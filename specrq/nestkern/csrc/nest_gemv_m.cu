// Small-M nested-VQ GEMV (M = 1..8) on CUDA cores: the k codewords of a group are summed once (fp32, one rounding to
// the activation dtype, as in nest_mma) and reused for all M rows. Same lane-permuted planar codes as nest_gemv
// (pack.kernel_codes), so a model needs no extra code layout. Targets latency-bound shapes (Qwen2.5-0.5B, Llama k/v)
// where the tensor-core path has a large fixed cost. One warp per output row, grid-stride over rows; x rows are read
// through L1 (coalesced per byte position thanks to the lane permutation).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <int KP> constexpr int repl_m() { return KP == 1 ? 8 : KP == 2 ? 4 : KP <= 4 ? 2 : 1; }

template <typename T> struct V;
template <> struct V<__nv_bfloat16> {
  __device__ __forceinline__ static float2 f2(uint32_t w) {
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
  }
  __device__ __forceinline__ static float rnd(float v) { return __bfloat162float(__float2bfloat16(v)); }
  __device__ __forceinline__ static __nv_bfloat16 out(float v) { return __float2bfloat16(v); }
  __device__ __forceinline__ static float in(__nv_bfloat16 v) { return __bfloat162float(v); }
};
template <> struct V<half> {
  __device__ __forceinline__ static float2 f2(uint32_t w) { return __half22float2(*reinterpret_cast<half2*>(&w)); }
  __device__ __forceinline__ static float rnd(float v) { return __half2float(__float2half(v)); }
  __device__ __forceinline__ static half out(float v) { return __float2half(v); }
  __device__ __forceinline__ static float in(half v) { return __half2float(v); }
};

__device__ __forceinline__ uint32_t cw(const int4& v, int i) {
  return i == 0 ? (uint32_t)v.x : i == 1 ? (uint32_t)v.y : i == 2 ? (uint32_t)v.z : (uint32_t)v.w;
}

template <typename T, int KP, int MM>
__global__ void __launch_bounds__(256) nest_gemv_m_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ codes, const int4* __restrict__ cb,
    const T* __restrict__ scales, T* __restrict__ y, int M, int N, int K, long long plane_u32) {
  constexpr int R = repl_m<KP>();
  extern __shared__ int4 sh[];
  for (int i = threadIdx.x; i < KP * 256; i += blockDim.x) {
    const int4 v = cb[i];
#pragma unroll
    for (int r = 0; r < R; r++) sh[i * R + r] = v;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int nch = K >> 5, rr = lane % R;
  const int4* xg = reinterpret_cast<const int4*>(x);
  const int kg = K / 8;  // int4 per x row
  for (int row = blockIdx.x * nw + warp; row < N; row += gridDim.x * nw) {
    const uint32_t* crow = codes + (long long)row * nch;
    float acc[MM];
#pragma unroll
    for (int m = 0; m < MM; m++) acc[m] = 0.f;
    for (int ch = lane; ch < nch; ch += 32) {
      uint32_t c[KP];
#pragma unroll
      for (int l = 0; l < KP; l++) c[l] = __ldg(crow + l * plane_u32 + ch);
      const int blk = ch >> 5, j = ch & 31, w = min(32, nch - 32 * blk);
#pragma unroll
      for (int g = 0; g < 4; g++) {
        float wf[8];
#pragma unroll
        for (int e = 0; e < 8; e++) wf[e] = 0.f;
#pragma unroll
        for (int l = 0; l < KP; l++) {
          const int4 v = sh[(l * 256 + ((c[l] >> (8 * g)) & 0xffu)) * R + rr];
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const float2 p = V<T>::f2(cw(v, q));
            wf[2 * q] += p.x;
            wf[2 * q + 1] += p.y;
          }
        }
        if constexpr (KP > 1) {
#pragma unroll
          for (int e = 0; e < 8; e++) wf[e] = V<T>::rnd(wf[e]);
        }
        const int gi = 128 * blk + g * w + j;
#pragma unroll
        for (int m = 0; m < MM; m++) {
          if (m < M) {
            const int4 xv = __ldg(xg + (long long)m * kg + gi);
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < 4; q++) {
              const float2 p = V<T>::f2(cw(xv, q));
              s = fmaf(p.x, wf[2 * q], s);
              s = fmaf(p.y, wf[2 * q + 1], s);
            }
            acc[m] += s;
          }
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MM; m++) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
    }
    if (lane == 0) {
      const float sc = V<T>::in(scales[row]);
#pragma unroll
      for (int m = 0; m < MM; m++)
        if (m < M) y[(long long)m * N + row] = V<T>::out(acc[m] * sc);
    }
  }
}

template <typename T, int KP, int MM>
void launch(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb, const torch::Tensor& sc,
            torch::Tensor& y, int M, int N, int K) {
  constexpr int R = repl_m<KP>(), THREADS = 256;
  const int smem = KP * 256 * R * 16;
  auto kern = nest_gemv_m_kernel<T, KP, MM>;
  int dev = x.get_device(), sms = 0, occ = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, THREADS, smem);
  const int need = (N + THREADS / 32 - 1) / (THREADS / 32);
  const int grid = std::max(1, std::min(sms * std::max(occ, 1), need));
  kern<<<grid, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(codes.data_ptr()),
      reinterpret_cast<const int4*>(cb.data_ptr()), reinterpret_cast<const T*>(sc.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()), M, N, K, (long long)N * (K / 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int MM>
void dispatch_k(int k, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, int M, int N, int K) {
  switch (k) {
    case 1: launch<T, 1, MM>(x, codes, cb, sc, y, M, N, K); break;
    case 2: launch<T, 2, MM>(x, codes, cb, sc, y, M, N, K); break;
    case 3: launch<T, 3, MM>(x, codes, cb, sc, y, M, N, K); break;
    case 4: launch<T, 4, MM>(x, codes, cb, sc, y, M, N, K); break;
    case 5: launch<T, 5, MM>(x, codes, cb, sc, y, M, N, K); break;
    default: TORCH_CHECK(false, "k in [1, 5]");
  }
}

template <typename T>
void dispatch_m(int k, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, int M, int N, int K) {
  if (M <= 2) dispatch_k<T, 2>(k, x, codes, cb, sc, y, M, N, K);
  else if (M <= 4) dispatch_k<T, 4>(k, x, codes, cb, sc, y, M, N, K);
  else dispatch_k<T, 8>(k, x, codes, cb, sc, y, M, N, K);
}

}  // namespace

torch::Tensor nest_gemv_m(torch::Tensor x, torch::Tensor codes, torch::Tensor codebooks, torch::Tensor scales,
                          int64_t k) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.size(0) >= 1 && x.size(0) <= 8, "x [M<=8, K] contiguous");
  TORCH_CHECK(codes.scalar_type() == at::kByte && codes.dim() == 3 && codes.is_contiguous(), "codes uint8 [L,N,K/8]");
  const int M = x.size(0), L = codes.size(0), N = codes.size(1), K = codes.size(2) * 8;
  TORCH_CHECK(x.size(1) == K && K % 32 == 0 && k >= 1 && k <= L, "shape / k");
  TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) >= k && codebooks.size(1) == 256 && codebooks.size(2) == 8 &&
                  codebooks.is_contiguous(), "codebooks [>=k, 256, 8]");
  TORCH_CHECK(scales.numel() == N && x.scalar_type() == codebooks.scalar_type() &&
                  x.scalar_type() == scales.scalar_type(), "scales / dtype");
  auto y = torch::empty({M, N}, x.options());
  if (x.scalar_type() == at::kBFloat16)
    dispatch_m<__nv_bfloat16>(k, x, codes, codebooks, scales, y, M, N, K);
  else
    dispatch_m<half>(k, x, codes, codebooks, scales, y, M, N, K);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("nest_gemv_m", &nest_gemv_m, "small-M nested VQ GEMV (CUDA cores)"); }
