// Nested additive-VQ GEMV (M=1), planar codes, prefix k of L codebooks, 8-bit codes, d=8.
//   y[n] = scale[n] * sum_g < x[8g:8g+8], sum_{l<k} CB[l][codes[l][n][g]] >
// Layout "planar_rowmajor_v1": codes uint8 [L][N][K/8]; codebooks [L][256][8] (fp16|bf16); scales [N].
// Codes are in the lane-permuted byte order (pack.kernel_codes): byte g of u32 word ch holds group
// 128*(ch/32) + g*w + ch%32, w = words in that 32-word block, so x reads are coalesced across lanes.
// Path (a) "dequant-in-register": codebooks of the k used planes live in shared memory (replicated R times so lanes
// in a quarter-warp hit distinct banks for 128-bit loads); each lane reads 4 codes (one u32) per plane per step,
// fetches codewords from smem and dots them with x (read through L1). One warp per output row, grid-stride rows.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <int KP> constexpr int repl_auto() { return KP == 1 ? 8 : KP == 2 ? 4 : KP <= 4 ? 2 : 1; }  // <=32 KB smem

template <typename T> struct Dot;
template <> struct Dot<half> {
  // 8-term dot in half2 (as AQLM's kernels), result in fp32
  __device__ __forceinline__ static float dot8(const int4& a, const int4& b) {
    const half2* x = reinterpret_cast<const half2*>(&a);
    const half2* w = reinterpret_cast<const half2*>(&b);
    half2 r = __hmul2(x[0], w[0]);
    r = __hfma2(x[1], w[1], r);
    r = __hfma2(x[2], w[2], r);
    r = __hfma2(x[3], w[3], r);
    float2 f = __half22float2(r);
    return f.x + f.y;
  }
  __device__ __forceinline__ static float tof(half v) { return __half2float(v); }
  __device__ __forceinline__ static half fromf(float v) { return __float2half(v); }
};
template <> struct Dot<__nv_bfloat16> {
  // bf16 has 8 mantissa bits: accumulate in fp32
  __device__ __forceinline__ static float dot8(const int4& a, const int4& b) {
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* w = reinterpret_cast<const __nv_bfloat162*>(&b);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float2 xf = __bfloat1622float2(x[j]);
      float2 wf = __bfloat1622float2(w[j]);
      s = fmaf(xf.x, wf.x, s);
      s = fmaf(xf.y, wf.y, s);
    }
    return s;
  }
  __device__ __forceinline__ static float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ __forceinline__ static __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }
};

template <typename T, int KP, int U, int R>
__global__ void __launch_bounds__(256) nest_gemv_kernel(
    const int4* __restrict__ x, const uint32_t* __restrict__ codes, const int4* __restrict__ cb,
    const T* __restrict__ scales, T* __restrict__ y, int N, int K, long long plane_u32) {
  extern __shared__ int4 sh[];  // [KP*256*R]
  for (int i = threadIdx.x; i < KP * 256; i += blockDim.x) {
    int4 v = cb[i];
#pragma unroll
    for (int r = 0; r < R; r++) sh[i * R + r] = v;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int nch = K >> 5;  // u32 code words per row (4 groups of 8 each)
  const int rr = lane % R;
  for (int row = blockIdx.x * nw + warp; row < N; row += gridDim.x * nw) {
    const uint32_t* crow = codes + (long long)row * nch;
    float acc = 0.f;
    for (int base = lane; base < nch; base += 32 * U) {
      uint32_t c[U][KP];
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int ch = base + 32 * u;
#pragma unroll
        for (int l = 0; l < KP; l++) c[u][l] = ch < nch ? __ldg(crow + l * plane_u32 + ch) : 0u;
      }
#pragma unroll
      for (int u = 0; u < U; u++) {
        const int ch = base + 32 * u;
        if (ch < nch) {
          const int blk = ch >> 5, j = ch & 31;
          const int w = min(32, nch - 32 * blk);
#pragma unroll
          for (int g = 0; g < 4; g++) {
            const int4 xv = __ldg(x + 128 * blk + g * w + j);
#pragma unroll
            for (int l = 0; l < KP; l++) {
              const uint32_t code = (c[u][l] >> (8 * g)) & 0xffu;
              acc += Dot<T>::dot8(xv, sh[(l * 256 + code) * R + rr]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = Dot<T>::fromf(acc * Dot<T>::tof(scales[row]));
  }
}

// Register-resident x variant for rows with nch <= 32*CPL code words (K <= 1024*CPL): each lane keeps the x
// slices of its CPL chunks in registers for all rows, and prefetches the next row's codes while computing.
template <typename T, int KP, int R, int CPL>
__global__ void __launch_bounds__(256) nest_gemv_xreg_kernel(
    const int4* __restrict__ x, const uint32_t* __restrict__ codes, const int4* __restrict__ cb,
    const T* __restrict__ scales, T* __restrict__ y, int N, int K, long long plane_u32) {
  extern __shared__ int4 sh[];
  for (int i = threadIdx.x; i < KP * 256; i += blockDim.x) {
    int4 v = cb[i];
#pragma unroll
    for (int r = 0; r < R; r++) sh[i * R + r] = v;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int nch = K >> 5;
  const int rr = lane % R;
  int4 xr[CPL][4];
  bool act[CPL];
#pragma unroll
  for (int u = 0; u < CPL; u++) {
    const int ch = lane + 32 * u;
    act[u] = ch < nch;
    const int blk = ch >> 5, j = ch & 31, w = min(32, nch - 32 * blk);
#pragma unroll
    for (int g = 0; g < 4; g++) xr[u][g] = act[u] ? __ldg(x + 128 * blk + g * w + j) : make_int4(0, 0, 0, 0);
  }
  __syncthreads();
  const int stride = gridDim.x * nw;
  int row = blockIdx.x * nw + warp;
  uint32_t c[CPL][KP];
  auto load = [&](int r_, uint32_t (&cc)[CPL][KP]) {
    const uint32_t* crow = codes + (long long)r_ * nch;
#pragma unroll
    for (int u = 0; u < CPL; u++)
#pragma unroll
      for (int l = 0; l < KP; l++) cc[u][l] = act[u] ? __ldg(crow + l * plane_u32 + lane + 32 * u) : 0u;
  };
  if (row < N) load(row, c);
  while (row < N) {
    const int nxt = row + stride;
    uint32_t cn[CPL][KP];
    if (nxt < N) load(nxt, cn);
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < CPL; u++) {
      if (act[u]) {
#pragma unroll
        for (int g = 0; g < 4; g++)
#pragma unroll
          for (int l = 0; l < KP; l++) {
            const uint32_t code = (c[u][l] >> (8 * g)) & 0xffu;
            acc += Dot<T>::dot8(xr[u][g], sh[(l * 256 + code) * R + rr]);
          }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = Dot<T>::fromf(acc * Dot<T>::tof(scales[row]));
#pragma unroll
    for (int u = 0; u < CPL; u++)
#pragma unroll
      for (int l = 0; l < KP; l++) c[u][l] = cn[u][l];
    row = nxt;
  }
}

template <typename T, typename Kern>
void run_kernel(Kern kern, int smem, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, int N, int K) {
  constexpr int THREADS = 256;
  TORCH_CHECK(smem <= 99 * 1024, "codebook smem ", smem, " B exceeds 99 KB");
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  int dev = x.get_device(), sms = 0, occ = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, THREADS, smem);
  int need = (N + THREADS / 32 - 1) / (THREADS / 32);
  int grid = std::max(1, std::min(sms * std::max(occ, 1), need));
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<grid, THREADS, smem, stream>>>(
      reinterpret_cast<const int4*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(codes.data_ptr()),
      reinterpret_cast<const int4*>(cb.data_ptr()), reinterpret_cast<const T*>(sc.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()), N, K, (long long)N * (K / 32));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int KP, int R>
void launch_r(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb, const torch::Tensor& sc,
              torch::Tensor& y, int N, int K) {
  constexpr int U = 4;
  run_kernel<T>(nest_gemv_kernel<T, KP, U, R>, KP * 256 * R * 16, x, codes, cb, sc, y, N, K);
}

// variant: 0 = auto (xreg when K <= 4096 and repl is auto), 1 = generic loop, 2 = force xreg
template <typename T, int KP>
bool try_xreg(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb, const torch::Tensor& sc,
              torch::Tensor& y, int N, int K) {
  constexpr int R = repl_auto<KP>();
  const int smem = KP * 256 * R * 16;
  const int cpl = (K / 32 + 31) / 32;
  switch (cpl) {
    case 1: run_kernel<T>(nest_gemv_xreg_kernel<T, KP, R, 1>, smem, x, codes, cb, sc, y, N, K); return true;
    case 2: run_kernel<T>(nest_gemv_xreg_kernel<T, KP, R, 2>, smem, x, codes, cb, sc, y, N, K); return true;
    case 3: run_kernel<T>(nest_gemv_xreg_kernel<T, KP, R, 3>, smem, x, codes, cb, sc, y, N, K); return true;
    case 4: run_kernel<T>(nest_gemv_xreg_kernel<T, KP, R, 4>, smem, x, codes, cb, sc, y, N, K); return true;
    default: return false;
  }
}

template <typename T, int KP>
void launch(int repl, int variant, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
            const torch::Tensor& sc, torch::Tensor& y, int N, int K) {
  if (variant != 1 && repl <= 0) {
    if (try_xreg<T, KP>(x, codes, cb, sc, y, N, K)) return;
    TORCH_CHECK(variant != 2, "xreg variant needs K <= 4096");
  }
  if (repl <= 0) repl = repl_auto<KP>();
  switch (repl) {
    case 1: launch_r<T, KP, 1>(x, codes, cb, sc, y, N, K); break;
    case 2: launch_r<T, KP, 2>(x, codes, cb, sc, y, N, K); break;
    case 4: launch_r<T, KP, 4>(x, codes, cb, sc, y, N, K); break;
    case 8: launch_r<T, KP, 8>(x, codes, cb, sc, y, N, K); break;
    default: TORCH_CHECK(false, "repl must be 1, 2, 4 or 8");
  }
}

template <typename T>
void dispatch_k(int k, int repl, int variant, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, int N, int K) {
  switch (k) {
    case 1: launch<T, 1>(repl, variant, x, codes, cb, sc, y, N, K); break;
    case 2: launch<T, 2>(repl, variant, x, codes, cb, sc, y, N, K); break;
    case 3: launch<T, 3>(repl, variant, x, codes, cb, sc, y, N, K); break;
    case 4: launch<T, 4>(repl, variant, x, codes, cb, sc, y, N, K); break;
    case 5: launch<T, 5>(repl, variant, x, codes, cb, sc, y, N, K); break;
    default: TORCH_CHECK(false, "prefix k must be in [1, 5], got ", k);
  }
}

}  // namespace

// x [1, K] or [K]; codes uint8 [L, N, K/8]; codebooks [L, 256, 8]; scales [N]; returns y [1, N]
torch::Tensor nest_gemv(torch::Tensor x, torch::Tensor codes, torch::Tensor codebooks, torch::Tensor scales,
                        int64_t k, int64_t repl, int64_t variant) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  TORCH_CHECK(x.is_cuda() && codes.is_cuda() && codebooks.is_cuda() && scales.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(codes.scalar_type() == at::kByte && codes.dim() == 3, "codes must be uint8 [L, N, K/8]");
  const int L = codes.size(0), N = codes.size(1), K = codes.size(2) * 8;
  TORCH_CHECK(k >= 1 && k <= L, "k=", k, " out of range for L=", L);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32");
  TORCH_CHECK(x.numel() == K, "nest_gemv is the M=1 path: x must have K elements (got ", x.numel(), ")");
  // codebooks may be the target's [L, 256, 8] or a separate draft set [k_d, 256, 8] (only planes < k are read)
  TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) >= k && codebooks.size(1) == 256 && codebooks.size(2) == 8 &&
                  codebooks.is_contiguous(), "codebooks must be contiguous [>=k, 256, 8]");
  TORCH_CHECK(scales.numel() == N, "scales must have N elements");
  TORCH_CHECK(x.scalar_type() == codebooks.scalar_type() && x.scalar_type() == scales.scalar_type(),
              "x, codebooks, scales must share dtype");
  TORCH_CHECK(x.is_contiguous() && codes.is_contiguous() && codebooks.is_contiguous() && scales.is_contiguous(),
              "contiguous tensors required");
  auto y = torch::empty({1, N}, x.options());
  if (x.scalar_type() == at::kHalf)
    dispatch_k<half>(k, repl, variant, x, codes, codebooks, scales, y, N, K);
  else if (x.scalar_type() == at::kBFloat16)
    dispatch_k<__nv_bfloat16>(k, repl, variant, x, codes, codebooks, scales, y, N, K);
  else
    TORCH_CHECK(false, "dtype must be float16 or bfloat16");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("nest_gemv", &nest_gemv, "nested VQ GEMV (M=1), planar codes"); }
