// Online input rotation x' = BlockFWHT(x * S) / sqrt(B) (DbD Alg. 4, specrq/dbd/src/hadamard.py::rotate_input).
// One warp per (row, block of B); lane holds EPL = B/32 contiguous elements; butterflies with stride < EPL are
// in-register, larger strides use __shfl_xor_sync. fp32 math, output in the input dtype.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ half fromf(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }

template <typename T, int B>
__global__ void fwht_kernel(const T* __restrict__ x, const float* __restrict__ signs, T* __restrict__ y, int M,
                            int K) {
  constexpr int EPL = B / 32;
  const int lane = threadIdx.x & 31;
  const long long wid = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  const int nblk = K / B;
  if (wid >= (long long)M * nblk) return;
  const int m = wid / nblk, blk = wid % nblk;
  const int base = blk * B + lane * EPL;
  const T* xr = x + (long long)m * K + base;
  float v[EPL];
#pragma unroll
  for (int e = 0; e < EPL; e++) v[e] = tof(xr[e]) * __ldg(signs + base + e);
#pragma unroll
  for (int h = 1; h < EPL; h <<= 1)
#pragma unroll
    for (int e = 0; e < EPL; e++)
      if (!(e & h)) {
        const float a = v[e], b = v[e + h];
        v[e] = a + b;
        v[e + h] = a - b;
      }
#pragma unroll
  for (int lm = 1; lm < 32; lm <<= 1) {
    const bool upper = lane & lm;
#pragma unroll
    for (int e = 0; e < EPL; e++) {
      const float o = __shfl_xor_sync(0xffffffffu, v[e], lm);
      v[e] = upper ? (o - v[e]) : (v[e] + o);
    }
  }
  const float sc = rsqrtf((float)B);
  T* yr = y + (long long)m * K + base;
#pragma unroll
  for (int e = 0; e < EPL; e++) yr[e] = fromf<T>(v[e] * sc);
}

template <typename T>
void launch(const torch::Tensor& x, const torch::Tensor& s, torch::Tensor& y, int M, int K, int B) {
  const long long warps = (long long)M * (K / B);
  const int threads = 128;
  const int grid = (int)((warps * 32 + threads - 1) / threads);
  auto st = at::cuda::getCurrentCUDAStream();
  auto xp = reinterpret_cast<const T*>(x.data_ptr());
  auto yp = reinterpret_cast<T*>(y.data_ptr());
  auto sp = s.data_ptr<float>();
  switch (B) {
    case 32: fwht_kernel<T, 32><<<grid, threads, 0, st>>>(xp, sp, yp, M, K); break;
    case 64: fwht_kernel<T, 64><<<grid, threads, 0, st>>>(xp, sp, yp, M, K); break;
    case 128: fwht_kernel<T, 128><<<grid, threads, 0, st>>>(xp, sp, yp, M, K); break;
    case 256: fwht_kernel<T, 256><<<grid, threads, 0, st>>>(xp, sp, yp, M, K); break;
    default: TORCH_CHECK(false, "B must be 32, 64, 128 or 256");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor block_fwht_signed(torch::Tensor x, torch::Tensor signs, int64_t B) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x [M, K] contiguous CUDA");
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % B == 0, "K must be divisible by B");
  TORCH_CHECK(signs.scalar_type() == at::kFloat && signs.numel() == K && signs.is_contiguous(), "signs fp32 [K]");
  auto y = torch::empty_like(x);
  if (x.scalar_type() == at::kHalf)
    launch<half>(x, signs, y, M, K, B);
  else if (x.scalar_type() == at::kBFloat16)
    launch<__nv_bfloat16>(x, signs, y, M, K, B);
  else
    TORCH_CHECK(false, "dtype must be fp16 or bf16");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("block_fwht_signed", &block_fwht_signed, "x*S -> block FWHT/sqrt(B)"); }
