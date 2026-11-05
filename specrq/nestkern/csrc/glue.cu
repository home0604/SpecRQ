// Fused decode glue for NestModel (bf16/fp16 io, fp32 math), replacing several small kernels per layer:
//   add_rms_rot: h' = h + d (rounded), x = w * round(h' * rsqrt(mean(h'^2) + eps)) (rounded, HF RMSNorm order),
//                then for each of S sign vectors: out_s = BlockFWHT(x * sign_s) / sqrt(C)   (S = 0: out = x)
//   act_rot    : a = round(round(silu(g)) * u), out = BlockFWHT(a * sign) / sqrt(C)        (no sign: out = a)
// Grid = (M rows, K / C chunks); C = Hadamard block size (or 128 without rotation); one thread per element of the
// chunk; add_rms_rot recomputes the row's sum of squares in every chunk block (the row is small and L2-resident).
// Rounding points match the unfused path (compiled RMSNorm -> bf16, nestkern fwht on the bf16 input).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float tof(half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T fromf(float v);
template <> __device__ __forceinline__ __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ half fromf(float v) { return __float2half(v); }
template <typename T> __device__ __forceinline__ float rnd(float v) { return tof<T>(fromf<T>(v)); }

template <int C>
__device__ __forceinline__ float fwht_chunk(float v, float* sh) {
  // in-place Walsh-Hadamard over C elements held one per thread (threadIdx.x < C)
  const int i = threadIdx.x;
  sh[i] = v;
  __syncthreads();
#pragma unroll
  for (int h = 1; h < C; h <<= 1) {
    const float a = sh[i], b = sh[i ^ h];
    __syncthreads();
    sh[i] = (i & h) ? (b - a) : (a + b);
    __syncthreads();
  }
  return sh[i] * rsqrtf((float)C);
}

template <typename T, int C>
__global__ void add_rms_rot_kernel(const T* __restrict__ h, const T* __restrict__ d, const T* __restrict__ w,
                                   const float* __restrict__ signs, T* __restrict__ h_out, T* __restrict__ out,
                                   int M, int K, int S, float eps) {
  __shared__ float red[32];
  __shared__ float sh[C];
  const int m = blockIdx.x, chunk = blockIdx.y, i = threadIdx.x;
  const T* hr = h + (long long)m * K;
  const T* dr = d ? d + (long long)m * K : nullptr;
  // row sum of squares of h' = round(h + d)
  float ss = 0.f;
  for (int j = i; j < K; j += C) {
    const float v = dr ? rnd<T>(tof<T>(hr[j]) + tof<T>(dr[j])) : tof<T>(hr[j]);
    ss += v * v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((i & 31) == 0) red[i >> 5] = ss;
  __syncthreads();
  if (i < 32) {
    float t = i < C / 32 ? red[i] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (i == 0) red[0] = t;
  }
  __syncthreads();
  const float inv = rsqrtf(red[0] / K + eps);
  const int col = chunk * C + i;
  const float hv = dr ? rnd<T>(tof<T>(hr[col]) + tof<T>(dr[col])) : tof<T>(hr[col]);
  if (h_out) h_out[(long long)m * K + col] = fromf<T>(hv);
  const float x = rnd<T>(tof<T>(w[col]) * rnd<T>(hv * inv));
  if (S == 0) {
    out[(long long)m * K + col] = fromf<T>(x);
    return;
  }
  for (int s = 0; s < S; s++) {
    __syncthreads();
    const float y = fwht_chunk<C>(x * signs[(long long)s * K + col], sh);
    out[((long long)s * M + m) * K + col] = fromf<T>(y);
  }
}

template <typename T, int C>
__global__ void act_rot_kernel(const T* __restrict__ g, const T* __restrict__ u, const float* __restrict__ sign,
                               T* __restrict__ out, int K) {
  __shared__ float sh[C];
  const int m = blockIdx.x, col = blockIdx.y * C + threadIdx.x;
  const long long o = (long long)m * K + col;
  const float gv = tof<T>(g[o]);
  const float a = rnd<T>(rnd<T>(gv / (1.f + __expf(-gv))) * tof<T>(u[o]));
  out[o] = fromf<T>(sign ? fwht_chunk<C>(a * sign[col], sh) : a);
}

template <typename T>
void launch_add_rms(const torch::Tensor& h, const c10::optional<torch::Tensor>& d, const torch::Tensor& w,
                    const c10::optional<torch::Tensor>& signs, torch::Tensor& h_out, torch::Tensor& out, int C,
                    double eps, bool write_h) {
  const int M = h.size(0), K = h.size(1), S = signs ? signs->size(0) : 0;
  dim3 grid(M, K / C);
  auto st = at::cuda::getCurrentCUDAStream();
  auto hp = reinterpret_cast<const T*>(h.data_ptr());
  auto dp = d ? reinterpret_cast<const T*>(d->data_ptr()) : nullptr;
  auto wp = reinterpret_cast<const T*>(w.data_ptr());
  auto sp = signs ? signs->data_ptr<float>() : nullptr;
  auto hop = write_h ? reinterpret_cast<T*>(h_out.data_ptr()) : nullptr;
  auto op = reinterpret_cast<T*>(out.data_ptr());
  switch (C) {
#define CASE(c) case c: add_rms_rot_kernel<T, c><<<grid, c, 0, st>>>(hp, dp, wp, sp, hop, op, M, K, S, (float)eps); break;
    CASE(32) CASE(64) CASE(128) CASE(256)
#undef CASE
    default: TORCH_CHECK(false, "chunk must be 32/64/128/256");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// returns (h_out or empty, out [S, M, K] or [M, K] when S == 0)
std::vector<torch::Tensor> add_rms_rot(torch::Tensor h, c10::optional<torch::Tensor> d, torch::Tensor w,
                                       c10::optional<torch::Tensor> signs, int64_t C, double eps) {
  const at::cuda::OptionalCUDAGuard guard(device_of(h));
  TORCH_CHECK(h.dim() == 2 && h.is_contiguous(), "h [M, K] contiguous");
  const int M = h.size(0), K = h.size(1);
  TORCH_CHECK(K % C == 0, "K % C != 0");
  if (d) TORCH_CHECK(d->sizes() == h.sizes() && d->is_contiguous() && d->scalar_type() == h.scalar_type(), "d like h");
  if (signs) TORCH_CHECK(signs->scalar_type() == at::kFloat && signs->dim() == 2 && signs->size(1) == K &&
                             signs->is_contiguous(), "signs fp32 [S, K]");
  const int S = signs ? signs->size(0) : 0;
  torch::Tensor h_out = d ? torch::empty_like(h) : torch::Tensor();
  torch::Tensor out = S ? torch::empty({S, M, K}, h.options()) : torch::empty_like(h);
  if (h.scalar_type() == at::kBFloat16)
    launch_add_rms<__nv_bfloat16>(h, d, w, signs, h_out, out, C, eps, d.has_value());
  else
    launch_add_rms<half>(h, d, w, signs, h_out, out, C, eps, d.has_value());
  return {h_out, out};
}

torch::Tensor act_rot(torch::Tensor g, torch::Tensor u, c10::optional<torch::Tensor> sign, int64_t C) {
  const at::cuda::OptionalCUDAGuard guard(device_of(g));
  TORCH_CHECK(g.dim() == 2 && g.is_contiguous() && u.sizes() == g.sizes() && u.is_contiguous(), "g, u [M, K]");
  const int M = g.size(0), K = g.size(1);
  TORCH_CHECK(K % C == 0, "K % C != 0");
  if (sign) TORCH_CHECK(sign->scalar_type() == at::kFloat && sign->numel() == K, "sign fp32 [K]");
  auto out = torch::empty_like(g);
  dim3 grid(M, K / C);
  auto st = at::cuda::getCurrentCUDAStream();
  const float* sp = sign ? sign->data_ptr<float>() : nullptr;
#define CASE(T, c) case c: act_rot_kernel<T, c><<<grid, c, 0, st>>>(reinterpret_cast<const T*>(g.data_ptr()), \
    reinterpret_cast<const T*>(u.data_ptr()), sp, reinterpret_cast<T*>(out.data_ptr()), K); break;
  if (g.scalar_type() == at::kBFloat16) {
    switch (C) { CASE(__nv_bfloat16, 32) CASE(__nv_bfloat16, 64) CASE(__nv_bfloat16, 128) CASE(__nv_bfloat16, 256)
      default: TORCH_CHECK(false, "bad C"); }
  } else {
    switch (C) { CASE(half, 32) CASE(half, 64) CASE(half, 128) CASE(half, 256) default: TORCH_CHECK(false, "bad C"); }
  }
#undef CASE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_rms_rot", &add_rms_rot, "residual add + RMSNorm + S block-FWHT rotations");
  m.def("act_rot", &act_rot, "silu(g)*u + block-FWHT rotation");
}
