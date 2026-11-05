// Nested additive-VQ GEMM for small M (1..16) with tensor cores: path (a) "dequant-in-register + mma".
//   y[m, n] = scale[n] * sum_k x[m, k] * W_k[n, k],   W_k = sum_{l<k} CB[l][codes[l][n][k/8]]
// Roles are swapped so the weight is the 16x16 A operand and x^T is the 16x8 B operand:
//   mma.m16n8k16 (A row-major = W tile [16 n x 16 k], B col-major = x^T [16 k x 8 m], C [16 n x 8 m] fp32).
// M <= 8 uses one n8 tile (NT=1), M <= 16 two (NT=2). Rows m >= M are zero in B and are not stored.
//
// k permutation inside a 64-wide super-tile (same permutation on W and x, so the result is unchanged): mma k-step s,
// fragment column kk in {2t, 2t+1, 2t+8, 2t+9} maps to physical k = 16t + 4s + {0, 1, 2, 3}. Hence lane (g, t)
//   - reads x[m][64ks + 16t .. +16] as two contiguous uint4 (b0 = word 2s, b1 = word 2s+1),
//   - needs, for rows g and g+8, the two codewords of groups 2t (k-steps 0,1) and 2t+1 (k-steps 2,3): one LDS.128 per
//     (row, group, plane) gives both halves (a0/a2 = words 2(s&1), 2(s&1)+1).
// Code layout "mma_tile_v1" (pack.mma_codes): per plane l, super-tiles of 16 rows x 64 k (= 8 groups) stored as
// 128 contiguous bytes, ordered [N/16][K/64]. Word (lane = 4g + t) holds bytes j = 0..3 with the codes of
//   row g + 8*(j>>1), group 2t + (j&1)
// so every lane loads its own u32 per plane per super-tile (a warp reads 128 contiguous bytes).
// Codebooks sit in shared memory replicated R times (copy = lane % R) so a quarter-warp's 8 LDS.128 hit distinct
// 16-byte bank groups when R = 8.
// Dequant: k codewords summed in fp32, one rounding to the mma dtype (bf16 / fp16).
// det = 1: each split writes its partial to a [splits, M, N] buffer and the last block sums them in split order
// (bitwise reproducible run to run). det = 0:
// Split-K over blockIdx.y: fp32 atomicAdd into a persistent zeroed workspace [M, N] (stays in L2); the last block of
// each N-block column converts to the output dtype and re-zeroes its slice (self-resetting counters too).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

// Tile config (template): WARPS warps per block, all along N; TPW 16-row tiles per warp; KU super-tiles (64 k each)
// loaded per pipeline step (next step's codes are prefetched while computing the current one).

template <int KP> constexpr int repl_mma() { return KP == 1 ? 8 : KP == 2 ? 4 : KP <= 4 ? 2 : 1; }

template <typename T> struct MT;
template <> struct MT<__nv_bfloat16> {
  __device__ __forceinline__ static void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  // sum of KP packed pairs, fp32 accumulate, single rounding
  template <int KP>
  __device__ __forceinline__ static uint32_t sum(const uint32_t (&v)[KP]) {
    if constexpr (KP == 1) {
      return v[0];
    } else {
      float lo = 0.f, hi = 0.f;
#pragma unroll
      for (int l = 0; l < KP; l++) {
        lo += __uint_as_float(v[l] << 16);
        hi += __uint_as_float(v[l] & 0xffff0000u);
      }
      __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
      return *reinterpret_cast<uint32_t*>(&r);
    }
  }
  __device__ __forceinline__ static __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }
  __device__ __forceinline__ static float2 unpack(uint32_t w) {
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
  }
  __device__ __forceinline__ static uint32_t pack(float a, float b) {
    __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&r);
  }
  __device__ __forceinline__ static float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
};
template <> struct MT<half> {
  __device__ __forceinline__ static void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  template <int KP>
  __device__ __forceinline__ static uint32_t sum(const uint32_t (&v)[KP]) {
    if constexpr (KP == 1) {
      return v[0];
    } else {
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int l = 0; l < KP; l++) {
        float2 f = __half22float2(*reinterpret_cast<const half2*>(&v[l]));
        acc.x += f.x;
        acc.y += f.y;
      }
      half2 r = __floats2half2_rn(acc.x, acc.y);
      return *reinterpret_cast<uint32_t*>(&r);
    }
  }
  __device__ __forceinline__ static half fromf(float v) { return __float2half(v); }
  __device__ __forceinline__ static float2 unpack(uint32_t w) {
    return __half22float2(*reinterpret_cast<const half2*>(&w));
  }
  __device__ __forceinline__ static uint32_t pack(float a, float b) {
    half2 r = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&r);
  }
  __device__ __forceinline__ static float tof(half v) { return __half2float(v); }
};

__device__ __forceinline__ uint32_t comp(const uint4& v, int s) {
  return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w;
}

// sum of KP codewords (8 elements each), fp32 accumulate, one rounding
template <typename T, int KP>
__device__ __forceinline__ uint4 sum_cw(const uint4 (&v)[KP]) {
  if constexpr (KP == 1) {
    return v[0];
  } else {
    float f[8];
#pragma unroll
    for (int e = 0; e < 8; e++) f[e] = 0.f;
#pragma unroll
    for (int l = 0; l < KP; l++) {
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float2 p = MT<T>::unpack(comp(v[l], w));
        f[2 * w] += p.x;
        f[2 * w + 1] += p.y;
      }
    }
    return make_uint4(MT<T>::pack(f[0], f[1]), MT<T>::pack(f[2], f[3]), MT<T>::pack(f[4], f[5]),
                      MT<T>::pack(f[6], f[7]));
  }
}

template <typename T, int KP, int NT, int WARPS, int TPW, int KU>
__global__ void __launch_bounds__(WARPS * 32) nest_mma_kernel(
    const T* __restrict__ x, const uint32_t* __restrict__ codes, const uint4* __restrict__ cb,
    const T* __restrict__ scales, T* __restrict__ y, float* __restrict__ ws, int* __restrict__ counters,
    int M, int N, int K, int ks_per_split, long long plane_u32, int det) {
  constexpr int R = repl_mma<KP>();
  extern __shared__ uint4 sh[];  // [KP][256][R] codewords
  for (int i = threadIdx.x; i < KP * 256; i += blockDim.x) {
    const uint4 v = cb[i];
#pragma unroll
    for (int r = 0; r < R; r++) sh[i * R + r] = v;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3, cp = lane % R;
  const int nsup = K >> 6;                         // super-tiles along K
  const int ks0 = blockIdx.y * ks_per_split;
  const int ks1 = min(nsup, ks0 + ks_per_split);
  constexpr int ROWS_PER_BLOCK = WARPS * TPW * 16;
  const int tile0 = (blockIdx.x * ROWS_PER_BLOCK) / 16 + warp * TPW;  // first 16-row tile of this warp
  const int ntiles = N >> 4;

  float acc[TPW][NT][4];
#pragma unroll
  for (int i = 0; i < TPW; i++)
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int q = 0; q < 4; q++) acc[i][j][q] = 0.f;

  bool tvalid[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) tvalid[i] = tile0 + i < ntiles;

  auto load_codes = [&](int ks, uint32_t (&c)[KU][TPW][KP]) {
#pragma unroll
    for (int u = 0; u < KU; u++)
#pragma unroll
      for (int i = 0; i < TPW; i++)
#pragma unroll
        for (int l = 0; l < KP; l++)
          c[u][i][l] = (tvalid[i] && ks + u < ks1)
                           ? __ldg(codes + l * plane_u32 + ((long long)(tile0 + i) * nsup + ks + u) * 32 + lane)
                           : 0u;
  };

  uint32_t cur[KU][TPW][KP];
  if (ks0 < ks1) load_codes(ks0, cur);
  for (int ks = ks0; ks < ks1; ks += KU) {
    uint32_t nxt[KU][TPW][KP];
    if (ks + KU < ks1) load_codes(ks + KU, nxt);
#pragma unroll
    for (int u = 0; u < KU; u++) {
      if (ks + u >= ks1) break;
      // x slice of this lane: x[m][64(ks+u) + 16t .. +16] -> 8 words; k-step s uses words 2s (b0), 2s+1 (b1)
      uint4 xv[NT][2];
#pragma unroll
      for (int j = 0; j < NT; j++) {
        const int m = j * 8 + g;
        if (m < M) {
          const uint4* xr = reinterpret_cast<const uint4*>(x + (long long)m * K + (ks + u) * 64 + 16 * t);
          xv[j][0] = __ldg(xr);
          xv[j][1] = __ldg(xr + 1);
        } else {
          xv[j][0] = xv[j][1] = make_uint4(0, 0, 0, 0);
        }
      }
#pragma unroll
      for (int i = 0; i < TPW; i++) {
        // dequantized codewords: w[rh][gp] for row g + 8*rh, group 2t + gp
        uint4 w[2][2];
#pragma unroll
        for (int rh = 0; rh < 2; rh++)
#pragma unroll
          for (int gp = 0; gp < 2; gp++) {
            uint4 v[KP];
#pragma unroll
            for (int l = 0; l < KP; l++) {
              const uint32_t code = (cur[u][i][l] >> (8 * (rh * 2 + gp))) & 0xffu;
              v[l] = sh[(l * 256 + code) * R + cp];
            }
            w[rh][gp] = sum_cw<T, KP>(v);
          }
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int gp = s >> 1, h = (s & 1) * 2;
          const uint32_t a[4] = {comp(w[0][gp], h), comp(w[1][gp], h), comp(w[0][gp], h + 1), comp(w[1][gp], h + 1)};
#pragma unroll
          for (int j = 0; j < NT; j++)
            MT<T>::mma(acc[i][j], a, comp(xv[j][s >> 1], (s & 1) * 2), comp(xv[j][s >> 1], (s & 1) * 2 + 1));
        }
      }
    }
#pragma unroll
    for (int u = 0; u < KU; u++)
#pragma unroll
      for (int i = 0; i < TPW; i++)
#pragma unroll
        for (int l = 0; l < KP; l++) cur[u][i][l] = nxt[u][i][l];
  }

  // epilogue: C frag c0,c1 -> (n = row g, m = 2t, 2t+1), c2,c3 -> (n = row g+8, m = 2t, 2t+1)
  const bool split = gridDim.y > 1;
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    if (!tvalid[i]) continue;
    const int nb = (tile0 + i) * 16 + g;
    const float s0 = MT<T>::tof(scales[nb]), s1 = MT<T>::tof(scales[nb + 8]);
#pragma unroll
    for (int j = 0; j < NT; j++) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int m = j * 8 + 2 * t + (q & 1);
        const int n = nb + (q >> 1) * 8;
        if (m < M) {
          const float v = acc[i][j][q] * ((q >> 1) ? s1 : s0);
          if (split && det)
            ws[((long long)blockIdx.y * M + m) * N + n] = v;  // per-split partial (deterministic order below)
          else if (split)
            atomicAdd(ws + (long long)m * N + n, v);  // L2 reduction into the persistent zeroed workspace
          else
            y[(long long)m * N + n] = MT<T>::fromf(v);
        }
      }
    }
  }
  if (!split) return;
  // last block of this N-block column reduces the gridDim.y partials
  __shared__ int is_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int prev = atomicAdd(&counters[blockIdx.x], 1);
    is_last = prev == (int)gridDim.y - 1;
  }
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int n0 = blockIdx.x * ROWS_PER_BLOCK;
  const int nrows = min(ROWS_PER_BLOCK, N - n0);
  for (int e = threadIdx.x; e < M * nrows; e += blockDim.x) {
    const int m = e / nrows, n = n0 + e % nrows;
    float v;
    if (det) {
      v = 0.f;
      for (int sk = 0; sk < (int)gridDim.y; sk++) v += __ldcg(ws + ((long long)sk * M + m) * N + n);
    } else {
      float* p = ws + (long long)m * N + n;
      v = __ldcg(p);
      __stcg(p, 0.f);  // leave the workspace zeroed for the next call
    }
    y[(long long)m * N + n] = MT<T>::fromf(v);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;  // self-reset for the next call
}

template <typename T, int KP, int NT, int WARPS, int TPW, int KU>
void launch_cfg(const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb, const torch::Tensor& sc,
                torch::Tensor& y, torch::Tensor& counters, torch::Tensor& ws, int M, int N, int K, int splits, int det) {
  constexpr int R = repl_mma<KP>();
  constexpr int ROWS_PER_BLOCK = WARPS * TPW * 16;
  const int smem = KP * 256 * R * 16;
  auto kern = nest_mma_kernel<T, KP, NT, WARPS, TPW, KU>;
  if (smem > 48 * 1024) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const int dev = x.get_device();
  int sms = 0, occ = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, WARPS * 32, smem);
  const int nbN = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  const int nsup = K / 64;
  if (splits <= 0) {  // auto: largest split count whose grid still fits in one wave of resident blocks
    const int target = sms * std::max(occ, 1);  // (ceil here gave a ~1.2-wave grid, i.e. a 2x tail on fused gate_up)
    splits = std::max(1, std::min(target / nbN, nsup / (2 * KU)));
  }
  splits = std::max(1, std::min(splits, nsup));
  const int per = (nsup + splits - 1) / splits;
  splits = (nsup + per - 1) / per;
  TORCH_CHECK(counters.numel() >= nbN, "counter buffer too small");
  TORCH_CHECK(ws.numel() >= (long)M * N && ws.scalar_type() == at::kFloat, "workspace too small");
  torch::Tensor wsd;  // deterministic mode: per-split partials [splits, M, N]
  if (det && splits > 1) wsd = torch::empty({(long)splits * M * N}, x.options().dtype(torch::kFloat32));
  dim3 grid(nbN, splits);
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<grid, WARPS * 32, smem, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(codes.data_ptr()),
      reinterpret_cast<const uint4*>(cb.data_ptr()), reinterpret_cast<const T*>(sc.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()), (det && splits > 1) ? wsd.data_ptr<float>() : ws.data_ptr<float>(),
      counters.data_ptr<int>(), M, N, K, per, (long long)N * K / 8 / 4, det);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// cfg id (WARPS, TPW, KU): 1 = (4,2,1); 2 = (8,2,1); 3 = (8,2,2); 4 = (8,1,2); 5 = (8,1,4); 6 = (4,2,2);
// 7 = (8,4,1); 8 = (4,4,1); 9 = (4,1,4). 0 = auto (heuristic below).
template <typename T, int KP, int NT>
void launch(int cfg, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
            const torch::Tensor& sc, torch::Tensor& y, torch::Tensor& cnt, torch::Tensor& ws, int M, int N, int K, int splits, int det) {
  if (cfg == 0)  // heuristic from the M3 sweep (Llama gate_up / q_o / down / k_v, bf16, graph_flush/read)
    cfg = N >= 8192 ? (KP <= 2 ? 2 : 8) : (N <= 2048 ? 4 : 2);
  switch (cfg) {
    case 1: launch_cfg<T, KP, NT, 4, 2, 1>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 2: launch_cfg<T, KP, NT, 8, 2, 1>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 3: launch_cfg<T, KP, NT, 8, 2, 2>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 4: launch_cfg<T, KP, NT, 8, 1, 2>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 5: launch_cfg<T, KP, NT, 8, 1, 4>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 6: launch_cfg<T, KP, NT, 4, 2, 2>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 7: launch_cfg<T, KP, NT, 8, 4, 1>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 8: launch_cfg<T, KP, NT, 4, 4, 1>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 9: launch_cfg<T, KP, NT, 4, 1, 4>(x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    default: TORCH_CHECK(false, "unknown cfg ", cfg);
  }
}

template <typename T, int NT>
void dispatch_k(int k, int cfg, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, torch::Tensor& cnt, torch::Tensor& ws, int M, int N, int K, int splits, int det) {
  switch (k) {
    case 1: launch<T, 1, NT>(cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 2: launch<T, 2, NT>(cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 3: launch<T, 3, NT>(cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 4: launch<T, 4, NT>(cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    case 5: launch<T, 5, NT>(cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det); break;
    default: TORCH_CHECK(false, "prefix k must be in [1, 5], got ", k);
  }
}

template <typename T>
void dispatch_m(int k, int cfg, const torch::Tensor& x, const torch::Tensor& codes, const torch::Tensor& cb,
                const torch::Tensor& sc, torch::Tensor& y, torch::Tensor& cnt, torch::Tensor& ws, int M, int N, int K, int splits, int det) {
  if (M <= 8)
    dispatch_k<T, 1>(k, cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det);
  else
    dispatch_k<T, 2>(k, cfg, x, codes, cb, sc, y, cnt, ws, M, N, K, splits, det);
}

}  // namespace

// x [M, K] (M <= 16); codes uint8 mma_tile_v1 [L, N*K/8] (see header); codebooks [L, 256, 8]; scales [N];
// counters int32 zero-initialised buffer (>= ceil(N/ROWS_PER_BLOCK) entries, reset by the kernel). Returns y [M, N].
torch::Tensor nest_mma(torch::Tensor x, torch::Tensor codes, torch::Tensor codebooks, torch::Tensor scales,
                       torch::Tensor counters, torch::Tensor ws, int64_t N, int64_t k, int64_t splits,
                       int64_t cfg, int64_t det) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA [M, K]");
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(M >= 1 && M <= 16, "nest_mma supports 1 <= M <= 16, got ", M);
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "need K % 64 == 0 and N % 16 == 0");
  TORCH_CHECK(codes.scalar_type() == at::kByte && codes.is_contiguous() && codes.dim() == 2 &&
                  codes.size(1) == N * K / 8, "codes must be uint8 [L, N*K/8] in mma_tile_v1 order");
  const int L = codes.size(0);
  TORCH_CHECK(k >= 1 && k <= L, "k out of range");
  // codebooks may be the target's [L, 256, 8] or a separate draft set [k_d, 256, 8] (only planes < k are read)
  TORCH_CHECK(codebooks.dim() == 3 && codebooks.size(0) >= k && codebooks.size(1) == 256 && codebooks.size(2) == 8 &&
                  codebooks.is_contiguous(), "codebooks must be contiguous [>=k, 256, 8]");
  TORCH_CHECK(scales.numel() == N && scales.is_contiguous(), "scales [N]");
  TORCH_CHECK(x.scalar_type() == codebooks.scalar_type() && x.scalar_type() == scales.scalar_type(), "dtype mismatch");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.is_cuda(), "counters int32 CUDA");
  auto y = torch::empty({M, (long)N}, x.options());
  if (x.scalar_type() == at::kBFloat16)
    dispatch_m<__nv_bfloat16>(k, cfg, x, codes, codebooks, scales, y, counters, ws, M, N, K, splits, (int)det);
  else if (x.scalar_type() == at::kHalf)
    dispatch_m<half>(k, cfg, x, codes, codebooks, scales, y, counters, ws, M, N, K, splits, (int)det);
  else
    TORCH_CHECK(false, "dtype must be bf16 or fp16");
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("nest_mma", &nest_mma, "nested VQ small-M GEMM (mma, planar)"); }
