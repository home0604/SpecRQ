// Generalized nested additive-VQ kernels ("nestg"): group dim d in {8, 16}, code bits b in {4, 8, 12, 16},
// L <= 10 codebooks with runtime prefix k, codebook values q in {bf16, int8 + per-plane scale, int4 + per-plane scale}.
// bf16 activations only. Two paths: M=1 GEMV (CUDA cores) and M<=16 tensor-core GEMM (mma.m16n8k16), as in
// nest_gemv.cu / nest_mma.cu (the specialized C1 kernels stay the default for d=8, b=8, q=bf16).
//
// Codes ("planar split" layout, pack.pack_general): each plane l is stored as sub-planes so every load is aligned
// and the bytes read are exactly k*b/8 per group:
//   byte  sub-plane  (b >= 8):  low 8 bits of each code
//   nibble sub-plane (b = 4, 12): 4 bits (b=4: the code; b=12: bits 8..11)
//   byte1 sub-plane  (b = 16):  bits 8..15
// GEMV order: groups lane-permuted as pack.kernel_codes, one "chunk" of 4 groups per lane read as u32 (byte) / u16 (nib).
// MMA order : super-tiles of 16 rows x 64 k; lane (g, t) holds n = 64/(4d)*2 codes per plane: d=8 -> 4 codes
//   (rows g, g+8 x groups 2t, 2t+1; j = rh*2 + gp), d=16 -> 2 codes (rows g, g+8 x group t; j = rh).
//   byte sub-plane per lane: d=8 u32, d=16 u16; nibble: d=8 u16, d=16 u8.
// Kernels are instantiated for an upper bound LM in {1, 2, 3, 5, 10} on the prefix k (smallest LM >= k at launch).
// mma k-permutation (same as nest_mma.cu): lane (g,t), step s uses physical k = 16t + 4s + {0..3}; the lane's 16
// physical columns are 16 consecutive elements = groups {2t, 2t+1} (d=8) or group t (d=16).
// Mixed per-plane bits (e.g. K = [4096, 4096, 256, 256]): instantiate B = 12; planes without a nibble sub-plane
// (PInfo.nib[l] = -1) are 8-bit. Codebook planes then differ in size (PInfo.cb_off).
// Codebooks: [L][2^b][d] in q format, plane bytes PB = 2^b * d * q/8. Copied to shared memory when k*PB <= smem_max
// (else read from global/L2: "fallback"). Dequant: k codeword values summed in fp32; mma rounds the sum once to bf16
// (as nest_mma), GEMV keeps fp32 (as nest_gemv).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int MAXL = 10;

// Per-plane layout (host-computed): codebook byte offset of plane l (cb_off[L] = total) and the index of plane l's
// nibble sub-plane (-1 when the plane has none). Lets planes mix 8-bit and 12-bit codes (B = 12 instantiation):
// an 8-bit plane simply has no nibble sub-plane and its code is the byte value.
struct PInfo {
  long long cb_off[MAXL + 1];
  int nib[MAXL];
  int R;  // shared-memory replication of each codeword (lane % R picks the copy); 1 for the global fallback
};

template <int D, int Q> struct Ent { static constexpr int E = D * Q / 8; };

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// w[0..D) += codeword(p) * sc   (p: generic pointer to smem or global, 16B/8B/4B aligned as needed)
template <int D, int Q>
__device__ __forceinline__ void add_entry(float* w, const uint8_t* p, float sc) {
  if constexpr (Q == 16) {
#pragma unroll
    for (int i = 0; i < D / 8; i++) {
      const uint4 v = reinterpret_cast<const uint4*>(p)[i];
      const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        w[8 * i + 2 * q] += bf_lo(u[q]);
        w[8 * i + 2 * q + 1] += bf_hi(u[q]);
      }
    }
  } else if constexpr (Q == 8) {
    uint32_t u[D / 4];
    if constexpr (D == 8) {
      const uint2 v = *reinterpret_cast<const uint2*>(p);
      u[0] = v.x; u[1] = v.y;
    } else {
      const uint4 v = *reinterpret_cast<const uint4*>(p);
      u[0] = v.x; u[1] = v.y; u[2] = v.z; u[3] = v.w;
    }
#pragma unroll
    for (int i = 0; i < D / 4; i++)
#pragma unroll
      for (int q = 0; q < 4; q++) w[4 * i + q] += (float)(int8_t)((u[i] >> (8 * q)) & 0xffu) * sc;
  } else {  // Q == 4: two signed nibbles per byte, element 2e in the low nibble
    uint32_t u[D / 8];
    if constexpr (D == 8) {
      u[0] = *reinterpret_cast<const uint32_t*>(p);
    } else {
      const uint2 v = *reinterpret_cast<const uint2*>(p);
      u[0] = v.x; u[1] = v.y;
    }
#pragma unroll
    for (int i = 0; i < D / 8; i++)
#pragma unroll
      for (int q = 0; q < 8; q++) {
        const int nib = (int)((u[i] >> (4 * q)) & 0xfu);
        w[8 * i + q] += (float)(nib >= 8 ? nib - 16 : nib) * sc;
      }
  }
}

template <int B>
__device__ __forceinline__ uint32_t code_of(uint32_t c0, uint32_t cn, uint32_t c1, int j) {
  if constexpr (B == 4) return (cn >> (4 * j)) & 0xfu;
  else if constexpr (B == 8) return (c0 >> (8 * j)) & 0xffu;
  else if constexpr (B == 12) return ((c0 >> (8 * j)) & 0xffu) | (((cn >> (4 * j)) & 0xfu) << 8);
  else return ((c0 >> (8 * j)) & 0xffu) | (((c1 >> (8 * j)) & 0xffu) << 8);
}

// copy planes [0, k) to shared memory, each E-byte codeword replicated R times: plane l at cb_off[l]*R,
// codeword c copy r at ((c*R + r) * E). R == 1: plain 16-byte copy. E is a power of two (4..32).
template <int E>
__device__ __forceinline__ void copy_cb(uint8_t* sh, const uint8_t* cb, const PInfo& pi, int k) {
  const int bytes = (int)pi.cb_off[k];
  if (pi.R == 1) {
    const uint4* s = reinterpret_cast<const uint4*>(cb);
    uint4* d = reinterpret_cast<uint4*>(sh);
    for (int i = threadIdx.x; i < bytes / 16; i += blockDim.x) d[i] = s[i];
  } else {
    constexpr int LOG_E = E == 4 ? 2 : E == 8 ? 3 : E == 16 ? 4 : 5;
    const uint32_t* s = reinterpret_cast<const uint32_t*>(cb);
    for (int w = threadIdx.x; w < bytes / 4; w += blockDim.x) {
      const int off = w * 4;
      int l = 0;
      while (l + 1 < k && off >= (int)pi.cb_off[l + 1]) l++;
      const int loc = off - (int)pi.cb_off[l], ent = loc >> LOG_E, within = loc & (E - 1);
      const uint32_t v = s[w];
      uint8_t* dst = sh + (int)pi.cb_off[l] * pi.R + within;
      for (int r = 0; r < pi.R; r++) *reinterpret_cast<uint32_t*>(dst + ((ent * pi.R + r) << LOG_E)) = v;
    }
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ GEMV (M = 1)
template <int D, int B, int Q, int LM>
__global__ void __launch_bounds__(256) nestg_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ p0, const uint16_t* __restrict__ pn,
    const uint32_t* __restrict__ p1, const uint8_t* __restrict__ cb, const float* __restrict__ cbs,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ y, int N, int K, int k, int use_smem,
    const PInfo pi) {
  constexpr int E = Ent<D, Q>::E;
  extern __shared__ __align__(16) uint8_t shm[];
  if (use_smem) copy_cb<E>(shm, cb, pi, k);
  const uint8_t* cbase = use_smem ? shm : cb;
  float sc[LM];
  int po[LM];  // plane byte offsets (replicated layout in smem), 32-bit
#pragma unroll
  for (int l = 0; l < LM; l++) {
    sc[l] = (Q < 16 && l < k) ? cbs[l] : 1.f;
    po[l] = (int)pi.cb_off[l] * (use_smem ? pi.R : 1);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int R = use_smem ? pi.R : 1, rr = lane % R;
  const int G = K / D, nch = G / 4;
  const long long pl = (long long)N * nch;  // u32 (byte) or u16 (nib) words per plane
  const uint4* xg = reinterpret_cast<const uint4*>(x);
  for (int row = blockIdx.x * nw + warp; row < N; row += gridDim.x * nw) {
    float acc = 0.f;
    uint32_t c0[LM], cn[LM], c1[LM], n0[LM], nn[LM], n1[LM];
    auto load = [&](int ch_, uint32_t (&a0)[LM], uint32_t (&an)[LM], uint32_t (&a1)[LM]) {
#pragma unroll
      for (int l = 0; l < LM; l++) {
        a0[l] = an[l] = a1[l] = 0u;
        if (l < k && ch_ < nch) {
          const long long o = l * pl + (long long)row * nch + ch_;
          if constexpr (B >= 8) a0[l] = __ldg(p0 + o);
          if constexpr (B == 4 || B == 12)
            if (pi.nib[l] >= 0) an[l] = __ldg(pn + pi.nib[l] * pl + (long long)row * nch + ch_);
          if constexpr (B == 16) a1[l] = __ldg(p1 + o);
        }
      }
    };
    load(lane, c0, cn, c1);
    for (int ch = lane; ch < nch; ch += 32) {
      load(ch + 32, n0, nn, n1);  // prefetch the next chunk's codes
      const int blk = ch >> 5, jj = ch & 31, wd = min(32, nch - 32 * blk);
#pragma unroll
      for (int g = 0; g < 4; g++) {
        float w[D];
#pragma unroll
        for (int e = 0; e < D; e++) w[e] = 0.f;
#pragma unroll
        for (int l = 0; l < LM; l++)
          if (l < k)
            add_entry<D, Q>(w, cbase + po[l] + ((int)code_of<B>(c0[l], cn[l], c1[l], g) * R + rr) * E, sc[l]);
        const int gi = 128 * blk + g * wd + jj;
#pragma unroll
        for (int i = 0; i < D / 8; i++) {
          const uint4 xv = __ldg(xg + (long long)gi * (D / 8) + i);
          const uint32_t u[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            acc = fmaf(bf_lo(u[q]), w[8 * i + 2 * q], acc);
            acc = fmaf(bf_hi(u[q]), w[8 * i + 2 * q + 1], acc);
          }
        }
      }
#pragma unroll
      for (int l = 0; l < LM; l++) { c0[l] = n0[l]; cn[l] = nn[l]; c1[l] = n1[l]; }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = __float2bfloat16(acc * __bfloat162float(scales[row]));
  }
}

// Register-resident x variant (as nest_gemv xreg): rows with nch <= 32*CPL chunks; each lane keeps the x slices of its
// CPL chunks in registers for all rows and prefetches the next row's codes. Used for LM <= 5.
template <int D, int B, int Q, int LM, int CPL>
__global__ void __launch_bounds__(256) nestg_gemv_xreg_kernel(
    const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ p0, const uint16_t* __restrict__ pn,
    const uint32_t* __restrict__ p1, const uint8_t* __restrict__ cb, const float* __restrict__ cbs,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ y, int N, int K, int k, int use_smem,
    const PInfo pi) {
  constexpr int E = Ent<D, Q>::E;
  extern __shared__ __align__(16) uint8_t shm[];
  if (use_smem) copy_cb<E>(shm, cb, pi, k);
  const uint8_t* cbase = use_smem ? shm : cb;
  float sc[LM];
  int po[LM];
#pragma unroll
  for (int l = 0; l < LM; l++) {
    sc[l] = (Q < 16 && l < k) ? cbs[l] : 1.f;
    po[l] = (int)pi.cb_off[l] * (use_smem ? pi.R : 1);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const int R = use_smem ? pi.R : 1, rr = lane % R;
  const int G = K / D, nch = G / 4;
  const long long pl = (long long)N * nch;
  const uint4* xg = reinterpret_cast<const uint4*>(x);
  uint4 xr[CPL][4][D / 8];
  bool act[CPL];
#pragma unroll
  for (int u = 0; u < CPL; u++) {
    const int ch = lane + 32 * u;
    act[u] = ch < nch;
    const int blk = ch >> 5, jj = ch & 31, wd = min(32, nch - 32 * blk);
#pragma unroll
    for (int g = 0; g < 4; g++)
#pragma unroll
      for (int i = 0; i < D / 8; i++)
        xr[u][g][i] = act[u] ? __ldg(xg + (long long)(128 * blk + g * wd + jj) * (D / 8) + i) : make_uint4(0, 0, 0, 0);
  }
  auto load = [&](int r, uint32_t (&a0)[CPL][LM], uint32_t (&an)[CPL][LM], uint32_t (&a1)[CPL][LM]) {
#pragma unroll
    for (int u = 0; u < CPL; u++)
#pragma unroll
      for (int l = 0; l < LM; l++) {
        a0[u][l] = an[u][l] = a1[u][l] = 0u;
        if (l < k && act[u]) {
          const long long o = (long long)r * nch + lane + 32 * u;
          if constexpr (B >= 8) a0[u][l] = __ldg(p0 + l * pl + o);
          if constexpr (B == 4 || B == 12)
            if (pi.nib[l] >= 0) an[u][l] = __ldg(pn + pi.nib[l] * pl + o);
          if constexpr (B == 16) a1[u][l] = __ldg(p1 + l * pl + o);
        }
      }
  };
  const int stride = gridDim.x * nw;
  int row = blockIdx.x * nw + warp;
  uint32_t c0[CPL][LM], cn[CPL][LM], c1[CPL][LM];
  if (row < N) load(row, c0, cn, c1);
  while (row < N) {
    const int nxt = row + stride;
    uint32_t n0[CPL][LM], nn[CPL][LM], n1[CPL][LM];
    if (nxt < N) load(nxt, n0, nn, n1);
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < CPL; u++) {
      if (!act[u]) continue;
#pragma unroll
      for (int g = 0; g < 4; g++) {
        float w[D];
#pragma unroll
        for (int e = 0; e < D; e++) w[e] = 0.f;
#pragma unroll
        for (int l = 0; l < LM; l++)
          if (l < k)
            add_entry<D, Q>(w, cbase + po[l] + ((int)code_of<B>(c0[u][l], cn[u][l], c1[u][l], g) * R + rr) * E, sc[l]);
#pragma unroll
        for (int i = 0; i < D / 8; i++) {
          const uint32_t uu[4] = {xr[u][g][i].x, xr[u][g][i].y, xr[u][g][i].z, xr[u][g][i].w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            acc = fmaf(bf_lo(uu[q]), w[8 * i + 2 * q], acc);
            acc = fmaf(bf_hi(uu[q]), w[8 * i + 2 * q + 1], acc);
          }
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = __float2bfloat16(acc * __bfloat162float(scales[row]));
#pragma unroll
    for (int u = 0; u < CPL; u++)
#pragma unroll
      for (int l = 0; l < LM; l++) { c0[u][l] = n0[u][l]; cn[u][l] = nn[u][l]; c1[u][l] = n1[u][l]; }
    row = nxt;
  }
}

// ------------------------------------------------------------------------------------------------ MMA (M <= 16)
constexpr int WARPS = 4, TPW = 2, ROWS_PER_BLOCK = WARPS * TPW * 16;

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf(float a, float b) {
  __nv_bfloat162 r = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t comp4(const uint4& v, int s) { return s == 0 ? v.x : s == 1 ? v.y : s == 2 ? v.z : v.w; }

template <int D, int B>
__device__ __forceinline__ void load_codes_mma(const uint8_t* __restrict__ p0, const uint8_t* __restrict__ pn,
                                               const uint8_t* __restrict__ p1, long long idx, long long plane_lanes,
                                               int l, int nl, uint32_t& c0, uint32_t& cn, uint32_t& c1) {
  // idx = (tile * nsup + ks) * 32 + lane; per-lane element = n codes; nl = nibble sub-plane index (-1: none)
  const long long o = l * plane_lanes + idx, on = (long long)nl * plane_lanes + idx;
  if constexpr (D == 8) {
    if constexpr (B >= 8) c0 = __ldg(reinterpret_cast<const uint32_t*>(p0) + o);
    if constexpr (B == 4 || B == 12)
      if (nl >= 0) cn = __ldg(reinterpret_cast<const uint16_t*>(pn) + on);
    if constexpr (B == 16) c1 = __ldg(reinterpret_cast<const uint32_t*>(p1) + o);
  } else {
    if constexpr (B >= 8) c0 = __ldg(reinterpret_cast<const uint16_t*>(p0) + o);
    if constexpr (B == 4 || B == 12)
      if (nl >= 0) cn = __ldg(pn + on);
    if constexpr (B == 16) c1 = __ldg(reinterpret_cast<const uint16_t*>(p1) + o);
  }
}

template <int D, int B, int Q, int NT, int LM>
__global__ void __launch_bounds__(WARPS * 32) nestg_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ p0, const uint8_t* __restrict__ pn,
    const uint8_t* __restrict__ p1, const uint8_t* __restrict__ cb, const float* __restrict__ cbs,
    const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
    int* __restrict__ counters, int M, int N, int K, int k, int ks_per_split, int use_smem, const PInfo pi) {
  constexpr int E = Ent<D, Q>::E;
  extern __shared__ __align__(16) uint8_t shm[];
  if (use_smem) copy_cb<E>(shm, cb, pi, k);
  const uint8_t* cbase = use_smem ? shm : cb;
  float sc[LM];
  int po[LM];
#pragma unroll
  for (int l = 0; l < LM; l++) {
    sc[l] = (Q < 16 && l < k) ? cbs[l] : 1.f;
    po[l] = (int)pi.cb_off[l] * (use_smem ? pi.R : 1);
  }

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int R = use_smem ? pi.R : 1, rr = lane % R;
  const int nsup = K >> 6, ks0 = blockIdx.y * ks_per_split, ks1 = min(nsup, ks0 + ks_per_split);
  const int tile0 = blockIdx.x * (ROWS_PER_BLOCK / 16) + warp * TPW, ntiles = N >> 4;
  const long long plane_lanes = (long long)ntiles * nsup * 32;
  float acc[TPW][NT][4] = {};
  bool tv[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) tv[i] = tile0 + i < ntiles;

  uint32_t c0[TPW][LM], cn[TPW][LM], c1[TPW][LM], n0[TPW][LM], nn[TPW][LM], n1[TPW][LM];
  auto load = [&](int ks_, uint32_t (&a0)[TPW][LM], uint32_t (&an)[TPW][LM], uint32_t (&a1)[TPW][LM]) {
#pragma unroll
    for (int i = 0; i < TPW; i++)
#pragma unroll
      for (int l = 0; l < LM; l++) {
        a0[i][l] = an[i][l] = a1[i][l] = 0u;
        if (l < k && tv[i] && ks_ < ks1)
          load_codes_mma<D, B>(p0, pn, p1, ((long long)(tile0 + i) * nsup + ks_) * 32 + lane, plane_lanes, l,
                               pi.nib[l], a0[i][l], an[i][l], a1[i][l]);
      }
  };
  load(ks0, c0, cn, c1);
  for (int ks = ks0; ks < ks1; ks++) {
    load(ks + 1, n0, nn, n1);  // prefetch next super-tile codes
    uint4 xv[NT][2];
#pragma unroll
    for (int j = 0; j < NT; j++) {
      const int m = j * 8 + g;
      if (m < M) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + (long long)m * K + ks * 64 + 16 * t);
        xv[j][0] = __ldg(xr);
        xv[j][1] = __ldg(xr + 1);
      } else {
        xv[j][0] = xv[j][1] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      if (!tv[i]) continue;
      uint32_t W[2][8];
#pragma unroll
      for (int rh = 0; rh < 2; rh++) {
        float w[16];
#pragma unroll
        for (int e = 0; e < 16; e++) w[e] = 0.f;
#pragma unroll
        for (int l = 0; l < LM; l++) {
          if (l < k) {
            const uint8_t* pb = cbase + po[l];
            if constexpr (D == 8) {
#pragma unroll
              for (int gp = 0; gp < 2; gp++)
                add_entry<8, Q>(w + 8 * gp,
                                pb + ((int)code_of<B>(c0[i][l], cn[i][l], c1[i][l], rh * 2 + gp) * R + rr) * E,
                                sc[l]);
            } else {
              add_entry<16, Q>(w, pb + ((int)code_of<B>(c0[i][l], cn[i][l], c1[i][l], rh) * R + rr) * E, sc[l]);
            }
          }
        }
#pragma unroll
        for (int e = 0; e < 8; e++) W[rh][e] = pack_bf(w[2 * e], w[2 * e + 1]);
      }
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const uint32_t a[4] = {W[0][2 * s], W[1][2 * s], W[0][2 * s + 1], W[1][2 * s + 1]};
#pragma unroll
        for (int j = 0; j < NT; j++)
          mma16816(acc[i][j], a, comp4(xv[j][s >> 1], (s & 1) * 2), comp4(xv[j][s >> 1], (s & 1) * 2 + 1));
      }
    }
#pragma unroll
    for (int i = 0; i < TPW; i++)
#pragma unroll
      for (int l = 0; l < LM; l++) { c0[i][l] = n0[i][l]; cn[i][l] = nn[i][l]; c1[i][l] = n1[i][l]; }
  }
  const bool split = gridDim.y > 1;
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    if (!tv[i]) continue;
    const int nb = (tile0 + i) * 16 + g;
    const float s0 = __bfloat162float(scales[nb]), s1 = __bfloat162float(scales[nb + 8]);
#pragma unroll
    for (int j = 0; j < NT; j++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int m = j * 8 + 2 * t + (q & 1), n = nb + (q >> 1) * 8;
        if (m < M) {
          const float v = acc[i][j][q] * ((q >> 1) ? s1 : s0);
          if (split) ws[((long long)blockIdx.y * M + m) * N + n] = v;
          else y[(long long)m * N + n] = __float2bfloat16(v);
        }
      }
  }
  if (!split) return;
  __shared__ int is_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = atomicAdd(&counters[blockIdx.x], 1) == (int)gridDim.y - 1;
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const int row0 = blockIdx.x * ROWS_PER_BLOCK, nrows = min(ROWS_PER_BLOCK, N - row0);
  for (int e = threadIdx.x; e < M * nrows; e += blockDim.x) {
    const int m = e / nrows, n = row0 + e % nrows;
    float v = 0.f;
    for (int sk = 0; sk < (int)gridDim.y; sk++) v += __ldcg(ws + ((long long)sk * M + m) * N + n);
    y[(long long)m * N + n] = __float2bfloat16(v);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;
}

// ------------------------------------------------------------------------------------------------ host
struct Plan {
  long long smem_need;  // k * PB
  int use_smem, smem_launch, blocks_per_sm, grid_x, grid_y, R;
};

int sms_of(int dev) {
  int s = 0;
  cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, dev);
  return s;
}

constexpr long long REPL_BUDGET = 32 * 1024;  // replicate codewords while the copies fit in 32 KB (as nest_gemv)

template <typename Kern>
Plan make_plan(Kern kern, int threads, PInfo& pi, int k, long long smem_max, int dev) {
  Plan p{};
  p.smem_need = pi.cb_off[k];
  p.use_smem = p.smem_need <= smem_max ? 1 : 0;
  pi.R = 1;
  if (p.use_smem)
    while (pi.R < 8 && p.smem_need * pi.R * 2 <= REPL_BUDGET) pi.R *= 2;
  p.smem_launch = p.use_smem ? (int)(p.smem_need * pi.R) : 0;
  if (p.smem_launch > 0) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, p.smem_launch);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p.blocks_per_sm, kern, threads, p.smem_launch);
  p.R = pi.R;
  return p;
}

template <int D, int B, int Q, int LM>
Plan gemv_launch(const torch::Tensor& x, const torch::Tensor& p0, const torch::Tensor& pn, const torch::Tensor& p1,
                 const torch::Tensor& cb, const torch::Tensor& cbs, const torch::Tensor& sc, torch::Tensor* y, int N,
                 int K, int k, long long smem_max, PInfo pi) {
  const int nch = K / D / 4;
  auto kern = nestg_gemv_kernel<D, B, Q, LM>;
  if constexpr (LM <= 5) {  // register-resident x when the row fits in 2 or 4 chunks per lane
    if (nch <= 64) kern = nestg_gemv_xreg_kernel<D, B, Q, LM, 2>;
    else if (nch <= 128) kern = nestg_gemv_xreg_kernel<D, B, Q, LM, 4>;
  }
  const int dev = x.get_device();
  Plan p = make_plan(kern, 256, pi, k, smem_max, dev);
  p.grid_x = std::max(1, std::min(sms_of(dev) * std::max(p.blocks_per_sm, 1), (N + 7) / 8));
  p.grid_y = 1;
  if (y) {
    kern<<<p.grid_x, 256, p.smem_launch, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        B >= 8 ? reinterpret_cast<const uint32_t*>(p0.data_ptr()) : nullptr,
        (B == 4 || B == 12) ? reinterpret_cast<const uint16_t*>(pn.data_ptr()) : nullptr,
        B == 16 ? reinterpret_cast<const uint32_t*>(p1.data_ptr()) : nullptr,
        reinterpret_cast<const uint8_t*>(cb.data_ptr()), Q < 16 ? cbs.data_ptr<float>() : nullptr,
        reinterpret_cast<const __nv_bfloat16*>(sc.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y->data_ptr()), N, K, k,
        p.use_smem, pi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return p;
}

template <int D, int B, int Q, int NT, int LM>
Plan mma_launch(const torch::Tensor& x, const torch::Tensor& p0, const torch::Tensor& pn, const torch::Tensor& p1,
                const torch::Tensor& cb, const torch::Tensor& cbs, const torch::Tensor& sc, torch::Tensor* y,
                torch::Tensor* counters, int M, int N, int K, int k, long long smem_max, int splits, PInfo pi) {
  auto kern = nestg_mma_kernel<D, B, Q, NT, LM>;
  const int dev = x.get_device();
  Plan p = make_plan(kern, WARPS * 32, pi, k, smem_max, dev);
  const int nbN = (N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, nsup = K / 64;
  if (splits <= 0) splits = std::max(1, std::min(sms_of(dev) * std::max(p.blocks_per_sm, 1) / nbN, nsup / 2));
  splits = std::max(1, std::min(splits, nsup));
  const int per = (nsup + splits - 1) / splits;
  splits = (nsup + per - 1) / per;
  p.grid_x = nbN;
  p.grid_y = splits;
  if (y) {
    TORCH_CHECK(counters->numel() >= nbN, "counter buffer too small");
    torch::Tensor wsd;
    if (splits > 1) wsd = torch::empty({(long)splits * M * N}, x.options().dtype(torch::kFloat32));
    kern<<<dim3(nbN, splits), WARPS * 32, p.smem_launch, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), B >= 8 ? reinterpret_cast<const uint8_t*>(p0.data_ptr()) : nullptr,
        (B == 4 || B == 12) ? reinterpret_cast<const uint8_t*>(pn.data_ptr()) : nullptr,
        B == 16 ? reinterpret_cast<const uint8_t*>(p1.data_ptr()) : nullptr, reinterpret_cast<const uint8_t*>(cb.data_ptr()),
        Q < 16 ? cbs.data_ptr<float>() : nullptr, reinterpret_cast<const __nv_bfloat16*>(sc.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y->data_ptr()), splits > 1 ? wsd.data_ptr<float>() : nullptr,
        counters->data_ptr<int>(), M, N, K, k, per, p.use_smem, pi);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return p;
}

#define DISPATCH_DBQ(D_, B_, Q_, ...)                                                              \
  [&] {                                                                                            \
    if (D_ == 8 && B_ == 4 && Q_ == 16) { constexpr int D = 8, B = 4, Q = 16; return __VA_ARGS__(); }   \
    if (D_ == 8 && B_ == 4 && Q_ == 8) { constexpr int D = 8, B = 4, Q = 8; return __VA_ARGS__(); }     \
    if (D_ == 8 && B_ == 4 && Q_ == 4) { constexpr int D = 8, B = 4, Q = 4; return __VA_ARGS__(); }     \
    if (D_ == 8 && B_ == 8 && Q_ == 16) { constexpr int D = 8, B = 8, Q = 16; return __VA_ARGS__(); }   \
    if (D_ == 8 && B_ == 8 && Q_ == 8) { constexpr int D = 8, B = 8, Q = 8; return __VA_ARGS__(); }     \
    if (D_ == 8 && B_ == 8 && Q_ == 4) { constexpr int D = 8, B = 8, Q = 4; return __VA_ARGS__(); }     \
    if (D_ == 8 && B_ == 12 && Q_ == 16) { constexpr int D = 8, B = 12, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 8 && B_ == 12 && Q_ == 8) { constexpr int D = 8, B = 12, Q = 8; return __VA_ARGS__(); }   \
    if (D_ == 8 && B_ == 12 && Q_ == 4) { constexpr int D = 8, B = 12, Q = 4; return __VA_ARGS__(); }   \
    if (D_ == 8 && B_ == 16 && Q_ == 16) { constexpr int D = 8, B = 16, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 8 && B_ == 16 && Q_ == 8) { constexpr int D = 8, B = 16, Q = 8; return __VA_ARGS__(); }   \
    if (D_ == 8 && B_ == 16 && Q_ == 4) { constexpr int D = 8, B = 16, Q = 4; return __VA_ARGS__(); }   \
    if (D_ == 16 && B_ == 4 && Q_ == 16) { constexpr int D = 16, B = 4, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 4 && Q_ == 8) { constexpr int D = 16, B = 4, Q = 8; return __VA_ARGS__(); }   \
    if (D_ == 16 && B_ == 4 && Q_ == 4) { constexpr int D = 16, B = 4, Q = 4; return __VA_ARGS__(); }   \
    if (D_ == 16 && B_ == 8 && Q_ == 16) { constexpr int D = 16, B = 8, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 8 && Q_ == 8) { constexpr int D = 16, B = 8, Q = 8; return __VA_ARGS__(); }   \
    if (D_ == 16 && B_ == 8 && Q_ == 4) { constexpr int D = 16, B = 8, Q = 4; return __VA_ARGS__(); }   \
    if (D_ == 16 && B_ == 12 && Q_ == 16) { constexpr int D = 16, B = 12, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 12 && Q_ == 8) { constexpr int D = 16, B = 12, Q = 8; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 12 && Q_ == 4) { constexpr int D = 16, B = 12, Q = 4; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 16 && Q_ == 16) { constexpr int D = 16, B = 16, Q = 16; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 16 && Q_ == 8) { constexpr int D = 16, B = 16, Q = 8; return __VA_ARGS__(); } \
    if (D_ == 16 && B_ == 16 && Q_ == 4) { constexpr int D = 16, B = 16, Q = 4; return __VA_ARGS__(); } \
    TORCH_CHECK(false, "unsupported (d, b, q) = (", D_, ", ", B_, ", ", Q_, ")");                 \
    return Plan{};                                                                                 \
  }()

void check_common(const torch::Tensor& x, int d, int b, int q, int k, int L) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2, "x bf16 [M, K] contiguous");
  TORCH_CHECK(d == 8 || d == 16, "d must be 8 or 16");
  TORCH_CHECK(b == 4 || b == 8 || b == 12 || b == 16, "b must be 4/8/12/16");
  TORCH_CHECK(q == 16 || q == 8 || q == 4, "q must be 16/8/4");
  TORCH_CHECK(k >= 1 && k <= L && L <= MAXL, "need 1 <= k <= L <= 10");
}

PInfo make_pinfo(const torch::Tensor& cb_off, const torch::Tensor& nib, int L) {
  TORCH_CHECK(cb_off.device().is_cpu() && cb_off.scalar_type() == at::kLong && cb_off.numel() == L + 1, "cb_off cpu long [L+1]");
  TORCH_CHECK(nib.device().is_cpu() && nib.scalar_type() == at::kLong && nib.numel() == L, "nib cpu long [L]");
  PInfo pi{};
  auto co = cb_off.accessor<int64_t, 1>();
  auto nb = nib.accessor<int64_t, 1>();
  for (int l = 0; l <= MAXL; l++) pi.cb_off[l] = l <= L ? co[l] : co[L];
  for (int l = 0; l < MAXL; l++) pi.nib[l] = l < L ? (int)nb[l] : -1;
  return pi;
}

std::vector<int64_t> plan_vec(const Plan& p) {
  return {p.smem_need, p.use_smem, p.smem_launch, p.blocks_per_sm, p.grid_x, p.grid_y, p.R};
}

}  // namespace

// launch helpers used inside DISPATCH_DBQ bodies (LM = smallest of {1,2,3,5,10} >= k)
#define G_(LM) return gemv_launch<D, B, Q, LM>(x, p0, pn, p1, cb, cbs, scales, yp, N, K, k, smem_max, pi)
#define M_(NT, LM) \
  return mma_launch<D, B, Q, NT, LM>(x, p0, pn, p1, cb, cbs, scales, yp, &counters, M, N, K, k, smem_max, splits, pi)
#define BY_K(NT) { if (k <= 1) M_(NT, 1); if (k <= 2) M_(NT, 2); if (k <= 3) M_(NT, 3); if (k <= 5) M_(NT, 5); M_(NT, 10); }

// p0/pn/p1: sub-planes (empty tensors when unused); cb: uint8 codebook bytes [L * PB]; cbs: fp32 [L] (q < 16);
// scales bf16 [N]. dry_run: return the launch plan only.
std::vector<torch::Tensor> nestg_gemv(torch::Tensor x, torch::Tensor p0, torch::Tensor pn, torch::Tensor p1,
                                      torch::Tensor cb, torch::Tensor cbs, torch::Tensor scales, torch::Tensor cb_off,
                                      torch::Tensor nib, int64_t N, int64_t d, int64_t b, int64_t q, int64_t L, int64_t k,
                                      int64_t smem_max, bool dry_run) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  check_common(x, d, b, q, k, L);
  const int K = x.size(1);
  TORCH_CHECK(x.size(0) == 1, "nestg_gemv is the M=1 path");
  TORCH_CHECK(K % (4 * d) == 0, "K must be a multiple of 4d");
  torch::Tensor y = dry_run ? torch::Tensor() : torch::empty({1, N}, x.options());
  const PInfo pi = make_pinfo(cb_off, nib, L);
  Plan p = DISPATCH_DBQ(d, b, q, [&] {
    torch::Tensor* yp = dry_run ? nullptr : &y;
    if (k <= 1) G_(1);
    if (k <= 2) G_(2);
    if (k <= 3) G_(3);
    if (k <= 5) G_(5);
    G_(10);
  });
  auto pv = plan_vec(p);
  return {y, torch::tensor(pv, torch::kLong)};
}

std::vector<torch::Tensor> nestg_mma(torch::Tensor x, torch::Tensor p0, torch::Tensor pn, torch::Tensor p1,
                                     torch::Tensor cb, torch::Tensor cbs, torch::Tensor scales, torch::Tensor counters,
                                     torch::Tensor cb_off, torch::Tensor nib, int64_t N, int64_t d, int64_t b, int64_t q,
                                     int64_t L, int64_t k, int64_t smem_max, int64_t splits, bool dry_run) {
  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  check_common(x, d, b, q, k, L);
  const int M = x.size(0), K = x.size(1);
  TORCH_CHECK(M >= 1 && M <= 16, "M <= 16");
  TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "K % 64 == 0 and N % 16 == 0");
  torch::Tensor y = dry_run ? torch::Tensor() : torch::empty({M, N}, x.options());
  const PInfo pi = make_pinfo(cb_off, nib, L);
  Plan p = DISPATCH_DBQ(d, b, q, [&] {
    torch::Tensor* yp = dry_run ? nullptr : &y;
    if (M <= 8) BY_K(1)
    BY_K(2)
  });
  auto pv = plan_vec(p);
  return {y, torch::tensor(pv, torch::kLong)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nestg_gemv", &nestg_gemv, "generalized nested-VQ GEMV (M=1)");
  m.def("nestg_mma", &nestg_mma, "generalized nested-VQ small-M GEMM (mma)");
}
