// E6 kvregime v2: nested VQ-KV decode attention with ADC/LUT scores (keys) and register-FMA values.
// Format (d = 8, D = 128, NSV = 16 sub-vectors): stage s has 2^bits[s] codewords (bits 8: one byte per sub-vector,
// bits 4: two sub-vectors per byte, low nibble = even sub-vector); planes are stage-major [B, Hkv, T, W_s] per K / V.
// Per block (b, kv-head h, split): LUT_s[j][c][r] = q_r[j] . Ck_s[c] (fp16, r = G*M stacked query rows, pre-scaled
// by D^-0.5 log2 e, q RoPE fused) built once in shared memory; V codebooks in shared memory.
// Each warp walks 32-token tiles of the block's chunk independently (tiles w, w + NW, ...):
//   K : lane = token; score_r = sum_{s<KST} sum_j LUT_s[j][code][r] (one 8/16-byte gather per (s, j) gives 4/8 rows)
//   softmax: warp shuffles (exact online softmax, fp32 p)
//   V : lane = (8-dim group dg, token parity); v = sum_s Cv_s[code] (16-byte gathers), acc[r][8] += p_r * v
// Each warp writes its own partial (O, m, l) as split sp * NW + warp; merged by the Triton combine.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>

using namespace nvcuda;

namespace {
constexpr int D = 128, DS = 8, NSV = 16, G = 4;

struct KVP {
  const uint8_t* kc[4];
  const uint8_t* vc[4];
  long sb[4], sh[4];  // plane strides (batch, head); token stride = plane width W_s, element stride 1
  const __nv_bfloat16* kcb[4];
  const __nv_bfloat16* vcb[4];
  long cbh[4];  // codebook head stride (codeword stride DS)
};

// format id -> per-stage code bits
template <int FMT>
struct Fmt;
template <>
struct Fmt<0> { static constexpr int S = 2; __host__ __device__ static constexpr int bits(int s) { return 8; } };
template <>
struct Fmt<1> { static constexpr int S = 4; __host__ __device__ static constexpr int bits(int s) { return 4; } };
template <>
struct Fmt<2> {
  static constexpr int S = 3;
  __host__ __device__ static constexpr int bits(int s) { return s < 2 ? 4 : 8; }
};

// The first KST stages are read as "groups": an 8-bit stage (G8), a lone 4-bit stage (G4), or a pair of consecutive
// 4-bit stages (G44) whose two nibbles form one 8-bit index into a 256-entry sum table (C_a[i] + C_b[j]) -- a pair of
// K16 stages then costs one gather per sub-vector, like one K256 stage.
enum { G8 = 0, G4 = 1, G44 = 2 };
template <int FMT, int KST, bool PAIR>
struct Grp {
  __host__ __device__ static constexpr int type_at(int s) {  // group type starting at stage s
    return Fmt<FMT>::bits(s) == 8 ? G8 : ((PAIR && s + 1 < KST && Fmt<FMT>::bits(s + 1) == 4) ? G44 : G4);
  }
  __host__ __device__ static constexpr int width(int s) { return type_at(s) == G44 ? 2 : 1; }
  __host__ __device__ static constexpr int start(int g) { return g == 0 ? 0 : start(g - 1) + width(start(g - 1)); }
  __host__ __device__ static constexpr int count(int s = 0) { return s >= KST ? 0 : 1 + count(s + width(s)); }
  static constexpr int NG = count();
  __host__ __device__ static constexpr int type(int g) { return type_at(start(g)); }
  __host__ __device__ static constexpr int C(int g) { return type(g) == G4 ? 16 : 256; }
};

__host__ __device__ constexpr int al16(int x) { return (x + 15) / 16 * 16; }
constexpr int LDV = D + 8, LDP = 40;  // TC path: V tile [32][LDV] bf16, P [BMQ][LDP] bf16

// M == 1: K16 pairs merged (LUT 16 x 256 x 4 rows fits), V by register FMA.
// M >= 2: no pairs (LUT size grows with rows), P V on tensor cores (WMMA) from a warp-private bf16 V tile.
template <int FMT, int KST, int M, int NW>
struct Lay {
  using GR = Grp<FMT, KST, M == 1>;
  static constexpr bool TC = M >= 2;
  static constexpr int NG = GR::NG;
  static constexpr int GM = G * M;
  static constexpr int BMQ = GM <= 16 ? 16 : 32;
  static constexpr int tc_v = 0, tc_p = al16(32 * LDV * 2), tc_a = al16(tc_p + BMQ * LDP * 2),
                       tc_bytes = al16(tc_a + BMQ * 4);
  __host__ __device__ static constexpr int vcb_off(int g) { return g == 0 ? 0 : vcb_off(g - 1) + GR::C(g - 1) * DS * 2; }
  __host__ __device__ static constexpr int lut_off(int g) {
    return g == 0 ? al16(vcb_off(NG)) : lut_off(g - 1) + NSV * GR::C(g - 1) * GM * 2;
  }
  static constexpr int warp_bytes = TC ? tc_bytes : NG * 32 * 16 + GM * 32 * 4;  // FMA: V codes + p [32][GM]
  static constexpr int warp_off = al16(lut_off(NG));
  static constexpr int qbytes = GM * D * 4;
  static constexpr int tail = NW * warp_bytes > qbytes ? NW * warp_bytes : qbytes;  // q scratch overlaps warps
  static constexpr int total = warp_off + tail;
};

__device__ __forceinline__ void add8(float* a, uint4 w) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(p[i]);
    a[2 * i] += f.x;
    a[2 * i + 1] += f.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float* a) {
  uint4 w;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) p[i] = __floats2bfloat162_rn(a[2 * i], a[2 * i + 1]);
  return w;
}

template <int GM>
__device__ __forceinline__ void add_lut(float* sc, const __half* e) {
  // GM halves (GM multiple of 4) -> sc[0..GM)
  if constexpr (GM % 8 == 0) {
#pragma unroll
    for (int q = 0; q < GM / 8; ++q) {
      uint4 w = *reinterpret_cast<const uint4*>(e + 8 * q);
      const __half2* h = reinterpret_cast<const __half2*>(&w);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(h[i]);
        sc[8 * q + 2 * i] += f.x;
        sc[8 * q + 2 * i + 1] += f.y;
      }
    }
  } else {  // rows of GM halves are only 8-byte aligned: GM / 4 eight-byte loads
#pragma unroll
    for (int q = 0; q < GM / 4; ++q) {
      uint2 w = *reinterpret_cast<const uint2*>(e + 4 * q);
      const __half2* h = reinterpret_cast<const __half2*>(&w);
      float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
      sc[4 * q] += a.x;
      sc[4 * q + 1] += a.y;
      sc[4 * q + 2] += b.x;
      sc[4 * q + 3] += b.y;
    }
  }
}

template <int FMT, int KST, int M, int NW>
__global__ void __launch_bounds__(NW * 32) vq_attn_lut_kernel(
    const __nv_bfloat16* __restrict__ qkv, KVP P, const float* __restrict__ invf, float* __restrict__ OP,
    float* __restrict__ MP, float* __restrict__ LP, int ctx, int chunk, int splits, int Hkv, int BM, long sq_r,
    float qk_scale) {
  using L = Lay<FMT, KST, M, NW>;
  constexpr int GM = L::GM;
  constexpr int NTH = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv;
  const int n = ctx + M, start = sp * chunk, end = min(start + chunk, n);
  float* sQ = reinterpret_cast<float*>(smem + L::warp_off);  // scratch, overlaps the warp region

  // ---- prologue: q rows (RoPE, scaled), per-group V tables (bf16) and K LUTs (fp16)
  using GR = typename L::GR;
  constexpr int NG = L::NG;
  for (int i = tid; i < GM * D; i += NTH) {
    const int r = i / D, c = i % D, m = r / G, g = r % G;
    const __nv_bfloat16* qr = qkv + (long)(b * M + m) * sq_r + (h * G + g) * D;
    float x = __bfloat162float(qr[c]), xp = __bfloat162float(qr[(c + D / 2) % D]);
    float sn, cs;
    sincosf((float)(ctx + m) * invf[c % (D / 2)], &sn, &cs);
    sQ[r * D + c] = __bfloat162float(__float2bfloat16((x * cs + (c < D / 2 ? -1.f : 1.f) * xp * sn) * qk_scale));
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int s0 = GR::start(g), ty = GR::type(g);
    __half* lut = reinterpret_cast<__half*>(smem + L::lut_off(g));
    __nv_bfloat16* vt = reinterpret_cast<__nv_bfloat16*>(smem + L::vcb_off(g));
    for (int c = tid; c < GR::C(g); c += NTH) {
      float cw[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, cv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (ty == G44) {  // index c = ca | cb << 4
        const int ca = c & 15, cb = c >> 4;
        add8(cw, *reinterpret_cast<const uint4*>(P.kcb[s0] + h * P.cbh[s0] + ca * DS));
        add8(cw, *reinterpret_cast<const uint4*>(P.kcb[s0 + 1] + h * P.cbh[s0 + 1] + cb * DS));
        add8(cv, *reinterpret_cast<const uint4*>(P.vcb[s0] + h * P.cbh[s0] + ca * DS));
        add8(cv, *reinterpret_cast<const uint4*>(P.vcb[s0 + 1] + h * P.cbh[s0 + 1] + cb * DS));
      } else {
        add8(cw, *reinterpret_cast<const uint4*>(P.kcb[s0] + h * P.cbh[s0] + c * DS));
        add8(cv, *reinterpret_cast<const uint4*>(P.vcb[s0] + h * P.cbh[s0] + c * DS));
      }
      *reinterpret_cast<uint4*>(vt + c * DS) = pack8(cv);
      for (int j = 0; j < NSV; ++j) {
#pragma unroll
        for (int r0 = 0; r0 < GM; r0 += 4) {
          __half v[4];
#pragma unroll
          for (int rr = 0; rr < 4; ++rr) {
            const float4 a = *reinterpret_cast<const float4*>(sQ + (r0 + rr) * D + j * DS);
            const float4 bq = *reinterpret_cast<const float4*>(sQ + (r0 + rr) * D + j * DS + 4);
            v[rr] = __float2half(a.x * cw[0] + a.y * cw[1] + a.z * cw[2] + a.w * cw[3] + bq.x * cw[4] +
                                 bq.y * cw[5] + bq.z * cw[6] + bq.w * cw[7]);
          }
          *reinterpret_cast<uint2*>(lut + ((long)(j * GR::C(g) + c) * GM + r0)) = *reinterpret_cast<uint2*>(v);
        }
      }
    }
  }
  __syncthreads();  // sQ region becomes the warp-private region below

  constexpr bool TC = L::TC;
  constexpr int BMQ = L::BMQ;
  uint8_t* wb = smem + L::warp_off + warp * L::warp_bytes;
  uint8_t* sVC = wb;                                         // FMA: [NG][32][16]
  float* sPw = reinterpret_cast<float*>(wb + NG * 32 * 16);  // FMA: [32][GM]
  __nv_bfloat16* sV = reinterpret_cast<__nv_bfloat16*>(wb + L::tc_v);  // TC: [32][LDV]
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(wb + L::tc_p);  // TC: [BMQ][LDP]
  float* sA = reinterpret_cast<float*>(wb + L::tc_a);                  // TC: [BMQ]
  constexpr int AR = TC ? 4 : GM;  // TC: unused (dead FMA path)
  float acc[AR][8];
#pragma unroll
  for (int r = 0; r < AR; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
  constexpr int OT = TC ? (BMQ / 16) * (D / 16) : 1;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[OT];
  if constexpr (TC) {
#pragma unroll
    for (int i = 0; i < OT; ++i) wmma::fill_fragment(fo[i], 0.f);
    for (int i = lane; i < BMQ * LDP; i += 32) sP[i] = __float2bfloat16(0.f);
    if (lane < BMQ) sA[lane] = 1.f;
    __syncwarp();
  }
  float mrow[GM], lrow[GM];
#pragma unroll
  for (int r = 0; r < GM; ++r) { mrow[r] = -INFINITY; lrow[r] = 0.f; }
  const int dg = lane & 15, tp = lane >> 4;

  for (int t0 = start + 32 * warp; t0 < end; t0 += 32 * NW) {
    const int t = t0 + lane;
    const bool ok = t < end;
    float sc[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) sc[r] = 0.f;
    uint32_t vcr[NG][4];
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      const int s0 = GR::start(g), ty = GR::type(g);
      const __half* lut = reinterpret_cast<const __half*>(smem + L::lut_off(g));
      const long off = b * P.sb[s0] + h * P.sh[s0];
      uint32_t kc[4] = {0, 0, 0, 0}, vc[4] = {0, 0, 0, 0};  // 16 byte-indices (G8, G44) or 16 nibbles (G4)
      if (ty == G8) {
        if (ok) {
          const uint4 kw = *reinterpret_cast<const uint4*>(P.kc[s0] + off + (long)t * 16);
          const uint4 vw = *reinterpret_cast<const uint4*>(P.vc[s0] + off + (long)t * 16);
          kc[0] = kw.x; kc[1] = kw.y; kc[2] = kw.z; kc[3] = kw.w;
          vc[0] = vw.x; vc[1] = vw.y; vc[2] = vw.z; vc[3] = vw.w;
        }
      } else if (ty == G4) {
        if (ok) {
          const uint2 kw = *reinterpret_cast<const uint2*>(P.kc[s0] + off + (long)t * 8);
          const uint2 vw = *reinterpret_cast<const uint2*>(P.vc[s0] + off + (long)t * 8);
          kc[0] = kw.x; kc[1] = kw.y;
          vc[0] = vw.x; vc[1] = vw.y;
        }
      } else {  // G44: interleave the two nibble planes into byte indices ca | cb << 4
        if (ok) {
          const long off1 = b * P.sb[s0 + 1] + h * P.sh[s0 + 1];
          const uint2 ka = *reinterpret_cast<const uint2*>(P.kc[s0] + off + (long)t * 8);
          const uint2 kb = *reinterpret_cast<const uint2*>(P.kc[s0 + 1] + off1 + (long)t * 8);
          const uint2 va = *reinterpret_cast<const uint2*>(P.vc[s0] + off + (long)t * 8);
          const uint2 vb = *reinterpret_cast<const uint2*>(P.vc[s0 + 1] + off1 + (long)t * 8);
          const uint32_t a2[2] = {ka.x, ka.y}, b2[2] = {kb.x, kb.y}, c2[2] = {va.x, va.y}, d2[2] = {vb.x, vb.y};
#pragma unroll
          for (int q = 0; q < 4; ++q) {  // output word q holds sub-vectors 4q .. 4q+3
            const uint32_t A = a2[q >> 1] >> (16 * (q & 1)), Bw = b2[q >> 1] >> (16 * (q & 1));
            const uint32_t Cw = c2[q >> 1] >> (16 * (q & 1)), Dw = d2[q >> 1] >> (16 * (q & 1));
            uint32_t ko = 0, vo = 0;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              ko |= (((A >> (4 * e)) & 15) | (((Bw >> (4 * e)) & 15) << 4)) << (8 * e);
              vo |= (((Cw >> (4 * e)) & 15) | (((Dw >> (4 * e)) & 15) << 4)) << (8 * e);
            }
            kc[q] = ko;
            vc[q] = vo;
          }
        }
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) vcr[g][q] = vc[q];
      if constexpr (!TC) *reinterpret_cast<uint4*>(sVC + (g * 32 + lane) * 16) = make_uint4(vc[0], vc[1], vc[2], vc[3]);
#pragma unroll
      for (int j = 0; j < NSV; ++j) {
        const int code = ty == G4 ? (kc[j >> 3] >> (4 * (j & 7))) & 0xF : (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
        add_lut<GM>(sc, lut + (long)(j * GR::C(g) + code) * GM);
      }
    }
    // ---- online softmax over this 32-token tile (lane = token)
    float alpha[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) {
      const bool valid = ok && t <= ctx + r / G;
      float v = valid ? sc[r] : -INFINITY;
      float mx = v;
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mn = fmaxf(mrow[r], mx);
      const float ms = mn == -INFINITY ? 0.f : mn;
      float p = valid ? exp2f(v - ms) : 0.f;
      if constexpr (TC) {
        const __nv_bfloat16 pb = __float2bfloat16(p);
        sP[r * LDP + lane] = pb;
        p = __bfloat162float(pb);
      } else {
        sPw[lane * GM + r] = p;
      }
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      alpha[r] = exp2f(mrow[r] - ms);
      lrow[r] = lrow[r] * alpha[r] + sum;
      mrow[r] = mn;
      if constexpr (TC) {
        if (lane == 0) sA[r] = alpha[r];
      }
    }
    if constexpr (TC) {
      // V of this lane's token -> sV row (bf16 group sums), then O = O * alpha + P V on tensor cores
#pragma unroll
      for (int j = 0; j < NSV; ++j) {
        float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int g = 0; g < NG; ++g) {
          const int code = GR::type(g) == G4 ? (vcr[g][j >> 3] >> (4 * (j & 7))) & 0xF
                                             : (vcr[g][j >> 2] >> (8 * (j & 3))) & 0xFF;
          add8(v8, *reinterpret_cast<const uint4*>(smem + L::vcb_off(g) + code * DS * 2));
        }
        *reinterpret_cast<uint4*>(sV + lane * LDV + j * DS) = pack8(v8);
      }
      __syncwarp();
#pragma unroll
      for (int i = 0; i < OT; ++i) {
        const int rb = i / (D / 16), cb = i % (D / 16);
#pragma unroll
        for (int e = 0; e < fo[i].num_elements; ++e) fo[i].x[e] *= sA[rb * 16 + (lane >> 2) + 8 * ((e >> 1) & 1)];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
          wmma::load_matrix_sync(fa, sP + rb * 16 * LDP + kk * 16, LDP);
          wmma::load_matrix_sync(fb, sV + kk * 16 * LDV + cb * 16, LDV);
          wmma::mma_sync(fo[i], fa, fb, fo[i]);
        }
      }
      __syncwarp();
      continue;
    }
    __syncwarp();
    // ---- V: lane = (dg, tp); tokens tp, tp+2, ...
#pragma unroll
    for (int r = 0; r < AR; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] *= alpha[r];
    const int nt = min(32, end - t0);
#pragma unroll 2
    for (int tt = tp; tt < nt; tt += 2) {
      float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        const uint8_t* row = sVC + (g * 32 + tt) * 16;
        const int code = GR::type(g) == G4 ? ((row[dg >> 1] >> (4 * (dg & 1))) & 0xF) : row[dg];
        add8(v8, *reinterpret_cast<const uint4*>(smem + L::vcb_off(g) + code * DS * 2));
      }
#pragma unroll
      for (int r0 = 0; r0 < AR; r0 += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(sPw + tt * GM + r0);
        const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r0 + rr][i] += pp[rr] * v8[i];
      }
    }
    __syncwarp();
  }
  const long o = (long)(bh * splits * NW + sp * NW + warp) * BM;
  if constexpr (TC) {  // O fragments -> staging (sV region, [16][68] fp32 halves) -> partial rows < GM
    float* stg = reinterpret_cast<float*>(sV);
#pragma unroll
    for (int rb = 0; rb < BMQ / 16; ++rb) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __syncwarp();
#pragma unroll
        for (int cb = 0; cb < 4; ++cb)
          wmma::store_matrix_sync(stg + cb * 16, fo[rb * (D / 16) + half * 4 + cb], 68, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * 64; i += 32) {
          const int r = rb * 16 + i / 64, c = i % 64;
          if (r < GM) OP[(o + r) * D + half * 64 + c] = stg[(i / 64) * 68 + c];
        }
      }
    }
  }
  // ---- FMA path: merge token parities, write this warp's partial
#pragma unroll
  for (int r = 0; r < AR; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] += __shfl_xor_sync(0xffffffff, acc[r][i], 16);
  if (!TC && tp == 0) {
#pragma unroll
    for (int r = 0; r < AR; ++r) {
      float4* dst = reinterpret_cast<float4*>(OP + (o + r) * D + dg * 8);
      dst[0] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
      dst[1] = make_float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < GM; ++r) { MP[o + r] = mrow[r]; LP[o + r] = lrow[r]; }
  }
}

template <int M>
constexpr int nw_for() { return M == 1 ? 16 : 4; }

template <int FMT, int KST, int M>
void launch(torch::Tensor qkv, const KVP& P, torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP,
            int ctx, int chunk, int splits, int B, int Hkv, int BM, float qk_scale) {
  constexpr int NW = nw_for<M>();
  using L = Lay<FMT, KST, M, NW>;
  if constexpr (L::total > 101376) {
    TORCH_CHECK(false, "vq_attn_lut: shared memory ", L::total, " B too large for this (fmt, kst, M)");
  } else {
    auto k = vq_attn_lut_kernel<FMT, KST, M, NW>;
    static bool set = false;
    if (!set) {
      TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess,
                  "smem attr");
      set = true;
    }
    dim3 grid(B * Hkv, splits);
    k<<<grid, NW * 32, L::total, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), P, invf.data_ptr<float>(), OP.data_ptr<float>(),
        MP.data_ptr<float>(), LP.data_ptr<float>(), ctx, chunk, splits, Hkv, BM, qkv.stride(0), qk_scale);
  }
}

template <int FMT, int KST>
void by_m(int64_t M, torch::Tensor qkv, const KVP& P, torch::Tensor invf, torch::Tensor OP, torch::Tensor MP,
          torch::Tensor LP, int ctx, int chunk, int splits, int B, int Hkv, int BM, float qk_scale) {
  switch (M) {
    case 1: launch<FMT, KST, 1>(qkv, P, invf, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, qk_scale); break;
    case 2: launch<FMT, KST, 2>(qkv, P, invf, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, qk_scale); break;
    case 3: launch<FMT, KST, 3>(qkv, P, invf, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, qk_scale); break;
    case 5: launch<FMT, KST, 5>(qkv, P, invf, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, qk_scale); break;
    default: TORCH_CHECK(false, "M must be 1/2/3/5");
  }
}

template <int FMT, int KST, int M>
int64_t smem_of() { return Lay<FMT, KST, M, nw_for<M>()>::total; }

template <int FMT, int KST>
int64_t smem_m(int64_t M) {
  return M == 1 ? smem_of<FMT, KST, 1>() : M == 2 ? smem_of<FMT, KST, 2>() : M == 3 ? smem_of<FMT, KST, 3>()
                                                                                    : smem_of<FMT, KST, 5>();
}
}  // namespace

// (shared memory bytes, warps per block) of the LUT kernel for (fmt, kst, M); smem > 101376 = not launchable
std::vector<int64_t> vq_attn_lut_plan(int64_t fmt, int64_t kst, int64_t M) {
  int64_t sm = 0;
  if (fmt == 0) sm = kst == 1 ? smem_m<0, 1>(M) : smem_m<0, 2>(M);
  else if (fmt == 1) sm = kst == 1 ? smem_m<1, 1>(M) : kst == 2 ? smem_m<1, 2>(M) : kst == 3 ? smem_m<1, 3>(M)
                                                                                               : smem_m<1, 4>(M);
  else sm = kst == 1 ? smem_m<2, 1>(M) : kst == 2 ? smem_m<2, 2>(M) : smem_m<2, 3>(M);
  return {sm, M == 1 ? 16 : 4};
}

void vq_attn_lut(torch::Tensor qkv, std::vector<torch::Tensor> kpl, std::vector<torch::Tensor> vpl,
                 std::vector<torch::Tensor> kcb, std::vector<torch::Tensor> vcb, torch::Tensor invf, torch::Tensor OP,
                 torch::Tensor MP, torch::Tensor LP, int64_t ctx, int64_t chunk, int64_t splits, int64_t B,
                 int64_t Hkv, int64_t BM, int64_t M, int64_t fmt, int64_t kst, double qk_scale) {
  KVP P{};
  TORCH_CHECK((int64_t)kpl.size() >= kst && (int64_t)vpl.size() >= kst, "planes");
  for (int s = 0; s < kst; ++s) {
    P.kc[s] = kpl[s].data_ptr<uint8_t>();
    P.vc[s] = vpl[s].data_ptr<uint8_t>();
    TORCH_CHECK(kpl[s].stride(0) == vpl[s].stride(0) && kpl[s].stride(1) == vpl[s].stride(1), "K/V plane strides");
    TORCH_CHECK(kpl[s].stride(3) == 1 && kpl[s].stride(2) == kpl[s].size(3), "plane token-contiguous");
    P.sb[s] = kpl[s].stride(0);
    P.sh[s] = kpl[s].stride(1);
    P.kcb[s] = reinterpret_cast<const __nv_bfloat16*>(kcb[s].data_ptr());
    P.vcb[s] = reinterpret_cast<const __nv_bfloat16*>(vcb[s].data_ptr());
    TORCH_CHECK(kcb[s].stride(0) == vcb[s].stride(0) && kcb[s].stride(2) == 1 && kcb[s].stride(1) == DS, "cb");
    P.cbh[s] = kcb[s].stride(0);
  }
#define A qkv, P, invf, OP, MP, LP, (int)ctx, (int)chunk, (int)splits, (int)B, (int)Hkv, (int)BM, (float)qk_scale
  if (fmt == 0) {
    if (kst == 1) by_m<0, 1>(M, A); else by_m<0, 2>(M, A);
  } else if (fmt == 1) {
    if (kst == 1) by_m<1, 1>(M, A); else if (kst == 2) by_m<1, 2>(M, A); else if (kst == 3) by_m<1, 3>(M, A);
    else by_m<1, 4>(M, A);
  } else {
    if (kst == 1) by_m<2, 1>(M, A); else if (kst == 2) by_m<2, 2>(M, A); else by_m<2, 3>(M, A);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_attn_lut", &vq_attn_lut);
  m.def("vq_attn_lut_plan", &vq_attn_lut_plan);
}
