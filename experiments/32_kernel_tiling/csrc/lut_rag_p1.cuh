// E10 phase 1: parameterized copy of the E7 LUT VQ-KV draft-attention partial kernel
// (experiments/26_magicvq/csrc/vqattn_lut_rag.cu; the original is NOT modified). Same inputs / partial math.
// Structural template knobs:
//   NW   warps per block (original 16); prologue / LUT build loop over NTH = NW * 32 threads
//   MINB __launch_bounds__ minBlocksPerSM (register target; smem may still limit blocks/SM)
//   RED  1: combine the NW warp partials inside the block (smem, fixed order) -> GM rows per block, SPT = splits
//        0: one partial per warp (original), SPT = NW * splits
//   PF   1: register prefetch of the next token tile's K/V codes (all stages) before processing the current tile
//   LUTM 0: fp16 LUT (original, 8 B per (j, code): 4 q-heads)
//        1: int8 LUT, symmetric per-(stage, subvector j, q-head) scale, float FMA accumulate (4 B per (j, code))
//        2: int8 LUT, offset-binary uint8, one scale per q-head (over all stages / subvectors), packed integer
//           accumulate (2 x 16-bit lanes per u32), scale applied once per token
//   SB   1: static score upper bound per (block, q-head) instead of the online max: bound = sum over stages and
//           subvectors of max_c LUT[g][j][c][r] (computed from the stored, possibly quantized, LUT entries after the
//           build; q . muk offset added at the store as for m). p = exp2(s - bound) <= 1: no running-max update, no
//           accumulator rescale, l reduced across lanes once per split. Underflow hazard if bound - s > ~126 (log2
//           units) for every token of a warp/split -> measured separately (sb_gap.py): real Llama data DOES exceed
//           it (gap > 126 for ~1-3% of (query, head)), so SB 1 is unsafe and kept only as a speed reference.
//        2: lazy online max (safe): reference max set by the first tile, rescale (warp max + acc/l rescale) only when
//           a warp vote finds a score > reference + 8; otherwise p = exp2(s - ref), no shuffles; l reduced at the end.
// LUTM 1/2 are NOT bit-identical to the fp16 LUT (scores quantized to 8 bit).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace e10p1 {
constexpr int D = 128, DS = 8, NSV = 16, G = 4, C = 256, GM = G;  // M = 1

struct KVP {
  const uint8_t* kc[2];
  const uint8_t* vc[2];
  long sb, sh;  // plane strides (batch, head); token stride 16 bytes
  const __nv_bfloat16* kcb[2];
  const __nv_bfloat16* vcb[2];
  long cbh;  // codebook head stride
};

struct RunArgs {
  const __nv_bfloat16* qkv;
  KVP P;
  const float* invf;
  float *OP, *MP, *LP;
  const int *LO, *HI, *QPOS;
  const float *SDK, *MUK, *SDV, *MUV;
  int chunk, splits, B, Hkv, BM, SPT, OFF;
  long sq_r;
  float qk_scale;
  cudaStream_t stream;
};

__host__ __device__ constexpr int al16(int x) { return (x + 15) / 16 * 16; }
__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

template <int KST, int NW, int RED, int LUTM, int SB = 0>
struct Lay {
  static constexpr int ESZ = LUTM == 0 ? 2 : 1;
  static constexpr int vcb_bytes = KST * C * DS * 2;
  static constexpr int lut_stage = NSV * C * GM * ESZ;
  static constexpr int vcb_off(int g) { return g * C * DS * 2; }
  static constexpr int lut_off(int g) { return al16(vcb_bytes) + g * lut_stage; }
  static constexpr int warp_bytes = KST * 32 * 16 + GM * 32 * 4;
  static constexpr int warp_off = al16(lut_off(KST));
  static constexpr int qbytes = GM * D * 4;
  static constexpr int tail = cmax(NW * warp_bytes, qbytes);
  static constexpr int qmu_off = warp_off + tail;  // 16 floats
  static constexpr int nscale = LUTM == 1 ? KST * NSV * GM : (LUTM == 2 ? GM : 0);
  static constexpr int sc_off = qmu_off + 64;
  static constexpr int bnd_off = sc_off + al16(nscale * 4);
  static constexpr int base = bnd_off + (SB == 1 ? KST * NSV * GM * 4 : 0);
  static constexpr int red_bytes = RED ? NW * GM * (D + 2) * 4 : 0;  // overlays offset 0 after the token loop
  static constexpr int total = cmax(base, red_bytes);
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

__device__ __forceinline__ void add_lut4(float* sc, const __half* e) {
  uint2 w = *reinterpret_cast<const uint2*>(e);
  const __half2* h = reinterpret_cast<const __half2*>(&w);
  float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
  sc[0] += a.x; sc[1] += a.y; sc[2] += b.x; sc[3] += b.y;
}

__device__ __forceinline__ float lut_dot(const float* sQ, int rr, int j, const float* cw) {
  const float4 a = *reinterpret_cast<const float4*>(sQ + rr * D + j * DS);
  const float4 bq = *reinterpret_cast<const float4*>(sQ + rr * D + j * DS + 4);
  return a.x * cw[0] + a.y * cw[1] + a.z * cw[2] + a.w * cw[3] + bq.x * cw[4] + bq.y * cw[5] + bq.z * cw[6] +
         bq.w * cw[7];
}

template <int KST, int NW, int MINB, int RED, int PF, int LUTM, int SB>
__global__ void __launch_bounds__(NW * 32, MINB) lut_p1_kernel(
    const __nv_bfloat16* __restrict__ qkv, KVP P, const float* __restrict__ invf, float* __restrict__ OP,
    float* __restrict__ MP, float* __restrict__ LP, const int* __restrict__ LO, const int* __restrict__ HI,
    const int* __restrict__ QPOS, const float* __restrict__ SDK, const float* __restrict__ MUK,
    const float* __restrict__ SDV, const float* __restrict__ MUV, int chunk, int Hkv, int BM, int SPT, int OFF,
    long sq_r, float qk_scale) {
  using L = Lay<KST, NW, RED, LUTM, SB>;
  constexpr int NTH = NW * 32;
  constexpr int CPT = (C + NTH - 1) / NTH;
  static_assert((GM * D) % NTH == 0 && GM * D / 32 == 16, "prologue: whole warps per iteration, 16 qmu slots");
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv;
  const int start = LO[bh] + sp * chunk, end = min(start + chunk, HI[bh]);
  const int qpos = QPOS[b];
  float* sQ = reinterpret_cast<float*>(smem + L::warp_off);
  float* sQmu = reinterpret_cast<float*>(smem + L::qmu_off);
  float* sScl = reinterpret_cast<float*>(smem + L::sc_off);
  if constexpr (LUTM != 0) {
    for (int i = tid; i < L::nscale; i += NTH) sScl[i] = 0.f;
  }
  // q RoPE + per-channel scale; q . muk per q-head via fixed-order reduction (16 warp-slot partials, as original)
  for (int idx = tid; idx < GM * D; idx += NTH) {
    const int r = idx / D, c = idx % D;
    const __nv_bfloat16* qr = qkv + (long)b * sq_r + (h * G + r) * D;
    float x = __bfloat162float(qr[c]), xp = __bfloat162float(qr[(c + D / 2) % D]);
    float sn, cs;
    sincosf((float)qpos * invf[c % (D / 2)], &sn, &cs);
    const float rot = __bfloat162float(__float2bfloat16(x * cs + (c < D / 2 ? -1.f : 1.f) * xp * sn));
    sQ[r * D + c] = rot * SDK[h * D + c] * qk_scale;
    float pm = rot * MUK[h * D + c] * qk_scale;
#pragma unroll
    for (int o = 16; o; o >>= 1) pm += __shfl_xor_sync(0xffffffff, pm, o);
    if (lane == 0) sQmu[idx >> 5] = pm;  // slot w holds row w / (D / 32)
  }
  __syncthreads();
  if constexpr (LUTM == 0) {
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      __half* lut = reinterpret_cast<__half*>(smem + L::lut_off(g));
      __nv_bfloat16* vt = reinterpret_cast<__nv_bfloat16*>(smem + L::vcb_off(g));
      for (int c = tid; c < C; c += NTH) {
        float cw[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, cv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        add8(cw, *reinterpret_cast<const uint4*>(P.kcb[g] + h * P.cbh + c * DS));
        add8(cv, *reinterpret_cast<const uint4*>(P.vcb[g] + h * P.cbh + c * DS));
        *reinterpret_cast<uint4*>(vt + c * DS) = pack8(cv);
        for (int j = 0; j < NSV; ++j) {
          __half v[4];
#pragma unroll
          for (int rr = 0; rr < 4; ++rr) v[rr] = __float2half(lut_dot(sQ, rr, j, cw));
          *reinterpret_cast<uint2*>(lut + ((long)(j * C + c) * GM)) = *reinterpret_cast<uint2*>(v);
        }
      }
    }
  } else {
    float cw[KST][CPT][8];
#pragma unroll
    for (int g = 0; g < KST; ++g)
#pragma unroll
      for (int k = 0; k < CPT; ++k) {
        const int c = tid + k * NTH;
#pragma unroll
        for (int i = 0; i < 8; ++i) cw[g][k][i] = 0.f;
        if (c < C) {
          float cv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
          add8(cw[g][k], *reinterpret_cast<const uint4*>(P.kcb[g] + h * P.cbh + c * DS));
          add8(cv, *reinterpret_cast<const uint4*>(P.vcb[g] + h * P.cbh + c * DS));
          *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(smem + L::vcb_off(g)) + c * DS) = pack8(cv);
        }
      }
    // pass 1: max |entry| per scale group (atomicMax on the bits of non-negative floats: order independent)
#pragma unroll
    for (int g = 0; g < KST; ++g)
      for (int j = 0; j < NSV; ++j) {
        float mx[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < CPT; ++k)
          if (tid + k * NTH < C)
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) mx[rr] = fmaxf(mx[rr], fabsf(lut_dot(sQ, rr, j, cw[g][k])));
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
          for (int o = 16; o; o >>= 1) mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffff, mx[rr], o));
          if (lane == 0)
            atomicMax(reinterpret_cast<unsigned*>(sScl + (LUTM == 1 ? (g * NSV + j) * GM + rr : rr)),
                      __float_as_uint(mx[rr]));
        }
      }
    __syncthreads();
    // pass 2: quantize
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      uint32_t* lut = reinterpret_cast<uint32_t*>(smem + L::lut_off(g));
      for (int j = 0; j < NSV; ++j) {
        float inv[4];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
          const float m = sScl[LUTM == 1 ? (g * NSV + j) * GM + rr : rr];
          inv[rr] = m > 0.f ? 127.f / m : 0.f;
        }
#pragma unroll
        for (int k = 0; k < CPT; ++k) {
          const int c = tid + k * NTH;
          if (c < C) {
            uint32_t w = 0;
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
              int q = __float2int_rn(lut_dot(sQ, rr, j, cw[g][k]) * inv[rr]);
              q = max(-127, min(127, q));
              const uint32_t byte = LUTM == 1 ? (uint32_t)(uint8_t)(int8_t)q : (uint32_t)(q + 128);
              w |= byte << (8 * rr);
            }
            lut[j * C + c] = w;
          }
        }
      }
    }
    __syncthreads();
    for (int i = tid; i < L::nscale; i += NTH) sScl[i] = sScl[i] * (1.f / 127.f);
  }
  float* sBnd = reinterpret_cast<float*>(smem + L::bnd_off);
  if constexpr (SB == 1) {
    __syncthreads();  // LUT (and int8 scales) complete
    for (int it = warp; it < KST * NSV; it += NW) {
      const int g = it / NSV, j = it % NSV;
      float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      for (int c = lane; c < C; c += 32) {
        float e[4];
        if constexpr (LUTM == 0) {
          const uint2 w = *reinterpret_cast<const uint2*>(smem + L::lut_off(g) + (j * C + c) * GM * 2);
          const __half2* hh = reinterpret_cast<const __half2*>(&w);
          const float2 x0 = __half22float2(hh[0]), x1 = __half22float2(hh[1]);
          e[0] = x0.x; e[1] = x0.y; e[2] = x1.x; e[3] = x1.y;
        } else {
          const uint32_t w = reinterpret_cast<const uint32_t*>(smem + L::lut_off(g))[j * C + c];
#pragma unroll
          for (int rr = 0; rr < 4; ++rr) {
            const int byte = (w >> (8 * rr)) & 0xFF;
            e[rr] = LUTM == 1 ? sScl[(g * NSV + j) * GM + rr] * (float)(int8_t)byte : (float)(byte - 128);
          }
        }
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) mx[rr] = fmaxf(mx[rr], e[rr]);
      }
#pragma unroll
      for (int rr = 0; rr < 4; ++rr) {
#pragma unroll
        for (int o = 16; o; o >>= 1) mx[rr] = fmaxf(mx[rr], __shfl_xor_sync(0xffffffff, mx[rr], o));
      }
      if (lane == 0)
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) sBnd[it * GM + rr] = mx[rr];
    }
  }
  float qmu[GM];
#pragma unroll
  for (int r = 0; r < GM; ++r) {
    float a = 0.f;
#pragma unroll
    for (int w = 0; w < D / 32; ++w) a += sQmu[r * (D / 32) + w];
    qmu[r] = a;
  }
  __syncthreads();  // sQ region becomes the warp-private region

  uint8_t* wb = smem + L::warp_off + warp * L::warp_bytes;
  uint8_t* sVC = wb;                                          // [KST][32][16]
  float* sPw = reinterpret_cast<float*>(wb + KST * 32 * 16);  // [32][GM]
  float acc[GM][8];
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
  float mrow[GM], lrow[GM];
#pragma unroll
  for (int r = 0; r < GM; ++r) { mrow[r] = -INFINITY; lrow[r] = 0.f; }
  float srow[GM];
  if constexpr (LUTM == 2) {
#pragma unroll
    for (int r = 0; r < GM; ++r) srow[r] = sScl[r];
  }
  float bnd[GM];
  if constexpr (SB == 1) {
#pragma unroll
    for (int r = 0; r < GM; ++r) {
      float a = 0.f;
      for (int it = 0; it < KST * NSV; ++it) a += sBnd[it * GM + r];
      if constexpr (LUTM == 2) a *= sScl[r];
      bnd[r] = a;
      mrow[r] = a;
    }
  }
  const int dg = lane & 15, tp = lane >> 4;
  const long off = b * P.sb + h * P.sh;

  uint4 kn[KST], vn[KST];  // PF: next tile's codes
  if constexpr (PF) {
    const int t = start + 32 * warp + lane;
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      kn[g] = make_uint4(0, 0, 0, 0);
      vn[g] = make_uint4(0, 0, 0, 0);
      if (t < end) {
        kn[g] = *reinterpret_cast<const uint4*>(P.kc[g] + off + (long)t * 16);
        vn[g] = *reinterpret_cast<const uint4*>(P.vc[g] + off + (long)t * 16);
      }
    }
  }

  for (int t0 = start + 32 * warp; t0 < end; t0 += 32 * NW) {
    const int t = t0 + lane;
    const bool ok = t < end;
    uint4 kcur[KST], vcur[KST];
    if constexpr (PF) {
#pragma unroll
      for (int g = 0; g < KST; ++g) { kcur[g] = kn[g]; vcur[g] = vn[g]; }
      const int tn = t + 32 * NW;
#pragma unroll
      for (int g = 0; g < KST; ++g) {
        kn[g] = make_uint4(0, 0, 0, 0);
        vn[g] = make_uint4(0, 0, 0, 0);
        if (tn < end) {
          kn[g] = *reinterpret_cast<const uint4*>(P.kc[g] + off + (long)tn * 16);
          vn[g] = *reinterpret_cast<const uint4*>(P.vc[g] + off + (long)tn * 16);
        }
      }
    }
    float sc[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) sc[r] = 0.f;
    uint32_t a02 = 0, a13 = 0;  // LUTM 2 packed accumulators
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      uint32_t kc[4] = {0, 0, 0, 0}, vc[4] = {0, 0, 0, 0};
      if constexpr (PF) {
        kc[0] = kcur[g].x; kc[1] = kcur[g].y; kc[2] = kcur[g].z; kc[3] = kcur[g].w;
        vc[0] = vcur[g].x; vc[1] = vcur[g].y; vc[2] = vcur[g].z; vc[3] = vcur[g].w;
      } else if (ok) {
        const uint4 kw = *reinterpret_cast<const uint4*>(P.kc[g] + off + (long)t * 16);
        const uint4 vw = *reinterpret_cast<const uint4*>(P.vc[g] + off + (long)t * 16);
        kc[0] = kw.x; kc[1] = kw.y; kc[2] = kw.z; kc[3] = kw.w;
        vc[0] = vw.x; vc[1] = vw.y; vc[2] = vw.z; vc[3] = vw.w;
      }
      *reinterpret_cast<uint4*>(sVC + (g * 32 + lane) * 16) = make_uint4(vc[0], vc[1], vc[2], vc[3]);
      if constexpr (LUTM == 0) {
        const __half* lut = reinterpret_cast<const __half*>(smem + L::lut_off(g));
#pragma unroll
        for (int j = 0; j < NSV; ++j) {
          const int code = (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
          add_lut4(sc, lut + (long)(j * C + code) * GM);
        }
      } else {
        const uint32_t* lut = reinterpret_cast<const uint32_t*>(smem + L::lut_off(g));
#pragma unroll
        for (int j = 0; j < NSV; ++j) {
          const int code = (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
          const uint32_t w = lut[j * C + code];
          if constexpr (LUTM == 1) {
            const float4 s4 = *reinterpret_cast<const float4*>(sScl + (g * NSV + j) * GM);
            sc[0] += s4.x * (float)(int8_t)(w & 0xFF);
            sc[1] += s4.y * (float)(int8_t)((w >> 8) & 0xFF);
            sc[2] += s4.z * (float)(int8_t)((w >> 16) & 0xFF);
            sc[3] += s4.w * (float)(int8_t)(w >> 24);
          } else {
            a02 += w & 0x00FF00FFu;
            a13 += (w >> 8) & 0x00FF00FFu;
          }
        }
      }
    }
    if constexpr (LUTM == 2) {
      constexpr int Z = 128 * NSV * KST;
      sc[0] = srow[0] * (float)((int)(a02 & 0xFFFFu) - Z);
      sc[1] = srow[1] * (float)((int)(a13 & 0xFFFFu) - Z);
      sc[2] = srow[2] * (float)((int)(a02 >> 16) - Z);
      sc[3] = srow[3] * (float)((int)(a13 >> 16) - Z);
    }
    float alpha[GM];
    if constexpr (SB == 2) {
      // lazy max: rescale only if some score exceeds the running reference by > 8 (log2 units); the first tile
      // always triggers (mrow = -inf). Exact up to fp32 (p <= 2^8, dropped terms < 2^-118 of the max term).
      bool need = false;
#pragma unroll
      for (int r = 0; r < GM; ++r) need |= ok && sc[r] > mrow[r] + 8.f;
      if (__any_sync(0xffffffff, need)) {
#pragma unroll
        for (int r = 0; r < GM; ++r) {
          float mx = ok ? sc[r] : -INFINITY;
#pragma unroll
          for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
          const float mn = fmaxf(mrow[r], mx);
          const float al = exp2f(mrow[r] - mn);
          lrow[r] *= al;
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r][i] *= al;
          mrow[r] = mn;
        }
      }
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        const float p = ok ? exp2f(sc[r] - mrow[r]) : 0.f;
        sPw[lane * GM + r] = p;
        lrow[r] += p;
      }
    } else if constexpr (SB == 1) {
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        const float p = ok ? exp2f(sc[r] - bnd[r]) : 0.f;
        sPw[lane * GM + r] = p;
        lrow[r] += p;  // per-lane partial; reduced across the warp after the loop
      }
    } else
#pragma unroll
    for (int r = 0; r < GM; ++r) {
      float v = ok ? sc[r] : -INFINITY;
      float mx = v;
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mn = fmaxf(mrow[r], mx);
      const float ms = mn == -INFINITY ? 0.f : mn;
      float p = ok ? exp2f(v - ms) : 0.f;
      sPw[lane * GM + r] = p;
      float sum = p;
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      alpha[r] = exp2f(mrow[r] - ms);
      lrow[r] = lrow[r] * alpha[r] + sum;
      mrow[r] = mn;
    }
    __syncwarp();
    if constexpr (!SB)
#pragma unroll
    for (int r = 0; r < GM; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] *= alpha[r];
    const int nt = min(32, end - t0);
#pragma unroll 2
    for (int tt = tp; tt < nt; tt += 2) {
      float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < KST; ++g) {
        const uint8_t* row = sVC + (g * 32 + tt) * 16;
        add8(v8, *reinterpret_cast<const uint4*>(smem + L::vcb_off(g) + row[dg] * DS * 2));
      }
      const float4 p4 = *reinterpret_cast<const float4*>(sPw + tt * GM);
      const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
      for (int rr = 0; rr < 4; ++rr)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[rr][i] += pp[rr] * v8[i];
    }
    __syncwarp();
  }
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] += __shfl_xor_sync(0xffffffff, acc[r][i], 16);
  if constexpr (SB) {
#pragma unroll
    for (int r = 0; r < GM; ++r) {
#pragma unroll
      for (int o = 16; o; o >>= 1) lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], o);
      if (lrow[r] == 0.f) mrow[r] = -INFINITY;  // no token in this warp's range (as the online-max path)
    }
  }
  if constexpr (!RED) {
    const long o = ((long)bh * SPT + OFF + sp * NW + warp) * BM;
    if (tp == 0) {
      const float* sv = SDV + h * D + dg * 8;
      const float* mv = MUV + h * D + dg * 8;
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        float y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) y[i] = acc[r][i] * sv[i] + mv[i] * lrow[r];
        float4* dst = reinterpret_cast<float4*>(OP + (o + r) * D + dg * 8);
        dst[0] = make_float4(y[0], y[1], y[2], y[3]);
        dst[1] = make_float4(y[4], y[5], y[6], y[7]);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        MP[o + r] = mrow[r] == -INFINITY ? -INFINITY : mrow[r] + qmu[r];
        LP[o + r] = lrow[r];
      }
    }
  } else {
    // in-block combine of the NW warp partials (fixed warp order) -> GM rows per block
    __syncthreads();  // all warps are done with the LUT / warp regions
    float* rA = reinterpret_cast<float*>(smem);   // [NW][GM][D]
    float* rM = rA + NW * GM * D;                 // [NW][GM]
    float* rL = rM + NW * GM;                     // [NW][GM]
    if (tp == 0) {
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        float4* dst = reinterpret_cast<float4*>(rA + (warp * GM + r) * D + dg * 8);
        dst[0] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
        dst[1] = make_float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < GM; ++r) { rM[warp * GM + r] = mrow[r]; rL[warp * GM + r] = lrow[r]; }
    }
    __syncthreads();
    const long o = ((long)bh * SPT + OFF + sp) * BM;
    for (int idx = tid; idx < GM * D; idx += NTH) {
      const int r = idx / D, c = idx % D;
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < NW; ++w) M = fmaxf(M, rM[w * GM + r]);
      const float ms = M == -INFINITY ? 0.f : M;
      float Oa = 0.f, La = 0.f;
#pragma unroll
      for (int w = 0; w < NW; ++w) {
        const float e = exp2f(rM[w * GM + r] - ms);
        Oa += e * rA[(w * GM + r) * D + c];
        La += e * rL[w * GM + r];
      }
      OP[(o + r) * D + c] = Oa * SDV[h * D + c] + MUV[h * D + c] * La;
      if (c == 0) {
        MP[o + r] = M == -INFINITY ? -INFINITY : M + qmu[r];
        LP[o + r] = La;
      }
    }
  }
}

template <int KST, int NW, int MINB, int RED, int PF, int LUTM, int SB>
cudaError_t run_t(const RunArgs& a) {
  using L = Lay<KST, NW, RED, LUTM, SB>;
  auto k = lut_p1_kernel<KST, NW, MINB, RED, PF, LUTM, SB>;
  static bool set = false;
  if (!set) {
    cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total);
    if (e != cudaSuccess) return e;
    set = true;
  }
  dim3 grid(a.B * a.Hkv, a.splits);
  k<<<grid, NW * 32, L::total, a.stream>>>(a.qkv, a.P, a.invf, a.OP, a.MP, a.LP, a.LO, a.HI, a.QPOS, a.SDK, a.MUK,
                                          a.SDV, a.MUV, a.chunk, a.Hkv, a.BM, a.SPT, a.OFF, a.sq_r, a.qk_scale);
  return cudaGetLastError();
}

// [blocks/SM (0 if not launchable), dyn smem, regs, local bytes, static smem, max threads/block, threads, vcb bytes,
//  lut bytes, warp_off, tail bytes, warp bytes, partial slots per split]
template <int KST, int NW, int MINB, int RED, int PF, int LUTM, int SB>
std::vector<int64_t> occ_t() {
  using L = Lay<KST, NW, RED, LUTM, SB>;
  auto k = lut_p1_kernel<KST, NW, MINB, RED, PF, LUTM, SB>;
  int nb = 0;
  cudaFuncAttributes fa{};
  cudaFuncGetAttributes(&fa, k);
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, NW * 32, L::total);
  cudaGetLastError();
  return {nb, L::total, fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes, fa.maxThreadsPerBlock,
          NW * 32, L::vcb_bytes, L::lut_off(KST) - L::lut_off(0), L::warp_off, L::tail, L::warp_bytes,
          RED ? 1 : NW};
}
}  // namespace e10p1
