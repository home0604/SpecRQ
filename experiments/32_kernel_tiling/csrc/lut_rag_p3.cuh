// E10 phase 3: A100-oriented variants of the LUT VQ-KV draft-attention partial kernel.
// Base = phase-1 best structure (lut_rag_p1.cuh w8_b1_r1_p1_l0_s2): in-block partial reduction (RED 1), register
// prefetch of the next tile's codes (PF 1), fp16 LUT scores, lazy online max (SB 2). Same inputs and the same partial
// contract as the phase-1 RED kernel (GM rows per block and split, SPT = splits), so the Triton combine is unchanged.
// The original (experiments/26_magicvq/csrc/vqattn_lut_rag.cu) and the p1 header are NOT modified.
//
// Template knobs:
//   NW    warps per block
//   VM    value path (P . V):
//         0  fp32 FFMA (= phase 1): lane (dg, tp) owns 8 dims of one token half; bf16 codewords summed in fp32
//         1  tensor core mma.sync m16n8k16 (f16 x f16 -> f32, sm_80+). Per 16-token k-tile: O^T[128 x 8] += V^T[128 x 16]
//            P^T[16 x 8] (N = 4 q-heads padded to 8). V^T rows are permuted so that thread (gid, tig) holds whole
//            sub-vectors 2gid, 2gid+1 (row gid of m-tile m = dim 16 gid + m, row gid + 8 = dim 16 gid + 8 + m): it gathers
//            the fp16 codewords of its 4 tokens (2 sub-vectors each), sums stages with HADD2, and builds the A fragments
//            with PRMT. P in fp16 (p <= 2^8 by the lazy max), fp32 accumulate.
//         2  packed HFMA2: VM 0 layout, fp16 codewords summed with HADD2, p * 2^-4 in fp16, per-tile half2 accumulators
//            (<= 16 tokens per lane) flushed into fp32 after every 32-token tile.
//   VREP  (VM 1/2) fp16 V codebook replicas per stage, interleaved over the 8 16-byte bank groups so that lanes of one
//         128-bit load phase (8 lanes) use different replicas: replica r owns bank groups [r*8/VREP, (r+1)*8/VREP); lane
//         uses r = lane % VREP. VREP 8 = conflict-free V gathers (smem 4 KB * VREP per stage).
//   HS    1: LUT scores accumulated in half2 over groups of 4 sub-vectors, then added in fp32 (fewer FP32 ops; extra
//         fp16 roundings of partial sums). 0: fp32 accumulate (original).
//   AB    timing-only ablations (NOT correct outputs): bit 0 = V accumulation removed (codes still loaded and staged in
//         smem), bit 1 = LUT score gather replaced by a cheap function of the K codes (loads kept live).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace e10p3 {
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

template <int KST, int NW, int VM, int VREP>
struct Lay {
  static constexpr int vstage = VM == 0 ? C * DS * 2 : C * DS * 2 * VREP;
  static constexpr int vcb_bytes = KST * vstage;
  __host__ __device__ static constexpr int vcb_off(int g) { return g * vstage; }
  static constexpr int lut_stage = NSV * C * GM * 2;
  __host__ __device__ static constexpr int lut_off(int g) { return al16(vcb_bytes) + g * lut_stage; }
  static constexpr int pbytes = VM == 1 ? GM * 32 * 2 : (VM == 2 ? 32 * GM * 2 : 32 * GM * 4);
  static constexpr int warp_bytes = KST * 32 * 16 + pbytes;
  static constexpr int warp_off = al16(lut_off(KST));
  static constexpr int qbytes = GM * D * 4;
  static constexpr int tail = cmax(NW * warp_bytes, qbytes);
  static constexpr int qmu_off = warp_off + tail;  // 16 floats
  static constexpr int base = qmu_off + 64;
  static constexpr int red_bytes = NW * GM * (D + 2) * 4;  // overlays offset 0 after the token loop
  static constexpr int total = cmax(base, red_bytes);
  // fp16 V codeword c of replica r (byte offset inside the stage)
  static __device__ __forceinline__ int vaddr(int c, int r) {
    constexpr int SPR = 8 / VREP;  // 16-byte slots per replica per 128-byte row
    return (c / SPR) * 128 + (r * SPR + c % SPR) * 16;
  }
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

__device__ __forceinline__ uint4 pack8h(const float* a) {
  uint4 w;
  __half2* p = reinterpret_cast<__half2*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) p[i] = __floats2half2_rn(a[2 * i], a[2 * i + 1]);
  return w;
}

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b) {
  __half2 r = __hadd2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ uint4 hadd8(uint4 a, uint4 b) {
  return make_uint4(hadd2u(a.x, b.x), hadd2u(a.y, b.y), hadd2u(a.z, b.z), hadd2u(a.w, b.w));
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

__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  __trap();
#endif
}

template <int KST, int NW, int VM, int VREP, int HS, int AB>
__global__ void __launch_bounds__(NW * 32, 1) lut_p3_kernel(
    const __nv_bfloat16* __restrict__ qkv, KVP P, const float* __restrict__ invf, float* __restrict__ OP,
    float* __restrict__ MP, float* __restrict__ LP, const int* __restrict__ LO, const int* __restrict__ HI,
    const int* __restrict__ QPOS, const float* __restrict__ SDK, const float* __restrict__ MUK,
    const float* __restrict__ SDV, const float* __restrict__ MUV, int chunk, int Hkv, int BM, int SPT, int OFF,
    long sq_r, float qk_scale) {
  using L = Lay<KST, NW, VM, VREP>;
  constexpr int NTH = NW * 32;
  static_assert((GM * D) % NTH == 0 && GM * D / 32 == 16, "prologue: whole warps per iteration, 16 qmu slots");
  static_assert(VM != 0 || VREP == 1, "VREP only for the fp16 value paths");
  static_assert(VREP == 1 || VREP == 2 || VREP == 4 || VREP == 8, "VREP");
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv;
  const int start = LO[bh] + sp * chunk, end = min(start + chunk, HI[bh]);
  const int qpos = QPOS[b];
  float* sQ = reinterpret_cast<float*>(smem + L::warp_off);
  float* sQmu = reinterpret_cast<float*>(smem + L::qmu_off);
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
    if (lane == 0) sQmu[idx >> 5] = pm;
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < KST; ++g) {
    __half* lut = reinterpret_cast<__half*>(smem + L::lut_off(g));
    for (int c = tid; c < C; c += NTH) {
      float cw[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, cv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      add8(cw, *reinterpret_cast<const uint4*>(P.kcb[g] + h * P.cbh + c * DS));
      add8(cv, *reinterpret_cast<const uint4*>(P.vcb[g] + h * P.cbh + c * DS));
      if constexpr (VM == 0) {
        *reinterpret_cast<uint4*>(smem + L::vcb_off(g) + c * DS * 2) = pack8(cv);
      } else {
        const uint4 w = pack8h(cv);  // bf16 -> fp16 (exact in the normal fp16 range)
#pragma unroll
        for (int r = 0; r < VREP; ++r) *reinterpret_cast<uint4*>(smem + L::vcb_off(g) + L::vaddr(c, r)) = w;
      }
      for (int j = 0; j < NSV; ++j) {
        __half v[4];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) v[rr] = __float2half(lut_dot(sQ, rr, j, cw));
        *reinterpret_cast<uint2*>(lut + ((long)(j * C + c) * GM)) = *reinterpret_cast<uint2*>(v);
      }
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
  uint8_t* sVC = wb;                    // [KST][32][16]
  uint8_t* sPb = wb + KST * 32 * 16;    // VM 0: float [32][GM]; VM 1: half [GM][32]; VM 2: half [32][GM]
  constexpr int NACC = VM == 1 ? 8 : GM;  // VM 1: 8 m-tiles x 4; else GM rows x 8 dims
  constexpr int NAI = VM == 1 ? 4 : 8;
  float acc[NACC][NAI];
#pragma unroll
  for (int r = 0; r < NACC; ++r)
#pragma unroll
    for (int i = 0; i < NAI; ++i) acc[r][i] = 0.f;
  float mrow[GM], lrow[GM];
#pragma unroll
  for (int r = 0; r < GM; ++r) { mrow[r] = -INFINITY; lrow[r] = 0.f; }
  const int dg = lane & 15, tp = lane >> 4;  // VM 0 / 2
  const int gid = lane >> 2, tig = lane & 3;  // VM 1
  const int rep = lane % VREP;
  const long off = b * P.sb + h * P.sh;
  uint32_t dummy = 0;  // AB: keeps ablated loads live

  uint4 kn[KST], vn[KST];
  {
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
#pragma unroll
    for (int g = 0; g < KST; ++g) { kcur[g] = kn[g]; vcur[g] = vn[g]; }
    {
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
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      const uint32_t kc[4] = {kcur[g].x, kcur[g].y, kcur[g].z, kcur[g].w};
      *reinterpret_cast<uint4*>(sVC + (g * 32 + lane) * 16) = vcur[g];
      if constexpr (AB & 2) {
        // ablation: no LUT gather; cheap score from the codes (keeps the K loads live)
#pragma unroll
        for (int r = 0; r < GM; ++r) sc[r] += (float)((kc[r] ^ kc[(r + 1) & 3]) & 0xFF) * (1.f / 64.f);
      } else {
        const __half* lut = reinterpret_cast<const __half*>(smem + L::lut_off(g));
        if constexpr (HS) {
#pragma unroll
          for (int jg = 0; jg < NSV / 4; ++jg) {
            __half2 h01 = __float2half2_rn(0.f), h23 = __float2half2_rn(0.f);
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
              const int j = jg * 4 + jj;
              const int code = (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
              const uint2 w = *reinterpret_cast<const uint2*>(lut + (long)(j * C + code) * GM);
              h01 = __hadd2(h01, *reinterpret_cast<const __half2*>(&w.x));
              h23 = __hadd2(h23, *reinterpret_cast<const __half2*>(&w.y));
            }
            const float2 a = __half22float2(h01), bb = __half22float2(h23);
            sc[0] += a.x; sc[1] += a.y; sc[2] += bb.x; sc[3] += bb.y;
          }
        } else {
#pragma unroll
          for (int j = 0; j < NSV; ++j) {
            const int code = (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
            add_lut4(sc, lut + (long)(j * C + code) * GM);
          }
        }
      }
    }
    // lazy online max: rescale only if some score exceeds the running reference by > 8 (log2 units); the first tile
    // always triggers (mrow = -inf). p <= 2^8.
    {
      bool need = false;
#pragma unroll
      for (int r = 0; r < GM; ++r) need |= ok && sc[r] > mrow[r] + 8.f;
      if (__any_sync(0xffffffff, need)) {
        float al[GM];
#pragma unroll
        for (int r = 0; r < GM; ++r) {
          float mx = ok ? sc[r] : -INFINITY;
#pragma unroll
          for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
          const float mn = fmaxf(mrow[r], mx);
          al[r] = exp2f(mrow[r] - mn);
          lrow[r] *= al[r];
          mrow[r] = mn;
        }
        if constexpr (VM == 1) {
          const float a0 = al[(2 * tig) & 3], a1 = al[(2 * tig + 1) & 3];
#pragma unroll
          for (int m = 0; m < 8; ++m) { acc[m][0] *= a0; acc[m][1] *= a1; acc[m][2] *= a0; acc[m][3] *= a1; }
        } else {
#pragma unroll
          for (int r = 0; r < GM; ++r)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[r][i] *= al[r];
        }
      }
      float p[GM];
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        p[r] = ok ? exp2f(sc[r] - mrow[r]) : 0.f;
        lrow[r] += p[r];
      }
      if constexpr (VM == 0) {
        *reinterpret_cast<float4*>(sPb + lane * 16) = make_float4(p[0], p[1], p[2], p[3]);
      } else if constexpr (VM == 1) {
        __half* sp_ = reinterpret_cast<__half*>(sPb);
#pragma unroll
        for (int r = 0; r < GM; ++r) sp_[r * 32 + lane] = __float2half_rn(p[r]);
      } else {
        const __half2 h0 = __floats2half2_rn(p[0] * 0.0625f, p[1] * 0.0625f);
        const __half2 h1 = __floats2half2_rn(p[2] * 0.0625f, p[3] * 0.0625f);
        uint2 w;
        w.x = *reinterpret_cast<const uint32_t*>(&h0);
        w.y = *reinterpret_cast<const uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(sPb + lane * 8) = w;
      }
    }
    __syncwarp();
    if constexpr (AB & 1) {
      dummy ^= *reinterpret_cast<const uint32_t*>(sVC + ((lane + 1) & 31) * 16) ^
               *reinterpret_cast<const uint32_t*>(sPb + ((lane + 3) & 31) * 4);
    } else if constexpr (VM == 0) {
      const int nt = min(32, end - t0);
#pragma unroll 2
      for (int tt = tp; tt < nt; tt += 2) {
        float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int g = 0; g < KST; ++g) {
          const uint8_t* row = sVC + (g * 32 + tt) * 16;
          add8(v8, *reinterpret_cast<const uint4*>(smem + L::vcb_off(g) + row[dg] * DS * 2));
        }
        const float4 p4 = *reinterpret_cast<const float4*>(sPb + tt * 16);
        const float pp[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[rr][i] += pp[rr] * v8[i];
      }
    } else if constexpr (VM == 2) {
      const int nt = min(32, end - t0);
      __half2 ha[GM][4];
#pragma unroll
      for (int r = 0; r < GM; ++r)
#pragma unroll
        for (int w = 0; w < 4; ++w) ha[r][w] = __float2half2_rn(0.f);
#pragma unroll 2
      for (int tt = tp; tt < nt; tt += 2) {
        uint4 x;
#pragma unroll
        for (int g = 0; g < KST; ++g) {
          const int code = sVC[(g * 32 + tt) * 16 + dg];
          const uint4 y = *reinterpret_cast<const uint4*>(smem + L::vcb_off(g) + L::vaddr(code, rep));
          x = g == 0 ? y : hadd8(x, y);
        }
        const uint2 pw = *reinterpret_cast<const uint2*>(sPb + tt * 8);
        const __half2 p01 = *reinterpret_cast<const __half2*>(&pw.x), p23 = *reinterpret_cast<const __half2*>(&pw.y);
        const __half2 pb[4] = {__low2half2(p01), __high2half2(p01), __low2half2(p23), __high2half2(p23)};
        const __half2* xv = reinterpret_cast<const __half2*>(&x);
#pragma unroll
        for (int r = 0; r < GM; ++r)
#pragma unroll
          for (int w = 0; w < 4; ++w) ha[r][w] = __hfma2(xv[w], pb[r], ha[r][w]);
      }
#pragma unroll
      for (int r = 0; r < GM; ++r)
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float2 f = __half22float2(ha[r][w]);
          acc[r][2 * w] += 16.f * f.x;
          acc[r][2 * w + 1] += 16.f * f.y;
        }
    } else {  // VM == 1: tensor-core P . V
      const __half* sp_ = reinterpret_cast<const __half*>(sPb);
      const int ntile = min(32, end - t0) > 16 ? 2 : 1;
#pragma unroll
      for (int kt = 0; kt < 2; ++kt) {
        if (kt < ntile) {
          const int tb = 16 * kt + 2 * tig;
          uint32_t b0 = 0, b1 = 0;
          if (gid < GM) {
            b0 = *reinterpret_cast<const uint32_t*>(sp_ + gid * 32 + tb);
            b1 = *reinterpret_cast<const uint32_t*>(sp_ + gid * 32 + tb + 8);
          }
          uint4 X[2][4];  // [sub-vector 2gid + s][token tb + {0, 1, 8, 9}]
#pragma unroll
          for (int ti = 0; ti < 4; ++ti) {
            const int tok = tb + (ti & 1) + 8 * (ti >> 1);
#pragma unroll
            for (int g = 0; g < KST; ++g) {
              const uint32_t c2 = *reinterpret_cast<const uint16_t*>(sVC + (g * 32 + tok) * 16 + 2 * gid);
              const uint8_t* vb = smem + L::vcb_off(g);
              const uint4 y0 = *reinterpret_cast<const uint4*>(vb + L::vaddr(c2 & 0xFF, rep));
              const uint4 y1 = *reinterpret_cast<const uint4*>(vb + L::vaddr(c2 >> 8, rep));
              X[0][ti] = g == 0 ? y0 : hadd8(X[0][ti], y0);
              X[1][ti] = g == 0 ? y1 : hadd8(X[1][ti], y1);
            }
          }
#pragma unroll
          for (int m = 0; m < 8; ++m) {
            const unsigned sel = (m & 1) ? 0x7632u : 0x5410u;
            const int w = m >> 1;
            auto W = [&](const uint4& u) { return w == 0 ? u.x : (w == 1 ? u.y : (w == 2 ? u.z : u.w)); };
            const uint32_t a0 = __byte_perm(W(X[0][0]), W(X[0][1]), sel);
            const uint32_t a1 = __byte_perm(W(X[1][0]), W(X[1][1]), sel);
            const uint32_t a2 = __byte_perm(W(X[0][2]), W(X[0][3]), sel);
            const uint32_t a3 = __byte_perm(W(X[1][2]), W(X[1][3]), sel);
            mma16816(acc[m], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
    __syncwarp();
  }
  if constexpr (VM != 1) {
#pragma unroll
    for (int r = 0; r < GM; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] += __shfl_xor_sync(0xffffffff, acc[r][i], 16);
  }
#pragma unroll
  for (int r = 0; r < GM; ++r) {
#pragma unroll
    for (int o = 16; o; o >>= 1) lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], o);
    if (lrow[r] == 0.f) mrow[r] = -INFINITY;  // no token in this warp's range
  }
  // in-block combine of the NW warp partials (fixed warp order) -> GM rows per block
  __syncthreads();
  float* rA = reinterpret_cast<float*>(smem);  // [NW][GM][D]
  float* rM = rA + NW * GM * D;                // [NW][GM]
  float* rL = rM + NW * GM;                    // [NW][GM]
  if constexpr (VM == 1) {
    if (tig < 2) {
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const int d0 = 16 * gid + m, d1 = 16 * gid + 8 + m;
        rA[(warp * GM + 2 * tig) * D + d0] = acc[m][0];
        rA[(warp * GM + 2 * tig + 1) * D + d0] = acc[m][1];
        rA[(warp * GM + 2 * tig) * D + d1] = acc[m][2];
        rA[(warp * GM + 2 * tig + 1) * D + d1] = acc[m][3];
      }
    }
  } else if (tp == 0) {
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
  if constexpr (AB & 1) {
    if (dummy == 0x9E3779B9u) OP[o * D] = 1.f;  // data-dependent: keeps the ablated loads
  }
}

template <int KST, int NW, int VM, int VREP, int HS, int AB>
cudaError_t run_t(const RunArgs& a) {
  using L = Lay<KST, NW, VM, VREP>;
  auto k = lut_p3_kernel<KST, NW, VM, VREP, HS, AB>;
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
template <int KST, int NW, int VM, int VREP, int HS, int AB>
std::vector<int64_t> occ_t() {
  using L = Lay<KST, NW, VM, VREP>;
  auto k = lut_p3_kernel<KST, NW, VM, VREP, HS, AB>;
  int nb = 0;
  cudaFuncAttributes fa{};
  cudaFuncGetAttributes(&fa, k);
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, NW * 32, L::total);
  cudaGetLastError();
  return {nb, L::total, fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes, fa.maxThreadsPerBlock,
          NW * 32, L::vcb_bytes, L::lut_off(KST) - L::lut_off(0), L::warp_off, L::tail, L::warp_bytes, 1};
}
}  // namespace e10p3
