// E10 phase 2: new structural axes for the LUT VQ-KV draft-attention partial kernel, starting from the phase-1
// global struct w8_b1_r1_p1_l0_s2 (csrc/lut_rag_p1.cuh: fp16 LUT, in-block partial reduction, lazy online max,
// MINB 1). Same inputs / math as experiments/26_magicvq/csrc/vqattn_lut_rag.cu (NOT modified).
// Template knobs (variant name k{KST}_w{NW}_s{LS}_d{DEC}_p{PIPE}_f{FUSE}):
//   NW    warps per block (4 / 8 / 16)
//   LS    LUT source. 0: every block builds the LUT of its (b, kv-head) in smem (phase 1).
//                     1: a separate build kernel (grid B*Hkv x KST x 4 j-groups) writes the fp16 LUT and q.muk to
//                        global (L2-resident workspace); the attention blocks copy it into smem (no rebuild per split).
//   DEC   work decomposition. 0: grid (B*Hkv, splits), one (bh, split) per block (phase 1).
//                     1: stream-K style: grid of P blocks over the flattened (bh, token) space (prefix offsets CU),
//                        each block takes `chunk` consecutive tokens and loops over the bh segments it covers
//                        (one partial slot per segment: slot = block - first block of the bh).
//   PIPE  code-tile pipeline. 0: register prefetch of the next tile (phase 1 PF 1).
//                     S in {2, 3}: per-warp S-stage cp.async ring of K+V code tiles in smem; the first S-1 tiles are
//                        issued before the LUT prologue (overlaps the code fetch with the LUT build / load).
//   FUSE  0: partials -> separate (Triton) log-sum-exp combine kernel.
//         1: combine fused into the partial kernel: a bh covered by one segment writes the bf16 output directly;
//            otherwise the last block of a bh (atomic counter, threadfence) merges the slots and resets the counter.
// The partial math per segment is the phase-1 struct (bit-identical LUT entries for LS 0/1).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace e10p2 {
constexpr int D = 128, DS = 8, NSV = 16, G = 4, C = 256, GM = G;  // M = 1
constexpr int LUT_STAGE_H = NSV * C * GM;                           // halves per stage (32 KB)

struct KVP {
  const uint8_t* kc[2];
  const uint8_t* vc[2];
  long sb, sh;  // plane strides (batch, head); token stride 16 bytes
  const __nv_bfloat16* kcb[2];
  const __nv_bfloat16* vcb[2];
  long cbh;  // codebook head stride
};

struct KArgs {
  const __nv_bfloat16* qkv;
  KVP P;
  const float* invf;
  float *OP, *MP, *LP;
  const int *LO, *HI, *QPOS;
  const float *SDK, *MUK, *SDV, *MUV;
  __half* LUTG;  // LS 1: [BH][KST][NSV][C][GM]
  float* QMUG;   // LS 1: [BH][GM]
  const int* CU;  // DEC 1: [BH + 1] prefix offsets of (HI - LO)
  __nv_bfloat16* OUT;  // FUSE 1: [B][Hkv * G * D]
  int* CNT;            // FUSE 1: [BH] zero-initialized, self-resetting
  int chunk, splits, B, Hkv, BM, SPT, OFF, nblk;
  long sq_r;
  float qk_scale;
};

struct RunArgs {
  KArgs k;
  cudaStream_t stream;
};

__host__ __device__ constexpr int al16(int x) { return (x + 15) / 16 * 16; }
__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

template <int KST, int NW, int PIPE>
struct Lay {
  static constexpr int vcb_bytes = KST * C * DS * 2;
  static constexpr int lut_stage = NSV * C * GM * 2;
  static constexpr int vcb_off(int g) { return g * C * DS * 2; }
  static constexpr int lut_off(int g) { return al16(vcb_bytes) + g * lut_stage; }
  static constexpr int ring = PIPE ? PIPE : 1;
  static constexpr int tile_bytes = PIPE ? KST * 2 * 32 * 16 : KST * 32 * 16;  // PIPE: K+V, else V only
  static constexpr int warp_bytes = ring * tile_bytes + GM * 32 * 4;
  static constexpr int warp_off = al16(lut_off(KST));
  static constexpr int qbytes = GM * D * 4;
  // sQ overlays the warp region (PIPE 0, as phase 1) or sits after it (PIPE > 0: ring is filled during the prologue)
  static constexpr int q_off = PIPE ? warp_off + NW * warp_bytes : warp_off;
  static constexpr int tail = PIPE ? NW * warp_bytes + qbytes : cmax(NW * warp_bytes, qbytes);
  static constexpr int qmu_off = warp_off + tail;  // 16 floats
  static constexpr int flag_off = qmu_off + 64;
  static constexpr int base = flag_off + 16;
  static constexpr int red_bytes = NW * GM * (D + 2) * 4;  // overlays offset 0 after the token loop
  static_assert(red_bytes <= warp_off, "reduction overlay must not touch the warp region");
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

__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// q RoPE + per-channel scale into sQ [GM][D]; q . muk partial sums into sQmu[16] (fixed order, as the original)
template <int NTH>
__device__ __forceinline__ void q_prologue(const KArgs& a, int b, int h, float* sQ, float* sQmu) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int qpos = a.QPOS[b];
  for (int idx = tid; idx < GM * D; idx += NTH) {
    const int r = idx / D, c = idx % D;
    const __nv_bfloat16* qr = a.qkv + (long)b * a.sq_r + (h * G + r) * D;
    float x = __bfloat162float(qr[c]), xp = __bfloat162float(qr[(c + D / 2) % D]);
    float sn, cs;
    sincosf((float)qpos * a.invf[c % (D / 2)], &sn, &cs);
    const float rot = __bfloat162float(__float2bfloat16(x * cs + (c < D / 2 ? -1.f : 1.f) * xp * sn));
    sQ[r * D + c] = rot * a.SDK[h * D + c] * a.qk_scale;
    float pm = rot * a.MUK[h * D + c] * a.qk_scale;
#pragma unroll
    for (int o = 16; o; o >>= 1) pm += __shfl_xor_sync(0xffffffff, pm, o);
    if (lane == 0) sQmu[idx >> 5] = pm;  // slot w holds row w / (D / 32)
  }
}

// LS 1: separate LUT build kernel. grid (BH, KST, NSV / 4), 256 threads (one code per thread, 4 subvectors).
template <int KST>
__global__ void __launch_bounds__(256) lut_build_kernel(const __grid_constant__ KArgs a) {
  __shared__ __align__(16) float sQ[GM * D];
  __shared__ float sQmu[16];
  const int bh = blockIdx.x, g = blockIdx.y, jg = blockIdx.z, b = bh / a.Hkv, h = bh % a.Hkv;
  const int tid = threadIdx.x;
  q_prologue<256>(a, b, h, sQ, sQmu);
  __syncthreads();
  if (g == 0 && jg == 0 && tid < GM) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < D / 32; ++w) s += sQmu[tid * (D / 32) + w];
    a.QMUG[bh * GM + tid] = s;
  }
  const int c = tid;
  float cw[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  add8(cw, *reinterpret_cast<const uint4*>(a.P.kcb[g] + h * a.P.cbh + c * DS));
  __half* lut = a.LUTG + ((long)bh * KST + g) * LUT_STAGE_H;
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    const int j = jg * 4 + jj;
    __half v[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) v[rr] = __float2half(lut_dot(sQ, rr, j, cw));
    *reinterpret_cast<uint2*>(lut + (long)(j * C + c) * GM) = *reinterpret_cast<uint2*>(v);
  }
}

// One (bh, token range) segment: LUT (build or load), token loop, in-block reduction, partial store or fused combine.
template <int KST, int NW, int LS, int PIPE, int FUSE>
__device__ __forceinline__ void segment(const KArgs& a, uint8_t* smem, int bh, int start, int end, int slot,
                                        int nseg) {
  using L = Lay<KST, NW, PIPE>;
  constexpr int NTH = NW * 32;
  constexpr int S = PIPE ? PIPE : 1;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = bh / a.Hkv, h = bh % a.Hkv;
  const KVP& P = a.P;
  const long off = b * P.sb + h * P.sh;
  uint8_t* wb = smem + L::warp_off + warp * L::warp_bytes;

  // PIPE: issue the first S-1 code tiles before the prologue
  auto issue = [&](int i) {
    if constexpr (PIPE) {
      const int t0 = start + 32 * warp + i * 32 * NW;
      if (t0 < end) {
        const int t = t0 + lane;
        const bool v = t < end;
        const long ts = (long)(v ? t : start) * 16;
        uint8_t* dst = wb + (i % S) * L::tile_bytes;
#pragma unroll
        for (int g = 0; g < KST; ++g) {
          cp_async16(dst + (g * 2 + 0) * 512 + lane * 16, P.kc[g] + off + ts, v ? 16 : 0);
          cp_async16(dst + (g * 2 + 1) * 512 + lane * 16, P.vc[g] + off + ts, v ? 16 : 0);
        }
      }
      cp_async_commit();
    }
  };
  if constexpr (PIPE) {
#pragma unroll
    for (int i = 0; i < S - 1; ++i) issue(i);
  }

  float qmu[GM];
  if constexpr (LS == 0) {
    float* sQ = reinterpret_cast<float*>(smem + L::q_off);
    float* sQmu = reinterpret_cast<float*>(smem + L::qmu_off);
    q_prologue<NTH>(a, b, h, sQ, sQmu);
    __syncthreads();
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
#pragma unroll
    for (int r = 0; r < GM; ++r) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < D / 32; ++w) s += sQmu[r * (D / 32) + w];
      qmu[r] = s;
    }
  } else {
    // copy the prebuilt LUT (L2) and the V codebooks into smem
    const uint4* src = reinterpret_cast<const uint4*>(a.LUTG + (long)bh * KST * LUT_STAGE_H);
    uint4* dst = reinterpret_cast<uint4*>(smem + L::lut_off(0));
    constexpr int NL = KST * LUT_STAGE_H * 2 / 16;
#pragma unroll 4
    for (int i = tid; i < NL; i += NTH) dst[i] = __ldcg(src + i);
#pragma unroll
    for (int g = 0; g < KST; ++g)
      for (int c = tid; c < C; c += NTH) {
        float cv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        add8(cv, *reinterpret_cast<const uint4*>(P.vcb[g] + h * P.cbh + c * DS));
        *reinterpret_cast<uint4*>(smem + L::vcb_off(g) + c * DS * 2) = pack8(cv);
      }
#pragma unroll
    for (int r = 0; r < GM; ++r) qmu[r] = __ldcg(a.QMUG + bh * GM + r);
  }
  __syncthreads();  // LUT complete; (PIPE 0) sQ region becomes the warp-private region

  float* sPw = reinterpret_cast<float*>(wb + S * L::tile_bytes);  // [32][GM]
  float acc[GM][8];
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
  float mrow[GM], lrow[GM];
#pragma unroll
  for (int r = 0; r < GM; ++r) { mrow[r] = -INFINITY; lrow[r] = 0.f; }
  const int dg = lane & 15, tp = lane >> 4;

  uint4 kn[KST], vn[KST];  // PIPE 0: next tile's codes in registers
  if constexpr (!PIPE) {
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

  int it = 0;
  for (int t0 = start + 32 * warp; t0 < end; t0 += 32 * NW, ++it) {
    const int t = t0 + lane;
    const bool ok = t < end;
    const uint8_t* vrow;  // V code rows of this tile: [KST][32][16] with stage stride vstr
    int vstr;
    uint32_t kc[KST][4];
    if constexpr (PIPE) {
      issue(it + S - 1);
      cp_async_wait<S - 1>();
      __syncwarp();
      const uint8_t* tb = wb + (it % S) * L::tile_bytes;
#pragma unroll
      for (int g = 0; g < KST; ++g) {
        const uint4 w = *reinterpret_cast<const uint4*>(tb + (g * 2 + 0) * 512 + lane * 16);
        kc[g][0] = w.x; kc[g][1] = w.y; kc[g][2] = w.z; kc[g][3] = w.w;
      }
      vrow = tb + 512;
      vstr = 1024;
    } else {
      uint4 kcur[KST], vcur[KST];
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
#pragma unroll
      for (int g = 0; g < KST; ++g) {
        kc[g][0] = kcur[g].x; kc[g][1] = kcur[g].y; kc[g][2] = kcur[g].z; kc[g][3] = kcur[g].w;
        *reinterpret_cast<uint4*>(wb + (g * 32 + lane) * 16) = vcur[g];
      }
      vrow = wb;
      vstr = 512;
    }
    float sc[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) sc[r] = 0.f;
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      const __half* lut = reinterpret_cast<const __half*>(smem + L::lut_off(g));
#pragma unroll
      for (int j = 0; j < NSV; ++j) {
        const int code = (kc[g][j >> 2] >> (8 * (j & 3))) & 0xFF;
        add_lut4(sc, lut + (long)(j * C + code) * GM);
      }
    }
    // lazy online max (phase-1 SB 2)
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
    __syncwarp();
    const int nt = min(32, end - t0);
#pragma unroll 2
    for (int tt = tp; tt < nt; tt += 2) {
      float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < KST; ++g) {
        const uint8_t* row = vrow + g * vstr + tt * 16;
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
  if constexpr (PIPE) cp_async_wait<0>();
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] += __shfl_xor_sync(0xffffffff, acc[r][i], 16);
#pragma unroll
  for (int r = 0; r < GM; ++r) {
#pragma unroll
    for (int o = 16; o; o >>= 1) lrow[r] += __shfl_xor_sync(0xffffffff, lrow[r], o);
    if (lrow[r] == 0.f) mrow[r] = -INFINITY;
  }
  // in-block combine of the NW warp partials (fixed warp order)
  __syncthreads();
  float* rA = reinterpret_cast<float*>(smem);  // [NW][GM][D]
  float* rM = rA + NW * GM * D;                // [NW][GM]
  float* rL = rM + NW * GM;                    // [NW][GM]
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
  const long obase = (long)bh * a.SPT + a.OFF;
  const long o = (obase + slot) * a.BM;
  __nv_bfloat16* orow = a.OUT + (long)b * (a.Hkv * G * D) + h * G * D;
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
    const float y = Oa * a.SDV[h * D + c] + a.MUV[h * D + c] * La;
    if (FUSE && nseg == 1) {
      orow[r * D + c] = __float2bfloat16(y / La);
    } else {
      a.OP[(o + r) * D + c] = y;
      if (c == 0) {
        a.MP[o + r] = M == -INFINITY ? -INFINITY : M + qmu[r];
        a.LP[o + r] = La;
      }
    }
  }
  if constexpr (FUSE) {
    if (nseg > 1) {
      int* flag = reinterpret_cast<int*>(smem + L::flag_off);
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const int prev = atomicAdd(a.CNT + bh, 1);
        const int last = prev == nseg - 1;
        if (last) a.CNT[bh] = 0;  // self-reset for the next call (all nseg blocks have arrived)
        *flag = last;
      }
      __syncthreads();
      if (*flag) {
        __threadfence();
        for (int idx = tid; idx < GM * D; idx += NTH) {
          const int r = idx / D, c = idx % D;
          float M = -INFINITY;
          for (int s = 0; s < nseg; ++s) M = fmaxf(M, __ldcg(a.MP + (obase + s) * a.BM + r));
          float Oa = 0.f, La = 0.f;
          for (int s = 0; s < nseg; ++s) {
            const long os = (obase + s) * a.BM + r;
            const float m = __ldcg(a.MP + os);
            const float e = m == -INFINITY ? 0.f : exp2f(m - M);
            Oa += e * __ldcg(a.OP + os * D + c);
            La += e * __ldcg(a.LP + os);
          }
          orow[r * D + c] = __float2bfloat16(Oa / La);
        }
      }
    }
  }
}

template <int KST, int NW, int LS, int DEC, int PIPE, int FUSE>
__global__ void __launch_bounds__(NW * 32, 1) lut_p2_kernel(const __grid_constant__ KArgs a) {
  extern __shared__ __align__(16) uint8_t smem[];
  if constexpr (DEC == 0) {
    const int bh = blockIdx.x, sp = blockIdx.y;
    const int start = a.LO[bh] + sp * a.chunk, end = min(start + a.chunk, a.HI[bh]);
    segment<KST, NW, LS, PIPE, FUSE>(a, smem, bh, start, end, sp, a.splits);
  } else {
    const int BH = a.B * a.Hkv;
    const int total = a.CU[BH];
    const int g0 = blockIdx.x * a.chunk, g1 = min(g0 + a.chunk, total);
    if (g0 >= g1) return;
    int lo = 0, hi = BH;  // largest bh with CU[bh] <= g0
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (a.CU[mid] <= g0) lo = mid; else hi = mid;
    }
    for (int bh = lo; bh < BH; ++bh) {
      const int c0 = a.CU[bh], c1 = a.CU[bh + 1];
      if (c0 >= g1) break;
      const int s0 = max(g0, c0), s1 = min(g1, c1);
      if (s0 >= s1) continue;
      const int first = c0 / a.chunk, last = (c1 - 1) / a.chunk;
      segment<KST, NW, LS, PIPE, FUSE>(a, smem, bh, a.LO[bh] + s0 - c0, a.LO[bh] + s1 - c0, blockIdx.x - first,
                                       last - first + 1);
      __syncthreads();
    }
  }
}

template <int KST, int NW, int LS, int DEC, int PIPE, int FUSE>
cudaError_t run_t(const RunArgs& r) {
  using L = Lay<KST, NW, PIPE>;
  auto k = lut_p2_kernel<KST, NW, LS, DEC, PIPE, FUSE>;
  static bool set = false;
  if (!set) {
    cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total);
    if (e != cudaSuccess) return e;
    set = true;
  }
  const KArgs& a = r.k;
  const int BH = a.B * a.Hkv;
  if constexpr (LS) lut_build_kernel<KST><<<dim3(BH, KST, NSV / 4), 256, 0, r.stream>>>(a);
  const dim3 grid = DEC ? dim3(a.nblk) : dim3(BH, a.splits);
  k<<<grid, NW * 32, L::total, r.stream>>>(a);
  return cudaGetLastError();
}

// [blocks/SM (0 if not launchable), dyn smem, regs, local bytes, static smem, max threads/block, threads,
//  warp bytes, build-kernel regs]
template <int KST, int NW, int LS, int DEC, int PIPE, int FUSE>
std::vector<int64_t> occ_t() {
  using L = Lay<KST, NW, PIPE>;
  auto k = lut_p2_kernel<KST, NW, LS, DEC, PIPE, FUSE>;
  int nb = 0;
  cudaFuncAttributes fa{}, fb{};
  cudaFuncGetAttributes(&fa, k);
  if (LS) cudaFuncGetAttributes(&fb, lut_build_kernel<KST>);
  if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, NW * 32, L::total);
  cudaGetLastError();
  return {nb, L::total, fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes, fa.maxThreadsPerBlock,
          NW * 32, L::warp_bytes, LS ? fb.numRegs : 0};
}
}  // namespace e10p2
