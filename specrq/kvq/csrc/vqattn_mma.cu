// E6 kvregime: VQ-KV decode attention, tensor-core variant (WMMA bf16, fp32 accumulate).
// One block (4 warps) per (b, kv-head h, split). Per 32-token tile:
//   codes for K and V (first KST stages) are prefetched into registers one tile ahead (vector loads),
//   dequantized with 16-byte codeword gathers from shared-memory codebooks into bf16 sK / sV tiles,
//   S = Q K^T and O += P V on tensor cores (Q = G*M stacked rows padded to BMQ = 16 or 32), online softmax (exp2).
// O accumulators stay in WMMA fragments; the per-row rescale uses the sm_80+ m16n16k16 fp32 accumulator layout
// (element i of lane l: row (l>>2) + 8*((i>>1)&1), col 2*(l&3) + (i&1) + 8*(i>>2)); verified by check_kernels.py.
// Partials (O, m, l) per split -> Triton _attn_combine (same layout as the other paths).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

using namespace nvcuda;

namespace {
constexpr int D = 128, NT = 128, C = 256, G = 4, TILE = 32;
constexpr int LDQ = D + 8, LDK = D + 8, LDSS = TILE + 4, LDP = TILE + 8, LDO = D + 4;

__host__ __device__ constexpr int al128(int x) { return (x + 127) / 128 * 128; }

template <int DSUB, int KST, int BMQ>
struct Smem {
  static constexpr int cb = 2 * KST * C * DSUB * 2;
  static constexpr int q = BMQ * LDQ * 2;
  static constexpr int kv = 2 * TILE * LDK * 2;  // sK + sV (reused as O staging [BMQ][LDO] fp32 at the end)
  static constexpr int s = BMQ * LDSS * 4;
  static constexpr int p = BMQ * LDP * 2;
  static constexpr int st = 3 * BMQ * 4;
  static constexpr int o_cb = 0, o_q = al128(o_cb + cb), o_kv = al128(o_q + q), o_s = al128(o_kv + kv),
                       o_p = al128(o_s + s), o_st = al128(o_p + p), total = al128(o_st + st);
  static_assert(BMQ * LDO * 4 <= kv, "O staging must fit in sK+sV");
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

// codes of one tile: TILE*NSV bytes per stage = 512 (d8) / 256 (d16) -> 4 / 2 bytes per thread
template <int NSV>
struct CodeT;
template <>
struct CodeT<16> { using T = uint32_t; static constexpr int BPT = 4; };
template <>
struct CodeT<8> { using T = uint16_t; static constexpr int BPT = 2; };

template <int DSUB, int KST, int BMQ>
__global__ void __launch_bounds__(NT) vq_attn_mma_kernel(
    const __nv_bfloat16* __restrict__ qkv, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const __nv_bfloat16* __restrict__ cbk, const __nv_bfloat16* __restrict__ cbv, const float* __restrict__ invf,
    float* __restrict__ OP, float* __restrict__ MP, float* __restrict__ LP, int ctx, int chunk, int splits, int Hkv,
    int BM, int M, long sq_r, long s_s, long s_b, long s_h, long s_t, long scb_h, long scb_s, float qk_scale) {
  constexpr int NSV = D / DSUB;
  using SM = Smem<DSUB, KST, BMQ>;
  using CT = typename CodeT<NSV>::T;
  constexpr int BPT = CodeT<NSV>::BPT;
  extern __shared__ __align__(128) uint8_t smem[];
  __nv_bfloat16* sCbK = reinterpret_cast<__nv_bfloat16*>(smem + SM::o_cb);
  __nv_bfloat16* sCbV = sCbK + KST * C * DSUB;
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SM::o_q);
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem + SM::o_kv);
  __nv_bfloat16* sV = sK + TILE * LDK;
  float* sS = reinterpret_cast<float*>(smem + SM::o_s);
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(smem + SM::o_p);
  float* sM = reinterpret_cast<float*>(smem + SM::o_st);
  float* sL = sM + BMQ;
  float* sA = sL + BMQ;

  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int GM = G * M;
  const int n = ctx + M, start = sp * chunk, end = min(start + chunk, n);
  {
    constexpr int V16 = C * DSUB * 2 / 16;
    for (int s = 0; s < KST; ++s) {
      const uint4* gk = reinterpret_cast<const uint4*>(cbk + h * scb_h + s * scb_s);
      const uint4* gv = reinterpret_cast<const uint4*>(cbv + h * scb_h + s * scb_s);
      uint4* dk = reinterpret_cast<uint4*>(sCbK + s * C * DSUB);
      uint4* dv = reinterpret_cast<uint4*>(sCbV + s * C * DSUB);
      for (int i = tid; i < V16; i += NT) { dk[i] = gk[i]; dv[i] = gv[i]; }
    }
  }
  for (int r = 0; r < BMQ; ++r) {
    float q = 0.f;
    if (r < GM) {
      const int m = r / G, g = r % G;
      const __nv_bfloat16* qr = qkv + (long)(b * M + m) * sq_r + (h * G + g) * D;
      float x = __bfloat162float(qr[tid]), xp = __bfloat162float(qr[(tid + D / 2) % D]);
      float sn, cs;
      sincosf((float)(ctx + m) * invf[tid % (D / 2)], &sn, &cs);
      q = (x * cs + (tid < D / 2 ? -1.f : 1.f) * xp * sn) * qk_scale;
    }
    sQ[r * LDQ + tid] = __float2bfloat16(q);
  }
  for (int i = tid; i < BMQ * LDP; i += NT) sP[i] = __float2bfloat16(0.f);
  if (tid < BMQ) { sM[tid] = -INFINITY; sL[tid] = 0.f; }

  // O fragments: (BMQ/16) x 8 output tiles over 4 warps
  constexpr int OT = (BMQ / 16) * (D / 16) / 4;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[OT];
#pragma unroll
  for (int i = 0; i < OT; ++i) wmma::fill_fragment(fo[i], 0.f);

  const uint8_t* kb = kc + b * s_b + h * s_h;
  const uint8_t* vb = vc + b * s_b + h * s_h;
  // this thread's code bytes within a tile: token tk, sub-vectors j0 .. j0+BPT-1
  const int tk = (tid * BPT) / NSV, j0 = (tid * BPT) % NSV;
  CT ck[KST], cv[KST];
  auto load_codes = [&](int t0) {
    const int t = t0 + tk;
    const bool ok = t < end;
#pragma unroll
    for (int s = 0; s < KST; ++s) {
      ck[s] = ok ? *reinterpret_cast<const CT*>(kb + s * s_s + (long)t * s_t + j0) : CT(0);
      cv[s] = ok ? *reinterpret_cast<const CT*>(vb + s * s_s + (long)t * s_t + j0) : CT(0);
    }
  };
  if (start < end) load_codes(start);
  __syncthreads();

  for (int t0 = start; t0 < end; t0 += TILE) {
    // ---- dequant this tile's K and V (bf16-rounded stage sums) into sK / sV
#pragma unroll
    for (int jj = 0; jj < BPT; ++jj) {
#pragma unroll
      for (int hf = 0; hf < DSUB / 8; ++hf) {
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < KST; ++s) {
          const int c1 = (ck[s] >> (8 * jj)) & 0xFF, c2 = (cv[s] >> (8 * jj)) & 0xFF;
          add8(a, *reinterpret_cast<const uint4*>(sCbK + (s * C + c1) * DSUB + hf * 8));
          add8(v, *reinterpret_cast<const uint4*>(sCbV + (s * C + c2) * DSUB + hf * 8));
        }
        const int col = (j0 + jj) * DSUB + hf * 8;
        *reinterpret_cast<uint4*>(sK + tk * LDK + col) = pack8(a);
        *reinterpret_cast<uint4*>(sV + tk * LDK + col) = pack8(v);
      }
    }
    if (t0 + TILE < end) load_codes(t0 + TILE);  // prefetch next tile (in flight during the MMAs)
    __syncthreads();
    // ---- S = Q K^T
    for (int ti = warp; ti < (BMQ / 16) * (TILE / 16); ti += 4) {
      const int rb = ti / (TILE / 16), cb = ti % (TILE / 16);
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
      wmma::fill_fragment(fs, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + rb * 16 * LDQ + kk * 16, LDQ);
        wmma::load_matrix_sync(fb, sK + cb * 16 * LDK + kk * 16, LDK);
        wmma::mma_sync(fs, fa, fb, fs);
      }
      wmma::store_matrix_sync(sS + rb * 16 * LDSS + cb * 16, fs, LDSS, wmma::mem_row_major);
    }
    __syncthreads();
    // ---- online softmax (rows < GM), lane = column (TILE == 32)
    for (int r = warp; r < GM; r += 4) {
      const int t = t0 + lane;
      float v = (t < end && t <= ctx + r / G) ? sS[r * LDSS + lane] : -INFINITY;
      float mx = v;
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mo = sM[r];
      const float mn = fmaxf(mo, mx);
      const float ms = mn == -INFINITY ? 0.f : mn;
      const __nv_bfloat16 pb = __float2bfloat16(exp2f(v - ms));
      sP[r * LDP + lane] = pb;
      float sum = __bfloat162float(pb);
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      if (lane == 0) {
        const float al = exp2f(mo - ms);
        sA[r] = al;
        sL[r] = sL[r] * al + sum;
        sM[r] = mn;
      }
    }
    for (int r = GM + warp; r < BMQ; r += 4)
      if (lane == 0) sA[r] = 1.f;
    __syncthreads();
    // ---- O = O * alpha + P V
#pragma unroll
    for (int i = 0; i < OT; ++i) {
      const int ti = warp + 4 * i, rb = ti / (D / 16), cb = ti % (D / 16);
#pragma unroll
      for (int e = 0; e < fo[i].num_elements; ++e) fo[i].x[e] *= sA[rb * 16 + (lane >> 2) + 8 * ((e >> 1) & 1)];
#pragma unroll
      for (int kk = 0; kk < TILE / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, sP + rb * 16 * LDP + kk * 16, LDP);
        wmma::load_matrix_sync(fb, sV + kk * 16 * LDK + cb * 16, LDK);
        wmma::mma_sync(fo[i], fa, fb, fo[i]);
      }
    }
    __syncthreads();
  }
  // ---- epilogue: O fragments -> smem staging (sK/sV region) -> partial buffers
  float* sO = reinterpret_cast<float*>(smem + SM::o_kv);
#pragma unroll
  for (int i = 0; i < OT; ++i) {
    const int ti = warp + 4 * i, rb = ti / (D / 16), cb = ti % (D / 16);
    wmma::store_matrix_sync(sO + rb * 16 * LDO + cb * 16, fo[i], LDO, wmma::mem_row_major);
  }
  __syncthreads();
  const long o = (long)(bh * splits + sp) * BM;
  for (int r = 0; r < GM; ++r) OP[(o + r) * D + tid] = sO[r * LDO + tid];
  if (tid < GM) { MP[o + tid] = sM[tid]; LP[o + tid] = sL[tid]; }
}

template <int DSUB, int KST, int BMQ>
void launch(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
            torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int ctx, int chunk, int splits,
            int B, int Hkv, int BM, int M, long s_s, long s_b, long s_h, long s_t, long scb_h, long scb_s,
            float qk_scale) {
  auto k = vq_attn_mma_kernel<DSUB, KST, BMQ>;
  constexpr int sm = Smem<DSUB, KST, BMQ>::total;
  static_assert(sm <= 101376, "smem");
  static bool set = false;
  if (!set) {
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess, "smem attr");
    set = true;
  }
  dim3 grid(B * Hkv, splits);
  k<<<grid, NT, sm, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cbk.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cbv.data_ptr()),
      invf.data_ptr<float>(), OP.data_ptr<float>(), MP.data_ptr<float>(), LP.data_ptr<float>(), ctx, chunk, splits,
      Hkv, BM, M, qkv.stride(0), s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale);
}
}  // namespace


// ------------------------------------------------------------------------------------------------ v2: warp-split
// Each warp independently walks 16-token sub-tiles of the block's chunk (token tiles w, w+NW, ...): codes prefetched one
// sub-tile ahead in registers (lane = token lane/2, half of its sub-vectors), K dequantized into a warp-private bf16
// tile, S = Q K^T (WMMA), V dequantized into the same tile, warp-private online softmax (lane = row), O += P V (WMMA,
// O in fragments). No block barriers inside the loop. Each warp writes its own partial (O, m, l) as split
// sp * NW + warp; the Triton combine merges splits * NW partials.
template <int BMQ>
struct WarpSm {
  static constexpr int kv = 16 * LDK * 2;
  static constexpr int s = BMQ * 20 * 4;
  static constexpr int p = BMQ * 24 * 2;
  static constexpr int st = 3 * BMQ * 4;
  static constexpr int o_kv = 0, o_s = al128(o_kv + kv), o_p = al128(o_s + s), o_st = al128(o_p + p),
                       total = al128(o_st + st);
};

template <int DSUB, int KST, int BMQ, int NW>
struct Smem2 {
  static constexpr int cb = 2 * KST * C * DSUB * 2;
  static constexpr int q = BMQ * LDQ * 2;
  static constexpr int o_q = al128(cb), o_w = al128(o_q + q), total = o_w + NW * WarpSm<BMQ>::total;
};

template <int NSV>
struct Code2;
template <>
struct Code2<16> { using T = uint2; static constexpr int BPL = 8; };
template <>
struct Code2<8> { using T = uint32_t; static constexpr int BPL = 4; };

__device__ __forceinline__ int byte_of(uint2 w, int i) { return ((i < 4 ? w.x : w.y) >> (8 * (i & 3))) & 0xFF; }
__device__ __forceinline__ int byte_of(uint32_t w, int i) { return (w >> (8 * i)) & 0xFF; }

template <int DSUB, int KST, int BMQ, int NW>
__global__ void __launch_bounds__(NW * 32) vq_attn_w_kernel(
    const __nv_bfloat16* __restrict__ qkv, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const __nv_bfloat16* __restrict__ cbk, const __nv_bfloat16* __restrict__ cbv, const float* __restrict__ invf,
    float* __restrict__ OP, float* __restrict__ MP, float* __restrict__ LP, int ctx, int chunk, int splits, int Hkv,
    int BM, int M, long sq_r, long s_s, long s_b, long s_h, long s_t, long scb_h, long scb_s, float qk_scale) {
  constexpr int NSV = D / DSUB;
  using SM = Smem2<DSUB, KST, BMQ, NW>;
  using WS = WarpSm<BMQ>;
  using CT = typename Code2<NSV>::T;
  constexpr int BPL = Code2<NSV>::BPL;
  constexpr int NTH = NW * 32;
  extern __shared__ __align__(128) uint8_t smem[];
  __nv_bfloat16* sCbK = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sCbV = sCbK + KST * C * DSUB;
  __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem + SM::o_q);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint8_t* wbase = smem + SM::o_w + warp * WS::total;
  __nv_bfloat16* sKV = reinterpret_cast<__nv_bfloat16*>(wbase + WS::o_kv);
  float* sS = reinterpret_cast<float*>(wbase + WS::o_s);
  __nv_bfloat16* sP = reinterpret_cast<__nv_bfloat16*>(wbase + WS::o_p);
  float* sM = reinterpret_cast<float*>(wbase + WS::o_st);
  float* sL = sM + BMQ;
  float* sA = sL + BMQ;

  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv;
  const int GM = G * M;
  const int n = ctx + M, start = sp * chunk, end = min(start + chunk, n);
  {
    constexpr int V16 = C * DSUB * 2 / 16;
    for (int s = 0; s < KST; ++s) {
      const uint4* gk = reinterpret_cast<const uint4*>(cbk + h * scb_h + s * scb_s);
      const uint4* gv = reinterpret_cast<const uint4*>(cbv + h * scb_h + s * scb_s);
      uint4* dk = reinterpret_cast<uint4*>(sCbK + s * C * DSUB);
      uint4* dv = reinterpret_cast<uint4*>(sCbV + s * C * DSUB);
      for (int i = tid; i < V16; i += NTH) { dk[i] = gk[i]; dv[i] = gv[i]; }
    }
  }
  for (int i = tid; i < BMQ * D; i += NTH) {
    const int r = i / D, c = i % D;
    float q = 0.f;
    if (r < GM) {
      const int m = r / G, g = r % G;
      const __nv_bfloat16* qr = qkv + (long)(b * M + m) * sq_r + (h * G + g) * D;
      float x = __bfloat162float(qr[c]), xp = __bfloat162float(qr[(c + D / 2) % D]);
      float sn, cs;
      sincosf((float)(ctx + m) * invf[c % (D / 2)], &sn, &cs);
      q = (x * cs + (c < D / 2 ? -1.f : 1.f) * xp * sn) * qk_scale;
    }
    sQ[r * LDQ + c] = __float2bfloat16(q);
  }
  for (int i = lane; i < BMQ * 24; i += 32) sP[i] = __float2bfloat16(0.f);
  if (lane < BMQ) { sM[lane] = -INFINITY; sL[lane] = 0.f; sA[lane] = 1.f; }
  constexpr int OT = (BMQ / 16) * (D / 16);
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo[OT];
#pragma unroll
  for (int i = 0; i < OT; ++i) wmma::fill_fragment(fo[i], 0.f);

  const uint8_t* kb = kc + b * s_b + h * s_h;
  const uint8_t* vb = vc + b * s_b + h * s_h;
  const int tk = lane >> 1, j0 = (lane & 1) * BPL;  // token in the 16-token sub-tile, first sub-vector
  CT ck[KST], cv[KST];
  auto load_codes = [&](int t0) {
    const int t = t0 + tk;
    const bool ok = t < end;
#pragma unroll
    for (int s = 0; s < KST; ++s) {
      if (ok) {
        ck[s] = *reinterpret_cast<const CT*>(kb + s * s_s + (long)t * s_t + j0);
        cv[s] = *reinterpret_cast<const CT*>(vb + s * s_s + (long)t * s_t + j0);
      } else {
        ck[s] = CT{};
        cv[s] = CT{};
      }
    }
  };
  auto dequant = [&](const __nv_bfloat16* cb, const CT* cc) {
#pragma unroll
    for (int jj = 0; jj < BPL; ++jj) {
#pragma unroll
      for (int hf = 0; hf < DSUB / 8; ++hf) {
        float a[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < KST; ++s)
          add8(a, *reinterpret_cast<const uint4*>(cb + (s * C + byte_of(cc[s], jj)) * DSUB + hf * 8));
        *reinterpret_cast<uint4*>(sKV + tk * LDK + (j0 + jj) * DSUB + hf * 8) = pack8(a);
      }
    }
  };
  __syncthreads();
  const int first = start + 16 * warp;
  if (first < end) load_codes(first);
  for (int t0 = first; t0 < end; t0 += 16 * NW) {
    CT nk[KST], nv[KST];
    dequant(sCbK, ck);
#pragma unroll
    for (int s = 0; s < KST; ++s) { nk[s] = cv[s]; }  // keep V codes of this sub-tile
    if (t0 + 16 * NW < end) load_codes(t0 + 16 * NW);  // prefetch (ck/cv now hold the next sub-tile)
#pragma unroll
    for (int s = 0; s < KST; ++s) { nv[s] = nk[s]; }
    __syncwarp();
#pragma unroll
    for (int rb = 0; rb < BMQ / 16; ++rb) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
      wmma::fill_fragment(fs, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
        wmma::load_matrix_sync(fa, sQ + rb * 16 * LDQ + kk * 16, LDQ);
        wmma::load_matrix_sync(fb, sKV + kk * 16, LDK);
        wmma::mma_sync(fs, fa, fb, fs);
      }
      wmma::store_matrix_sync(sS + rb * 16 * 20, fs, 20, wmma::mem_row_major);
    }
    __syncwarp();
    dequant(sCbV, nv);  // V of this sub-tile into the same tile buffer
    if (lane < GM) {  // softmax, lane = row
      const int r = lane;
      const int lim = ctx + r / G;
      float v[16];
      float mx = -INFINITY;
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const int t = t0 + c;
        v[c] = (t < end && t <= lim) ? sS[r * 20 + c] : -INFINITY;
        mx = fmaxf(mx, v[c]);
      }
      const float mo = sM[r];
      const float mn = fmaxf(mo, mx);
      const float ms = mn == -INFINITY ? 0.f : mn;
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) {
        const __nv_bfloat16 pb = __float2bfloat16(exp2f(v[c] - ms));
        sP[r * 24 + c] = pb;
        sum += __bfloat162float(pb);
      }
      const float al = exp2f(mo - ms);
      sA[r] = al;
      sL[r] = sL[r] * al + sum;
      sM[r] = mn;
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < OT; ++i) {
      const int rb = i / (D / 16), cb = i % (D / 16);
#pragma unroll
      for (int e = 0; e < fo[i].num_elements; ++e) fo[i].x[e] *= sA[rb * 16 + (lane >> 2) + 8 * ((e >> 1) & 1)];
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fb;
      wmma::load_matrix_sync(fa, sP + rb * 16 * 24, 24);
      wmma::load_matrix_sync(fb, sKV + cb * 16, LDK);
      wmma::mma_sync(fo[i], fa, fb, fo[i]);
    }
    __syncwarp();
  }
  // partial of this warp -> split index sp * NW + warp (staging through the warp's sKV/sS region as fp32 [16][D+4])
  const long o = (long)(bh * splits * NW + sp * NW + warp) * BM;
  float* stg = reinterpret_cast<float*>(wbase + WS::o_kv);  // [16][68] fp32 = 4352 bytes = the sKV tile
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
  __syncwarp();
  if (lane < GM) { MP[o + lane] = sM[lane]; LP[o + lane] = sL[lane]; }
}

template <int DSUB, int KST, int BMQ, int NW>
void launch_w(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
              torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int ctx, int chunk,
              int splits, int B, int Hkv, int BM, int M, long s_s, long s_b, long s_h, long s_t, long scb_h,
              long scb_s, float qk_scale) {
  auto k = vq_attn_w_kernel<DSUB, KST, BMQ, NW>;
  constexpr int sm = Smem2<DSUB, KST, BMQ, NW>::total;
  static_assert(sm <= 101376, "smem");
  static_assert(16 * 68 * 4 <= WarpSm<BMQ>::kv, "staging");
  static bool set = false;
  if (!set) {
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess, "smem attr");
    set = true;
  }
  dim3 grid(B * Hkv, splits);
  k<<<grid, NW * 32, sm, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cbk.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cbv.data_ptr()),
      invf.data_ptr<float>(), OP.data_ptr<float>(), MP.data_ptr<float>(), LP.data_ptr<float>(), ctx, chunk, splits,
      Hkv, BM, M, qkv.stride(0), s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale);
}

// warps per block for (d, kst, bmq): 4 unless the shared memory needs 2
int64_t vq_attn_w_nw(int64_t dsub, int64_t kst, int64_t bmq) {
  return (dsub == 16 && kst == 4) ? 2 : 4;
}

void vq_attn_w(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
               torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int64_t ctx, int64_t chunk,
               int64_t splits, int64_t B, int64_t Hkv, int64_t BM, int64_t M, int64_t dsub, int64_t kst, int64_t s_s,
               int64_t s_b, int64_t s_h, int64_t s_t, int64_t scb_h, int64_t scb_s, double qk_scale) {
  TORCH_CHECK(G * M <= 32, "G*M <= 32");
  const int bmq = G * M <= 16 ? 16 : 32;
#define ARGW qkv, kc, vc, cbk, cbv, invf, OP, MP, LP, (int)ctx, (int)chunk, (int)splits, (int)B, (int)Hkv, (int)BM, \
             (int)M, s_s, s_b, s_h, s_t, scb_h, scb_s, (float)qk_scale
#define CASE_W(DS, KS, NW_) \
  if (bmq == 16) launch_w<DS, KS, 16, NW_>(ARGW); else launch_w<DS, KS, 32, NW_>(ARGW);
  if (dsub == 8) {
    if (kst == 1) { CASE_W(8, 1, 4) } else if (kst == 2) { CASE_W(8, 2, 4) } else TORCH_CHECK(false, "d8: kst 1/2");
  } else if (dsub == 16) {
    if (kst == 1) { CASE_W(16, 1, 4) } else if (kst == 2) { CASE_W(16, 2, 4) } else if (kst == 4) { CASE_W(16, 4, 2) }
    else TORCH_CHECK(false, "d16: kst 1/2/4");
  } else {
    TORCH_CHECK(false, "dsub 8/16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t vq_attn_mma_smem(int64_t dsub, int64_t kst, int64_t bmq) {
#define SZ(DS, KS) (bmq == 16 ? Smem<DS, KS, 16>::total : Smem<DS, KS, 32>::total)
  if (dsub == 8) return kst == 1 ? SZ(8, 1) : SZ(8, 2);
  return kst == 1 ? SZ(16, 1) : (kst == 2 ? SZ(16, 2) : SZ(16, 4));
}

void vq_attn_mma(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
                 torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int64_t ctx, int64_t chunk,
                 int64_t splits, int64_t B, int64_t Hkv, int64_t BM, int64_t M, int64_t dsub, int64_t kst, int64_t s_s,
                 int64_t s_b, int64_t s_h, int64_t s_t, int64_t scb_h, int64_t scb_s, double qk_scale) {
  TORCH_CHECK(chunk % TILE == 0, "chunk must be a multiple of 32");
  TORCH_CHECK(G * M <= 32, "G*M <= 32");
  const int bmq = G * M <= 16 ? 16 : 32;
#define ARGS qkv, kc, vc, cbk, cbv, invf, OP, MP, LP, (int)ctx, (int)chunk, (int)splits, (int)B, (int)Hkv, (int)BM, \
             (int)M, s_s, s_b, s_h, s_t, scb_h, scb_s, (float)qk_scale
#define CASE_B(DS, KS) \
  if (bmq == 16) launch<DS, KS, 16>(ARGS); else launch<DS, KS, 32>(ARGS);
  if (dsub == 8) {
    if (kst == 1) { CASE_B(8, 1) } else if (kst == 2) { CASE_B(8, 2) } else TORCH_CHECK(false, "d8: kst 1/2");
  } else if (dsub == 16) {
    if (kst == 1) { CASE_B(16, 1) } else if (kst == 2) { CASE_B(16, 2) } else if (kst == 4) { CASE_B(16, 4) }
    else TORCH_CHECK(false, "d16: kst 1/2/4");
  } else {
    TORCH_CHECK(false, "dsub 8/16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_attn_mma", &vq_attn_mma);
  m.def("vq_attn_mma_smem", &vq_attn_mma_smem);
  m.def("vq_attn_w", &vq_attn_w);
  m.def("vq_attn_w_nw", &vq_attn_w_nw);
}
