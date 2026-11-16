// E6 kvregime: decode attention over a nested VQ KV cache, CUDA (codebooks in shared memory).
// One block (128 threads) per (sequence b, kv-head h, context split sp). D = 128, G = 4 query heads per kv-head,
// M new query rows per sequence (GM = G*M stacked rows r = m*G + g). Per 128-token tile:
//   K phase : thread = token; its K codes (NSV bytes per stage, vector load) -> codeword gathers from smem (16 B vector),
//             summed over the first KST stages, dotted with all GM q rows (q in smem, broadcast reads) -> scores in smem.
//   softmax : online (exp2; q pre-scaled by D^-0.5 log2 e), one warp per row.
//   V phase : thread = (8-dim group dg, token group tg); V codes staged in smem; acc[GM][8] in registers.
// Partial (acc, m, l) per split -> combined by the Triton _attn_combine kernel (same layout as the Triton path).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int D = 128, TILE = 128, NT = 128, C = 256, G = 4;

__device__ __forceinline__ void add8(float* a, uint4 w) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(p[i]);
    a[2 * i] += f.x;
    a[2 * i + 1] += f.y;
  }
}

template <int DSUB, int KST, int M>
__global__ void __launch_bounds__(NT) vq_attn_kernel(
    const __nv_bfloat16* __restrict__ qkv, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const __nv_bfloat16* __restrict__ cbk, const __nv_bfloat16* __restrict__ cbv, const float* __restrict__ invf,
    float* __restrict__ OP, float* __restrict__ MP, float* __restrict__ LP, int ctx, int chunk, int splits, int Hkv,
    int BM, long sq_r, long s_s, long s_b, long s_h, long s_t, long scb_h, long scb_s, float qk_scale) {
  constexpr int NSV = D / DSUB;
  constexpr int GM = G * M;
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* sCbK = reinterpret_cast<__nv_bfloat16*>(smem);           // [KST][C][DSUB]
  __nv_bfloat16* sCbV = sCbK + KST * C * DSUB;                             // [KST][C][DSUB]
  float* sQ = reinterpret_cast<float*>(sCbV + KST * C * DSUB);             // [GM][D]
  float* sS = sQ + GM * D;                                                 // [GM][TILE] (reused as acc [GM][D])
  float* sM = sS + GM * TILE;                                              // [GM]
  float* sL = sM + GM;                                                     // [GM]
  float* sA = sL + GM;                                                     // [GM]
  uint8_t* sVC = reinterpret_cast<uint8_t*>(sM + ((3 * GM + 3) / 4) * 4);  // [KST][TILE][NSV]

  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv, tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int n = ctx + M, start = sp * chunk, end = min(start + chunk, n);

  {  // codebooks -> smem (16 B vectors)
    constexpr int V16 = C * DSUB * 2 / 16;
    for (int s = 0; s < KST; ++s) {
      const uint4* gk = reinterpret_cast<const uint4*>(cbk + h * scb_h + s * scb_s);
      const uint4* gv = reinterpret_cast<const uint4*>(cbv + h * scb_h + s * scb_s);
      uint4* dk = reinterpret_cast<uint4*>(sCbK + s * C * DSUB);
      uint4* dv = reinterpret_cast<uint4*>(sCbV + s * C * DSUB);
      for (int i = tid; i < V16; i += NT) { dk[i] = gk[i]; dv[i] = gv[i]; }
    }
  }
  // q rows with RoPE (rotate-half), pre-scaled; element tid of each row
#pragma unroll
  for (int r = 0; r < GM; ++r) {
    const int m = r / G, g = r % G;
    const __nv_bfloat16* qr = qkv + (long)(b * M + m) * sq_r + (h * G + g) * D;
    float x = __bfloat162float(qr[tid]), xp = __bfloat162float(qr[(tid + D / 2) % D]);
    float ang = (float)(ctx + m) * invf[tid % (D / 2)];
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float q = (x * cs + (tid < D / 2 ? -1.f : 1.f) * xp * sn) * qk_scale;
    sQ[r * D + tid] = __bfloat162float(__float2bfloat16(q));
  }
  if (tid < GM) { sM[tid] = -INFINITY; sL[tid] = 0.f; }
  float acc[GM][8];
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
  const int dg = tid & 15, tg = tid >> 4;  // V phase mapping: 16 dim groups of 8, 8 token groups
  const int vj = (dg * 8) / DSUB, voff = (dg * 8) % DSUB;
  const uint8_t* kb = kc + b * s_b + h * s_h;
  const uint8_t* vb = vc + b * s_b + h * s_h;
  __syncthreads();

  for (int t0 = start; t0 < end; t0 += TILE) {
    const int t = t0 + tid;
    const bool valid = t < end;
    // ---- K phase
    float sc[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) sc[r] = 0.f;
    if (valid) {
      uint8_t code[KST][NSV];
#pragma unroll
      for (int s = 0; s < KST; ++s) {
        if constexpr (NSV == 16) {
          *reinterpret_cast<uint4*>(code[s]) = *reinterpret_cast<const uint4*>(kb + s * s_s + (long)t * s_t);
        } else {
          *reinterpret_cast<uint2*>(code[s]) = *reinterpret_cast<const uint2*>(kb + s * s_s + (long)t * s_t);
        }
        // stage V codes of this token into smem for the V phase
        if constexpr (NSV == 16) {
          *reinterpret_cast<uint4*>(sVC + (s * TILE + tid) * NSV) =
              *reinterpret_cast<const uint4*>(vb + s * s_s + (long)t * s_t);
        } else {
          *reinterpret_cast<uint2*>(sVC + (s * TILE + tid) * NSV) =
              *reinterpret_cast<const uint2*>(vb + s * s_s + (long)t * s_t);
        }
      }
#pragma unroll
      for (int j = 0; j < NSV; ++j) {
#pragma unroll
        for (int hf = 0; hf < DSUB / 8; ++hf) {
          float k8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int s = 0; s < KST; ++s)
            add8(k8, *reinterpret_cast<const uint4*>(sCbK + (s * C + code[s][j]) * DSUB + hf * 8));
          // keys are bf16-rounded sums (as the Triton path / reference)
#pragma unroll
          for (int i = 0; i < 8; ++i) k8[i] = __bfloat162float(__float2bfloat16(k8[i]));
          const int col = j * DSUB + hf * 8;
#pragma unroll
          for (int r = 0; r < GM; ++r) {
            float4 a = *reinterpret_cast<const float4*>(sQ + r * D + col);
            float4 c = *reinterpret_cast<const float4*>(sQ + r * D + col + 4);
            sc[r] += a.x * k8[0] + a.y * k8[1] + a.z * k8[2] + a.w * k8[3] + c.x * k8[4] + c.y * k8[5] +
                     c.z * k8[6] + c.w * k8[7];
          }
        }
      }
    }
#pragma unroll
    for (int r = 0; r < GM; ++r) sS[r * TILE + tid] = (valid && t <= ctx + r / G) ? sc[r] : -INFINITY;
    __syncthreads();
    // ---- online softmax, one warp per row
    for (int r = warp; r < GM; r += NT / 32) {
      float v[TILE / 32];
      float mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < TILE / 32; ++i) { v[i] = sS[r * TILE + lane + 32 * i]; mx = fmaxf(mx, v[i]); }
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float mo = sM[r];
      const float mn = fmaxf(mo, mx);
      const float ms = mn == -INFINITY ? 0.f : mn;
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < TILE / 32; ++i) {
        float p = exp2f(v[i] - ms);
        sS[r * TILE + lane + 32 * i] = p;
        sum += p;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      if (lane == 0) {
        const float al = exp2f(mo - ms);
        sA[r] = al;
        sL[r] = sL[r] * al + sum;
        sM[r] = mn;
      }
    }
    __syncthreads();
    // ---- V phase
#pragma unroll
    for (int r = 0; r < GM; ++r) {
      const float al = sA[r];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[r][i] *= al;
    }
    const int nt = min(TILE, end - t0);
    for (int tt = tg; tt < nt; tt += NT / 16) {
      float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < KST; ++s)
        add8(v8, *reinterpret_cast<const uint4*>(sCbV + (s * C + sVC[(s * TILE + tt) * NSV + vj]) * DSUB + voff));
#pragma unroll
      for (int i = 0; i < 8; ++i) v8[i] = __bfloat162float(__float2bfloat16(v8[i]));
#pragma unroll
      for (int r = 0; r < GM; ++r) {
        // p rounded to bf16 like the Triton path (tl.dot(p.to(bf16), v))
        const float p = __bfloat162float(__float2bfloat16(sS[r * TILE + tt]));
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][i] += p * v8[i];
      }
    }
    __syncthreads();
  }
  // ---- reduce acc over the 8 token groups (sS reused as [GM][D]) and write partials
  for (int i = tid; i < GM * D; i += NT) sS[i] = 0.f;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) atomicAdd(&sS[r * D + dg * 8 + i], acc[r][i]);
  __syncthreads();
  const long o = (long)(bh * splits + sp) * BM;
#pragma unroll
  for (int r = 0; r < GM; ++r) OP[(o + r) * D + tid] = sS[r * D + tid];
  if (tid < GM) { MP[o + tid] = sM[tid]; LP[o + tid] = sL[tid]; }
}

template <int DSUB, int KST, int M>
size_t smem_bytes() {
  constexpr int NSV = D / DSUB, GM = G * M;
  size_t s = 2 * KST * C * DSUB * 2 + GM * D * 4 + GM * TILE * 4 + ((3 * GM + 3) / 4) * 4 * 4 + KST * TILE * NSV;
  return s;
}

template <int DSUB, int KST, int M>
void launch(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
            torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int ctx, int chunk, int splits,
            int B, int Hkv, int BM, long s_s, long s_b, long s_h, long s_t, long scb_h, long scb_s, float qk_scale) {
  auto k = vq_attn_kernel<DSUB, KST, M>;
  size_t sm = smem_bytes<DSUB, KST, M>();
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    set = true;
  }
  dim3 grid(B * Hkv, splits);
  k<<<grid, NT, sm, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cbk.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(cbv.data_ptr()),
      invf.data_ptr<float>(), OP.data_ptr<float>(), MP.data_ptr<float>(), LP.data_ptr<float>(), ctx, chunk, splits,
      Hkv, BM, qkv.stride(0), s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale);
}
}  // namespace

void vq_attn_split(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor cbk, torch::Tensor cbv,
                   torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, int64_t ctx,
                   int64_t chunk, int64_t splits, int64_t B, int64_t Hkv, int64_t BM, int64_t M, int64_t dsub,
                   int64_t kst, int64_t s_s, int64_t s_b, int64_t s_h, int64_t s_t, int64_t scb_h, int64_t scb_s,
                   double qk_scale) {
  TORCH_CHECK(chunk % TILE == 0, "chunk must be a multiple of 128");
#define ARGS qkv, kc, vc, cbk, cbv, invf, OP, MP, LP, (int)ctx, (int)chunk, (int)splits, (int)B, (int)Hkv, (int)BM, \
             s_s, s_b, s_h, s_t, scb_h, scb_s, (float)qk_scale
#define CASE_M(DS, KS)                                    \
  switch (M) {                                            \
    case 1: launch<DS, KS, 1>(ARGS); break;               \
    case 2: launch<DS, KS, 2>(ARGS); break;               \
    case 3: launch<DS, KS, 3>(ARGS); break;               \
    case 5: launch<DS, KS, 5>(ARGS); break;               \
    default: TORCH_CHECK(false, "M must be 1/2/3/5");     \
  }
  if (dsub == 8) {
    if (kst == 1) { CASE_M(8, 1) } else if (kst == 2) { CASE_M(8, 2) } else TORCH_CHECK(false, "d8: kst 1/2");
  } else if (dsub == 16) {
    if (kst == 1) { CASE_M(16, 1) } else if (kst == 2) { CASE_M(16, 2) } else if (kst == 4) { CASE_M(16, 4) }
    else TORCH_CHECK(false, "d16: kst 1/2/4");
  } else {
    TORCH_CHECK(false, "dsub 8/16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("vq_attn_split", &vq_attn_split); }
