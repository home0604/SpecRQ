// E7 magicvq v5 (real e2e): ragged copy of specrq/kvq/csrc/vqattn_lut.cu (LUT scores / register-FMA values), reduced
// to what the real decode loop needs: M = 1 draft rows, uniform 8-bit stages (d = 8, K256 x S, FMT 0), KST 1 or 2.
// Changes vs the original (specrq/kvq is not modified):
//   * per-(b, kv-head) token range [LO[bh], HI[bh]) of the code planes (device tensors: CUDA-graph safe, ragged);
//     no causal mask (the VQ part never contains the query's own rows);
//   * q RoPE angle from QPOS[b] (true position of the query row);
//   * static per-channel standardization of the codebooks (calib --chnorm): K_hat = sdk * sum_s cb_s + muk,
//     V_hat = sdv * sum_s cb_s + muv, folded exactly: q' = q * sdk for the LUT, score offset q . muk added to the
//     partial max, O = sdv * O' + muv * l at the partial store;
//   * partials written at split offset OFF of a shared workspace with SPT splits per (b, kv-head), so a bf16 partial
//     kernel can write the other splits and one combine merges both sets.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int D = 128, DS = 8, NSV = 16, G = 4, C = 256, NW = 16, GM = G;  // M = 1

struct KVP {
  const uint8_t* kc[2];
  const uint8_t* vc[2];
  long sb, sh;  // plane strides (batch, head); token stride 16 bytes
  const __nv_bfloat16* kcb[2];
  const __nv_bfloat16* vcb[2];
  long cbh;  // codebook head stride
};

__host__ __device__ constexpr int al16(int x) { return (x + 15) / 16 * 16; }
template <int KST>
struct Lay {
  static constexpr int vcb_off(int g) { return g * C * DS * 2; }
  static constexpr int lut_off(int g) { return al16(vcb_off(KST)) + g * NSV * C * GM * 2; }
  static constexpr int warp_bytes = KST * 32 * 16 + GM * 32 * 4;
  static constexpr int warp_off = al16(lut_off(KST));
  static constexpr int qbytes = GM * D * 4;
  static constexpr int tail = NW * warp_bytes > qbytes ? NW * warp_bytes : qbytes;
  static constexpr int qmu_off = warp_off + tail;
  static constexpr int total = qmu_off + 64;
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

template <int KST>
__global__ void __launch_bounds__(NW * 32) lut_rag_kernel(
    const __nv_bfloat16* __restrict__ qkv, KVP P, const float* __restrict__ invf, float* __restrict__ OP,
    float* __restrict__ MP, float* __restrict__ LP, const int* __restrict__ LO, const int* __restrict__ HI,
    const int* __restrict__ QPOS, const float* __restrict__ SDK, const float* __restrict__ MUK,
    const float* __restrict__ SDV, const float* __restrict__ MUV, int chunk, int Hkv, int BM, int SPT, int OFF,
    long sq_r, float qk_scale) {
  using L = Lay<KST>;
  constexpr int NTH = NW * 32;
  extern __shared__ __align__(16) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bh = blockIdx.x, sp = blockIdx.y, b = bh / Hkv, h = bh % Hkv;
  const int start = LO[bh] + sp * chunk, end = min(start + chunk, HI[bh]);
  const int qpos = QPOS[b];
  float* sQ = reinterpret_cast<float*>(smem + L::warp_off);
  float* sQmu = reinterpret_cast<float*>(smem + L::qmu_off);
  // q . muk per q-head: deterministic fixed-order reduction (was a shared-memory float atomicAdd, whose order varied
  // between launches -> draft logits not bit-reproducible). One (r, c) per thread: warp shuffle tree, then the
  // D / 32 warp partials of each row are summed in fixed order below.
  static_assert(NTH == GM * D && NW * 4 <= 64, "one (r, c) per thread; NW partials fit the 64-byte qmu slot");
  {
    const int r = tid / D, c = tid % D;  // M = 1: r = q-head within the group
    const __nv_bfloat16* qr = qkv + (long)b * sq_r + (h * G + r) * D;
    float x = __bfloat162float(qr[c]), xp = __bfloat162float(qr[(c + D / 2) % D]);
    float sn, cs;
    sincosf((float)qpos * invf[c % (D / 2)], &sn, &cs);
    const float rot = __bfloat162float(__float2bfloat16(x * cs + (c < D / 2 ? -1.f : 1.f) * xp * sn));
    sQ[r * D + c] = rot * SDK[h * D + c] * qk_scale;
    float pm = rot * MUK[h * D + c] * qk_scale;
#pragma unroll
    for (int o = 16; o; o >>= 1) pm += __shfl_xor_sync(0xffffffff, pm, o);
    if (lane == 0) sQmu[warp] = pm;  // warp w holds row w / (D / 32)
  }
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
        for (int rr = 0; rr < 4; ++rr) {
          const float4 a = *reinterpret_cast<const float4*>(sQ + rr * D + j * DS);
          const float4 bq = *reinterpret_cast<const float4*>(sQ + rr * D + j * DS + 4);
          v[rr] = __float2half(a.x * cw[0] + a.y * cw[1] + a.z * cw[2] + a.w * cw[3] + bq.x * cw[4] + bq.y * cw[5] +
                               bq.z * cw[6] + bq.w * cw[7]);
        }
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
  const int dg = lane & 15, tp = lane >> 4;
  const long off = b * P.sb + h * P.sh;

  for (int t0 = start + 32 * warp; t0 < end; t0 += 32 * NW) {
    const int t = t0 + lane;
    const bool ok = t < end;
    float sc[GM];
#pragma unroll
    for (int r = 0; r < GM; ++r) sc[r] = 0.f;
#pragma unroll
    for (int g = 0; g < KST; ++g) {
      const __half* lut = reinterpret_cast<const __half*>(smem + L::lut_off(g));
      uint32_t kc[4] = {0, 0, 0, 0}, vc[4] = {0, 0, 0, 0};
      if (ok) {
        const uint4 kw = *reinterpret_cast<const uint4*>(P.kc[g] + off + (long)t * 16);
        const uint4 vw = *reinterpret_cast<const uint4*>(P.vc[g] + off + (long)t * 16);
        kc[0] = kw.x; kc[1] = kw.y; kc[2] = kw.z; kc[3] = kw.w;
        vc[0] = vw.x; vc[1] = vw.y; vc[2] = vw.z; vc[3] = vw.w;
      }
      *reinterpret_cast<uint4*>(sVC + (g * 32 + lane) * 16) = make_uint4(vc[0], vc[1], vc[2], vc[3]);
#pragma unroll
      for (int j = 0; j < NSV; ++j) {
        const int code = (kc[j >> 2] >> (8 * (j & 3))) & 0xFF;
        add_lut4(sc, lut + (long)(j * C + code) * GM);
      }
    }
    float alpha[GM];
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
  const long o = ((long)bh * SPT + OFF + sp * NW + warp) * BM;
#pragma unroll
  for (int r = 0; r < GM; ++r)
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[r][i] += __shfl_xor_sync(0xffffffff, acc[r][i], 16);
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
}

template <int KST>
void launch(torch::Tensor qkv, const KVP& P, torch::Tensor invf, torch::Tensor OP, torch::Tensor MP, torch::Tensor LP,
            torch::Tensor LO, torch::Tensor HI, torch::Tensor QPOS, torch::Tensor SDK, torch::Tensor MUK,
            torch::Tensor SDV, torch::Tensor MUV, int chunk, int splits, int B, int Hkv, int BM, int SPT, int OFF,
            float qk_scale) {
  using L = Lay<KST>;
  static_assert(L::total <= 101376, "smem");
  auto k = lut_rag_kernel<KST>;
  static bool set = false;
  if (!set) {
    TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess, "smem");
    set = true;
  }
  dim3 grid(B * Hkv, splits);
  k<<<grid, NW * 32, L::total, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()), P, invf.data_ptr<float>(), OP.data_ptr<float>(),
      MP.data_ptr<float>(), LP.data_ptr<float>(), LO.data_ptr<int>(), HI.data_ptr<int>(), QPOS.data_ptr<int>(),
      SDK.data_ptr<float>(), MUK.data_ptr<float>(), SDV.data_ptr<float>(), MUV.data_ptr<float>(), chunk, Hkv, BM, SPT,
      OFF, qkv.stride(0), qk_scale);
}
}  // namespace

// kpl / vpl: per stage [B, Hkv, T, 16] uint8 token-contiguous planes; kcb / vcb: per stage [Hkv, 256, 8] bf16.
// Partials: (bh * SPT + OFF + sp * 16 + warp) * BM rows; the caller reserves splits * 16 slots at OFF.
void lut_rag(torch::Tensor qkv, std::vector<torch::Tensor> kpl, std::vector<torch::Tensor> vpl,
             std::vector<torch::Tensor> kcb, std::vector<torch::Tensor> vcb, torch::Tensor invf, torch::Tensor OP,
             torch::Tensor MP, torch::Tensor LP, torch::Tensor LO, torch::Tensor HI, torch::Tensor QPOS,
             torch::Tensor SDK, torch::Tensor MUK, torch::Tensor SDV, torch::Tensor MUV, int64_t chunk, int64_t splits,
             int64_t B, int64_t Hkv, int64_t BM, int64_t SPT, int64_t OFF, int64_t kst, double qk_scale) {
  KVP P{};
  TORCH_CHECK(kst == 1 || kst == 2, "kst");
  for (int s = 0; s < kst; ++s) {
    P.kc[s] = kpl[s].data_ptr<uint8_t>();
    P.vc[s] = vpl[s].data_ptr<uint8_t>();
    TORCH_CHECK(kpl[s].stride(3) == 1 && kpl[s].size(3) == 16 && kpl[s].stride(2) == 16, "plane layout");
    TORCH_CHECK(kpl[s].stride(0) == vpl[s].stride(0) && kpl[s].stride(1) == vpl[s].stride(1), "K/V strides");
    TORCH_CHECK(s == 0 || (kpl[s].stride(0) == kpl[0].stride(0) && kpl[s].stride(1) == kpl[0].stride(1)), "stages");
    P.kcb[s] = reinterpret_cast<const __nv_bfloat16*>(kcb[s].data_ptr());
    P.vcb[s] = reinterpret_cast<const __nv_bfloat16*>(vcb[s].data_ptr());
    TORCH_CHECK(kcb[s].stride(2) == 1 && kcb[s].stride(1) == DS && kcb[s].stride(0) == vcb[s].stride(0), "cb");
  }
  P.sb = kpl[0].stride(0);
  P.sh = kpl[0].stride(1);
  P.cbh = kcb[0].stride(0);
  TORCH_CHECK(LO.dtype() == torch::kInt32 && HI.dtype() == torch::kInt32 && QPOS.dtype() == torch::kInt32, "int32");
  if (kst == 1)
    launch<1>(qkv, P, invf, OP, MP, LP, LO, HI, QPOS, SDK, MUK, SDV, MUV, chunk, splits, B, Hkv, BM, SPT, OFF,
              qk_scale);
  else
    launch<2>(qkv, P, invf, OP, MP, LP, LO, HI, QPOS, SDK, MUK, SDV, MUV, chunk, splits, B, Hkv, BM, SPT, OFF,
              qk_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("lut_rag", &lut_rag); }
