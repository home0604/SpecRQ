// E7 e2e opt-in (MV_LUT_VARIANT, e2e_kern.py): bindings for the E10 phase-3 LUT VQ-KV partial kernels
// (experiments/32_kernel_tiling/csrc/lut_rag_p3.cuh, unmodified; variants in vqattn_lut_p3_vars.h, instantiated in
// vqattn_lut_p3_k{1,2}.cu). Same inputs as csrc/vqattn_lut_rag.cu (lut_rag), but the in-block reduced partial
// contract of phase 1: ONE partial slot per split, written at (bh * SPT + OFF + sp) * BM (original: 16 per split at
// OFF + sp * 16 + warp). The e2e_kern wrapper merges with a combine that reads only the OFF + splits used slots.
// Always uses the lazy online max (exact up to fp32, p <= 2^8); fp16 V codewords (VM 1/2).
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <string>
#include <tuple>

#include "lut_rag_p3.cuh"
#include "vqattn_lut_p3_vars.h"

namespace e10p3 {
#define MV_EXT(K, W, V, R, H) extern template cudaError_t run_t<K, W, V, R, H, 0>(const RunArgs&); \
  extern template std::vector<int64_t> occ_t<K, W, V, R, H, 0>();
MV_P3_VARS_K1(MV_EXT)
MV_P3_VARS_K2(MV_EXT)
}  // namespace e10p3
using namespace e10p3;

namespace {
struct VEnt {
  const char* name;
  int f[5];
  cudaError_t (*run)(const RunArgs&);
  std::vector<int64_t> (*occ)();
};
#define MV_ENT(K, W, V, R, H) \
  {"k" #K "_w" #W "_v" #V "_r" #R "_h" #H, {K, W, V, R, H}, &run_t<K, W, V, R, H, 0>, &occ_t<K, W, V, R, H, 0>},
const VEnt TABLE[] = {MV_P3_VARS_K1(MV_ENT) MV_P3_VARS_K2(MV_ENT)};
constexpr int NV = sizeof(TABLE) / sizeof(TABLE[0]);
}  // namespace

std::vector<std::tuple<std::string, std::vector<int64_t>>> variants() {
  std::vector<std::tuple<std::string, std::vector<int64_t>>> r;
  for (int i = 0; i < NV; ++i) r.emplace_back(TABLE[i].name, std::vector<int64_t>(TABLE[i].f, TABLE[i].f + 5));
  return r;
}

// [blocks/SM on the current device (0 = not launchable, e.g. smem), dyn smem, regs, local bytes, ...] (occ_t)
std::vector<int64_t> occupancy(int64_t vid) {
  TORCH_CHECK(vid >= 0 && vid < NV);
  return TABLE[vid].occ();
}

void lut_p3(int64_t vid, torch::Tensor qkv, std::vector<torch::Tensor> kpl, std::vector<torch::Tensor> vpl,
            std::vector<torch::Tensor> kcb, std::vector<torch::Tensor> vcb, torch::Tensor invf, torch::Tensor OP,
            torch::Tensor MP, torch::Tensor LP, torch::Tensor LO, torch::Tensor HI, torch::Tensor QPOS,
            torch::Tensor SDK, torch::Tensor MUK, torch::Tensor SDV, torch::Tensor MUV, int64_t chunk, int64_t splits,
            int64_t B, int64_t Hkv, int64_t BM, int64_t SPT, int64_t OFF, double qk_scale) {
  TORCH_CHECK(vid >= 0 && vid < NV, "vid");
  const int kst = TABLE[vid].f[0];
  TORCH_CHECK((int)kpl.size() >= kst && (int)vpl.size() >= kst && (int)kcb.size() >= kst && (int)vcb.size() >= kst,
              "kst");
  TORCH_CHECK(BM >= GM && OFF + splits <= SPT, "partial slots");
  RunArgs a{};
  for (int s = 0; s < kst; ++s) {
    a.P.kc[s] = kpl[s].data_ptr<uint8_t>();
    a.P.vc[s] = vpl[s].data_ptr<uint8_t>();
    TORCH_CHECK(kpl[s].stride(3) == 1 && kpl[s].size(3) == 16 && kpl[s].stride(2) == 16, "plane layout");
    TORCH_CHECK(kpl[s].stride(0) == vpl[s].stride(0) && kpl[s].stride(1) == vpl[s].stride(1), "K/V strides");
    TORCH_CHECK(s == 0 || (kpl[s].stride(0) == kpl[0].stride(0) && kpl[s].stride(1) == kpl[0].stride(1)), "stages");
    a.P.kcb[s] = reinterpret_cast<const __nv_bfloat16*>(kcb[s].data_ptr());
    a.P.vcb[s] = reinterpret_cast<const __nv_bfloat16*>(vcb[s].data_ptr());
    TORCH_CHECK(kcb[s].stride(2) == 1 && kcb[s].stride(1) == DS && kcb[s].stride(0) == vcb[s].stride(0), "cb");
  }
  a.P.sb = kpl[0].stride(0);
  a.P.sh = kpl[0].stride(1);
  a.P.cbh = kcb[0].stride(0);
  TORCH_CHECK(LO.dtype() == torch::kInt32 && HI.dtype() == torch::kInt32 && QPOS.dtype() == torch::kInt32, "int32");
  a.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  a.invf = invf.data_ptr<float>();
  a.OP = OP.data_ptr<float>();
  a.MP = MP.data_ptr<float>();
  a.LP = LP.data_ptr<float>();
  a.LO = LO.data_ptr<int>();
  a.HI = HI.data_ptr<int>();
  a.QPOS = QPOS.data_ptr<int>();
  a.SDK = SDK.data_ptr<float>();
  a.MUK = MUK.data_ptr<float>();
  a.SDV = SDV.data_ptr<float>();
  a.MUV = MUV.data_ptr<float>();
  a.chunk = chunk;
  a.splits = splits;
  a.B = B;
  a.Hkv = Hkv;
  a.BM = BM;
  a.SPT = SPT;
  a.OFF = OFF;
  a.sq_r = qkv.stride(0);
  a.qk_scale = qk_scale;
  a.stream = c10::cuda::getCurrentCUDAStream();
  const cudaError_t e = TABLE[vid].run(a);
  TORCH_CHECK(e == cudaSuccess, "launch ", TABLE[vid].name, ": ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("variants", &variants);
  m.def("occupancy", &occupancy);
  m.def("lut_p3", &lut_p3);
}
