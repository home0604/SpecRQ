// Phase-2 copy of the phase-0 wrapper (lut_rag_occ.cu), adapted to commit ce9668d (LAZY template parameter and
// `lazy` argument of the original kernel; 'orig' = lazy 0, the original online-max path). Phase-0 note: the unmodified E7 LUT kernel (experiments/26_magicvq/csrc/vqattn_lut_rag.cu, included as-is)
// plus occupancy / function-attribute queries. The included file's PYBIND11_MODULE is renamed to an unused static
// function so that this translation unit can define its own module.
#include <torch/extension.h>
#pragma push_macro("PYBIND11_MODULE")
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, var) [[maybe_unused]] static void vq_unused_bind_(pybind11::module_& var)
#include "vqattn_lut_rag.cu"
#pragma pop_macro("PYBIND11_MODULE")

namespace {
template <int KST>
std::vector<int64_t> occ_impl() {
  using L = Lay<KST>;
  auto k = lut_rag_kernel<KST, false>;
  TORCH_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, L::total) == cudaSuccess, "smem");
  int nb = 0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, NW * 32, L::total) == cudaSuccess, "occ");
  cudaFuncAttributes a;
  TORCH_CHECK(cudaFuncGetAttributes(&a, k) == cudaSuccess, "attr");
  return {nb, L::total, a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, a.maxThreadsPerBlock,
          NW * 32, L::vcb_off(KST), L::lut_off(KST) - L::lut_off(0), L::warp_off, L::tail, L::warp_bytes};
}
}  // namespace

// returns [blocks_per_SM, dyn_smem_bytes, regs/thread, local_bytes, static_smem, max_threads_per_block, threads,
//          vcb_bytes, lut_bytes, warp_off, tail_bytes, warp_bytes]
std::vector<int64_t> occupancy(int64_t kst) { return kst == 1 ? occ_impl<1>() : occ_impl<2>(); }

std::vector<int64_t> device_limits() {
  int dev = 0;
  cudaGetDevice(&dev);
  auto q = [&](cudaDeviceAttr a) { int v = 0; cudaDeviceGetAttribute(&v, a, dev); return (int64_t)v; };
  return {q(cudaDevAttrMultiProcessorCount), q(cudaDevAttrMaxSharedMemoryPerMultiprocessor),
          q(cudaDevAttrMaxSharedMemoryPerBlockOptin), q(cudaDevAttrReservedSharedMemoryPerBlock),
          q(cudaDevAttrMaxRegistersPerMultiprocessor), q(cudaDevAttrMaxThreadsPerMultiProcessor),
          q(cudaDevAttrMaxBlocksPerMultiprocessor), q(cudaDevAttrL2CacheSize), q(cudaDevAttrComputeCapabilityMajor),
          q(cudaDevAttrComputeCapabilityMinor)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lut_rag", [](torch::Tensor qkv, std::vector<torch::Tensor> kpl, std::vector<torch::Tensor> vpl,
                      std::vector<torch::Tensor> kcb, std::vector<torch::Tensor> vcb, torch::Tensor invf,
                      torch::Tensor OP, torch::Tensor MP, torch::Tensor LP, torch::Tensor LO, torch::Tensor HI,
                      torch::Tensor QPOS, torch::Tensor SDK, torch::Tensor MUK, torch::Tensor SDV, torch::Tensor MUV,
                      int64_t chunk, int64_t splits, int64_t B, int64_t Hkv, int64_t BM, int64_t SPT, int64_t OFF,
                      int64_t kst, double qk_scale) {
    lut_rag(qkv, kpl, vpl, kcb, vcb, invf, OP, MP, LP, LO, HI, QPOS, SDK, MUK, SDV, MUV, chunk, splits, B, Hkv, BM, SPT,
            OFF, kst, qk_scale, 0);
  });
  m.def("occupancy", &occupancy);
  m.def("device_limits", &device_limits);
}
