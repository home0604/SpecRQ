// explicit instantiations of the KST 2 p3 variants (separate TU: parallel build)
#include "lut_rag_p3.cuh"
#include "vqattn_lut_p3_vars.h"
namespace e10p3 {
#define MV_INST(K, W, V, R, H) template cudaError_t run_t<K, W, V, R, H, 0>(const RunArgs&); \
  template std::vector<int64_t> occ_t<K, W, V, R, H, 0>();
MV_P3_VARS_K2(MV_INST)
}  // namespace e10p3
