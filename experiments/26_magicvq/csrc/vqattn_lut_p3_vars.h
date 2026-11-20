// E10 phase 3 -> E7 e2e: the p3 LUT kernel variants compiled into the e2e extension (mv_lut_p3), as
// X(KST, NW, VM, VREP, HS); kernel = experiments/32_kernel_tiling/csrc/lut_rag_p3.cuh (not modified), AB = 0 only.
// Presets / fallback order: experiments/26_magicvq/e2e_kern.py (MV_LUT_VARIANT).
#pragma once
#define MV_P3_VARS_K1(X) X(1, 8, 1, 8, 0) X(1, 8, 1, 4, 0) X(1, 8, 1, 2, 1) X(1, 8, 1, 2, 0) X(1, 8, 1, 1, 1) \
  X(1, 8, 1, 1, 0) X(1, 8, 2, 1, 1)
#define MV_P3_VARS_K2(X) X(2, 8, 1, 8, 0) X(2, 8, 1, 4, 0) X(2, 8, 1, 2, 0) X(2, 8, 1, 1, 0) X(2, 8, 2, 1, 0)
