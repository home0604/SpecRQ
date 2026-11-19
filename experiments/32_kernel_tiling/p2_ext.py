"""E10 phase 2: build the phase-2 LUT kernel (csrc/lut_rag_p2.cuh) for a list of variants.

A variant is (kst, nw, ls, dec, pipe, fuse); name = k{kst}_w{nw}_s{ls}_d{dec}_p{pipe}_f{fuse}.
One generated .cu per variant (ninja compiles them in parallel) + one binding .cu; sources in
/data/lsh/temp/e10_p2_src/<hash>/ (override with E10_SRC_DIR). Arch: TORCH_CUDA_ARCH_LIST if set (lets the build run
without a GPU), else the current device.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HDR = HERE / "csrc" / "lut_rag_p2.cuh"
FIELDS = ("kst", "nw", "ls", "dec", "pipe", "fuse")


def vname(v):
    return "k{}_w{}_s{}_d{}_p{}_f{}".format(*v)


def parse(name):
    return tuple(int(x[1:]) for x in name.split("_"))


def _env():
    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        cap = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    for k, v in (("CC", "/usr/bin/gcc-11"), ("CXX", "/usr/bin/g++-11"), ("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11"),
                 ("CUDA_HOME", "/usr/local/cuda-12.1")):
        if Path(v.split()[-1]).exists():
            os.environ.setdefault(k, v)
    os.environ.setdefault("MAX_JOBS", "16")
    return os.environ["TORCH_CUDA_ARCH_LIST"].replace(".", "")


MAIN = r"""#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "lut_rag_p2.cuh"
namespace e10p2 {
%(decl)s
}
using namespace e10p2;
struct VEnt { const char* name; int f[6]; cudaError_t (*run)(const RunArgs&); std::vector<int64_t> (*occ)(); };
static const VEnt TABLE[] = {
%(table)s
};
static const int NV = sizeof(TABLE) / sizeof(TABLE[0]);

std::vector<std::tuple<std::string, std::vector<int64_t>>> variants() {
  std::vector<std::tuple<std::string, std::vector<int64_t>>> r;
  for (int i = 0; i < NV; ++i) r.emplace_back(TABLE[i].name, std::vector<int64_t>(TABLE[i].f, TABLE[i].f + 6));
  return r;
}
std::vector<int64_t> occupancy(int64_t vid) { TORCH_CHECK(vid >= 0 && vid < NV); return TABLE[vid].occ(); }

// LUTG / QMUG / CU / OUT / CNT: pass any tensor when the variant does not use it
void run(int64_t vid, torch::Tensor qkv, std::vector<torch::Tensor> kpl, std::vector<torch::Tensor> vpl,
         std::vector<torch::Tensor> kcb, std::vector<torch::Tensor> vcb, torch::Tensor invf, torch::Tensor OP,
         torch::Tensor MP, torch::Tensor LP, torch::Tensor LO, torch::Tensor HI, torch::Tensor QPOS,
         torch::Tensor SDK, torch::Tensor MUK, torch::Tensor SDV, torch::Tensor MUV, torch::Tensor LUTG,
         torch::Tensor QMUG, torch::Tensor CU, torch::Tensor OUT, torch::Tensor CNT, int64_t chunk, int64_t splits,
         int64_t nblk, int64_t B, int64_t Hkv, int64_t BM, int64_t SPT, int64_t OFF, double qk_scale) {
  TORCH_CHECK(vid >= 0 && vid < NV);
  const int* f = TABLE[vid].f;
  const int kst = f[0];
  TORCH_CHECK((int)kpl.size() == kst && (int)vpl.size() == kst, "kst");
  RunArgs r{};
  KArgs& a = r.k;
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
  a.P.sb = kpl[0].stride(0); a.P.sh = kpl[0].stride(1); a.P.cbh = kcb[0].stride(0);
  TORCH_CHECK(LO.dtype() == torch::kInt32 && HI.dtype() == torch::kInt32 && QPOS.dtype() == torch::kInt32, "int32");
  const int64_t BHn = B * Hkv;
  if (f[2]) {
    TORCH_CHECK(LUTG.scalar_type() == torch::kHalf && LUTG.numel() >= BHn * kst * LUT_STAGE_H, "LUTG");
    TORCH_CHECK(QMUG.scalar_type() == torch::kFloat && QMUG.numel() >= BHn * GM, "QMUG");
    a.LUTG = reinterpret_cast<__half*>(LUTG.data_ptr()); a.QMUG = QMUG.data_ptr<float>();
  }
  if (f[3]) {
    TORCH_CHECK(CU.dtype() == torch::kInt32 && CU.numel() == BHn + 1, "CU");
    a.CU = CU.data_ptr<int>();
  }
  if (f[5]) {
    TORCH_CHECK(OUT.scalar_type() == torch::kBFloat16 && OUT.is_contiguous() && OUT.numel() == B * Hkv * G * D, "OUT");
    TORCH_CHECK(CNT.dtype() == torch::kInt32 && CNT.numel() >= BHn, "CNT");
    a.OUT = reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()); a.CNT = CNT.data_ptr<int>();
  }
  a.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  a.invf = invf.data_ptr<float>(); a.OP = OP.data_ptr<float>(); a.MP = MP.data_ptr<float>(); a.LP = LP.data_ptr<float>();
  a.LO = LO.data_ptr<int>(); a.HI = HI.data_ptr<int>(); a.QPOS = QPOS.data_ptr<int>();
  a.SDK = SDK.data_ptr<float>(); a.MUK = MUK.data_ptr<float>(); a.SDV = SDV.data_ptr<float>(); a.MUV = MUV.data_ptr<float>();
  a.chunk = chunk; a.splits = splits; a.nblk = nblk; a.B = B; a.Hkv = Hkv; a.BM = BM; a.SPT = SPT; a.OFF = OFF;
  a.sq_r = qkv.stride(0); a.qk_scale = qk_scale; r.stream = c10::cuda::getCurrentCUDAStream();
  cudaError_t e = TABLE[vid].run(r);
  TORCH_CHECK(e == cudaSuccess, "launch ", TABLE[vid].name, ": ", cudaGetErrorString(e));
}

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
  m.def("variants", &variants);
  m.def("occupancy", &occupancy);
  m.def("run", &run);
  m.def("device_limits", &device_limits);
}
"""

VAR = r"""#include "lut_rag_p2.cuh"
namespace e10p2 {
cudaError_t run_%(n)s(const RunArgs& a) { return run_t<%(t)s>(a); }
std::vector<int64_t> occ_%(n)s() { return occ_t<%(t)s>(); }
}
"""

OCC_KEYS = ["blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads_per_block", "threads",
            "warp_bytes", "build_regs"]


class Ext:
    """Wrapper: .names, .fields[name], .occ[name] (dict, needs a GPU), .run(name, ...)."""

    def __init__(self, mod, with_occ=True):
        self.mod = mod
        self.vid, self.fields, self.occ = {}, {}, {}
        for i, (n, f) in enumerate(mod.variants()):
            self.vid[n] = i
            self.fields[n] = dict(zip(FIELDS, f))
            if with_occ:
                self.occ[n] = dict(zip(OCC_KEYS, mod.occupancy(i)))
        self.names = list(self.vid)

    def run(self, name, *args):
        self.mod.run(self.vid[name], *args)

    def device_limits(self):
        return dict(zip(["sm_count", "smem_per_sm", "smem_per_block_optin", "reserved_smem_per_block", "regs_per_sm",
                         "threads_per_sm", "blocks_per_sm", "l2_bytes", "cc_major", "cc_minor"],
                        self.mod.device_limits()))


def build(variants, verbose=False, with_occ=True):
    arch = _env()
    variants = sorted(set(tuple(v) for v in variants))
    hdr = HDR.read_text()
    key = hashlib.sha1((repr(variants) + hdr + MAIN + VAR + arch).encode()).hexdigest()[:12]
    src = Path(os.environ.get("E10_SRC_DIR", "/data/lsh/temp/e10_p2_src")) / key
    src.mkdir(parents=True, exist_ok=True)
    files, decl, table = [], [], []
    for v in variants:
        n = vname(v)
        t = ",".join(str(x) for x in v)
        f = src / f"v_{n}.cu"
        if not f.exists():
            f.write_text(VAR % dict(n=n, t=t))
        files.append(str(f))
        decl.append(f"cudaError_t run_{n}(const RunArgs&); std::vector<int64_t> occ_{n}();")
        table.append(f'  {{"{n}", {{{t}}}, &run_{n}, &occ_{n}}},')
    m = src / "main.cu"
    m.write_text(MAIN % dict(decl="\n".join(decl), table="\n".join(table)))
    files.insert(0, str(m))
    from torch.utils.cpp_extension import load
    mod = load(name=f"e10p2_{key}_sm{arch}", sources=files, extra_include_paths=[str(HERE / "csrc")],
               extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-Xptxas=-v"], verbose=verbose)
    return Ext(mod, with_occ)
