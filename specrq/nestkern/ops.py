"""JIT loader for the nestkern CUDA extension (build settings per CLAUDE.md: g++-11, CUDA 12.1, sm_89)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
os.environ.setdefault("CC", "/usr/bin/gcc-11")
os.environ.setdefault("CXX", "/usr/bin/g++-11")
os.environ.setdefault("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/lsh/torch_extensions")
if "/usr/local/cuda-12.1/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ.get("PATH", "")

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

_CSRC = Path(__file__).resolve().parent / "csrc"
_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load(name="nestkern_cuda", sources=[str(_CSRC / "nest_gemv.cu")],
                    extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _ext


def nest_gemv(x: torch.Tensor, kcodes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor, k: int,
              repl: int = 0, variant: int = 0):
    """y [1, N] = x [1, K] @ W_k^T with W_k the prefix-k dequantized planar VQ weight.
    kcodes: lane-permuted planar codes from pack.kernel_codes. repl: codebook smem replication (0 = auto).
    variant: 0 auto (register-resident x when K <= 4096 and repl auto), 1 generic loop, 2 force register-x."""
    return ext().nest_gemv(x.contiguous(), kcodes, codebooks, scales, int(k), int(repl), int(variant))
