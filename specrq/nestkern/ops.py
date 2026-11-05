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


_ext_mma = None
_counters = {}


def ext_mma():
    global _ext_mma
    if _ext_mma is None:
        _ext_mma = load(name="nestkern_mma", sources=[str(_CSRC / "nest_mma.cu")],
                        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _ext_mma


_ws = {}


def _cnt(device) -> torch.Tensor:
    c = _counters.get(device)
    if c is None:
        c = _counters[device] = torch.zeros(1 << 16, dtype=torch.int32, device=device)  # self-resetting
    return c


def _wsp(device, numel) -> torch.Tensor:
    """Persistent zeroed fp32 split-K workspace (the kernel re-zeroes what it uses)."""
    w = _ws.get(device)
    if w is None or w.numel() < numel:
        w = _ws[device] = torch.zeros(max(numel, 16 * 32768), dtype=torch.float32, device=device)
    return w


def nest_mma(x: torch.Tensor, mcodes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor, k: int,
             splits: int = 0, cfg: int = 0):
    """y [M, N] = x [M, K] @ W_k^T for 1 <= M <= 16 (tensor-core path). mcodes from pack.mma_codes.
    splits: split-K factor (0 = auto). cfg: tile config id (0 = auto, see csrc/nest_mma.cu)."""
    N = scales.numel()
    return ext_mma().nest_mma(x.contiguous(), mcodes, codebooks, scales, _cnt(x.device), _wsp(x.device, x.shape[0] * N),
                              N, int(k), int(splits), int(cfg))


_ext_fwht = None


def block_fwht_signed(x: torch.Tensor, signs: torch.Tensor, block_size: int) -> torch.Tensor:
    """x' = BlockFWHT(x * signs) / sqrt(B) along the last dim (== specrq.dbd.src.hadamard.rotate_input)."""
    global _ext_fwht
    if _ext_fwht is None:
        _ext_fwht = load(name="nestkern_fwht", sources=[str(_CSRC / "fwht.cu")],
                         extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    shp = x.shape
    return _ext_fwht.block_fwht_signed(x.reshape(-1, shp[-1]).contiguous(), signs.float().contiguous(),
                                       int(block_size)).reshape(shp)
