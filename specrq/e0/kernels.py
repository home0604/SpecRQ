"""E0 kernel wrappers: build random weights for each (kernel, config) and return a callable y = f(x).

Every builder returns a Case with
  fn(x)       -> y (M, N) fp16, the kernel under test
  ref(x)      -> y_ref (M, N) fp32 from the dequantized weight (correctness reference)
  bytes_w     -> bytes the kernel must read for the weight representation (container sizes, incl. codebooks/scales)
  info        -> dict of config details
Weights are random (timing only). Shapes are (N_out, K_in).

Sources: AQLM inference_lib @ /data/lsh/src/AQLM (e79a896) loaded from source via sys.path (not pip-installed);
Marlin @ /data/lsh/src/marlin (1f25790), pip -e into env nestvq.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

AQLM_SRC = "/data/lsh/src/AQLM/inference_lib/src"

# JIT build settings for the AQLM CUDA extension (system g++ (12) has no cc1plus; use g++-11).
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
os.environ.setdefault("CC", "/usr/bin/gcc-11")
os.environ.setdefault("CXX", "/usr/bin/g++-11")
os.environ.setdefault("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/lsh/torch_extensions")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/lsh/triton_cache")
if "/usr/local/cuda-12.1/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/usr/local/cuda-12.1/bin:" + os.environ.get("PATH", "")
if AQLM_SRC not in sys.path:
    sys.path.insert(0, AQLM_SRC)

DTYPE = torch.float16  # activation/weight dtype for the run; bench --dtype sets it. Marlin/AQLM need fp16.
D = 8  # in_group_size (vector dim) for every VQ config here

# name -> (num_codebooks L, bits per code b)
VQ_CONFIGS = {
    "C1": (5, 8),    # 5 codebooks x 8 bit, d=8 -> 5 bpw (nested prefixes k=1..5 -> 1..5 bpw)
    "C2": (2, 12),   # 2 codebooks x 12 bit, d=8 -> 3 bpw
    "A1x16": (1, 16),  # AQLM native 2 bpw
    "A2x8": (2, 8),    # AQLM native 2 bpw == C1 prefix k=2 layout
    "A1x8": (1, 8),    # AQLM native 1 bpw == C1 prefix k=1 layout
}


@dataclass
class Case:
    fn: Callable[[torch.Tensor], torch.Tensor]
    ref: Callable[[torch.Tensor], torch.Tensor]
    bytes_w: int
    info: dict = field(default_factory=dict)
    max_m: int | None = None  # largest M the kernel accepts (None = any)


class Unsupported(Exception):
    pass


# ----------------------------------------------------------------------------------------------- fp16
def build_fp16(N, K, dev="cuda"):
    """Dense cuBLAS baseline in DTYPE (name kept as 'fp16' for continuity; rows carry 'dtype')."""
    W = (torch.randn(N, K, device=dev) / K ** 0.5).to(DTYPE)
    return Case(fn=lambda x: F.linear(x, W), ref=lambda x: x.float() @ W.float().t(),
                bytes_w=2 * N * K, info={"bpw": 16})


# ----------------------------------------------------------------------------------------------- marlin
def _need_fp16(name):
    if DTYPE != torch.float16:
        raise Unsupported(f"{name} supports fp16 only (run dtype {DTYPE})")


def build_marlin(N, K, dev="cuda", groupsize=128):
    _need_fp16("Marlin")
    import marlin  # noqa
    if K % 128 or N % 128:
        raise Unsupported(f"Marlin needs N%128==0 and K%128==0 for M<=16 (got N={N}, K={K})")
    W = torch.randn(N, K, device=dev) / K ** 0.5
    Wg = W.view(N, K // groupsize, groupsize)
    s = (Wg.abs().amax(-1) / 7).clamp_min(1e-8)  # (N, K/g); symmetric 4-bit as in marlin test.py
    q = torch.clamp(torch.round(Wg / s[..., None]) + 8, 0, 15)
    Wdq = ((q - 8) * s[..., None]).view(N, K).half()
    lin = nn.Linear(K, N, bias=False).to(dev).half()
    lin.weight.data = Wdq.clone()
    # marlin.Layer.__init__ insists on N%256; the kernel itself only needs N%thread_n (=128 for M<=16).
    layer = marlin.Layer.__new__(marlin.Layer)
    nn.Module.__init__(layer)
    layer.k, layer.n, layer.groupsize = K, N, groupsize
    layer.register_buffer("B", torch.empty((K // 16, N * 16 // 8), dtype=torch.int, device=dev))
    layer.register_buffer("s", torch.empty((K // groupsize, N), dtype=torch.half, device=dev))
    layer.register_buffer("workspace", torch.zeros(N // 128 * 16, dtype=torch.int, device=dev), persistent=False)
    layer.pack(lin, s.half())
    del lin
    Wref = Wdq.float()
    return Case(fn=layer.forward, ref=lambda x: x.float() @ Wref.t(),
                bytes_w=N * K // 2 + 2 * N * (K // groupsize), info={"bpw": 4, "groupsize": groupsize})


# ----------------------------------------------------------------------------------------------- AQLM
def _rand_vq(N, K, L, b, dev):
    from aqlm.utils import pack_int_data
    codes_u = torch.randint(0, 2 ** b, (N, K // D, L), device=dev, dtype=torch.int64)
    codes = pack_int_data(codes_u.clone(), b)  # int8 for b<=8, int16 for b<=16 (AQLM storage)
    codebooks = (torch.randn(L, 2 ** b, 1, D, device=dev) * (1.0 / (K * L) ** 0.5) * 3).half()
    scales = (torch.rand(N, 1, 1, 1, device=dev) + 0.5).half()
    return codes, codebooks, scales


def _vq_bytes(N, K, L, b, codes):
    return codes.numel() * codes.element_size() + L * (2 ** b) * D * 2 + 2 * N


def _vq_ref(codes, codebooks, scales):
    from aqlm.utils import _dequantize_weight, unpack_int_data
    nbits = codebooks.shape[1].bit_length() - 1
    W = _dequantize_weight(unpack_int_data(codes, nbits), codebooks.float(), scales.float())
    return lambda x: x.float() @ W.t()


def _cuda_op(L, b, dequant: bool):
    import aqlm.inference_kernels.cuda_kernel  # noqa: F401  (JIT-builds codebook_cuda, registers ops)
    suffix = "_dequant" if dequant else ""
    table = {(1, 16): "code1x16_matmat", (2, 8): "code2x8_matmat", (1, 8): "code1x8_matmat"}
    if (L, b) not in table:
        raise Unsupported(f"AQLM CUDA kernels exist only for 1x16, 2x8, 1x8 (CodeKx8 instantiated for K=1,2); "
                          f"got {L}x{b}")
    return getattr(torch.ops.aqlm, table[(L, b)] + suffix)


def build_aqlm_cuda(N, K, cfg, dev="cuda", dequant=False):
    _need_fp16("AQLM CUDA (as benchmarked)")
    L, b = VQ_CONFIGS[cfg]
    op = _cuda_op(L, b, dequant)
    codes, cbs, sc = _rand_vq(N, K, L, b, dev)
    return Case(fn=lambda x: op(x, codes, cbs, sc, None), ref=_vq_ref(codes, cbs, sc),
                bytes_w=_vq_bytes(N, K, L, b, codes),
                info={"L": L, "b": b, "d": D, "bpw": L * b / D, "code_dtype": str(codes.dtype),
                      "path": "AQLM CUDA dequant-kernel + matmul" if dequant else "AQLM CUDA matvec (row loop)"})


def build_aqlm_triton(N, K, cfg, dev="cuda"):
    _need_fp16("AQLM Triton (as benchmarked)")
    from aqlm.inference_kernels.triton_kernel import triton_matmul
    L, b = VQ_CONFIGS[cfg]
    if L & (L - 1):
        raise Unsupported(f"AQLM Triton kernel uses tl.arange(0, num_codebooks): needs power-of-2 codebooks, got {L}")
    codes, cbs, sc = _rand_vq(N, K, L, b, dev)
    return Case(fn=lambda x: triton_matmul(x, codes, cbs, sc, None), ref=_vq_ref(codes, cbs, sc),
                bytes_w=_vq_bytes(N, K, L, b, codes),
                info={"L": L, "b": b, "d": D, "bpw": L * b / D, "code_dtype": str(codes.dtype),
                      "path": "AQLM Triton gemv (row loop)"})


def build_dequant(N, K, cfg, dev="cuda"):
    _need_fp16("AQLM dequantize_gemm (as benchmarked)")
    from aqlm.inference_kernels.dequantization import dequantize_gemm
    L, b = VQ_CONFIGS[cfg]
    codes, cbs, sc = _rand_vq(N, K, L, b, dev)
    return Case(fn=lambda x: dequantize_gemm(x, codes, cbs, sc, None), ref=_vq_ref(codes, cbs, sc),
                bytes_w=_vq_bytes(N, K, L, b, codes),
                info={"L": L, "b": b, "d": D, "bpw": L * b / D, "code_dtype": str(codes.dtype),
                      "path": "AQLM dequantize_gemm (embedding_bag -> F.linear)"})


# ----------------------------------------------------------------------------------------------- nestkern (ours)
def build_nest_gemv(N, K, cfg, dev="cuda"):
    """Our planar nested-VQ GEMV (path (a), M=1). cfg 'C1k<k>': C1 codes (L=5, 8 bit, d=8), prefix k."""
    from specrq.nestkern.ops import nest_gemv
    from specrq.nestkern.pack import PlanarVQ, kernel_codes, ref_matmul
    L, b = VQ_CONFIGS["C1"]
    # cfg grammar: 'C1k<k>[r<repl>][g]'; r = codebook smem replication, g = force generic (no register-x) path
    body = cfg[3:]
    variant = 1 if body.endswith("g") else 0
    body = body.rstrip("g")
    k, _, repl = body.partition("r")
    k, repl = int(k), int(repl or 0)
    if K % 32:
        raise Unsupported(f"nest_gemv needs K%32==0 (got {K})")
    codes = torch.randint(0, 256, (L, N, K // D), device=dev, dtype=torch.uint8)
    cbs = (torch.randn(L, 256, D, device=dev) * (3.0 / (K * L) ** 0.5)).to(DTYPE)
    sc = (torch.rand(N, device=dev) + 0.5).to(DTYPE)
    p = PlanarVQ(codes, cbs, sc, N, K)
    kc = kernel_codes(codes)
    return Case(fn=lambda x: nest_gemv(x, kc, cbs, sc, k, repl, variant), ref=lambda x: ref_matmul(x, p, k), bytes_w=p.nbytes(k),
                info={"L": L, "b": b, "d": D, "k": k, "bpw": k * b / D, "code_dtype": "uint8 planar",
                      "path": "nestkern planar GEMV (a), M=1", "repl": repl,
                      "variant": "generic" if variant == 1 or repl or K > 4096 else "xreg"}, max_m=1)


def build_nest_mma(N, K, cfg, dev="cuda"):
    """Our tensor-core nested-VQ GEMM (path (a) + mma.m16n8k16), 1 <= M <= 16. cfg 'C1k<k>[s<splits>][c<cfg>]'."""
    from specrq.nestkern.ops import nest_mma
    from specrq.nestkern.pack import PlanarVQ, mma_codes, ref_matmul
    L, b = VQ_CONFIGS["C1"]
    body, _, tc = cfg[3:].partition("c")
    k, _, sp = body.partition("s")
    k, sp, tc = int(k), int(sp or 0), int(tc or 0)
    if K % 64 or N % 16:
        raise Unsupported(f"nest_mma needs K%64==0, N%16==0 (got N={N}, K={K})")
    codes = torch.randint(0, 256, (L, N, K // D), device=dev, dtype=torch.uint8)
    cbs = (torch.randn(L, 256, D, device=dev) * (3.0 / (K * L) ** 0.5)).to(DTYPE)
    sc = (torch.rand(N, device=dev) + 0.5).to(DTYPE)
    p = PlanarVQ(codes, cbs, sc, N, K)
    mc = mma_codes(codes)
    return Case(fn=lambda x: nest_mma(x, mc, cbs, sc, k, sp, tc), ref=lambda x: ref_matmul(x, p, k), bytes_w=p.nbytes(k),
                info={"L": L, "b": b, "d": D, "k": k, "bpw": k * b / D, "code_dtype": "uint8 mma_tile_v1",
                      "path": "nestkern planar mma (a), M<=16", "splits": sp, "tile_cfg": tc}, max_m=16)


# kernel name -> (builder, list of configs or [None])
KERNELS = {
    "fp16": (build_fp16, [None]),
    "marlin": (build_marlin, [None]),
    "aqlm_cuda": (build_aqlm_cuda, ["A1x16", "A2x8", "A1x8", "C1", "C2"]),
    "aqlm_cuda_dq": (lambda N, K, cfg: build_aqlm_cuda(N, K, cfg, dequant=True), ["A1x16", "A2x8", "A1x8", "C1", "C2"]),
    "aqlm_triton": (build_aqlm_triton, ["A2x8", "C2", "C1"]),
    "dequant": (build_dequant, ["C1", "C2"]),
    "nest_gemv": (build_nest_gemv, ["C1k1", "C1k2", "C1k3", "C1k4", "C1k5",
                                    "C1k1g", "C1k2g", "C1k5g", "C1k3r4", "C1k5r2"]),
    "nest_mma": (build_nest_mma, ["C1k1", "C1k2", "C1k3", "C1k4", "C1k5"]),
}


def build(kernel, N, K, cfg):
    fn, _ = KERNELS[kernel]
    return fn(N, K) if cfg is None else fn(N, K, cfg)
