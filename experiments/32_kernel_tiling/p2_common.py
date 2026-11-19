"""E10 phase 2 helpers: per-call plan (splits / stream-K block count), workspaces, and one call of a phase-2 variant.

Plan for a variant at (lengths per bh, knob):
  DEC 0: knob = chunk -> splits = ceil(n_max / chunk), grid (BH, splits), SPT = splits (1 slot per split, in-block red.)
  DEC 1: knob = P (target block count) -> chunk = ceil32(ceil(N / P)) over the flattened N = sum(len) tokens,
         nblk = ceil(N / chunk), SPT = max over bh of the blocks covering it (unused slots stay M=-inf, L=0, O=0).
"""
from __future__ import annotations

from pathlib import Path

import torch

D, G = 128, 4
LUT_STAGE_H = 16 * 256 * 4


def ceil_div(a, b):
    return -(-a // b)


def plan(dec, lens, knob):
    """lens: python list of per-bh token counts. Returns dict(chunk, splits, nblk, spt, cu(list))."""
    cu = [0]
    for x in lens:
        cu.append(cu[-1] + x)
    if dec == 0:
        chunk = knob
        splits = ceil_div(max(lens), chunk)
        return dict(chunk=chunk, splits=splits, nblk=len(lens) * splits, spt=splits, cu=cu)
    N = cu[-1]
    chunk = ceil_div(ceil_div(N, knob), 32) * 32
    nblk = ceil_div(N, chunk)
    spt = max((cu[i + 1] - 1) // chunk - cu[i] // chunk + 1 for i in range(len(lens)) if lens[i] > 0)
    return dict(chunk=chunk, splits=0, nblk=nblk, spt=spt, cu=cu)


def workspace(BH, spt, BM=16, dev="cuda"):
    n = BH * spt * BM
    return (torch.zeros(n, D, dtype=torch.float32, device=dev), torch.full((n,), float("-inf"), device=dev),
            torch.zeros(n, device=dev), BM)


class Bufs:
    """Per-(B, Hkv, kst) extra buffers: LUT workspace, q.muk, fused-combine counters, output."""

    def __init__(self, B, Hkv, kst, dev="cuda"):
        BH = B * Hkv
        self.lutg = torch.empty(BH * kst * LUT_STAGE_H, dtype=torch.half, device=dev)
        self.qmug = torch.empty(BH * G, dtype=torch.float32, device=dev)
        self.cnt = torch.zeros(BH, dtype=torch.int32, device=dev)
        self.out = torch.empty(B, Hkv * G * D, dtype=torch.bfloat16, device=dev)


def call(E, name, qkv, planes, cb, inv, ws, lo, hi, qpos, sd, mu, pl, cu_t, bufs, B, Hkv, kst, d_scale):
    """planes: [2, kst, B, Hkv, n, 16] uint8; cb [2, Hkv, kst, 256, 8] bf16. Partial (+ fused combine) only."""
    E.run(name, qkv, [planes[0, s] for s in range(kst)], [planes[1, s] for s in range(kst)],
          [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv, ws[0], ws[1], ws[2], lo, hi, qpos,
          sd[0], mu[0], sd[1], mu[1], bufs.lutg, bufs.qmug, cu_t, bufs.out, bufs.cnt, pl["chunk"], pl["splits"],
          pl["nblk"], B, Hkv, ws[3], pl["spt"], 0, d_scale)


def build_orig():
    """Original kernel (experiments/26_magicvq/csrc/vqattn_lut_rag.cu, lazy=0 path) via csrc/lut_rag_occ_p2.cu; same
    interface as bench_p0.build_ext() (whose wrapper no longer compiles after commit ce9668d)."""
    import os
    from torch.utils.cpp_extension import load
    import p2_ext as PX
    arch = PX._env()
    here = Path(__file__).resolve().parent
    return load(name=f"e10_lut_rag_occ_p2_sm{arch}", sources=[str(here / "csrc" / "lut_rag_occ_p2.cu")],
                extra_include_paths=[str(here.parent / "26_magicvq" / "csrc")],
                extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-Xptxas=-v"], verbose=False)
