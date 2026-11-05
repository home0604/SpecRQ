"""M3 correctness: nest_mma vs fp32 prefix-k reference, M in {1..16}, random + real ckpt layer.

Floor (unavoidable rounding of a dtype tensor-core path): the dequantized weight sum is rounded once to dtype and
the output is rounded to dtype: floor = rel(((x @ round(W_sum)^T) * scale).to(dtype), y_fp32).
Pass if rel_err < 1e-3 or rel_err < 1.5 * floor (same rule as M2 with the weight rounding added).
Usage: python -m specrq.nestkern.test_mma [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.nestkern.ops import nest_mma
from specrq.nestkern.pack import mma_codes, pack_planar, ref_matmul
from specrq.nestkern.test_gemv import SHAPES, rand_planar

REPO = Path(__file__).resolve().parents[2]
MS = [1, 2, 3, 4, 5, 8, 16]


def floor_ref(x, p, k, dtype):
    G = p.K // 8
    acc = torch.zeros(p.N, G, 8, device=x.device)
    for l in range(k):
        acc += p.codebooks[l].float()[p.codes[l].long()]
    w = acc.to(dtype).float().reshape(p.N, p.K)
    return ((x.float() @ w.t()) * p.scales.float()[None]).to(dtype)


def relerr(y, yr):
    return ((y.float() - yr).norm() / yr.norm()).item()


def check(p, dtype, src, rows, ms=MS, splits=(0,)):
    mc = mma_codes(p.codes)
    for M in ms:
        x = torch.randn(M, p.K, device="cuda").to(dtype)
        for k in range(1, p.L + 1):
            yr = ref_matmul(x, p, k)
            fl = relerr(floor_ref(x, p, k, dtype), yr)
            for sp in splits:
                e = relerr(nest_mma(x, mc, p.codebooks, p.scales, k, sp), yr)
                rows.append({"src": src, "dtype": str(dtype), "N": p.N, "K": p.K, "M": M, "k": k, "splits": sp,
                             "rel_err": e, "floor": fl, "pass": e < 1e-3 or e < 1.5 * fl})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rows = []
    for dtype in (torch.bfloat16, torch.float16):
        for N, K in SHAPES:
            check(rand_planar(N, K, dtype=dtype), dtype, "random", rows, splits=(0, 1, 3))
    sys.path.insert(0, str(REPO / "specrq" / "dbd"))
    layer = torch.load(f"{a.ckpt}/{a.layer}.pth", map_location="cpu", weights_only=False)
    for name, mod in layer.named_modules():
        if hasattr(mod, "quantized_weight"):
            q = mod.quantized_weight
            for dtype in (torch.bfloat16, torch.float16):
                p = pack_planar(q.get_codes(), q.get_codebooks().detach(), q.get_scales().detach()).to("cuda", dtype)
                check(p, dtype, f"ckpt:{a.layer}:{name}", rows)
    for dt in ("torch.bfloat16", "torch.float16"):
        sel = [r for r in rows if r["dtype"] == dt]
        print(dt, "n", len(sel), "max rel", f"{max(r['rel_err'] for r in sel):.2e}",
              "max rel/floor", f"{max(r['rel_err'] / r['floor'] for r in sel):.3f}")
    bad = [r for r in rows if not r["pass"]]
    print("PASS" if not bad else f"FAIL {len(bad)}", bad[:5])
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"setting": "M3 nest_mma correctness vs fp32 prefix-k ref",
                                                          "ckpt": a.ckpt, "layer": a.layer, "ms": MS,
                                                          "rule": "rel<1e-3 or rel<1.5*floor(weight+output rounding)"},
                                   metrics={"rows": rows, "pass": not bad}))


if __name__ == "__main__":
    main()
