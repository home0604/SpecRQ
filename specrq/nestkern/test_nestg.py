"""Correctness of the generalized nestg kernels (GEMV M=1 and MMA M<=16) vs the fp32 prefix-k reference, for every
(d, b, q) in {8,16} x {4,8,12,16} x {16,8,4}, L = 10, prefix k in {1, 3, 10}, plus the smem/global fallback plan.
Reference uses the dequantized codebook (quantize_codebooks); floor = output rounding (GEMV) or weight sum rounded
once to bf16 + output rounding (MMA). Pass: rel < 1e-3 or rel < 1.5 * floor (same rule as test_gemv / test_mma).
Usage: python -m specrq.nestkern.test_nestg [--save RUN]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

from specrq.nestkern.ops import GenVQ

REPO = Path(__file__).resolve().parents[2]


def ref_weight(g: GenVQ, codes, k, round_bf16):
    N, G = codes.shape[1], codes.shape[2]
    acc = torch.zeros(N, G, g.d, device=codes.device)
    for l in range(k):
        acc += g.cb_deq[l][codes[l].long()]
    if round_bf16:
        acc = acc.to(torch.bfloat16).float()
    return acc.reshape(N, -1) * g.scales.float()[:, None]


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    ap.add_argument("--L", type=int, default=10)
    ap.add_argument("--mixed-only", action="store_true")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rows = []
    shapes = [(256, 1024), (128, 896)]
    grid = [(d, b, q) for d, b, q in itertools.product((8, 16), (4, 8, 12, 16), (16, 8, 4))]
    if a.mixed_only:
        grid = []
    grid += [(8, bl, q) for bl in ([12, 12, 8, 8], [8, 8, 12, 12]) for q in (16, 8, 4)]  # BF, BL (per-plane bits)
    for d, b, q in grid:
        for N, K in shapes:
            bits = b if isinstance(b, list) else [b] * a.L
            L = len(bits)
            codes = torch.stack([torch.randint(0, 2 ** bl, (N, K // d), device="cuda") for bl in bits])
            cbf = [torch.randn(2 ** bl, d, device="cuda") * (3.0 / (K * L) ** 0.5) for bl in bits]
            g = GenVQ(codes, cbf, torch.rand(N, device="cuda") + 0.5, bits, d, q)
            bname = str(bits) if len(set(bits)) > 1 else bits[0]
            for k in sorted({1, 2, 3, L}):
                W32 = ref_weight(g, codes, k, False)
                Wbf = ref_weight(g, codes, k, True)
                for M, splits in ((1, 0), (3, 0), (8, 3), (16, 1)):
                    x = torch.randn(M, K, device="cuda").bfloat16()
                    yr = x.float() @ W32.t()
                    cases = [("mma", *g.mma(x, k, splits))]
                    if M == 1:
                        cases.append(("gemv", *g.gemv(x, k)))
                    for path, y, plan in cases:
                        fl = rel(((x.float() @ (Wbf if path == "mma" else W32).t())).to(torch.bfloat16), yr)
                        e = rel(y, yr)
                        rows.append({"d": d, "b": bname, "q": q, "L": L, "N": N, "K": K, "k": k, "M": M, "path": path,
                                     "splits_req": splits, "rel_err": e, "floor": fl,
                                     "pass": e < 1e-3 or e < 1.5 * fl, **plan})
            last = [r for r in rows if (r["d"], r["b"], r["q"], r["N"]) == (d, bname, q, N)]
            print(f"d={d:2d} b={bname} q={q:2d} N={N} K={K}: max rel/floor "
                  f"{max(r['rel_err'] / r['floor'] for r in last):.3f} pass={all(r['pass'] for r in last)} "
                  f"smem(k=L)={last[-1]['smem_need']} use_smem={last[-1]['use_smem']}", flush=True)
            del g, codes, cbf
    bad = [r for r in rows if not r["pass"]]
    print("PASS" if not bad else f"FAIL {len(bad)}", bad[:5])
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"setting": "nestg correctness (d, b, q) grid, bf16", "L": a.L,
                                                          "shapes": shapes, "ks": "1, 3, L", "Ms": [1, 3, 8, 16]},
                                   metrics={"rows": rows, "pass": not bad}))


if __name__ == "__main__":
    main()
