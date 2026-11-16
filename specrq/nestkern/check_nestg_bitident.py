"""Bit-identity of the nestg kernels after the cachefit2 extension (per-plane q patterns, B = 0 bit mix) on the
pre-existing (d, B, uniform q) paths. The reference outputs were dumped with the HEAD (7ca2554) nestg build:
inputs (codes, float codebooks, row scales, x) and outputs y of mma (and gemv for M = 1) for the v1-v3 test grid
(24 uniform (d, b, q) + BF/BL mixes, L = 10, every k, M in {1,2,3,5,8,16}) and Llama-3.1-8B shapes of the
cachefit configs (every k, M in {1,2,5,8}). Here the same GenVQ is rebuilt from the saved inputs with the current
code and every output is compared with torch.equal (bitwise, bf16).
Usage: python -m specrq.nestkern.check_nestg_bitident --ref /data/lsh/temp/nestg_old/ref_outputs.pt [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.nestkern.ops import GenVQ

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/data/lsh/temp/nestg_old/ref_outputs.pt")
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    ref = torch.load(a.ref)
    rows, n_eq, n_all = [], 0, 0
    for rec in ref:
        g = GenVQ(rec["codes"].cuda(), [c.cuda() for c in rec["cbf"]], rec["sc"].cuda(), rec["bits"], rec["d"], rec["q"])
        neq = 0
        for r in rec["runs"]:
            x = r["x"].cuda()
            outs = {"mma": g.mma(x, r["k"], r["splits"])[0].cpu()}
            if "gemv" in r:
                outs["gemv"] = g.gemv(x, r["k"])[0].cpu()
            for path, y in outs.items():
                eq = torch.equal(y, r[path])
                n_all += 1
                n_eq += eq
                neq += not eq
                if not eq:
                    rows.append({"d": rec["d"], "bits": rec["bits"], "q": rec["q"], "N": rec["N"], "K": rec["K"],
                                 "k": r["k"], "M": r["M"], "path": path,
                                 "max_abs_diff": (y.float() - r[path].float()).abs().max().item()})
        print(f"d={rec['d']} bits={rec['bits']} q={rec['q']} N={rec['N']} K={rec['K']}: "
              f"{len(rec['runs'])} runs, not-equal {neq}", flush=True)
        del g
    print(f"bitwise equal {n_eq}/{n_all}", "PASS" if n_eq == n_all else "FAIL", rows[:5])
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={
            "setting": "nestg bit-identity after cachefit2 extension vs HEAD 7ca2554 build (dumped outputs)",
            "ref": a.ref, "cases": [{k: rec[k] for k in ("d", "bits", "q", "N", "K")} for rec in ref]},
            metrics={"n_outputs": n_all, "n_bitwise_equal": n_eq, "mismatches": rows, "pass": n_eq == n_all}))


if __name__ == "__main__":
    main()
