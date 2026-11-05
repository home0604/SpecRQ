"""I2 (Hadamard): NestLinear (block FWHT kernel + nest kernels) vs DbD QuantizedLinear forward with rotation_signs.

Per linear of one layer pickle, M in {1, 5, 16}, prefix k in {5, 3}; reference = DbD module in fp32 (rotation in
fp32 + fp32 dequant), compared with the DbD module in bf16 and the runner in bf16. Read-only on the checkpoint.
Usage: python -m specrq.infer.check_hadamard --ckpt DIR --layer 0 [--save RUN]
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

from specrq.infer.model import DBD, LINEARS, NestLinear

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(DBD))
    lay = torch.load(f"{a.ckpt}/{a.layer}.pth", map_location="cuda", weights_only=False)
    mods = dict(lay.named_modules())
    rel = lambda x, y: ((x.float() - y.float()).norm() / y.float().norm()).item()
    rows = []
    for name in LINEARS:
        qb = mods[name].to(torch.bfloat16)
        q32 = copy.deepcopy(qb).float()
        nl = NestLinear(qb, "cuda", torch.bfloat16)
        for M in (1, 5, 16):
            x = torch.randn(M, nl.K, device="cuda").bfloat16()
            for k in (5, 3):
                qb.active_num_codebooks = q32.active_num_codebooks = k
                with torch.no_grad():
                    y32 = q32(x.float())
                    r = {"module": name, "M": M, "k": k, "hb": nl.hb, "rotated": nl.signs is not None,
                         "runner_vs_fp32": rel(nl(x, k), y32), "dbd_bf16_vs_fp32": rel(qb(x), y32)}
                rows.append(r)
                print(f"{name:18} hb={r['hb']:3d} rot={r['rotated']} M={M:2d} k={k} runner {r['runner_vs_fp32']:.2e} "
                      f"| dbd bf16 {r['dbd_bf16_vs_fp32']:.2e}", flush=True)
    worst = max(r["runner_vs_fp32"] / r["dbd_bf16_vs_fp32"] for r in rows)
    print("max runner/dbd-bf16 error ratio", worst)
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E2", a.save, config={"setting": "I2 Hadamard linear check", "ckpt": a.ckpt,
                                                          "layer": a.layer}, metrics={"rows": rows, "max_ratio": worst}))


if __name__ == "__main__":
    main()
