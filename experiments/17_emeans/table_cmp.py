"""Summarize init_compare.py jsonl files: per (cfg, variant) aggregates over the layer x linear grid.

Columns: init_s (sum over linears), init loss_rel / final loss_rel (geomean ratio vs torch:0 per linear, and mean),
epochs (mean), aq time (sum, incl. init). 'ratio' = geometric mean over linears of variant/torch:0.
Usage: python experiments/17_emeans/table_cmp.py FILE.jsonl [...] [--per_linear]
"""
import json
import math
import sys
from collections import defaultdict

files = [f for f in sys.argv[1:] if not f.startswith("--")]
per_linear = "--per_linear" in sys.argv
rows = [json.loads(l) for f in files for l in open(f)]
by = defaultdict(dict)
for r in rows:
    by[(r["cfg"], r["layer"], r["name"])][f'{r["variant"]}:{r["seed"]}'] = r


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


cfgs = sorted({k[0] for k in by})
for cfg in cfgs:
    keys = [k for k in by if k[0] == cfg]
    variants = sorted({v for k in keys for v in by[k]}, key=lambda v: (not v.startswith("torch"), v))
    print(f"\n== {cfg}: {len(keys)} linears ==")
    print(f"{'variant':12s} {'n':>3s} {'init_s':>7s} {'wmse_r':>7s} {'init_r':>7s} {'final_r':>8s} {'final_rel':>9s}"
          f" {'epochs':>6s} {'total_s':>8s} {'win%':>5s}")
    for v in variants:
        ks = [k for k in keys if v in by[k] and "torch:0" in by[k]]
        rs = [by[k][v] for k in ks]
        base = [by[k]["torch:0"] for k in ks]
        wr = gm([r["init_wmse_rel"] / b["init_wmse_rel"] for r, b in zip(rs, base)])
        ir = gm([r["init_loss"] / b["init_loss"] for r, b in zip(rs, base)])
        fr = gm([r["final_loss"] / b["final_loss"] for r, b in zip(rs, base)])
        win = 100 * sum(r["final_loss"] < b["final_loss"] for r, b in zip(rs, base)) / len(rs)
        print(f"{v:12s} {len(rs):3d} {sum(r['init_s'] for r in rs):7.1f} {wr:7.4f} {ir:7.4f} {fr:8.4f}"
              f" {sum(r['final_loss_rel'] for r in rs) / len(rs):9.5f} {sum(r['epochs'] for r in rs) / len(rs):6.2f}"
              f" {sum(r['total_s'] for r in rs):8.1f} {win:5.0f}")
    if per_linear:
        for k in sorted(keys):
            print(" ", k[1], k[2], " ".join(f"{v}={by[k][v]['final_loss_rel']:.5f}/{by[k][v]['epochs']}"
                                            for v in variants if v in by[k]))
