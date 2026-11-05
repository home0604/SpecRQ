"""Print E0 tables from a result JSON: T(M)/T(1) per kernel and speed vs fp16 at M=1 / M=4 / M=8.

Usage: python -m specrq.e0.summarize results/E0/<run>.json [--layers q_o gate_up ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--layers", nargs="+", default=None)
    a = ap.parse_args(argv)
    rec = json.load(open(a.path))
    rows = rec["metrics"]["rows"]
    by = defaultdict(dict)
    for r in rows:
        if a.layers and r["layer"] not in a.layers:
            continue
        by[(r["model"], r["layer"], r["N"], r["K"], r["kernel"], r["cfg"])][r["M"]] = r
    ms = sorted({r["M"] for r in rows})
    hdr = f"{'model/layer':28} {'kernel:cfg':20} {'T1 us':>7} {'GB/s':>6} {'x fp16':>6} | " + \
          " ".join(f"{'T'+str(m)+'/T1':>7}" for m in ms[1:]) + f" | {'maxrel':>7}"
    print(hdr)
    last = None
    for key, d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][4], str(kv[0][5]))):
        model, layer, N, K, kern, cfg = key
        if 1 not in d:
            continue
        if (model, layer) != last:
            print("-" * len(hdr))
            last = (model, layer)
        r1 = d[1]
        sp = r1["speedup_vs_fp16"]
        print(f"{model[:5]+'/'+layer+' '+str(N)+'x'+str(K):28} {kern+':'+str(cfg):20} {r1['t_us']:7.1f} "
              f"{r1['gbps']:6.0f} {sp:6.2f} | " +
              " ".join(f"{d[m]['t_over_t1']:7.2f}" if m in d else f"{'-':>7}" for m in ms[1:]) +
              f" | {max(x['rel_err'] for x in d.values()):7.1e}")
    for u in {(u["kernel"], u["cfg"], u["reason"]) for u in rec["metrics"]["unsupported"]}:
        print("[unsupported]", *u)
    for e in rec["metrics"]["errors"]:
        print("[error]", e)


if __name__ == "__main__":
    main()
