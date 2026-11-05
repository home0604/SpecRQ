"""M=1 table across E0 result files: t_us and GB/s per (shape, kernel:cfg), one column group per file (mode).

Usage: python -m specrq.nestkern.summarize_m1 results/E0/nest_m2_gemv_m1_*.json
"""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    recs = [json.load(open(p)) for p in a.paths]
    modes = [f"{r['config']['mode']}/{r['config'].get('flush_kind', 'write')}" for r in recs]
    tab = {}
    for i, r in enumerate(recs):
        for row in r["metrics"]["rows"]:
            if row["M"] != 1:
                continue
            key = (row["model"], row["layer"], row["N"], row["K"])
            tab.setdefault(key, {}).setdefault(f"{row['kernel']}:{row['cfg']}", {})[i] = row
    print(f"{'shape':30} {'kernel:cfg':18} " + " ".join(f"{m:>22}" for m in modes))
    for key in sorted(tab):
        print("-" * (50 + 23 * len(modes)))
        fp = tab[key].get("fp16:None", {})
        for kc in sorted(tab[key]):
            cells = []
            for i in range(len(recs)):
                r = tab[key][kc].get(i)
                if r is None:
                    cells.append(f"{'-':>22}")
                    continue
                sp = fp[i]["t_us"] / r["t_us"] if i in fp else float("nan")
                cells.append(f"{r['t_us']:7.1f}us {r['gbps']:4.0f}GB/s x{sp:4.2f}")
            print(f"{key[0][:5]}/{key[1]} {key[2]}x{key[3]:<12} {kc:18} " + " ".join(cells))


if __name__ == "__main__":
    main()
