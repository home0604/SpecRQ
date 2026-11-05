"""M3 tables from E0 result files: per (shape, kernel:cfg) T(M) [us], T(M)/T(1), and ratio vs dense / Marlin at the
same M (ratio > 1 means ours is faster). Dense and Marlin are taken from the same file (same dtype); Marlin only
exists in fp16 runs.

Usage: python -m specrq.nestkern.summarize_m3 results/E0/<run>.json [--kernels nest_mma nest_gemv] [--layers ...]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--kernels", nargs="+", default=["nest_mma", "nest_gemv", "aqlm_cuda"])
    ap.add_argument("--layers", nargs="+", default=None)
    a = ap.parse_args(argv)
    rec = json.load(open(a.path))
    rows = rec["metrics"]["rows"]
    t = defaultdict(dict)
    for r in rows:
        if a.layers and r["layer"] not in a.layers:
            continue
        t[(r["model"], r["layer"], r["N"], r["K"], r["kernel"], str(r["cfg"]))][r["M"]] = r["t_us"]
    ms = sorted({r["M"] for r in rows})
    print(f"file {a.path}  mode {rec['config'].get('mode')}/{rec['config'].get('flush_kind')} "
          f"dtype {rec['config'].get('dtype', 'fp16')}")
    hdr = f"{'shape':26} {'kernel:cfg':16} " + " ".join(f"{'M' + str(m):>6}" for m in ms)
    for key in sorted(t):
        model, layer, N, K, kern, cfg = key
        if kern not in a.kernels:
            continue
        d = t[key]
        dense = t.get((model, layer, N, K, "fp16", "None"), {})
        marlin = t.get((model, layer, N, K, "marlin", "None"), {})
        tag = f"{model[:5]}/{layer} {N}x{K}"
        print(hdr if kern == a.kernels[0] and cfg.endswith("k1") else "", end="")
        print(f"\n{tag:26} {kern + ':' + cfg:16} " + " ".join(f"{d.get(m, float('nan')):6.1f}" for m in ms) + "  T[us]")
        print(f"{'':26} {'':16} " + " ".join(f"{d.get(m, float('nan')) / d[1]:6.2f}" for m in ms) + "  T(M)/T(1)"
              if 1 in d else "")
        if dense:
            print(f"{'':26} {'':16} " + " ".join(f"{dense.get(m, float('nan')) / d.get(m, float('nan')):6.2f}"
                                                 for m in ms) + "  dense/ours")
        if marlin:
            print(f"{'':26} {'':16} " + " ".join(f"{marlin.get(m, float('nan')) / d.get(m, float('nan')):6.2f}"
                                                 for m in ms) + "  marlin/ours")
    for base in ("fp16", "marlin"):
        for key in sorted(t):
            if key[4] == base:
                d = t[key]
                if 1 in d:
                    print(f"{key[0][:5]}/{key[1]:14} {base:7} T1={d[1]:6.1f}us  T(M)/T(1): " +
                          " ".join(f"{d.get(m, float('nan')) / d[1]:5.2f}" for m in ms))


if __name__ == "__main__":
    main()
