"""Tables from an E6 kvregime record (results/E6/<run>.json or the per-(B,ctx) jsonl written during the run).
Usage: python experiments/24_kvregime/summarize.py results/E6/llama8b_kvregime_v1.json  (or /data/lsh/logs/<run>.jsonl)
"""
from __future__ import annotations

import json
import sys


def load(path):
    if path.endswith(".jsonl"):
        return {"grid": [json.loads(l) for l in open(path)], "fit": [], "weights_paths": []}
    return json.load(open(path))["metrics"]


def pct(r, k):
    s = r["sum_components_us"]
    return 100 * r[k]["us"] / s if k in r else 0.0


def main():
    m = load(sys.argv[1])
    if m.get("fit"):
        print("## KV fit (free GiB after weights)")
        for f in m["fit"]:
            print(f"B={f['B']:>3} ctx={f['ctx']:>6} {f['format']:>6}: KV {f['kv_GiB']:6.2f} GiB / free {f['free_GiB']:5.1f} "
                  f"-> {'fits' if f['fits'] else 'NO'}")
    if m.get("weights_paths"):
        print("\n## weight path per (shape, R, kw): us for 32 layers")
        for w in m["weights_paths"]:
            t = w["us_32layers"]
            print(f"{w['shape']:>5} R={w['R']:>3} kw={w['kw']} nest={t['nest']:8.0f}"
                  + (f" deq={t['deq']:8.0f}" if "deq" in t else " " * 13) + f" -> {w['best']}")
    print("\n## per (B, ctx, format): step us (full graph) | breakdown % of component sum | attn backend, GB/s")
    for rec in m["grid"]:
        for fname, fr in rec["formats"].items():
            if "oom" in fr:
                print(f"B={rec['B']} ctx={rec['ctx']} {fname}: OOM during run")
            for name, r in fr.items():
                if not isinstance(r, dict) or "full" not in r:
                    if isinstance(r, dict) and "attention" in r:
                        print(f"B={rec['B']:>3} ctx={rec['ctx']:>6} {fname:>5} {name:>10}: attn {r['attention']['us']:8.0f} "
                              f"({r['attn_GBps']:.0f} GB/s)")
                    continue
                print(f"B={rec['B']:>3} ctx={rec['ctx']:>6} {fname:>5} {name:>10}: T={r['full']['us']:8.0f} "
                      f"[{r['full']['min']:.0f}-{r['full']['max']:.0f}] sum={r['sum_components_us']:8.0f} | "
                      f"W {pct(r, 'weights'):4.1f} A {pct(r, 'attention'):4.1f} E {pct(r, 'encode'):4.1f} "
                      f"F {pct(r, 'fixed'):4.1f} | {r.get('attn_backend')} {r.get('attn_GBps', 0):.0f} GB/s")
            if "derived" in fr:
                for d, v in fr["derived"].items():
                    g = " ".join(f"g{k[1:]}: tau*={x['tau_breakeven']:.2f} ceil={x['ceiling_speedup']:.2f}"
                                 for k, x in v.items() if k.startswith("g"))
                    print(f"      {fname} {d:>11}: c={v['c']:.3f}  {g}")


if __name__ == "__main__":
    main()
