"""Tables from results/E4/l1_llama31_config_sweep_<tag>_*.json:
  (1) per config x q: target bpw -> AR step, draft bpw (per k) -> draft step, verify M=2/3/5/8
  (2) tau* / alpha* / upper bound per draft k and gamma (composed round time, [I])
  (3) kernel plans: smem bytes, smem vs global fallback, replication R, blocks/SM per linear shape and k
Usage: python -m specrq.infer.summarize_sweep --tag v1 [--md out.md]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    files = sorted(glob.glob(str(REPO / f"results/E4/l1_llama31_config_sweep_{a.tag}_*.json")))
    out = []
    p = out.append
    for f in files:
        r = json.load(open(f))
        m, c = r["metrics"], r["config"]
        L = m["L"]
        name = f"{m['name']} q{m['q']}" + (" (C1 specialized)" if m["specialized_c1"] else "")
        p(f"\n### {name}  bits={m['bits']} d={m['d']}  target {m['target_bpw_index']:.2f} bpw  "
          f"co-tenant start/end: {c.get('co_tenant_start')}/{c.get('co_tenant_end')}")
        ar = {int(k): v for k, v in m["ar"].items()}
        fwd = {int(k): v for k, v in m["fwd"].items()}
        p(f"AR k=L: {ar[L]['step_us']:.1f} us | verify M2/M3/M5/M8: "
          + " / ".join(f"{fwd[M]:.1f}" for M in (2, 3, 5, 8)) + " us")
        p("| k | draft bpw | draft step (V_d=32k) us | catch-up M=2 us | " +
          " | ".join(f"g={g} tau*/alpha*/ub" for g in c["gammas"]) + " |")
        p("|---|---|---|---|" + "---|" * len(c["gammas"]))
        model = m["model"]
        for k in range(1, L):
            cells = []
            for g in c["gammas"]:
                x = next(y for y in model if y["kd"] == k and y["gamma"] == g)
                al = x["alpha_required_iid"]
                cells.append(f"{x['tau_required']:.2f}/{al:.3f}/{x['speedup_upper_bound']:.2f}" if al is not None
                             else f"{x['tau_required']:.2f}/-/{x['speedup_upper_bound']:.2f}")
            p(f"| {k} | {m['draft_bpw_index'][str(k)]:.2f} | {ar[k]['step_us']:.1f} | {m['catchup'][str(k)]:.1f} | "
              + " | ".join(cells) + " |")
        if m.get("plans"):
            p("plans (k: gemv smem/use/R/bps | mma smem/use/R/bps | M=1 path):")
            for sn, pk in m["plans"].items():
                cells = []
                for k, pl in pk.items():
                    g, mm = pl["gemv"], pl["mma_M5"]
                    cells.append(f"k{k}: {g['smem_need'] // 1024}K/{'S' if g['use_smem'] else 'G'}/R{g['R']}/"
                                 f"{g['blocks_per_sm']} | {mm['smem_need'] // 1024}K/{'S' if mm['use_smem'] else 'G'}/"
                                 f"R{mm['R']}/{mm['blocks_per_sm']} | {pl.get('m1_path')}")
                p(f"  {sn}: " + "; ".join(cells))
        if m.get("compose_check"):
            p("compose check (measured vs composed round ms): " + ", ".join(
                f"g{x['gamma']}k{x['kd']} {x['measured_round_ms']:.2f}/{x['composed_round_ms']:.2f}"
                for x in m["compose_check"]))
    text = "\n".join(out)
    print(text)
    if a.md:
        Path(a.md).write_text(text)


if __name__ == "__main__":
    main()
