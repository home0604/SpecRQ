"""E10 phase 3 tables from bench_p3 raw JSON(s) (+ optional check_p3 JSON).
Metric: t_tot_us (partial + combine, median of reps); oracle = min over the measured chunks per (impl, condition).
With several raws (replicates on different GPUs) the per-(impl, condition) oracle values are averaged and the
replicate spread is reported. Also: min-over-reps oracle (t_tot_min), fixed-splits table, ablation table.
Usage: python analyze_p3.py --raw r1.json [r2.json] [--check check.json] --out tables.md"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--check")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.load(open(p)) for p in a.raw]
    L = []
    P = L.append
    dev = recs[0]["device"]
    P(f"# E10 phase 3: A100-oriented LUT VQ-KV kernel variants ({dev})\n")
    P("- raw: " + ", ".join(f"`{p}`" for p in a.raw) + "; metric `t_tot_us` = partial + Triton combine, us per call "
      "(median of reps, CUDA graph over code copies >= 3x L2: codes stream from DRAM); oracle = min over measured "
      "chunks (splits) per implementation and condition.")
    for p, r in zip(a.raw, recs):
        mine = [x for x in r["apps_before"] + r["apps_after"] if r["gpu_uuid"] in x]
        P(f"- `{p}`: GPU {r['cuda_visible']} {r['gpu_uuid']}, arch {r['arch']}, ref read BW {r['bw_ref_GBps']:.0f} GB/s, "
          f"sweep {r['sweep_s']:.0f} s, warm-up {r['args'].get('warm_ms')} ms, reps {r['args']['reps']}; "
          f"co-tenant check: processes on this GPU before/after = {mine if mine else 'none'} "
          f"(apps after include our own process)")
    P(f"- not launchable on this device (smem): {recs[0]['skipped']}\n")
    P("Variant name: orig_k = original kernel (26_magicvq, lazy flag 0), origlazy_k = original with lazy=1, "
      "p1: = phase-1 struct (NW8, in-block reduce, prefetch, lazy max), k{KST}_w{NW}_v{VM}_r{VREP}_h{HS}_a{AB}: "
      "VM 0 fp32 FFMA, 1 tensor-core mma P.V, 2 packed HFMA2; VREP = fp16 V codebook replicas; HS 1 = half2 score "
      "partial sums; AB = timing-only ablation (1 no V accumulation, 2 no LUT gather, 3 both).\n")

    P("## 0. Resources\n")
    P("| impl | regs | local B | dyn smem B | blocks/SM |")
    P("|---|---|---|---|---|")
    for nm, v in sorted(recs[0]["variants"].items()):
        o = v["occ"]
        P(f"| {nm} | {o['regs']} | {o['local_bytes']} | {o['dyn_smem']} | {o['blocks_per_sm']} |")

    # oracle per (rec, impl, cond)
    orc = [defaultdict(lambda: None) for _ in recs]
    orc_min = [defaultdict(lambda: None) for _ in recs]
    fixed = [dict() for _ in recs]
    for i, r in enumerate(recs):
        for row in r["rows"]:
            key = (row["variant"], row["kst"], row["n"], row["bh"])
            if orc[i][key] is None or row["t_tot_us"] < orc[i][key][0]:
                orc[i][key] = (row["t_tot_us"], row["splits"])
            if orc_min[i][key] is None or row["t_tot_min"] < orc_min[i][key]:
                orc_min[i][key] = row["t_tot_min"]
            fixed[i][key + (row["splits"],)] = (row["t_tot_us"], row["t_tot_min"])
    keys = sorted(set(k for o in orc for k in o))
    impls = sorted(set(k[0] for k in keys))
    conds = sorted(set(k[1:] for k in keys))

    def T(nm, c):
        vs = [o[(nm,) + c][0] for o in orc if o.get((nm,) + c) is not None]
        return statistics.mean(vs) if vs else None

    def Tmin(nm, c):
        vs = [o[(nm,) + c] for o in orc_min if o.get((nm,) + c) is not None]
        return min(vs) if vs else None

    def S(nm, c):
        return orc[0][(nm,) + c][1] if orc[0].get((nm,) + c) else None

    if len(recs) > 1:
        sp = []
        for k in keys:
            vs = [o[k][0] for o in orc if o.get(k) is not None]
            if len(vs) > 1:
                sp.append(max(vs) / min(vs) - 1)
        P(f"\n- replicate spread of oracle times (max/min - 1): max {100 * max(sp):.1f}% / median "
          f"{100 * statistics.median(sp):.1f}% (n={len(sp)})")

    exact = [nm for nm in impls if not nm.startswith(("orig", "p1:")) and nm.endswith("_a0")]
    abl = [nm for nm in impls if not nm.startswith(("orig", "p1:")) and not nm.endswith("_a0")]
    for kst in sorted(set(c[0] for c in conds)):
        P(f"\n## 1.{kst} KST{kst}: oracle t_tot (us) and speed-up vs the original\n")
        cs = [c for c in conds if c[0] == kst]
        ex = [nm for nm in exact if nm.startswith(f"k{kst}_")]
        P("| n | B*Hkv | orig (splits) | origlazy | p1 struct | best p3 | best p3 t | best vs orig | best vs p1 | "
          "VM1 w8 r1 h0 | VM2 w8 r1 h0 | GB/s best |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in cs:
            to, tl = T(f"orig_k{kst}", c), T(f"origlazy_k{kst}", c)
            t1 = T(f"p1:k{kst}_w8_b1_r1_p1_l0_s2", c)
            bt = min((T(nm, c), nm) for nm in ex if T(nm, c) is not None)
            gbs = 2 * kst * c[2] * c[1] * 16 / (bt[0] * 1e-6) / 1e9
            P(f"| {c[1]} | {c[2]} | {to:.1f} ({S(f'orig_k{kst}', c)}) | {tl if tl is None else f'{tl:.1f}'} | "
              f"{t1 if t1 is None else f'{t1:.1f}'} | {bt[1]} | {bt[0]:.1f} ({S(bt[1], c)}) | {to / bt[0]:.2f} | "
              f"{'' if t1 is None else f'{t1 / bt[0]:.2f}'} | {T(f'k{kst}_w8_v1_r1_h0_a0', c) or 0:.1f} | "
              f"{T(f'k{kst}_w8_v2_r1_h0_a0', c) or 0:.1f} | {gbs:.0f} |")
        P(f"\n### 1.{kst}b per-variant geomean speed-up over conditions (oracle splits)\n")
        P("| variant | vs orig geomean (min, max) | vs p1 geomean (min, max) | min-over-reps oracle vs orig geomean |")
        P("|---|---|---|---|")
        for nm in ex + [f"origlazy_k{kst}", f"p1:k{kst}_w8_b1_r1_p1_l0_s2"]:
            ro = [T(f"orig_k{kst}", c) / T(nm, c) for c in cs if T(nm, c)]
            r1 = [T(f"p1:k{kst}_w8_b1_r1_p1_l0_s2", c) / T(nm, c) for c in cs
                  if T(nm, c) and T(f"p1:k{kst}_w8_b1_r1_p1_l0_s2", c)]
            rm = [Tmin(f"orig_k{kst}", c) / Tmin(nm, c) for c in cs if Tmin(nm, c)]
            if not ro:
                continue
            P(f"| {nm} | {gm(ro):.3f} ({min(ro):.2f}, {max(ro):.2f}) | "
              f"{(f'{gm(r1):.3f} ({min(r1):.2f}, {max(r1):.2f})') if r1 else '-'} | {gm(rm):.3f} |")

    # fixed splits (A100 reference: 4 / 8 / 16)
    req = sorted(set(recs[0]["args"].get("splits") or []))
    if req:
        P("\n## 2. Fixed splits (median us / min us over reps), first raw\n")
        sel = sorted(set(k[0] for k in fixed[0]))
        for c in conds:
            P(f"\n### KST{c[0]} n {c[1]} B*Hkv {c[2]}\n")
            P("| impl | " + " | ".join(f"splits {s}" for s in req) + " | oracle (splits) |")
            P("|---|" + "---|" * (len(req) + 1))
            for nm in sel:
                if not (nm.startswith(f"k{c[0]}_") or nm.endswith(f"k{c[0]}") or f":k{c[0]}_" in nm):
                    continue
                cells = []
                for s in req:
                    x = fixed[0].get((nm,) + c + (s,))
                    cells.append(f"{x[0]:.1f} / {x[1]:.1f}" if x else "-")
                o = orc[0].get((nm,) + c)
                P(f"| {nm} | " + " | ".join(cells) + (f" | {o[0]:.1f} ({o[1]}) |" if o else " | - |"))

    if abl:
        P("\n## 3. Timing-only ablations (oracle t_tot; ratio to the same struct with AB=0)\n")
        P("AB 1 = V accumulation removed (codes still loaded + staged), AB 2 = LUT score gather replaced by a cheap op "
          "on the K codes, AB 3 = both (loads + softmax bookkeeping + prologue/LUT build + combine only).\n")
        P("| KST | n | B*Hkv | struct | AB0 us | AB1 us (ratio) | AB2 us (ratio) | AB3 us (ratio) | code-read floor us "
          "at ref BW |")
        P("|---|---|---|---|---|---|---|---|---|")
        bases = sorted(set(nm.rsplit("_a", 1)[0] for nm in abl))
        for c in conds:
            for bse in bases:
                if not bse.startswith(f"k{c[0]}_"):
                    continue
                t0 = T(bse + "_a0", c)
                if t0 is None:
                    continue
                cells = []
                for ab in (1, 2, 3):
                    t = T(f"{bse}_a{ab}", c)
                    cells.append(f"{t:.1f} ({t / t0:.2f})" if t else "-")
                floor = 2 * c[0] * c[1] * c[2] * 16 / (recs[0]["bw_ref_GBps"] * 1e9) * 1e6
                P(f"| {c[0]} | {c[1]} | {c[2]} | {bse} | {t0:.1f} | " + " | ".join(cells) + f" | {floor:.1f} |")

    if a.check:
        ck = json.load(open(a.check))
        P(f"\n## 4. Correctness (`{a.check}`, {ck['device']}): all passed = {ck['all_passed']}\n")
        P("Max rel = max |o - x| / max |x| over the bf16 attention outputs (after combine), over "
          f"{len(ck['cases'])} cases (KST 1/2, ragged, q scale 0.5 and 4).\n")
        P("| variant | max rel vs orig | max rel vs fp64 | orig max rel vs fp64 | max ratio (variant / orig vs fp64) | "
          "rel-Frobenius vs fp64 (orig) | passed |")
        P("|---|---|---|---|---|---|---|")
        for nm, s in sorted(ck["summary"].items()):
            P(f"| {nm} | {s['max_rel_vs_orig']:.2e} | {s['max_rel_vs_ref']:.2e} | {s['orig_max_rel_vs_ref']:.2e} | "
              f"{s['max_ratio_vs_ref_over_orig']:.3f} | {s['max_relfro_vs_ref']:.2e} ({s['orig_max_relfro_vs_ref']:.2e}) "
              f"| {s['passed']} |")
        P("\nPer case (q scale 0.5 cases only, where the fp16-LUT original itself is close to fp64):\n")
        P("| KST | n | chunk | orig vs fp64 | origlazy vs orig | worst p3 variant vs fp64 | worst p3 vs orig |")
        P("|---|---|---|---|---|---|---|")
        for c in ck["cases"]:
            w = max(c["var"].items(), key=lambda kv: kv[1]["vs_ref_maxrel"])
            wo = max(c["var"].items(), key=lambda kv: kv[1]["vs_orig_maxrel"])
            P(f"| {c['kst']} | {c['n']} (qs {c['qscale']}) | {c['chunk']} | {c['orig_vs_ref']:.2e} | "
              f"{c.get('origlazy_vs_orig', float('nan')):.1e} | {w[0]} {w[1]['vs_ref_maxrel']:.2e} | "
              f"{wo[0]} {wo[1]['vs_orig_maxrel']:.2e} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
