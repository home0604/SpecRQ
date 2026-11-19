"""E10 phase 2 analysis: regret of the new structural axes of the LUT VQ-KV draft-attention kernel.
Usage: python analyze_p2.py OUT_STEM raw1.json [raw2.json ...] -> OUT_STEM_tables.md + OUT_STEM_summary.json

struct = phase-2 variant without the KST field: w{NW}_s{LS}_d{DEC}_p{PIPE}_f{FUSE}; references 'p1' (phase-1 global
struct w8_b1_r1_p1_l0_s2, own extension) and 'orig'. Condition c = (KST, n, B*Hkv). T(c, struct) = min over measured
runtime knobs (oracle chunk / block count); with several raws: per raw, and 'pooled' = min over raws of the medians.
All phase-2 structs are exact (fp16 LUT, same math; check_p2.py).
Regret R(policy, c) = T(c, policy) / T(c, best p2 struct) - 1.
  - global struct: argmin over structs of max_c R (ties -> median)
  - axis A fixed to value a, other axes free per condition: R_A(a, c) = min_{struct: A=a} T / opt - 1;
    per-condition argmax of A; best fixed value (minimax over c) -> 'axis needs to be condition-dependent' if > 5%
  - transfer: R[i, j] = T_i(opt struct of j) / T_i(opt_i) - 1 over condition pairs differing in one condition axis
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

stem = Path(sys.argv[1])
paths = sys.argv[2:]
raws = [json.load(open(p)) for p in paths]
AX = {"w": "NW (warps/block)", "s": "LUT source (0 per-block build, 1 shared prebuilt LUT)",
      "d": "decomposition (0 split grid, 1 stream-K)", "p": "code pipeline (0 reg prefetch, 2/3 cp.async stages)",
      "f": "combine (0 separate Triton, 1 fused last-block)"}


def skey(r):
    if r["kind"] == "orig":
        return "orig"
    if r["kind"] == "p1":
        return "p1"
    return r["variant"].split("_", 1)[1]


def fields(s):
    return {p[0]: int(p[1:]) for p in s.split("_")}


def table(rows):
    T = defaultdict(dict)  # cond -> struct -> (t, row)
    for r in rows:
        c = (r["kst"], r["n"], r["bh"])
        s = skey(r)
        if s not in T[c] or r["t_tot_us"] < T[c][s][0]:
            T[c][s] = (r["t_tot_us"], r)
    return T


TS = [table(R["rows"]) for R in raws]
pooled = defaultdict(dict)
for T in TS:
    for c, d in T.items():
        for s, (t, r) in d.items():
            if s not in pooled[c] or t < pooled[c][s][0]:
                pooled[c][s] = (t, r)
conds = sorted(pooled)


def p2structs(T, cs):
    sets = [set(s for s in T[c] if s not in ("orig", "p1")) for c in cs]
    return sorted(set.intersection(*sets)) if sets else []


def opt(T, c, pool=None):
    ss = [s for s in T[c] if s not in ("orig", "p1") and (pool is None or s in pool)]
    s = min(ss, key=lambda s: T[c][s][0])
    return s, T[c][s][0]


def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"max {100 * max(xs):.1f}% / median {100 * statistics.median(xs):.1f}% (n={len(xs)})"


def gmean(xs):
    return math.exp(sum(map(math.log, xs)) / len(xs))


def analyze(T, label, out, J):
    P = out.append
    cs = [c for c in conds if c in T]
    if not cs:
        return
    S = p2structs(T, cs)
    P(f"\n## {label}\n")
    if not S:
        P("(no struct measured on every condition)")
        return
    reg = {s: {c: T[c][s][0] / opt(T, c, S)[1] - 1 for c in cs} for s in S}
    g = min(S, key=lambda s: (max(reg[s].values()), statistics.median(reg[s].values())))
    base = "w8_s0_d0_p0_f0"
    J[label] = dict(global_struct=g, global_max=max(reg[g].values()), global_median=statistics.median(reg[g].values()))
    P(f"- one global struct (minimax over {len(cs)} conditions, {len(S)} structs): `{g}` -> {summ(reg[g].values())}")
    if base in reg:
        P(f"- phase-1 struct re-implemented (`{base}`): {summ(reg[base].values())}")
        J[label]["p2base_max"] = max(reg[base].values())
    if all("p1" in T[c] for c in cs):
        r1 = [T[c]["p1"][0] / opt(T, c, S)[1] - 1 for c in cs]
        P(f"- phase-1 struct (own p1 extension, `p1`): {summ(r1)}")
        J[label]["p1_max"] = max(r1)
        J[label]["p1_median"] = statistics.median(r1)
    # per condition table
    P("\n| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | "
      "p1 t us (regret) | orig t us | best vs p1 | best vs orig |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    J[label]["conds"] = {}
    for c in cs:
        s, t = opt(T, c, S)
        r = T[c][s][1]
        oth = sorted((T[c][x][0], x) for x in S if x != s)
        sec = f"{oth[0][1]} ({100 * (oth[0][0] / t - 1):.1f}%)" if oth else "-"
        tp1 = T[c]["p1"][0] if "p1" in T[c] else None
        to = T[c]["orig"][0] if "orig" in T[c] else None
        P(f"| {c[0]} | {c[1]} | {c[2]} | {s} | {r['knob']} | {r['chunk']} | {r['nblk']} | {t:.1f} | {sec} | "
          f"{100 * reg[g][c]:.1f}% | " + (f"{tp1:.1f} ({100 * (tp1 / t - 1):.1f}%)" if tp1 else "-") + " | "
          + (f"{to:.1f}" if to else "-") + " | " + (f"{tp1 / t:.3f}" if tp1 else "-") + " | "
          + (f"{to / t:.3f}" if to else "-") + " |")
        J[label]["conds"][f"k{c[0]}_n{c[1]}_bh{c[2]}"] = dict(best=s, t=t, knob=r["knob"], t_p1=tp1, t_orig=to,
                                                              global_regret=reg[g][c], second=sec)
    if all("p1" in T[c] and "orig" in T[c] for c in cs):
        for k in sorted({c[0] for c in cs}):
            ck = [c for c in cs if c[0] == k]
            P(f"\n- KST{k}: best-per-cond vs p1 geomean {gmean([T[c]['p1'][0] / opt(T, c, S)[1] for c in ck]):.3f}, "
              f"global struct `{g}` vs p1 geomean {gmean([T[c]['p1'][0] / T[c][g][0] for c in ck]):.3f} "
              f"(min {min(T[c]['p1'][0] / T[c][g][0] for c in ck):.3f}, "
              f"max {max(T[c]['p1'][0] / T[c][g][0] for c in ck):.3f}); global vs orig geomean "
              f"{gmean([T[c]['orig'][0] / T[c][g][0] for c in ck]):.3f}")
    # per axis
    P("\n### Per axis: fix the axis value, other axes free per condition (regret vs best struct)\n")
    P("| axis | value | regret over conditions | conditions where this value is the axis argmax |")
    P("|---|---|---|---|")
    J[label]["axes"] = {}
    for ax in AX:
        vals = sorted({fields(s)[ax] for s in S})
        if len(vals) < 2:
            continue
        ra = {}
        for v in vals:
            ss = [s for s in S if fields(s)[ax] == v]
            ra[v] = {c: min(T[c][s][0] for s in ss) / opt(T, c, S)[1] - 1 for c in cs}
        am = {c: min(vals, key=lambda v: ra[v][c]) for c in cs}
        best_v = min(vals, key=lambda v: (max(ra[v].values()), statistics.median(ra[v].values())))
        for v in vals:
            P(f"| {AX[ax]} | {v}{' (best fixed)' if v == best_v else ''} | {summ(ra[v].values())} | "
              f"{sum(1 for c in cs if am[c] == v)} / {len(cs)} |")
        # near-tie-aware: value needed if fixing the best value costs > x
        J[label]["axes"][ax] = dict(best_fixed=best_v, best_fixed_max=max(ra[best_v].values()),
                                    best_fixed_median=statistics.median(ra[best_v].values()),
                                    per_value={str(v): dict(max=max(ra[v].values()),
                                                            median=statistics.median(ra[v].values()),
                                                            argmax_count=sum(1 for c in cs if am[c] == v))
                                               for v in vals},
                                    argmax={f"k{c[0]}_n{c[1]}_bh{c[2]}": am[c] for c in cs})
    P("\nPer-condition axis argmax (value; regret of the best-fixed value at that condition):\n")
    P("| KST | n | B*Hkv | " + " | ".join(AX[a].split(" (")[0] for a in J[label]["axes"]) + " |")
    P("|---|---|---|" + "---|" * len(J[label]["axes"]))
    for c in cs:
        cells = []
        for ax, d in J[label]["axes"].items():
            ss = [s for s in S if fields(s)[ax] == d["best_fixed"]]
            rb = min(T[c][s][0] for s in ss) / opt(T, c, S)[1] - 1
            cells.append(f"{d['argmax'][f'k{c[0]}_n{c[1]}_bh{c[2]}']} ({100 * rb:.1f}%)")
        P(f"| {c[0]} | {c[1]} | {c[2]} | " + " | ".join(cells) + " |")
    # transfer along condition axes
    P("\n### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)\n")
    P("| condition axis | regret |")
    P("|---|---|")
    J[label]["transfer"] = {}
    best = {c: opt(T, c, S)[0] for c in cs}
    for ai, an in ((0, "KST"), (1, "n"), (2, "B*Hkv")):
        xs = []
        for ci in cs:
            for cj in cs:
                if ci != cj and all(ci[k] == cj[k] for k in range(3) if k != ai):
                    xs.append(T[ci][best[cj]][0] / T[ci][best[ci]][0] - 1)
        P(f"| {an} | {summ(xs)} |")
        J[label]["transfer"][an] = dict(max=max(xs), median=statistics.median(xs)) if xs else None
    # full transfer matrix
    P("\nFull transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:\n")
    lab = [f"k{c[0]}n{c[1] // 1024}k b{c[2]}" for c in cs]
    P("| i \\ j | " + " | ".join(lab) + " |")
    P("|---|" + "---|" * len(cs))
    for ci, li in zip(cs, lab):
        P(f"| {li} | " + " | ".join(f"{100 * (T[ci][best[cj]][0] / T[ci][best[ci]][0] - 1):.1f}" for cj in cs) + " |")


out = []
P = out.append
R0 = raws[0]
P(f"# E10 phase 2: new structural axes of the LUT VQ-KV draft attention ({R0['device']})\n")
P(f"- raw: {', '.join('`' + p + '`' for p in paths)}; metric `t_tot_us` (LUT build if any + partial + combine, us per "
  f"call, median of {R0['args']['reps']} CUDA-graph reps, codes stream from DRAM); oracle runtime knob per struct.")
P(f"- limits: {R0['limits']}; ref read BW " + ", ".join(f"{R['bw_ref_GBps']:.0f}" for R in raws) + " GB/s")
for p, R in zip(paths, raws):
    P(f"- `{p}`: GPU {R['cuda_visible']} uuid {R['gpu_uuid']}; sweep {R['sweep_s']:.0f} s; apps before "
      f"{R['apps_before']}; after {R['apps_after']}")
P("- struct name: w = warps/block, s = LUT source (0 build per block, 1 separate build kernel + copy from L2), "
  "d = decomposition (0 (bh, split) grid, 1 stream-K over flattened tokens), p = code pipeline (0 register "
  "prefetch = phase 1, 2/3 = cp.async ring stages issued before the LUT prologue), f = fused combine")
J = {"raws": paths}
# resources
P("\n## 0. Variant resources\n")
P("| variant | regs | local B | smem B | blocks/SM | build regs |")
P("|---|---|---|---|---|---|")
for nm, v in sorted(R0["variants"].items()):
    o = v["occ"]
    P(f"| {nm} | {o['regs']} | {o['local_bytes']} | {o['dyn_smem']} | {o['blocks_per_sm']} | {o.get('build_regs', '-')} |")
if len(raws) > 1:
    # rep noise: same (struct, cond, knob) across raws
    key = lambda r: (r["variant"], r["kst"], r["n"], r["bh"], r["knob"])  # noqa: E731
    m = [dict((key(r), r["t_tot_us"]) for r in R["rows"]) for R in raws]
    common = set.intersection(*[set(x) for x in m])
    d = [max(x[k] for x in m) / min(x[k] for x in m) - 1 for k in common]
    if d:
        P(f"\n- rep-to-rep spread of identical (variant, condition, knob) medians: {summ(d)}; "
          f"p90 {100 * sorted(d)[int(0.9 * len(d))]:.1f}%")
        J["rep_spread"] = dict(max=max(d), median=statistics.median(d), p90=sorted(d)[int(0.9 * len(d))])
if len(raws) > 1:
    for i, T in enumerate(TS):
        analyze(T, f"raw {i + 1} ({paths[i]})", out, J)
analyze(pooled, "pooled (min over raws)" if len(raws) > 1 else "results", out, J)
for k in sorted({c[0] for c in conds}):  # KST is known at compile time: per-KST global struct / axes
    analyze({c: d for c, d in pooled.items() if c[0] == k}, f"pooled, KST{k} only", out, J)
stem.parent.mkdir(parents=True, exist_ok=True)
Path(str(stem) + "_tables.md").write_text("\n".join(out) + "\n")
json.dump(J, open(str(stem) + "_summary.json", "w"), indent=1)
print("\n".join(out))
