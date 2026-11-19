"""E10 phase 1 analysis: structural-config regret of the LUT VQ-KV draft-attention kernel.
Usage: python analyze_p1.py OUT_STEM raw1.json [raw2.json ...]  -> OUT_STEM tables.md (+ heatmap.png)

Structural config ("struct") = variant without the KST field (nw, minb, red, pf, lutm, sb) or 'orig' (the unmodified
kernel). Splits/chunk is a runtime knob: per (struct, condition) it is picked either by the oracle over measured chunks
(T_or) or by the wave-aware rule recomputed with the struct's blocks/SM and slots (T_rule).
Classes: exact = fp16 LUT and SB in {0, 2} (same math up to reassociation); int8 = adds LUTM 1/2 (approximate scores);
sb1 = static bound (unsafe on real data: speed reference only)."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

stem = Path(sys.argv[1])
raws = [json.load(open(p)) for p in sys.argv[2:]]
METRIC = "t_tot_us"
rows = [r for R in raws for r in R["rows"]]


def skey(r):
    return "orig" if r["orig"] else r["variant"].split("_", 1)[1]


INFO = {}
for R in raws:
    for nm, v in R["variants"].items():
        s = "orig" if v["orig"] else nm.split("_", 1)[1]
        INFO[(s, v["kst"])] = v

# T[cond][struct] -> list of rows ; conds = (kst, n, bh)
T = defaultdict(lambda: defaultdict(list))
for r in rows:
    T[(r["kst"], r["n"], r["bh"])][skey(r)].append(r)
conds = sorted(T)
ksts = sorted({c[0] for c in conds})
ns = sorted({c[1] for c in conds})
bhs = sorted({c[2] for c in conds})
structs = sorted({s for c in conds for s in T[c]})


def cls(s):
    if s == "orig":
        return "exact"
    f = dict(x.split(":") for x in [])  # noqa
    parts = {p[0]: int(p[1:]) for p in s.split("_")}
    if parts["s"] == 1:
        return "sb1"
    return "exact" if parts["l"] == 0 else "int8"


CLASSES = {"exact": {"exact"}, "exact+int8": {"exact", "int8"}, "all": {"exact", "int8", "sb1"}}


def rule_splits(n, bh, sms, bps, slots, F_TOK=1500, CS_TOK=200):  # = bench_p1.rule_splits
    best = None
    for s in range(1, max(1, -(-n // 64)) + 1):
        cost = math.ceil(bh * s / (sms * bps)) * (bps * math.ceil(n / s) + F_TOK) + CS_TOK * s * slots / 16
        if best is None or cost < best[0]:
            best = (cost, s)
    return best[1]


def t_or(c, s):
    rs = T[c].get(s)
    if not rs:
        return None
    return min(r[METRIC] for r in rs)  # duplicates (several raws): min of medians


def t_rule(c, s):
    rs = [r for r in T[c].get(s, []) if r.get("is_rule")]
    if s == "orig" or not rs:  # orig: phase-0 rule (bps 1, 16 slots) == p1 rule with bps 1 / slots 16
        rs2 = T[c].get(s, [])
        if not rs2:
            return None
        if s == "orig":
            # recompute the rule on the measured orig chunks
            sp = rule_splits(c[1], c[2], raws[0]["limits"]["sm_count"], 1, 16)
            best = min(rs2, key=lambda r: (abs(math.log(r["splits"] / sp)), r["chunk"]))
            return best[METRIC]
        return None
    return min(r[METRIC] for r in rs)


def best_or_row(c, s):
    rs = T[c].get(s)
    return min(rs, key=lambda r: r[METRIC]) if rs else None


def avail(c, klass):
    return [s for s in T[c] if cls(s) in CLASSES[klass] and t_or(c, s) is not None]


def opt(c, klass, tf=t_or):
    ss = [s for s in avail(c, klass) if tf(c, s) is not None]
    s = min(ss, key=lambda s: tf(c, s))
    return s, tf(c, s)


def reg(c, s, klass, tf=t_or, ref_tf=None):
    t = tf(c, s)
    if t is None:
        return None
    return t / opt(c, klass, ref_tf or tf)[1] - 1


def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"max {100 * max(xs):.1f}% / median {100 * statistics.median(xs):.1f}% (n={len(xs)})"


def common(cs, klass):
    """structs measured on every condition in cs"""
    sets = [set(avail(c, klass)) for c in cs]
    return set.intersection(*sets) if sets else set()


out = []
P = out.append
R0 = raws[0]
P(f"# E10 phase 1: structural knobs of the LUT VQ-KV draft attention ({R0['device']})\n")
P(f"- raw: {', '.join('`' + p + '`' for p in sys.argv[2:])}; metric `{METRIC}` (partial + combine, us per call, "
  f"median of {R0['args']['reps']} CUDA-graph reps, codes stream from DRAM).")
P(f"- limits: {R0['limits']}; ref read BW {R0['bw_ref_GBps']:.0f} GB/s")
for R in raws:
    P(f"- `{R['args']['out']}`: GPU {R['cuda_visible']} uuid {R['gpu_uuid']}; apps before {R['apps_before']}; "
      f"after {R['apps_after']}")
P("- struct name: w = warps/block, b = launch-bounds minBlocks, r = in-block partial reduction, p = code prefetch, "
  "l = LUT (0 fp16, 1 int8 per-(stage,j,head) scale, 2 int8 per-head packed), s = max (0 online, 1 static LUT "
  "bound [unsafe], 2 lazy online max)\n")

# ---------------------------------------------------------------- 0. variant resources
P("## 0. Variant resources (per KST)\n")
P("| struct | KST1 regs / local B / smem B / blocks/SM / slots | KST2 regs / local B / smem B / blocks/SM / slots |")
P("|---|---|---|")
for s in structs:
    cells = []
    for k in (1, 2):
        v = INFO.get((s, k))
        if v is None:
            cells.append("-")
            continue
        o = v["occ"]
        cells.append(f"{o['regs']} / {o['local_bytes']} / {o['dyn_smem']} / {o['blocks_per_sm']} / "
                     f"{o.get('slots_per_split', 16)}")
    P(f"| {s} | " + " | ".join(cells) + " |")

# ---------------------------------------------------------------- 1. per-variant speed vs orig
P("\n## 1. Per-struct speed vs the original kernel (oracle splits for both): geomean / min / max of T_orig / T_struct\n")
P("| struct | class | KST1 geomean | KST1 min..max | KST2 geomean | KST2 min..max | #conds |")
P("|---|---|---|---|---|---|---|")
sp_tab = {}
for s in structs:
    cells = []
    nn = 0
    for k in ksts:
        xs = [t_or(c, "orig") / t_or(c, s) for c in conds if c[0] == k and t_or(c, s) and t_or(c, "orig")]
        nn += len(xs)
        if xs:
            gm = math.exp(sum(map(math.log, xs)) / len(xs))
            sp_tab[(s, k)] = gm
            cells += [f"{gm:.3f}", f"{min(xs):.2f}..{max(xs):.2f}"]
        else:
            cells += ["-", "-"]
    P(f"| {s} | {cls(s)} | " + " | ".join(cells) + f" | {nn} |")

# ---------------------------------------------------------------- 2. per-condition optimum
for klass in ("exact", "exact+int8"):
    P(f"\n## 2{'a' if klass == 'exact' else 'b'}. Optimum per condition, class `{klass}` (oracle splits)\n")
    P("| KST | n | B*Hkv | best struct | chunk | splits | t us | orig best us | speedup vs orig | 2nd best (regret) | "
      "best with rule splits (regret vs oracle opt) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in conds:
        s, t = opt(c, klass)
        r = best_or_row(c, s)
        to = t_or(c, "orig")
        others = sorted((t_or(c, x), x) for x in avail(c, klass) if x != s)
        sec = f"{others[0][1]} ({100 * (others[0][0] / t - 1):.1f}%)" if others else "-"
        sr, trr = opt(c, klass, t_rule)
        P(f"| {c[0]} | {c[1]} | {c[2]} | {s} | {r['chunk']} | {r['splits']} | {t:.1f} | "
          f"{to:.1f} | {to / t:.3f} | {sec} | {sr} ({100 * (trr / t - 1):.1f}%) |")

# ---------------------------------------------------------------- 3. regret of structural policies
for klass in ("exact", "exact+int8"):
    P(f"\n## 3{'a' if klass == 'exact' else 'b'}. Structural-config regret, class `{klass}`\n")
    P("Regret = T(policy struct, splits by oracle | rule) / T(best struct at oracle splits) - 1.\n")
    for tfn, tf in (("oracle splits", t_or), ("rule splits", t_rule)):
        cs_all = [c for c in conds]
        cand = common(cs_all, klass)
        cand = [s for s in cand if all(tf(c, s) is not None for c in cs_all)]
        if cand:
            g = min(cand, key=lambda s: (max(reg(c, s, klass, tf, t_or) for c in cs_all),
                                         statistics.median(reg(c, s, klass, tf, t_or) for c in cs_all)))
            P(f"- [{tfn}] one global struct: `{g}` -> {summ([reg(c, g, klass, tf, t_or) for c in cs_all])}")
        for k in ksts:
            cs = [c for c in conds if c[0] == k]
            cand = [s for s in common(cs, klass) if all(tf(c, s) is not None for c in cs)]
            if not cand:
                continue
            g = min(cand, key=lambda s: (max(reg(c, s, klass, tf, t_or) for c in cs),
                                         statistics.median(reg(c, s, klass, tf, t_or) for c in cs)))
            oth = [c for c in conds if c[0] != k]
            P(f"  - [{tfn}] best struct for KST{k} only: `{g}` -> on KST{k}: "
              f"{summ([reg(c, g, klass, tf, t_or) for c in cs])}; transferred to the other KST: "
              f"{summ([reg(c, g, klass, tf, t_or) for c in oth])}")
        P(f"  - [{tfn}] original kernel: {summ([reg(c, 'orig', klass, tf, t_or) for c in cs_all])}")
    # axis transfer
    P("\n| axis | splits | regret of using cond j's best struct on cond i (i, j differ only in that axis) |")
    P("|---|---|---|")
    for nm, ax in (("KST", 0), ("n", 1), ("B*Hkv", 2)):
        for tfn, tf in (("oracle", t_or), ("rule", t_rule)):
            rs = []
            for i in conds:
                for j in conds:
                    if i == j or any(i[a] != j[a] for a in range(3) if a != ax):
                        continue
                    sj, _ = opt(j, klass)
                    rs.append(reg(i, sj, klass, tf, t_or))
            P(f"| {nm} | {tfn} | {summ(rs)} |")
    P("\nKST transfer per (n, B*Hkv): best struct KST1 / KST2; regret of KST2-best on KST1 / KST1-best on KST2 "
      "(oracle splits)\n")
    P("| n \\ B*Hkv | " + " | ".join(str(b) for b in bhs) + " |")
    P("|---|" + "---|" * len(bhs))
    for n in ns:
        cells = []
        for b in bhs:
            c1, c2 = (1, n, b), (2, n, b)
            if c1 in T and c2 in T:
                s1, s2 = opt(c1, klass)[0], opt(c2, klass)[0]
                a = reg(c1, s2, klass)
                bb = reg(c2, s1, klass)
                f = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
                cells.append(f"{s1} / {s2}: {f(a)} / {f(bb)}")
            else:
                cells.append("-")
        P(f"| {n} | " + " | ".join(cells) + " |")

# ---------------------------------------------------------------- 3c. one knob at a time
P("\n## 3c. One knob at a time around a base struct (oracle splits)\n")
P("For each knob, the other fields are fixed at the base; per condition the best value of the knob; regret = "
  "T(value) / T(best value of that knob at that condition) - 1. 'x-KST' = regret of the other KST's best value "
  "at the same (n, B*Hkv); 'one value' = best single value of the knob over all conditions.\n")


def parts(s):
    return {p[0]: int(p[1:]) for p in s.split("_")}


def mk(d):
    return "_".join(f"{k}{d[k]}" for k in "wbrpls")


for base in [b for b in ("w8_b1_r1_p1_l0_s2", "w16_b1_r1_p0_l0_s2", "w16_b1_r0_p0_l0_s0") if b in structs]:
    P(f"\nbase `{base}`\n")
    P("| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value "
      "(value) | max regret of per-KST value (values) |")
    P("|---|---|---|---|---|---|")
    bp = parts(base)
    for kn in "wbrpls":
        vals = sorted({parts(x)[kn] for x in structs if x != "orig" and
                       all(parts(x)[o] == bp[o] for o in "wbrpls" if o != kn)})
        if len(vals) < 2:
            continue
        sname = {v: mk({**bp, kn: v}) for v in vals}

        def tv(c, v):
            return t_or(c, sname[v])
        cs = [c for c in conds if all(tv(c, v) is not None for v in vals)]
        if not cs:
            continue
        bestv = {c: min(vals, key=lambda v: tv(c, v)) for c in cs}
        rg = lambda c, v: tv(c, v) / tv(c, bestv[c]) - 1  # noqa: E731
        xk = [rg(c, bestv[(3 - c[0], c[1], c[2])]) for c in cs if (3 - c[0], c[1], c[2]) in bestv]
        one = min(vals, key=lambda v: max(rg(c, v) for c in cs))
        perk = {}
        for k in ksts:
            ck = [c for c in cs if c[0] == k]
            if ck:
                perk[k] = min(vals, key=lambda v: max(rg(c, v) for c in ck))
        pk = max(rg(c, perk[c[0]]) for c in cs)
        cell = " ; ".join(",".join(f"{c[1] // 1024}k/{c[2]}:{bestv[c]}" for c in cs if c[0] == k) for k in ksts)
        P(f"| {kn} | {vals} | {cell} | {100 * max(xk):.1f}% | {100 * max(rg(c, one) for c in cs):.1f}% ({one}) | "
          f"{100 * pk:.1f}% ({perk}) |")

# ---------------------------------------------------------------- 4. KST1/KST2 ratio
P("\n## 4. KST1 / KST2 time ratio: original vs best struct per KST (oracle splits)\n")
P("| n | B*Hkv | orig ratio | best exact ratio (KST1 / KST2 struct) | best exact+int8 ratio | best incl. sb1 ratio |")
P("|---|---|---|---|---|---|")
for n in ns:
    for b in bhs:
        c1, c2 = (1, n, b), (2, n, b)
        if c1 not in T or c2 not in T:
            continue
        o = t_or(c1, "orig") / t_or(c2, "orig")
        e1, e2 = opt(c1, "exact"), opt(c2, "exact")
        i1, i2 = opt(c1, "exact+int8"), opt(c2, "exact+int8")
        a1, a2 = opt(c1, "all"), opt(c2, "all")
        P(f"| {n} | {b} | {o:.3f} | {e1[1] / e2[1]:.3f} ({e1[0]} / {e2[0]}) | {i1[1] / i2[1]:.3f} | "
          f"{a1[1] / a2[1]:.3f} |")

# ---------------------------------------------------------------- 5. full table of oracle times
P("\n## 5. Oracle-splits time per struct and condition (us; * = class-exact optimum, + = overall optimum)\n")
P("| struct | " + " | ".join(f"K{c[0]} {c[1] // 1024}k {c[2]}" for c in conds) + " |")
P("|---|" + "---|" * len(conds))
for s in structs:
    cells = []
    for c in conds:
        t = t_or(c, s)
        if t is None:
            cells.append("-")
            continue
        mk = ("*" if opt(c, "exact")[0] == s else "") + ("+" if opt(c, "all")[0] == s else "")
        cells.append(f"{t:.0f}{mk}")
    P(f"| {s} | " + " | ".join(cells) + " |")

Path(str(stem) + "tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
print("wrote", str(stem) + "tables.md")
