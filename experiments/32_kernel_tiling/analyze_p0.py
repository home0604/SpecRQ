"""E10 phase 0 analysis: optimal chunk per condition (kst, n, bh), cross-condition regret, policy regrets, KST ratio
breakdown, heatmaps.  Usage: python analyze_p0.py results/E10/lut_tiling_p0_raw.json [metric=t_tot_us]
Writes <stem>tables.md and <stem>heatmap.png next to the raw json (stem = name without 'raw.json')."""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

raw = Path(sys.argv[1])
METRIC = sys.argv[2] if len(sys.argv) > 2 else "t_tot_us"
R = json.load(open(raw))
rows = R["rows"]
TUNE_SET = (1024, 2048, 4096, 8192)  # e2e_real.tune_plan VQ candidates
FIXED_CHUNKS = (256, 512, 1024, 2048, 4096, 8192, 16384)

T = defaultdict(dict)  # cond -> chunk -> row
for r in rows:
    T[(r["kst"], r["n"], r["bh"])][r["chunk"]] = r
conds = sorted(T)
ksts = sorted({c[0] for c in conds})
ns = sorted({c[1] for c in conds})
bhs = sorted({c[2] for c in conds})


def t_at(c, chunk):
    """time of condition c at an absolute chunk (chunk >= n -> splits = 1 row); None if not measured."""
    d = T[c]
    if chunk >= c[1]:
        cs = [k for k in d if k >= c[1]]
        return d[min(cs)][METRIC] if cs else None
    return d[chunk][METRIC] if chunk in d else None


def t_splits(c, splits):
    d = T[c]
    best = [r for r in d.values() if r["splits"] == splits]
    return min(r[METRIC] for r in best) if best else None


def t_grid(c, grid):  # total-programs policy: splits = ceil(grid / bh), nearest measured splits
    want = max(1, -(-grid // c[2]))
    d = T[c]
    if not d:
        return None
    r = min(d.values(), key=lambda r: (abs(math.log(r["splits"] / want)), r["chunk"]))
    return r[METRIC]


def opt(c):
    d = T[c]
    k = min(d, key=lambda k: d[k][METRIC])
    return k, d[k]


def reg(c, t):
    return None if t is None else t / opt(c)[1][METRIC] - 1


def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"max {100 * max(xs):.1f}% / median {100 * statistics.median(xs):.1f}% (n={len(xs)})"


out = []
P = out.append
lim, occ = R["limits"], R["occupancy"]
P(f"# E10 phase 0: LUT VQ-KV draft attention, chunk/splits sweep ({R['device']})\n")
P(f"- raw: `{raw}`; metric = `{METRIC}` (us per call, median of {R['args']['reps']} reps of CUDA-graph replays; "
  f"t_tot = lut_rag partial + Triton combine; each graph cycles over R code copies >= {R['args']['l2_mult']}x L2 so "
  "codes stream from DRAM).")
P(f"- device limits: {lim}; reference read BW (torch.sum 512 MiB fp32) = {R['bw_ref_GBps']:.0f} GB/s")
for k, o in occ.items():
    P(f"- KST{k}: dyn smem {o['dyn_smem']} B ({o['dyn_smem'] / 1024:.2f} KiB: value cb {o['vcb_bytes']}, LUT "
      f"{o['lut_bytes']}, warp tail {o['tail_bytes']}, qmu 64), regs/thread {o['regs']}, local {o['local_bytes']} B, "
      f"512 threads -> {o['regs'] * 512} regs/block; cudaOccupancy blocks/SM = {o['blocks_per_sm']}")
P(f"- GPU apps before: {R['apps_before']}; after (own process included, uuid {R['gpu_uuid']}): {R['apps_after']}\n")

# ---------------------------------------------------------------- 1. per-condition optimum
P("## 1. Optimum per condition\n")
P("| KST | n | B*Hkv | opt chunk | splits | grid | waves | t_opt us | part / comb / fixed us | code GB/s | "
  "regret: other-KST opt | tune_plan set | best fixed-chunk (global) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
# global best fixed chunk (min of max regret over all conditions)
glob = {}
for ch in FIXED_CHUNKS:
    rs = [reg(c, t_at(c, ch)) for c in conds]
    if all(x is not None for x in rs):
        glob[ch] = (max(rs), statistics.median(rs))
gbest = min(glob, key=lambda k: glob[k]) if glob else None
for c in conds:
    k, r = opt(c)
    oth = [o for o in ksts if o != c[0]]
    ro = reg(c, t_at(c, opt((oth[0], c[1], c[2]))[0])) if oth and (oth[0], c[1], c[2]) in T else None
    rt = reg(c, min(x for x in (t_at(c, ch) for ch in TUNE_SET) if x is not None))
    rg = reg(c, t_at(c, gbest)) if gbest else None
    f = lambda x: "-" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    P(f"| {c[0]} | {c[1]} | {c[2]} | {k} | {r['splits']} | {r['grid']} | {r['waves']} | {r[METRIC]:.1f} | "
      f"{r['t_part_us']:.1f} / {r['t_comb_us']:.1f} / {r['t_fixed_us']:.1f} | {r['code_GBps']:.0f} | {f(ro)} | "
      f"{f(rt)} | {f(rg)} |")

# ---------------------------------------------------------------- 2. policy regrets
P("\n## 2. Policy regret (T(policy) / T(opt) - 1 over all conditions)\n")
P(f"- (a) one global fixed chunk: best = {gbest} -> {summ([reg(c, t_at(c, gbest)) for c in conds])}")
for ch in FIXED_CHUNKS:
    if ch in glob:
        P(f"  - chunk {ch}: max {100 * glob[ch][0]:.1f}% / median {100 * glob[ch][1]:.1f}%")
for k in ksts:
    cs = [c for c in conds if c[0] == k]
    best = min((ch for ch in FIXED_CHUNKS if all(t_at(c, ch) is not None for c in cs)),
               key=lambda ch: max(reg(c, t_at(c, ch)) for c in cs))
    P(f"- (b) best fixed chunk per KST: KST{k} -> {best}: {summ([reg(c, t_at(c, best)) for c in cs])}")
P(f"- (c) current tune_plan VQ set {TUNE_SET}, oracle pick per condition: "
  f"{summ([reg(c, min(x for x in (t_at(c, ch) for ch in TUNE_SET) if x is not None)) for c in conds])}")
for k in ksts:
    cs = [c for c in conds if c[0] == k]
    P(f"  - KST{k}: {summ([reg(c, min(x for x in (t_at(c, ch) for ch in TUNE_SET) if x is not None)) for c in cs])}")
# total-grid policy
grids = sorted({r["grid"] for r in rows})
gl = {}
for g in (128, 256, 512, 1024, 2048):
    gl[g] = [reg(c, t_grid(c, g)) for c in conds]
gb = min(gl, key=lambda g: max(x for x in gl[g] if x is not None))
P(f"- (d) one global total-program target (splits = ceil(G / bh), nearest measured): best G = {gb}: {summ(gl[gb])}")
for g in gl:
    P(f"  - G {g}: {summ(gl[g])}")
SMS = R["limits"]["sm_count"]


def t_near_splits(c, want):
    d = T[c]
    r = min(d.values(), key=lambda r: (abs(math.log(r["splits"] / want)), r["chunk"]))
    return r[METRIC]


for nm, fn in (("ceil(SMs / bh)", lambda b: max(1, -(-SMS // b))), ("floor(SMs / bh)", lambda b: max(1, SMS // b))):
    P(f"- (e) one-wave rule splits = {nm} (SMs = {SMS}, 1 block/SM): "
      f"{summ([reg(c, t_near_splits(c, fn(c[2]))) for c in conds])}")
# wave-aware analytic rule: per-SM work = waves * (chunk + F) token-equivalents + combine Cs * splits
# (F, Cs fitted in-sample on the phase-0 sweeps: F = 1500 tokens, Cs = 200 tokens; KST-independent)
F_TOK, CS_TOK = 1500, 200


def rule_splits(c):
    ms = sorted({r["splits"] for r in T[c].values()})
    return min(ms, key=lambda s: math.ceil(c[2] * s / SMS) * (math.ceil(c[1] / s) + F_TOK) + CS_TOK * s)


P(f"- (f) wave-aware rule argmin_s ceil(bh*s/SMs)*(ceil(n/s)+{F_TOK}) + {CS_TOK}*s over measured splits "
  f"(constants fitted in-sample, KST-independent): "
  f"{summ([reg(c, min(r[METRIC] for r in T[c].values() if r['splits'] == rule_splits(c))) for c in conds])}")

# ---------------------------------------------------------------- 3. axis transfer regret
P("\n## 3. Cross-condition transfer regret by axis (use opt chunk of j on i; i, j differ only in that axis)\n")
P("| axis | carried as | regret summary |")
P("|---|---|---|")
axes = {"KST": 0, "n": 1, "B*Hkv": 2}
for nm, ax in axes.items():
    for how in ("chunk", "splits"):
        rs = []
        for i in conds:
            for j in conds:
                if i == j or any(i[a] != j[a] for a in range(3) if a != ax):
                    continue
                kj, rj = opt(j)
                t = t_at(i, kj) if how == "chunk" else t_splits(i, rj["splits"])
                rs.append(reg(i, t))
        P(f"| {nm} | {how} | {summ(rs)} |")
# KST-only, by (n, bh)
P("\nKST transfer per (n, B*Hkv): regret of KST2-opt chunk on KST1 / KST1-opt chunk on KST2\n")
P("| n \\ B*Hkv | " + " | ".join(str(b) for b in bhs) + " |")
P("|---|" + "---|" * len(bhs))
for n in ns:
    cells = []
    for b in bhs:
        if (1, n, b) in T and (2, n, b) in T:
            a = reg((1, n, b), t_at((1, n, b), opt((2, n, b))[0]))
            c2 = reg((2, n, b), t_at((2, n, b), opt((1, n, b))[0]))
            cells.append(f"{opt((1, n, b))[0]}/{opt((2, n, b))[0]}: {100 * (a or 0):.1f}% / {100 * (c2 or 0):.1f}%")
        else:
            cells.append("-")
    P(f"| {n} | " + " | ".join(cells) + " |")

# ---------------------------------------------------------------- 4. KST1 / KST2 ratio + breakdown
P("\n## 4. KST1 / KST2 time ratio and breakdown\n")
P("T = fixed (q RoPE + LUT build + partial store, measured with an empty token range, same grid) + stream "
  "(t_part - t_fixed) + combine.\n")
P("| n | B*Hkv | ratio at own opt (t_tot) | chunk | ratio t_part same chunk | fixed1/fixed2 us | stream1/stream2 us "
  "| stream ratio | comb us | code GB/s K1/K2 |")
P("|---|---|---|---|---|---|---|---|---|---|")
for n in ns:
    for b in bhs:
        c1, c2 = (1, n, b), (2, n, b)
        if c1 not in T or c2 not in T:
            continue
        k2, r2 = opt(c2)
        k1, r1 = opt(c1)
        ch = k2 if k2 in T[c1] else k1
        a, bb = T[c1].get(ch), T[c2].get(ch)
        if a is None or bb is None:
            continue
        s1, s2 = a["t_part_us"] - a["t_fixed_us"], bb["t_part_us"] - bb["t_fixed_us"]
        P(f"| {n} | {b} | {r1[METRIC] / r2[METRIC]:.3f} | {ch} | {a['t_part_us'] / bb['t_part_us']:.3f} | "
          f"{a['t_fixed_us']:.1f}/{bb['t_fixed_us']:.1f} | {s1:.1f}/{s2:.1f} | {s1 / s2:.3f} | "
          f"{a['t_comb_us']:.1f}/{bb['t_comb_us']:.1f} | {a['code_GBps']:.0f}/{bb['code_GBps']:.0f} |")

# ---------------------------------------------------------------- 5. full chunk curves
P("\n## 5. Full sweep (t_tot us per chunk; * = optimum)\n")
allch = sorted({r["chunk"] for r in rows})
P("| KST | n | B*Hkv | " + " | ".join(str(c) for c in allch) + " |")
P("|---|---|---|" + "---|" * len(allch))
for c in conds:
    k, _ = opt(c)
    P(f"| {c[0]} | {c[1]} | {c[2]} | " + " | ".join(
        ("-" if ch not in T[c] else f"{T[c][ch][METRIC]:.0f}{'*' if ch == k else ''}") for ch in allch) + " |")

stem = raw.name.replace("raw.json", "")
(raw.parent / f"{stem}tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:80]))

# ---------------------------------------------------------------- heatmaps
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fig, axs = plt.subplots(1, len(ksts) + 1, figsize=(5.2 * (len(ksts) + 1), 4.6), constrained_layout=True)
for ai, k in enumerate(ksts):
    ax = axs[ai]
    M = np.full((len(ns), len(bhs)), np.nan)
    lab = [["" for _ in bhs] for _ in ns]
    for i, n in enumerate(ns):
        for j, b in enumerate(bhs):
            c = (k, n, b)
            if c not in T:
                continue
            ch, r = opt(c)
            M[i, j] = math.log2(ch)
            o = (3 - k, n, b)
            ro = reg(c, t_at(c, opt(o)[0])) if o in T else None
            lab[i][j] = f"{ch} (g{r['grid']})\n{r[METRIC]:.0f}us" + (f"\nx-KST {100 * ro:.0f}%" if ro is not None else "")
    im = ax.imshow(M, cmap="Blues", vmin=6, vmax=14, aspect="auto")
    for i in range(len(ns)):
        for j in range(len(bhs)):
            if lab[i][j]:
                ax.text(j, i, lab[i][j], ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 11 else "#1a1a1a")
    ax.set_xticks(range(len(bhs)), [str(b) for b in bhs])
    ax.set_yticks(range(len(ns)), [f"{n // 1024}k" for n in ns])
    ax.set_xlabel("B*Hkv (grid x)")
    ax.set_ylabel("VQ tokens per row n")
    ax.set_title(f"KST{k}: optimal chunk (t_opt; regret of other-KST opt)", fontsize=9)
    fig.colorbar(im, ax=ax, label="log2(opt chunk)", shrink=0.8)
ax = axs[-1]
N = len(conds)
RM = np.full((N, N), np.nan)
for i, ci in enumerate(conds):
    for j, cj in enumerate(conds):
        t = t_at(ci, opt(cj)[0])
        if t is not None:
            RM[i, j] = 100 * reg(ci, t)
im = ax.imshow(np.clip(RM, 0, 100), cmap="Blues", vmin=0, vmax=100, aspect="auto")
ticks = [f"K{c[0]} {c[1] // 1024}k {c[2]}" for c in conds]
ax.set_xticks(range(N), ticks, rotation=90, fontsize=4)
ax.set_yticks(range(N), ticks, fontsize=4)
ax.set_xlabel("config source j (opt chunk of j)")
ax.set_ylabel("evaluated condition i")
ax.set_title("regret R[i,j] % (clipped at 100)", fontsize=9)
fig.colorbar(im, ax=ax, label="regret %", shrink=0.8)
fig.savefig(raw.parent / f"{stem}heatmap.png", dpi=160)
print("wrote", raw.parent / f"{stem}tables.md", raw.parent / f"{stem}heatmap.png")
