"""E7 c-alpha theory map: measured e2e points (SnapKV vs VQ/hybrid t2w) on the analytical speculative-decoding map.

Theory (experiments/28_sd_theory/breakeven.py): S(gamma) = tau / (gamma*c + v), tau = (1 - a^(gamma+1)) / (1 - a),
S_max = max over gamma. c = draft step / AR step, v = rest of the round / AR step, a = per-token acceptance (i.i.d.).

Per measured row (all-active accounting, the headline; median-tok/s timed rep of 3 (= the rep behind S_meas)):
  MEASURED  T_AR = 1000*B / AR tok/s of the same process (ms per AR step); T_round = aa wall / aa rounds;
            T_draft = aa draft_s / rounds; tau = aa tokens / (B * aa rounds); S_meas = aa tok/s / AR tok/s.
  DERIVED   c = T_draft / (gamma * T_AR);  v = (T_round - T_draft) / T_AR (verify + accept + encode + host);
            alpha = root of tau(alpha, gamma) = tau_measured (bisection).
  CHECKS    (1) S_model = tau / (gamma*c + v) vs S_meas: an accounting identity (same wall clock), so it only checks
                that the record's tau / phase times / AR baseline are mutually consistent.
            (2) cross-gamma: alpha from one gamma predicts tau and S at another measured gamma of the same draft /
                process (c, v of the target gamma measured). This is the real test of the i.i.d.-alpha model.
            (3) S_map: the background value at the plotted (c, alpha), i.e. max over gamma 1..8 with the
                representative v(gamma) fit below, vs S_meas.
Background v(gamma): least-squares line v = v0 + v1*gamma over all rows of the plotted points (stated on the figure);
the S = 1 contour is also drawn for the two regime fits (32k B16; 16k B16 + 32k B8) to show its sensitivity to v.

Data: results/E7/llama8b_magicvq_{v6e2e,v7,v7_g6,v7mse}_*.json (4x RTX 4090, TP=4, bf16 Llama-3.1-8B-Instruct,
bf16 target KV). Best = max all-active speedup over every measured SnapKV budget / gamma (resp. VQ/hybrid with the
t2w codebook, resp. StreamingLLM-512) across all processes of a point, as in e2e_summarize_v7.py.
Panel B: the same for the default (def) codebook, at the points that have def rows (v6e2e / v7mse).
Usage: taskset -c 16-31,48-63 nice -n 10 /hai/anaconda3/bin/python3 experiments/26_magicvq/plot_c_alpha.py
"""
import glob
import json
import os
import statistics

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(REPO, "results", "E7")
OUTP = os.path.join(RES, "llama8b_magicvq_c_alpha_map")
FILES = ["v6e2e", "v7", "v7mse"]  # v7 glob also covers v7_g6_*
GMAP = np.arange(1, 9)  # gamma range of the background map
YT = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]  # y axis is -log10(1 - alpha), labelled in alpha

# palette (dataviz reference, light): validated --pairs all, see report
SNAP, VQ, STREAM = "#eb6834", "#2a78d6", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"

POINTS = [  # (task, ctx, B, gen, label on figure)
    ("syn_quote", 32768, 16, 256, "syn_quote 32k B16"),
    ("pg19", 32768, 16, 256, "PG-19 32k B16"),
    ("syn_quote", 32768, 8, 256, "syn_quote 32k B8"),
    ("gov_report", 32768, 8, 256, "gov_report 32k B8"),
    ("hotpotqa_q", 16384, 16, 256, "hotpotqa_q 16k B16"),
    ("gov_report", 16384, 16, 256, "gov_report 16k B16"),
    ("qmsum", 16384, 16, 256, "qmsum 16k B16"),
    ("repobench-p", 16384, 16, 256, "repobench-p 16k B16"),
    ("syn_quote", 32768, 16, 512, "syn_quote 32k B16 gen512"),  # table only
]
TABLE_ONLY = {("syn_quote", 32768, 16, 512)}
LABELS = {"snapkv2048": "SnapKV-2048", "snapkv4096": "SnapKV-4096", "stream512": "StreamingLLM-512",
          "vqpL2+r128": "RQ2+r128", "hyb512pL2": "hybrid SnapKV512+RQ2"}


def tau_of(a, g):
    a = np.asarray(a, float)
    return np.where(a < 1, (1 - a ** (g + 1)) / np.maximum(1 - a, 1e-12), g + 1.0)


def alpha_of(tau, g):
    """Invert tau(alpha, gamma) (monotone in alpha on [0, 1]) by bisection."""
    if tau >= g + 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if tau_of(m, g) < tau else (lo, m)
    return 0.5 * (lo + hi)


def load_rows():
    rows = []
    for tag in FILES:
        for f in sorted(glob.glob(f"{RES}/llama8b_magicvq_{tag}_*B*_gen*.json")):
            d = json.load(open(f))
            m, cfg = d["metrics"], d["config"]
            B = cfg["B"]
            for r in m["rows"]:
                ar = m["procs"][r["proc"]]["ar_tok_s"] if "procs" in m else m["proc_ar"][r["proc"]]
                base, _, cb = r["config"].split(":")[0].partition("@")
                g = r["gamma"]
                am = sorted(r["aa_runs"], key=lambda x: x["tok_per_s"])[len(r["aa_runs"]) // 2]  # median rep, as S_meas
                t_ar = 1000.0 * B / ar
                t_round = am["wall_s"] / am["rounds"] * 1e3
                t_draft = am["draft_s"] / am["rounds"] * 1e3
                tau = am["tokens"] / (B * am["rounds"])
                c = t_draft / (g * t_ar)
                v = (t_round - t_draft) / t_ar
                rows.append(dict(file=os.path.basename(f), proc=r["proc"], task=cfg["task"], ctx=cfg["T"], B=B,
                                 gen=cfg["gen"], draft=base, cb=cb or "def", gamma=g, t_ar=t_ar, t_round=t_round,
                                 t_draft=t_draft, t_verify=am["verify_s"] / am["rounds"] * 1e3, tau=tau,
                                 tau_rec=r["aa_tau"], c=c, v=v, alpha=alpha_of(tau, g), s_meas=r["aa_speedup"],
                                 s_model=tau / (g * c + v), cotenant=r.get("cotenant", False)))
    return rows


def kind(r):
    if r["draft"].startswith("snapkv"):
        return "snapkv"
    if r["draft"].startswith("stream"):
        return "stream"
    return "vq_" + r["cb"]


def fit_v(rows):
    g = np.array([r["gamma"] for r in rows], float)
    v = np.array([r["v"] for r in rows])
    A = np.stack([np.ones_like(g), g], 1)
    (v0, v1), *_ = np.linalg.lstsq(A, v, rcond=None)
    res = v - (v0 + v1 * g)
    return v0, v1, float(np.max(np.abs(res)))


def smax(a, c, v0, v1):
    a = np.asarray(a, float)[..., None]
    s = tau_of(a, GMAP) / (GMAP * np.asarray(c, float)[..., None] + v0 + v1 * GMAP)
    return s.max(-1), GMAP[s.argmax(-1)]


def cross_gamma(rows):
    """alpha from gamma_a predicts S at gamma_b (same point, file, process, draft, codebook)."""
    out = []
    grp = {}
    for r in rows:
        grp.setdefault((r["file"], r["proc"], r["draft"], r["cb"]), []).append(r)
    for k, rs in grp.items():
        for ra in rs:
            for rb in rs:
                if ra["gamma"] == rb["gamma"]:
                    continue
                tp = float(tau_of(ra["alpha"], rb["gamma"]))
                sp = tp / (rb["gamma"] * rb["c"] + rb["v"])
                out.append(dict(key=k, ga=ra["gamma"], gb=rb["gamma"], a_a=ra["alpha"], a_b=rb["alpha"],
                                tau_pred=tp, tau_meas=rb["tau"], s_pred=sp, s_meas=rb["s_meas"],
                                err=sp / rb["s_meas"] - 1))
    return out


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=3)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def background(ax, v0, v1, regimes, xlim, ylim):
    cs = np.linspace(*xlim, 260)
    al = 1 - 10 ** -np.linspace(-np.log10(1 - ylim[0]), -np.log10(1 - ylim[1]), 300)  # uniform on the plotted scale
    Z = np.array([smax(al, c, v0, v1)[0] for c in cs]).T
    # S < 1 (slower than AR) = light gray wash; S >= 1 = surface. Contours in muted gray, break-even in ink.
    ax.contourf(cs, al, Z, levels=[0, 1], colors=["#ecebe6"], zorder=0.5)
    lev = [1.25, 1.5, 1.75]  # S_max < 2 everywhere in the plotted range
    cl = ax.contour(cs, al, Z, levels=lev, colors=[MUTED], linewidths=0.8, zorder=1)
    # contour labels near the left/bottom edge, away from the data labels (clabel snaps to the nearest contour)
    ax.clabel(cl, fmt=lambda x: f"{x:g}×", fontsize=8, colors=INK2, inline_spacing=3,
              manual=[(0.42, 0.815), (0.41, 0.93), (0.41, 0.975)])
    for (rv0, rv1, name) in regimes:
        Zr = np.array([smax(al, c, rv0, rv1)[0] for c in cs]).T
        ax.contour(cs, al, Zr, levels=[1], colors=[INK2], linewidths=0.8, linestyles=[(0, (3, 2))], zorder=1)
    ax.contour(cs, al, Z, levels=[1], colors=[INK], linewidths=1.8, zorder=1.2)


def best(rows, pt, k):
    rs = [r for r in rows if (r["task"], r["ctx"], r["B"], r["gen"]) == pt and kind(r) == k]
    return max(rs, key=lambda r: r["s_meas"]) if rs else None


def draw_points(ax, rows, vqkind, offsets, show_stream, v0, v1):
    plotted = []
    for task, ctx, B, gen, lab in POINTS:
        pt = (task, ctx, B, gen)
        if pt in TABLE_ONLY:
            continue
        s, q = best(rows, pt, "snapkv"), best(rows, pt, vqkind)
        if s is None or q is None:
            continue
        if show_stream:
            t = best(rows, pt, "stream")
            if t is not None:
                ax.plot(t["c"], t["alpha"], marker="D", ms=5.5, color=STREAM, alpha=0.55, mec=SURF, mew=1,
                        ls="none", zorder=3)
                ax.plot([t["c"], s["c"]], [t["alpha"], s["alpha"]], color=STREAM, alpha=0.35, lw=0.8, zorder=2)
        ax.annotate("", xy=(q["c"], q["alpha"]), xytext=(s["c"], s["alpha"]),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, shrinkA=5, shrinkB=6,
                                    mutation_scale=9), zorder=3)
        ax.plot(s["c"], s["alpha"], marker="o", ms=8, color=SNAP, mec=SURF, mew=2, ls="none", zorder=4)
        filled = vqkind == "vq_t2w"
        ax.plot(q["c"], q["alpha"], marker="s", ms=8, color=VQ if filled else SURF, mec=VQ if not filled else SURF,
                mew=2 if filled else 1.8, ls="none", zorder=4)
        tx, ty, ha = offsets[pt]  # label position in data coordinates (placed by looking at the render)
        ratio = q["s_meas"] / s["s_meas"]
        far = abs(tx - q["c"]) > 0.012 or abs(np.log10((1 - ty) / (1 - q["alpha"]))) > 0.08
        ax.annotate(f"{lab}\n{s['s_meas']:.2f}× → {q['s_meas']:.2f}× ({ratio:.2f})", xy=(q["c"], q["alpha"]),
                    xytext=(tx, ty), textcoords="data", ha=ha, va="center", fontsize=7.6, color=INK,
                    linespacing=1.15, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.7, shrinkA=2, shrinkB=5) if far else None)
        plotted.append((pt, lab, s, q))
    return plotted


def main():
    rows = load_rows()
    # rows that enter the figure / table (all points incl. table-only, all kinds)
    pts = [p[:4] for p in POINTS]
    use = [r for r in rows if (r["task"], r["ctx"], r["B"], r["gen"]) in pts]
    v0, v1, vres = fit_v(use)
    reg_a = [r for r in use if r["ctx"] == 32768 and r["B"] == 16]
    reg_b = [r for r in use if not (r["ctx"] == 32768 and r["B"] == 16)]
    fa, fb = fit_v(reg_a), fit_v(reg_b)
    regimes = [(fa[0], fa[1], "32k B16"), (fb[0], fb[1], "16k B16 / 32k B8")]
    ident = [abs(r["s_model"] / r["s_meas"] - 1) for r in use]
    cg = cross_gamma(use)

    xlim, ylim = (0.39, 0.70), (0.60, 0.997)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.patch.set_facecolor(SURF)
    for ax in axs:
        style(ax)
        background(ax, v0, v1, regimes, xlim, ylim)
        ax.set_xlim(*xlim)
        ax.set_yscale("function", functions=(lambda a: -np.log10(np.clip(1 - a, 1e-6, None)),
                                             lambda y: 1 - 10.0 ** -y))
        ax.set_ylim(*ylim)
        ax.set_yticks(YT)
        ax.set_yticklabels([f"{t:g}" for t in YT])
        ax.minorticks_off()
        ax.set_xlabel("draft cost  c = draft step / AR step  (measured)", color=INK2, fontsize=9.5)
    axs[0].set_ylabel("per-token acceptance  α  (derived from measured τ; axis ∝ −log(1−α))", color=INK2, fontsize=9.5)

    # label offsets in points from the VQ marker (tuned by looking at the render)
    off_a = {("syn_quote", 32768, 16, 256): (0.467, 0.9955, "left"),
             ("pg19", 32768, 16, 256): (0.456, 0.966, "left"),
             ("syn_quote", 32768, 8, 256): (0.595, 0.9945, "left"),
             ("hotpotqa_q", 16384, 16, 256): (0.612, 0.9885, "right"),
             ("repobench-p", 16384, 16, 256): (0.645, 0.9765, "left"),
             ("qmsum", 16384, 16, 256): (0.645, 0.960, "left"),
             ("gov_report", 16384, 16, 256): (0.645, 0.940, "left"),
             ("gov_report", 32768, 8, 256): (0.645, 0.905, "left")}
    pa = draw_points(axs[0], use, "vq_t2w", off_a, True, v0, v1)
    off_b = {("syn_quote", 32768, 16, 256): (0.468, 0.9835, "left"),
             ("pg19", 32768, 16, 256): (0.458, 0.935, "left"),
             ("syn_quote", 32768, 8, 256): (0.595, 0.988, "left"),
             ("hotpotqa_q", 16384, 16, 256): (0.634, 0.952, "left")}
    pb = draw_points(axs[1], use, "vq_def", off_b, False, v0, v1)
    axs[0].set_title("A. Best SnapKV → best VQ/hybrid, attention-trained codebook (t2w)",
                     loc="left", fontsize=10.5, color=INK, pad=8)
    axs[1].set_title("B. Same with the default codebook", loc="left",
                     fontsize=10.5, color=INK, pad=8)
    leg = [Line2D([], [], marker="o", ms=8, color=SNAP, mec=SURF, mew=2, ls="none", label="best SnapKV (2048/4096)"),
           Line2D([], [], marker="s", ms=8, color=VQ, mec=SURF, mew=2, ls="none", label="best VQ/hybrid, t2w codebook"),
           Line2D([], [], marker="s", ms=8, color=SURF, mec=VQ, mew=1.8, ls="none",
                  label="best VQ/hybrid, default codebook"),
           Line2D([], [], marker="D", ms=5.5, color=STREAM, alpha=0.55, ls="none", label="StreamingLLM-512 (LongBench only)"),
           Line2D([], [], color=INK, lw=1.8, label=f"break-even S = 1, v(γ) = {v0:.2f} + {v1:.3f}γ (all rows)"),
           Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)),
                  label=f"S = 1 with regime fits: 32k B16 v = {fa[0]:.2f}+{fa[1]:.3f}γ;  16k B16 / 32k B8 v = {fb[0]:.2f}+{fb[1]:.3f}γ"),
           Line2D([], [], color=MUTED, lw=0.8, label="model S_max contours (γ ≤ 8); gray wash = slower than AR")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, fontsize=8.2, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Gains come from α, not c: compressed-KV self-speculation, Llama-3.1-8B, 4×RTX 4090 TP=4 "
                 "(labels: measured all-active speedup SnapKV → VQ, and ratio)",
                 x=0.01, ha="left", fontsize=11, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(OUTP + ".png", dpi=300, facecolor=SURF)
    fig.savefig(OUTP + ".pdf", facecolor=SURF)

    # ---------------- table ----------------
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    L = ["# E7 c–α theory map: measured e2e points on the analytical speculative-decoding map", "",
         f"Figure: `results/E7/llama8b_magicvq_c_alpha_map.{{png,pdf}}`, script `experiments/26_magicvq/plot_c_alpha.py`.",
         "Data: `results/E7/llama8b_magicvq_{v6e2e,v7,v7_g6,v7mse}_*.json` (4×RTX 4090 TP=4, bf16 Llama-3.1-8B-Instruct, "
         "bf16 target KV, all-active accounting = headline; median-tok/s timed rep of 3).", "",
         "- **Measured:** T_AR (= 1000·B / AR tok/s of the same process), draft time per round, round time, τ "
         "(= all-active tokens / (B·rounds)), S_meas (all-active tok/s ÷ AR tok/s).",
         "- **Derived:** c = T_draft / (γ·T_AR); v = (T_round − T_draft) / T_AR (verify + accept + encode + host); "
         "α = root of τ(α, γ) = (1 − α^(γ+1)) / (1 − α) at the row's γ (i.i.d. acceptance model).",
         "- S_model = τ / (γc + v) from the row's own c, v, τ. This equals S_meas by construction (same wall clock), so "
         f"the error only checks record consistency: max |S_model/S_meas − 1| = {max(ident):.1e} over {len(ident)} rows.",
         f"- S_map = background value at (c, α): max over γ ∈ 1..8 of τ(α, γ) / (γc + v0 + v1·γ) with the pooled fit "
         f"v = {v0:.3f} + {v1:.4f}·γ (max residual {vres:.3f}). Regime fits: 32k B16 v = {fa[0]:.3f} + {fa[1]:.4f}·γ "
         f"(max res {fa[2]:.3f}); 16k B16 and 32k B8 v = {fb[0]:.3f} + {fb[1]:.4f}·γ (max res {fb[2]:.3f}). "
         "S_map can exceed S_meas because it also optimizes γ (and SnapKV was measured at γ ≤ 4 on LongBench).",
         "- Best = max S_meas over all measured budgets / γ / processes of the point (VQ/hybrid: RQ2+r128 and the "
         "SnapKV512+RQ2 hybrid, t2w codebook; panel B: default codebook).", ""]
    hdr = ("| point | draft (best) | γ | T_AR ms | c | v | τ | α | S_meas | S_map (γ*) | VQ ÷ SnapKV meas |",
           "|---|---|---|---|---|---|---|---|---|---|---|")
    for title, vk in (("Panel A: SnapKV vs VQ/hybrid t2w (and StreamingLLM-512)", "vq_t2w"),
                      ("Panel B: SnapKV vs VQ/hybrid default codebook", "vq_def")):
        L += [f"## {title}", "", *hdr]
        for task, ctx, B, gen, lab in POINTS:
            pt = (task, ctx, B, gen)
            s, q = best(use, pt, "snapkv"), best(use, pt, vk)
            if s is None or q is None:
                continue
            ks = [("snapkv", s), (vk, q)]
            if vk == "vq_t2w":
                t = best(use, pt, "stream")
                if t is not None:
                    ks.insert(0, ("stream", t))
            for k, r in ks:
                sm, gm = smax([r["alpha"]], r["c"], v0, v1)
                rr = f"**{q['s_meas'] / s['s_meas']:.3f}**" if k == vk else ""
                nm = LABELS.get(r["draft"], r["draft"]) + (f" [{r['cb']}]" if k.startswith("vq") else "")
                L.append(f"| {lab}{' (table only)' if pt in TABLE_ONLY else ''} | {nm} | {r['gamma']} | "
                         f"{r['t_ar']:.2f} | {f3(r['c'])} | {f3(r['v'])} | {r['tau']:.2f} | {f3(r['alpha'])} | "
                         f"{f3(r['s_meas'])} | {sm[0]:.3f} (γ{gm[0]}) | {rr} |")
        L.append("")
    # cross-gamma check
    errs = [abs(x["err"]) for x in cg]
    L += ["## Check: does one α predict another γ? (i.i.d.-acceptance model test)", "",
          f"For each draft / process with ≥ 2 measured γ: α from γ_a, τ and S predicted at γ_b with the measured c, v of "
          f"γ_b, compared with measured S at γ_b. {len(cg)} pairs: median |S error| {statistics.median(errs) * 100:.1f}%, "
          f"max {max(errs) * 100:.1f}%. Positive error = model overpredicts.", "",
          "| point file | proc | draft | cb | γ_a → γ_b | α(γ_a) | α(γ_b) | τ pred / meas | S pred / meas | error |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in sorted(cg, key=lambda x: (x["key"], x["ga"], x["gb"])):
        f, p, d, cb = x["key"]
        L.append(f"| {f.replace('llama8b_magicvq_', '').replace('.json', '')} | {p} | {LABELS.get(d, d)} | {cb} | "
                 f"{x['ga']} → {x['gb']} | {x['a_a']:.3f} | {x['a_b']:.3f} | {x['tau_pred']:.2f} / {x['tau_meas']:.2f} | "
                 f"{x['s_pred']:.3f} / {x['s_meas']:.3f} | {x['err'] * 100:+.1f}% |")
    # S_map vs S_meas summary over plotted rows
    pm = []
    for _, _, s, q in pa + pb:
        for r in (s, q):
            pm.append(smax([r["alpha"]], r["c"], v0, v1)[0][0] / r["s_meas"] - 1)
    L += ["", f"S_map / S_meas − 1 over the plotted markers: median {statistics.median(pm) * 100:+.1f}%, "
          f"range {min(pm) * 100:+.1f}% … {max(pm) * 100:+.1f}%.", ""]
    open(OUTP + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", OUTP + ".{png,pdf,md}")


if __name__ == "__main__":
    main()
