"""E8a: tables (markdown) and the scatter figure from analyze.py outputs.
Usage: /hai/anaconda3/bin/python3 experiments/29_accept_pred/report.py [--tag v1]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
E8 = os.path.join(REPO, "results", "E8")

KEY_TARGETS = ["tau4_snapkv4096", "alpha1_snapkv4096", "tau4_hyb512pL2", "tau4_vqpL2+r128", "logratio4_hyb_vs_snap4096",
               "logratio4_vq_vs_snap4096", "logratio4_hybt2w_vs_snap4096"]
KEY_FEATS = ["obs_cov4096_mean", "obs_cov512_mean", "obs_far4096_mean", "obs_covho4096_mean", "obs_cov4096_q10",
             "obs_ent_mean", "obs_logn90_mean", "obs_mfar128_mean", "obs_o_snap4096_mean", "obs_o_snap512_mean",
             "obs_o_vq_def_mean", "obs_o_hyb_def_mean", "obs_o_hyb_t2w_mean", "obs_d_snap4096_hyb_def",
             "obs_d_snap4096_vq_def", "P"]


def f3(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    R = json.load(open(os.path.join(E8, f"accept_pred_{a.tag}_analysis.json")))
    per = json.load(open(os.path.join(E8, f"accept_pred_{a.tag}_per_seq.json")))
    L = [f"# E8a accept_pred {a.tag}: is per-sequence draft acceptance predictable at prefill? (go/no-go)", ""]
    L += [f"- sequences: {R['n_rows']} with features, {R['n_kept']} kept (tf_agree >= 0.97 and prompt length = record); "
          f"excluded: {R['excluded']}",
          "- tau = tau(gamma=4) per sequence from the E7 lb_tau records (v4/v4b/v6); features from one target prefill "
          "(observation window = last 32 prompt queries) and, as a non-causal control, from the target's attention at "
          "the generated positions ('oracle').",
          "- within-task = Pearson of task-demeaned within-task ranks (pooled); mean task rho = n-weighted mean of per-task "
          "Spearman (tasks with n >= 6).", ""]
    for ctx, C in R["corr"].items():
        for kind in ("prefill", "oracle"):
            L += [f"## Correlations, ctx {ctx}, {kind} features", "",
                  "| feature | " + " | ".join(KEY_TARGETS) + " |", "|---" * (len(KEY_TARGETS) + 1) + "|"]
            for fe in KEY_FEATS:
                fe2 = fe if kind == "prefill" else fe.replace("obs_", "orc_")
                cells = []
                for t in KEY_TARGETS:
                    d = C[kind].get(f"{t}|{fe2}")
                    cells.append("-" if d is None else
                                 f"{f3(d['pooled_spearman'])} / {f3(d['within_spearman_ranks'])} / {f3(d['mean_task_spearman'])}")
                if any(c != "-" for c in cells):
                    L.append(f"| {fe2} | " + " | ".join(cells) + " |")
            L += ["", "cells: pooled Spearman / within-task (rank, pooled) / mean per-task Spearman", ""]
        # best within-task feature per target
        L += [f"### ctx {ctx}: strongest prefill features by |mean per-task Spearman|", "",
              "| target | feature | pooled | within (ranks) | mean task rho | per task |", "|---|---|---|---|---|---|"]
        for t in KEY_TARGETS:
            cand = [(k.split("|")[1], d) for k, d in C["prefill"].items() if k.startswith(t + "|") and k.split("|")[1] != "P"]
            cand = [c for c in cand if np.isfinite(c[1]["mean_task_spearman"])]
            for fe, d in sorted(cand, key=lambda c: -abs(c[1]["mean_task_spearman"]))[:3]:
                L.append(f"| {t} | {fe} | {f3(d['pooled_spearman'])} | {f3(d['within_spearman_ranks'])} | "
                         f"{f3(d['mean_task_spearman'])} | {d['per_task_spearman']} |")
        L.append("")
    L += ["## Does the prefill (obs-window) feature track the same feature at the generated positions?", "",
          "| ctx | obs -> oracle | pooled Spearman | within-task (ranks) | mean task rho |", "|---|---|---|---|---|"]
    for ctx, D in R["obs_vs_orc"].items():
        for k, d in D.items():
            L.append(f"| {ctx} | {k} | {f3(d['pooled_spearman'])} | {f3(d['within_spearman_ranks'])} | {f3(d['mean_task_spearman'])} |")
    L.append("")
    L += ["## Per-sequence draft selection (speedup = tau * T_AR / T_round, measured B16 TP4 step model, all-active)", ""]
    sm = R["step_model"]
    for ctx, s in sm.items():
        L.append(f"- step model ctx {ctx} ({s['source']}): T_AR {s['T_AR']:.2f} ms, t_draft/step "
                 + ", ".join(f"{k} {v:.2f}" for k, v in s["t_d"].items())
                 + f" ms, t_verify = {s['v0']:.2f} + {s['v1']:.2f}*gamma ms")
    L.append("")
    for key, S in R["selection"].items():
        L += [f"### {key} (n = {S['n']}, gamma per draft {S['gamma']})", "",
              f"- pooled mean speedup per draft: {S['mean_speedup']}; best single = {S['best_single']} {S['best_single_S']:.3f}",
              f"- oracle best-of-3 per sequence: {S['oracle3_S']:.3f} (**{100 * S['oracle3_gain_vs_best_single']:+.1f}%** vs best single); "
              f"task oracle (best draft per task): {S['task_oracle_S']:.3f} ({100 * S['task_oracle_gain_vs_best_single']:+.1f}%), "
              f"task best = {S['task_best']}",
              f"- upper bound, per-sequence oracle over (draft, gamma): {S['oracle_draft_gamma_S']:.3f} "
              f"({100 * S['oracle_draft_gamma_gain_vs_best_single']:+.1f}%)",
              f"- binary {S['binary_pair']}: best single {S['best_single2_S']:.3f}, oracle {S['oracle2_S']:.3f} "
              f"({100 * S['oracle2_gain_vs_best_single']:+.1f}%), share of sequences where the VQ-side draft wins {S['frac_vq_better']:.2f}", "",
              "| predictor (leave-one-task-out) | S | gain vs best single | share of oracle gain | accuracy |", "|---|---|---|---|---|"]
        for n, p in S["predictors"].items():
            L.append(f"| {n} | {p['S']:.3f} | {100 * p['gain_vs_best_single']:+.1f}% | {(f"{100 * p['recovered_frac_of_oracle_gain']:.0f}%" if S["oracle2_S"] - S["best_single2_S"] > 1e-3 else "n/a (oracle gain <0.1%)")} | {p['acc']:.2f} |")
        L += ["", "| task | n | mean speedup per draft | oracle3 | share VQ-side better |", "|---|---|---|---|---|"]
        for t, d in S["per_task"].items():
            L.append(f"| {t} | {d['n']} | {d['mean']} | {d['oracle3']} | {d['frac_vq_better']} |")
        L.append("")
    L += ["## Per-head heterogeneity (obs window; layer 0 excluded; quantiles across the 31x8 (layer, kv-head) per sequence, "
          "median over sequences)", "",
          "| ctx | stat | q10 | q50 | q90 | q90/q10 | head-rank stability same task | diff task | frac heads cov<0.9 | share of total err in worst 10% heads |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ctx, D in R["heads"].items():
        for s, d in D.items():
            L.append(f"| {ctx} | {s} | {d['across_heads_q10_median']:.3f} | {d['across_heads_q50_median']:.3f} | "
                     f"{d['across_heads_q90_median']:.3f} | {d['q90_over_q10_median']:.2f} | "
                     f"{d['head_rank_stability_same_task'] if d['head_rank_stability_same_task'] is None else round(d['head_rank_stability_same_task'], 2)} | "
                     f"{d['head_rank_stability_diff_task'] if d['head_rank_stability_diff_task'] is None else round(d['head_rank_stability_diff_task'], 2)} | "
                     f"{'-' if d['frac_heads_below_0.9'] is None else round(d['frac_heads_below_0.9'], 3)} | "
                     f"{'-' if d['share_of_total_from_worst_10pct_heads'] is None else round(d['share_of_total_from_worst_10pct_heads'], 3)} |")
    open(os.path.join(E8, f"accept_pred_{a.tag}_tables.md"), "w").write("\n".join(L) + "\n")

    # ---- figure: 2x2 scatter, colored by task (ctx 16k circles, 32k triangles)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = sorted({r["task"] for r in per})
    cmap = plt.get_cmap("tab10")
    col = {t: cmap(i % 10) for i, t in enumerate(tasks)}
    panels = [("obs_cov4096_mean", "tau4:snapkv4096", "prefill SnapKV-4096 coverage (obs window)", "tau(g4) SnapKV-4096"),
              ("orc_cov4096_mean", "tau4:snapkv4096", "oracle SnapKV-4096 coverage (generated queries)", "tau(g4) SnapKV-4096"),
              ("obs_d_snap4096_hyb_def", "lr", "prefill err(SnapKV-4096) - err(hybrid), after o_proj", "log tau(hyb512pL2) / tau(SnapKV-4096)"),
              ("orc_d_snap4096_hyb_def", "lr", "oracle err(SnapKV-4096) - err(hybrid), after o_proj", "log tau(hyb512pL2) / tau(SnapKV-4096)")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (fx, fy, xl, yl) in zip(axs.ravel(), panels):
        for r in per:
            y = np.log(r["tau4:hyb512pL2"] / r["tau4:snapkv4096"]) if fy == "lr" else r[fy]
            ax.scatter(r[fx], y, s=14, color=col[r["task"]], marker="o" if r["ctx"] == 16384 else "^", alpha=0.75)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        if fy == "lr":
            ax.axhline(0, color="k", lw=0.6)
    hs = [plt.Line2D([], [], marker="o", ls="", color=col[t], label=t) for t in tasks]
    hs += [plt.Line2D([], [], marker="o", ls="", color="k", mfc="none", label="16k"), plt.Line2D([], [], marker="^", ls="", color="k", mfc="none", label="32k")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("E8a: per-sequence acceptance vs prefill / oracle attention features (Llama-3.1-8B, E7 drafts)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(os.path.join(E8, f"accept_pred_{a.tag}_scatter.png"), dpi=130)
    print("wrote tables + figure")


if __name__ == "__main__":
    main()
