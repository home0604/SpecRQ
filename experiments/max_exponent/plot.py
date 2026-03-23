"""
Max Exponent 실험 결과 시각화.

사용법:
  python plot_max_exponent.py                          # max_exp_results.json 사용
  python plot_max_exponent.py --input my_results.json  # 다른 파일
  python plot_max_exponent.py --output fig.pdf          # PDF 출력
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    args = data["args"]
    results = data["results"]  # [sample][layer]{stats}
    # flatten to per-layer list with sample index
    flat = []
    for s_idx, sample in enumerate(results):
        for l_idx, stats in enumerate(sample):
            flat.append({"sample": s_idx, "layer": l_idx, **stats})
    num_layers = len(results[0])
    num_samples = len(results)
    return args, results, flat, num_layers, num_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="max_exp_results.json")
    parser.add_argument("--output", default="max_exponent_plots.png")
    opts = parser.parse_args()

    args, results, flat, num_layers, num_samples = load_results(opts.input)
    layers = np.arange(num_layers)

    model_name = args.get("model", "unknown").split("/")[-1]
    ctx_len = args.get("context_length", "?")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Max Exponent Analysis — {model_name} (ctx={ctx_len}, {num_samples} samples)",
                 fontsize=14, fontweight="bold")

    # ── 1. Per-layer error distribution (mean ± p90/p99) ──
    ax = axes[0, 0]
    for s_idx, sample in enumerate(results):
        means = [s["error_mean"] for s in sample]
        p90s = [s["error_p90"] for s in sample]
        p99s = [s["error_p99"] for s in sample]
        alpha = 0.6 if num_samples > 1 else 1.0
        label = f"sample {s_idx}" if num_samples > 1 else None
        ax.plot(layers, means, "o-", markersize=3, alpha=alpha, label=label)
        ax.fill_between(layers, means, p90s, alpha=0.1)
        ax.fill_between(layers, p90s, p99s, alpha=0.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|target_max − draft_max|")
    ax.set_title("Per-row error by layer\n(line=mean, light=p90, lighter=p99)")
    if num_samples > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Tight bound error vs actual error (per layer, sample-averaged) ──
    ax = axes[0, 1]
    avg_error = np.array([np.mean([results[s][l]["error_mean"] for s in range(num_samples)])
                          for l in range(num_layers)])
    avg_bound = np.array([np.mean([results[s][l]["tight_err_mean"] for s in range(num_samples)])
                          for l in range(num_layers)])
    ax.bar(layers - 0.2, avg_bound, width=0.4, label="tight bound err", color="C1", alpha=0.7)
    ax.bar(layers + 0.2, avg_error, width=0.4, label="actual err", color="C0", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Error")
    ax.set_title("Tight bound vs actual error (mean)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Overestimate ratio per layer ──
    ax = axes[0, 2]
    avg_ratio = [np.mean([results[s][l]["overestimate_ratio_mean"] for s in range(num_samples)])
                 for l in range(num_layers)]
    med_ratio = [np.mean([results[s][l]["overestimate_ratio_p50"] for s in range(num_samples)])
                 for l in range(num_layers)]
    ax.bar(layers, avg_ratio, alpha=0.5, label="mean ratio", color="C3")
    ax.bar(layers, med_ratio, alpha=0.7, label="median ratio", color="C4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("bound_err / actual_err")
    ax.set_title("Overestimate ratio by layer")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. δk: theory vs actual (per layer) ──
    ax = axes[1, 0]
    for s_idx, sample in enumerate(results):
        dk_theory = [s["delta_k_theory_max"] for s in sample]
        dk_actual = [s["delta_k_actual_max"] for s in sample]
        alpha = 0.6 if num_samples > 1 else 1.0
        ax.plot(layers, dk_theory, "s--", markersize=4, alpha=alpha, color=f"C{s_idx}",
                label=f"theory (s{s_idx})" if s_idx == 0 else None)
        ax.plot(layers, dk_actual, "o-", markersize=4, alpha=alpha, color=f"C{s_idx}",
                label=f"actual (s{s_idx})" if s_idx == 0 else None)
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖δk‖_∞ max")
    ax.set_title("δk theory (8.5·scale) vs actual max")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 5. ‖q‖₁ distribution per layer ──
    ax = axes[1, 1]
    avg_q_mean = [np.mean([results[s][l]["q_l1_mean"] for s in range(num_samples)])
                  for l in range(num_layers)]
    avg_q_p50 = [np.mean([results[s][l]["q_l1_p50"] for s in range(num_samples)])
                 for l in range(num_layers)]
    avg_q_p99 = [np.mean([results[s][l]["q_l1_p99"] for s in range(num_samples)])
                 for l in range(num_layers)]
    ax.fill_between(layers, avg_q_p50, avg_q_p99, alpha=0.2, color="C2", label="p50–p99")
    ax.plot(layers, avg_q_mean, "o-", markersize=3, color="C2", label="mean")
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖q‖₁")
    ax.set_title("Query L1 norm by layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 6. Error percentile histogram (all layers aggregated) ──
    ax = axes[1, 2]
    percentiles = ["error_p50", "error_p90", "error_p95", "error_p99", "error_max"]
    labels = ["p50", "p90", "p95", "p99", "max"]
    vals = [np.mean([f[p] for f in flat]) for p in percentiles]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("|target_max − draft_max|")
    ax.set_title("Error percentiles (all layers avg)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(opts.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {opts.output}")
    plt.show()


if __name__ == "__main__":
    main()
