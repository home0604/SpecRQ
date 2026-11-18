"""Analytical speculative-decoding speedup vs draft cost c and per-token acceptance alpha.
S(c, alpha) = max_gamma tau(gamma, alpha) / (gamma*c + v(gamma)),  tau = (1 - alpha^(gamma+1)) / (1 - alpha)
c = T_draft_step / T_AR_step,  v(gamma) = T_verify(gamma+1 tokens) / T_AR_step,  alpha i.i.d. per token (greedy or sampling).
Upper bound: alpha -> 1 gives S <= (gamma+1)/(gamma*c+v) -> 1/c.  Break-even at gamma=1 with v=1: alpha = c.
Usage: /hai/anaconda3/bin/python3 experiments/28_sd_theory/breakeven.py  (writes results/E7/sd_theory_*.{png,md})"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = np.arange(1, 33)
VS = {"v1": ("verify = 1 AR step (ideal, memory-bound)", lambda g: np.ones_like(g, float)),
      "v105": ("verify = 1 + 0.05γ AR steps (≈ our 4×4090, B16×32k)", lambda g: 1.0 + 0.05 * g)}


LEV = [0, 1, 1.25, 1.5, 2, 3, 5, 100]
COLS = ["#e5e5e5", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]  # gray = slower than AR


def S(a, c, v):
    a = np.asarray(a, float)[..., None]
    tau = np.where(a < 1, (1 - a ** (G + 1)) / np.maximum(1 - a, 1e-12), G + 1)
    s = tau / (G * c + v(G))
    return s.max(-1), G[s.argmax(-1)]


def table(v):
    al = np.linspace(0, 0.999, 4000)
    rows = []
    for c in [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        s, _ = S(al, c, v)
        need = [f"{al[np.argmax(s >= t)]:.2f}" if (s >= t).any() else "—" for t in (1.0, 1.5, 2.0, 3.0)]
        s9, g9 = S([0.9], c, v); s8, g8 = S([0.8], c, v)
        rows.append(f"| {c} | {need[0]} | {need[1]} | {need[2]} | {need[3]} | {s8[0]:.2f} (γ{g8[0]}) | {s9[0]:.2f} (γ{g9[0]}) | {1/c:.1f} |")
    return "\n".join(["| c | α for 1.0× | α for 1.5× | α for 2× | α for 3× | S at α=0.8 | S at α=0.9 | bound 1/c |",
                      "|---|---|---|---|---|---|---|---|"] + rows)


def main():
    md = ["# Analytical spec-decoding speedup vs draft cost c and acceptance α", "",
          "S = max_γ τ(γ,α)/(γc + v(γ)), τ = (1−α^(γ+1))/(1−α); c = draft step / AR step; v = verify step / AR step.",
          "α = per-token acceptance (i.i.d. model); τ = tokens per round (MagicDec Ω). Upper bound S ≤ 1/c.", ""]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cs = np.linspace(0.01, 0.9, 300); al = np.linspace(0.3, 0.99, 300)
    for ax, (k, (title, v)) in zip(axs, VS.items()):
        Z = np.array([S(al, c, v)[0] for c in cs]).T
        cf = ax.contourf(cs, al, Z, levels=LEV, colors=COLS)
        cl = ax.contour(cs, al, Z, levels=[1, 1.25, 1.5, 2, 3], colors="k", linewidths=[2, 1, 1, 1, 1])
        ax.clabel(cl, fmt=lambda x: f"{x:g}×", fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_xlabel("draft cost c = T_draft step / T_AR step")
        ax.grid(alpha=0.2)
        md += [f"## {title}", "", table(v), ""]
    axs[0].set_ylabel("per-token acceptance α")
    cb = fig.colorbar(cf, ax=axs, ticks=[0.5, 1.125, 1.375, 1.75, 2.5, 4, 52.5]); cb.ax.set_yticklabels(["<1×", "1–1.25×", "1.25–1.5×", "1.5–2×", "2–3×", "3–5×", ">5×"]); cb.set_label("best-γ speedup S (analytical)")
    fig.suptitle("Speculative decoding speedup (formula, not measured): thick line = break-even 1×", fontsize=11)
    fig.savefig("results/E7/sd_theory_speedup_vs_c_alpha.png", dpi=150, bbox_inches="tight")
    open("results/E7/sd_theory_tables.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
