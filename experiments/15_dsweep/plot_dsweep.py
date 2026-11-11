"""PPL@4096 (wt2, c4) and KL(fp16||target) vs bpw incl. codebooks+scales for all d-sweep records
(results/E5/qwen05_dsweep_*.json): VQ target-only (T) per d, nested (N), GPTQ INTb. Also the same against index bpw.
Needs only json + matplotlib (system python3 has it; env nestvq does not).
Usage: python3 experiments/15_dsweep/plot_dsweep.py [--out results/E5/qwen05_dsweep_ppl_vs_bpw.png]
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(REPO, "results/E5/qwen05_dsweep_ppl_vs_bpw.png")
pts = []
for p in sorted(glob.glob(os.path.join(REPO, "results/E5/qwen05_dsweep_*.json"))):
    if p.endswith("_kd24.json"):  # eval-only add-on of the T run (same target)
        continue
    r = json.load(open(p)); m, c = r["metrics"], r["config"]
    L = c.get("L", 1)
    if c["cfg"].startswith("gptq"):
        grp = "GPTQ g128"
    else:
        grp = f"VQ d{c['d']} K{c['K']} {c.get('objective', '')}" if c.get("objective") == "T" else f"VQ d{c['d']} K{c['K']} nested"
    tq = m.get("target_quality", {})
    pts.append({"grp": grp, "label": c["cfg"] + ("" if c.get("objective", "T") in ("T", None) else "_" + c["objective"]), "x_incl": m["target_bpw"]["bpw_with_codebooks_scales"],
                "x_idx": m["target_bpw"]["index_bpw"], "wt2": m["ppl_4096"].get(f"wikitext2_k{L}"),
                "c4": m["ppl_4096"].get(f"c4_k{L}"), "kl": tq.get("wikitext2", {}).get("kl_fp16_target")})
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for row, xk, xl in ((0, "x_incl", "bpw incl. codebooks + scales"), (1, "x_idx", "index bpw")):
    for col, yk, yl in ((0, "wt2", "PPL@4096 wikitext2"), (1, "c4", "PPL@4096 c4"), (2, "kl", "KL(fp16||target) wt2")):
        a = ax[row, col]
        for g in sorted({q["grp"] for q in pts}):
            s = sorted([q for q in pts if q["grp"] == g and q[yk] is not None], key=lambda q: q[xk])
            if not s:
                continue
            a.plot([q[xk] for q in s], [q[yk] for q in s], "o-" if len(s) > 1 and "nested" not in g else "o", label=g)
            for q in s:
                a.annotate(q["label"], (q[xk], q[yk]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        a.set_yscale("log"); a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=7)
fig.suptitle("Qwen2.5-0.5B-Instruct claim-1 d-sweep (no FT, Hadamard)")
fig.tight_layout(); fig.savefig(out, dpi=130)
print(out)
