"""E7 magicvq v6e2e: v6 draft-trained codebook (t2w) vs default ('p' greedy) codebook, fully real e2e, both batch
accountings, after the deterministic LUT-kernel fix (csrc/vqattn_lut_rag.cu q.mu reduction).

Per point one run (e2e_real.py --accounting both --cbx t2w=...): same AR baseline / prompts / plans; configs
'<draft>:<g>' use the default codebook, '<draft>@t2w:<g>' the v6 t2w codebook.
Columns: static and all-active measured speedups vs the AR of the same run; best VQ/hybrid / best SnapKV per codebook
and accounting; default-codebook rows vs v5aa (pre-fix kernel); measured tau ratio t2w / default (static tau = mean
accepted+1 over all rounds of the static run) vs the v6 computed tau ratio (results/E7/llama8b_magicvq_v6_tables.md,
same task / ctx / gamma, '-' when v6 has no such gamma).
Usage: python experiments/26_magicvq/e2e_summarize_v6e2e.py [--tag v6e2e] [--save]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e2e_summarize_aa import LABEL, OUT, RES, REPO, analyze, v5_derived  # noqa: E402

sys.path.insert(0, REPO)
V6NAME = {"vqpL2+r128": "vqpt2wL2+r128", "hyb512pL2": "hyb512pt2wL2"}


def v6_ratios():
    """{(task, ctx, trained draft): {3: r, 4: r, 6: r}} from the v6 tau table."""
    d = {}
    for line in open(f"{RES}/llama8b_magicvq_v6_tables.md"):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 9 and c[1].isdigit() and c[2] in V6NAME.values():
            d[(c[0], int(c[1]), c[2])] = {3: float(c[5]), 4: float(c[6].split()[0]), 6: float(c[7].split()[0])}
    return d


def v5aa_rows(p):
    f = f"{RES}/llama8b_magicvq_v5aa_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}.json"
    if not os.path.exists(f):
        return {}
    return {r["config"]: r for r in json.load(open(f))["metrics"]["rows"]}


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def lab(r):
    return LABEL.get(r["draft"], r["draft"])


def best(rows, k):
    return max(rows, key=lambda r: r[k]) if rows else None


def tables(pts, tag):
    L = [f"# E7 magicvq {tag}: v6 draft-trained codebook (t2w) vs default codebook, measured e2e, static and all-active "
         "(TP=4 RTX 4090, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, v5 prompts / drafts / gammas)", "",
         "- Kernel: deterministic VQ LUT draft kernel (fixed-order q.mu reduction replaces the shared-memory float "
         "atomicAdd; csrc/vqattn_lut_rag.cu). Default-codebook rows are therefore re-measured; 'v5aa' columns are the "
         "pre-fix values.",
         "- One run per point (e2e_real.py --accounting both --cbx t2w=cb_d8K256x2_post_cn_t2w.pt): one AR baseline, "
         "default ('def', cb_d8K256x2_post_cn_greedy.pt) and t2w drafts in the same process; static and all-active reps "
         "interleaved, warm-up excluded, 3 timed reps each, median.",
         "- static: every row exactly gen tokens (finished rows idle). all-active (headline, MagicDec accounting): no row "
         "frozen, stop when the longest row has >= gen tokens, all tokens / decode wall.",
         "- tau ratio meas = tau_static(t2w) / tau_static(def) for the same draft and gamma (same prompts, same AR "
         "target trajectory up to divergence); v6 computed = paired tau ratio of results/E7/llama8b_magicvq_v6_tables.md "
         "(same task / ctx / gamma; v6 used more samples and a replay-style tau eval, not this batch).", ""]
    summ = ["## Summary: best VQ/hybrid / best SnapKV (measured)", "",
            "| point | AR tok/s | static: SnapKV / VQ def / VQ t2w | static ratio def / t2w | all-active: SnapKV / VQ def / "
            "VQ t2w | **all-active ratio def / t2w** | v5aa aa ratio (pre-fix) |",
            "|---|---|---|---|---|---|---|"]
    tau = ["## tau: t2w vs default codebook (measured static tau ratio vs v6 computed ratio)", "",
           "| point | draft | gamma | tau static def -> t2w | tau aa def -> t2w | meas ratio (static) | v6 computed ratio | "
           "aa speedup def -> t2w |", "|---|---|---|---|---|---|---|---|"]
    v6r = v6_ratios()
    body = []
    for p in pts:
        v5 = v5aa_rows(p)
        body += [f"## {p['name']}", "",
                 f"AR {p['ar_tok_s']:.1f} tok/s (step {p['ar_step_ms']:.2f} ms){' *co-tenant*' if p['ar_cotenant'] else ''}"
                 f" (v5aa AR {p['v5aa_ar']:.1f}); co-tenants on GPUs 4-7 at start: {p['cotenants_start'] or 'none'}, end: "
                 f"{p['cotenants_end'] or 'none'}; any config flagged: {'YES (rows marked *)' if p['cotenant_any'] else 'no'}",
                 "",
                 "| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | "
                 "aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | "
                 "aa change vs v5aa |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["rows"]:
            a0, a1 = (x / p["proc_ar"][r["proc"]] * p["ar_tok_s"] for x in r["aa_tok_s_min_max"])
            o = v5.get(r["config"]) if r["cb"] == "def" else None
            body.append(
                f"| {lab(r)}{' *' if r['cotenant'] else ''} | {r['cb']}{' (sep. run)' if r['proc'] != 'main' else ''} | {r['gamma']} | {r['static_speedup']:.3f} | "
                f"**{r['aa_speedup']:.3f}** ({a0 / p['ar_tok_s']:.3f}-{a1 / p['ar_tok_s']:.3f}) | "
                f"{r['tau_static']:.2f} / {r['aa_tau']:.2f} | {r['static_round_ms']:.1f} / {r['aa_round_ms']:.1f} | "
                f"{r['aa_n_rows_identical_overlap']}/{p['B']} | {'yes' if r['aa_rep_ngen_differs'] else 'no'} | "
                + (f"{o['static_speedup']:.3f} / {o['aa_speedup']:.3f} | {r['aa_speedup'] / o['aa_speedup'] - 1:+.2%} |"
                   if o else "- | - |"))
        snap_all = [r for r in p["rows"] if r["draft"].startswith("snapkv")]
        snap = [r for r in snap_all if r["proc"] == "main"]
        if len(p["proc_ar"]) > 1:
            body += ["", "*sep. run*: t2w drafts measured in a separate process "
                     f"(AR {p['proc_ar']['t']:.1f} tok/s there; speedups vs that AR); the summary t2w ratio uses the "
                     "best SnapKV of that same process.", ""]
        cells = {}
        for k in ("static_speedup", "aa_speedup"):
            bs = best(snap, k)
            bd = best([r for r in p["rows"] if r["cb"] == "def" and not r["draft"].startswith("snapkv")], k)
            bt = best([r for r in p["rows"] if r["cb"] == "t2w"], k)
            if bt is not None and bt["proc"] != "main":  # separate process: compare with its own SnapKV rows
                bs_t = best([r for r in snap_all if r["proc"] == bt["proc"]], k)
                bt = dict(bt, **{k: bt[k] * bs[k] / bs_t[k]})  # expressed relative to the main-run best SnapKV
            cells[k] = (bs, bd, bt)
        p["summary"] = {k: {"snapkv": [lab(b[0]), b[0]["gamma"], b[0][k]],
                            "vq_def": [lab(b[1]), b[1]["gamma"], b[1][k]],
                            "vq_t2w": [lab(b[2]), b[2]["gamma"], b[2][k]] if b[2] else None,
                            "ratio_def": b[1][k] / b[0][k], "ratio_t2w": b[2][k] / b[0][k] if b[2] else None}
                        for k, b in cells.items()}
        v5best = None
        if v5:
            v5r = list(v5.values())
            v5best = (best([r for r in v5r if not r["draft"].startswith("snapkv")], "aa_speedup")["aa_speedup"] /
                      best([r for r in v5r if r["draft"].startswith("snapkv")], "aa_speedup")["aa_speedup"])

        def cell(k):
            bs, bd, bt = cells[k]
            s = f"{bs[k]:.3f} (g{bs['gamma']}) / {bd[k]:.3f} ({lab(bd)} g{bd['gamma']}) / "
            s += f"{bt[k]:.3f} ({lab(bt)} g{bt['gamma']})" if bt else "-"
            return s, f"{bd[k] / bs[k]:.3f} / {bt[k] / bs[k]:.3f}" if bt else f"{bd[k] / bs[k]:.3f} / -"
        cs, ca = cell("static_speedup"), cell("aa_speedup")
        summ.append(f"| {p['name']} | {p['ar_tok_s']:.1f} | {cs[0]} | {cs[1]} | {ca[0]} | **{ca[1]}** | {f3(v5best)} |")
        body += ["", f"best VQ/hybrid / best SnapKV (def / t2w): static {cs[1]}, all-active **{ca[1]}**", ""]
        byc = {(r["draft"], r["gamma"], r["cb"]): r for r in p["rows"]}
        for (dr, g, cb), r in byc.items():
            if cb != "def" or (dr, g, "t2w") not in byc:
                continue
            t = byc[(dr, g, "t2w")]
            v6 = v6r.get((p["task"], p["ctx"], V6NAME.get(dr)), {}).get(g)
            mr = t["tau_static"] / r["tau_static"]
            r["tau_ratio_t2w_meas"], r["tau_ratio_t2w_v6"] = mr, v6
            tau.append(f"| {p['name']} | {lab(r)} | {g} | {r['tau_static']:.2f} -> {t['tau_static']:.2f} | "
                       f"{r['aa_tau']:.2f} -> {t['aa_tau']:.2f} | {mr:.3f} | {f3(v6)} | "
                       f"{r['aa_speedup']:.3f} -> {t['aa_speedup']:.3f} |")
    return "\n".join(L + summ + [""] + tau + [""] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6e2e")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    v5d = v5_derived()
    order = {"syn_quote": 0, "hotpotqa_q": 1, "pg19": 2}
    files = sorted(glob.glob(f"{OUT}/run_{a.tag}_*.json")) + sorted(glob.glob(f"{OUT}/run_{a.tag}g512_*.json"))
    pts = [analyze(f, v5d) for f in files]
    for p in pts:
        p["proc_ar"] = {"main": p["ar_tok_s"]}
        for r in p["rows"]:
            r["proc"] = "main"
    # separate t2w processes (tag + 't', e.g. pg19: the 10-draft process OOMs in graph capture): rows keep speedups
    # vs the AR of their own process; their t2w ratio uses the SnapKV rows of the same process
    for f in sorted(glob.glob(f"{OUT}/run_{a.tag}t_*.json")):
        q = analyze(f, v5d)
        p = next(x for x in pts if (x["task"], x["ctx"], x["B"], x["gen"]) == (q["task"], q["ctx"], q["B"], q["gen"]))
        p["proc_ar"]["t"] = q["ar_tok_s"]
        p["raw"] += f"; {q['raw']}"
        p["cotenant_any"] = p["cotenant_any"] or q["cotenant_any"]
        for r in q["rows"]:
            r["proc"] = "t"
            p["rows"].append(r)
    for p in pts:
        for r in p["rows"]:
            r["cb"] = "t2w" if "@t2w" in r["draft"] else "def"
            r["draft"] = r["draft"].split("@")[0]
            r.pop("v5_static_measured", None), r.pop("v5_aa_derived", None)
        v5 = v5aa_rows(p)
        f = f"{RES}/llama8b_magicvq_v5aa_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}.json"
        p["v5aa_ar"] = json.load(open(f))["metrics"]["ar_tok_s"] if v5 else float("nan")
    pts.sort(key=lambda p: (order.get(p["task"], 9), -p["ctx"], -p["B"], p["gen"]))
    md = tables(pts, a.tag)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_{a.tag}_tables.md", "w").write(md + "\n")
        from tools.results import save_result
        for p in pts:
            save_result("E7", f"llama8b_magicvq_{a.tag}_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), fully real "
                                "e2e self-spec (e2e_real.py --accounting both --cbx t2w=...), deterministic LUT kernel",
                                "B": p["B"], "T": p["ctx"], "gen": p["gen"], "task": p["task"],
                                "configs": [r["config"] for r in p["rows"]],
                                "cb_default": "/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_greedy.pt",
                                "cb_t2w": "/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt"},
                        metrics={k: p[k] for k in ("ar_tok_s", "ar_step_ms", "ar_tok_s_runs", "ar_cotenant", "rows",
                                                   "summary", "v5aa_ar", "proc_ar", "cotenant_any", "cotenants_start",
                                                   "cotenants_end", "nvsmi_apps_start", "nvsmi_apps_end")},
                        model_dir="/data/models/llama3.1_8b_instruct",
                        notes=f"raw: {p['raw']}; static and all-active both measured wall clock in one run")


if __name__ == "__main__":
    main()
