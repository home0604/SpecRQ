"""E7 magicvq v5aa: static vs MEASURED all-active batch accounting (e2e_real.py --accounting both).

Per (point, config), all from one run (shared plans / graphs / AR baseline):
  static measured     : B*gen / decode wall (finished rows idle), speedup vs AR tok/s of the same run.
  all-active measured : all tokens of all rows / decode wall, batch stops when the longest row has >= gen tokens.
  all-active derived  : tau_static x T_AR / T_round_static of this run (the v5_accounting formula), and the value of
                        results/E7/llama8b_magicvq_v5_accounting.md (derived from the v5 run).
Usage: python experiments/26_magicvq/e2e_summarize_aa.py --tag v5aa [--save]
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
OUT = "/data/lsh/results_accept/magicvq/e2e_real"
RES = os.path.join(REPO, "results", "E7")
LABEL = {"snapkv2048": "SnapKV-2048", "snapkv4096": "SnapKV-4096", "vqpL2+r128": "RQ2+r128",
         "hyb512pL2": "SnapKV512+RQ2 hybrid"}
med = statistics.median


def v5_derived():
    """{(section name, label, gamma): (static measured v5, all-active derived v5)} from the v5 accounting table."""
    d, sec = {}, None
    for l in open(f"{RES}/llama8b_magicvq_v5_accounting.md"):
        if l.startswith("## "):
            sec = l[3:].strip()
            continue
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if sec and len(c) == 6 and c[1].isdigit():
            d[(sec, c[0], int(c[1]))] = (float(c[4]), float(c[5]))
    return d


def cof(r):
    b, a = r.get("co_before") or {}, r.get("co_after") or {}
    return bool(b.get("cotenant") or a.get("cotenant")), (b.get("others") or []) + (a.get("others") or [])


def analyze(path, v5d):
    R = json.load(open(path))
    a = R["args"]
    task, ctx, B, gen = a["task"], a["ctx"], a["B"], a["gen"]
    sec = f"{task} ctx {ctx}, B={B}, gen {gen}"
    ar = R["configs"]["ar"]
    ar_tps = med([x["tok_per_s"] for x in ar["runs"]])
    t_ar = ar["step_ms"]
    rows = []
    for c, r in R["configs"].items():
        if c == "ar":
            continue
        name, g = c.split(":")
        g = int(g)
        st = r["runs"]
        aa = r["all_active"]
        s_tps = med([x["tok_per_s"] for x in st])
        a_tps = med([x["tok_per_s"] for x in aa["runs"]])
        acc = [x + 1 for row in r["acc_log"] for x in row if x >= 0]
        tau_s = sum(acc) / len(acc)
        ms = sorted(st, key=lambda x: x["wall_s"])[len(st) // 2]
        t_round = ms["wall_s"] / ms["rounds"] * 1e3
        am = sorted(aa["runs"], key=lambda x: x["wall_s"])[len(aa["runs"]) // 2]
        v5 = v5d.get((sec, LABEL.get(name, name), g))
        fd = aa["first_divergence_vs_ar"]
        co, apps = cof(r)
        rows.append({
            "config": c, "draft": name, "gamma": g,
            "static_tok_s": s_tps, "static_tok_s_min_max": [min(x["tok_per_s"] for x in st), max(x["tok_per_s"] for x in st)],
            "static_speedup": s_tps / ar_tps, "static_rounds": ms["rounds"], "tau_static": tau_s,
            "static_round_ms": t_round,
            "aa_tok_s": a_tps, "aa_tok_s_min_max": [min(x["tok_per_s"] for x in aa["runs"]),
                                                    max(x["tok_per_s"] for x in aa["runs"])],
            "aa_speedup": a_tps / ar_tps, "aa_rounds": am["rounds"], "aa_tokens": am["tokens"],
            "aa_tau": aa["tau"], "aa_round_ms": am["wall_s"] / am["rounds"] * 1e3,
            "aa_ngen_min_max": [min(am["ngen"]), max(am["ngen"])], "aa_ngen": am["ngen"],
            "aa_rep_ngen_differs": aa.get("rep_ngen_differs"),
            "aa_per_round_ms": {k: am[k] / am["rounds"] * 1e3 for k in ("draft_s", "verify_s", "accept_s", "encode_s",
                                                                       "host_s")},
            "aa_runs": aa["runs"], "static_runs": st,
            "aa_derived_this_run": tau_s * t_ar / t_round,
            "v5_static_measured": v5[0] if v5 else None, "v5_aa_derived": v5[1] if v5 else None,
            "aa_first_divergence_vs_ar": fd, "aa_n_rows_identical_overlap": sum(1 for f in fd if f is None),
            "aa_match_vs_ar_overlap_mean": sum(aa["match_vs_ar_overlap"]) / B,
            "cotenant": co, "cotenant_apps": apps})
    co_any = R.get("cotenant_any", False)
    return {"name": sec, "task": task, "ctx": ctx, "B": B, "gen": gen, "ar_tok_s": ar_tps, "ar_step_ms": t_ar,
            "ar_tok_s_runs": [x["tok_per_s"] for x in ar["runs"]], "ar_cotenant": cof(ar)[0], "rows": rows,
            "cotenant_any": co_any, "cotenants_start": R["cotenants_start"], "cotenants_end": R["cotenants_end"],
            "nvsmi_apps_start": R["nvsmi_apps_start"], "nvsmi_apps_end": R["nvsmi_apps_end"], "raw": path}


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def tables(pts, tag):
    L = [f"# E7 magicvq {tag}: static vs MEASURED all-active batch accounting (TP=4 RTX 4090, GPUs 4-7, bf16 "
         "Llama-3.1-8B-Instruct, v5 prompts / drafts / gammas / default codebooks)", "",
         "- One run per point (e2e_real.py --accounting both): same AR baseline, plans, CUDA graphs; static and "
         "all-active reps interleaved (warm-up of each excluded, 3 timed reps each, median).",
         "- **static (measured):** every row makes exactly gen tokens; finished rows idle until the slowest row is done. "
         "Speedup = B*gen/wall vs AR tok/s.",
         "- **all-active (measured):** no row is frozen; the batch stops when the LONGEST row has >= gen new tokens "
         "(EOS ignored); throughput = all tokens of all rows / decode wall (MagicDec official accounting, minus EOS stop).",
         "- **all-active derived:** tau_static x T_AR / T_round_static, (a) from this run, (b) v5 value from "
         "results/E7/llama8b_magicvq_v5_accounting.md (v5 run).",
         "- AR: every row gen tokens (no stragglers). Speedups vs the AR tok/s of the same run.", ""]
    summ = ["## Summary: best VQ/hybrid / best SnapKV", "",
            "| point | static meas: SnapKV / VQ / ratio | all-active meas: SnapKV / VQ / ratio | all-active derived v5: SnapKV / VQ / ratio |",
            "|---|---|---|---|"]
    body = []
    for p in pts:
        body += [f"## {p['name']}", "",
                 f"AR {p['ar_tok_s']:.1f} tok/s (step {p['ar_step_ms']:.2f} ms){' *co-tenant*' if p['ar_cotenant'] else ''}; "
                 f"co-tenants on GPUs 4-7 at start: {p['cotenants_start'] or 'none'}, end: {p['cotenants_end'] or 'none'}; "
                 f"any config flagged: {'YES (rows marked *)' if p['cotenant_any'] else 'no'}", "",
                 "| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | "
                 "tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | "
                 "aa rows identical to AR on overlap | aa per-row lengths vary across reps |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["rows"]:
            a0, a1 = r["aa_tok_s_min_max"]
            body.append(
                f"| {LABEL.get(r['draft'], r['draft'])}{' *' if r['cotenant'] else ''} | {r['gamma']} | "
                f"{r['static_speedup']:.3f} | **{r['aa_speedup']:.3f}** ({a0 / p['ar_tok_s']:.3f}-{a1 / p['ar_tok_s']:.3f}) | "
                f"{r['aa_derived_this_run']:.3f} | {f3(r['v5_aa_derived'])} | {r['tau_static']:.2f} / {r['aa_tau']:.2f} | "
                f"{r['static_rounds']} / {r['aa_rounds']} | {r['aa_tokens']} ({r['aa_ngen_min_max'][0]}-{r['aa_ngen_min_max'][1]}) | "
                f"{r['static_round_ms']:.1f} / {r['aa_round_ms']:.1f} | {r['aa_n_rows_identical_overlap']}/{p['B']} | {'yes' if r['aa_rep_ngen_differs'] else 'no'} |")
        snap = [r for r in p["rows"] if r["draft"].startswith("snapkv")]
        vq = [r for r in p["rows"] if not r["draft"].startswith("snapkv")]
        cells = []
        for k in ("static_speedup", "aa_speedup", "v5_aa_derived"):
            if any(r[k] is None for r in p["rows"]):
                cells.append("-")
                continue
            bs, bv = max(snap, key=lambda r: r[k]), max(vq, key=lambda r: r[k])
            cells.append(f"{bs[k]:.3f} (g{bs['gamma']}) / {bv[k]:.3f} ({LABEL.get(bv['draft'], bv['draft'])} g{bv['gamma']}) / "
                         f"**{bv[k] / bs[k]:.3f}**")
        body += ["", f"best VQ/hybrid / best SnapKV: static meas {cells[0].split('/ **')[-1].rstrip('*')}, "
                 f"all-active meas {cells[1].split('/ **')[-1].rstrip('*')}, all-active derived v5 "
                 f"{cells[2].split('/ **')[-1].rstrip('*')}", ""]
        summ.append(f"| {p['name']} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(L + summ + [""] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v5aa")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    v5d = v5_derived()
    order = {"syn_quote": 0, "hotpotqa_q": 1, "pg19": 2}
    files = sorted(glob.glob(f"{OUT}/run_{a.tag}_*.json")) + sorted(glob.glob(f"{OUT}/run_{a.tag}g512_*.json"))
    pts = [analyze(f, v5d) for f in files]
    pts.sort(key=lambda p: (order.get(p["task"], 9), -p["ctx"], -p["B"], p["gen"]))
    md = tables(pts, a.tag)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_{a.tag}_tables.md", "w").write(md + "\n")
        from tools.results import save_result
        for p in pts:
            save_result("E7", f"llama8b_magicvq_{a.tag}_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), fully real "
                                "e2e self-spec (e2e_real.py --accounting both)", "B": p["B"], "T": p["ctx"],
                                "gen": p["gen"], "task": p["task"], "configs": [r["config"] for r in p["rows"]]},
                        metrics={k: p[k] for k in ("ar_tok_s", "ar_step_ms", "ar_tok_s_runs", "ar_cotenant", "rows",
                                                   "cotenant_any", "cotenants_start", "cotenants_end",
                                                   "nvsmi_apps_start", "nvsmi_apps_end")},
                        model_dir="/data/models/llama3.1_8b_instruct",
                        notes=f"raw: {p['raw']}; static and all-active both measured wall clock in one run")


if __name__ == "__main__":
    main()
