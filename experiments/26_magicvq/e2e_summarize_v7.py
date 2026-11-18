"""E7 magicvq v7: more fully real e2e points with the v6 t2w codebook (e2e_real.py --accounting both --cbx t2w=...).

Points: (1) larger gamma (5, 6) for t2w at hotpotqa_q 16k B16 and PG-19 32k B16 (tag v7g6); (2) real LongBench tasks
(tag v7, plus v7b = second process of the same point when the drafts are split over two processes).
Per point: best SnapKV (snapkv2048 / snapkv4096), best StreamingLLM-512 (stream512 = MagicDec StreamingLLM: 16 sinks +
last 496 prompt tokens, then the growing window, same draft-cache semantics as our SnapKV), best VQ/hybrid with the t2w
codebook ('@t2w'); static and all-active (headline) measured speedups vs the AR of the SAME process. Ratio = best VQ
speedup / best SnapKV speedup (speedups of different processes are each relative to their own AR).
v6 computed = results/E7/llama8b_magicvq_v6_tables.md "vs best SnapKV" of the best of hyb512pt2wL2 / vqpt2wL2+r128 at
the same task / B / ctx (computed from replay tau x TP4 step times, not measured; hotpotqa_q 32k: v6 used the pre-fix
prompts).
Usage: python experiments/26_magicvq/e2e_summarize_v7.py [--save]
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
LAB = dict(LABEL, stream512="StreamingLLM-512")
KINDS = (("snapkv", lambda d: d.startswith("snapkv")), ("stream", lambda d: d.startswith("stream")),
         ("vq_t2w", lambda d: d.endswith("@t2w")))


def v6_pred():
    """{(task, B, ctx): (ratio, draft, gamma)} best trained-2-bit draft vs best SnapKV, computed (v6 table)."""
    d, sec = {}, None
    for line in open(f"{RES}/llama8b_magicvq_v6_tables.md"):
        if line.startswith("## B"):
            sec = line[3:].split(":")[0]
            B, ctx = int(sec.split("_")[0][1:]), int(sec.split("ctx")[1])
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if sec and len(c) == 7 and c[2] in ("hyb512pt2wL2", "vqpt2wL2+r128"):
            r = float(c[5].split()[0])
            k = (c[0], B, ctx)
            if k not in d or r > d[k][0]:
                d[k] = (r, c[2], c[4].split("(")[1].rstrip(")"))
    return d


def lab(r):
    n = r["draft"]
    base, _, cb = n.partition("@")
    return LAB.get(base, base) + (f" [{cb}]" if cb else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    v5d = v5_derived()
    pts = {}
    for tag in ("v7g6", "v7g6v", "v7g6s", "v7", "v7b"):
        for f in sorted(glob.glob(f"{OUT}/run_{tag}_*.json")):
            q = analyze(f, v5d)
            raw = json.load(open(f))
            k = ("v7g6" if tag.startswith("v7g6") else "v7", q["task"], q["ctx"], q["B"], q["gen"])
            for r in q["rows"]:
                r["proc"], r["proc_ar"] = tag, q["ar_tok_s"]
                c = raw["configs"][r["config"]]
                r["rep_out_differs"] = bool(c.get("rep_out_differs") or c["all_active"].get("rep_out_differs"))
                rd = c.get("rep_out_differs") or []  # static: per timed rep, first divergence per row vs rep 0
                r["rep_diff_static_rows"] = [sum(x >= 0 for x in d["first_div_per_row"]) for d in rd]
                r["rep_diff_reps123_same_pattern"] = (len({tuple(d["first_div_per_row"]) for d in rd}) == 1) if rd else None
                r["rep_diff_aa_rows"] = [d["rows"] for d in c["all_active"].get("rep_out_differs") or []]
            if k not in pts:
                q["procs"] = {tag: {"ar_tok_s": q["ar_tok_s"], "ar_step_ms": q["ar_step_ms"], "raw": f,
                                    "cotenant_any": q["cotenant_any"], "partial": raw.get("partial")}}
                q["P"] = json.load(open(f))["prefill"]["P"]
                pts[k] = q
            else:
                p = pts[k]
                p["procs"][tag] = {"ar_tok_s": q["ar_tok_s"], "ar_step_ms": q["ar_step_ms"], "raw": f,
                                   "cotenant_any": q["cotenant_any"], "partial": raw.get("partial")}
                p["rows"] += q["rows"]
                p["cotenant_any"] = p["cotenant_any"] or q["cotenant_any"]
    v6 = v6_pred()
    L = ["# E7 magicvq v7: more measured e2e points with the v6 t2w codebook (TP=4 RTX 4090, GPUs 4-7, bf16 "
         "Llama-3.1-8B-Instruct, gen 256, EOS ignored)", "",
         "- e2e_real.py --accounting both --cbx t2w=cb_d8K256x2_post_cn_t2w.pt; static and all-active reps interleaved, "
         "warm-up excluded, 3 timed reps each, median. Speedup vs the AR of the same process.",
         "- all-active (headline, MagicDec accounting): no row frozen, stop when the longest row has >= 256 tokens. "
         "static: every row exactly 256 tokens.",
         "- StreamingLLM-512 (new draft kind 'stream512' in e2e_real.py): 16 sinks + last 496 prompt tokens + the "
         "growing generated window (MagicDec prefill_draft semantics, same cache handling as our SnapKV draft).",
         "- LongBench samples: the v4 (+v4b) tau samples in order (passed lb_tau's min_gen 128 filter), first B. "
         "hotpotqa_q 32k prompts rebuilt after the sorted(own) determinism fix (identical under PYTHONHASHSEED 1 and 2), "
         "so they differ from the v4/v6 32k prompts.",
         "- ratio = best VQ/hybrid (t2w) speedup / best SnapKV speedup over ALL measured SnapKV budgets / gammas (and / best StreamingLLM). v6 computed = best "
         "trained-2-bit / best SnapKV from results/E7/llama8b_magicvq_v6_tables.md (replay tau x step times).", ""]
    S = ["## Summary per point", "",
         "| point | P min-max | AR tok/s | all-active: SnapKV / StreamingLLM / VQ t2w | **aa ratio VQ/SnapKV** | aa VQ/Stream | "
         "static: SnapKV / StreamingLLM / VQ t2w | static ratio VQ/SnapKV | v6 computed VQ/SnapKV | SnapKV gammas measured "
         "(g5/6?) | co-tenant |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    body = []
    order = {"hotpotqa_q": 0, "pg19": 1, "gov_report": 2, "qmsum": 3, "repobench-p": 4, "multi_news": 5}
    keys = sorted(pts, key=lambda k: (k[0] != "v7g6", order.get(k[1], 9) if k[0] == "v7g6" else -k[2], order.get(k[1], 9)))
    for k in keys:
        p = pts[k]
        name = f"{p['task']} ctx {p['ctx']} B={p['B']}" + (" (gamma 5/6 for t2w)" if k[0] == "v7g6" else "")
        summ = {}
        for acc in ("aa_speedup", "static_speedup"):
            summ[acc] = {}
            for kn, f in KINDS:
                rs = [r for r in p["rows"] if f(r["draft"])]
                b = max(rs, key=lambda r: r[acc]) if rs else None
                summ[acc][kn] = None if b is None else {"draft": b["draft"], "gamma": b["gamma"], "speedup": b[acc]}
            s, t, v = (summ[acc][x] for x in ("snapkv", "stream", "vq_t2w"))
            summ[acc]["ratio_vq_snapkv"] = v["speedup"] / s["speedup"] if v and s else None
            summ[acc]["ratio_vq_stream"] = v["speedup"] / t["speedup"] if v and t else None
        p["summary"] = summ
        pr = v6.get((p["task"], p["B"], p["ctx"]))
        p["v6_computed"] = pr

        def cell(acc):
            out = []
            for kn in ("snapkv", "stream", "vq_t2w"):
                x = summ[acc][kn]
                out.append("-" if x is None else f"{x['speedup']:.3f} ({lab(x)} g{x['gamma']})".replace(" [t2w]", ""))
            return " / ".join(out)
        f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        ars = " / ".join(f"{v['ar_tok_s']:.1f}" for v in p["procs"].values())
        sgs = sorted({(r["draft"][6:], r["gamma"]) for r in p["rows"] if r["draft"].startswith("snapkv")})
        sg = ", ".join(f"{w}:g{g}" for w, g in sgs)
        g56 = any(g >= 5 for _, g in sgs)
        p["snapkv_gammas"], p["snapkv_g56_measured"] = sgs, g56
        S.append(f"| {name} | {min(p['P'])}-{max(p['P'])} | {ars} | {cell('aa_speedup')} | "
                 f"**{f3(summ['aa_speedup']['ratio_vq_snapkv'])}** | {f3(summ['aa_speedup']['ratio_vq_stream'])} | "
                 f"{cell('static_speedup')} | {f3(summ['static_speedup']['ratio_vq_snapkv'])} | "
                 f"{'-' if pr is None else f'{pr[0]:.3f} ({pr[1]} {pr[2]})'} | {sg} ({'yes' if g56 else 'NO'}) | "
                 f"{'YES' if p['cotenant_any'] else 'no'} |")
        body += [f"## {name}", "",
                 "processes: " + "; ".join(f"{t}: AR {v['ar_tok_s']:.1f} tok/s (step {v['ar_step_ms']:.2f} ms), "
                                           f"co-tenant {'YES' if v['cotenant_any'] else 'no'}, {os.path.basename(v['raw'])}"
                                           + (f" (PARTIAL: {v['partial']})" if v.get("partial") else "")
                                           for t, v in p["procs"].items()),
                 f"prompt lengths P: {p['P']}", "",
                 "| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | "
                 "aa rows identical to AR on overlap | co-tenant | outputs differ across reps |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["rows"]:
            a0, a1 = (x / r["proc_ar"] for x in r["aa_tok_s_min_max"])
            body.append(f"| {lab(r)} | {r['proc']} | {r['gamma']} | {r['static_speedup']:.3f} | **{r['aa_speedup']:.3f}** "
                        f"({a0:.3f}-{a1:.3f}) | {r['tau_static']:.2f} / {r['aa_tau']:.2f} | {r['static_round_ms']:.1f} / "
                        f"{r['aa_round_ms']:.1f} | {r['aa_n_rows_identical_overlap']}/{p['B']} | "
                        f"{'YES' if r['cotenant'] else 'no'} | "
                        + (f"YES: static rows vs rep0 {r['rep_diff_static_rows']}, aa rows {r['rep_diff_aa_rows']}, "
                           f"reps 1-3 same divergence pattern: {r['rep_diff_reps123_same_pattern']}"
                           if r['rep_out_differs'] else "no") + " |")
        body.append("")
    md = "\n".join(L + S + [""] + body)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_v7_tables.md", "w").write(md + "\n")
        from tools.results import save_result
        for k in keys:
            p = pts[k]
            save_result("E7", f"llama8b_magicvq_v7_{'g6_' if k[0] == 'v7g6' else ''}{p['task']}_ctx{p['ctx']}_B{p['B']}"
                        f"_gen{p['gen']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), fully real "
                                "e2e self-spec (e2e_real.py --accounting both --cbx t2w=...), deterministic LUT kernel",
                                "B": p["B"], "T": p["ctx"], "gen": p["gen"], "task": p["task"], "P": p["P"],
                                "configs": [r["config"] for r in p["rows"]],
                                "cb_t2w": "/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt"},
                        metrics={"procs": p["procs"], "rows": p["rows"], "summary": p["summary"],
                                 "snapkv_gammas": p["snapkv_gammas"], "snapkv_g56_measured": p["snapkv_g56_measured"],
                                 "v6_computed": p["v6_computed"], "cotenant_any": p["cotenant_any"],
                                 "cotenants_start": p["cotenants_start"], "cotenants_end": p["cotenants_end"]},
                        model_dir="/data/models/llama3.1_8b_instruct",
                        notes="raw: " + "; ".join(v["raw"] for v in p["procs"].values()))


if __name__ == "__main__":
    main()
