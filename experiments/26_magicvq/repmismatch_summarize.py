"""E7 gamma6 rep-mismatch: old (pre-fix T, v7 / v8 processes) vs fixed-T reruns of every config that ran in or after
an overflowing gamma-6 config. Writes results/E7/llama8b_magicvq_repmismatch_{tables.md,<point>.json}.
Usage: python experiments/26_magicvq/repmismatch_summarize.py"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.results import save_result  # noqa: E402

RAW = "/data/lsh/results_accept/magicvq/e2e_real"
OLD = {  # new tag -> old processes searched in order for the same config string
    ("rmfixa", "gov_report"): ["v8e2ea"], ("rmfixa", "hotpotqa_q"): ["v8e2ea"],
    ("rmfix7", "gov_report"): ["v7"], ("rmfix7", "qmsum"): ["v7"], ("rmfix7", "repobench-p"): ["v7"],
    ("rmfix7g6", "hotpotqa_q"): ["v7g6", "v7g6s", "v7g6v"], ("rmfixmse", "hotpotqa_q"): ["v7mse"],
    ("rmlegacy", "gov_report"): ["v8e2ea"],
}
CTX, B = 16384, 16
med = statistics.median


def rows(path):
    R = json.load(open(path))
    ar = R["configs"]["ar"]
    ar_tps = med(x["tok_per_s"] for x in ar["runs"])
    out = {"_ar": ar_tps, "_P": R["prefill"]["P"]}
    for c, r in R["configs"].items():
        if c == "ar":
            continue
        aa = r["all_active"]
        acc = [x + 1 for row in r["acc_log"] for x in row if x >= 0]
        a = [x["tok_per_s"] / ar_tps for x in aa["runs"]]
        s = [x["tok_per_s"] / ar_tps for x in r["runs"]]
        fd = aa.get("first_divergence_vs_ar") or []
        out[c] = {"aa": med(a), "aa_min": min(a), "aa_max": max(a), "static": med(s), "tau_aa": aa["tau"],
                  "tau_aa_runs": [x["tokens"] / (B * x["rounds"]) for x in aa["runs"]],
                  "tau_static_rep0": sum(acc) / len(acc),
                  "rep_diff_static": bool(r.get("rep_out_differs")), "rep_diff_aa": bool(aa.get("rep_out_differs")),
                  "rep_ngen_differs": aa.get("rep_ngen_differs"),
                  "aa_identical_ar": sum(1 for f in fd if f is None),
                  "prompt_head_after": r.get("prompt_head_after"), "cache_overflow": r.get("cache_overflow"),
                  "cotenant": bool((r.get("co_before") or {}).get("cotenant") or (r.get("co_after") or {}).get("cotenant"))}
    return out, R


def main():
    md = ["# E7 gamma-6 rep-mismatch: target-cache overflow in static accounting (fix + reruns)", "",
          "- Cause: static mode keeps frozen rows (ngen >= gen) running draft + verify at their final length "
          "(<= P + gen + gamma) until the slowest row finishes, so they append K/V up to position P + gen + 2 gamma. "
          "Old T = ceil64(Pmax + gen + gmax + 2): at gamma 6 with P near Pmax (16k LongBench points, slack 9 < 12) "
          "the write wraps into position 0.. of the next (seq, kv-head) [or the next k/v plane / layer] of the TARGET "
          "cache (BOS / attention sinks) and stays corrupted for the rest of the process (all later configs and reps).",
          "- Fix: T = ceil64(Pmax + gen + 2 gmax + 2) + guard (assert length.max + gamma < T after each rep) + "
          "prompt-head integrity check (first 32 positions of every head vs the post-prefill snapshot, after every "
          "config). Fixed-T reruns: same prompts / configs / codebooks, --accounting both --soft_rep_check, reps 3.",
          "- speedup = config tok/s / AR tok/s of the same process (median of timed reps); aa = all-active (headline).",
          "- tau static = rep-0 acc_log mean (as in the v7/v8 tables); tau aa = median over timed reps.", ""]
    allrec = {}
    for f in sorted(glob.glob(f"{RAW}/run_rm*_ctx{CTX}_B{B}.json")):
        stem = Path(f).name[4:-5]
        tag, rest = stem.split("_", 1)
        task = rest.rsplit(f"_ctx{CTX}", 1)[0]
        new, Rn = rows(f)
        olds = []
        for ot in OLD.get((tag, task), []):
            p = f"{RAW}/run_{ot}_{task}_ctx{CTX}_B{B}.json"
            if Path(p).exists():
                olds.append((ot, rows(p)[0]))
        md += [f"## {task} 16k B16 - new process `{tag}`{' (LEGACY T, bug reproduction, reps 1)' if Rn['args'].get('legacy_T') else ''}",
               "", f"AR tok/s new {new['_ar']:.1f}; old " + ", ".join(f"{t} {o['_ar']:.1f}" for t, o in olds)
               + f"; co-tenant any (new): {Rn.get('cotenant_any')}", "",
               "| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | "
               "tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |",
               "|---|---|---|---|---|---|---|---|---|"]
        rec = {}
        for c, v in new.items():
            if c.startswith("_"):
                continue
            o = next((x[c] for _, x in olds if c in x), None)
            ot = next((t for t, x in olds if c in x), None)
            ph = v["prompt_head_after"] or {}
            fmt = lambda x: f"{x['aa']:.3f} ({x['aa_min']:.3f}-{x['aa_max']:.3f})"  # noqa: E731
            rd = lambda x: ("yes" if (x["rep_diff_static"] or x["rep_diff_aa"]) else "no")  # noqa: E731
            g = lambda k, f: (format(o[k], f) if o else "-")  # noqa: E731
            old_aa = f"{fmt(o)} [{ot}]" if o else "-"
            md.append(f"| {c} | {old_aa} | **{fmt(v)}** | {g('static', '.3f')} / {v['static']:.3f} | "
                      f"{g('tau_aa', '.2f')} / {v['tau_aa']:.2f} | {g('tau_static_rep0', '.2f')} / "
                      f"{v['tau_static_rep0']:.2f} | {rd(o) if o else '-'} / {rd(v)} | "
                      f"{g('aa_identical_ar', 'd')} / {v['aa_identical_ar']} | "
                      f"{ph.get('n_heads_changed', '-')} heads, rows {ph.get('rows', '-')}, pos {ph.get('positions', '-')} |")
            rec[c] = {"new": v, "old": o, "old_proc": ot}
        md.append("")
        allrec[stem] = {"new_file": f, "old_files": [t for t, _ in olds], "ar_new": new["_ar"],
                        "ar_old": {t: x["_ar"] for t, x in olds}, "configs": rec,
                        "legacy_T": bool(Rn["args"].get("legacy_T")), "cotenant_any": Rn.get("cotenant_any")}
    out = REPO / "results/E7/llama8b_magicvq_repmismatch_tables.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    for stem, rec in allrec.items():
        try:
            save_result("E7", f"llama8b_magicvq_repmismatch_{stem}", {"script": "experiments/26_magicvq/repmismatch_run.sh",
                                                                     "raw": rec["new_file"]}, rec,
                        model_dir="/data/models/llama3.1_8b_instruct")
        except FileExistsError as e:
            print("skip", e)


if __name__ == "__main__":
    main()
