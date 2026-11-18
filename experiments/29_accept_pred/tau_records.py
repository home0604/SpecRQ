"""E8a: merge the existing per-sequence tau records of E7 (lb_tau.py runs v4, v4b, v4bh, v6a, v6a2, v6b) into one table
keyed by (task, ctx, id). No GPU, no torch: plain json.

Per (task, ctx, id, draft): tau_g1..g6 (tokens per round of the greedy round walk) and alpha1 (= step-1 match rate),
plus the target text (for the prompt-reproducibility check of the feature run). Drafts measured in several runs
(e.g. snapkv4096 in v4 and v6a on the same ids) are checked for agreement and the first run's value is kept.
Draft aliases: the v6 trained-codebook drafts are renamed to the e2e names (vqpt2wL2+r128 -> vqpL2+r128@t2w,
hyb512pt2wL2 -> hyb512pL2@t2w).
Usage: python experiments/29_accept_pred/tau_records.py   (prints counts and duplicate agreement)
"""
from __future__ import annotations

import glob
import json
import os
import re

LBDIR = "/data/lsh/results_accept/magicvq/lb"
RUNS = ["v4", "v4b", "v4bh", "v6a", "v6a2", "v6b"]  # earlier runs win on duplicates
ALIAS = {"vqpt2wL2+r128": "vqpL2+r128@t2w", "hyb512pt2wL2": "hyb512pL2@t2w", "vqpt2L2": "vqpL2@t2", "vqpt1L1": "vqpL1@t1",
         "vqpt1wL1+r128": "vqpL1+r128@t1w", "hyb512pt1wL1": "hyb512pL1@t1w"}
# hotpotqa_q 32k prompts depend on PYTHONHASHSEED (set order, fixed in lb_tasks.py for v7): not reproducible -> excluded
UNREPRODUCIBLE = {("hotpotqa_q", 32768)}


def load(verbose=False):
    recs = {}  # (task, ctx, id) -> {"P", "N", "text", "run", "drafts": {name: {...}}}
    dup = {}   # draft -> list of |tau_g4 diff|
    for run in RUNS:
        for f in sorted(glob.glob(f"{LBDIR}/tau_{run}_*_ctx*.json")):
            m = re.fullmatch(rf"tau_{run}_(.+)_ctx(\d+)\.json", os.path.basename(f))
            if not m:
                continue
            task, ctx = m.group(1), int(m.group(2))
            d = json.load(open(f))
            for r in d["per_prompt"]:
                key = (task, ctx, r["id"])
                e = recs.setdefault(key, {"P": r["P"], "N": r["N"], "gen": r["gen"], "text": r["text"], "run": run,
                                          "drafts": {}, "runs": []})
                e["runs"].append(run)
                if r["text"] != e["text"]:
                    e.setdefault("text_mismatch_runs", []).append(run)
                for n, x in r["drafts"].items():
                    n = ALIAS.get(n, n)
                    v = {f"tau_g{g}": x[f"g{g}"][0] / x[f"g{g}"][1] for g in range(1, 7) if f"g{g}" in x}
                    v["alpha1"] = x["match_rate_step"][0]
                    v["alpha_step"] = x["match_rate_step"]
                    if n in e["drafts"]:
                        dup.setdefault(n, []).append(abs(e["drafts"][n].get("tau_g4", 0) - v.get("tau_g4", 0)))
                        continue
                    e["drafts"][n] = v
    if verbose:
        from collections import Counter
        c = Counter((k[0], k[1]) for k in recs)
        for k in sorted(c):
            n4 = sum(1 for kk, e in recs.items() if kk[:2] == k and "snapkv4096" in e["drafts"])
            nt = sum(1 for kk, e in recs.items() if kk[:2] == k and "hyb512pL2@t2w" in e["drafts"])
            print(k, "n", c[k], "with snapkv4096", n4, "with t2w", nt, "excluded" if k in UNREPRODUCIBLE else "")
        for n, v in dup.items():
            print("dup", n, len(v), "max |dtau_g4|", round(max(v), 4), "mean", round(sum(v) / len(v), 4))
        print("text mismatches across runs:", sum(1 for e in recs.values() if e.get("text_mismatch_runs")))
    return recs


if __name__ == "__main__":
    load(verbose=True)
