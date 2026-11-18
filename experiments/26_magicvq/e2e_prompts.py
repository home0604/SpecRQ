"""E7 magicvq v5: freeze the prompts of the fully real e2e runs (token ids) + the v4/v4b tau records of the same samples.

syn_quote / hotpotqa_q: the v4 + v4b tau samples (lb_tau.py order: v4 ids first, then v4b), rebuilt with lb_tasks.py and
checked against the stored prompt length P. pg19: tau_eval.py build_prompts (BOS + first ctx-1 tokens of book i).
Output: /data/lsh/results_accept/magicvq/e2e_real/prompts_<task>_ctx<ctx>.json
Usage: python experiments/26_magicvq/e2e_prompts.py   (CPU only)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_common import MODEL, OUTROOT  # noqa: E402

OUT = os.path.join(OUTROOT, "e2e_real")
DRAFTS = ["snapkv2048", "snapkv4096", "vqpL2+r128", "hyb512pL2"]


def main():
    from transformers import AutoTokenizer
    from lb_tasks import lb_prompts, load_haystack_books, syn_prompts
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    for task, ctx in (("syn_quote", 32768), ("hotpotqa_q", 16384)):
        recs = []
        cfg = {}
        for run in ("v4", "v4b"):
            d = json.load(open(f"{OUTROOT}/lb/tau_{run}_{task}_ctx{ctx}.json"))
            cfg[run] = d["config"]
            for r in d["per_prompt"]:
                recs.append({"id": r["id"], "P": r["P"], "gen": r["gen"], "N": r["N"], "run": run, "text": r["text"],
                             "tau_rec": {n: {k: v for k, v in r["drafts"][n].items() if k.startswith("g")}
                                         for n in DRAFTS if n in r["drafts"]}})
        if task.startswith("syn"):
            cands = syn_prompts(tok, task, ctx, 80, load_haystack_books(tok))
        else:
            assert cfg["v4"]["min_frac"] == cfg["v4b"]["min_frac"]
            cands = lb_prompts(tok, task, ctx, 80, cfg["v4"]["min_frac"])
        by = {c["id"]: c["ids"] for c in cands}
        for r in recs:
            r["ids"] = by[r["id"]]
            assert len(r["ids"]) == r["P"], (r["id"], len(r["ids"]), r["P"])
        json.dump(recs, open(f"{OUT}/prompts_{task}_ctx{ctx}.json", "w"))
        print(task, ctx, len(recs), "prompts, P ok", flush=True)
    from mv_common import build_prompts, load_books
    books = load_books(tok, 32, 8192 + 32768 + 1024)
    ps = build_prompts(tok, books, 32768, 16, ("pg19",))["pg19"]
    json.dump([{"id": f"pg19_{i}", "P": len(p), "ids": p.tolist(), "tau_rec": {}} for i, p in enumerate(ps)],
              open(f"{OUT}/prompts_pg19_ctx32768.json", "w"))
    print("pg19", len(ps), flush=True)


if __name__ == "__main__":
    main()
