"""E7 magicvq v5: freeze the prompts of the fully real e2e runs (token ids) + the v4/v4b tau records of the same samples.

syn_quote / hotpotqa_q: the v4 + v4b tau samples (lb_tau.py order: v4 ids first, then v4b), rebuilt with lb_tasks.py and
checked against the stored prompt length P. pg19: tau_eval.py build_prompts (BOS + first ctx-1 tokens of book i).
Output: /data/lsh/results_accept/magicvq/e2e_real/prompts_<task>_ctx<ctx>.json
Usage: python experiments/26_magicvq/e2e_prompts.py   (CPU only)
       python experiments/26_magicvq/e2e_prompts.py gov_report:32768 qmsum:16384 ...   (v7: LongBench tasks, lb_main)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_common import MODEL, OUTROOT  # noqa: E402

OUT = os.path.join(OUTROOT, "e2e_real")
DRAFTS = ["snapkv2048", "snapkv4096", "vqpL2+r128", "hyb512pL2"]


def lb_main(specs):
    """v7: real LongBench tasks. spec = <task>:<ctx>. Samples = the v4 (+ v4b) tau samples in lb_tau.py order (they passed
    lb_tau's min_gen filter), rebuilt with lb_tasks.lb_prompts at the min_frac of those runs. hotpotqa_q ctx > 17000:
    lb_tasks now orders the own passages deterministically (sorted, was list(set)), so the prompts differ from the v4
    ones; the P check is skipped there (old P kept as 'P_v4') and tau_rec is dropped."""
    from transformers import AutoTokenizer
    from lb_tasks import lb_prompts
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    for spec in specs:
        task, ctx = spec.split(":")
        ctx = int(ctx)
        changed = task == "hotpotqa_q" and ctx > 17000
        recs, fr = [], set()
        for run in ("v4", "v4b"):
            f = f"{OUTROOT}/lb/tau_{run}_{task}_ctx{ctx}.json"
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            fr.add(d["config"]["min_frac"])
            for r in d["per_prompt"]:
                recs.append({"id": r["id"], "P": r["P"], "gen": r["gen"], "N": r["N"], "run": run,
                             "tau_rec": {} if changed else {n: {k: v for k, v in r["drafts"][n].items() if k.startswith("g")}
                                                            for n in DRAFTS if n in r["drafts"]}})
        assert len(fr) == 1, fr
        cands = lb_prompts(tok, task, ctx, 80, fr.pop())
        by = {c["id"]: c for c in cands}
        for r in recs:
            c = by[r["id"]]
            r["ids"] = c["ids"]
            r["orig_tokens"] = c["orig_tokens"]
            if changed:
                r["P_v4"], r["P"] = r["P"], len(c["ids"])
            assert len(r["ids"]) == r["P"], (r["id"], len(r["ids"]), r["P"])
        json.dump(recs, open(f"{OUT}/prompts_{task}_ctx{ctx}.json", "w"))
        print(task, ctx, len(recs), "prompts, P", min(r["P"] for r in recs), "-", max(r["P"] for r in recs),
              "(rebuilt, P changed)" if changed else "ok", flush=True)


def main():
    if len(sys.argv) > 1:  # v7: python e2e_prompts.py gov_report:32768 qmsum:16384 ...
        return lb_main(sys.argv[1:])
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
