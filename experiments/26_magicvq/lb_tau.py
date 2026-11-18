"""E7 magicvq v4: greedy acceptance tau of compressed-KV drafts on LongBench / synthetic far-context workloads.

Same machinery as tau_eval.py (bf16 target, bf16 KV, target greedy trajectory, teacher-forced G-stream drafts, round
walk); only the prompts differ (lb_tasks.py). Candidates whose target output is shorter than --min_gen tokens are
skipped (logged) until --n samples are accepted. Resumable: completed samples in the .partial file are kept.
Usage: python experiments/26_magicvq/lb_tau.py --ctx 16384 --tasks gov_report syn_kv --n 20 --run v4a --drafts ...
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tau_eval as TE  # noqa: E402
from mv_common import MODEL, OUTROOT, SINKS, bits_of, draft_specs, kv_tokens_equiv, make_quant  # noqa: E402
from lb_tasks import LB_TASKS, SYN_TASKS, lb_prompts, load_haystack_books, syn_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", nargs="+", default=LB_TASKS + SYN_TASKS)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_cand", type=int, default=60)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--min_gen", type=int, default=128)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--drafts", nargs="+", required=True)
    ap.add_argument("--run", default="v4")
    ap.add_argument("--min_frac", type=float, default=0.75)
    ap.add_argument("--skip_run", default=None, help="v4b: skip candidate ids already used or skipped in "
                    "tau_<skip_run>_<task>_ctx<ctx>.json (adds disjoint samples; --n counts the new ones)")
    ap.add_argument("--dump_matches", nargs="*", default=[], help="drafts whose per-position step-m match traces are "
                    "stored (for e2e_loop.py replay)")
    ap.add_argument("--only_run", default=None, help="v4b: evaluate only the ids of tau_<only_run>_<task>_ctx<ctx>.json "
                    "(e.g. a draft missing from that run, on the same samples)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    books = None
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    f = TE.M(model)
    specs = draft_specs(a.drafts)
    specs = {n: s for n, s in specs.items() if not (s["kind"] in ("evict", "snapkv", "topk") and s["budget"] >= a.ctx)}
    TE.SNAP["maxb"] = max([s["budget"] for s in specs.values() if s["kind"] in ("snapkv", "hyb")], default=0)
    kvqs = {}
    for s in specs.values():
        if s["kind"] in ("vq", "hyb") and s["fmt"] not in kvqs:
            kvqs[s["fmt"]] = make_quant(s["fmt"])
            kvqs[s["fmt"]].keep_first = 0 if (s["fmt"].startswith("cv") and s["fmt"].endswith("n")) else SINKS
    maxT = a.ctx + a.gen + 8
    Kst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    Vst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    cs = f.cos_sin(torch.arange(maxT, device="cuda"))
    out_dir = os.path.join(OUTROOT, "lb")
    os.makedirs(out_dir, exist_ok=True)
    gammas = list(range(1, a.G + 1))
    for task in a.tasks:
        out_path = os.path.join(out_dir, f"tau_{a.run}_{task}_ctx{a.ctx}.json")
        if os.path.exists(out_path):
            print("exists, skip", out_path, flush=True)
            continue
        res = {"config": {**vars(a), "task": task, "specs": specs, "bits": {n: bits_of(s, kvqs) for n, s in specs.items()},
                          "kv_tokens_equiv": {n: kv_tokens_equiv(s, kvqs, a.ctx) for n, s in specs.items()},
                          "model": MODEL}, "per_prompt": [], "skipped": []}
        if os.path.exists(out_path + ".partial"):
            old = json.load(open(out_path + ".partial"))
            if set(old["config"]["specs"]) == set(specs):
                res["per_prompt"], res["skipped"] = old["per_prompt"], old["skipped"]
        done = {r["id"] for r in res["per_prompt"]} | {r["id"] for r in res["skipped"]}
        only = None
        for flag, rn in (("skip", a.skip_run), ("only", a.only_run)):
            if rn:
                ref = json.load(open(os.path.join(out_dir, f"tau_{rn}_{task}_ctx{a.ctx}.json")))
                if flag == "skip":
                    done |= {r["id"] for r in ref["per_prompt"]} | {r["id"] for r in ref["skipped"]}
                else:
                    only = {r["id"]: r["text"] for r in ref["per_prompt"]}
        if task in SYN_TASKS:
            if books is None:
                books = load_haystack_books(tok)
            cands = syn_prompts(tok, task, a.ctx, a.n_cand, books)
        else:
            cands = lb_prompts(tok, task, a.ctx, a.n_cand, a.min_frac)
        print(task, "candidates", len(cands), "P", [len(c["ids"]) for c in cands[:8]], f"{time.time() - t0:.0f}s",
              flush=True)
        for c in cands:
            if len(res["per_prompt"]) >= a.n:
                break
            if c["id"] in done or (only is not None and c["id"] not in only):
                continue
            t1 = time.time()
            ids = torch.tensor(c["ids"], device="cuda")
            P = ids.shape[0]
            path = TE.prefill_and_generate(f, ids, a.gen, Kst, Vst, cs)
            t2 = time.time()
            meta = {k: v for k, v in c.items() if k != "ids"}
            text = tok.decode(path, skip_special_tokens=True)
            if len(path) < a.min_gen + 1:
                res["skipped"].append({"id": c["id"], "P": P, "gen": len(path), "text": text[:400]})
                print(task, "skip", c["id"], P, len(path), flush=True)
                json.dump(res, open(out_path + ".partial", "w"))
                continue
            full = torch.cat([ids, torch.tensor(path, device="cuda")])
            TE.TIMES = {}
            tgt, preds = TE.tf_pass(f, full, P, Kst, Vst, cs, specs, kvqs, a.G)
            N = tgt.shape[0]
            pathn = torch.tensor(path[1:], device="cuda")
            if only is not None:
                meta["same_text_as_" + a.only_run] = text == only[c["id"]]
            rec = {"set": task, "id": c["id"], "P": P, "gen": len(path), "N": N, "meta": meta, "text": text,
                   "target_tf_vs_gen": (tgt == pathn).float().mean().item(), "group_s": TE.TIMES, "drafts": {}}
            for n, pr in preds.items():
                mt = (pr == tgt[None])
                matches = mt.tolist()
                d = {"match_rate_step": mt.float().mean(1).tolist()}
                if n in a.dump_matches:
                    d["matches"] = [[int(x) for x in row] for row in matches]
                for g in gammas:
                    d[f"g{g}"] = TE.walk(matches, g, N)
                rec["drafts"][n] = d
            res["per_prompt"].append(rec)
            torch.cuda.empty_cache()
            print(task, len(res["per_prompt"]), c["id"], P, len(path), f"gen {t2 - t1:.0f}s tf {time.time() - t2:.0f}s",
                  {n: round(d["g4"][0] / d["g4"][1], 2) for n, d in rec["drafts"].items()},
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", {k: round(v) for k, v in TE.TIMES.items()},
                  flush=True)
            json.dump(res, open(out_path + ".partial", "w"))
        per = res["per_prompt"]
        agg = {"n_prompts": len(per), "n_skipped": len(res["skipped"]), "mean_P": sum(r["P"] for r in per) / max(1, len(per)),
               "mean_N": sum(r["N"] for r in per) / max(1, len(per)), "drafts": {}}
        if per:
            agg["target_tf_vs_gen"] = sum(r["target_tf_vs_gen"] for r in per) / len(per)
            for n in specs:
                A = {"alpha_step": [sum(r["drafts"][n]["match_rate_step"][m] * r["N"] for r in per) / sum(r["N"] for r in per)
                                    for m in range(a.G)]}
                for g in gammas:
                    toks = sum(r["drafts"][n][f"g{g}"][0] for r in per)
                    its = sum(r["drafts"][n][f"g{g}"][1] for r in per)
                    A[f"tau_g{g}"] = toks / its
                    A[f"tau_g{g}_per_prompt"] = [r["drafts"][n][f"g{g}"][0] / r["drafts"][n][f"g{g}"][1] for r in per]
                agg["drafts"][n] = A
        res["agg"] = agg
        res["wall_s"] = time.time() - t0
        json.dump(res, open(out_path, "w"), indent=1)
        if per:
            print(task, "AGG", json.dumps({n: [round(v["alpha_step"][0], 3), round(v["tau_g3"], 2), round(v["tau_g6"], 2)]
                                            for n, v in agg["drafts"].items()}), flush=True)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
