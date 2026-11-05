"""
Self-speculative decoding benchmark: acceptance + measured and modeled speed-up.

For each prompt: AR with the exact target (reference speed and greedy output), then speculative
decoding with the chosen KV representation. Reports acceptance, tokens/iteration, greedy
losslessness (spec output == AR output), measured wall-clock speed-up of this PyTorch
reference, and the memory-traffic-modeled speed-up for a range of (ctx, batch).

Examples:
  # sanity: draft == target -> acceptance 1.0, spec output == AR output
  python -m specdec.run --rep fp16 --gpus 1 2
  # INT2 draft KV (per-token groups of 32), exact target
  python -m specdec.run --rep int --bits 2 --gpus 1
  # RQ: draft reads m=2 stages, target all stages
  python -m specdec.run --rep rq --codebook /data/lsh/ckpts/wikitext/<cfg>/meta_model.pt --m 2 --gpus 1 2
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from specdec.costmodel import modeled_speedup
from specdec.decoding import autoregressive, speculative
from specdec.kvstore import FP16Rep, IntRep, QuantSpecRep, RQRep, SnapKVRep, StreamingRep
from specdec.model import Llama


PG19_TEST = "/data/lsh/datasets/pg19-test/data"


def load_prompts(tok, ctx, n, dataset):
    """`ctx`-token prompts (BOS + text). PG-19: the first ctx tokens of each long-enough test book."""
    bos = torch.tensor([tok.bos_token_id]) if tok.bos_token_id is not None else torch.tensor([], dtype=torch.long)
    if dataset == "pg19":
        import glob
        import pandas as pd
        df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19_TEST}/*.parquet")))
        prompts = []
        for text in df["text"]:
            ids = tok(text[: ctx * 8], return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(ids) >= ctx:
                prompts.append(torch.cat([bos, ids[: ctx - len(bos)]]))
            if len(prompts) == n:
                break
        return prompts
    if dataset == "wikitext":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
        stride = max(ctx, (len(ids) - ctx) // n)
        return [torch.cat([bos, ids[i * stride: i * stride + ctx - len(bos)]]) for i in range(n)]
    raise ValueError(dataset)


def build_rep(args, model):
    budget = args.budget or int(args.ctx * args.budget_frac)
    if args.rep == "fp16":
        return FP16Rep()
    if args.rep == "int":
        return IntRep(bits=args.bits, group=args.group)
    if args.rep == "quantspec":
        return QuantSpecRep(group=args.group, bu=args.kv_bits, bl=args.kv_bits, target=args.target,
                            scale=args.kv_scale)
    if args.rep == "streaming":
        return StreamingRep(budget, sink=args.sink)
    if args.rep == "snapkv":
        return SnapKVRep(budget, window=args.window, kernel=args.kernel)
    if args.rep == "rq":
        return RQRep(model, args.codebook, m=args.m, target=args.target, draft_new=args.draft_new,
                     target_stages=args.target_stages, window=args.rq_window or None,
                     cw_format=args.cw_format, cw_stages=args.cw_stages, cw_format_rest=args.cw_format_rest)
    raise ValueError(args.rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama2_7b_32k_instruct")
    ap.add_argument("--rep", choices=["fp16", "int", "quantspec", "streaming", "snapkv", "rq"], default="fp16")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--kv-bits", type=int, default=4, help="quantspec: upper = lower bits (4 -> INT4 + INT4)")
    ap.add_argument("--kv-scale", choices=["rtn", "paper"], default="rtn",
                    help="quantspec hierarchy scales; 'paper' = QuantSpec's formula (used in quantspec_repro, baseline50)")
    ap.add_argument("--group", type=int, default=None, help="int: 32, quantspec: 128")
    ap.add_argument("--budget", type=int, default=None, help="sparse draft KV budget (tokens)")
    ap.add_argument("--budget-frac", type=float, default=0.25, help="budget = ctx * frac if --budget unset (QuantSpec)")
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--draft-wbits", choices=["2", "3", "4", "16", "n4", "n6", "n8"], default="16",
                    help="4: separate RTN INT4 draft (QuantSpec); n8: nested INT8 weights (draft = upper 4 bits)")
    ap.add_argument("--codebook", default=None)
    ap.add_argument("--cw-format", default=None, help="rq: shape-codeword format (specrq.formats) of stages < --cw-stages")
    ap.add_argument("--cw-stages", type=int, default=None, help="rq: stages stored in --cw-format (default all)")
    ap.add_argument("--cw-format-rest", default=None, help="rq: format of the remaining stages (default as stored)")
    ap.add_argument("--m", type=int, default=2, help="RQ stages read by the draft")
    ap.add_argument("--target-stages", type=int, default=None, help="RQ stages read by the target (--target full)")
    ap.add_argument("--rq-window", type=int, default=128, help="recent tokens kept exact (QuantSpec policy); 0 = none")
    ap.add_argument("--target", choices=["full", "fp16", "sep2", "sep3", "sep4"], default="fp16",
                    help="target KV for rq/quantspec: exact (default; isolates draft error), full reconstruction, or "
                         "sepB (quantspec only: independent B-bit RTN target + separate draft copy)")
    ap.add_argument("--draft-new", choices=["quant", "fp16"], default="quant",
                    help="draft's provisional KV for its own proposed tokens")
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=2048, help="prompt length (Llama-3-8B max 8192)")
    ap.add_argument("--n-new", type=int, default=90)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--dataset", choices=["pg19", "wikitext"], default="pg19")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0], help="layers are split evenly across these")
    ap.add_argument("--model-ctxs", type=int, nargs="+", default=[2048, 8192, 32768, 131072])
    ap.add_argument("--model-batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.group is None:
        args.group = 128 if args.rep == "quantspec" else 32
    torch.cuda.set_device(args.gpus[0])
    t0 = time.time()
    model = Llama(args.model, devices=[f"cuda:{g}" for g in args.gpus], draft_wbits=args.draft_wbits if args.draft_wbits.startswith("n") else int(args.draft_wbits),
                  max_seq_len=args.ctx + args.n_new + args.gamma + 8)
    tok = AutoTokenizer.from_pretrained(args.model)
    rep = build_rep(args, model)
    prompts = load_prompts(tok, args.ctx, args.num_prompts, args.dataset)
    print(f"model {args.model} loaded in {time.time()-t0:.0f}s | rep={rep.name} "
          f"draft={rep.bits_draft:.2f}b target={rep.bits_target:.2f}b W{args.draft_wbits} | "
          f"γ={args.gamma} T={args.temperature} ctx={args.ctx} prompts={len(prompts)}")

    exact = FP16Rep()
    # warm-up (kernels, allocator)
    autoregressive(model, exact, prompts[0][:256], 8)
    speculative(model, rep, prompts[0][:256], 8, args.gamma, args.temperature)

    rows = []
    for i, p in enumerate(prompts):
        ar_out, ar = autoregressive(model, exact, p, args.n_new, args.temperature, args.seed + i)
        sp_out, sp = speculative(model, rep, p, args.n_new, args.gamma, args.temperature, args.seed + i)
        match = sum(a == b for a, b in zip(ar_out, sp_out)) / len(ar_out)
        first_div = next((j for j, (a, b) in enumerate(zip(ar_out, sp_out)) if a != b), len(ar_out))
        r = {"prompt": i, "accept_rate": sp["accept_rate"], "tokens_per_iter": sp["tokens_per_iter"],
             "accept_hist": sp["accept_hist"], "ar_tok_s": ar["decode_tok_per_s"],
             "spec_tok_s": sp["decode_tok_per_s"], "measured_speedup": sp["decode_tok_per_s"] / ar["decode_tok_per_s"],
             "draft_s": sp["time"]["draft"], "verify_s": sp["time"]["verify"],
             "greedy_match": match, "first_divergence": first_div}
        rows.append(r)
        print(f"[{i}] α={r['accept_rate']:.3f} tok/iter={r['tokens_per_iter']:.2f} hist={r['accept_hist']} | "
              f"AR {r['ar_tok_s']:.1f} tok/s, spec {r['spec_tok_s']:.1f} tok/s (×{r['measured_speedup']:.2f}) | "
              f"draft {r['draft_s']:.2f}s verify {r['verify_s']:.2f}s"
              + (f" | greedy match {match:.3f} (first diff @{first_div})" if args.temperature == 0 else ""),
              flush=True)

    mean = lambda k: statistics.mean(r[k] for r in rows)
    tpi = mean("tokens_per_iter")
    summary = {"accept_rate": mean("accept_rate"), "tokens_per_iter": tpi,
               "measured_speedup": mean("measured_speedup"), "greedy_match": mean("greedy_match")}
    modeled = {f"ctx{c}_b{b}": modeled_speedup(model, rep, tpi, args.gamma, c, b)
               for c in args.model_ctxs for b in args.model_batches}
    print(f"\nmean: α={summary['accept_rate']:.3f} tokens/iter={tpi:.2f} measured ×{summary['measured_speedup']:.2f}"
          + (f" greedy match={summary['greedy_match']:.3f}" if args.temperature == 0 else ""))
    print("modeled speed-up (memory traffic, same-precision weights):")
    print("  ctx \\ batch " + "".join(f"{b:>8}" for b in args.model_batches))
    for c in args.model_ctxs:
        print(f"  {c:>11} " + "".join(f"{modeled[f'ctx{c}_b{b}']:>8.2f}" for b in args.model_batches))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "rep": rep.name, "bits_draft": rep.bits_draft,
                   "bits_target": rep.bits_target, "summary": summary, "modeled": modeled, "rows": rows},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
