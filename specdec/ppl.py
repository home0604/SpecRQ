"""
Target-quality check: teacher-forced perplexity of the *target* path (weights as loaded, KV via rep's
target reconstruction) on the same prompts used by run.py.

  python -m specdec.ppl --draft-wbits n8 --rep quantspec --ctx 8192 --gpus 6 7

--ref also runs the exact model (bf16 weights, FP16 KV) on the same tokens and reports, per position,
KL(exact || target) and top-1 agreement with the exact model (= teacher-forced acceptance of the target
as a γ=1 draft for the exact model).
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from specdec.kvstore import FP16Rep, KVStore
from specdec.model import Llama
from specdec.run import build_rep, load_prompts


@torch.no_grad()
def nll(model, rep, ids, chunk=2048, ref=False):
    kv = KVStore(model, rep, 1, ids.numel())
    kv_ref = KVStore(model, FP16Rep(), 1, ids.numel()) if ref else None
    tot, n, kl, agree = 0.0, 0, 0.0, 0
    for s in range(0, ids.numel() - 1, chunk):
        e = min(ids.numel(), s + chunk)
        x = ids[s:e].view(1, -1).to(model.device)
        lg = model.forward(x, s, kv, "target")[0]
        tgt = ids[s + 1:e + 1].to(model.device)
        lg = lg[: tgt.numel()]
        tot += F.cross_entropy(lg, tgt, reduction="sum").item()
        n += tgt.numel()
        if ref:
            lr = model.forward(x, s, kv_ref, "target")[0][: tgt.numel()].float()
            lp, lq = F.log_softmax(lr, -1), F.log_softmax(lg.float(), -1)
            kl += (lp.exp() * (lp - lq)).sum().item()
            agree += (lr.argmax(-1) == lg.argmax(-1)).sum().item()
            kv_ref.commit(e)
        kv.commit(e)  # block reps (QuantSpec) quantize as the context grows, like during decoding
    return tot, n, kl, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama2_7b_32k_instruct")
    ap.add_argument("--rep", default="fp16")
    ap.add_argument("--draft-wbits", choices=["2", "3", "4", "16", "n4", "n6", "n8"], default="16")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--dataset", default="pg19")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", action="store_true", help="also report KL / top-1 agreement vs the exact model")
    # rep args used by build_rep
    for a, d in [("--bits", 4), ("--group", None), ("--budget", None), ("--budget-frac", 0.25), ("--sink", 4),
                 ("--window", 32), ("--kernel", 5), ("--codebook", None), ("--m", 2), ("--target", "fp16"),
                 ("--draft-new", "quant"), ("--kv-bits", 4), ("--kv-scale", "rtn"), ("--target-stages", None), ("--rq-window", 128),
                 ("--cw-format", None), ("--cw-stages", None), ("--cw-format-rest", None)]:
        ap.add_argument(a, type=type(d) if d is not None else None, default=d)
    args = ap.parse_args()
    args.target_stages = int(args.target_stages) if args.target_stages else None
    args.cw_stages = int(args.cw_stages) if args.cw_stages else None
    if args.group is None:
        args.group = 128 if args.rep == "quantspec" else 32
    else:
        args.group = int(args.group)
    torch.cuda.set_device(args.gpus[0])
    wb = args.draft_wbits if args.draft_wbits.startswith("n") else int(args.draft_wbits)
    model = Llama(args.model, devices=[f"cuda:{g}" for g in args.gpus], draft_wbits=wb, max_seq_len=args.ctx + 8)
    tok = AutoTokenizer.from_pretrained(args.model)
    rep = build_rep(args, model)
    tot, n, kl, agree = 0.0, 0, 0.0, 0
    for p in load_prompts(tok, args.ctx, args.num_prompts, args.dataset):
        a, b, c, d = nll(model, rep, p, ref=args.ref)
        tot, n, kl, agree = tot + a, n + b, kl + c, agree + d
    ppl = math.exp(tot / n)
    print(f"target PPL ({rep.name}, W{args.draft_wbits} target weights, ctx {args.ctx}) = {ppl:.4f}")
    res = {"args": vars(args), "rep": rep.name, "ppl": ppl, "tokens": n}
    if args.ref:
        res.update(kl=kl / n, top1_agree=agree / n)
        print(f"vs exact: KL {kl / n:.5f} nats/token, top-1 agreement {agree / n:.4f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
