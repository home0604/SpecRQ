"""Practical baseline: AR decode step of a SYNTHETIC Llama-3.1-8B with Marlin FP16 x INT4 (g128) linears (random
weights, real embed / norms / lm_head, fp16, same runner: CUDA graph, fused glue, fp32 decode attention).
Timing only. Usage: python -m specrq.infer.marlin_ar --run NAME
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

from specrq.infer.decode import ARDecoder
from specrq.infer.model import NestModel

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    m = NestModel.synthetic(a.model, dtype=torch.float16, linear_impl="marlin", verbose=False)
    prompt = torch.randint(0, m.cfg.vocab_size, (a.P,), device="cuda",
                           generator=torch.Generator(device="cuda").manual_seed(0))
    ar = ARDecoder(m, a.P + a.n_new + 16)
    ar.generate(prompt, 8)
    st = [ar.generate(prompt, a.n_new)[1] for _ in range(a.repeats)]
    step = [1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st]
    r = {"step_us": statistics.median(step), "step_us_range": [min(step), max(step)],
         "tokens_per_s": statistics.median(s["tokens_per_s"] for s in st),
         "prefill_s": statistics.median(s["t_prefill_s"] for s in st),
         "linear_bytes_per_token": m.linear_bytes(1), "lm_head_bytes": m.lm_head.numel() * 2}
    print(r)
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": "SYNTHETIC Llama-3.1-8B, Marlin INT4 g128 linears (random), fp16, AR only; timing only",
            "model": a.model, "P": a.P, "n_new": a.n_new, "repeats": a.repeats}, metrics=r, model_dir=a.model))


if __name__ == "__main__":
    main()
