"""P2 check: fused glue (add+RMSNorm+FWHT, SiLU*mul+FWHT kernels) vs the unfused decode path on the same model.
Synthetic Llama-3.1-8B (Hadamard B=256 on every linear) by default; logits of decode-sized forwards (M in 1,2,5,8)
after a prefill. Reports rel L2 / max |diff| / top-1 agreement between the two paths (both bf16).
Usage: python -m specrq.infer.check_fused [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.infer.model import NestModel

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    m = NestModel.synthetic(a.model, verbose=False)
    g = torch.Generator(device="cuda").manual_seed(1)
    prompt = torch.randint(0, m.cfg.vocab_size, (64,), device="cuda", generator=g)
    rows = []
    for k in (5, 3, 1):
        for M in (1, 2, 5, 8):
            toks = torch.randint(0, m.cfg.vocab_size, (M,), device="cuda", generator=g)
            out = {}
            for fused in (False, True):
                m.fused_glue = fused
                c = m.new_cache(96)
                with torch.no_grad():
                    m.prefill(prompt, c, k)
                    out[fused] = m.forward(toks, torch.tensor(64, device="cuda"), c, k).float()
            d = out[True] - out[False]
            r = {"k": k, "M": M, "rel": (d.norm() / out[False].norm()).item(), "maxabs": d.abs().max().item(),
                 "top1": (out[True].argmax(-1) == out[False].argmax(-1)).float().mean().item()}
            rows.append(r)
            print(r, flush=True)
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E2", a.save, config={"setting": "P2 fused glue vs unfused decode path, SYNTHETIC "
                                                          "Llama-3.1-8B C1 + Hadamard (B=256)"}, metrics={"rows": rows},
                                   model_dir=a.model))


if __name__ == "__main__":
    main()
