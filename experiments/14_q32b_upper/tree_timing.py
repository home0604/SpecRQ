"""E5/E4 tree-drafting timing: forward-time M-curves on the SYNTHETIC Qwen2.5-32B nested 2-bit model (d8 K256 L2,
Hadamard, same construction and runner as q32b_upper.py / results/E4/q32b_L2_synth_prefix_layerskip.json).
  target AR k2 step (ARDecoder loop, --n-new tokens), measured at start and end -> T_AR
  verify  : target k2 forward over M tokens at context P (all-row lm_head + argmax), M in --verify-ms
  draft   : k1 forward over M tokens (all-row lm_head over full vocab or the FR-Spec V_d subset + top-8), M in --draft-ms
Each forward is a CUDA graph of --iters calls, replayed --reps times (warm-up/capture excluded) -> median [min, max].
[I] A tree verify/draft needs a tree attention mask instead of the causal mask used here; the attention share of a
forward at this context is small, so the mask's cost difference is not measured. M > 16 takes the non-compiled glue
path of NestModel.forward (as it would in a real run).
Usage: python experiments/14_q32b_upper/tree_timing.py --out JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "14_q32b_upper"))

import torch  # noqa: E402
import torch._dynamo  # noqa: E402

torch._dynamo.config.cache_size_limit = 256  # one compiled specialization per M (default 8 is hit at M=12)
torch._dynamo.config.accumulated_cache_size_limit = 1024

from q32b_upper import ar_rows, gpu_util, graph_us  # noqa: E402
from specrq.infer.model import NestModel  # noqa: E402
from specrq.nestkern.gpucheck import cotenants  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_32b_instruct")
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--verify-ms", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--draft-ms", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    co0, util0 = cotenants() or "none", gpu_util()
    t0 = time.time()
    m = NestModel.synthetic(a.model, L=2, hadamard=True, verbose=True)
    from specrq.infer.frspec import default_freq_dir, load_subset
    m.set_draft_vocab(load_subset(default_freq_dir(m.cfg), a.vd, None))
    V = m.lm_head.shape[0]
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(0, m.cfg.vocab_size, (a.P,), device="cuda", generator=gen)
    Mx = max(a.verify_ms + a.draft_ms)
    T = a.P + a.n_new + Mx + 24
    res = {"args": vars(a), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "ar": [], "verify": [],
           "draft": []}
    res["ar"].append(ar_rows(m, prompt, T, None, None, a.n_new, 3))
    print("[ar start]", res["ar"][-1], flush=True)
    cache = m.new_cache(T)
    m.prefill(prompt, cache, m.L)
    pos = torch.tensor(a.P, device="cuda")
    toks = torch.randint(0, m.cfg.vocab_size, (Mx,), device="cuda", generator=gen)
    with torch.no_grad():
        for M in a.verify_ms:
            r = {"k": m.L, "M": M, "us": graph_us(lambda: m.forward(toks[:M], pos, cache, m.L).argmax(-1),
                                                  iters=a.iters, reps=a.reps)}
            res["verify"].append(r)
            print("[verify]", r, flush=True)
        for vd in (a.vd, None):
            for M in a.draft_ms:
                r = {"k": 1, "vd": vd or V, "M": M,
                     "us": graph_us(lambda: m.to_token_ids(m.forward(toks[:M], pos, cache, 1, vd=vd).topk(8, -1)[1],
                                                           vd), iters=a.iters, reps=a.reps)}
                res["draft"].append(r)
                print("[draft]", r, flush=True)
    del cache
    torch.cuda.empty_cache()
    res["ar"].append(ar_rows(m, prompt, T, None, None, a.n_new, 3))
    print("[ar end]", res["ar"][-1], flush=True)
    res["T_ar_ms"] = 0.5 * (res["ar"][0]["step_us"]["median"] + res["ar"][-1]["step_us"]["median"]) / 1e3
    res["co_tenant"] = [co0, cotenants() or "none"]
    res["gpu_util"] = [util0, gpu_util()]
    res["wall_s"] = time.time() - t0
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    json.dump(res, open(a.out, "w"), indent=1)
    print("[tree_timing] done", res["T_ar_ms"], res["wall_s"], flush=True)


if __name__ == "__main__":
    main()
