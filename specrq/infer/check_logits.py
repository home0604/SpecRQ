"""I2: runner logits vs the DbD/HF PyTorch path (main.py --load equivalent: src.modelutils.get_model(path, load)).

For each prefix k (default 5 = target and 3):
  prefill  : runner.forward on T tokens (mma, chunks of 16) vs HF forward on the same ids
  decode   : runner prefill on the first T0 tokens, then M=1 steps (gemv/mma dispatch + KV cache) vs HF logits
Noise floor: HF sdpa vs HF eager attention on the same quantized model (both bf16) -- the size of differences that
pure bf16 kernel-order effects produce. Accuracy reference: the same DbD model in fp32 (codebooks/scales/activations
fp32, eager attention); runner and the bf16 HF paths are each compared to it. Metrics: rel L2, max |diff|, top-1 agreement, mean KL(ref || runner).
Usage: python -m specrq.infer.check_logits --ckpt DIR [--model /data/models/qwen2.5_0.5b_instruct] [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.infer.model import DBD, NestModel

REPO = Path(__file__).resolve().parents[2]
TEXT = ("The history of natural language processing generally started in the 1950s, although work can be found "
        "from earlier periods. In 1950, Alan Turing published an article titled Computing Machinery and "
        "Intelligence which proposed what is now called the Turing test as a criterion of intelligence. "
        "The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian "
        "sentences into English. The authors claimed that within three or five years, machine translation would be "
        "a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, funding for "
        "machine translation was dramatically reduced.")


def metrics(a, ref):
    a, ref = a.float(), ref.float()
    lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(a, -1)
    return {"rel": ((a - ref).norm() / ref.norm()).item(), "maxabs": (a - ref).abs().max().item(),
            "top1": (a.argmax(-1) == ref.argmax(-1)).float().mean().item(),
            "kl": (lp.exp() * (lp - lq)).sum(-1).mean().item()}


def set_k(model, k):
    from src.aq import QuantizedLinear
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m.active_num_codebooks = k


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 3])
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--T0", type=int, default=96)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(DBD))
    from src.modelutils import get_model
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(TEXT, return_tensors="pt").input_ids[0]
    while ids.numel() < a.T:
        ids = torch.cat([ids, ids])
    ids = ids[: a.T].cuda()

    runner = NestModel(a.model, a.ckpt)
    refs = {impl: get_model(a.model, a.ckpt, "auto", attn_implementation=impl).cuda().eval() for impl in ("sdpa", "eager")}
    refs["fp32"] = get_model(a.model, a.ckpt, "auto", attn_implementation="eager").float().cuda().eval()
    out = {}
    with torch.no_grad():
        for k in a.ks:
            for m in refs.values():
                set_k(m, k)
            ref = refs["sdpa"](ids[None]).logits[0]
            ref_eager = refs["eager"](ids[None]).logits[0]
            ref32 = refs["fp32"](ids[None]).logits[0]
            cache = runner.new_cache(a.T + 8)
            pos0 = torch.zeros((), dtype=torch.long, device="cuda")
            pre = runner.forward(ids, pos0, cache, k)
            # incremental decode: prefill T0 then M=1 steps
            cache2 = runner.new_cache(a.T + 8)
            runner.forward(ids[: a.T0], pos0, cache2, k, last_only=True)
            steps = []
            for t in range(a.T0, a.T):
                steps.append(runner.forward(ids[t:t + 1], torch.tensor(t, device="cuda"), cache2, k)[0])
            dec = torch.stack(steps)
            # verify-shaped calls: blocks of 5 tokens (M=5, compiled decode path) after the same T0 prefill
            cache3 = runner.new_cache(a.T + 8)
            runner.forward(ids[: a.T0], pos0, cache3, k, last_only=True)
            ver = torch.cat([runner.forward(ids[t:min(t + 5, a.T)], torch.tensor(t, device="cuda"), cache3, k)
                             for t in range(a.T0, a.T, 5)])
            r = {"prefill_vs_hf_sdpa": metrics(pre, ref), "decode_vs_hf_sdpa": metrics(dec, ref[a.T0:]),
                 "floor_hf_eager_vs_sdpa": metrics(ref_eager, ref),
                 "floor_hf_eager_vs_sdpa_decode_rows": metrics(ref_eager[a.T0:], ref[a.T0:]),
                 "runner_prefill_vs_fp32": metrics(pre, ref32), "runner_decode_vs_fp32": metrics(dec, ref32[a.T0:]),
                 "runner_verify5_vs_fp32": metrics(ver, ref32[a.T0:]),
                 "hf_sdpa_bf16_vs_fp32": metrics(ref, ref32), "hf_eager_bf16_vs_fp32": metrics(ref_eager, ref32)}
            out[f"k{k}"] = r
            for name, m in r.items():
                print(f"k={k} {name:36} rel {m['rel']:.2e}  maxabs {m['maxabs']:.3f}  top1 {m['top1']:.4f}  "
                      f"KL {m['kl']:.2e}", flush=True)
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E2", a.save, config={"setting": "I2 runner logits vs DbD/HF path", "ckpt": a.ckpt,
                                                          "model": a.model, "T": a.T, "T0": a.T0, "ks": a.ks,
                                                          "dtype": "bf16"},
                                   metrics=out, model_dir=a.model))


if __name__ == "__main__":
    main()
