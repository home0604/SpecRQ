"""I3/I4: greedy AR baseline vs greedy self-speculative decoding on NestModel (CUDA graphs, static KV).

For each prompt: AR target (k_t) and spec (gamma, k_d) runs; output checked against AR with first_divergence.
Timing (Notion 03): warm-up run per decoder excluded, `repeats` timed runs per (prompt, config), median and
[min, max]; decode throughput = confirmed tokens / decode wall time (draft + verify + overhead), prefill separate.
Also reports the bf16 lm_head GEMV time (M=1 and M=gamma+1) and its share of an AR step.
Usage: python -m specrq.infer.run_decode --ckpt DIR [--gammas 1 2 4] [--kds 3 2] [--n-new 128] [--run NAME]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from specrq.infer.decode import ARDecoder, SpecDecoder, first_divergence
from specrq.infer.model import NestLinear, NestModel

REPO = Path(__file__).resolve().parents[2]
PROMPTS = [
    "The history of natural language processing generally started in the 1950s, although work can be found from "
    "earlier periods. In 1950, Alan Turing published an article titled Computing Machinery and Intelligence which",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods "
    "from carbon dioxide and water. In plants, photosynthesis generally involves the green pigment chlorophyll and",
    "def quicksort(arr):\n    \"\"\"Sort a list of integers using the quicksort algorithm.\"\"\"\n    if len(arr) <= 1:\n",
]


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def lm_head_us(model, M, iters=50):
    h = torch.randn(M, model.cfg.hidden_size, device=model.device, dtype=model.dtype)
    for _ in range(3):
        F.linear(h, model.lm_head)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            F.linear(h, model.lm_head)
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--draft-sd", default=None, help="DbD draft_state_dict file (use_draft=True)")
    ap.add_argument("--kt", type=int, default=None)
    ap.add_argument("--kds", type=int, nargs="+", default=[3, 2])
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run", default=None)
    ap.add_argument("--nondet", action="store_true", help="fp32-atomic split-K (faster, not bitwise reproducible)")
    a = ap.parse_args(argv)
    NestLinear.DETERMINISTIC = not a.nondet
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = NestModel(a.model, a.ckpt)
    use_draft = False
    if a.draft_sd:
        n = m.load_draft(torch.load(a.draft_sd, map_location="cpu", weights_only=False))
        use_draft = n > 0
        print(f"[run_decode] installed draft codebooks on {n} linears")
    prompts = [tok(p, return_tensors="pt").input_ids[0].cuda() for p in PROMPTS]
    T = max(p.numel() for p in prompts) + a.n_new + max(a.gammas) + 8
    rows = []

    ar = ARDecoder(m, T, k=a.kt)
    ar.generate(prompts[0], 16)  # warm-up
    ar_ref = {}
    for pi, p in enumerate(prompts):
        runs = [ar.generate(p, a.n_new) for _ in range(a.repeats)]
        toks = runs[0][0]
        ar_ref[pi] = toks
        det = all(r[0] == toks for r in runs)
        st = [r[1] for r in runs]
        rows.append({"mode": "ar", "prompt": pi, "kt": a.kt or m.L, "reproducible": det,
                     "tokens_per_s": med([s["tokens_per_s"] for s in st]),
                     "t_decode_s": med([s["t_decode_s"] for s in st]), "t_prefill_s": med([s["t_prefill_s"] for s in st]),
                     "step_us": med([1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st])})
        print(f"[AR] prompt {pi}: {rows[-1]['tokens_per_s']['median']:.1f} tok/s, step "
              f"{rows[-1]['step_us']['median']:.1f} us, reproducible={det}", flush=True)
    ar_tps = {r["prompt"]: r["tokens_per_s"]["median"] for r in rows}

    for g in a.gammas:
        for kd in a.kds:
            sd = SpecDecoder(m, T, g, kd, a.kt, use_draft)
            sd.generate(prompts[0], 16)  # warm-up
            for pi, p in enumerate(prompts):
                runs = [sd.generate(p, a.n_new) for _ in range(a.repeats)]
                st = [r[1] for r in runs]
                fd = [first_divergence(r[0], ar_ref[pi]) for r in runs]
                r = {"mode": "spec", "prompt": pi, "gamma": g, "kd": kd, "kt": a.kt or m.L, "use_draft": use_draft,
                     "first_divergence": fd, "greedy_match": [f is None for f in fd]}
                for key in ("tokens_per_s", "t_decode_s", "t_prefill_s", "tau", "accept_rate", "t_draft_round_ms",
                            "t_verify_ms", "t_overhead_ms"):
                    r[key] = med([s[key] for s in st])
                r["speedup_vs_ar"] = r["tokens_per_s"]["median"] / ar_tps[pi]
                r["share"] = {k2: statistics.median(s[k2] for s in st) / statistics.median(s["t_decode_s"] for s in st)
                              for k2 in ("t_draft_s", "t_verify_s", "t_overhead_s")}
                rows.append(r)
                print(f"[spec g={g} kd={kd}] prompt {pi}: {r['tokens_per_s']['median']:.1f} tok/s "
                      f"(x{r['speedup_vs_ar']:.2f} AR) tau={r['tau']['median']:.2f} acc={r['accept_rate']['median']:.2f} "
                      f"draft {r['t_draft_round_ms']['median']:.2f}ms verify {r['t_verify_ms']['median']:.2f}ms "
                      f"ovh {r['t_overhead_ms']['median']:.2f}ms first_div={fd}", flush=True)
            del sd
            torch.cuda.empty_cache()
    lm = {f"M{M}": lm_head_us(m, M) for M in sorted({1, *[g + 1 for g in a.gammas]})}
    ar_step = statistics.median(r["step_us"]["median"] for r in rows if r["mode"] == "ar")
    print(f"lm_head bf16 GEMV us: {lm}; share of AR step {lm['M1'] / ar_step:.2%}")
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": "greedy AR vs self-spec, NestModel CUDA graphs, batch 1", "model": a.model, "ckpt": a.ckpt,
            "draft_sd": a.draft_sd, "kt": a.kt or m.L, "kds": a.kds, "gammas": a.gammas, "n_new": a.n_new,
            "repeats": a.repeats, "prompts": PROMPTS, "deterministic_splitk": not a.nondet, "dtype": "bf16", "T_max": T,
            "linear_bytes_per_token": {f"k{k}": m.linear_bytes(k) for k in range(1, m.L + 1)},
            "lm_head_bytes": m.lm_head.numel() * 2},
            metrics={"rows": rows, "lm_head_us": lm, "lm_head_share_of_ar_step": lm["M1"] / ar_step},
            model_dir=a.model))


if __name__ == "__main__":
    main()
