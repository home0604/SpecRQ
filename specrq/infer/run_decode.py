"""I3/I4: greedy AR baseline vs greedy self-speculative decoding on NestModel (CUDA graphs, static KV).

For each prompt: AR target (k_t) and spec (gamma, k_d) runs; output checked against AR with first_divergence.
Timing (Notion 03): warm-up run per decoder excluded, `repeats` timed runs per (prompt, config), median and
[min, max]; decode throughput = confirmed tokens / decode wall time (draft + verify + overhead), prefill separate.
Also reports the bf16 lm_head GEMV time (M=1 and M=gamma+1) and its share of an AR step.
FR-Spec: --vds V_d ... (0 = full vocab) runs every (gamma, kd) with the draft restricted to the V_d most frequent
tokens (frequency table --freq-dir, default specrq.infer.frspec.FREQ_DIRS); verify stays full-vocab.
EOS stop (default on; --no-eos-stop = previous fixed-length behaviour): stop ids = generation_config eos; a request
ends at the first confirmed stop token or n_new confirmed tokens; tokens/s = confirmed decode tokens / decode wall.
--chat adds INSTRUCT_PROMPTS (chat template) to the 3 text prompts. Draft vocab keeps chat specials (--no-chat-forced
= stop tokens only, as in results/E4/frspec_qwen05_*). Default V_d = frspec.DEFAULT_VD (32768).
Usage: python -m specrq.infer.run_decode --ckpt DIR [--gammas 1 2 4] [--kds 3 2] [--vds 0 32768] [--n-new 128] [--run NAME]
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
INSTRUCT_PROMPTS = [  # same as experiments/06_accept/draft_target_eval.py
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean at night.",
    "What are three common causes of high blood pressure?",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Give step-by-step instructions to make a cup of green tea.",
    "Why is the sky blue? Answer for a ten-year-old.",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "List five tips for writing clear technical documentation.",
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
    ap.add_argument("--vds", type=int, nargs="+", default=[32768], help="FR-Spec draft vocab sizes (0 = full)")
    ap.add_argument("--no-eos-stop", action="store_true", help="fixed n_new tokens (no EOS stop), as before")
    ap.add_argument("--no-chat-forced", action="store_true", help="draft vocab forces stop tokens only")
    ap.add_argument("--chat", action="store_true", help="also run the chat-template INSTRUCT_PROMPTS")
    ap.add_argument("--freq-dir", default=None)
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
    from specrq.infer.frspec import coverage, default_freq_dir, load_subset
    freq_dir = a.freq_dir or default_freq_dir(m.cfg)
    vd_cov = {}
    for vd in a.vds:
        if vd:
            ids = load_subset(freq_dir, vd, m.lm_head.shape[0], chat=not a.no_chat_forced)
            m.set_draft_vocab(ids)
            vd_cov[vd] = coverage(freq_dir, ids)
    prompt_texts = list(PROMPTS)
    prompts = [tok(p, return_tensors="pt").input_ids[0].cuda() for p in PROMPTS]
    if a.chat:
        for p in INSTRUCT_PROMPTS:
            s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            prompt_texts.append(s)
            prompts.append(tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda())
    eos = None
    if not a.no_eos_stop:
        from transformers import GenerationConfig
        e = GenerationConfig.from_pretrained(a.model).eos_token_id
        eos = sorted(set(e if isinstance(e, list) else [e]))
    print(f"[run_decode] eos stop ids: {eos}", flush=True)
    T = max(p.numel() for p in prompts) + a.n_new + max(a.gammas) + 8
    rows = []

    ar = ARDecoder(m, T, k=a.kt, eos=eos)
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
                     "step_us": med([1e6 * s["t_decode_s"] / max(s["n_decode_tokens"], 1) for s in st]),
                     "n_tokens": len(toks), "eos_hit": runs[0][1].get("eos_hit")})
        print(f"[AR] prompt {pi}: {rows[-1]['tokens_per_s']['median']:.1f} tok/s, step "
              f"{rows[-1]['step_us']['median']:.1f} us, reproducible={det}, n_tokens={len(toks)} "
              f"eos_hit={rows[-1]['eos_hit']}", flush=True)
    ar_tps = {r["prompt"]: r["tokens_per_s"]["median"] for r in rows}

    for g in a.gammas:
        for kd, vd in [(kd, vd) for kd in a.kds for vd in a.vds]:
            sd = SpecDecoder(m, T, g, kd, a.kt, use_draft, vd=vd or None, eos=eos)
            sd.generate(prompts[0], 16)  # warm-up
            for pi, p in enumerate(prompts):
                runs = [sd.generate(p, a.n_new) for _ in range(a.repeats)]
                st = [r[1] for r in runs]
                fd = [first_divergence(r[0], ar_ref[pi]) for r in runs]
                r = {"mode": "spec", "prompt": pi, "gamma": g, "kd": kd, "vd": vd or m.lm_head.shape[0],
                     "kt": a.kt or m.L, "use_draft": use_draft,
                     "first_divergence": fd, "greedy_match": [f is None for f in fd],
                     "n_tokens": [len(x[0]) for x in runs], "eos_hit": [x[1].get("eos_hit") for x in runs]}
                for key in ("tokens_per_s", "t_decode_s", "t_prefill_s", "tau", "accept_rate", "t_draft_round_ms",
                            "t_verify_ms", "t_overhead_ms"):
                    r[key] = med([s[key] for s in st])
                r["speedup_vs_ar"] = r["tokens_per_s"]["median"] / ar_tps[pi] if ar_tps[pi] else None
                r["share"] = {k2: statistics.median(s[k2] for s in st) / statistics.median(s["t_decode_s"] for s in st)
                              for k2 in ("t_draft_s", "t_verify_s", "t_overhead_s")}
                rows.append(r)
                print(f"[spec g={g} kd={kd} vd={vd}] prompt {pi}: {r['tokens_per_s']['median']:.1f} tok/s "
                      f"(x{r['speedup_vs_ar'] or 0:.2f} AR) n={r['n_tokens']} tau={r['tau']['median']:.2f} acc={r['accept_rate']['median']:.2f} "
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
            "repeats": a.repeats, "prompts": prompt_texts, "eos_stop_ids": eos,
            "chat_forced_draft_vocab": not a.no_chat_forced, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), "deterministic_splitk": not a.nondet, "dtype": "bf16", "T_max": T,
            "linear_bytes_per_token": {f"k{k}": m.linear_bytes(k) for k in range(1, m.L + 1)},
            "lm_head_bytes": m.lm_head.numel() * 2, "vds": a.vds, "freq_dir": freq_dir,
            "draft_vocab_bytes": m.draft_vocab_bytes(), "draft_vocab_pileval_coverage": vd_cov},
            metrics={"rows": rows, "lm_head_us": lm, "lm_head_share_of_ar_step": lm["M1"] / ar_step},
            model_dir=a.model))


if __name__ == "__main__":
    main()
