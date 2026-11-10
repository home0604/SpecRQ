"""AR baselines on the same runner (NestModel forward, CUDA-graph ARDecoder, fp32 decode attention, fused glue,
static KV, full lm_head): C1 VQ target (bf16, DbD ckpt), real Marlin INT4 (fp16, experiments/13_int4_marlin ckpt)
and the dense model (bf16, cuBLAS). Only the linears (and the activation dtype for Marlin) differ.
Prompts = run_decode --chat --wt2 8 --no-text-prompts (D0/D2: 8 chat-template + 8 wikitext2 prompts); T_max =
max prompt + n_new + t_pad_gamma + 8 (t_pad_gamma=3 reproduces D2's T_max=203). Per model: one warm-up generate
(excluded), then `repeats` timed generates per prompt; per-prompt median [min, max]; aggregate = median over prompts.
Fixed length (EOS ignored). --order runs the models in the given order in one process (e.g. vq marlin dense vq to
check drift); each model is freed before the next is loaded. Co-tenant check at start and after every model.
Usage: python -m specrq.infer.ar_baselines --vq-ckpt DIR --marlin-ckpt FILE [--order vq marlin dense vq] [--run NAME]
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path

import torch

from specrq.infer.decode import ARDecoder
from specrq.infer.model import NestModel
from specrq.infer.run_decode import INSTRUCT_PROMPTS, med

REPO = Path(__file__).resolve().parents[2]


def build_prompts(tok, model_path, n_wt2):
    texts, prompts, kinds = [], [], []
    for p in INSTRUCT_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        texts.append(s)
        prompts.append(tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda())
        kinds.append("chat")
    sys.path.insert(0, str(REPO / "specrq" / "dbd"))
    from src.datautils import get_loaders
    wt = get_loaders("wikitext2", seed=0, model_path=model_path, seqlen=512, eval_mode=True)
    for i in range(n_wt2):
        ids = wt[0, (100 + i) * 512:(100 + i) * 512 + 64]
        texts.append(tok.decode(ids))
        prompts.append(ids.cuda())
        kinds.append("wikitext2")
    return texts, prompts, kinds


def run_ar(m, prompts, kinds, T, n_new, repeats):
    ar = ARDecoder(m, T)
    ar.generate(prompts[0], 16)  # warm-up (graph already captured in __init__)
    rows = []
    for pi, p in enumerate(prompts):
        runs = [ar.generate(p, n_new) for _ in range(repeats)]
        st = [r[1] for r in runs]
        rows.append({"prompt": pi, "kind": kinds[pi], "reproducible": all(r[0] == runs[0][0] for r in runs),
                     "tokens_per_s": med([s["tokens_per_s"] for s in st]),
                     "step_us": med([1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st]),
                     "t_prefill_s": med([s["t_prefill_s"] for s in st]), "tokens": runs[0][0]})
        print(f"  prompt {pi} ({kinds[pi]}): {rows[-1]['tokens_per_s']['median']:.2f} tok/s "
              f"[{rows[-1]['tokens_per_s']['min']:.2f}, {rows[-1]['tokens_per_s']['max']:.2f}]", flush=True)
    del ar
    agg = {}
    for kind in sorted(set(kinds)) + ["all"]:
        sel = [r for r in rows if kind == "all" or r["kind"] == kind]
        tps = [r["tokens_per_s"]["median"] for r in sel]
        agg[kind] = {"n_prompts": len(sel), "tokens_per_s": statistics.median(tps), "tokens_per_s_prompt_min": min(tps),
                     "tokens_per_s_prompt_max": max(tps),
                     "tokens_per_s_repeat_spread_max": max(r["tokens_per_s"]["max"] - r["tokens_per_s"]["min"]
                                                           for r in sel),
                     "step_us": statistics.median(r["step_us"]["median"] for r in sel)}
    return rows, agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--vq-ckpt", default="/data/lsh/ckpts/weightvq/llama31_8b/c1_had_uniform_pile_fp32_ft5")
    ap.add_argument("--marlin-ckpt", default=None)
    ap.add_argument("--order", nargs="+", default=["vq", "marlin", "dense", "vq"])
    ap.add_argument("--wt2", type=int, default=8)
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--t-pad-gamma", type=int, default=3)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from specrq.nestkern.gpucheck import cotenants, require_alone
    tok = AutoTokenizer.from_pretrained(a.model)
    texts, prompts, kinds = build_prompts(tok, a.model, a.wt2)
    T = max(p.numel() for p in prompts) + a.n_new + a.t_pad_gamma + 8
    cot = {"start": require_alone() if a.run else (cotenants() or "none")}
    results, seen = [], {}
    for name in a.order:
        torch.cuda.empty_cache()
        print(f"[ar_baselines] ===== {name}", flush=True)
        if name == "vq":
            m = NestModel(a.model, a.vq_ckpt, verbose=False)
        elif name == "marlin":
            m = NestModel.from_hf(a.model, "marlin", a.marlin_ckpt, dtype=torch.float16, verbose=False)
        elif name == "dense":
            m = NestModel.from_hf(a.model, "dense", dtype=torch.bfloat16, verbose=False)
        else:
            raise ValueError(name)
        rows, agg = run_ar(m, prompts, kinds, T, a.n_new, a.repeats)
        tag = name if name not in seen else f"{name}_rep{seen[name]}"
        seen[name] = seen.get(name, 0) + 1
        print(f"[ar_baselines] {tag}: {agg}", flush=True)
        print(f"[ar_baselines] {tag} prompt 0 output: {tok.decode(rows[0]['tokens'][:48])!r}", flush=True)
        results.append({"name": tag, "impl": name, "ckpt": m.ckpt_dir, "dtype": str(m.dtype),
                        "linear_bytes_per_token": m.linear_bytes(m.L), "lm_head_bytes": m.lm_head.numel() * 2,
                        "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "aggregate": agg, "rows": rows})
        cot[f"after_{tag}"] = cotenants() or "none"
        del m
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # greedy-output agreement between implementations (diagnostic; tokens differ legitimately across weights)
    by = {r["name"]: r for r in results}
    summ = {r["name"]: {"tokens_per_s_all": r["aggregate"]["all"]["tokens_per_s"],
                        "step_us_all": r["aggregate"]["all"]["step_us"]} for r in results}
    print("[ar_baselines] summary", summ, "co-tenants", cot, flush=True)
    if "vq" in by and "vq_rep1" in by:
        same = sum(x["tokens"] == y["tokens"] for x, y in zip(by["vq"]["rows"], by["vq_rep1"]["rows"]))
        print(f"[ar_baselines] vq vs vq_rep1 identical outputs on {same}/{len(prompts)} prompts", flush=True)
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        import os
        print("saved", save_result("E4", a.run, config={
            "setting": "greedy AR, batch 1, NestModel runner (CUDA graph, fused glue, fp32 decode attn, static KV, "
                       "full lm_head); only linears (+ activation dtype fp16 for Marlin) differ",
            "model": a.model, "vq_ckpt": a.vq_ckpt, "marlin_ckpt": a.marlin_ckpt, "order": a.order,
            "n_new": a.n_new, "repeats": a.repeats, "T_max": T, "prompt_kinds": kinds, "prompts": texts,
            "main_metric": "fixed length n_new (EOS ignored); tokens/s = (n_new-1) / decode wall",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "co_tenants": cot},
            metrics={"summary": summ, "models": [{k: v for k, v in r.items()} for r in results]},
            model_dir=a.model))


if __name__ == "__main__":
    main()
