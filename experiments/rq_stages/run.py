"""
RQ Stage별 Reconstruction Error & Acceptance Rate 예측
=====================================================
실제 모델의 KV cache를 RQ codebook으로 encode한 후,
draft stage 수에 따른 reconstruction quality를 측정.

측정 항목:
  1. Reconstruction error: ||K_orig - K_recon|| / ||K_orig||
  2. Max exponent error: |max_j(Q·K_orig) - max_j(Q·K_recon)| (acceptance rate 예측)
  3. Score distribution 비교

Usage:
  python exp_rq_stages.py --model /data/models/llama2_7b \
      --codebook /data/home/lsh2/ckpts/wikitext/both_quant_Key_K[12]... \
      --context-length 512 --num-samples 3
"""

import argparse
import json
import math
import os
from contextlib import contextmanager

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from rq_kv_cache import RQKVCache
from dataset import load_dataset_prompts


# ── SDPA Hook (from exp_max_exponent.py) ──

@contextmanager
def hook_sdpa():
    """SDPA를 hooking하여 각 attention layer의 (Q, K, V)를 캡처."""
    captured = []
    orig = torch.nn.functional.scaled_dot_product_attention

    def hooked(query, key, value, *args, **kwargs):
        captured.append((query.detach().clone(), key.detach().clone(), value.detach().clone()))
        return orig(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = hooked
    try:
        yield captured
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


def analyze_layer_rq(Q, K, V, rq_cache, layer_idx, draft_stages_list):
    """
    하나의 layer에 대해 RQ stage별 분석.

    Args:
        Q, K, V: (B, H, S, D) — 실제 모델에서 캡처된 텐서
        rq_cache: RQKVCache with loaded codebooks
        layer_idx: layer index
        draft_stages_list: [1, 2, 3, 4, 6, 8, 12] 등

    Returns:
        dict with per-stage metrics
    """
    B, H_q, S, D = Q.shape
    H_kv = K.shape[1]
    inv_sqrt_d = 1.0 / math.sqrt(D)

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    # Encode with RQ
    rq_cache.encode_and_store(K, V, layer_idx)

    # Target scores (original FP16 KV)
    # For prefill: each query position attends to all previous + self
    # Simplified: compute full Q×K^T (no causal mask, for error analysis)
    target_scores = torch.matmul(Q_f, K_f.transpose(-2, -1)) * inv_sqrt_d  # (B, H_q, S, S_kv)
    target_max = target_scores.amax(dim=-1)  # (B, H_q, S)

    results = {}
    for ns in draft_stages_list:
        # Decode with N stages
        K_recon = rq_cache.decode(rq_cache.key_indices[layer_idx], layer_idx, "key", ns)
        V_recon = rq_cache.decode(rq_cache.value_indices[layer_idx], layer_idx, "value", ns)

        K_recon_f = K_recon.float()
        V_recon_f = V_recon.float()

        # 1. Reconstruction error (relative L2)
        # GQA: K has H_kv heads, need to match with Q's H_q heads
        key_err = (K_f - K_recon_f).norm() / K_f.norm()
        val_err = (V_f - V_recon_f).norm() / V_f.norm()

        # 2. Max exponent error
        # Handle GQA: expand K_recon to match Q heads
        if H_q != H_kv:
            repeat = H_q // H_kv
            K_recon_expanded = K_recon_f.repeat_interleave(repeat, dim=1)
        else:
            K_recon_expanded = K_recon_f

        draft_scores = torch.matmul(Q_f, K_recon_expanded.transpose(-2, -1)) * inv_sqrt_d
        draft_max = draft_scores.amax(dim=-1)

        # Handle GQA for target too
        if H_q != H_kv:
            K_expanded = K_f.repeat_interleave(repeat, dim=1)
            target_scores_full = torch.matmul(Q_f, K_expanded.transpose(-2, -1)) * inv_sqrt_d
            target_max_full = target_scores_full.amax(dim=-1)
        else:
            target_max_full = target_max

        max_exp_error = (target_max_full - draft_max).abs()

        # 3. Score distribution stats
        score_diff = (target_scores.reshape(-1) - draft_scores.reshape(-1)).float()

        results[ns] = {
            "key_recon_err": key_err.item(),
            "val_recon_err": val_err.item(),
            "max_exp_error_mean": max_exp_error.mean().item(),
            "max_exp_error_p50": torch.quantile(max_exp_error.reshape(-1).float(), 0.5).item(),
            "max_exp_error_p90": torch.quantile(max_exp_error.reshape(-1).float(), 0.9).item(),
            "max_exp_error_p99": torch.quantile(max_exp_error.reshape(-1).float(), 0.99).item(),
            "max_exp_error_max": max_exp_error.amax().item(),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RQ stage별 reconstruction error & acceptance rate 예측")
    parser.add_argument("--model", type=str, default="/data/models/llama2_7b")
    parser.add_argument("--codebook", type=str,
                        default="/data/home/lsh2/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256]_Value_K[12]C[256]D[256]_SubVec1_Stages1/meta_model.pt")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["pg19", "wikitext", "c4", "gsm8k", "default"])
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache-dir", type=str, default="/data/models")
    parser.add_argument("--draft-stages", type=int, nargs="+",
                        default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    model_name = args.model
    if "/" in model_name or os.sep in model_name:
        model_name = os.path.abspath(model_name)

    print("=" * 80)
    print("RQ Stage Analysis: Reconstruction Error & Max Exponent Error")
    print(f"  Model: {model_name}")
    print(f"  Codebook: {args.codebook}")
    print(f"  Dataset: {args.dataset}, context: {args.context_length}")
    print(f"  Draft stages to test: {args.draft_stages}")
    print("=" * 80)

    # ── Load RQ codebook ──
    rq_cache = RQKVCache(device=args.device)
    rq_cache.load_codebooks(args.codebook, args.num_kv_heads, args.head_dim)

    # ── Load model ──
    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    print("\nLoading model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=args.device,
        attn_implementation="sdpa", use_safetensors=False, **load_kw,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──
    prompts = load_dataset_prompts(
        args.dataset, tokenizer, args.context_length,
        num_samples=args.num_samples, max_new_tokens=10,
    )
    if not prompts:
        prompts = [tokenizer.encode("The future of artificial intelligence is",
                                     return_tensors="pt")]

    # ── Run analysis ──
    all_results = []

    for s_idx, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids[:, :args.context_length].to(args.device)
        seq_len = input_ids.shape[1]

        print(f"\n{'=' * 80}")
        print(f"Sample {s_idx + 1}/{len(prompts)} (seq_len={seq_len})")
        print(f"{'=' * 80}")

        with hook_sdpa() as captured:
            with torch.no_grad():
                model(input_ids)

        num_layers = len(captured)
        sample_results = []

        # Analyze each layer
        for l_idx in range(num_layers):
            Q, K, V = captured[l_idx]
            layer_res = analyze_layer_rq(Q, K, V, rq_cache, l_idx, args.draft_stages)
            sample_results.append(layer_res)

        # ── Print per-stage summary (averaged over layers) ──
        print(f"\n{'Stage':>6s} │ {'Key Err':>8s} {'Val Err':>8s} │ "
              f"{'MaxExp Mean':>11s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s}")
        print("─" * 75)

        for ns in args.draft_stages:
            avg_key = np.mean([r[ns]["key_recon_err"] for r in sample_results])
            avg_val = np.mean([r[ns]["val_recon_err"] for r in sample_results])
            avg_me_mean = np.mean([r[ns]["max_exp_error_mean"] for r in sample_results])
            avg_me_p50 = np.mean([r[ns]["max_exp_error_p50"] for r in sample_results])
            avg_me_p90 = np.mean([r[ns]["max_exp_error_p90"] for r in sample_results])
            avg_me_p99 = np.mean([r[ns]["max_exp_error_p99"] for r in sample_results])
            avg_me_max = np.max([r[ns]["max_exp_error_max"] for r in sample_results])

            bpa = ns * math.log2(rq_cache.codebook_size) / rq_cache.codebook_dim
            print(f"{ns:>5d}  │ {avg_key:>8.4f} {avg_val:>8.4f} │ "
                  f"{avg_me_mean:>11.4f} {avg_me_p50:>8.4f} {avg_me_p90:>8.4f} "
                  f"{avg_me_p99:>8.4f} {avg_me_max:>8.4f}  (BPA={bpa:.3f})")

        # ── Per-layer detail for best/worst stages ──
        print(f"\nPer-layer Key Reconstruction Error:")
        print(f"{'Layer':>6s}", end="")
        for ns in args.draft_stages:
            print(f" │ {f's={ns}':>8s}", end="")
        print()
        print("─" * (8 + 11 * len(args.draft_stages)))

        for l_idx in range(num_layers):
            print(f"{l_idx:>6d}", end="")
            for ns in args.draft_stages:
                err = sample_results[l_idx][ns]["key_recon_err"]
                print(f" │ {err:>8.4f}", end="")
            print()

        all_results.append(sample_results)
        del captured
        torch.cuda.empty_cache()

    # ── Overall summary ──
    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY (across all samples × layers)")
    print(f"{'=' * 80}")

    print(f"\n  QuantSpec INT4 비교 (from exp_max_exponent):")
    print(f"    INT4 upper-4bit: key_err ≈ 0.02-0.05, max_exp_error ≈ 0.1-0.5")
    print(f"    → 이 수치보다 낮으면 RQ가 INT4보다 나음\n")

    print(f"  {'Stages':>7s} {'BPA':>6s} │ {'Key Err':>8s} {'Val Err':>8s} │ {'MaxExp':>8s} │ 비고")
    print(f"  {'─' * 60}")

    for ns in args.draft_stages:
        all_key = [r[ns]["key_recon_err"] for sample in all_results for r in sample]
        all_val = [r[ns]["val_recon_err"] for sample in all_results for r in sample]
        all_me = [r[ns]["max_exp_error_mean"] for sample in all_results for r in sample]
        bpa = ns * math.log2(rq_cache.codebook_size) / rq_cache.codebook_dim

        avg_key = np.mean(all_key)
        avg_val = np.mean(all_val)
        avg_me = np.mean(all_me)

        note = ""
        if ns == 2:
            note = "← draft candidate"
        elif ns == max(args.draft_stages):
            note = "← target (full)"

        print(f"  {ns:>7d} {bpa:>6.3f} │ {avg_key:>8.4f} {avg_val:>8.4f} │ {avg_me:>8.4f} │ {note}")

    # ── Save ──
    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "config": vars(args),
                "rq_config": {
                    "stages_key": rq_cache.num_stages_key,
                    "stages_value": rq_cache.num_stages_value,
                    "codebook_size": rq_cache.codebook_size,
                    "codebook_dim": rq_cache.codebook_dim,
                    "num_pq": rq_cache.num_pq_codebooks,
                },
                "results": [[{str(k): v for k, v in layer.items()} for layer in sample]
                             for sample in all_results],
            }, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
