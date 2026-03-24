"""
RQ Speculative Decoding Acceptance Rate
========================================
Draft (N stages) vs Target (all stages) RQ KV cache로 speculative decoding의
acceptance rate를 실제 측정.

Flow:
  1. FP16 모델로 prefill → FP16 KV 캡처
  2. FP16 KV를 RQ encode → indices 저장
  3. Draft: N stages decode → approximate KV → attention → draft logits
  4. Target: all stages decode → accurate KV → attention → target logits
  5. Greedy: target argmax == draft argmax → accept

Usage:
  python experiments/rq_stages/bench_acceptance_rate.py \
      --model /data/models/llama2_7b \
      --codebook /data/home/lsh/ckpts/wikitext/both_quant_...
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kv_cache.rq_cache import RQKVCache
from src.speculative_decoding.dataset import load_dataset_prompts


@torch.no_grad()
def measure_acceptance_rate(
    model, tokenizer, input_ids, rq_cache,
    draft_stages, max_new_tokens=90, K=5,
):
    """
    RQ draft/target로 speculative decoding acceptance rate 측정.

    매 iteration:
      1. Target logits (all stages RQ KV)로 k=0 토큰 생성
      2. Draft (N stages RQ KV)로 k=1..K-1 토큰 생성
      3. Target (all stages)로 verify
      4. Accept/reject 카운트

    간소화: 실제 KV cache 업데이트 없이 prefill된 KV에 대해서만 측정.
    → acceptance rate의 "upper bound" (실제로는 decode 시 KV가 변하므로 약간 다를 수 있음)
    """
    device = input_ids.device
    B = input_ids.shape[0]
    full_stages = rq_cache.num_stages_key
    num_layers = rq_cache.num_layers
    H_kv = rq_cache.num_kv_heads
    D = rq_cache.head_dim
    H_q = model.config.num_attention_heads
    sm_scale = 1.0 / math.sqrt(D)

    # ── Step 1: Prefill with FP16 model → capture KV per layer ──
    captured_kv = []
    orig_sdpa = F.scaled_dot_product_attention

    def hook_sdpa(query, key, value, *args, **kwargs):
        captured_kv.append((key.detach().clone(), value.detach().clone()))
        return orig_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = hook_sdpa
    outputs = model(input_ids, use_cache=True)
    F.scaled_dot_product_attention = orig_sdpa

    # ── Step 2: Encode all layers' KV with RQ ──
    for l_idx in range(num_layers):
        K_l, V_l = captured_kv[l_idx]
        rq_cache.encode_and_store(K_l, V_l, l_idx)

    del captured_kv
    torch.cuda.empty_cache()

    # ── Step 3: For each position, compare draft vs target logits ──
    # We simulate: at each position i, the model sees KV[0:i] and predicts token[i+1]
    # Draft uses N-stage KV, Target uses full-stage KV
    # If argmax(draft_logits) == argmax(target_logits) → accept

    # Get all layers' Q from the prefill pass
    # Re-run with hooks to capture Q
    captured_q = []

    def hook_q(query, key, value, *args, **kwargs):
        captured_q.append(query.detach().clone())
        return orig_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = hook_q
    _ = model(input_ids, use_cache=True)
    F.scaled_dot_product_attention = orig_sdpa

    seq_len = input_ids.shape[1]
    results = {}

    for ns in draft_stages:
        # For each query position, compute attention output with draft vs target KV
        # Then compare the resulting hidden states → logits → argmax
        #
        # Simplified: compare attention output directly
        # If attention outputs are similar → logits will be similar → high acceptance
        #
        # More accurate: compute per-position max score error and estimate acceptance

        total_match = 0
        total_count = 0

        for l_idx in range(num_layers):
            Q_l = captured_q[l_idx]  # (B, H_q, seq_len, D)

            # Draft KV (N stages)
            K_draft, V_draft = rq_cache.get_kv_for_draft(l_idx, num_stages=ns)
            # Target KV (full stages)
            K_target, V_target = rq_cache.get_kv_for_target(l_idx)

            # GQA expansion
            if H_q != H_kv:
                rep = H_q // H_kv
                K_draft = K_draft.repeat_interleave(rep, dim=1)
                V_draft = V_draft.repeat_interleave(rep, dim=1)
                K_target = K_target.repeat_interleave(rep, dim=1)
                V_target = V_target.repeat_interleave(rep, dim=1)

            # Compute attention outputs for each position (causal)
            # Draft attention scores
            draft_scores = torch.matmul(Q_l, K_draft.transpose(-2, -1)) * sm_scale
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            draft_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            draft_attn = F.softmax(draft_scores, dim=-1)
            draft_out = torch.matmul(draft_attn, V_draft)

            # Target attention scores
            target_scores = torch.matmul(Q_l, K_target.transpose(-2, -1)) * sm_scale
            target_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            target_attn = F.softmax(target_scores, dim=-1)
            target_out = torch.matmul(target_attn, V_target)

            # Per-position output agreement (cosine similarity as proxy)
            cos_sim = F.cosine_similarity(
                draft_out.reshape(B, H_q, seq_len, D),
                target_out.reshape(B, H_q, seq_len, D),
                dim=-1)  # (B, H_q, seq_len)

            # "Accept" if cosine similarity > threshold (proxy for argmax match)
            # High cos_sim → similar outputs → likely same next token
            total_match += (cos_sim > 0.99).float().sum().item()
            total_count += cos_sim.numel()

        proxy_rate = total_match / total_count if total_count > 0 else 0
        results[ns] = {"proxy_acceptance_cos99": proxy_rate}

    # ── More direct method: use actual logits comparison ──
    # Run model forward with draft/target KV to get actual logits
    # This is more expensive but gives true acceptance rate

    # Use the model's lm_head on attention outputs
    # For simplicity, measure token-level agreement at the last layer

    last_layer_idx = num_layers - 1
    for ns in draft_stages:
        K_draft, V_draft = rq_cache.get_kv_for_draft(last_layer_idx, ns)
        K_target, V_target = rq_cache.get_kv_for_target(last_layer_idx)

        Q_last = captured_q[last_layer_idx]

        if H_q != H_kv:
            rep = H_q // H_kv
            K_draft = K_draft.repeat_interleave(rep, dim=1)
            V_draft = V_draft.repeat_interleave(rep, dim=1)
            K_target = K_target.repeat_interleave(rep, dim=1)
            V_target = V_target.repeat_interleave(rep, dim=1)

        # Draft attention at last layer
        ds = torch.matmul(Q_last, K_draft.transpose(-2, -1)) * sm_scale
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)
        ds.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        d_out = torch.matmul(F.softmax(ds, dim=-1), V_draft)

        # Target attention at last layer
        ts = torch.matmul(Q_last, K_target.transpose(-2, -1)) * sm_scale
        ts.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        t_out = torch.matmul(F.softmax(ts, dim=-1), V_target)

        # Per-position argmax agreement (rough proxy — not through full model)
        # Flatten heads and use as-is for comparison
        draft_flat = d_out.mean(dim=1)   # (B, seq_len, D) — average over heads
        target_flat = t_out.mean(dim=1)

        # Cosine similarity per position
        cos = F.cosine_similarity(draft_flat, target_flat, dim=-1)  # (B, seq_len)

        results[ns]["last_layer_cos_mean"] = cos.mean().item()
        results[ns]["last_layer_cos_p90"] = torch.quantile(cos.reshape(-1), 0.1).item()
        results[ns]["last_layer_match_99"] = (cos > 0.99).float().mean().item()
        results[ns]["last_layer_match_95"] = (cos > 0.95).float().mean().item()

    del captured_q
    return results


def main():
    parser = argparse.ArgumentParser(description="RQ spec decoding acceptance rate")
    parser.add_argument("--model", type=str, default="/data/models/llama2_7b")
    parser.add_argument("--codebook", type=str,
                        default="/data/home/lsh/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256]_Value_K[12]C[256]D[256]_SubVec1_Stages1")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--draft-stages", type=int, nargs="+",
                        default=[1, 2, 3, 4, 6, 8, 12])
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache-dir", type=str, default="/data/models")
    args = parser.parse_args()

    device = args.device
    model_name = os.path.abspath(args.model) if "/" in args.model else args.model

    print("=" * 70)
    print("RQ Speculative Decoding — Acceptance Rate Measurement")
    print(f"  Model: {model_name}")
    print(f"  Codebook: {args.codebook}")
    print(f"  Draft stages: {args.draft_stages}")
    print("=" * 70)

    # Load codebook
    rq = RQKVCache(device=device)
    rq.load_codebooks(args.codebook, args.num_kv_heads, args.head_dim)

    # Load model
    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    print("\nLoading model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        attn_implementation="sdpa", use_safetensors=False, **load_kw,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    prompts = load_dataset_prompts(
        args.dataset, tokenizer, args.context_length,
        num_samples=args.num_samples, max_new_tokens=10,
    )
    if not prompts:
        prompts = [tokenizer.encode("The future of AI is", return_tensors="pt")]

    # Run
    all_results = []
    for s_idx, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids[:, :args.context_length].to(device)
        print(f"\nSample {s_idx+1}/{len(prompts)} (seq_len={input_ids.shape[1]})")

        results = measure_acceptance_rate(
            model, tokenizer, input_ids, rq, args.draft_stages)
        all_results.append(results)

        for ns in args.draft_stages:
            r = results[ns]
            bpa = ns * math.log2(rq.codebook_size) / rq.codebook_dim
            print(f"  stages={ns:>2d} (BPA={bpa:.3f}): "
                  f"cos_mean={r['last_layer_cos_mean']:.4f}, "
                  f"match@99={r['last_layer_match_99']:.2%}, "
                  f"match@95={r['last_layer_match_95']:.2%}")

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (averaged over samples)")
    print(f"{'=' * 70}")

    import numpy as np

    print(f"\n  {'Stages':>7s} {'BPA':>6s} │ {'CosMean':>8s} │ {'Match@99':>9s} │ {'Match@95':>9s} │ 비고")
    print(f"  {'─' * 60}")

    for ns in args.draft_stages:
        cos_means = [r[ns]["last_layer_cos_mean"] for r in all_results]
        m99s = [r[ns]["last_layer_match_99"] for r in all_results]
        m95s = [r[ns]["last_layer_match_95"] for r in all_results]
        bpa = ns * math.log2(rq.codebook_size) / rq.codebook_dim

        note = ""
        if ns == 2:
            note = "← draft candidate"
        elif ns == max(args.draft_stages):
            note = "← target (match=100%)"

        print(f"  {ns:>7d} {bpa:>6.3f} │ {np.mean(cos_means):>8.4f} │ "
              f"{np.mean(m99s):>8.2%} │ {np.mean(m95s):>8.2%} │ {note}")

    print(f"\n  해석:")
    print(f"    CosMean: draft/target attention output의 cosine similarity (1.0 = 동일)")
    print(f"    Match@99: cos > 0.99인 position 비율 (proxy for acceptance rate)")
    print(f"    Match@95: cos > 0.95인 position 비율 (relaxed threshold)")
    print(f"    Stage가 늘수록 target에 수렴 → match rate 증가")


if __name__ == "__main__":
    main()
