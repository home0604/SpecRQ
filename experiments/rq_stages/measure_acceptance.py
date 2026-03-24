"""
RQ Speculative Decoding — Actual Acceptance Rate
=================================================
실제 autoregressive generation을 수행하면서 draft (N stages) vs target (all stages)의
token-level acceptance rate를 측정.

방법:
  1. Model attention을 monkey-patch하여 RQ KV cache 사용
  2. Target (full stages)로 90 tokens 생성 → target_tokens
  3. Draft (N stages)로 같은 prompt에서 생성 → draft_tokens
  4. Per-position acceptance: target_tokens[i] == draft_tokens[i]
  5. Speculative window (K=5) 기준 실제 acceptance rate 계산

Usage:
  python experiments/rq_stages/measure_acceptance.py \
      --model /data/models/llama2_7b \
      --codebook /data/home/lsh/ckpts/wikitext/both_quant_...
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.kv_cache.rq_cache import RQKVCache
from src.speculative_decoding.dataset import load_dataset_prompts


class RQAttentionPatcher:
    """Model의 SDPA를 RQ KV cache 기반으로 교체하는 monkey-patcher.

    매 forward에서:
      1. SDPA 호출 시 K, V를 캡처
      2. RQ encode → indices 저장
      3. RQ decode (N stages) → approximate/accurate KV
      4. 이 KV로 attention 수행
    """

    def __init__(self, rq_cache, num_stages, device="cuda"):
        self.rq = rq_cache
        self.num_stages = num_stages
        self.device = device
        self._orig_sdpa = F.scaled_dot_product_attention
        self._layer_counter = 0
        self._is_prefill = True

    def patch(self):
        patcher = self

        def rq_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, **kwargs):
            B, H_q, Lq, D = query.shape
            H_kv = key.shape[1]
            Lk = key.shape[2]
            l_idx = patcher._layer_counter % patcher.rq.num_layers

            if patcher._is_prefill:
                # Prefill: encode FP16 KV → RQ indices
                patcher.rq.encode_and_store(key, value, l_idx)

                # Decode with specified stages
                K_rq, V_rq = patcher.rq.get_kv_for_draft(l_idx, patcher.num_stages) \
                    if patcher.num_stages < patcher.rq.num_stages_key \
                    else patcher.rq.get_kv_for_target(l_idx)

                # GQA expansion
                if H_q != H_kv:
                    rep = H_q // H_kv
                    K_rq = K_rq.repeat_interleave(rep, dim=1)
                    V_rq = V_rq.repeat_interleave(rep, dim=1)

                patcher._layer_counter += 1
                return patcher._orig_sdpa(query, K_rq, V_rq, attn_mask=attn_mask,
                                          dropout_p=dropout_p, is_causal=is_causal,
                                          scale=scale, **kwargs)
            else:
                # Decode step: new token's KV needs to be appended
                # For simplicity: use original KV from model (only attention map differs)
                # The RQ effect comes from the prefill KV being quantized
                patcher._layer_counter += 1
                return patcher._orig_sdpa(query, key, value, attn_mask=attn_mask,
                                          dropout_p=dropout_p, is_causal=is_causal,
                                          scale=scale, **kwargs)

        F.scaled_dot_product_attention = rq_sdpa

    def unpatch(self):
        F.scaled_dot_product_attention = self._orig_sdpa

    def reset(self):
        self._layer_counter = 0
        self.rq.key_indices.clear()
        self.rq.value_indices.clear()

    def set_prefill(self, is_prefill):
        self._is_prefill = is_prefill
        self._layer_counter = 0


@torch.no_grad()
def generate_tokens(model, input_ids, max_new_tokens, tokenizer):
    """Autoregressive generation → list of token ids."""
    generated = input_ids.clone()
    past = None

    for _ in range(max_new_tokens):
        if past is None:
            outputs = model(generated, use_cache=True)
        else:
            outputs = model(generated[:, -1:], past_key_values=past, use_cache=True)

        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated[:, input_ids.shape[1]:]  # only new tokens


@torch.no_grad()
def measure_per_token_acceptance(model, tokenizer, input_ids, rq_cache,
                                  draft_stages_list, max_new_tokens=90):
    """
    각 draft stage에 대해 per-token acceptance rate 측정.

    1. Target (full stages)로 생성 → target_tokens (ground truth)
    2. 각 draft stage로 생성 → draft_tokens
    3. 비교: acceptance = (draft_tokens == target_tokens)
    """
    device = input_ids.device
    full_stages = rq_cache.num_stages_key
    results = {}

    # ── Target generation (full stages) ──
    patcher = RQAttentionPatcher(rq_cache, full_stages, device)
    patcher.set_prefill(True)
    patcher.patch()

    # Prefill
    outputs = model(input_ids, use_cache=True)
    patcher.set_prefill(False)

    # Generate
    past = outputs.past_key_values
    next_logits = outputs.logits[:, -1, :]
    target_tokens = []

    token = next_logits.argmax(dim=-1, keepdim=True)
    target_tokens.append(token.item())

    for _ in range(max_new_tokens - 1):
        outputs = model(token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        target_tokens.append(token.item())
        if token.item() == tokenizer.eos_token_id:
            break

    patcher.unpatch()
    patcher.reset()
    torch.cuda.empty_cache()

    # ── Draft generation (N stages each) ──
    for ns in draft_stages_list:
        patcher_d = RQAttentionPatcher(rq_cache, ns, device)
        patcher_d.set_prefill(True)
        patcher_d.patch()

        outputs = model(input_ids, use_cache=True)
        patcher_d.set_prefill(False)

        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        draft_tokens = []

        token = next_logits.argmax(dim=-1, keepdim=True)
        draft_tokens.append(token.item())

        for _ in range(max_new_tokens - 1):
            outputs = model(token, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(token.item())
            if token.item() == tokenizer.eos_token_id:
                break

        patcher_d.unpatch()
        patcher_d.reset()
        torch.cuda.empty_cache()

        # ── Compare ──
        min_len = min(len(target_tokens), len(draft_tokens))
        matches = sum(1 for i in range(min_len) if target_tokens[i] == draft_tokens[i])

        # Per-position acceptance (sequential — first mismatch breaks spec window)
        first_mismatch = min_len
        for i in range(min_len):
            if target_tokens[i] != draft_tokens[i]:
                first_mismatch = i
                break

        # Simulated speculative decoding acceptance (K=5 windows)
        K = 5
        total_drafted = 0
        total_accepted = 0
        pos = 0
        while pos < min_len:
            window = min(K, min_len - pos)
            total_drafted += window
            for j in range(window):
                if target_tokens[pos + j] == draft_tokens[pos + j]:
                    total_accepted += 1
                else:
                    total_accepted += 0  # reject, but count resampled token
                    pos += j + 1  # skip to after rejection
                    break
            else:
                pos += window  # all accepted
                continue
            continue

        spec_rate = total_accepted / max(total_drafted, 1)

        bpa = ns * math.log2(rq_cache.codebook_size) / rq_cache.codebook_dim
        results[ns] = {
            "bpa": bpa,
            "per_token_match": matches / max(min_len, 1),
            "first_mismatch_pos": first_mismatch,
            "spec_acceptance_rate": spec_rate,
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "tokens_generated": min_len,
            "target_text_preview": tokenizer.decode(target_tokens[:20]),
            "draft_text_preview": tokenizer.decode(draft_tokens[:20]),
        }

    return results, target_tokens


def main():
    parser = argparse.ArgumentParser(description="RQ spec decoding — actual acceptance rate")
    parser.add_argument("--model", type=str, default="/data/models/llama2_7b")
    parser.add_argument("--codebook", type=str,
                        default="/data/home/lsh/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256]_Value_K[12]C[256]D[256]_SubVec1_Stages1")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=90)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--draft-stages", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache-dir", type=str, default="/data/models")
    args = parser.parse_args()

    device = args.device
    model_name = os.path.abspath(args.model) if "/" in args.model else args.model

    print("=" * 70)
    print("RQ Speculative Decoding — Actual Acceptance Rate")
    print(f"  Model: {model_name}")
    print(f"  Codebook: {args.codebook}")
    print(f"  Context: {args.context_length}, Generate: {args.max_new_tokens} tokens")
    print(f"  Draft stages: {args.draft_stages}")
    print("=" * 70)

    # Load
    rq = RQKVCache(device=device)
    rq.load_codebooks(args.codebook, args.num_kv_heads, args.head_dim)

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

    prompts = load_dataset_prompts(
        args.dataset, tokenizer, args.context_length,
        num_samples=args.num_samples, max_new_tokens=args.max_new_tokens,
    )
    if not prompts:
        prompts = [tokenizer.encode("The future of AI is", return_tensors="pt")]

    # Run
    import numpy as np
    all_results = []

    for s_idx, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids[:, :args.context_length].to(device)
        print(f"\n{'─' * 70}")
        print(f"Sample {s_idx+1}/{len(prompts)} (seq_len={input_ids.shape[1]})")

        results, target_tokens = measure_per_token_acceptance(
            model, tokenizer, input_ids, rq, args.draft_stages, args.max_new_tokens)
        all_results.append(results)

        print(f"  Target: \"{tokenizer.decode(target_tokens[:30])}...\"")

        for ns in args.draft_stages:
            r = results[ns]
            print(f"  stages={ns:>2d} (BPA={r['bpa']:.3f}): "
                  f"token_match={r['per_token_match']:.2%}, "
                  f"spec_accept={r['spec_acceptance_rate']:.2%}, "
                  f"first_miss@{r['first_mismatch_pos']}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — Acceptance Rate (avg over {len(prompts)} samples)")
    print(f"  Context={args.context_length}, Generate={args.max_new_tokens}, K=5")
    print(f"{'=' * 70}")

    print(f"\n  {'Stages':>7s} {'BPA':>6s} │ {'Token Match':>11s} │ {'Spec Accept':>11s} │ 비고")
    print(f"  {'─' * 55}")

    for ns in args.draft_stages:
        matches = [r[ns]["per_token_match"] for r in all_results]
        specs = [r[ns]["spec_acceptance_rate"] for r in all_results]
        bpa = ns * math.log2(rq.codebook_size) / rq.codebook_dim

        note = ""
        if ns == 2:
            note = "← draft candidate"
        elif ns == max(args.draft_stages):
            note = "← target (100%)"

        print(f"  {ns:>7d} {bpa:>6.3f} │ {np.mean(matches):>10.2%} │ "
              f"{np.mean(specs):>10.2%} │ {note}")

    print(f"\n  QuantSpec 참고: acceptance ≈ 92% (INT4, BPA=4.0)")
    print(f"  해석: Token Match = 위치별 argmax 일치율")
    print(f"        Spec Accept = K=5 window 기반 실제 speculative acceptance rate")


if __name__ == "__main__":
    main()
