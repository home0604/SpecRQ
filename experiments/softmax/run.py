"""
Safe Softmax Verification with Tight Bound Max Exponent

m_bound = m_draft + ‖q‖₁ · 7.5·max(scale) / √d 를 safe softmax의
max exponent로 사용했을 때, NaN/Inf 없이 안전하게 동작하는지 검증.

검증 흐름:
  1. 모델 forward → 각 layer의 Q, K, V 캡처
  2. K를 INT8 quantize → upper 4-bit만 dequant → K_draft
  3. target scores = Q @ K^T / √d  (causal mask 적용)
  4. draft scores  = Q @ K_draft^T / √d  (causal mask 적용)
  5. 3가지 softmax + attention output 계산:
     A. baseline:    softmax(target_scores − m_target) @ V
     B. with_bound:  softmax(target_scores − m_bound)  @ V   ← 핵심 검증 대상
     C. draft_attn:  softmax(draft_scores  − m_draft)  @ V
  6. 각 단계에서 NaN/Inf 발생 여부 체크

사용법:
  python exp_safe_softmax.py --model togethercomputer/Llama-2-7B-32K-Instruct \\
      --context-length 512 --device cuda:0
"""

import argparse
import json
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_dataset_prompts
from kv_cache import HierarchicalKVCache


@contextmanager
def hook_sdpa():
    """SDPA를 hooking하여 각 attention layer의 (Q, K, V)를 캡처."""
    captured = []
    orig = torch.nn.functional.scaled_dot_product_attention

    def hooked(query, key, value, *args, **kwargs):
        captured.append((
            query.detach().clone(),
            key.detach().clone(),
            value.detach().clone(),
        ))
        return orig(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = hooked
    try:
        yield captured
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


def safe_softmax(scores: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """
    scores:  (B, H, S_q, S_k)
    max_val: (B, H, S_q, 1)
    """
    shifted = scores - max_val
    exp_s = torch.exp(shifted)
    denom = exp_s.sum(dim=-1, keepdim=True)
    return exp_s / denom


def analyze_layer(Q, K, V, group_size=128, chunk=64):
    """하나의 attention layer에서 NaN/Inf 발생 여부 + 정확도 검증."""
    B, H, S, D = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(D)

    Q_f, K_f, V_f = Q.float(), K.float(), V.float()

    # Draft KV
    hc = HierarchicalKVCache(group_size=group_size)
    upper, lower, scale, zp = hc._quantize_to_int4_pair(K)
    K_draft = hc._dequantize_upper_only(upper, scale, zp).half().reshape(K.shape)
    K_draft_f = K_draft.float()

    # δk = 7.5·scale (lower=1111: 7, INT8 rounding: 0.5)
    delta_k_per_key = (7.5 * scale.squeeze(-1)).float()
    max_delta_k = delta_k_per_key.amax(dim=-1, keepdim=True)  # (B, H, 1)

    # ── NaN/Inf counters ──
    checks = {
        "baseline_w": 0, "baseline_out": 0,
        "bound_w": 0,    "bound_out": 0,
        "draft_w": 0,    "draft_out": 0,
    }

    out_baseline_parts = []
    out_bound_parts = []
    out_draft_parts = []

    col_idx = torch.arange(S, device=Q.device)

    for i in range(0, S, chunk):
        cs = min(chunk, S - i)
        q_c = Q_f[:, :, i:i + cs, :]

        s_target = torch.matmul(q_c, K_f.transpose(-2, -1)) * inv_sqrt_d
        s_draft = torch.matmul(q_c, K_draft_f.transpose(-2, -1)) * inv_sqrt_d

        # Causal mask
        row_idx = torch.arange(i, i + cs, device=Q.device).unsqueeze(1)
        causal = col_idx.unsqueeze(0) > row_idx
        mask = causal.unsqueeze(0).unsqueeze(0)
        s_target = s_target.masked_fill(mask, float('-inf'))
        s_draft = s_draft.masked_fill(mask, float('-inf'))

        m_target = s_target.amax(dim=-1, keepdim=True)
        m_draft = s_draft.amax(dim=-1, keepdim=True)

        q_l1 = q_c.abs().sum(dim=-1, keepdim=True)
        m_bound = m_draft + q_l1 * max_delta_k.unsqueeze(-1) * inv_sqrt_d

        # A. Baseline
        w_base = safe_softmax(s_target, m_target)
        o_base = torch.matmul(w_base, V_f)

        # B. Bound
        w_bound = safe_softmax(s_target, m_bound)
        o_bound = torch.matmul(w_bound, V_f)

        # C. Draft
        w_draft = safe_softmax(s_draft, m_draft)
        o_draft = torch.matmul(w_draft, V_f)

        # ── NaN/Inf 체크: weight + output 모두 ──
        checks["baseline_w"] += (w_base.isnan() | w_base.isinf()).sum().item()
        checks["baseline_out"] += (o_base.isnan() | o_base.isinf()).sum().item()
        checks["bound_w"] += (w_bound.isnan() | w_bound.isinf()).sum().item()
        checks["bound_out"] += (o_bound.isnan() | o_bound.isinf()).sum().item()
        checks["draft_w"] += (w_draft.isnan() | w_draft.isinf()).sum().item()
        checks["draft_out"] += (o_draft.isnan() | o_draft.isinf()).sum().item()

        out_baseline_parts.append(o_base)
        out_bound_parts.append(o_bound)
        out_draft_parts.append(o_draft)

        del s_target, s_draft, w_base, w_bound, w_draft

    # ── Accuracy (NaN 없을 때만 의미있음) ──
    out_base = torch.cat(out_baseline_parts, dim=2)
    out_bound = torch.cat(out_bound_parts, dim=2)
    out_draft = torch.cat(out_draft_parts, dim=2)

    flat_base = out_base.reshape(-1, D)
    cos_bound = F.cosine_similarity(flat_base, out_bound.reshape(-1, D), dim=-1)
    cos_draft = F.cosine_similarity(flat_base, out_draft.reshape(-1, D), dim=-1)

    return {
        **checks,
        # Bound vs Baseline
        "cos_bound_mean": cos_bound.mean().item(),
        "cos_bound_min":  cos_bound.min().item(),
        "mse_bound":      ((out_base - out_bound) ** 2).mean().item(),
        # Draft vs Baseline
        "cos_draft_mean": cos_draft.mean().item(),
        "cos_draft_min":  cos_draft.min().item(),
        "mse_draft":      ((out_base - out_draft) ** 2).mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Safe Softmax Verification with Tight Bound")
    parser.add_argument("--model", type=str,
                        default="togethercomputer/Llama-2-7B-32K-Instruct")
    parser.add_argument("--dataset", type=str, default="pg19",
                        choices=["pg19", "wikitext", "c4", "default"])
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="safe_softmax_results.json")
    args = parser.parse_args()

    print("=" * 90)
    print("Safe Softmax Verification")
    print(f"  Model:  {args.model}")
    print(f"  Data:   {args.dataset}, context={args.context_length}, samples={args.num_samples}")
    print(f"  Bound:  m_bound = m_draft + ‖q‖₁ · 7.5·max(scale) / √d")
    print("=" * 90)

    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (FP16, sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device,
        use_safetensors=True, attn_implementation="sdpa", **load_kw,
    ).eval()

    prompts = load_dataset_prompts(
        args.dataset, tokenizer, args.context_length,
        num_samples=args.num_samples, max_new_tokens=10,
    )
    if not prompts:
        prompts = [tokenizer.encode(
            "The future of artificial intelligence is", return_tensors="pt")]

    all_results = []

    for s_idx, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids[:, :args.context_length].to(args.device)
        seq_len = input_ids.shape[1]

        print(f"\n{'=' * 90}")
        print(f"Sample {s_idx + 1}/{len(prompts)} (seq_len={seq_len})")
        print(f"{'=' * 90}")

        with hook_sdpa() as captured:
            with torch.no_grad():
                model(input_ids)

        num_layers = len(captured)
        total_elements = seq_len * 32  # S × H (per layer)

        # ── NaN/Inf Check Table ──
        print(f"\n[NaN/Inf Check]  (0 = PASS)")
        hdr = (f"{'Layer':>5} │ "
               f"{'baseline_w':>10} {'baseline_o':>10} │ "
               f"{'bound_w':>10} {'bound_o':>10} │ "
               f"{'draft_w':>10} {'draft_o':>10} │ {'result':>6}")
        print(hdr)
        print("─" * len(hdr))

        sample_results = []
        sample_all_pass = True

        for l_idx in range(num_layers):
            Q, K, V = captured[l_idx]
            stats = analyze_layer(Q, K, V, args.group_size, args.chunk_size)
            sample_results.append(stats)

            bound_ok = (stats['bound_w'] == 0 and stats['bound_out'] == 0)
            all_ok = bound_ok and (stats['baseline_w'] == 0 and stats['baseline_out'] == 0
                                   and stats['draft_w'] == 0 and stats['draft_out'] == 0)
            if not bound_ok:
                sample_all_pass = False

            tag = "PASS" if all_ok else ("FAIL" if not bound_ok else "warn")

            print(f"{l_idx:>5} │ "
                  f"{stats['baseline_w']:>10} {stats['baseline_out']:>10} │ "
                  f"{stats['bound_w']:>10} {stats['bound_out']:>10} │ "
                  f"{stats['draft_w']:>10} {stats['draft_out']:>10} │ "
                  f"{tag:>6}")

        # ── Accuracy Table (bound vs baseline) ──
        print(f"\n[Accuracy: bound softmax vs baseline]")
        hdr2 = (f"{'Layer':>5} │ {'cos_sim':>12} │ {'cos_min':>12} │ {'MSE':>12}")
        print(hdr2)
        print("─" * len(hdr2))
        for l_idx, s in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{s['cos_bound_mean']:>12.8f} │ "
                  f"{s['cos_bound_min']:>12.8f} │ "
                  f"{s['mse_bound']:>12.2e}")

        # ── Accuracy Table (draft KV vs baseline) ──
        print(f"\n[Accuracy: draft KV attention vs baseline]")
        print(hdr2)
        print("─" * len(hdr2))
        for l_idx, s in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{s['cos_draft_mean']:>12.8f} │ "
                  f"{s['cos_draft_min']:>12.8f} │ "
                  f"{s['mse_draft']:>12.2e}")

        # ── Sample verdict ──
        n = num_layers
        avg = lambda key: sum(s[key] for s in sample_results) / n
        mn = lambda key: min(s[key] for s in sample_results)

        total_bound_nan = sum(s['bound_w'] + s['bound_out'] for s in sample_results)

        print(f"\n  ── Sample {s_idx+1} Result ──")
        if sample_all_pass:
            print(f"  PASS: 전 layer NaN/Inf 없음 (bound softmax weight + output 모두 정상)")
        else:
            print(f"  FAIL: NaN/Inf 발생! (총 {total_bound_nan}개)")
        print(f"  Bound cos_sim:  mean={avg('cos_bound_mean'):.8f}  min={mn('cos_bound_min'):.8f}")
        print(f"  Draft cos_sim:  mean={avg('cos_draft_mean'):.8f}  min={mn('cos_draft_min'):.8f}")

        all_results.append(sample_results)
        del captured
        torch.cuda.empty_cache()

    # ── Final Verdict ──
    flat = [s for sample in all_results for s in sample]
    total_bound_nan = sum(s['bound_w'] + s['bound_out'] for s in flat)
    total_draft_nan = sum(s['draft_w'] + s['draft_out'] for s in flat)
    avg_all = lambda key: sum(s[key] for s in flat) / len(flat)
    mn_all = lambda key: min(s[key] for s in flat)

    print(f"\n{'=' * 90}")
    print("FINAL VERDICT")
    print(f"{'=' * 90}")
    print(f"  Samples: {len(all_results)}, Layers: {len(all_results[0])}")
    print()

    if total_bound_nan == 0:
        print(f"  ✓ SAFE: m_bound 사용 시 NaN/Inf 0건")
        print(f"          tight bound를 safe softmax의 max로 사용해도 안전함")
    else:
        print(f"  ✗ UNSAFE: m_bound 사용 시 NaN/Inf {total_bound_nan}건 발생")

    print()
    print(f"  Bound softmax accuracy (vs ground truth):")
    print(f"    cos_sim mean: {avg_all('cos_bound_mean'):.8f}")
    print(f"    cos_sim min:  {mn_all('cos_bound_min'):.8f}")
    print(f"    MSE:          {avg_all('mse_bound'):.2e}")

    print(f"  Draft KV accuracy (vs ground truth):")
    print(f"    cos_sim mean: {avg_all('cos_draft_mean'):.8f}")
    print(f"    cos_sim min:  {mn_all('cos_draft_min'):.8f}")
    print(f"    MSE:          {avg_all('mse_draft'):.2e}")

    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
