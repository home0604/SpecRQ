"""
Max Exponent Approximation Experiment for Hierarchical Quantized KV Cache

Softmax는 per-row로 적용되므로, 각 query position(row)별 max exponent를 비교:
  1. Target max exponent: max_j(Q_i · K_j / √d) — FP16 KV (ground truth)
  2. Draft max exponent:  max_j(Q_i · K'_j / √d) — INT8 상위 4bit만 dequantize한 KV
  3. Tight upper bound:   draft_max + ‖Q_i‖₁ · δk_max / √d
     여기서 δk_max = 7 · scale (하위 4bit가 1111일 때의 이론적 최대 오차)

이론:
  INT8 = upper*16 + lower,  draft 근사 = upper*16 + 8 (중간값)
  lower=0000(=0)일 때: |0 - 8| = 8 → 이론적 δk = 8 · scale (+ rounding 0.5)
  per-row error bound: |m_target_i − m_draft_i| ≤ ‖q_i‖₁ · max_j(8.5·scale_j) / √d

사용법:
  python exp_max_exponent.py --model togethercomputer/Llama-2-7B-32K-Instruct \\
      --context-length 512 --device cuda:0 --dataset pg19
"""

import argparse
import json
import math
import os
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_dataset_prompts
from kv_cache import HierarchicalKVCache


# ── SDPA Hook ──────────────────────────────────────────────────────────────────

@contextmanager
def hook_sdpa():
    """SDPA를 hooking하여 각 attention layer의 (Q, K)를 캡처."""
    captured = []
    orig = torch.nn.functional.scaled_dot_product_attention

    def hooked(query, key, value, *args, **kwargs):
        captured.append((query.detach().clone(), key.detach().clone()))
        return orig(query, key, value, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = hooked
    try:
        yield captured
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig


# ── Model Parameter Extraction ────────────────────────────────────────────────

def _get_decoder_layers(model):
    """모델 아키텍처에 따라 decoder layer 리스트를 반환."""
    # Llama, Mistral, Qwen2, Gemma, Phi-3, etc.
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    # OPT
    if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        return model.model.decoder.layers
    # GPT-2, GPT-Neo, GPT-J
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    # Falcon
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise AttributeError(
        f"Cannot find decoder layers for {type(model).__name__}. "
        f"Top-level attrs: {[a for a in dir(model) if not a.startswith('_')]}")


def _get_q_proj_weight(layer):
    """Attention layer에서 Q projection weight를 추출."""
    attn = getattr(layer, 'self_attn', None) or getattr(layer, 'attn', None)
    if attn is None:
        raise AttributeError(f"No self_attn/attn in {type(layer).__name__}")
    # q_proj (Llama, OPT, Mistral, etc.)
    if hasattr(attn, 'q_proj'):
        return attn.q_proj.weight
    # GPT-2 style: c_attn (combined QKV)
    if hasattr(attn, 'c_attn'):
        w = attn.c_attn.weight  # (d_model, 3*d_model) or (3*d_model, d_model)
        d = w.shape[-1] // 3 if w.shape[-1] % 3 == 0 else w.shape[0] // 3
        if w.shape[-1] % 3 == 0:
            return w[:, :d]
        return w[:d, :]
    raise AttributeError(f"Cannot find Q projection in {type(attn).__name__}")


def _get_layernorm_weight(layer):
    """Layer에서 input layernorm weight를 추출."""
    for attr in ['input_layernorm', 'self_attn_layer_norm', 'ln_1', 'ln_1']:
        ln = getattr(layer, attr, None)
        if ln is not None and hasattr(ln, 'weight'):
            return ln.weight
    raise AttributeError(
        f"Cannot find input layernorm in {type(layer).__name__}")


def extract_model_params(model):
    """각 layer의 W_Q spectral norm(최대 특잇값)과 LayerNorm/RMSNorm γ max 추출.

    - σ_max(W_Q): Q projection의 spectral norm (최대 특잇값).
      δh를 δQ로 변환할 때의 최대 증폭 계수.
    - max(|γ|): input LayerNorm/RMSNorm의 학습 파라미터 최대값.
      정규화 출력의 L2 norm 상한: ‖h_norm‖₂ ≤ √d_model · max(|γ|)
    """
    params = []
    layers = _get_decoder_layers(model)
    print(f"\nExtracting model params ({len(layers)} layers): σ_max(W_Q), max|γ|...")
    for l_idx, layer in enumerate(layers):
        wq = _get_q_proj_weight(layer).float()
        sigma_max = torch.linalg.svdvals(wq)[0].item()

        gamma = _get_layernorm_weight(layer).float()
        gamma_max = gamma.abs().max().item()

        params.append({
            "spectral_norm_wq": sigma_max,
            "gamma_max": gamma_max,
        })
        if (l_idx + 1) % 8 == 0 or l_idx == len(layers) - 1:
            print(f"  Layer {l_idx + 1}/{len(layers)}: "
                  f"σ_max(W_Q)={sigma_max:.4f}, max|γ|={gamma_max:.4f}")
    return params


# ── Per-layer Analysis ─────────────────────────────────────────────────────────

def analyze_layer(Q: torch.Tensor, K: torch.Tensor,
                  group_size: int = 128, chunk: int = 64,
                  layer_params: dict | None = None):
    """
    하나의 attention layer에 대해 per-row max exponent 분석.

    Q, K: (B, H, S, D) — SDPA에서 캡처된 텐서 (RoPE 적용 후).
    softmax는 per row 적용이므로, 모든 통계는 row=(b,h,i) 단위의 분포로 산출.
    """
    B, H, S, D = Q.shape
    G = group_size
    inv_sqrt_d = 1.0 / math.sqrt(D)

    Q_f = Q.float()
    K_f = K.float()

    # ── Quantize K → upper 4-bit dequant ──
    hc = HierarchicalKVCache(group_size=G)
    upper, lower, scale, zp = hc._quantize_to_int4_pair(K)
    K_draft_f = hc._dequantize_upper_only(upper, scale, zp).reshape(K.shape).float()

    # ── Theoretical δk: lower=0000(0)일 때 |0-8|=8, INT8 rounding ≤0.5 → δk = 8.5·scale ──
    # scale shape: (B, H, S, D//G).  D=G=128 → (B, H, S, 1)
    # per key position j: ‖δk_j‖_∞ = 8.5 · scale_j  (그룹이 1개이므로)
    delta_k_per_key = (8.5 * scale.squeeze(-1)).float()  # (B, H, S)
    # max over all key positions → per (b, h)
    max_delta_k = delta_k_per_key.amax(dim=-1, keepdim=True)  # (B, H, 1)

    # 실제 δk (비교용)
    actual_k_err = (K_f - K_draft_f).abs()
    actual_delta_k_inf = actual_k_err.amax(dim=-1)           # (B, H, S)
    actual_max_delta_k = actual_delta_k_inf.amax(dim=-1, keepdim=True)  # (B, H, 1)

    # ── Chunked per-row max exponent 계산 + score 분포 수집 ──
    target_maxes = []
    draft_maxes = []
    # score element 분포용 (메모리 절약: running stats)
    score_sum = 0.0
    score_sq_sum = 0.0
    score_min = float('inf')
    score_max = float('-inf')
    score_count = 0
    # reservoir sampling for histogram (최대 100k개)
    reservoir_size = 100_000
    score_reservoir = []

    for i in range(0, S, chunk):
        q_c = Q_f[:, :, i:i + chunk, :]
        s_t = torch.matmul(q_c, K_f.transpose(-2, -1)) * inv_sqrt_d
        s_d = torch.matmul(q_c, K_draft_f.transpose(-2, -1)) * inv_sqrt_d
        target_maxes.append(s_t.amax(dim=-1))
        draft_maxes.append(s_d.amax(dim=-1))

        # causal mask 적용 전 score element 통계 (upper triangle 포함)
        s_flat = s_t.reshape(-1)
        n = s_flat.numel()
        score_sum += s_flat.sum().item()
        score_sq_sum += (s_flat ** 2).sum().item()
        score_min = min(score_min, s_flat.min().item())
        score_max = max(score_max, s_flat.max().item())
        score_count += n

        # reservoir sampling
        if len(score_reservoir) < reservoir_size:
            take = min(reservoir_size - len(score_reservoir), n)
            idx = torch.randperm(n, device=s_flat.device)[:take]
            score_reservoir.append(s_flat[idx].cpu())
        else:
            # 확률적으로 교체
            take = min(reservoir_size // 10, n)
            idx = torch.randperm(n, device=s_flat.device)[:take]
            samples = s_flat[idx].cpu()
            for val in samples:
                j = torch.randint(0, score_count, (1,)).item()
                if j < reservoir_size:
                    score_reservoir_cat = torch.cat(score_reservoir) if isinstance(score_reservoir, list) else score_reservoir
                    score_reservoir_cat[j] = val
                    score_reservoir = [score_reservoir_cat]

        del s_t, s_d

    # score reservoir 정리
    if isinstance(score_reservoir, list) and score_reservoir:
        score_reservoir = torch.cat(score_reservoir)
    score_pcts = {}
    for p in [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]:
        score_pcts[f"p{int(p*100):02d}"] = torch.quantile(
            score_reservoir.float(), p).item()
    score_mean = score_sum / score_count
    score_std = math.sqrt(score_sq_sum / score_count - score_mean ** 2)

    target_max = torch.cat(target_maxes, dim=2)  # (B, H, S) — per-row max exp
    draft_max = torch.cat(draft_maxes, dim=2)

    # ── Per-row Q L1 norm ──
    q_l1 = Q_f.abs().sum(dim=-1)  # (B, H, S)

    # ── Per-row error ──
    error = (target_max - draft_max).abs()  # (B, H, S)

    # ── Tight bound (이론적, lower=1111): per-row ──
    # bound_err_i = ‖q_i‖₁ · max_j(8.5·scale_j) / √d
    tight_bound_err = q_l1 * max_delta_k * inv_sqrt_d      # (B, H, S)
    tight_bound = draft_max + tight_bound_err

    # ── δQ error bound (weight quantization → query error) ──
    # Q_t·K_t - Q_d·K_d = δQ·K_t + Q_d·δK
    # |Δm| ≤ ‖δQ‖₂·max_j‖K_j‖₂/√d  +  ‖Q‖₁·‖δK‖_∞/√d
    #         ^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^
    #         δQ term (new)               δK term (existing tight_bound_err)
    # ‖δQ‖₂ ≤ σ_max(W_Q) · ‖δh_norm‖₂ ≤ σ_max(W_Q) · 2√d_model · max(|γ|)
    if layer_params is not None:
        sigma_wq = layer_params["spectral_norm_wq"]
        gamma_max = layer_params["gamma_max"]
        d_model = D * H  # hidden_size = head_dim × num_heads

        # max ‖K_j‖₂ per (b, h): 모든 key position 중 최대
        max_k_l2 = K_f.norm(dim=-1).amax(dim=-1, keepdim=True)  # (B, H, 1)

        # ‖δQ‖₂ 상한 (worst-case: δh_norm이 최대 2배)
        delta_q_l2_bound = sigma_wq * 2 * math.sqrt(d_model) * gamma_max

        # δQ contribution: per (b, h) — broadcast to (B, H, S)
        q_error_term = delta_q_l2_bound * max_k_l2 * inv_sqrt_d  # (B, H, 1)

        # Combined bound: δK term + δQ term
        combined_bound_err = tight_bound_err + q_error_term       # (B, H, S)
        combined_bound = draft_max + combined_bound_err
    else:
        sigma_wq = gamma_max = delta_q_l2_bound = None
        q_error_term = torch.zeros_like(tight_bound_err)
        combined_bound_err = tight_bound_err
        combined_bound = tight_bound

    # ── Naive bound: per-row ──
    naive_bound = draft_max + q_l1

    # ── Per-row 분포 통계 ──
    err_flat = error.reshape(-1).float()
    tb_err_flat = tight_bound_err.reshape(-1).float()
    q_l1_flat = q_l1.reshape(-1).float()

    pcts = [0.50, 0.90, 0.95, 0.99]
    target_flat = target_max.reshape(-1).float()
    draft_flat = draft_max.reshape(-1).float()
    target_pcts = {f"p{int(p*100)}": torch.quantile(target_flat, p).item() for p in pcts}
    draft_pcts = {f"p{int(p*100)}": torch.quantile(draft_flat, p).item() for p in pcts}
    err_pcts = {f"p{int(p*100)}": torch.quantile(err_flat, p).item() for p in pcts}
    tb_err_pcts = {f"p{int(p*100)}": torch.quantile(tb_err_flat, p).item() for p in pcts}

    return {
        # Per-row max exponent 분포
        "target_max_mean": target_max.mean().item(),
        "target_max_p50":  target_pcts["p50"],
        "target_max_p90":  target_pcts["p90"],
        "target_max_p99":  target_pcts["p99"],
        "target_max_max":  target_max.amax().item(),

        "draft_max_mean": draft_max.mean().item(),
        "draft_max_p50":  draft_pcts["p50"],
        "draft_max_p90":  draft_pcts["p90"],
        "draft_max_p99":  draft_pcts["p99"],
        "draft_max_max":  draft_max.amax().item(),

        # Per-row error 분포
        "error_mean": error.mean().item(),
        "error_p50":  err_pcts["p50"],
        "error_p90":  err_pcts["p90"],
        "error_p95":  err_pcts["p95"],
        "error_p99":  err_pcts["p99"],
        "error_max":  error.amax().item(),

        # Per-row tight bound error 분포
        "tight_err_mean": tight_bound_err.mean().item(),
        "tight_err_p50":  tb_err_pcts["p50"],
        "tight_err_p90":  tb_err_pcts["p90"],
        "tight_err_p95":  tb_err_pcts["p95"],
        "tight_err_p99":  tb_err_pcts["p99"],
        "tight_err_max":  tight_bound_err.amax().item(),

        # Bound hold 비율 (per-row)
        "tight_bound_holds_pct": (target_max <= tight_bound + 1e-6).float().mean().item() * 100,
        "combined_bound_holds_pct": (target_max <= combined_bound + 1e-6).float().mean().item() * 100,
        "naive_bound_holds_pct": (target_max <= naive_bound + 1e-6).float().mean().item() * 100,

        # Per-row bound tightness: tight_bound_err / actual_error
        "overestimate_ratio_mean": (tight_bound_err / error.clamp(min=1e-8)).mean().item(),
        "overestimate_ratio_p50":  torch.quantile(
            (tight_bound_err / error.clamp(min=1e-8)).reshape(-1).float(), 0.50).item(),

        # δk 정보
        "delta_k_theory_max": max_delta_k.amax().item(),
        "delta_k_actual_max": actual_max_delta_k.amax().item(),

        # δQ error bound (weight quantization → query error)
        "spectral_norm_wq": sigma_wq,
        "gamma_max": gamma_max,
        "delta_q_l2_bound": delta_q_l2_bound,
        "q_error_term_mean": q_error_term.mean().item() if isinstance(q_error_term, torch.Tensor) else 0,
        "q_error_term_max": q_error_term.amax().item() if isinstance(q_error_term, torch.Tensor) else 0,
        "combined_err_mean": combined_bound_err.mean().item(),
        "combined_err_max": combined_bound_err.amax().item(),

        # Q L1 norm 분포
        "q_l1_mean": q_l1.mean().item(),
        "q_l1_p50":  torch.quantile(q_l1_flat, 0.50).item(),
        "q_l1_p99":  torch.quantile(q_l1_flat, 0.99).item(),

        # Softmax 입력 x = Q·K^T/√d 의 element 분포
        "score_mean": score_mean,
        "score_std":  score_std,
        "score_min":  score_min,
        "score_max":  score_max,
        "score_p01":  score_pcts["p01"],
        "score_p05":  score_pcts["p05"],
        "score_p10":  score_pcts["p10"],
        "score_p50":  score_pcts["p50"],
        "score_p90":  score_pcts["p90"],
        "score_p95":  score_pcts["p95"],
        "score_p99":  score_pcts["p99"],
        "score_count": score_count,
    }


# ── Decoding-phase Analysis ────────────────────────────────────────────────────

def analyze_decoding_step(Q: torch.Tensor, K: torch.Tensor,
                          group_size: int = 128):
    """
    Decoding 중 한 step의 분석. Q: (B, H, 1, D), K: (B, H, S_kv, D).
    Single query가 전체 KV context에 attend → extreme score 발생 가능.
    """
    B, H, S_q, D = Q.shape
    S_kv = K.shape[2]
    inv_sqrt_d = 1.0 / math.sqrt(D)

    Q_f = Q.float()
    K_f = K.float()

    # Quantize K
    hc = HierarchicalKVCache(group_size=group_size)
    upper, lower, scale, zp = hc._quantize_to_int4_pair(K)
    K_draft_f = hc._dequantize_upper_only(upper, scale, zp).reshape(K.shape).float()

    # Scores: (B, H, 1, S_kv)
    s_target = torch.matmul(Q_f, K_f.transpose(-2, -1)) * inv_sqrt_d
    s_draft = torch.matmul(Q_f, K_draft_f.transpose(-2, -1)) * inv_sqrt_d

    # Per-row max (B, H, 1)
    target_max = s_target.amax(dim=-1)
    draft_max = s_draft.amax(dim=-1)
    error = (target_max - draft_max).abs()

    # δk
    delta_k_per_key = (8.5 * scale.squeeze(-1)).float()
    max_delta_k = delta_k_per_key.amax(dim=-1, keepdim=True)
    q_l1 = Q_f.abs().sum(dim=-1)
    tight_bound_err = q_l1 * max_delta_k * inv_sqrt_d

    # Score element 분포
    s_flat = s_target.reshape(-1)
    s_draft_flat = s_draft.reshape(-1)

    return {
        "target_max_mean": target_max.mean().item(),
        "target_max_max":  target_max.amax().item(),
        "draft_max_mean":  draft_max.mean().item(),
        "draft_max_max":   draft_max.amax().item(),
        "error_mean":      error.mean().item(),
        "error_max":       error.amax().item(),
        "tight_err_mean":  tight_bound_err.mean().item(),
        "tight_bound_holds_pct": (target_max <= draft_max + tight_bound_err + 1e-6).float().mean().item() * 100,
        "score_mean":      s_flat.mean().item(),
        "score_std":       s_flat.std().item(),
        "score_min":       s_flat.min().item(),
        "score_max":       s_flat.max().item(),
        "score_p01":       torch.quantile(s_flat.float(), 0.01).item(),
        "score_p50":       torch.quantile(s_flat.float(), 0.50).item(),
        "score_p99":       torch.quantile(s_flat.float(), 0.99).item(),
        "kv_len":          S_kv,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Max Exponent Approximation: target vs draft (upper-4bit KV)")
    parser.add_argument("--model", type=str,
                        default="data/models/llama2_7b")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["pg19", "wikitext", "c4", "gsm8k", "default"])
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="Q chunking size for OOM 방지")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cache-dir", type=str, default="/data/models")
    parser.add_argument("--output", type=str, default="max_exp_results.json")
    parser.add_argument("--plot", type=str, default="max_exponent_plots.png",
                        help="Plot 저장 경로 (--no-plot으로 비활성화)")
    parser.add_argument("--no-plot", action="store_true", help="Plot 생성 비활성화")
    parser.add_argument("--decode-tokens", type=int, default=0,
                        help="Prefill 후 autoregressive decoding할 토큰 수 (0=prefill만)")
    args = parser.parse_args()

    print("=" * 100)
    print("Max Exponent Approximation Experiment (per-row analysis)")
    print(f"  Model:          {args.model}")
    print(f"  Dataset:        {args.dataset}")
    print(f"  Context length: {args.context_length}")
    print(f"  Group size:     {args.group_size}")
    print(f"  Num samples:    {args.num_samples}")
    print(f"  Device:         {args.device}")
    print(f"  Decode tokens:  {args.decode_tokens}")
    print(f"  Tight bound:    δk = 8.5·scale (lower=1111 이론치)")
    print("=" * 100)

    # ── Load model (FP16, single model) ──
    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model (FP16, sdpa)...")
    # 로컬에 .safetensors 샤드 없고 .bin만 있으면 use_safetensors=False 로 PyTorch 가중치 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=args.device,
        attn_implementation="sdpa",
        use_safetensors=False,
        **load_kw,
    ).eval()

    # ── Load dataset ──
    prompts = load_dataset_prompts(
        args.dataset, tokenizer, args.context_length,
        num_samples=args.num_samples, max_new_tokens=10,
    )
    if not prompts:
        print("No valid prompts. Using default short prompt.")
        prompts = [tokenizer.encode(
            "The future of artificial intelligence is", return_tensors="pt")]

    # ── Extract model params (spectral norm, RMSNorm γ) ──
    model_params = extract_model_params(model)

    # ── Run experiment ──
    all_results = []

    for s_idx, prompt_ids in enumerate(prompts):
        input_ids = prompt_ids[:, :args.context_length].to(args.device)
        seq_len = input_ids.shape[1]

        print(f"\n{'=' * 100}")
        print(f"Sample {s_idx + 1}/{len(prompts)} (seq_len={seq_len})")
        print(f"  Each row = one softmax row (batch, head, query_pos)")
        print(f"  Total rows per layer: {seq_len} positions × 32 heads = {seq_len * 32}")
        print(f"{'=' * 100}")

        with hook_sdpa() as captured:
            with torch.no_grad():
                model(input_ids)

        num_layers = len(captured)

        sample_results = []
        for l_idx in range(num_layers):
            Q, K = captured[l_idx]
            stats = analyze_layer(Q, K, args.group_size, args.chunk_size,
                                  layer_params=model_params[l_idx])
            sample_results.append(stats)

        # ── Table 0: Per-row max exponent 값 (target vs draft) ──
        print(f"\n[Per-row max exponent]  target = FP16 K,  draft = upper-4bit K")
        hdr0 = (f"{'Layer':>5} │ "
                f"{'t_mean':>8} {'t_p50':>8} {'t_p90':>8} {'t_p99':>8} {'t_max':>8} │ "
                f"{'d_mean':>8} {'d_p50':>8} {'d_p90':>8} {'d_p99':>8} {'d_max':>8}")
        print(hdr0)
        print("─" * len(hdr0))
        for l_idx, stats in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{stats['target_max_mean']:>8.3f} "
                  f"{stats['target_max_p50']:>8.3f} "
                  f"{stats['target_max_p90']:>8.3f} "
                  f"{stats['target_max_p99']:>8.3f} "
                  f"{stats['target_max_max']:>8.3f} │ "
                  f"{stats['draft_max_mean']:>8.3f} "
                  f"{stats['draft_max_p50']:>8.3f} "
                  f"{stats['draft_max_p90']:>8.3f} "
                  f"{stats['draft_max_p99']:>8.3f} "
                  f"{stats['draft_max_max']:>8.3f}")

        # ── Table 0.5: Softmax 입력 x = Q·K^T/√d element 분포 ──
        print(f"\n[Softmax input x distribution]  x = Q·K^T/√d (all elements per layer)")
        hdr05 = (f"{'Layer':>5} │ {'mean':>8} {'std':>8} │ "
                 f"{'p01':>8} {'p05':>8} {'p10':>8} {'p50':>8} "
                 f"{'p90':>8} {'p95':>8} {'p99':>8} │ {'min':>8} {'max':>8}")
        print(hdr05)
        print("─" * len(hdr05))
        for l_idx, stats in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{stats['score_mean']:>8.3f} "
                  f"{stats['score_std']:>8.3f} │ "
                  f"{stats['score_p01']:>8.3f} "
                  f"{stats['score_p05']:>8.3f} "
                  f"{stats['score_p10']:>8.3f} "
                  f"{stats['score_p50']:>8.3f} "
                  f"{stats['score_p90']:>8.3f} "
                  f"{stats['score_p95']:>8.3f} "
                  f"{stats['score_p99']:>8.3f} │ "
                  f"{stats['score_min']:>8.3f} "
                  f"{stats['score_max']:>8.3f}")

        # ── Table 1: Per-row error 분포 ──
        print(f"\n[Per-row |error| distribution]  (|target_max − draft_max| per softmax row)")
        hdr1 = (f"{'Layer':>5} │ {'mean':>8} │ {'p50':>8} │ {'p90':>8} │ "
                f"{'p95':>8} │ {'p99':>8} │ {'max':>8}")
        print(hdr1)
        print("─" * len(hdr1))

        for l_idx, stats in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{stats['error_mean']:>8.4f} │ "
                  f"{stats['error_p50']:>8.4f} │ "
                  f"{stats['error_p90']:>8.4f} │ "
                  f"{stats['error_p95']:>8.4f} │ "
                  f"{stats['error_p99']:>8.4f} │ "
                  f"{stats['error_max']:>8.4f}")

        # ── Table 2: Tight bound (δK only) 분석 ──
        print(f"\n[Tight bound per row]  δK term = ‖q‖₁ · 8.5·max(scale) / √d")
        hdr2 = (f"{'Layer':>5} │ {'δK_term':>9} │ {'actual_err':>10} │ "
                f"{'overest':>7} │ {'holds%':>6} │ "
                f"{'δk_theory':>9} │ {'δk_actual':>9} │ {'‖q‖₁':>8}")
        print(hdr2)
        print("─" * len(hdr2))

        for l_idx, stats in enumerate(sample_results):
            print(f"{l_idx:>5} │ "
                  f"{stats['tight_err_mean']:>9.4f} │ "
                  f"{stats['error_mean']:>10.4f} │ "
                  f"{stats['overestimate_ratio_mean']:>6.1f}x │ "
                  f"{stats['tight_bound_holds_pct']:>5.1f}% │ "
                  f"{stats['delta_k_theory_max']:>9.4f} │ "
                  f"{stats['delta_k_actual_max']:>9.4f} │ "
                  f"{stats['q_l1_mean']:>8.1f}")

        # ── Table 2.5: Combined bound (δK + δQ) ──
        print(f"\n[Combined bound]  |Δm| ≤ ‖δQ‖₂·max‖K‖₂/√d + ‖Q‖₁·‖δK‖_∞/√d")
        print(f"  δQ bound: ‖δQ‖₂ ≤ σ_max(W_Q) · 2√d_model · max|γ|")
        hdr25 = (f"{'Layer':>5} │ {'σ_max':>8} {'max|γ|':>8} │ "
                 f"{'δQ_term':>9} {'δK_term':>9} {'combined':>9} │ "
                 f"{'actual':>9} │ {'comb_holds':>10}")
        print(hdr25)
        print("─" * len(hdr25))
        for l_idx, stats in enumerate(sample_results):
            sigma = stats.get('spectral_norm_wq')
            gamma = stats.get('gamma_max')
            print(f"{l_idx:>5} │ "
                  f"{sigma:>8.3f} {gamma:>8.4f} │ "
                  f"{stats['q_error_term_mean']:>9.4f} "
                  f"{stats['tight_err_mean']:>9.4f} "
                  f"{stats['combined_err_mean']:>9.4f} │ "
                  f"{stats['error_mean']:>9.4f} │ "
                  f"{stats['combined_bound_holds_pct']:>9.1f}%")

        # ── Per-sample summary ──
        n = num_layers
        avg = lambda key: sum(s[key] for s in sample_results) / n
        mx = lambda key: max(s[key] for s in sample_results)

        print(f"\n  Summary:")
        print(f"    Per-row |error|:  mean={avg('error_mean'):.4f}  "
              f"p95={avg('error_p95'):.4f}  p99={avg('error_p99'):.4f}  "
              f"max={mx('error_max'):.4f}")
        print(f"    δK bound (tight): mean={avg('tight_err_mean'):.4f}  "
              f"overestimate={avg('overestimate_ratio_mean'):.1f}x  "
              f"holds={avg('tight_bound_holds_pct'):.1f}%")
        print(f"    δQ bound term:    mean={avg('q_error_term_mean'):.4f}  "
              f"max={mx('q_error_term_max'):.4f}")
        print(f"    Combined (δK+δQ): mean={avg('combined_err_mean'):.4f}  "
              f"holds={avg('combined_bound_holds_pct'):.1f}%")
        print(f"    Naive bound:      holds={avg('naive_bound_holds_pct'):.1f}%")
        print(f"    δk (8.5·scale):   theory_max={mx('delta_k_theory_max'):.4f}  "
              f"actual_max={mx('delta_k_actual_max'):.4f}")

        all_results.append(sample_results)
        del captured
        torch.cuda.empty_cache()

        # ── Decoding phase ──
        if args.decode_tokens > 0:
            print(f"\n[Decoding phase: {args.decode_tokens} tokens]")
            decode_results = []
            generated = input_ids.clone()

            for t_idx in range(args.decode_tokens):
                with hook_sdpa() as dec_captured:
                    with torch.no_grad():
                        outputs = model(generated, use_cache=False)

                # 마지막 토큰의 Q, K만 분석 (decoding attention)
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                generated = torch.cat([generated, next_token], dim=1)

                # 각 layer에서 Q의 마지막 row, K 전체
                step_stats = []
                for l_idx in range(len(dec_captured)):
                    Q_dec, K_dec = dec_captured[l_idx]
                    # Q: (B,H,S,D) → 마지막 position만
                    Q_last = Q_dec[:, :, -1:, :]
                    stats = analyze_decoding_step(Q_last, K_dec, args.group_size)
                    step_stats.append(stats)

                decode_results.append(step_stats)
                del dec_captured
                if (t_idx + 1) % 20 == 0:
                    print(f"  ... {t_idx + 1}/{args.decode_tokens} tokens generated")

            # Decode summary
            dec_flat = [s for step in decode_results for s in step]
            dec_score_min = min(s["score_min"] for s in dec_flat)
            dec_score_max = max(s["score_max"] for s in dec_flat)
            dec_score_p01 = sum(s["score_p01"] for s in dec_flat) / len(dec_flat)
            dec_score_p99 = sum(s["score_p99"] for s in dec_flat) / len(dec_flat)
            dec_err_max = max(s["error_max"] for s in dec_flat)
            dec_bound_holds = sum(s["tight_bound_holds_pct"] for s in dec_flat) / len(dec_flat)

            print(f"\n  ── Decoding Score Distribution (sample {s_idx+1}) ──")
            print(f"    Score range: [{dec_score_min:.1f}, {dec_score_max:.1f}]")
            print(f"    Score p01={dec_score_p01:.3f}  p99={dec_score_p99:.3f}")
            print(f"    Max |error|: {dec_err_max:.4f}")
            print(f"    Tight bound holds: {dec_bound_holds:.1f}%")

            # Per-layer decoding score extremes
            print(f"\n  [Decoding per-layer score extremes]")
            hdr_dec = f"{'Layer':>5} │ {'score_min':>10} {'score_max':>10} │ {'err_max':>8} │ {'holds%':>6}"
            print(hdr_dec)
            print("─" * len(hdr_dec))
            for l_idx in range(num_layers):
                layer_stats = [decode_results[t][l_idx] for t in range(len(decode_results))]
                l_smin = min(s["score_min"] for s in layer_stats)
                l_smax = max(s["score_max"] for s in layer_stats)
                l_emax = max(s["error_max"] for s in layer_stats)
                l_holds = sum(s["tight_bound_holds_pct"] for s in layer_stats) / len(layer_stats)
                print(f"{l_idx:>5} │ {l_smin:>10.1f} {l_smax:>10.1f} │ "
                      f"{l_emax:>8.4f} │ {l_holds:>5.1f}%")

            # decoding 결과도 저장
            if len(all_results) > 0:
                all_results[-1] = {
                    "prefill": sample_results,
                    "decoding": decode_results,
                }

            del generated
            torch.cuda.empty_cache()

    # ── Overall summary ──
    # prefill 결과 추출 (decoding이 있으면 dict, 없으면 list)
    prefill_results = []
    decoding_results = []
    for sample in all_results:
        if isinstance(sample, dict):
            prefill_results.append(sample["prefill"])
            decoding_results.append(sample["decoding"])
        else:
            prefill_results.append(sample)

    flat = [s for sample in prefill_results for s in sample]
    n_total = len(flat)
    avg_all = lambda key: sum(s[key] for s in flat) / n_total
    mx_all = lambda key: max(s[key] for s in flat)

    print(f"\n{'=' * 100}")
    print("OVERALL SUMMARY (across all samples × layers)")
    print(f"{'=' * 100}")

    print(f"\n  Per-row max exponent:")
    print(f"    target: mean={avg_all('target_max_mean'):.3f}  "
          f"p50={avg_all('target_max_p50'):.3f}  "
          f"p99={avg_all('target_max_p99'):.3f}  "
          f"max={mx_all('target_max_max'):.3f}")
    print(f"    draft:  mean={avg_all('draft_max_mean'):.3f}  "
          f"p50={avg_all('draft_max_p50'):.3f}  "
          f"p99={avg_all('draft_max_p99'):.3f}  "
          f"max={mx_all('draft_max_max'):.3f}")

    print(f"\n  Per-row |error| (|target_max − draft_max|):")
    print(f"    mean:  {avg_all('error_mean'):.4f}")
    print(f"    p50:   {avg_all('error_p50'):.4f}")
    print(f"    p90:   {avg_all('error_p90'):.4f}")
    print(f"    p95:   {avg_all('error_p95'):.4f}")
    print(f"    p99:   {avg_all('error_p99'):.4f}")
    print(f"    max:   {mx_all('error_max'):.4f}")

    print(f"\n  Tight bound — δK only (‖Q‖₁·‖δK‖_∞/√d):")
    print(f"    bound_err mean:    {avg_all('tight_err_mean'):.4f}")
    print(f"    overestimate:      {avg_all('overestimate_ratio_mean'):.1f}x")
    print(f"    holds:             {avg_all('tight_bound_holds_pct'):.1f}%")

    has_dq = flat[0].get('spectral_norm_wq') is not None
    if has_dq:
        print(f"\n  δQ bound (‖δQ‖₂·max‖K‖₂/√d, from weight quantization):")
        print(f"    σ_max(W_Q):  mean={avg_all('spectral_norm_wq'):.4f}  "
              f"max={mx_all('spectral_norm_wq'):.4f}")
        print(f"    max|γ|:      mean={avg_all('gamma_max'):.4f}  "
              f"max={mx_all('gamma_max'):.4f}")
        print(f"    ‖δQ‖₂ bound: max={mx_all('delta_q_l2_bound'):.2f}")
        print(f"    δQ term:     mean={avg_all('q_error_term_mean'):.4f}  "
              f"max={mx_all('q_error_term_max'):.4f}")

        print(f"\n  Combined bound (δK + δQ):")
        print(f"    combined_err mean: {avg_all('combined_err_mean'):.4f}")
        print(f"    combined_err max:  {mx_all('combined_err_max'):.4f}")
        print(f"    holds:             {avg_all('combined_bound_holds_pct'):.1f}%")

    print(f"\n  Naive bound (‖q‖₁):")
    print(f"    holds:             {avg_all('naive_bound_holds_pct'):.1f}%")

    print(f"\n  δk (quantization error ‖δk‖_∞):")
    print(f"    theory (8.5·scale):  max={mx_all('delta_k_theory_max'):.4f}")
    print(f"    actual:            max={mx_all('delta_k_actual_max'):.4f}")

    print(f"\n  ‖q‖₁:")
    print(f"    mean: {avg_all('q_l1_mean'):.1f}  "
          f"p50: {avg_all('q_l1_p50'):.1f}  "
          f"p99: {avg_all('q_l1_p99'):.1f}")

    mn_all = lambda key: min(s[key] for s in flat)
    print(f"\n  Softmax input x = Q·K^T/√d (element-wise):")
    print(f"    mean={avg_all('score_mean'):.3f}  std={avg_all('score_std'):.3f}")
    print(f"    p01={avg_all('score_p01'):.3f}  p05={avg_all('score_p05'):.3f}  "
          f"p50={avg_all('score_p50'):.3f}  p95={avg_all('score_p95'):.3f}  "
          f"p99={avg_all('score_p99'):.3f}")
    print(f"    min={mn_all('score_min'):.3f}  max={mx_all('score_max'):.3f}")

    if decoding_results:
        dec_all = [s for sample in decoding_results for step in sample for s in step]
        print(f"\n  Decoding phase ({args.decode_tokens} tokens × {len(decoding_results)} samples):")
        print(f"    Score range: [{min(s['score_min'] for s in dec_all):.1f}, "
              f"{max(s['score_max'] for s in dec_all):.1f}]")
        print(f"    Score p01={sum(s['score_p01'] for s in dec_all)/len(dec_all):.3f}  "
              f"p99={sum(s['score_p99'] for s in dec_all)/len(dec_all):.3f}")
        print(f"    Max |error|: {max(s['error_max'] for s in dec_all):.4f}")
        print(f"    Tight bound holds: "
              f"{sum(s['tight_bound_holds_pct'] for s in dec_all)/len(dec_all):.1f}%")

    # ── Save results ──
    with open(args.output, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Plot ──
    if not args.no_plot:
        plot_results(vars(args), prefill_results, args.plot,
                     decoding_results=decoding_results if decoding_results else None)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_results(args: dict, all_results: list, save_path: str,
                 decoding_results=None):
    """실험 결과를 2x4 panel figure로 시각화."""
    num_samples = len(all_results)
    num_layers = len(all_results[0])
    layers = np.arange(num_layers)

    model_name = args.get("model", "unknown").split("/")[-1]
    ctx_len = args.get("context_length", "?")

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        f"Max Exponent Analysis — {model_name} (ctx={ctx_len}, {num_samples} samples)",
        fontsize=14, fontweight="bold")

    # ── 1. Per-layer error distribution (mean + p90/p99 band) ──
    ax = axes[0, 0]
    for s_idx, sample in enumerate(all_results):
        means = [s["error_mean"] for s in sample]
        p90s = [s["error_p90"] for s in sample]
        p99s = [s["error_p99"] for s in sample]
        alpha = 0.6 if num_samples > 1 else 1.0
        label = f"sample {s_idx}" if num_samples > 1 else None
        ax.plot(layers, means, "o-", markersize=3, alpha=alpha, label=label)
        ax.fill_between(layers, means, p90s, alpha=0.1)
        ax.fill_between(layers, p90s, p99s, alpha=0.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("|target_max − draft_max|")
    ax.set_title("Per-row error by layer\n(line=mean, light=p90, lighter=p99)")
    if num_samples > 1:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Tight bound error vs actual error ──
    ax = axes[0, 1]
    avg_error = np.array([np.mean([all_results[s][l]["error_mean"]
                                   for s in range(num_samples)])
                          for l in range(num_layers)])
    avg_bound = np.array([np.mean([all_results[s][l]["tight_err_mean"]
                                   for s in range(num_samples)])
                          for l in range(num_layers)])
    ax.bar(layers - 0.2, avg_bound, width=0.4,
           label="tight bound err", color="C1", alpha=0.7)
    ax.bar(layers + 0.2, avg_error, width=0.4,
           label="actual err", color="C0", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Error")
    ax.set_title("Tight bound vs actual error (mean)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 3. Overestimate ratio per layer ──
    ax = axes[0, 2]
    avg_ratio = [np.mean([all_results[s][l]["overestimate_ratio_mean"]
                          for s in range(num_samples)])
                 for l in range(num_layers)]
    med_ratio = [np.mean([all_results[s][l]["overestimate_ratio_p50"]
                          for s in range(num_samples)])
                 for l in range(num_layers)]
    ax.bar(layers, avg_ratio, alpha=0.5, label="mean ratio", color="C3")
    ax.bar(layers, med_ratio, alpha=0.7, label="median ratio", color="C4")
    ax.set_xlabel("Layer")
    ax.set_ylabel("bound_err / actual_err")
    ax.set_title("Overestimate ratio by layer")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. δk: theory vs actual ──
    ax = axes[1, 0]
    for s_idx, sample in enumerate(all_results):
        dk_theory = [s["delta_k_theory_max"] for s in sample]
        dk_actual = [s["delta_k_actual_max"] for s in sample]
        alpha = 0.6 if num_samples > 1 else 1.0
        ax.plot(layers, dk_theory, "s--", markersize=4, alpha=alpha,
                color=f"C{s_idx}",
                label="theory" if s_idx == 0 else None)
        ax.plot(layers, dk_actual, "o-", markersize=4, alpha=alpha,
                color=f"C{s_idx}",
                label="actual" if s_idx == 0 else None)
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖δk‖_∞ max")
    ax.set_title("δk theory (8.5·scale) vs actual max")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 5. ‖q‖₁ distribution per layer ──
    ax = axes[1, 1]
    avg_q_mean = [np.mean([all_results[s][l]["q_l1_mean"]
                           for s in range(num_samples)])
                  for l in range(num_layers)]
    avg_q_p50 = [np.mean([all_results[s][l]["q_l1_p50"]
                          for s in range(num_samples)])
                 for l in range(num_layers)]
    avg_q_p99 = [np.mean([all_results[s][l]["q_l1_p99"]
                          for s in range(num_samples)])
                 for l in range(num_layers)]
    ax.fill_between(layers, avg_q_p50, avg_q_p99,
                    alpha=0.2, color="C2", label="p50–p99")
    ax.plot(layers, avg_q_mean, "o-", markersize=3, color="C2", label="mean")
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖q‖₁")
    ax.set_title("Query L1 norm by layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 6. Error percentile histogram ──
    ax = axes[1, 2]
    flat = [s for sample in all_results for s in sample]
    percentiles = ["error_p50", "error_p90", "error_p95", "error_p99", "error_max"]
    labels = ["p50", "p90", "p95", "p99", "max"]
    vals = [np.mean([f[p] for f in flat]) for p in percentiles]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.bar(labels, vals, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("|target_max − draft_max|")
    ax.set_title("Error percentiles (all layers avg)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── 7. Score x = Q·K^T/√d: mean ± std per layer ──
    ax = axes[0, 3]
    avg_score_mean = [np.mean([all_results[s][l]["score_mean"]
                               for s in range(num_samples)])
                      for l in range(num_layers)]
    avg_score_std = [np.mean([all_results[s][l]["score_std"]
                              for s in range(num_samples)])
                     for l in range(num_layers)]
    avg_score_mean = np.array(avg_score_mean)
    avg_score_std = np.array(avg_score_std)
    ax.plot(layers, avg_score_mean, "o-", markersize=3, color="C0", label="prefill mean")
    ax.fill_between(layers,
                    avg_score_mean - avg_score_std,
                    avg_score_mean + avg_score_std,
                    alpha=0.2, color="C0", label="prefill ±1std")
    if decoding_results:
        dec_score_mean = []
        dec_score_std = []
        for l in range(num_layers):
            l_means = [step[l]["score_mean"]
                       for sample in decoding_results for step in sample]
            l_stds = [step[l]["score_std"]
                      for sample in decoding_results for step in sample]
            dec_score_mean.append(np.mean(l_means))
            dec_score_std.append(np.mean(l_stds))
        dec_score_mean = np.array(dec_score_mean)
        dec_score_std = np.array(dec_score_std)
        ax.plot(layers, dec_score_mean, "s--", markersize=3, color="C3",
                label="decode mean")
        ax.fill_between(layers,
                        dec_score_mean - dec_score_std,
                        dec_score_mean + dec_score_std,
                        alpha=0.15, color="C3", label="decode ±1std")
    ax.set_xlabel("Layer")
    ax.set_ylabel("x = Q·K^T/√d")
    ax.set_title("Score mean ± std by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 8. Aggregated score density (all layers, Gaussian mixture) ──
    ax = axes[1, 3]
    from scipy.stats import norm as _norm

    # Prefill: 각 레이어의 (mean, std, count)로 Gaussian mixture 구성
    pf_means, pf_stds, pf_counts = [], [], []
    for l in range(num_layers):
        for s in range(num_samples):
            st = all_results[s][l]
            pf_means.append(st["score_mean"])
            pf_stds.append(max(st["score_std"], 1e-8))
            pf_counts.append(st["score_count"])
    pf_means = np.array(pf_means)
    pf_stds = np.array(pf_stds)
    pf_weights = np.array(pf_counts, dtype=float)
    pf_weights /= pf_weights.sum()

    # x 범위: 전체 min/max에서 약간 여유
    global_min = min(all_results[s][l]["score_min"]
                     for s in range(num_samples) for l in range(num_layers))
    global_max = max(all_results[s][l]["score_max"]
                     for s in range(num_samples) for l in range(num_layers))
    margin = (global_max - global_min) * 0.05
    x_grid = np.linspace(global_min - margin, global_max + margin, 1000)

    # 가중 Gaussian mixture density
    pf_density = np.zeros_like(x_grid)
    for mu, sigma, w in zip(pf_means, pf_stds, pf_weights):
        pf_density += w * _norm.pdf(x_grid, mu, sigma)

    ax.fill_between(x_grid, pf_density, alpha=0.3, color="C0", label="prefill")
    ax.plot(x_grid, pf_density, color="C0", linewidth=1.0)

    # Decoding density
    if decoding_results:
        dc_means, dc_stds, dc_counts = [], [], []
        for sample in decoding_results:
            for step in sample:
                for l in range(num_layers):
                    st = step[l]
                    dc_means.append(st["score_mean"])
                    dc_stds.append(max(st["score_std"], 1e-8))
                    # decoding은 score_count 없을 수 있음 → kv_len × heads
                    dc_counts.append(st.get("score_count", st.get("kv_len", 1)))
        dc_means = np.array(dc_means)
        dc_stds = np.array(dc_stds)
        dc_weights = np.array(dc_counts, dtype=float)
        dc_weights /= dc_weights.sum()

        dc_density = np.zeros_like(x_grid)
        for mu, sigma, w in zip(dc_means, dc_stds, dc_weights):
            dc_density += w * _norm.pdf(x_grid, mu, sigma)

        ax.fill_between(x_grid, dc_density, alpha=0.2, color="C3", label="decode")
        ax.plot(x_grid, dc_density, color="C3", linewidth=1.0, linestyle="--")

    # 99.99% 범위 계산 (prefill CDF에서)
    pf_cdf = np.zeros_like(x_grid)
    for mu, sigma, w in zip(pf_means, pf_stds, pf_weights):
        pf_cdf += w * _norm.cdf(x_grid, mu, sigma)

    lo_idx = np.searchsorted(pf_cdf, 0.00005)
    hi_idx = np.searchsorted(pf_cdf, 0.99995)
    lo_val = x_grid[min(lo_idx, len(x_grid) - 1)]
    hi_val = x_grid[min(hi_idx, len(x_grid) - 1)]
    ax.axvline(lo_val, color="red", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(hi_val, color="red", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvspan(lo_val, hi_val, alpha=0.06, color="red")
    ax.text(hi_val, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 0.5,
            f" 99.99%\n [{lo_val:.1f}, {hi_val:.1f}]",
            fontsize=7, color="red", va="center")

    ax.set_xlabel("x = Q·K^T/√d")
    ax.set_ylabel("Probability Density")
    ax.set_title("Score distribution (all layers)\nred = 99.99% range")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")


if __name__ == "__main__":
    main()
