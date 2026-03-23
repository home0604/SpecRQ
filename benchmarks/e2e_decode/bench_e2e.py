"""
End-to-End Decode Latency Breakdown for Llama-2-7B
===================================================
Measures per-token decode latency with component breakdown:
  - RMSNorm
  - Q/K/V/O Linear projections
  - Attention (QK + softmax + PV)
  - MLP (gate_proj, up_proj, down_proj, SiLU)
  - Residual connections

Two profiling modes:
  1. CUDA events: per-component wall-clock time (accurate, low overhead)
  2. torch.profiler: full trace exportable to Chrome (chrome://tracing)

Usage:
  python bench_e2e_decode.py --model meta-llama/Llama-2-7b-hf --context-length 1024
  python bench_e2e_decode.py --model /data/models/llama2_7b --cache-dir /data/models
"""

import argparse
import json
import math
import time
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── CUDA Event Profiling ──────────────────────────────────────────────────────

class DecodeProfiler:
    """CUDA event 기반 per-component latency 측정.

    Hook을 등록하여 각 module의 forward 전후에 CUDA event를 기록.
    """

    def __init__(self, model):
        self.model = model
        self.events = {}    # name → (start_event, end_event) list
        self.hooks = []
        self._setup_hooks()

    def _make_hook(self, name):
        """Module forward 전후에 CUDA event 기록하는 hook 생성."""
        def pre_hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            if name not in self.events:
                self.events[name] = []
            self.events[name].append([start, None])

        def post_hook(module, input, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events[name][-1][1] = end

        return pre_hook, post_hook

    def _classify_module(self, full_name, module):
        """Module 이름으로 component 분류."""
        name = full_name.split(".")[-1]
        cls_name = type(module).__name__

        if "layernorm" in name.lower() or "rmsnorm" in cls_name.lower() or "norm" in name.lower():
            if "input" in name or "self_attn" in full_name.split(".")[-2] if "." in full_name else "":
                return "attn_norm"
            elif "post" in name or "mlp" in full_name.split(".")[-2] if "." in full_name else "":
                return "mlp_norm"
            elif "final" in name or "model.norm" in full_name:
                return "final_norm"
            return "norm"
        elif name == "q_proj":
            return "q_proj"
        elif name == "k_proj":
            return "k_proj"
        elif name == "v_proj":
            return "v_proj"
        elif name == "o_proj":
            return "o_proj"
        elif name == "gate_proj":
            return "gate_proj"
        elif name == "up_proj":
            return "up_proj"
        elif name == "down_proj":
            return "down_proj"
        elif name == "self_attn":
            return "self_attn"  # 전체 attention block (QKV + RoPE + core + O 포함)
        elif "attention" in cls_name.lower() or "sdpa" in cls_name.lower():
            return None  # skip sub-wrappers
        elif "mlp" == name.lower():
            return "mlp"
        elif name == "lm_head":
            return "lm_head"
        elif "embed" in name.lower():
            return "embed"
        elif "rotary" in name.lower() or "rope" in cls_name.lower():
            return "rotary"

        return None

    def _setup_hooks(self):
        """모델의 각 component에 profiling hook 등록."""
        # Layer-level hooks for attention and MLP blocks
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            layers = self.model.model.decoder.layers

        if layers is not None:
            for l_idx, layer in enumerate(layers):
                for name, module in layer.named_modules():
                    if name == "":  # skip the layer itself
                        continue
                    component = self._classify_module(f"layer.{l_idx}.{name}", module)
                    if component:
                        pre, post = self._make_hook(component)
                        self.hooks.append(module.register_forward_pre_hook(pre))
                        self.hooks.append(module.register_forward_hook(post))

        # Top-level modules
        for name, module in self.model.named_modules():
            if name in ("model.embed_tokens", "lm_head", "model.norm"):
                component = name.split(".")[-1]
                if component == "embed_tokens":
                    component = "embed"
                elif component == "norm":
                    component = "final_norm"
                pre, post = self._make_hook(component)
                self.hooks.append(module.register_forward_pre_hook(pre))
                self.hooks.append(module.register_forward_hook(post))

    def reset(self):
        """이전 측정 결과 클리어."""
        self.events.clear()

    def get_results(self):
        """각 component의 총 시간 (ms) 반환."""
        torch.cuda.synchronize()
        results = {}
        for name, event_pairs in self.events.items():
            total_ms = 0.0
            for start, end in event_pairs:
                if end is not None:
                    total_ms += start.elapsed_time(end)
            results[name] = total_ms
        return results

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Grouped Summary ───────────────────────────────────────────────────────────

def group_results(results, num_tokens):
    """Component별 결과를 카테고리로 그룹화.

    self_attn은 QKV + RoPE + attention core + O proj을 포함하므로,
    Attention Core = self_attn - (q_proj + k_proj + v_proj + o_proj + rotary)로 분리.
    """
    # self_attn에서 sub-component 빼서 core attention 추출
    self_attn_ms = results.get("self_attn", 0)
    sub_ms = sum(results.get(k, 0) for k in ["q_proj", "k_proj", "v_proj", "o_proj", "rotary"])
    attn_core_ms = max(self_attn_ms - sub_ms, 0)

    groups = {
        "Embedding":      ["embed"],
        "Attn Norm":      ["attn_norm", "norm"],
        "QKV Proj":       ["q_proj", "k_proj", "v_proj"],
        "Attn Core":      [],  # 특별 처리
        "O Proj":         ["o_proj"],
        "MLP Norm":       ["mlp_norm"],
        "MLP":            ["mlp", "gate_proj", "up_proj", "down_proj"],
        "Final Norm":     ["final_norm"],
        "LM Head":        ["lm_head"],
        "Rotary":         ["rotary"],
    }

    grouped = {}
    accounted = set()
    for group_name, keys in groups.items():
        if group_name == "Attn Core":
            if attn_core_ms > 0:
                grouped[group_name] = attn_core_ms
            accounted.add("self_attn")
            continue
        total = 0.0
        for k in keys:
            if k in results:
                total += results[k]
                accounted.add(k)
        if total > 0:
            grouped[group_name] = total

    # Unaccounted
    other = sum(v for k, v in results.items() if k not in accounted)
    if other > 0:
        grouped["Other"] = other

    return grouped


def print_breakdown(grouped, num_tokens, label=""):
    """Component breakdown 출력."""
    total = sum(grouped.values())
    per_token = total / max(num_tokens, 1)

    print(f"\n{'─' * 60}")
    if label:
        print(f"  {label}")
    print(f"  Total: {total:.2f} ms | {num_tokens} tokens | {per_token:.3f} ms/token")
    print(f"{'─' * 60}")
    print(f"  {'Component':>18s} │ {'Time(ms)':>10s} │ {'%':>6s} │ {'ms/tok':>8s}")
    print(f"  {'─' * 50}")

    for name, ms in sorted(grouped.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total > 0 else 0
        tok_ms = ms / max(num_tokens, 1)
        print(f"  {name:>18s} │ {ms:>10.2f} │ {pct:>5.1f}% │ {tok_ms:>8.4f}")

    print(f"  {'─' * 50}")

    # High-level: Linear vs Attention vs Other
    linear_keys = {"QKV Proj", "O Proj", "MLP", "LM Head"}
    norm_keys = {"Attn Norm", "MLP Norm", "Final Norm"}
    attn_keys = {"Attn Core"}

    linear_ms = sum(grouped.get(k, 0) for k in linear_keys)
    norm_ms = sum(grouped.get(k, 0) for k in norm_keys)
    attn_ms = sum(grouped.get(k, 0) for k in attn_keys)
    other_ms = total - linear_ms - norm_ms - attn_ms

    print(f"\n  High-level breakdown:")
    print(f"    Linear (QKV+O+MLP+Head): {linear_ms:>8.2f} ms ({linear_ms/total*100:>5.1f}%)")
    print(f"    Attention:               {attn_ms:>8.2f} ms ({attn_ms/total*100:>5.1f}%)")
    print(f"    Norm:                    {norm_ms:>8.2f} ms ({norm_ms/total*100:>5.1f}%)")
    print(f"    Other:                   {other_ms:>8.2f} ms ({other_ms/total*100:>5.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E2E decode latency breakdown for Llama-2-7B")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--context-length", type=int, default=1024,
                        help="Prefill context length")
    parser.add_argument("--decode-tokens", type=int, default=32,
                        help="Number of decode tokens to generate")
    parser.add_argument("--warmup-tokens", type=int, default=5,
                        help="Warmup decode tokens (not measured)")
    parser.add_argument("--cache-dir", type=str, default="\data\models")
    parser.add_argument("--attn-backend", type=str, nargs="+",
                        default=["sdpa", "splitk", "splitk_oracle", "oracle_async"],
                        help="Attention backends to benchmark (space-separated). "
                             "Options: sdpa, splitk, splitk_oracle, splitk_async, oracle_async")
    parser.add_argument("--num-splits", type=int, default=32,
                        help="Split-K splits (for splitk backends)")
    parser.add_argument("--trace", type=str, default=None,
                        help="torch.profiler trace 저장 경로 (e.g., trace.json)")
    parser.add_argument("--save", type=str, default=None,
                        help="결과 JSON 저장 경로")
    args = parser.parse_args()

    device = args.device
    model_name = args.model
    if "/" in model_name or os.sep in model_name:
        model_name = os.path.abspath(model_name)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context length: {args.context_length}")
    print(f"Decode tokens: {args.decode_tokens} (+ {args.warmup_tokens} warmup)")
    print()

    # ── Load model ──
    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    # 모델은 항상 sdpa로 로드 (splitk backends는 monkey-patch로 교체)
    print("Loading model (FP16, sdpa attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
        use_safetensors=False,
        **load_kw,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_params = sum(p.numel() for p in model.parameters())
    attn_impl = getattr(model.config, '_attn_implementation', 'unknown')
    print(f"  Attention impl: {attn_impl}")
    print(f"  Parameters: {num_params / 1e9:.2f}B")
    print(f"  Memory: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")

    # ── Prepare input (batch_size × context_length) ──
    B = args.batch_size
    base_text = "The future of artificial intelligence is fascinating and complex. "
    tokens_per_rep = len(tokenizer.encode(base_text, add_special_tokens=False))
    reps_needed = (args.context_length // max(tokens_per_rep, 1)) + 10
    text = base_text * reps_needed
    single_ids = tokenizer.encode(text, return_tensors="pt",
                                   add_special_tokens=True,
                                   max_length=args.context_length,
                                   truncation=True).to(device)
    if single_ids.shape[1] < args.context_length:
        pad_len = args.context_length - single_ids.shape[1]
        pad_ids = torch.randint(100, 10000, (1, pad_len), device=device)
        single_ids = torch.cat([single_ids, pad_ids], dim=1)
    single_ids = single_ids[:, :args.context_length]
    input_ids = single_ids.expand(B, -1).contiguous()
    print(f"  Input: batch={B}, tokens={input_ids.shape[1]}")
    print()

    # ── Attention backend setup ──
    import splitk_kernels as sk
    import math as _math
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    _active_backend = ["sdpa"]  # mutable container for closure

    def custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, **kwargs):
        _, _, Lq, D = query.shape
        if Lq > 1:
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)

        sm_scale = scale if scale is not None else 1.0 / _math.sqrt(D)
        q = query.contiguous().half()
        k = key.contiguous().half()
        v = value.contiguous().half()
        ns = args.num_splits
        backend = _active_backend[0]

        if backend == "sdpa":
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)
        elif backend == "splitk":
            return sk.attention(q, k, v, sm_scale, ns)
        elif backend == "splitk_oracle":
            B_q, H_q = q.shape[0], q.shape[1]
            gm = torch.full((B_q * H_q,), 6.5, device=q.device, dtype=torch.float32)
            return sk.attention_oracle(q, k, v, gm, sm_scale, ns)
        elif backend == "splitk_async":
            return sk.attention_localmax_async(q, k, v, sm_scale, ns)
        elif backend == "oracle_async":
            B_q, H_q = q.shape[0], q.shape[1]
            gm = torch.full((B_q * H_q,), 6.5, device=q.device, dtype=torch.float32)
            return sk.attention_oracle_async(q, k, v, gm, sm_scale, ns)
        else:
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = custom_sdpa

    # ── Helper: run decode and measure throughput ──
    def run_decode(backend_name):
        """Prefill → warmup → measured decode. Returns (wall_ms, tok/s)."""
        _active_backend[0] = "sdpa"  # prefill은 항상 sdpa
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        _active_backend[0] = backend_name  # decode에서 선택된 backend 사용

        # Warmup
        token = next_token
        past = past_kv
        with torch.no_grad():
            for _ in range(args.warmup_tokens):
                out = model(token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        # Measured decode
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.decode_tokens):
                out = model(token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        wall_ms = (t_end - t_start) * 1000
        tok_s = args.decode_tokens / (wall_ms / 1000)
        return wall_ms, tok_s

    # ── Run all backends ──
    backends = args.attn_backend
    results_summary = {}

    for backend in backends:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend}" + (f" (splits={args.num_splits})" if backend != "sdpa" else ""))
        print(f"{'=' * 60}")

        # 3 runs, take best
        best_ms = float('inf')
        best_toks = 0
        for trial in range(3):
            torch.cuda.empty_cache()
            wall_ms, tok_s = run_decode(backend)
            if wall_ms < best_ms:
                best_ms = wall_ms
                best_toks = tok_s
            print(f"  Run {trial+1}: {wall_ms:.2f} ms, {tok_s:.1f} tok/s")

        results_summary[backend] = {
            "wall_ms": best_ms,
            "per_token_ms": best_ms / args.decode_tokens,
            "tokens_per_sec": best_toks,
        }
        print(f"  Best: {best_ms:.2f} ms, {best_toks:.1f} tok/s")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY — Llama-2-7B Decode Throughput")
    print(f"  batch={args.batch_size}, ctx={args.context_length}, "
          f"decode={args.decode_tokens} tokens, splits={args.num_splits}")
    print(f"{'=' * 70}")
    print(f"  {'Backend':>18s} │ {'ms/token':>10s} │ {'tok/s':>10s} │ {'vs sdpa':>10s}")
    print(f"  {'─' * 55}")

    sdpa_toks = results_summary.get("sdpa", {}).get("tokens_per_sec", 0)
    for backend, res in results_summary.items():
        ms_tok = res["per_token_ms"]
        tok_s = res["tokens_per_sec"]
        if sdpa_toks > 0 and backend != "sdpa":
            speedup = tok_s / sdpa_toks
            vs_str = f"{speedup:.2f}x"
        else:
            vs_str = "baseline"
        print(f"  {backend:>18s} │ {ms_tok:>10.3f} │ {tok_s:>10.1f} │ {vs_str:>10s}")

    print(f"  {'─' * 55}")

    # ── Save results ──
    if args.save:
        save_data = {
            "config": {
                "model": args.model,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "decode_tokens": args.decode_tokens,
                "num_splits": args.num_splits,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0),
            },
            "results": results_summary,
        }
        with open(args.save, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.save}")


if __name__ == "__main__":
    main()
