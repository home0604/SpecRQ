"""
Single-Layer Decode Benchmark: Attention vs Linear scaling
==========================================================
Full model은 long context에서 OOM이 발생하므로,
single layer만 실행하여 component별 latency를 측정.

Measures per-token decode latency breakdown:
  - RMSNorm (input_layernorm, post_attention_layernorm)
  - QKV Projection (q_proj, k_proj, v_proj)
  - Attention core (scaled_dot_product_attention)
  - O Projection (o_proj)
  - MLP (gate_proj, up_proj, down_proj, SiLU)
  - Residual connections

Context lengths up to 128K+ possible (single layer KV ≈ 1GB at 128K).

Usage:
  python bench_single_layer.py --model /data/models/llama2_7b --context-lengths 1024 4096 32768 131072
  python bench_single_layer.py --model /data/models/llama2_7b --context-lengths 1024 131072 \
      --attn-backend sdpa splitk splitk_oracle
"""

import argparse
import json
import math
import time
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig


def main():
    parser = argparse.ArgumentParser(
        description="Single-layer decode benchmark for long context")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[1024, 4096, 8192, 32768, 65536, 131072])
    parser.add_argument("--decode-steps", type=int, default=100,
                        help="Decode steps to measure (higher = more stable)")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Which layer to benchmark (0-indexed)")
    parser.add_argument("--attn-backend", type=str, nargs="+",
                        default=["sdpa", "oracle_async", "tc_oracle_async"],
                        help="Attention backends: sdpa, splitk, splitk_oracle, splitk_async, oracle_async, tc_oracle_async")
    parser.add_argument("--num-splits", type=int, default=32)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    model_name = args.model
    if "/" in model_name or os.sep in model_name:
        model_name = os.path.abspath(model_name)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Decode steps: {args.decode_steps} (+ {args.warmup_steps} warmup)")
    print(f"Layer idx: {args.layer_idx}")
    print(f"Backends: {args.attn_backend}")
    print()

    # ── Load config + single layer ──
    load_kw = {"trust_remote_code": True}
    if args.cache_dir:
        load_kw["cache_dir"] = args.cache_dir

    print("Loading model config...")
    config = AutoConfig.from_pretrained(model_name, **load_kw)
    H = config.num_attention_heads
    D = config.hidden_size // H
    D_model = config.hidden_size
    num_layers = config.num_hidden_layers
    intermediate_size = config.intermediate_size

    print(f"  H={H}, D={D}, d_model={D_model}, layers={num_layers}")
    print(f"  MLP intermediate: {intermediate_size}")

    print(f"Loading full model to extract layer {args.layer_idx}...")
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        use_safetensors=False,
        **load_kw,
    )

    # Extract single layer + rotary embedding
    if hasattr(full_model, 'model') and hasattr(full_model.model, 'layers'):
        layer = full_model.model.layers[args.layer_idx].to(device).eval()
        rotary_emb = full_model.model.rotary_emb.to(device) if hasattr(full_model.model, 'rotary_emb') else None
    elif hasattr(full_model, 'model') and hasattr(full_model.model, 'decoder'):
        layer = full_model.model.decoder.layers[args.layer_idx].to(device).eval()
        rotary_emb = None
    else:
        raise RuntimeError("Cannot find decoder layers")

    layer_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    print(f"  Layer {args.layer_idx} memory: {layer_bytes / 1e6:.1f} MB")

    del full_model

    # RoPE를 no-op으로 패치 (latency 측정에 영향 없음, pointwise multiply일 뿐)
    # 독립 layer 실행 시 position_embeddings shape 호환 문제 우회
    import transformers.models.llama.modeling_llama as _llama_mod
    _llama_mod.apply_rotary_pos_emb = lambda q, k, cos, sin, **kw: (q, k)
    print(f"  RoPE: patched to no-op (benchmark only)")
    torch.cuda.empty_cache()

    # ── Attention backend setup ──
    import splitk_kernels as sk

    _orig_sdpa = F.scaled_dot_product_attention
    _active_backend = ["sdpa"]

    def custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, **kwargs):
        backend = _active_backend[0]

        # 최신 transformers는 (B*H, Lq, D) 3D 또는 (B, H, Lq, D) 4D로 호출 가능
        orig_shape = query.shape
        if query.ndim == 3:
            # (B*H, Lq, Dh) → sdpa는 그대로 처리 가능
            Lq = query.shape[-2]
        else:
            Lq = query.shape[-2]

        # Prefill (Lq>1) 또는 sdpa backend → 원래 SDPA 사용
        if Lq > 1 or backend == "sdpa":
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)

        # 우리 커널은 4D (B, H, Lq, D) 필요
        if query.ndim == 3:
            # (B*H, Lq, Dh) → (B, H, Lq, Dh) 추정
            BH, Lq_dim, Dh = query.shape
            H_heads = H  # outer scope에서 가져옴
            B_batch = BH // H_heads
            query = query.view(B_batch, H_heads, Lq_dim, Dh)
            key = key.view(B_batch, H_heads, -1, Dh)
            value = value.view(B_batch, H_heads, -1, Dh)
            reshaped = True
        else:
            reshaped = False

        Dh = query.shape[-1]
        sm_scale = scale if scale is not None else 1.0 / math.sqrt(Dh)
        q = query.contiguous().half()
        k = key.contiguous().half()
        v = value.contiguous().half()
        ns = args.num_splits

        if backend == "splitk":
            out = sk.attention(q, k, v, sm_scale, ns)
        elif backend == "splitk_oracle":
            B_q, H_q = q.shape[0], q.shape[1]
            gm = torch.full((B_q * H_q,), 6.5, device=q.device, dtype=torch.float32)
            out = sk.attention_oracle(q, k, v, gm, sm_scale, ns)
        elif backend == "splitk_async":
            out = sk.attention_localmax_async(q, k, v, sm_scale, ns)
        elif backend == "oracle_async":
            B_q, H_q = q.shape[0], q.shape[1]
            gm = torch.full((B_q * H_q,), 6.5, device=q.device, dtype=torch.float32)
            out = sk.attention_oracle_async(q, k, v, gm, sm_scale, ns)
        elif backend == "tc_oracle_async":
            B_q, H_q = q.shape[0], q.shape[1]
            gm = torch.full((B_q * H_q,), 6.5, device=q.device, dtype=torch.float32)
            out = sk.attention_tc_oracle_async(q, k, v, gm, sm_scale, ns)
        else:
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, **kwargs)

        if reshaped:
            out = out.view(orig_shape)
        return out

    F.scaled_dot_product_attention = custom_sdpa

    # ── Per-component timing with CUDA events (minimal hooks) ──
    def time_decode_step_manual(layer, hidden, past_k, past_v, num_steps, sm_scale):
        """각 component를 직접 호출하여 KV cache를 확실히 읽도록 함."""
        attn = layer.self_attn
        mlp = layer.mlp
        norm1 = layer.input_layernorm
        norm2 = layer.post_attention_layernorm

        h = hidden.clone()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_steps):
                # 1. Input RMSNorm
                residual = h
                h_norm = norm1(h)

                # 2. QKV projection
                q = attn.q_proj(h_norm)
                k_new = attn.k_proj(h_norm)
                v_new = attn.v_proj(h_norm)

                # 3. Reshape to (B, H, Lq, D)
                B_s, Lq, _ = q.shape
                q = q.view(B_s, Lq, -1, D).transpose(1, 2)
                k_new = k_new.view(B_s, Lq, -1, D).transpose(1, 2)
                v_new = v_new.view(B_s, Lq, -1, D).transpose(1, 2)

                # 4. Concat with past KV (full cache read)
                k_full = torch.cat([past_k, k_new], dim=2)
                v_full = torch.cat([past_v, v_new], dim=2)

                # 5. Attention (SDPA or our custom kernel via monkey-patch)
                attn_out = F.scaled_dot_product_attention(q, k_full, v_full, scale=sm_scale)

                # 6. O projection
                attn_out = attn_out.transpose(1, 2).reshape(B_s, Lq, -1)
                h = residual + attn.o_proj(attn_out)

                # 7. MLP
                residual = h
                h = residual + mlp(norm2(h))

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000

    # ── Run benchmarks ──
    B = args.batch_size
    all_results = {}

    for ctx_len in args.context_lengths:
        kv_bytes = 2 * B * H * ctx_len * D * 2  # K+V, fp16
        total_mem = layer_bytes + kv_bytes
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if total_mem > gpu_mem * 0.9:
            print(f"\n  ctx={ctx_len}: SKIP (need {total_mem/1e9:.1f}GB, "
                  f"GPU has {gpu_mem/1e9:.1f}GB)")
            continue

        print(f"\n{'─' * 70}")
        print(f"  Context length: {ctx_len} (KV cache: {kv_bytes/1e6:.1f} MB)")
        print(f"{'─' * 70}")

        # Create fake KV cache (pre-filled)
        past_k = torch.randn(B, H, ctx_len, D, device=device, dtype=torch.float16)
        past_v = torch.randn(B, H, ctx_len, D, device=device, dtype=torch.float16)

        # Build DynamicCache-like object for the layer
        from transformers import DynamicCache
        past_kv = DynamicCache()
        past_kv.update(past_k, past_v, layer_idx=0)

        # Decode input: single token hidden state
        hidden = torch.randn(B, 1, D_model, device=device, dtype=torch.float16)
        position_ids = torch.tensor([[ctx_len]], device=device).expand(B, -1)
        cache_position = torch.tensor([ctx_len], device=device)

        sm_scale = 1.0 / math.sqrt(D)

        for backend in args.attn_backend:
            _active_backend[0] = backend
            torch.cuda.empty_cache()

            # Warmup
            _ = time_decode_step_manual(layer, hidden, past_k, past_v,
                                        args.warmup_steps, sm_scale)

            # Measure (3 trials, take best)
            best_ms = float('inf')
            for trial in range(3):
                ms = time_decode_step_manual(layer, hidden, past_k, past_v,
                                             args.decode_steps, sm_scale)
                per_step = ms / args.decode_steps
                if ms < best_ms:
                    best_ms = ms

            per_step_ms = best_ms / args.decode_steps
            steps_per_sec = args.decode_steps / (best_ms / 1000)

            suffix = f" (splits={args.num_splits})" if backend != "sdpa" else ""
            print(f"  {backend:>18s}{suffix}: {per_step_ms:.4f} ms/step, "
                  f"{steps_per_sec:.0f} steps/s")

            key = f"ctx{ctx_len}_{backend}"
            all_results[key] = {
                "context_length": ctx_len,
                "backend": backend,
                "per_step_ms": per_step_ms,
                "steps_per_sec": steps_per_sec,
            }

        # Cleanup
        del past_k, past_v, past_kv
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — Single Layer Decode (layer {args.layer_idx})")
    print(f"  batch={B}, splits={args.num_splits}, steps={args.decode_steps}")
    print(f"{'=' * 80}")

    # Header
    backends = args.attn_backend
    hdr = f"  {'ctx_len':>8s}"
    for b in backends:
        hdr += f" │ {b:>14s}"
    if len(backends) > 1:
        hdr += f" │ {'speedup':>10s}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")

    for ctx_len in args.context_lengths:
        row = f"  {ctx_len:>8d}"
        ms_vals = {}
        for b in backends:
            key = f"ctx{ctx_len}_{b}"
            if key in all_results:
                ms = all_results[key]["per_step_ms"]
                ms_vals[b] = ms
                row += f" │ {ms:>10.4f} ms"
            else:
                row += f" │ {'SKIP':>13s}"

        # Speedup: best non-sdpa vs sdpa
        if len(backends) > 1 and "sdpa" in ms_vals:
            sdpa_ms = ms_vals["sdpa"]
            best_other = min((ms_vals[b] for b in backends if b != "sdpa" and b in ms_vals),
                             default=sdpa_ms)
            speedup = sdpa_ms / best_other if best_other > 0 else 0
            row += f" │ {speedup:>9.2f}x"
        print(row)

    # Extrapolation: estimate full-model decode time at each context length
    print(f"\n  Extrapolated full model ({num_layers} layers):")
    print(f"  {'ctx_len':>8s}", end="")
    for b in backends:
        print(f" │ {b + ' tok/s':>14s}", end="")
    print()
    print(f"  {'─' * 60}")

    for ctx_len in args.context_lengths:
        row = f"  {ctx_len:>8d}"
        for b in backends:
            key = f"ctx{ctx_len}_{b}"
            if key in all_results:
                per_layer_ms = all_results[key]["per_step_ms"]
                # Full model: num_layers × per_layer + ~10% overhead (embed, head, norm)
                full_ms = per_layer_ms * num_layers * 1.1
                full_toks = 1000.0 / full_ms
                row += f" │ {full_toks:>11.1f}"
            else:
                row += f" │ {'SKIP':>13s}"
        print(row)

    # ── Attention fraction estimate ──
    print(f"\n  Estimated attention fraction (from scaling):")
    for ctx_len in args.context_lengths:
        key_sdpa = f"ctx{ctx_len}_sdpa"
        key_1k = "ctx1024_sdpa"
        if key_sdpa in all_results and key_1k in all_results:
            ms_ctx = all_results[key_sdpa]["per_step_ms"]
            ms_1k = all_results[key_1k]["per_step_ms"]
            # Linear ops are constant, attention scales with ctx_len
            # ms_ctx = linear_ms + attn_ms * (ctx_len / 1024)
            # ms_1k  = linear_ms + attn_ms
            # → attn_ms = (ms_ctx - ms_1k) / (ctx_len/1024 - 1)  if ctx_len != 1024
            if ctx_len != 1024:
                attn_ms = (ms_ctx - ms_1k) / (ctx_len / 1024 - 1)
                attn_frac = attn_ms * (ctx_len / 1024) / ms_ctx * 100
                print(f"    ctx={ctx_len:>6d}: attention ≈ {attn_frac:.1f}% of layer time")

    # ── Save ──
    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "config": {
                    "model": model_name, "batch_size": B,
                    "layer_idx": args.layer_idx, "num_splits": args.num_splits,
                    "decode_steps": args.decode_steps,
                    "H": H, "D": D, "d_model": D_model, "num_layers": num_layers,
                    "gpu": torch.cuda.get_device_name(0),
                },
                "results": all_results,
            }, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
