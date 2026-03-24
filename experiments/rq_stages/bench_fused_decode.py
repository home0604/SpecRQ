"""
Fused RQ Decode Benchmark (using VQLLM's Triton kernel)
=======================================================
VQLLM의 _residual_kernel을 직접 호출하여 fused RQ decode latency 측정.
Python loop (192 kernel launches) 대신 single Triton kernel로 전체 RQ decode.

GSRQ Table 7 재현:
  FP16 vs RQ per-layer latency, batch size sweep

Usage:
  conda activate sinkspec
  CUDA_VISIBLE_DEVICES=1 python experiments/rq_stages/bench_fused_decode.py \
      --codebook /data/home/lsh/ckpts/wikitext/both_quant_...
"""

import argparse
import math
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# VQLLM Triton kernel import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "VQLLM"))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench

# Mock cupy/cuml (only needed for K-means init, not inference)
import types
for mod_name in ['cupy', 'cuml', 'cuml.cluster']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
if not hasattr(sys.modules['cuml.cluster'], 'KMeans'):
    sys.modules['cuml.cluster'].KMeans = type('KMeans', (), {})

# Import VQLLM's fused Triton kernel directly
from vqllm.quantizer.residual_kernel import _residual_kernel


def load_codebook_tensors(ckpt_dir, layer_idx=0, kv_type="key", device="cuda"):
    """체크포인트에서 codebook/gain 텐서를 _residual_kernel 형식으로 준비.

    Returns:
        shape: (num_PQ * num_stages * C, D) — concatenated codebooks
        shape_normed: (D, num_PQ * num_stages * C) — transposed
        gain: (num_PQ * num_stages * C,) — concatenated gains
        config: dict with num_stages, C, D, num_PQ, etc.
    """
    ckpt = {}
    for f in sorted(os.listdir(ckpt_dir)):
        if f.endswith(".pt") and "model" in f:
            part = torch.load(os.path.join(ckpt_dir, f), map_location="cpu", weights_only=False)
            ckpt.update(part)

    # Detect key prefix
    prefix = None
    num_stages = 0
    num_pq = 0
    C, D = 0, 0
    for k in ckpt.keys():
        m = re.match(
            r"((?:model\.)?layers\.\d+\.(?:self_attn|attention)\.quantizer)\."
            + re.escape(kv_type) + r"\.(\d+)\.codebooks\.(\d+)\.", k)
        if m:
            prefix = m.group(1).replace(f"layers.{layer_idx}", "layers.{l}")
            num_stages = max(num_stages, int(m.group(2)) + 1)
            num_pq = max(num_pq, int(m.group(3)) + 1)
            if C == 0:
                sample = ckpt[k]
                C, D = sample.shape[0], sample.shape[1]

    print(f"  Layer {layer_idx}, {kv_type}: {num_stages} stages, {num_pq} PQ, C={C}, D={D}")

    # Build concatenated tensors like residual_kernel expects:
    # shape[pq_idx] = concat([stage_0_cb, stage_1_cb, ...], dim=0) → (num_stages*C, D)
    # Then stack across PQ: (num_PQ, num_stages*C, D) → view(-1, D)
    shape_list = []
    gain_list = []

    for pq in range(num_pq):
        pq_cbs = []
        pq_gains = []
        pfx = prefix.format(l=layer_idx)
        for s in range(num_stages):
            cb_key = f"{pfx}.{kv_type}.{s}.codebooks.{pq}.weight"
            g_key = f"{pfx}.{kv_type}.{s}.gains"
            pq_cbs.append(ckpt[cb_key].float())
            pq_gains.append(ckpt[g_key][pq].float())

        shape_list.append(torch.cat(pq_cbs, dim=0))    # (num_stages*C, D)
        gain_list.append(torch.cat(pq_gains, dim=0))    # (num_stages*C,)

    shape = torch.stack(shape_list, dim=0).to(device)          # (num_PQ, num_stages*C, D)
    gain = torch.stack(gain_list, dim=0).to(device)            # (num_PQ, num_stages*C)

    shape_flat = shape.view(-1, D).contiguous()                 # (num_PQ*num_stages*C, D)
    shape_normed = shape_flat.t().contiguous()                  # (D, num_PQ*num_stages*C)
    gain_flat = gain.view(-1).contiguous()                      # (num_PQ*num_stages*C,)

    config = {
        "num_stages": num_stages, "C": C, "D": D, "num_PQ": num_pq,
        "bpa": num_stages * math.log2(C) / D,
    }
    return shape_flat, shape_normed, gain_flat, config


def run_fused_rq_decode(x, shape, shape_normed, gain, config, num_stages=None):
    """_residual_kernel을 직접 호출하여 fused RQ decode.

    Args:
        x: (1, seq_len, H_kv, 1, D_head) or reshaped to (num_PQ*seq_len, D)
        shape, shape_normed, gain: from load_codebook_tensors
        config: codebook config dict
        num_stages: number of stages to decode (None=all)
    Returns:
        xq: same shape as x
    """
    if num_stages is None:
        num_stages = config["num_stages"]

    C = config["C"]
    D = config["D"]
    num_PQ = config["num_PQ"]

    x_shape = x.shape
    max_seq_len = x_shape[1] if x.ndim == 5 else x_shape[0]

    # Reshape like VQLLM: (1, seq_len, H, 1, D_head) → (num_PQ*seq_len, D)
    if x.ndim == 5:
        x_flat = x.view(1, max_seq_len, -1, D).transpose(1, 2).contiguous().view(-1, D)
    else:
        x_flat = x.contiguous()

    xq = torch.empty_like(x_flat)

    # Adjust shape/gain for partial stages (only use first num_stages)
    if num_stages < config["num_stages"]:
        # shape is (num_PQ, total_stages*C, D) stacked → need first num_stages*C per PQ
        full_stages = config["num_stages"]
        # Rebuild partial tensors
        shape_partial_list = []
        gain_partial_list = []
        shape_full = shape.view(num_PQ, full_stages * C, D)
        gain_full = gain.view(num_PQ, full_stages * C)
        for pq in range(num_PQ):
            shape_partial_list.append(shape_full[pq, :num_stages * C, :])
            gain_partial_list.append(gain_full[pq, :num_stages * C])
        s = torch.stack(shape_partial_list, dim=0).reshape(-1, D).contiguous()
        sn = s.t().contiguous()
        g = torch.stack(gain_partial_list, dim=0).reshape(-1).contiguous()
    else:
        s, sn, g = shape, shape_normed, gain

    # Kernel config — D=256일 때 smem = BLOCK_SIZE_X * D + BLOCK_SIZE_C * D
    # 166KB limit: 32*256 + 64*256 = 24KB (safe)
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_C = 64
    num_warps = 4

    M, N = x_flat.shape
    grid = (triton.cdiv(max_seq_len, BLOCK_SIZE_X), num_PQ)

    _residual_kernel[grid](
        x_flat, xq, s, sn, g,
        x_flat.stride(0), xq.stride(0), s.stride(0), sn.stride(0), g.stride(0),
        M, N, max_seq_len,
        num_stages, C,
        BLOCK_SIZE_X, BLOCK_SIZE_C,
        num_warps=num_warps,
    )

    # Restore shape
    if x.ndim == 5:
        total_subvectors = num_PQ
        xq = xq.view(total_subvectors, -1, D).transpose(0, 1).contiguous().view(*x_shape)

    return xq


def main():
    parser = argparse.ArgumentParser(description="Fused RQ decode benchmark (VQLLM Triton)")
    parser.add_argument("--codebook", type=str,
                        default="/data/home/lsh/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256]_Value_K[12]C[256]D[256]_SubVec1_Stages1")
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--draft-stages", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    H_kv = args.num_kv_heads
    D_head = args.head_dim

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Fused RQ Decode Benchmark (VQLLM Triton kernel)")
    print(f"  Seq len: {args.seq_len}, H_kv: {H_kv}, D: {D_head}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print()

    # Load codebooks for layer 0
    shape_k, shape_normed_k, gain_k, cfg_k = load_codebook_tensors(
        args.codebook, layer_idx=0, kv_type="key", device=device)
    shape_v, shape_normed_v, gain_v, cfg_v = load_codebook_tensors(
        args.codebook, layer_idx=0, kv_type="value", device=device)

    print(f"\n  Config: {cfg_k['num_stages']} stages, C={cfg_k['C']}, D={cfg_k['D']}, "
          f"PQ={cfg_k['num_PQ']}, BPA={cfg_k['bpa']:.3f}")

    # ── Benchmark ──
    print(f"\n{'=' * 80}")
    print(f"  GSRQ Table 7 재현: Layer-wise latency per token (ms)")
    print(f"  Context = {args.seq_len}, A100")
    print(f"{'=' * 80}")

    # Header
    hdr = f"  {'Batch':>6s} │ {'FP16':>8s}"
    for ns in args.draft_stages:
        bpa = ns * math.log2(cfg_k['C']) / cfg_k['D']
        hdr += f" │ {'s=' + str(ns) + f' ({bpa:.2f}b)':>14s}"
    print(hdr)
    print(f"  {'─' * (len(hdr) - 2)}")

    for B in args.batch_sizes:
        Lk = args.seq_len

        # FP16 baseline: SDPA attention only (per layer)
        Q_fp16 = torch.randn(B, H_kv, 1, D_head, device=device, dtype=torch.float16)
        K_fp16 = torch.randn(B, H_kv, Lk, D_head, device=device, dtype=torch.float16)
        V_fp16 = torch.randn(B, H_kv, Lk, D_head, device=device, dtype=torch.float16)
        sm = 1.0 / math.sqrt(D_head)

        fp16_ms = do_bench(
            lambda: F.scaled_dot_product_attention(Q_fp16, K_fp16, V_fp16, scale=sm),
            warmup=args.warmup, rep=args.rep)

        row = f"  {B:>6d} │ {fp16_ms:>8.3f}"

        # RQ fused decode + attention
        # Input: (1, B*Lk, H_kv, 1, D_head)
        x_key = torch.randn(1, B * Lk, H_kv, 1, D_head, device=device, dtype=torch.float32)
        x_val = torch.randn(1, B * Lk, H_kv, 1, D_head, device=device, dtype=torch.float32)

        for ns in args.draft_stages:
            if ns > cfg_k['num_stages']:
                row += f" │ {'N/A':>14s}"
                continue

            def fused_rq_attn():
                kq = run_fused_rq_decode(x_key, shape_k, shape_normed_k, gain_k, cfg_k, ns)
                vq = run_fused_rq_decode(x_val, shape_v, shape_normed_v, gain_v, cfg_v, ns)
                # Reshape for attention: (1, B*Lk, H, 1, D) → (B, H, Lk, D)
                K_rq = kq.view(B, Lk, H_kv, D_head).permute(0, 2, 1, 3).half()
                V_rq = vq.view(B, Lk, H_kv, D_head).permute(0, 2, 1, 3).half()
                return F.scaled_dot_product_attention(Q_fp16, K_rq, V_rq, scale=sm)

            rq_ms = do_bench(fused_rq_attn, warmup=args.warmup, rep=args.rep)
            speedup = fp16_ms / rq_ms if rq_ms > 0 else 0
            row += f" │ {rq_ms:>7.3f} ({speedup:.1f}x)"

        print(row)

        del Q_fp16, K_fp16, V_fp16, x_key, x_val
        torch.cuda.empty_cache()

    # ── GSRQ reference ──
    print(f"\n  GSRQ Table 7 참고 (LLaMA-3-8B, ctx=1K, A100 80GB SXM):")
    print(f"    Batch  FP16   GSRQ   Speedup")
    print(f"      8    2.3ms  1.2ms  1.9×")
    print(f"     16    3.9ms  1.7ms  2.3×")
    print(f"     32    7.9ms  2.3ms  3.4×")


if __name__ == "__main__":
    main()
