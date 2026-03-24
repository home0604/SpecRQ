"""
RQ Decode Latency Benchmark
============================
RQ codebook lookup의 stage별 latency를 측정.
Draft (소수 stage) vs Target (전체 stage)의 실제 시간 차이 확인.

측정:
  1. Stage별 decode latency (1, 2, 4, 8, 12 stages)
  2. Sequence length별 scaling
  3. SDPA attention 시간과 비교 (RQ decode + attention vs FP16 attention)

Usage:
  python experiments/rq_stages/bench_decode_latency.py \
      --codebook /data/home/lsh/ckpts/wikitext/both_quant_... \
      --seq-lens 1024 4096 32768 131072
"""

import argparse
import math
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from src.kv_cache.rq_cache import RQKVCache


def main():
    parser = argparse.ArgumentParser(description="RQ decode latency benchmark")
    parser.add_argument("--codebook", type=str,
                        default="/data/home/lsh/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256]_Value_K[12]C[256]D[256]_SubVec1_Stages1")
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536])
    parser.add_argument("--draft-stages", type=int, nargs="+",
                        default=[1, 2, 4, 8, 12])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device
    B = args.batch_size
    H_kv = args.num_kv_heads
    H_q = args.num_q_heads
    D = args.head_dim

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"RQ Decode Latency Benchmark")
    print(f"  Batch: {B}, H_kv: {H_kv}, H_q: {H_q}, D: {D}")
    print(f"  Seq lengths: {args.seq_lens}")
    print(f"  Draft stages: {args.draft_stages}")
    print()

    # ── Load codebook ──
    rq = RQKVCache(device=device)
    rq.load_codebooks(args.codebook, H_kv, D)

    full_stages = rq.num_stages_key
    num_pq = rq.num_pq_codebooks
    C = rq.codebook_size
    D_cb = rq.codebook_dim
    total_kv_dim = H_kv * D

    print(f"\n  Full stages: {full_stages}")
    print(f"  Codebook: {C}C × {D_cb}D × {num_pq} PQ")
    bpa_full = full_stages * math.log2(C) / D_cb
    print(f"  BPA (full): {bpa_full:.4f}")

    # ── Benchmark results ──
    results = {}

    for Lk in args.seq_lens:
        print(f"\n{'─' * 70}")
        print(f"  Seq length: {Lk}")
        print(f"{'─' * 70}")

        # Pre-create fake indices (simulating stored RQ cache)
        indices = torch.randint(0, C, (B, Lk, full_stages, num_pq),
                                device=device, dtype=torch.int16)

        # FP16 KV for SDPA comparison
        K_fp16 = torch.randn(B, H_kv, Lk, D, device=device, dtype=torch.float16)
        V_fp16 = torch.randn(B, H_kv, Lk, D, device=device, dtype=torch.float16)
        Q = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float16)
        sm_scale = 1.0 / math.sqrt(D)

        # ── 1. SDPA baseline (FP16 KV, no RQ) ──
        sdpa_ms = do_bench(
            lambda: F.scaled_dot_product_attention(Q, K_fp16, V_fp16, scale=sm_scale),
            warmup=args.warmup, rep=args.rep)

        # ── 2. RQ decode only (per stage count) ──
        print(f"\n  {'Stages':>7s} {'BPA':>6s} │ {'Decode(ms)':>10s} │ {'SDPA(ms)':>9s} │ "
              f"{'Total(ms)':>10s} │ {'vs FP16':>8s} │ {'Decode%':>8s}")
        print(f"  {'─' * 72}")

        # FP16 baseline
        print(f"  {'FP16':>7s} {'16.00':>6s} │ {'0.000':>10s} │ {sdpa_ms:>9.4f} │ "
              f"{sdpa_ms:>10.4f} │ {'1.00x':>8s} │ {'0.0%':>8s}")

        for ns in args.draft_stages:
            if ns > full_stages:
                continue

            # RQ decode: indices → KV tensors
            decode_ms = do_bench(
                lambda: rq.decode(indices, layer_idx=0, kv_type="key", num_stages=ns),
                warmup=args.warmup, rep=args.rep)

            # RQ decode + SDPA attention
            def rq_then_attn():
                K_rq = rq.decode(indices, 0, "key", ns)
                V_rq = rq.decode(indices, 0, "value", ns)
                # GQA expansion if needed
                if H_q != H_kv:
                    K_rq = K_rq.repeat_interleave(H_q // H_kv, dim=1)
                    V_rq = V_rq.repeat_interleave(H_q // H_kv, dim=1)
                return F.scaled_dot_product_attention(Q, K_rq, V_rq, scale=sm_scale)

            total_ms = do_bench(rq_then_attn, warmup=args.warmup, rep=args.rep)

            bpa = ns * math.log2(C) / D_cb
            vs_fp16 = total_ms / sdpa_ms if sdpa_ms > 0 else 0
            decode_pct = decode_ms / total_ms * 100 if total_ms > 0 else 0

            print(f"  {ns:>7d} {bpa:>6.3f} │ {decode_ms:>10.4f} │ {sdpa_ms:>9.4f} │ "
                  f"{total_ms:>10.4f} │ {vs_fp16:>7.2f}x │ {decode_pct:>7.1f}%")

            results[f"Lk{Lk}_s{ns}"] = {
                "seq_len": Lk, "stages": ns, "bpa": bpa,
                "decode_ms": decode_ms, "sdpa_ms": sdpa_ms,
                "total_ms": total_ms, "vs_fp16": vs_fp16,
                "decode_pct": decode_pct,
            }

        del indices, K_fp16, V_fp16, Q
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: RQ Decode Overhead")
    print(f"{'=' * 70}")
    print(f"\n  RQ decode = codebook lookup × stages × PQ subvectors")
    print(f"  Total = RQ decode (K+V) + SDPA attention")
    print(f"  vs FP16 = (RQ decode + attention) / FP16 attention\n")

    # Table: seq_len × stages
    print(f"  {'':>8s}", end="")
    for ns in args.draft_stages:
        if ns <= full_stages:
            print(f" │ {'s=' + str(ns):>10s}", end="")
    print(f" │ {'FP16':>10s}")
    print(f"  {'─' * (12 + 13 * (len(args.draft_stages) + 1))}")

    for Lk in args.seq_lens:
        print(f"  {Lk:>8d}", end="")
        for ns in args.draft_stages:
            key = f"Lk{Lk}_s{ns}"
            if key in results:
                print(f" │ {results[key]['total_ms']:>9.3f}ms", end="")
        # FP16
        key0 = f"Lk{Lk}_s{args.draft_stages[0]}"
        if key0 in results:
            print(f" │ {results[key0]['sdpa_ms']:>9.3f}ms", end="")
        print()

    # Speedup of draft vs target
    print(f"\n  Draft speedup (stage N → fewer lookups → faster):")
    for Lk in args.seq_lens:
        target_key = f"Lk{Lk}_s{full_stages}"
        if target_key not in results:
            continue
        target_ms = results[target_key]["total_ms"]
        parts = []
        for ns in args.draft_stages:
            key = f"Lk{Lk}_s{ns}"
            if key in results:
                speedup = target_ms / results[key]["total_ms"]
                parts.append(f"s={ns}: {speedup:.2f}x")
        print(f"    Lk={Lk}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
