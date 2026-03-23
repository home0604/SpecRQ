"""
Flash Attention: Max Exponent Oracle Benchmark
==============================================
Softmax의 max exponent를 미리 알고 있을 때 attention 연산이
얼마나 빨라지는지 측정하는 실험.

Target model로 먼저 정답 max를 구해놓고 이를 oracle로 사용.

4가지 설정 비교:
  1. naive_attention       : 표준 O(N^2) 메모리 attention
  2. flash_attention       : Online softmax (Dao 2022 style, pure PyTorch)
  3. flash_local_oracle    : Flash + 블록별 local max를 미리 알고 있음
  4. flash_global_oracle   : Flash + row 전체 global max를 미리 알고 있음
"""

import torch
import torch.nn.functional as F
import time
import math
import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    batch_size: int = 4
    n_heads: int = 32
    seq_len: int = 2048       # KV cache length (context)
    draft_len: int = 8        # speculative draft tokens to verify
    head_dim: int = 128
    block_size: int = 256     # flash attention tile size
    dtype: str = "float16"
    device: str = "cuda"
    warmup_iters: int = 10
    bench_iters: int = 100

    @property
    def torch_dtype(self):
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]


# ---------------------------------------------------------------------------
# 1. Naive Attention (baseline)
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V, scale):
    """Standard materialized attention. O(N^2) memory."""
    # Q: (B, H, Lq, D), K/V: (B, H, Lk, D)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B,H,Lq,Lk)
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, V)
    return out


# ---------------------------------------------------------------------------
# 2. Flash Attention (online softmax, tiled)
# ---------------------------------------------------------------------------

def flash_attention(Q, K, V, scale, block_size):
    """
    Pure PyTorch flash attention with online softmax.
    Tiles over KV dimension to avoid materializing full attention matrix.
    """
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    n_blocks = math.ceil(Lk / block_size)

    # Running accumulators
    O = torch.zeros(B, H, Lq, D, dtype=Q.dtype, device=Q.device)
    m = torch.full((B, H, Lq, 1), float('-inf'), dtype=torch.float32, device=Q.device)
    l = torch.zeros((B, H, Lq, 1), dtype=torch.float32, device=Q.device)

    for j in range(n_blocks):
        start = j * block_size
        end = min(start + block_size, Lk)

        Kj = K[:, :, start:end, :]    # (B, H, bs, D)
        Vj = V[:, :, start:end, :]

        # Local scores
        S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale  # (B, H, Lq, bs)
        S_f32 = S.float()

        # Online softmax update
        m_local = S_f32.max(dim=-1, keepdim=True).values     # (B, H, Lq, 1)
        m_new = torch.maximum(m, m_local)

        # Correction factor for old accumulator
        correction_old = torch.exp(m - m_new)
        # Exp of local scores with new max
        P = torch.exp(S_f32 - m_new)

        l = correction_old * l + P.sum(dim=-1, keepdim=True)
        O = (correction_old * O.float() + torch.matmul(P.to(Q.dtype), Vj).float())

        m = m_new

    O = (O / l).to(Q.dtype)
    return O


# ---------------------------------------------------------------------------
# 3. Flash Attention + Local Max Oracle
#    각 블록의 local max를 미리 알고 있어서 initial m을 정확하게 세팅.
#    하지만 블록 간 correction은 여전히 필요.
# ---------------------------------------------------------------------------

def precompute_local_maxes(Q, K, scale, block_size):
    """Target model로 미리 계산해둔 블록별 local max (oracle)."""
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    n_blocks = math.ceil(Lk / block_size)
    local_maxes = []
    for j in range(n_blocks):
        start = j * block_size
        end = min(start + block_size, Lk)
        Kj = K[:, :, start:end, :]
        S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale
        local_maxes.append(S.float().max(dim=-1, keepdim=True).values)
    return local_maxes


def flash_attention_local_oracle(Q, K, V, scale, block_size, local_maxes):
    """
    Flash attention where each block's local max is known in advance.
    Skips per-block max computation, but still needs inter-block correction.
    """
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    n_blocks = math.ceil(Lk / block_size)

    O = torch.zeros(B, H, Lq, D, dtype=Q.dtype, device=Q.device)
    m = torch.full((B, H, Lq, 1), float('-inf'), dtype=torch.float32, device=Q.device)
    l = torch.zeros((B, H, Lq, 1), dtype=torch.float32, device=Q.device)

    for j in range(n_blocks):
        start = j * block_size
        end = min(start + block_size, Lk)

        Kj = K[:, :, start:end, :]
        Vj = V[:, :, start:end, :]

        S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale
        S_f32 = S.float()

        # Use precomputed local max instead of computing it
        m_local = local_maxes[j]
        m_new = torch.maximum(m, m_local)

        correction_old = torch.exp(m - m_new)
        P = torch.exp(S_f32 - m_new)

        l = correction_old * l + P.sum(dim=-1, keepdim=True)
        O = (correction_old * O.float() + torch.matmul(P.to(Q.dtype), Vj).float())

        m = m_new

    O = (O / l).to(Q.dtype)
    return O


# ---------------------------------------------------------------------------
# 4. Flash Attention + Global Max Oracle
#    Row 전체의 global max를 미리 알고 있어서 correction 완전 제거.
# ---------------------------------------------------------------------------

def precompute_global_max(Q, K, scale, block_size):
    """Target model로 미리 계산해둔 row-wise global max (oracle)."""
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    n_blocks = math.ceil(Lk / block_size)
    global_max = torch.full((B, H, Lq, 1), float('-inf'), dtype=torch.float32, device=Q.device)
    for j in range(n_blocks):
        start = j * block_size
        end = min(start + block_size, Lk)
        Kj = K[:, :, start:end, :]
        S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale
        local_m = S.float().max(dim=-1, keepdim=True).values
        global_max = torch.maximum(global_max, local_m)
    return global_max


def flash_attention_global_oracle(Q, K, V, scale, block_size, global_max):
    """
    Flash attention where the global row-wise max is known.
    No inter-block correction needed — single-pass exp + accumulate.
    """
    B, H, Lq, D = Q.shape
    Lk = K.shape[2]
    n_blocks = math.ceil(Lk / block_size)

    O = torch.zeros(B, H, Lq, D, dtype=Q.dtype, device=Q.device)
    l = torch.zeros((B, H, Lq, 1), dtype=torch.float32, device=Q.device)

    for j in range(n_blocks):
        start = j * block_size
        end = min(start + block_size, Lk)

        Kj = K[:, :, start:end, :]
        Vj = V[:, :, start:end, :]

        S = torch.matmul(Q, Kj.transpose(-2, -1)) * scale
        S_f32 = S.float()

        # No correction needed — global max is fixed
        P = torch.exp(S_f32 - global_max)

        l += P.sum(dim=-1, keepdim=True)
        O = O.float() + torch.matmul(P.to(Q.dtype), Vj).float()

    O = (O / l).to(Q.dtype)
    return O


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(cfg: BenchConfig):
    """Verify all implementations produce the same output."""
    torch.manual_seed(42)
    B, H, D = cfg.batch_size, cfg.n_heads, cfg.head_dim
    Lq, Lk = cfg.draft_len, cfg.seq_len
    device = cfg.device
    dtype = cfg.torch_dtype
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    ref = naive_attention(Q, K, V, scale)
    out_flash = flash_attention(Q, K, V, scale, cfg.block_size)
    local_maxes = precompute_local_maxes(Q, K, scale, cfg.block_size)
    out_local = flash_attention_local_oracle(Q, K, V, scale, cfg.block_size, local_maxes)
    global_max = precompute_global_max(Q, K, scale, cfg.block_size)
    out_global = flash_attention_global_oracle(Q, K, V, scale, cfg.block_size, global_max)

    def max_diff(a, b, name):
        d = (a.float() - b.float()).abs().max().item()
        status = "PASS" if d < 1e-2 else "FAIL"
        print(f"  {name:30s} max_diff={d:.6e}  [{status}]")
        return d < 1e-2

    print("Correctness check:")
    all_ok = True
    all_ok &= max_diff(ref, out_flash, "flash vs naive")
    all_ok &= max_diff(ref, out_local, "flash_local_oracle vs naive")
    all_ok &= max_diff(ref, out_global, "flash_global_oracle vs naive")
    return all_ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_fn(fn, args, warmup, iters, device):
    """Benchmark a function with proper synchronization."""
    for _ in range(warmup):
        fn(*args)

    if device == "cpu":
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args)
            times.append((time.perf_counter() - t0) * 1000)
    else:
        torch.cuda.synchronize(device)
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            start_events[i].record()
            fn(*args)
            end_events[i].record()
        torch.cuda.synchronize(device)
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": sorted(times)[len(times) // 2],
    }


def run_benchmark(cfg: BenchConfig):
    torch.manual_seed(42)
    B, H, D = cfg.batch_size, cfg.n_heads, cfg.head_dim
    Lq, Lk = cfg.draft_len, cfg.seq_len
    device = cfg.device
    dtype = cfg.torch_dtype
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    # Precompute oracles (target model에서 미리 구한 정답 max)
    local_maxes = precompute_local_maxes(Q, K, scale, cfg.block_size)
    global_max = precompute_global_max(Q, K, scale, cfg.block_size)

    results = {}

    # 1. Naive
    print("  Benchmarking naive_attention ...")
    results["naive_attention"] = bench_fn(
        naive_attention, (Q, K, V, scale),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    # 2. Flash
    print("  Benchmarking flash_attention ...")
    results["flash_attention"] = bench_fn(
        flash_attention, (Q, K, V, scale, cfg.block_size),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    # 3. Flash + local oracle
    print("  Benchmarking flash_local_oracle ...")
    results["flash_local_oracle"] = bench_fn(
        flash_attention_local_oracle, (Q, K, V, scale, cfg.block_size, local_maxes),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    # 4. Flash + global oracle
    print("  Benchmarking flash_global_oracle ...")
    results["flash_global_oracle"] = bench_fn(
        flash_attention_global_oracle, (Q, K, V, scale, cfg.block_size, global_max),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    # Also measure oracle precompute cost
    print("  Benchmarking oracle precompute (local) ...")
    results["precompute_local_maxes"] = bench_fn(
        precompute_local_maxes, (Q, K, scale, cfg.block_size),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    print("  Benchmarking oracle precompute (global) ...")
    results["precompute_global_max"] = bench_fn(
        precompute_global_max, (Q, K, scale, cfg.block_size),
        cfg.warmup_iters, cfg.bench_iters, device
    )

    return results


# ---------------------------------------------------------------------------
# Sweep over sequence lengths
# ---------------------------------------------------------------------------

def run_sweep(cfg: BenchConfig, seq_lens: list[int]):
    all_results = {}
    for sl in seq_lens:
        cfg.seq_len = sl
        print(f"\n--- seq_len={sl}, draft_len={cfg.draft_len}, block_size={cfg.block_size} ---")
        all_results[sl] = run_benchmark(cfg)
    return all_results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_results(all_results):
    print("\n" + "=" * 90)
    print(f"{'seq_len':>8} | {'naive':>10} | {'flash':>10} | {'local_orc':>10} | {'global_orc':>10} | {'pre_local':>10} | {'pre_global':>10}")
    print("-" * 90)
    for sl, res in sorted(all_results.items()):
        print(f"{sl:>8} | "
              f"{res['naive_attention']['median_ms']:>9.3f}ms | "
              f"{res['flash_attention']['median_ms']:>9.3f}ms | "
              f"{res['flash_local_oracle']['median_ms']:>9.3f}ms | "
              f"{res['flash_global_oracle']['median_ms']:>9.3f}ms | "
              f"{res['precompute_local_maxes']['median_ms']:>9.3f}ms | "
              f"{res['precompute_global_max']['median_ms']:>9.3f}ms")
    print("=" * 90)

    # Speedup analysis
    print(f"\n{'seq_len':>8} | {'flash/naive':>12} | {'local_orc/flash':>16} | {'global_orc/flash':>17} | {'net_global_orc':>15}")
    print("-" * 80)
    for sl, res in sorted(all_results.items()):
        naive_t = res['naive_attention']['median_ms']
        flash_t = res['flash_attention']['median_ms']
        local_t = res['flash_local_oracle']['median_ms']
        global_t = res['flash_global_oracle']['median_ms']
        pre_global_t = res['precompute_global_max']['median_ms']

        # Net = global oracle attention only (precompute는 별도 pass에서 수행)
        net_global = global_t

        print(f"{sl:>8} | "
              f"{flash_t / naive_t:>11.3f}x | "
              f"{local_t / flash_t:>15.3f}x | "
              f"{global_t / flash_t:>16.3f}x | "
              f"{net_global / flash_t:>14.3f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flash Attention Max Oracle Benchmark")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--draft_len", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        draft_len=args.draft_len,
        block_size=args.block_size,
        dtype=args.dtype,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    print(f"Config: B={cfg.batch_size}, H={cfg.n_heads}, D={cfg.head_dim}, "
          f"draft_len={cfg.draft_len}, block_size={cfg.block_size}, dtype={cfg.dtype}")

    if not torch.cuda.is_available():
        print("CUDA not available, switching to CPU (timings will not be accurate)")
        cfg.device = "cpu"

    # Correctness
    ok = check_correctness(cfg)
    if not ok:
        print("CORRECTNESS CHECK FAILED — aborting benchmark")
        return

    # Benchmark sweep
    all_results = run_sweep(cfg, args.seq_lens)
    print_results(all_results)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"config": asdict(cfg), "results": all_results}, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
