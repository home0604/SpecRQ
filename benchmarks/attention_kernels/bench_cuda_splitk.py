"""
Single-kernel Split-K Flash Decoding: measure __syncthreads__ overhead
======================================================================
FlashDecoding++ claims ~18.8% overhead from synchronized partial softmax
update. Our Triton 2-kernel approach can't measure this because the sync
cost is hidden in kernel launch + global memory.

This benchmark uses a CUDA kernel where phase1 + reduction happen in ONE
CTA via shared memory + __syncthreads__, directly measuring the sync cost.

Two variants:
  1. Standard: online softmax per split → __syncthreads__ → reduction WITH correction
  2. Oracle:   global max per split    → __syncthreads__ → reduction WITHOUT correction
  Difference = partial softmax sync overhead (the 18.8% claim)
"""

import torch
import torch.nn.functional as F
import math
import argparse
import json

import splitk_kernels as sk

# ── Benchmark ─────────────────────────────────────────────────────────────────

compute_global_max = sk.compute_global_max


def naive_attention(Q, K, V, sm_scale):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def main():
    from triton.testing import do_bench

    parser = argparse.ArgumentParser(
        description="Single-kernel split-K: measure __syncthreads__ overhead (18.8%% claim)")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[64, 1024, 2048, 4096, 32768, 65536])
    parser.add_argument("--num_splits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--rep", type=int, default=2000)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.float16
    H, D = args.n_heads, args.head_dim
    Lq = 16

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Single-kernel split-K decode: H={H}, D={D}, Lq={Lq}")
    print(f"Batch sizes: {args.batch_size}, Cache lengths: {args.seq_lens}")
    print(f"Splits: {args.num_splits}")
    print()

    # Kernels loaded from splitk_kernels package (pip install -e flash/splitk_kernels)

    # ── Correctness check ──
    B_chk = 1
    Lk_chk = args.seq_lens[0] if args.seq_lens[0] >= 64 else 64
    ns_chk = args.num_splits[0]
    sm = 1.0 / math.sqrt(D)

    Q_chk = torch.randn(B_chk, H, 1, D, device=device, dtype=dtype)
    K_chk = torch.randn(B_chk, H, Lk_chk, D, device=device, dtype=dtype)
    V_chk = torch.randn(B_chk, H, Lk_chk, D, device=device, dtype=dtype)
    gm_chk = compute_global_max(Q_chk, K_chk, sm)

    ref = naive_attention(Q_chk, K_chk, V_chk, sm)
    out_std = sk.launch_standard(Q_chk, K_chk, V_chk, sm, ns_chk)
    out_orc = sk.launch_oracle(Q_chk, K_chk, V_chk, gm_chk.view(-1), sm, ns_chk)

    def chk(a, b, name, tol=5e-2):
        d = (a.float() - b.float()).abs().max().item()
        status = "PASS" if d < tol else "FAIL"
        print(f"  {name:40s} max_diff={d:.6e}  [{status}]")
        return d < tol

    # Local-max async (2-kernel)
    out_lma = sk.launch_localmax_async(Q_chk, K_chk, V_chk, sm, ns_chk)

    print("Correctness:")
    ok = True
    ok &= chk(ref, out_std, f"standard (splits={ns_chk}) vs naive")
    ok &= chk(ref, out_orc, f"oracle (splits={ns_chk}) vs naive")
    ok &= chk(ref, out_lma, f"localmax_async (splits={ns_chk}) vs naive")
    if not ok:
        print("CORRECTNESS FAILED — check kernel!")
    print()

    # ── Occupancy analysis ──
    print("Occupancy analysis (theoretical):")
    print(f"{'Kernel':>20s} {'splits':>6s} {'thr/blk':>8s} {'smem':>8s} │ "
          f"{'blk/SM':>6s} {'occ%':>6s} {'grid':>6s} {'active':>8s} {'SM_util':>8s}")
    print("-" * 95)
    for ns in args.num_splits:
        threads_std = ns * 32
        smem_std = (ns * D + ns * 2) * 4
        grid_std = args.batch_size[0] * H

        threads_lma = 32
        grid_lma = args.batch_size[0] * H * ns

        for kname, thr, smem, grid in [
            ("standard", threads_std, smem_std, grid_std),
            ("oracle", threads_std, smem_std, grid_std),
            ("localmax", threads_lma, 0, grid_lma),
        ]:
            try:
                occ = sk.query_occupancy(kname, thr, smem, grid)
                blk_sm, max_thr_sm, num_sms, occ_pct, total_active = occ
                sm_util = total_active / (blk_sm * num_sms) * 100 if blk_sm > 0 else 0
                print(f"{kname:>20s} {ns:>6d} {thr:>8d} {smem:>7d}B │ "
                      f"{int(blk_sm):>6d} {occ_pct:>5.1f}% {grid:>6d} {int(total_active):>8d} {sm_util:>7.1f}%")
            except Exception as e:
                print(f"{kname:>20s} {ns:>6d} — error: {e}")
    print()

    # ── Benchmark ──
    all_results = {}

    for B in args.batch_size:
        print(f"{'=' * 90}")
        print(f"Batch size = {B}")
        print(f"{'=' * 90}")

        hdr = "%6s %6s | %9s %9s %7s | %9s %7s | %9s" % (
            "splits", "Lk",
            "std(ms)", "oracle", "saved%",
            "lm_async", "lma_sv%",
            "sdpa(ms)")
        print(hdr)
        print("-" * len(hdr))

        for Lk in args.seq_lens:
            sm_scale = 1.0 / math.sqrt(D)
            torch.manual_seed(42)
            Q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
            K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
            V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
            gm = compute_global_max(Q, K, sm_scale)
            gm_flat = gm.view(-1)

            sdpa_ms = do_bench(
                lambda: F.scaled_dot_product_attention(Q, K, V, scale=sm_scale),
                warmup=args.warmup, rep=args.rep)

            for ns in args.num_splits:
                # Scalar versions
                std_ms = do_bench(
                    lambda: sk.launch_standard(Q, K, V, sm_scale, ns),
                    warmup=args.warmup, rep=args.rep)
                orc_ms = do_bench(
                    lambda: sk.launch_oracle(Q, K, V, gm_flat, sm_scale, ns),
                    warmup=args.warmup, rep=args.rep)
                saved = (1 - orc_ms / std_ms) * 100 if std_ms > 0 else 0

                # Local-max async (2-kernel)
                lma_ms = do_bench(
                    lambda: sk.launch_localmax_async(Q, K, V, sm_scale, ns),
                    warmup=args.warmup, rep=args.rep)
                lma_saved = (1 - lma_ms / std_ms) * 100 if std_ms > 0 else 0

                print("%6d %6d | %8.4fms %8.4fms %6.1f%% | %8.4fms %6.1f%% | %8.4fms" %
                      (ns, Lk,
                       std_ms, orc_ms, saved,
                       lma_ms, lma_saved,
                       sdpa_ms))

                all_results[f"B{B}_s{ns}_Lk{Lk}"] = {
                    "batch_size": B, "num_splits": ns, "cache_len": Lk,
                    "standard_ms": std_ms, "oracle_ms": orc_ms, "saved_pct": saved,
                    "localmax_async_ms": lma_ms, "lma_saved_pct": lma_saved,
                    "sdpa_ms": sdpa_ms,
                }
            print()

    # ── Summary ──
    print(f"\n{'=' * 90}")
    print("Summary: Speedup of oracle over standard = __syncthreads__ + correction overhead")
    print(f"{'=' * 90}")

    import functools, operator
    def geo_mean(vals):
        if not vals:
            return 0
        return functools.reduce(operator.mul, vals, 1.0) ** (1.0 / len(vals))

    saved_vals = [r["saved_pct"] for r in all_results.values()]
    avg_saved = sum(saved_vals) / len(saved_vals) if saved_vals else 0
    print(f"  Average saved%: {avg_saved:.1f}%")
    print(f"  FlashDecoding++ claim: ~18.8% (Llama-2-7B, A100, 1024 input)")

    # Per Lk breakdown
    for Lk in args.seq_lens:
        lk_vals = [r["saved_pct"] for r in all_results.values() if r["cache_len"] == Lk]
        if lk_vals:
            print(f"  Lk={Lk:>6d}: avg saved={sum(lk_vals)/len(lk_vals):.1f}%")

    # ── Time breakdown via clock64() profiling (보류) ──
    # clock64()는 per-warp timeline만 측정하여 cross-warp latency hiding 효과를 반영하지 못함.
    # effective bandwidth 비교나 ncu profiling이 더 적합.
    # 코드: launch_profiled(), launch_profiled_oracle() 사용 가능.
    pass

    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "config": {"H": H, "D": D, "dtype": "float16",
                           "gpu": torch.cuda.get_device_name(0)},
                "results": all_results,
            }, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
