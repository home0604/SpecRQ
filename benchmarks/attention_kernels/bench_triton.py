"""
Triton Flash Attention: Global Max Oracle Benchmark
====================================================
global max exponent를 미리 알고 있을 때의 성능 이득을 측정.

두 가지 모드:
  - prefill : Lq == Lk (원본 flash attention 구조 그대로)
  - decode  : Lq << Lk (Lq=1 or draft_len, Lk=context length)

비교 설정:
  1. naive_attention      : torch matmul 기반 표준 attention
  2. triton_flash         : Triton flash attention (online softmax)
  3. triton_global_oracle : global max oracle로 correction 제거
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import argparse
import json


# ===================================================================
# PREFILL KERNELS (Lq == Lk, 원본 triton.ops 구조)
# ===================================================================

@triton.jit
def _fwd_kernel_prefill(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    Z_H_N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm

    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, vk_offset),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(vk_offset, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)
    q = (q * qk_scale).to(K.dtype.element_ty)

    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    O_block_ptr = tl.make_block_ptr(
        base=Out, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(vk_offset + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


@triton.jit
def _fwd_kernel_prefill_oracle(
    Q, K, V, sm_scale,
    GlobalMax, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    Z_H_N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm

    K_block_ptr = tl.make_block_ptr(
        base=K, shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, vk_offset),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(vk_offset, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    gm_ptrs = GlobalMax + off_hz * N_CTX + offs_m
    m_global = tl.load(gm_ptrs)

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)
    q = (q * qk_scale).to(K.dtype.element_ty)

    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k)
        p = tl.math.exp2(qk - m_global[:, None])
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i += tl.sum(p, 1)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_global + tl.math.log2(l_i))
    O_block_ptr = tl.make_block_ptr(
        base=Out, shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(vk_offset + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


# ===================================================================
# DECODE KERNELS (Lq != Lk)
# Q, K, V offset을 별도로 계산 (stride가 다르므로)
# ===================================================================

@triton.jit
def _fwd_kernel_decode(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX_Q,   # query length (small: 1 or draft_len)
    N_CTX_K,   # key/value length (context)
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Separate offsets for Q and K/V
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs, mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX_K, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    O_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(O_ptrs, acc.to(K.dtype.element_ty), mask=q_mask)


@triton.jit
def _fwd_kernel_decode_oracle(
    Q, K, V, sm_scale,
    GlobalMax,  # (Z*H, N_CTX_Q) in log2 scale
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX_Q,
    N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    Q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs, mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    # Load precomputed global max
    gm_ptrs = GlobalMax + off_hz * N_CTX_Q + offs_m
    m_global = tl.load(gm_ptrs, mask=offs_m < N_CTX_Q, other=0.0)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX_K, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.dot(q, k)
        p = tl.math.exp2(qk - m_global[:, None])
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i += tl.sum(p, 1)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX_Q + offs_m
    tl.store(l_ptrs, m_global + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    O_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    tl.store(O_ptrs, acc.to(K.dtype.element_ty), mask=q_mask)


# ===================================================================
# Python wrappers
# ===================================================================

BLOCK_M = 128
BLOCK_N = 64


def _launch_prefill(q, k, v, sm_scale, causal, global_max=None):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    L = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    num_warps = 4 if D <= 64 else 8
    if global_max is None:
        _fwd_kernel_prefill[grid](
            q, k, v, sm_scale, L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N, B * H * N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            IS_CAUSAL=causal,
            num_warps=num_warps, num_stages=4,
        )
    else:
        _fwd_kernel_prefill_oracle[grid](
            q, k, v, sm_scale, global_max, L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, N, B * H * N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            IS_CAUSAL=causal,
            num_warps=num_warps, num_stages=4,
        )
    return o, L


def _launch_decode(q, k, v, sm_scale, global_max=None):
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    o = torch.empty_like(q)
    L = torch.empty((B * H, Lq), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Lq, BLOCK_M), B * H)
    num_warps = 4 if D <= 64 else 8
    if global_max is None:
        _fwd_kernel_decode[grid](
            q, k, v, sm_scale, L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, Lq, Lk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            num_warps=num_warps, num_stages=4,
        )
    else:
        _fwd_kernel_decode_oracle[grid](
            q, k, v, sm_scale, global_max, L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, Lq, Lk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            num_warps=num_warps, num_stages=4,
        )
    return o, L


def precompute_global_max(q, k, sm_scale):
    """정답 global max 계산 (log2 scale)."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    qk_scale = sm_scale * 1.44269504
    global_max_list = []
    for b in range(B):
        for h in range(H):
            s = torch.matmul(q[b, h].float(), k[b, h].float().T) * qk_scale
            global_max_list.append(s.max(dim=-1).values)
    return torch.stack(global_max_list).contiguous()  # (B*H, Lq)


def naive_attention(Q, K, V, sm_scale):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


# ===================================================================
# Correctness
# ===================================================================

def check_correctness(B, H, Lq, Lk, D, device, dtype, mode):
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    ref = naive_attention(Q, K, V, sm_scale)

    if mode == "prefill":
        out_flash, _ = _launch_prefill(Q, K, V, sm_scale, causal=False)
        gm = precompute_global_max(Q, K, sm_scale)
        out_oracle, _ = _launch_prefill(Q, K, V, sm_scale, causal=False, global_max=gm)
    else:
        out_flash, _ = _launch_decode(Q, K, V, sm_scale)
        gm = precompute_global_max(Q, K, sm_scale)
        out_oracle, _ = _launch_decode(Q, K, V, sm_scale, global_max=gm)

    def check(a, b, name):
        d = (a.float() - b.float()).abs().max().item()
        status = "PASS" if d < 1e-2 else "FAIL"
        print(f"  {name:35s} max_diff={d:.6e}  [{status}]")
        return d < 1e-2

    print(f"Correctness ({mode}, Lq={Lq}, Lk={Lk}):")
    ok = True
    ok &= check(ref, out_flash, "triton_flash vs naive")
    ok &= check(ref, out_oracle, "triton_global_oracle vs naive")
    ok &= check(out_flash, out_oracle, "oracle vs flash")
    return ok


# ===================================================================
# Benchmark
# ===================================================================

def bench_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
    }


def run_single(B, H, Lq, Lk, D, device, dtype, warmup, iters, mode):
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    gm = precompute_global_max(Q, K, sm_scale)

    if mode == "prefill":
        flash_fn = lambda: _launch_prefill(Q, K, V, sm_scale, causal=False)
        oracle_fn = lambda: _launch_prefill(Q, K, V, sm_scale, causal=False, global_max=gm)
    else:
        flash_fn = lambda: _launch_decode(Q, K, V, sm_scale)
        oracle_fn = lambda: _launch_decode(Q, K, V, sm_scale, global_max=gm)

    results = {}

    print(f"  naive ...", end="", flush=True)
    results["naive"] = bench_fn(
        lambda: naive_attention(Q, K, V, sm_scale), warmup, iters)
    print(f" {results['naive']['median_ms']:.4f}ms")

    print(f"  triton_flash ...", end="", flush=True)
    results["triton_flash"] = bench_fn(flash_fn, warmup, iters)
    print(f" {results['triton_flash']['median_ms']:.4f}ms")

    print(f"  triton_global_oracle ...", end="", flush=True)
    results["triton_global_oracle"] = bench_fn(oracle_fn, warmup, iters)
    print(f" {results['triton_global_oracle']['median_ms']:.4f}ms")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--draft_lens", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={args.batch_size}, H={args.n_heads}, D={args.head_dim}, dtype={args.dtype}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}\n")

    all_results = {}

    # ---- Prefill (Lq == Lk) ----
    print("=" * 85)
    print("PREFILL (Lq == Lk)")
    print("=" * 85)

    ok = check_correctness(args.batch_size, args.n_heads, args.seq_lens[0], args.seq_lens[0],
                           args.head_dim, device, dtype, "prefill")
    if not ok:
        print("PREFILL CORRECTNESS FAILED")
        return

    for sl in args.seq_lens:
        key = f"prefill_Lq{sl}_Lk{sl}"
        print(f"\n--- prefill: Lq=Lk={sl} ---")
        all_results[key] = run_single(
            args.batch_size, args.n_heads, sl, sl,
            args.head_dim, device, dtype, args.warmup, args.iters, "prefill")

    # ---- Decode (Lq << Lk) ----
    print("\n" + "=" * 85)
    print("DECODE (Lq << Lk)")
    print("=" * 85)

    for dl in args.draft_lens:
        ok = check_correctness(args.batch_size, args.n_heads, dl, args.seq_lens[0],
                               args.head_dim, device, dtype, "decode")
        if not ok:
            print(f"DECODE CORRECTNESS FAILED (Lq={dl})")
            return

        for sl in args.seq_lens:
            key = f"decode_Lq{dl}_Lk{sl}"
            print(f"\n--- decode: Lq={dl}, Lk={sl} ---")
            all_results[key] = run_single(
                args.batch_size, args.n_heads, dl, sl,
                args.head_dim, device, dtype, args.warmup, args.iters, "decode")

    # ---- Summary ----
    print("\n\n" + "=" * 90)
    print(f"{'mode':>10} {'Lq':>5} {'Lk':>6} | {'naive':>10} | {'flash':>10} | {'oracle':>10} | {'speedup':>8} | {'saved':>6}")
    print("-" * 90)
    for key, res in all_results.items():
        parts = key.split("_")
        mode = parts[0]
        lq = int(parts[1][2:])
        lk = int(parts[2][2:])
        n_t = res["naive"]["median_ms"]
        f_t = res["triton_flash"]["median_ms"]
        o_t = res["triton_global_oracle"]["median_ms"]
        speedup = f_t / o_t if o_t > 0 else float('inf')
        saved = (1 - o_t / f_t) * 100
        print(f"{mode:>10} {lq:>5} {lk:>6} | {n_t:>9.4f}ms | {f_t:>9.4f}ms | {o_t:>9.4f}ms | {speedup:>7.3f}x | {saved:>5.1f}%")
    print("=" * 90)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "config": {
                    "batch_size": args.batch_size, "n_heads": args.n_heads,
                    "head_dim": args.head_dim, "dtype": args.dtype,
                    "gpu": torch.cuda.get_device_name(0),
                },
                "results": all_results,
            }, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
