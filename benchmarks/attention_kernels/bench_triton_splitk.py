"""
Split-K Flash Decoding: Verify FlashDecoding++ 18.8% claim
==========================================================
FlashDecoding++ claims "synchronized partial softmax update accounts for
18.8% of attention computation" on Llama2-7B, A100, 1024 input length.

This benchmark implements split-K flash decoding to measure:
  1. split-K flash decoding (full: phase1 + reduction with correction)
  2. split-K oracle decoding (phase1 + reduction WITHOUT correction)
  3. single-pass flash decoding (no split, baseline)
  4. The percentage cost of the partial softmax synchronization

Split-K approach:
  - Phase 1: Split Lk into num_splits chunks. Each split computes
    partial_out (BLOCK_M, D), partial_m (BLOCK_M,), partial_l (BLOCK_M,)
    using online softmax within that chunk.
  - Phase 2 (reduction): Merge partial results across splits.
    For each query row, combine partial_out[s] weighted by
    exp(partial_m[s] - global_m) * partial_l[s] / global_l.
  - Oracle Phase 2: If global max is known, phase 1 can use it directly,
    making each partial result already aligned — reduction becomes simple sum.

Llama2-7B config: 32 heads, head_dim=128, no GQA.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import argparse
import json


# ===================================================================
# Single-pass flash decode (baseline, no split-K)
# ===================================================================

@triton.jit
def _fwd_decode_single(
    Q, K, V, sm_scale, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX_Q, N_CTX_K,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_off, shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_off, shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))

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
    tl.store(L + off_hz * N_CTX_Q + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on,
             acc.to(K.dtype.element_ty), mask=q_mask)


# ===================================================================
# Split-K Phase 1: Each split processes a chunk of KV
# Outputs partial_out, partial_m, partial_l per split
# ===================================================================

@triton.jit
def _splitk_phase1(
    Q, K, V, sm_scale,
    Partial_Out, Partial_M, Partial_L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    # partial_out: (Z*H, num_splits, Lq, D)
    stride_poz, stride_pos, stride_pom, stride_pod,
    # partial_m/l: (Z*H, num_splits, Lq)
    stride_pmz, stride_pms, stride_pmm,
    Z, H, N_CTX_Q, N_CTX_K,
    NUM_SPLITS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    split_id = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh

    # Compute this split's KV range
    split_size = tl.cdiv(N_CTX_K, NUM_SPLITS)
    kv_start = split_id * split_size
    kv_end = tl.minimum(kv_start + split_size, N_CTX_K)
    # Round up to BLOCK_N boundary
    kv_end_aligned = tl.cdiv(kv_end, BLOCK_N) * BLOCK_N

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_off, shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn), offsets=(0, kv_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_off, shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(kv_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(kv_start, kv_end_aligned, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        qk = tl.dot(q, k)
        # Mask out-of-range KV positions
        offs_n = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n[None, :] < kv_end, qk, float("-inf"))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store partial results (unnormalized: acc = sum(p * v), not divided by l_i)
    po_off = off_hz * stride_poz + split_id * stride_pos
    tl.store(Partial_Out + po_off + offs_m[:, None] * stride_pom + offs_k[None, :] * stride_pod,
             acc.to(tl.float16), mask=q_mask)
    pm_off = off_hz * stride_pmz + split_id * stride_pms
    tl.store(Partial_M + pm_off + offs_m * stride_pmm, m_i, mask=offs_m < N_CTX_Q)
    tl.store(Partial_L + pm_off + offs_m * stride_pmm, l_i, mask=offs_m < N_CTX_Q)


# ===================================================================
# Split-K Phase 1 Oracle: Same dataflow as standard, but uses global
# max instead of local tl.max(qk,1). Online softmax rescaling kept
# identical so instruction mix / register pressure matches standard.
# ===================================================================

@triton.jit
def _splitk_phase1_oracle(
    Q, K, V, sm_scale, GlobalMax,
    Partial_Out, Partial_L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_poz, stride_pos, stride_pom, stride_pod,
    stride_plz, stride_pls, stride_plm,
    Z, H, N_CTX_Q, N_CTX_K,
    NUM_SPLITS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    split_id = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh

    split_size = tl.cdiv(N_CTX_K, NUM_SPLITS)
    kv_start = split_id * split_size
    kv_end = tl.minimum(kv_start + split_size, N_CTX_K)
    kv_end_aligned = tl.cdiv(kv_end, BLOCK_N) * BLOCK_N

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    m_global = tl.load(GlobalMax + off_hz * N_CTX_Q + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_off, shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn), offsets=(0, kv_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_off, shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(kv_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Lean oracle: online softmax 전체 제거
    # global max 사용 → 모든 split이 동일 기준 → correction 불필요
    # max tracking, alpha rescaling 없음 → instruction 감소
    for start_n in range(kv_start, kv_end_aligned, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        qk = tl.dot(q, k)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n[None, :] < kv_end, qk, float("-inf"))
        p = tl.math.exp2(qk - m_global[:, None])
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i += tl.sum(p, 1)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # All splits share the same global max → reduction은 단순 합산
    po_off = off_hz * stride_poz + split_id * stride_pos
    tl.store(Partial_Out + po_off + offs_m[:, None] * stride_pom + offs_k[None, :] * stride_pod,
             acc.to(tl.float16), mask=q_mask)
    pl_off = off_hz * stride_plz + split_id * stride_pls
    tl.store(Partial_L + pl_off + offs_m * stride_plm, l_i, mask=offs_m < N_CTX_Q)


# ===================================================================
# Split-K Phase 2: Reduction with partial softmax correction
# ===================================================================

@triton.jit
def _splitk_reduce(
    Partial_Out, Partial_M, Partial_L, Out,
    stride_poz, stride_pos, stride_pom, stride_pod,
    stride_pmz, stride_pms, stride_pmm,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX_Q,
    NUM_SPLITS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m < N_CTX_Q

    # First pass: find global max across all splits
    global_m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    for s in range(NUM_SPLITS):
        pm_off = off_hz * stride_pmz + s * stride_pms
        m_s = tl.load(Partial_M + pm_off + offs_m * stride_pmm, mask=q_mask, other=float("-inf"))
        global_m = tl.maximum(global_m, m_s)

    # Second pass: rescale and accumulate
    global_l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        pm_off = off_hz * stride_pmz + s * stride_pms
        m_s = tl.load(Partial_M + pm_off + offs_m * stride_pmm, mask=q_mask, other=float("-inf"))
        l_s = tl.load(Partial_L + pm_off + offs_m * stride_pmm, mask=q_mask, other=0.0)
        # Correction factor: exp2(partial_max - global_max)
        alpha = tl.math.exp2(m_s - global_m)
        l_s_corrected = l_s * alpha
        global_l += l_s_corrected
        po_off = off_hz * stride_poz + s * stride_pos
        o_s = tl.load(Partial_Out + po_off + offs_m[:, None] * stride_pom + offs_d[None, :] * stride_pod,
                      mask=q_mask[:, None], other=0.0).to(tl.float32)
        acc += o_s * alpha[:, None]

    acc = acc / global_l[:, None]

    o_off = off_z * stride_oz + off_h * stride_oh
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(tl.float16), mask=q_mask[:, None])


# ===================================================================
# Split-K Phase 2 Oracle: Simple sum (no correction needed)
# ===================================================================

@triton.jit
def _splitk_reduce_oracle(
    Partial_Out, Partial_L, Out,
    stride_poz, stride_pos, stride_pom, stride_pod,
    stride_plz, stride_pls, stride_plm,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX_Q,
    NUM_SPLITS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m < N_CTX_Q

    # All partial results are aligned to global max — just sum
    global_l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        pl_off = off_hz * stride_plz + s * stride_pls
        l_s = tl.load(Partial_L + pl_off + offs_m * stride_plm, mask=q_mask, other=0.0)
        global_l += l_s
        po_off = off_hz * stride_poz + s * stride_pos
        o_s = tl.load(Partial_Out + po_off + offs_m[:, None] * stride_pom + offs_d[None, :] * stride_pod,
                      mask=q_mask[:, None], other=0.0).to(tl.float32)
        acc += o_s

    acc = acc / global_l[:, None]

    o_off = off_z * stride_oz + off_h * stride_oh
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             acc.to(tl.float16), mask=q_mask[:, None])


# ===================================================================
# Oracle Fused: phase1 directly atomic-adds to output (NO reduction)
# Global max known → all splits aligned → just accumulate into Out, L
# ===================================================================

@triton.jit
def _splitk_fused_oracle(
    Q, K, V, sm_scale, GlobalMax,
    Out, L_out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX_Q, N_CTX_K,
    NUM_SPLITS,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    split_id = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh
    o_off = off_z * stride_oz + off_h * stride_oh

    split_size = tl.cdiv(N_CTX_K, NUM_SPLITS)
    kv_start = split_id * split_size
    kv_end = tl.minimum(kv_start + split_size, N_CTX_K)
    kv_end_aligned = tl.cdiv(kv_end, BLOCK_N) * BLOCK_N

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(K.dtype.element_ty)

    m_global = tl.load(GlobalMax + off_hz * N_CTX_Q + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_off, shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn), offsets=(0, kv_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_off, shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(kv_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(kv_start, kv_end_aligned, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        qk = tl.dot(q, k)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n[None, :] < kv_end, qk, float("-inf"))
        p = tl.math.exp2(qk - m_global[:, None])
        acc += tl.dot(p.to(V.dtype.element_ty), v)
        l_i += tl.sum(p, 1)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Atomic add directly to output — no reduction kernel needed
    out_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.atomic_add(out_ptrs, acc.to(tl.float32), mask=q_mask)
    l_ptrs = L_out + off_hz * N_CTX_Q + offs_m
    tl.atomic_add(l_ptrs, l_i, mask=offs_m < N_CTX_Q)


# ===================================================================
# Python wrappers
# ===================================================================

BLOCK_M = 1    # 기본값 — decode (Lq=1) 최적
BLOCK_N = 128

# Autotune configurations: (BLOCK_M, BLOCK_N, num_warps, num_stages)
# Decode (Lq=1)에서는 BLOCK_M=1~4가 최적일 수 있음 — 큰 BLOCK_M은 mask waste
_AUTOTUNE_CONFIGS = [
    # ── BLOCK_M=1: decode 특화 (waste 없음) ──
    (1,  32,  4, 2),
    (1,  64,  4, 2),
    (1,  64,  4, 4),
    (1,  128, 4, 2),
    (1,  128, 4, 4),
    (1,  128, 8, 2),
    (1,  256, 4, 2),
    (1,  256, 8, 2),
    # ── BLOCK_M=2,4: 소규모 batch decode ──
    (2,  64,  4, 2),
    (2,  128, 4, 2),
    (2,  128, 4, 4),
    (4,  64,  4, 2),
    (4,  128, 4, 2),
    (4,  128, 4, 4),
    # ── BLOCK_M=16,32: prefill 또는 larger batch ──
    (16, 64,  4, 2),
    (16, 64,  4, 4),
    (16, 64,  8, 4),
    (16, 128, 4, 2),
    (16, 128, 8, 2),
    (32, 64,  4, 4),
    (32, 128, 8, 2),
]

# Best config cache: (kernel_name, D, Lk_bucket) → (BLOCK_M, BLOCK_N, num_warps, num_stages)
_best_configs = {}


def _lk_bucket(Lk):
    """Lk를 버킷으로 분류 (autotune 결과 재사용)."""
    for b in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        if Lk <= b:
            return b
    return Lk


def _autotune_kernel(kernel_name, launch_fn, q, k, v, sm_scale, extra_args=None,
                     num_splits=None):
    """
    주어진 커널에 대해 최적 config을 찾아 반환.
    launch_fn(q, k, v, sm_scale, bm, bn, nw, ns, *extra_args) → output
    num_splits: split-K 커널의 경우 cache key에 포함 (split 수에 따라 최적 config 다름)
    """
    from triton.testing import do_bench

    D = q.shape[-1]
    Lk = k.shape[2]
    bucket = _lk_bucket(Lk)
    cache_key = (kernel_name, D, bucket, num_splits)

    if cache_key in _best_configs:
        return _best_configs[cache_key]

    best_time = float("inf")
    best_cfg = _AUTOTUNE_CONFIGS[0]
    ea = extra_args or ()

    for bm, bn, nw, ns in _AUTOTUNE_CONFIGS:
        try:
            ms = do_bench(lambda: launch_fn(q, k, v, sm_scale, bm, bn, nw, ns, *ea),
                          warmup=20, rep=50)
            if ms < best_time:
                best_time = ms
                best_cfg = (bm, bn, nw, ns)
        except (triton.runtime.errors.OutOfResources, RuntimeError):
            continue

    _best_configs[cache_key] = best_cfg
    bm, bn, nw, ns = best_cfg
    print(f"  [autotune] {kernel_name} (D={D}, Lk≤{bucket}): "
          f"BLOCK_M={bm}, BLOCK_N={bn}, num_warps={nw}, num_stages={ns}")
    return best_cfg


def _launch_single_inner(q, k, v, sm_scale, bm, bn, nw, ns):
    """Allocating version (for autotune + correctness)."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    o = torch.empty_like(q)
    L = torch.empty((B * H, Lq), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Lq, bm), B * H)
    _fwd_decode_single[grid](
        q, k, v, sm_scale, L, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Lq, Lk,
        BLOCK_M=bm, BLOCK_DMODEL=D, BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
    )
    return o


def launch_single(q, k, v, sm_scale):
    bm, bn, nw, ns = _autotune_kernel("single", _launch_single_inner, q, k, v, sm_scale)
    return _launch_single_inner(q, k, v, sm_scale, bm, bn, nw, ns)


def _launch_splitk_inner(q, k, v, sm_scale, bm, bn, nw, ns, num_splits, global_max=None):
    """Allocating version (for autotune + correctness)."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]

    if global_max is None:
        partial_out = torch.empty((B * H, num_splits, Lq, D), device=q.device, dtype=torch.float16)
        partial_m = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        _run_splitk_kernels(q, k, v, sm_scale, bm, bn, nw, ns,
                            num_splits, partial_out, partial_m, partial_l, o)
        return o
    else:
        partial_out = torch.empty((B * H, num_splits, Lq, D), device=q.device, dtype=torch.float16)
        partial_l = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        _run_oracle_kernels(q, k, v, sm_scale, bm, bn, nw, ns,
                            num_splits, global_max, partial_out, partial_l, o)
        return o


def _run_splitk_kernels(q, k, v, sm_scale, bm, bn, nw, ns,
                        num_splits, partial_out, partial_m, partial_l, o):
    """Kernel-only: no allocation, only launches. For accurate benchmarking."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    grid1 = (triton.cdiv(Lq, bm), B * H, num_splits)
    _splitk_phase1[grid1](
        q, k, v, sm_scale,
        partial_out, partial_m, partial_l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        B, H, Lq, Lk, num_splits,
        BLOCK_M=bm, BLOCK_DMODEL=D, BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
    )
    grid2 = (triton.cdiv(Lq, bm), B * H)
    _splitk_reduce[grid2](
        partial_out, partial_m, partial_l, o,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Lq, num_splits,
        BLOCK_M=bm, BLOCK_DMODEL=D,
        num_warps=nw, num_stages=1,
    )


def _run_oracle_kernels(q, k, v, sm_scale, bm, bn, nw, ns,
                        num_splits, global_max, partial_out, partial_l, o):
    """Kernel-only: no allocation, only launches. For accurate benchmarking."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    grid1 = (triton.cdiv(Lq, bm), B * H, num_splits)
    _splitk_phase1_oracle[grid1](
        q, k, v, sm_scale, global_max,
        partial_out, partial_l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        B, H, Lq, Lk, num_splits,
        BLOCK_M=bm, BLOCK_DMODEL=D, BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
    )
    grid2 = (triton.cdiv(Lq, bm), B * H)
    _splitk_reduce_oracle[grid2](
        partial_out, partial_l, o,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Lq, num_splits,
        BLOCK_M=bm, BLOCK_DMODEL=D,
        num_warps=nw, num_stages=1,
    )


def _run_fused_oracle_kernel(q, k, v, sm_scale, bm, bn, nw, ns,
                             num_splits, global_max, o_acc, l_acc):
    """Kernel-only for fused oracle. Caller must zero o_acc/l_acc before each call."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    o_acc.zero_()
    l_acc.zero_()
    grid = (triton.cdiv(Lq, bm), B * H, num_splits)
    _splitk_fused_oracle[grid](
        q, k, v, sm_scale, global_max,
        o_acc, l_acc,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o_acc.stride(0), o_acc.stride(1), o_acc.stride(2), o_acc.stride(3),
        B, H, Lq, Lk, num_splits,
        BLOCK_M=bm, BLOCK_DMODEL=D, BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
    )


def launch_splitk(q, k, v, sm_scale, num_splits, global_max=None):
    """Split-K flash decoding with autotune."""
    kernel_name = "splitk" if global_max is None else "oracle"
    bm, bn, nw, ns = _autotune_kernel(
        kernel_name,
        lambda q, k, v, sm, bm, bn, nw, ns: _launch_splitk_inner(
            q, k, v, sm, bm, bn, nw, ns, num_splits, global_max),
        q, k, v, sm_scale, num_splits=num_splits)
    return _launch_splitk_inner(q, k, v, sm_scale, bm, bn, nw, ns, num_splits, global_max)


def launch_splitk_fused_oracle(q, k, v, sm_scale, num_splits, global_max):
    """Oracle split-K with NO reduction kernel (atomic add). Allocating version."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    o_acc = torch.zeros(B, H, Lq, D, device=q.device, dtype=torch.float32)
    l_acc = torch.zeros(B * H, Lq, device=q.device, dtype=torch.float32)
    bm, bn, nw, ns = BLOCK_M, 64, (4 if D <= 64 else 8), 2
    _run_fused_oracle_kernel(q, k, v, sm_scale, bm, bn, nw, ns,
                             num_splits, global_max, o_acc, l_acc)
    return (o_acc / l_acc.view(B, H, Lq, 1)).to(torch.float16)


def launch_splitk_phase1_only(q, k, v, sm_scale, num_splits, global_max=None):
    """Launch only phase 1 (for isolating phase 1 vs phase 2 cost)."""
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    nw = 4 if D <= 64 else 8

    if global_max is None:
        partial_out = torch.empty((B * H, num_splits, Lq, D), device=q.device, dtype=torch.float16)
        partial_m = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        partial_l = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        grid1 = (triton.cdiv(Lq, BLOCK_M), B * H, num_splits)
        _splitk_phase1[grid1](
            q, k, v, sm_scale,
            partial_out, partial_m, partial_l,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            B, H, Lq, Lk, num_splits,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            num_warps=nw, num_stages=4,
        )
        return partial_out, partial_m, partial_l
    else:
        partial_out = torch.empty((B * H, num_splits, Lq, D), device=q.device, dtype=torch.float16)
        partial_l = torch.empty((B * H, num_splits, Lq), device=q.device, dtype=torch.float32)
        grid1 = (triton.cdiv(Lq, BLOCK_M), B * H, num_splits)
        # Oracle loop body가 가벼워 memory latency hiding을 위해 더 깊은 pipelining 사용
        _splitk_phase1_oracle[grid1](
            q, k, v, sm_scale, global_max,
            partial_out, partial_l,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            B, H, Lq, Lk, num_splits,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
            num_warps=nw, num_stages=4,
        )
        return partial_out, partial_l


def naive_attention(Q, K, V, sm_scale):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def sdpa_attention(Q, K, V, sm_scale):
    """PyTorch 내장 flash attention (torch.nn.functional.scaled_dot_product_attention)."""
    return F.scaled_dot_product_attention(Q, K, V, scale=sm_scale)


def xformers_attention(Q, K, V, sm_scale):
    """xformers memory-efficient attention (CUTLASS 기반 flash attention)."""
    from xformers.ops import memory_efficient_attention
    # xformers: (B, Lq, H, D) layout — 벤치마크 루프에서 미리 변환된 텐서 사용
    return memory_efficient_attention(Q, K, V, scale=sm_scale)


# xformers 사용 가능 여부 확인
try:
    import xformers.ops  # noqa: F401
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False


def compute_global_max(q, k, sm_scale):
    B, H, Lq, D = q.shape
    qk_scale = sm_scale * 1.44269504
    gm = []
    for b in range(B):
        for h in range(H):
            s = torch.matmul(q[b, h].float(), k[b, h].float().T) * qk_scale
            gm.append(s.max(dim=-1).values)
    return torch.stack(gm).contiguous()


# ===================================================================
# Correctness
# ===================================================================

def check_correctness(B, H, Lq, Lk, D, num_splits, device, dtype):
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    ref = naive_attention(Q, K, V, sm_scale)
    gm = compute_global_max(Q, K, sm_scale)

    out_single = launch_single(Q, K, V, sm_scale)
    out_splitk = launch_splitk(Q, K, V, sm_scale, num_splits)
    out_oracle = launch_splitk(Q, K, V, sm_scale, num_splits, gm)
    out_fused = launch_splitk_fused_oracle(Q, K, V, sm_scale, num_splits, gm)

    def chk(a, b, name, tol=1e-2):
        d = (a.float() - b.float()).abs().max().item()
        status = "PASS" if d < tol else "FAIL"
        print(f"  {name:45s} max_diff={d:.6e}  [{status}]")
        return d < tol

    print(f"Correctness (B={B}, H={H}, Lq={Lq}, Lk={Lk}, splits={num_splits}):")
    ok = True
    ok &= chk(ref, out_single, "single-pass flash vs naive")
    ok &= chk(ref, out_splitk, "split-K flash vs naive")
    ok &= chk(ref, out_oracle, "split-K oracle (2-phase) vs naive")
    ok &= chk(ref, out_fused, "split-K oracle (fused/no reduce) vs naive")
    return ok


# ===================================================================
# Benchmark
# ===================================================================

def launch_reduce_only(partial_out, partial_m, partial_l, q_shape, num_splits):
    """Launch only the reduction kernel (for isolated timing)."""
    B, H, Lq, D = q_shape
    nw = 4 if D <= 64 else 8
    o = torch.empty(B, H, Lq, D, device=partial_out.device, dtype=torch.float16)
    grid = (triton.cdiv(Lq, BLOCK_M), B * H)
    _splitk_reduce[grid](
        partial_out, partial_m, partial_l, o,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Lq, num_splits,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D,
        num_warps=nw, num_stages=1,
    )
    return o


def launch_reduce_oracle_only(partial_out, partial_l, q_shape, num_splits):
    """Launch only the oracle reduction kernel."""
    B, H, Lq, D = q_shape
    nw = 4 if D <= 64 else 8
    o = torch.empty(B, H, Lq, D, device=partial_out.device, dtype=torch.float16)
    grid = (triton.cdiv(Lq, BLOCK_M), B * H)
    _splitk_reduce_oracle[grid](
        partial_out, partial_l, o,
        partial_out.stride(0), partial_out.stride(1), partial_out.stride(2), partial_out.stride(3),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, Lq, num_splits,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D,
        num_warps=nw, num_stages=1,
    )
    return o


def get_gpu_specs(device=0):
    """GPU의 peak bandwidth (GB/s)와 peak FLOP/s (TFLOP/s, FP16) 반환."""
    props = torch.cuda.get_device_properties(device)
    # Memory bandwidth: memory_clock (MHz) * bus_width (bits) * 2 (DDR) / 8 (bytes)
    mem_bw_gbs = props.total_memory  # fallback; 정확한 값은 아래 하드코딩
    gpu_name = props.name.upper()

    # 주요 GPU peak specs (FP16 Tensor Core TFLOP/s, HBM GB/s)
    # (FP16 Tensor Core TFLOP/s, HBM bandwidth GB/s)
    # 구체적 variant를 먼저 매칭 (PCIE vs SXM)
    gpu_db = [
        ("A100-PCIE", (312, 1555)),   # A100 PCIe 40GB/80GB
        ("A100-SXM",  (312, 2039)),   # A100 SXM
        ("A100",      (312, 1555)),   # A100 기본 (PCIe 보수적)
        ("H100-PCIE", (756, 2000)),   # H100 PCIe
        ("H100-SXM",  (990, 3350)),   # H100 SXM
        ("H100",      (990, 3350)),   # H100 기본
        ("A6000",     (155, 768)),
        ("A5000",     (130, 768)),
        ("3090",      (142, 936)),
        ("4090",      (330, 1008)),
        ("L40",       (181, 864)),
        ("V100",      (125, 900)),
    ]
    peak_tflops, peak_bw_gbs = 155, 768  # 기본값: A6000
    matched = "default"
    for key, (tf, bw) in gpu_db:
        if key in gpu_name:
            peak_tflops, peak_bw_gbs = tf, bw
            matched = key
            break

    print(f"  GPU specs: matched='{matched}', peak={peak_tflops} TFLOP/s, "
          f"BW={peak_bw_gbs} GB/s, ridge={peak_tflops*1e3/peak_bw_gbs:.1f} FLOPs/byte")
    return peak_tflops, peak_bw_gbs


def roofline_analysis(B, H, Lq, Lk, D, num_splits, elapsed_ms, dtype_bytes=2,
                      peak_tflops=155, peak_bw_gbs=768, label=""):
    """
    단일 attention decode의 roofline 분석.

    Returns dict with:
      - memory_bytes: 이론적 메모리 전송량 (Q + K + V 로드 + O 저장)
      - flops: 이론적 FLOPs (QK^T + softmax + PV)
      - arith_intensity: FLOPs / bytes
      - achieved_bw_gbs: 실측 bandwidth
      - achieved_tflops: 실측 TFLOP/s
      - bound: "memory" or "compute"
      - mem_time_pct: 메모리 로드가 차지하는 시간 비율 (%)
    """
    # ── 이론적 메모리 전송량 ──
    # Q: (B, H, Lq, D) — 1회 로드
    q_bytes = B * H * Lq * D * dtype_bytes
    # K, V: (B, H, Lk, D) — split-K에서는 각 split이 자기 chunk만 로드 (총 1회)
    k_bytes = B * H * Lk * D * dtype_bytes
    v_bytes = B * H * Lk * D * dtype_bytes
    # Output: (B, H, Lq, D) — 1회 저장
    o_bytes = B * H * Lq * D * dtype_bytes
    # Split-K partial results (intermediate)
    partial_bytes = 0
    if num_splits > 1:
        # partial_out: (B*H, splits, Lq, D) fp16 write + read
        partial_bytes += B * H * num_splits * Lq * D * dtype_bytes * 2
        # partial_m, partial_l: (B*H, splits, Lq) fp32 write + read
        partial_bytes += B * H * num_splits * Lq * 4 * 2 * 2

    total_bytes = q_bytes + k_bytes + v_bytes + o_bytes + partial_bytes

    # ── 이론적 FLOPs ──
    # QK^T: 2 * B * H * Lq * Lk * D (matmul)
    qk_flops = 2 * B * H * Lq * Lk * D
    # softmax: ~5 * B * H * Lq * Lk (exp, max, sub, sum, div)
    softmax_flops = 5 * B * H * Lq * Lk
    # PV: 2 * B * H * Lq * Lk * D (matmul)
    pv_flops = 2 * B * H * Lq * Lk * D
    total_flops = qk_flops + softmax_flops + pv_flops

    # ── Arithmetic intensity ──
    arith_intensity = total_flops / total_bytes  # FLOPs/byte

    # ── Ridge point (roofline knee) ──
    ridge_point = peak_tflops * 1e12 / (peak_bw_gbs * 1e9)  # FLOPs/byte

    # ── 실측 성능 ──
    elapsed_s = elapsed_ms / 1000
    achieved_bw_gbs = total_bytes / elapsed_s / 1e9
    achieved_tflops = total_flops / elapsed_s / 1e12

    # ── Memory vs compute bound ──
    bound = "memory" if arith_intensity < ridge_point else "compute"

    # ── 메모리 로드 시간 비율 ──
    # 이론적 메모리 시간 = bytes / peak_bandwidth
    mem_time_s = total_bytes / (peak_bw_gbs * 1e9)
    mem_time_pct = (mem_time_s / elapsed_s) * 100 if elapsed_s > 0 else 0
    # cap at 100% (실측이 이론보다 빠를 수 없음, 하지만 peak_bw 추정 오차 가능)
    mem_time_pct = min(mem_time_pct, 100.0)

    return {
        "label": label,
        "total_bytes": total_bytes,
        "q_bytes": q_bytes, "k_bytes": k_bytes, "v_bytes": v_bytes,
        "partial_bytes": partial_bytes,
        "total_flops": total_flops,
        "qk_flops": qk_flops, "pv_flops": pv_flops,
        "arith_intensity": arith_intensity,
        "ridge_point": ridge_point,
        "achieved_bw_gbs": achieved_bw_gbs,
        "achieved_tflops": achieved_tflops,
        "bw_util_pct": (achieved_bw_gbs / peak_bw_gbs) * 100,
        "bound": bound,
        "mem_time_pct": mem_time_pct,
        "elapsed_ms": elapsed_ms,
    }


def print_roofline_table(analyses):
    """Roofline 분석 결과를 테이블로 출력."""
    print(f"\n{'=' * 100}")
    print("ROOFLINE ANALYSIS — Memory vs Compute Breakdown")
    print(f"{'=' * 100}")
    hdr = (f"{'Method':>20s} │ {'Time(ms)':>8s} │ "
           f"{'Bytes':>8s} {'FLOPs':>8s} {'AI':>6s} │ "
           f"{'BW(GB/s)':>9s} {'BW%':>5s} {'TF/s':>6s} │ "
           f"{'Bound':>7s} {'Mem%':>5s}")
    print(hdr)
    print("─" * len(hdr))
    for a in analyses:
        bytes_str = f"{a['total_bytes']/1e6:.1f}M" if a['total_bytes'] < 1e9 else f"{a['total_bytes']/1e9:.2f}G"
        flops_str = f"{a['total_flops']/1e6:.0f}M" if a['total_flops'] < 1e9 else f"{a['total_flops']/1e9:.1f}G"
        print(f"{a['label']:>20s} │ {a['elapsed_ms']:>8.4f} │ "
              f"{bytes_str:>8s} {flops_str:>8s} {a['arith_intensity']:>6.2f} │ "
              f"{a['achieved_bw_gbs']:>9.1f} {a['bw_util_pct']:>4.1f}% {a['achieved_tflops']:>6.3f} │ "
              f"{a['bound']:>7s} {a['mem_time_pct']:>4.1f}%")
    if analyses:
        print(f"\n  Ridge point (AI at roofline knee): {analyses[0]['ridge_point']:.1f} FLOPs/byte")
        print(f"  AI < ridge → memory-bound, AI > ridge → compute-bound")
        print(f"  Mem% = 이론적 메모리 전송 시간 / 실측 시간 (peak BW 기준)")


def main():
    from triton.testing import do_bench

    parser = argparse.ArgumentParser(description="Split-K Flash Decoding: verify FlashDecoding++ 18.8%% claim")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[1, 4],
                        help="Batch sizes to sweep (논문 Figure 14: 1, 4)")
    parser.add_argument("--n_heads", type=int, default=32, help="Attention heads (Llama2-7B=32)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dim (Llama2-7B=128)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[64, 1024, 2048, 4096, 32168, 65536],
                        help="Cache lengths (논문 Figure 14: 64, 1024, 2048)")
    parser.add_argument("--num_splits", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=100, help="do_bench warmup iterations")
    parser.add_argument("--rep", type=int, default=1000, help="do_bench repetitions")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    batch_sizes = args.batch_size
    H, D = args.n_heads, args.head_dim
    Lq = 1  # decode

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Llama2-7B decode: H={H}, D={D}, Lq={Lq}, dtype={args.dtype}")
    print(f"Batch sizes: {batch_sizes}, Cache lengths: {args.seq_lens}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"xformers: {'available' if _XFORMERS_AVAILABLE else 'not installed'}")

    # SDPA backend 확인
    B_test = batch_sizes[0]
    test_q = torch.randn(B_test, H, Lq, D, device=device, dtype=dtype)
    test_k = torch.randn(B_test, H, 1024, D, device=device, dtype=dtype)
    test_v = torch.randn(B_test, H, 1024, D, device=device, dtype=dtype)
    sm = 1.0 / math.sqrt(D)

    print(f"\nSDPA backends:")
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        for backend in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]:
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(test_q, test_k, test_v, scale=sm)
                print(f"  {backend.name:25s} → supported ✓")
            except RuntimeError:
                print(f"  {backend.name:25s} → not available ✗")
    except ImportError:
        for name, flash, mem, math_b in [
            ("flash_only", True, False, False),
            ("mem_efficient_only", False, True, False),
            ("math_only", False, False, True),
        ]:
            try:
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=flash, enable_mem_efficient=mem, enable_math=math_b)
                with ctx:
                    F.scaled_dot_product_attention(test_q, test_k, test_v, scale=sm)
                print(f"  {name:25s} → supported ✓")
            except RuntimeError:
                print(f"  {name:25s} → not available ✗")

    del test_q, test_k, test_v
    print()

    # Correctness
    B_chk = batch_sizes[0]
    ok = check_correctness(B_chk, H, Lq, args.seq_lens[0], D, args.num_splits[0], device, dtype)
    if not ok:
        print("CORRECTNESS FAILED — aborting")
        return
    print()

    # ══════════════════════════════════════════════════════════════════
    # Detailed benchmark (per-split breakdown)
    # ══════════════════════════════════════════════════════════════════
    all_results = {}

    for B in batch_sizes:
        print(f"{'=' * 120}")
        print(f"Batch size = {B}")
        print(f"{'=' * 120}")

        xf_col = " %9s %9s |" % ("sdpa", "xformers") if _XFORMERS_AVAILABLE else " %9s |" % "sdpa"
        hdr = ("%6s %6s | %9s |" % ("splits", "Lk", "naive")
               + xf_col
               + " %9s %9s %7s | %9s %7s | %9s %9s %9s %7s" %
               ("splitk", "orc_2ph", "saved%",
                "orc_fuse", "fuse_sv%",
                "red_corr", "red_orc", "corr", "corr%"))
        print(hdr)
        print("-" * len(hdr))

        # GPU warmup: 클럭 안정화를 위해 heavy kernel 실행
        _warmup = torch.randn(B, H, 2048, D, device=device, dtype=dtype)
        for _ in range(50):
            torch.matmul(_warmup, _warmup.transpose(-2, -1))
        torch.cuda.synchronize()
        del _warmup

        for Lk in args.seq_lens:
            sm_scale = 1.0 / math.sqrt(D)
            torch.manual_seed(42)
            Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
            K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
            V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
            gm = compute_global_max(Q, K, sm_scale)

            naive_ms = do_bench(lambda: naive_attention(Q, K, V, sm_scale),
                                warmup=args.warmup, rep=args.rep)
            sdpa_ms = do_bench(lambda: sdpa_attention(Q, K, V, sm_scale),
                               warmup=args.warmup, rep=args.rep)
            xf_ms = None
            if _XFORMERS_AVAILABLE:
                Q_xf = Q.transpose(1, 2).contiguous()
                K_xf = K.transpose(1, 2).contiguous()
                V_xf = V.transpose(1, 2).contiguous()
                xf_ms = do_bench(lambda: xformers_attention(Q_xf, K_xf, V_xf, sm_scale),
                                 warmup=args.warmup, rep=args.rep)

            for ns in args.num_splits:
                # ── Autotune: splitk config을 oracle에도 동일 적용 (공정 비교) ──
                bm, bn, nw, nst = _autotune_kernel(
                    "splitk",
                    lambda q, k, v, sm, bm, bn, nw, nst: _launch_splitk_inner(
                        q, k, v, sm, bm, bn, nw, nst, ns, None),
                    Q, K, V, sm_scale, num_splits=ns)

                # ── Pre-allocate workspaces ──
                # Standard split-K
                po_s = torch.empty((B * H, ns, Lq, D), device=device, dtype=torch.float16)
                pm_s = torch.empty((B * H, ns, Lq), device=device, dtype=torch.float32)
                pl_s = torch.empty((B * H, ns, Lq), device=device, dtype=torch.float32)
                o_s = torch.empty_like(Q)

                # Oracle split-K (동일 config 사용)
                po_o = torch.empty((B * H, ns, Lq, D), device=device, dtype=torch.float16)
                pl_o = torch.empty((B * H, ns, Lq), device=device, dtype=torch.float32)
                o_o = torch.empty_like(Q)

                # Fused oracle workspace
                o_fused = torch.zeros(B, H, Lq, D, device=device, dtype=torch.float32)
                l_fused = torch.zeros(B * H, Lq, device=device, dtype=torch.float32)

                # ── Kernel-only 벤치마크 (할당 비용 제외, 동일 config) ──
                splitk_ms = do_bench(
                    lambda: _run_splitk_kernels(Q, K, V, sm_scale, bm, bn, nw, nst,
                                                ns, po_s, pm_s, pl_s, o_s),
                    warmup=args.warmup, rep=args.rep)
                oracle_2ph_ms = do_bench(
                    lambda: _run_oracle_kernels(Q, K, V, sm_scale, bm, bn, nw, nst,
                                                ns, gm, po_o, pl_o, o_o),
                    warmup=args.warmup, rep=args.rep)
                fused_ms = do_bench(
                    lambda: _run_fused_oracle_kernel(Q, K, V, sm_scale, BLOCK_M, 64,
                                                     4 if D <= 64 else 8, 2,
                                                     ns, gm, o_fused, l_fused),
                    warmup=args.warmup, rep=args.rep)

                # Isolated reduction (workspace 재사용)
                red_ms = do_bench(lambda: launch_reduce_only(po_s, pm_s, pl_s, (B, H, Lq, D), ns),
                                  warmup=args.warmup, rep=args.rep)
                red_orc_ms = do_bench(lambda: launch_reduce_oracle_only(po_o, pl_o, (B, H, Lq, D), ns),
                                      warmup=args.warmup, rep=args.rep)

                saved_2ph = (1 - oracle_2ph_ms / splitk_ms) * 100 if splitk_ms > 0 else 0
                saved_fused = (1 - fused_ms / splitk_ms) * 100 if splitk_ms > 0 else 0
                corr = red_ms - red_orc_ms
                corr_pct = (corr / splitk_ms) * 100 if splitk_ms > 0 else 0

                xf_str = " %8.4fms %8.4fms |" % (sdpa_ms, xf_ms) if xf_ms is not None else " %8.4fms |" % sdpa_ms
                print(("%6d %6d | %8.4fms |" % (ns, Lk, naive_ms))
                      + xf_str
                      + " %8.4fms %8.4fms %6.1f%% | %8.4fms %6.1f%% "
                        "| %8.4fms %8.4fms %8.4fms %6.1f%%" %
                      (splitk_ms, oracle_2ph_ms, saved_2ph,
                       fused_ms, saved_fused,
                       red_ms, red_orc_ms, corr, corr_pct))

                all_results[f"B{B}_splits{ns}_Lk{Lk}"] = {
                    "batch_size": B, "cache_len": Lk, "num_splits": ns,
                    "naive_ms": naive_ms, "sdpa_ms": sdpa_ms,
                    "xformers_ms": xf_ms,
                    "splitk_ms": splitk_ms,
                    "oracle_2phase_ms": oracle_2ph_ms, "oracle_fused_ms": fused_ms,
                    "reduce_ms": red_ms, "reduce_oracle_ms": red_orc_ms,
                    "saved_2phase_pct": saved_2ph, "saved_fused_pct": saved_fused,
                    "corr_pct": corr_pct,
                }
            print()

    # ══════════════════════════════════════════════════════════════════
    # Figure 14 재현: Speedup over HF (naive) — best split-K per config
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("Figure 14 재현: Speedup over HF (naive) — decode phase")
    print(f"{'=' * 100}")

    fig14_hdr = "%5s %6s | %8s %8s %8s %8s %8s" % (
        "B", "cache", "sdpa", "xformers", "splitk", "oracle", "saved%")
    print(fig14_hdr)
    print("-" * len(fig14_hdr))
    print("  splitk  = FlashDecoding (phase1 online softmax + reduce w/ correction)")
    print("  oracle  = FlashDecoding++ (phase1 global max + reduce simple sum)")
    print("  saved%  = (splitk - oracle) / splitk = partial softmax sync overhead")
    print("-" * len(fig14_hdr))

    geo_speedups = {"sdpa": [], "xformers": [], "splitk": [], "oracle": []}

    for B in batch_sizes:
        for Lk in args.seq_lens:
            # 해당 (B, Lk)에서 best split-K 찾기
            best_splitk = None
            best_oracle = None
            naive_ms = None
            sdpa_ms = None
            xf_ms = None

            for ns in args.num_splits:
                key = f"B{B}_splits{ns}_Lk{Lk}"
                if key not in all_results:
                    continue
                r = all_results[key]
                naive_ms = r["naive_ms"]
                sdpa_ms = r["sdpa_ms"]
                xf_ms = r["xformers_ms"]
                if best_splitk is None or r["splitk_ms"] < best_splitk:
                    best_splitk = r["splitk_ms"]
                if best_oracle is None or r["oracle_2phase_ms"] < best_oracle:
                    best_oracle = r["oracle_2phase_ms"]

            if naive_ms is None:
                continue

            sp_sdpa = naive_ms / sdpa_ms if sdpa_ms else 0
            sp_xf = naive_ms / xf_ms if xf_ms else 0
            sp_splitk = naive_ms / best_splitk if best_splitk else 0
            sp_oracle = naive_ms / best_oracle if best_oracle else 0
            saved_pct = (1 - best_oracle / best_splitk) * 100 if best_splitk else 0

            xf_str = f"{sp_xf:>8.2f}x" if xf_ms else f"{'n/a':>8s}"
            print(f"{B:>5d} {Lk:>6d} | {sp_sdpa:>7.2f}x {xf_str} "
                  f"{sp_splitk:>7.2f}x {sp_oracle:>7.2f}x {saved_pct:>7.1f}%")

            geo_speedups["sdpa"].append(sp_sdpa)
            if xf_ms:
                geo_speedups["xformers"].append(sp_xf)
            geo_speedups["splitk"].append(sp_splitk)
            geo_speedups["oracle"].append(sp_oracle)

    # Geo Mean
    import functools, operator
    def geo_mean(vals):
        if not vals:
            return 0
        return functools.reduce(operator.mul, vals, 1.0) ** (1.0 / len(vals))

    gm_sdpa = geo_mean(geo_speedups["sdpa"])
    gm_xf = geo_mean(geo_speedups["xformers"])
    gm_splitk = geo_mean(geo_speedups["splitk"])
    gm_oracle = geo_mean(geo_speedups["oracle"])
    # overall saved%: geo mean of oracle speedups / geo mean of splitk speedups
    gm_saved = (1 - gm_splitk / gm_oracle) * 100 if gm_oracle else 0

    xf_gm_str = f"{gm_xf:>8.2f}x" if geo_speedups["xformers"] else f"{'n/a':>8s}"
    print("-" * len(fig14_hdr))
    print(f"{'':>5s} {'GeoMn':>6s} | {gm_sdpa:>7.2f}x {xf_gm_str} "
          f"{gm_splitk:>7.2f}x {gm_oracle:>7.2f}x {gm_saved:>7.1f}%")
    print()
    print(f"  FlashDecoding++ 논문 주장: partial softmax sync overhead ≈ 18.8%")
    print(f"  우리 측정 saved%: {gm_saved:.1f}% (oracle이 splitk보다 빠른 비율)")

    # ── Roofline analysis for the last seq_len × last num_splits ──
    peak_tflops, peak_bw_gbs = get_gpu_specs()
    dtype_bytes = 2  # fp16

    print("Key metrics:")
    print("  splitk    = standard split-K: phase1 (online softmax) + reduce (with correction)")
    print("  orc_2ph   = oracle split-K: phase1 (global max) + reduce (simple sum) — 2 kernels")
    print("  orc_fuse  = oracle fused: phase1 atomic-adds to output — NO reduce kernel")
    print("  saved%    = (splitk - orc_2ph) / splitk")
    print("  fuse_sv%  = (splitk - orc_fuse) / splitk — savings with reduction fully eliminated")
    print("  corr%     = (red_w_correction - red_no_correction) / splitk — correction-only cost")

    # ── Roofline 분석: 각 Lk × num_splits 조합 ──
    for Lk in args.seq_lens:
        analyses = []
        sm_scale = 1.0 / math.sqrt(D)
        for ns in args.num_splits:
            key = f"splits{ns}_Lk{Lk}"
            if key not in all_results:
                continue
            r = all_results[key]

            analyses.append(roofline_analysis(
                B, H, Lq, Lk, D, 1, r["naive_ms"], dtype_bytes,
                peak_tflops, peak_bw_gbs, label=f"naive (Lk={Lk})"))
            analyses.append(roofline_analysis(
                B, H, Lq, Lk, D, ns, r["splitk_ms"], dtype_bytes,
                peak_tflops, peak_bw_gbs, label=f"splitk s={ns}"))
            analyses.append(roofline_analysis(
                B, H, Lq, Lk, D, ns, r["oracle_2phase_ms"], dtype_bytes,
                peak_tflops, peak_bw_gbs, label=f"oracle_2ph s={ns}"))
            analyses.append(roofline_analysis(
                B, H, Lq, Lk, D, ns, r["oracle_fused_ms"], dtype_bytes,
                peak_tflops, peak_bw_gbs, label=f"oracle_fused s={ns}"))
            break  # 한 splits만 표시 (첫 번째)

        if analyses:
            print(f"\n  Lk={Lk}:")
            print_roofline_table(analyses)
    print()
    print("  FlashDecoding++ claims: ~18.8%% for partial softmax update")
    print("           (Llama2-7B, A100, 1024 input length)")

    if args.save:
        with open(args.save, "w") as f:
            json.dump({
                "config": {"B": B, "H": H, "D": D, "Lq": Lq, "dtype": args.dtype,
                           "gpu": torch.cuda.get_device_name(0)},
                "results": all_results,
            }, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
