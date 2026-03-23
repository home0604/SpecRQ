"""
Triton Flash Attention: KIVI 4-bit KV Cache + Global Max Oracle
================================================================
Decode 단계에서 KV cache를 KIVI 방식 INT4로 압축 저장.
K/V load bandwidth를 ~4x 줄여서 compute 비중을 높이고,
global max oracle의 이득이 커지는지 측정.

비교 설정 (모두 decode, Lq << Lk):
  1. fp16 flash           : 기존 fp16 KV, online softmax
  2. fp16 oracle          : 기존 fp16 KV, global max oracle
  3. int4 flash           : KIVI 4-bit KV, online softmax
  4. int4 oracle          : KIVI 4-bit KV, global max oracle

KIVI 4-bit 양자화 (Asymmetric per-group along D):
  - group_size along D dimension (default 32)
  - scale = (max - min) / 15, zero = min
  - q_val = round((x - zero) / scale), clipped [0, 15]
  - dequant: x ≈ q_val * scale + zero
  - Packing: int32, 8 consecutive D values per int32
  - Kernel: KIVI-style modular index trick for single full-D dot product
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import math
import argparse
import json


# ===================================================================
# 4-bit Quantization (KIVI-style, pack along D)
# ===================================================================

def quantize_4bit(x, group_size=32):
    """
    KIVI-style asymmetric 4-bit quantization along D dimension.
    x: (B, H, T, D) fp16
    → packed (B, H, T, D//8) int32, scale (B, H, T, D//gs) fp16, zero (B, H, T, D//gs) fp16

    Packing: 8 consecutive D values into one int32
      packed[..., j] = x_q[..., 8j] | (x_q[..., 8j+1] << 4) | ... | (x_q[..., 8j+7] << 28)
    """
    B, H, T, D = x.shape
    assert D % group_size == 0 and D % 8 == 0

    x_f = x.float()
    x_g = x_f.view(B, H, T, D // group_size, group_size)
    x_min = x_g.amin(dim=-1, keepdim=True)
    x_max = x_g.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min) / 15.0).clamp(min=1e-8)
    zero = x_min

    x_q = torch.clamp(torch.round((x_g - zero) / scale), 0, 15).to(torch.int32)
    x_q = x_q.view(B, H, T, D)

    # Pack 8 consecutive D values into int32
    x_q = x_q.view(B, H, T, D // 8, 8)
    packed = torch.zeros(B, H, T, D // 8, dtype=torch.int32, device=x.device)
    for i in range(8):
        packed |= (x_q[..., i] & 0xF) << (i * 4)

    scale = scale.squeeze(-1).to(x.dtype)   # (B, H, T, D//gs)
    zero = zero.squeeze(-1).to(x.dtype)     # (B, H, T, D//gs)
    return packed.contiguous(), scale.contiguous(), zero.contiguous()


def dequantize_4bit(packed, scale, zero, D, group_size=32):
    """Dequantize for correctness verification."""
    shape = packed.shape  # (B, H, T, D//8)
    B, H, T, D8 = shape
    assert D == D8 * 8

    vals = []
    for i in range(8):
        vals.append(((packed >> (i * 4)) & 0xF).to(torch.float32))
    x_int = torch.stack(vals, dim=-1).reshape(B, H, T, D)  # (B, H, T, D)

    # Dequant per group
    x_int = x_int.view(B, H, T, D // group_size, group_size)
    x_deq = x_int * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)
    return x_deq.view(B, H, T, D).to(torch.float16)


# ===================================================================
# FP16 Decode Kernels (baseline, from bench_triton.py)
# ===================================================================

@triton.jit
def _fwd_decode_fp16(
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


@triton.jit
def _fwd_decode_fp16_oracle(
    Q, K, V, sm_scale, GlobalMax, L, Out,
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

    m_global = tl.load(GlobalMax + off_hz * N_CTX_Q + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_off, shape=(BLOCK_DMODEL, N_CTX_K),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_off, shape=(N_CTX_K, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk), offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))

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
    tl.store(L + off_hz * N_CTX_Q + offs_m, m_global + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on,
             acc.to(K.dtype.element_ty), mask=q_mask)


# ===================================================================
# INT4 Decode Kernels (KIVI-style int32 packing, modular index trick)
# K_packed, V_packed: (B, H, Lk, D//8) int32 — 8 × 4-bit per int32
# K_scale, V_scale, K_zero, V_zero: (B, H, Lk, D//group_size) fp16
# Dequant: val * scale + zero (asymmetric)
# ===================================================================

@triton.jit
def _fwd_decode_int4(
    Q, K_packed, K_scale, K_zero, V_packed, V_scale, V_zero,
    sm_scale, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX_Q, N_CTX_K,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    o_off = off_z * stride_oz + off_h * stride_oh

    D_PACKED: tl.constexpr = BLOCK_DMODEL // 8
    NUM_GROUPS: tl.constexpr = BLOCK_DMODEL // GROUP_SIZE

    # KV cache base offsets (contiguous: B, H, Lk, D//8 or D//gs)
    kv_packed_base = (off_z * H + off_h) * N_CTX_K * D_PACKED
    kv_scale_base = (off_z * H + off_h) * N_CTX_K * NUM_GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    # Load Q: (BLOCK_M, D) fp16
    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(tl.float16)

    # Precompute index helpers for modular dequant
    d_packed_idx = offs_d // 8          # which int32 in D//8
    d_shift = (offs_d % 8) * 4          # bit shift for this nibble
    d_group_idx = offs_d // GROUP_SIZE   # which group in D//gs

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # --- Dequant K (transposed): k[d, n] = K_packed[n, d//8] ---
        k_ptrs = K_packed + kv_packed_base + offs_n[None, :] * D_PACKED + d_packed_idx[:, None]
        k_packed_val = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0)
        k_int = (k_packed_val >> d_shift[:, None]) & 0xF  # (D, BLOCK_N)

        ks_ptrs = K_scale + kv_scale_base + offs_n[None, :] * NUM_GROUPS + d_group_idx[:, None]
        kz_ptrs = K_zero + kv_scale_base + offs_n[None, :] * NUM_GROUPS + d_group_idx[:, None]
        ks = tl.load(ks_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0.0)
        kz = tl.load(kz_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0.0)
        k = (k_int.to(tl.float16) * ks + kz).to(tl.float16)  # (D, BLOCK_N) fp16

        # QK = Q @ K^T
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N)

        # Online softmax
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # --- Dequant V: v[n, d] = V_packed[n, d//8] ---
        v_ptrs = V_packed + kv_packed_base + offs_n[:, None] * D_PACKED + d_packed_idx[None, :]
        v_packed_val = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0)
        v_int = (v_packed_val >> d_shift[None, :]) & 0xF  # (BLOCK_N, D)

        vs_ptrs = V_scale + kv_scale_base + offs_n[:, None] * NUM_GROUPS + d_group_idx[None, :]
        vz_ptrs = V_zero + kv_scale_base + offs_n[:, None] * NUM_GROUPS + d_group_idx[None, :]
        vs = tl.load(vs_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)
        vz = tl.load(vz_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)
        v = (v_int.to(tl.float16) * vs + vz).to(tl.float16)  # (BLOCK_N, D) fp16

        acc *= alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = acc / l_i[:, None]
    tl.store(L + off_hz * N_CTX_Q + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on,
             acc.to(tl.float16), mask=q_mask)


@triton.jit
def _fwd_decode_int4_oracle(
    Q, K_packed, K_scale, K_zero, V_packed, V_scale, V_zero,
    sm_scale, GlobalMax, L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX_Q, N_CTX_K,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_off = off_z * stride_qz + off_h * stride_qh
    o_off = off_z * stride_oz + off_h * stride_oh

    D_PACKED: tl.constexpr = BLOCK_DMODEL // 8
    NUM_GROUPS: tl.constexpr = BLOCK_DMODEL // GROUP_SIZE

    kv_packed_base = (off_z * H + off_h) * N_CTX_K * D_PACKED
    kv_scale_base = (off_z * H + off_h) * N_CTX_K * NUM_GROUPS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_mask = offs_m[:, None] < N_CTX_Q

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
                mask=q_mask, other=0.0)
    qk_scale = sm_scale * 1.44269504
    q = (q * qk_scale).to(tl.float16)

    m_global = tl.load(GlobalMax + off_hz * N_CTX_Q + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

    d_packed_idx = offs_d // 8
    d_shift = (offs_d % 8) * 4
    d_group_idx = offs_d // GROUP_SIZE

    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, N_CTX_K, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # --- Dequant K (transposed) ---
        k_ptrs = K_packed + kv_packed_base + offs_n[None, :] * D_PACKED + d_packed_idx[:, None]
        k_packed_val = tl.load(k_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0)
        k_int = (k_packed_val >> d_shift[:, None]) & 0xF

        ks_ptrs = K_scale + kv_scale_base + offs_n[None, :] * NUM_GROUPS + d_group_idx[:, None]
        kz_ptrs = K_zero + kv_scale_base + offs_n[None, :] * NUM_GROUPS + d_group_idx[:, None]
        ks = tl.load(ks_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0.0)
        kz = tl.load(kz_ptrs, mask=offs_n[None, :] < N_CTX_K, other=0.0)
        k = (k_int.to(tl.float16) * ks + kz).to(tl.float16)

        qk = tl.dot(q, k)

        # No correction — global max known
        p = tl.math.exp2(qk - m_global[:, None])

        # --- Dequant V ---
        v_ptrs = V_packed + kv_packed_base + offs_n[:, None] * D_PACKED + d_packed_idx[None, :]
        v_packed_val = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0)
        v_int = (v_packed_val >> d_shift[None, :]) & 0xF

        vs_ptrs = V_scale + kv_scale_base + offs_n[:, None] * NUM_GROUPS + d_group_idx[None, :]
        vz_ptrs = V_zero + kv_scale_base + offs_n[:, None] * NUM_GROUPS + d_group_idx[None, :]
        vs = tl.load(vs_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)
        vz = tl.load(vz_ptrs, mask=offs_n[:, None] < N_CTX_K, other=0.0)
        v = (v_int.to(tl.float16) * vs + vz).to(tl.float16)

        acc += tl.dot(p.to(tl.float16), v)
        l_i += tl.sum(p, 1)

    acc = acc / l_i[:, None]
    tl.store(L + off_hz * N_CTX_Q + offs_m, m_global + tl.math.log2(l_i), mask=offs_m < N_CTX_Q)
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on,
             acc.to(tl.float16), mask=q_mask)


# ===================================================================
# Python wrappers
# ===================================================================

BLOCK_M = 128
BLOCK_N = 64
GROUP_SIZE = 32


def launch_fp16(q, k, v, sm_scale, global_max=None):
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    o = torch.empty_like(q)
    L = torch.empty((B * H, Lq), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Lq, BLOCK_M), B * H)
    nw = 4 if D <= 64 else 8
    common = dict(
        stride_qz=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_kz=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
        stride_vz=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
        stride_oz=o.stride(0), stride_oh=o.stride(1), stride_om=o.stride(2), stride_on=o.stride(3),
        Z=B, H=H, N_CTX_Q=Lq, N_CTX_K=Lk,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
        num_warps=nw, num_stages=4,
    )
    if global_max is None:
        _fwd_decode_fp16[grid](q, k, v, sm_scale, L, o, **common)
    else:
        _fwd_decode_fp16_oracle[grid](q, k, v, sm_scale, global_max, L, o, **common)
    return o


def launch_int4(q, kp, ks, kz, vp, vs, vz, sm_scale, global_max=None):
    B, H, Lq, D = q.shape
    Lk = kp.shape[2]
    o = torch.empty_like(q)
    L = torch.empty((B * H, Lq), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(Lq, BLOCK_M), B * H)
    nw = 4 if D <= 64 else 8
    common = dict(
        stride_qz=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qk=q.stride(3),
        stride_oz=o.stride(0), stride_oh=o.stride(1), stride_om=o.stride(2), stride_on=o.stride(3),
        Z=B, H=H, N_CTX_Q=Lq, N_CTX_K=Lk,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
        num_warps=nw, num_stages=4,
    )
    # Reduce stages for int4 (more tensors per iteration → more shared memory)
    common['num_stages'] = 1
    if global_max is None:
        _fwd_decode_int4[grid](q, kp, ks, kz, vp, vs, vz, sm_scale, L, o, **common)
    else:
        _fwd_decode_int4_oracle[grid](q, kp, ks, kz, vp, vs, vz, sm_scale, global_max, L, o, **common)
    return o


def naive_attention(Q, K, V, sm_scale):
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def compute_global_max(q, k, sm_scale):
    """Oracle: true row-wise global max in log2 scale."""
    B, H, Lq, D = q.shape
    qk_scale = sm_scale * 1.44269504
    gm = []
    for b in range(B):
        for h in range(H):
            s = torch.matmul(q[b, h].float(), k[b, h].float().T) * qk_scale
            gm.append(s.max(dim=-1).values)
    return torch.stack(gm).contiguous()


def compute_global_max_int4(q, kp, ks, kz, sm_scale, D):
    """Oracle for int4 KV: compute from dequantized K."""
    K_deq = dequantize_4bit(kp, ks, kz, D, GROUP_SIZE)
    return compute_global_max(q, K_deq, sm_scale)


# ===================================================================
# Correctness
# ===================================================================

def check_correctness(B, H, Lq, Lk, D, device, dtype):
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    ref = naive_attention(Q, K, V, sm_scale)

    # FP16
    out_fp16 = launch_fp16(Q, K, V, sm_scale)
    gm_fp16 = compute_global_max(Q, K, sm_scale)
    out_fp16_orc = launch_fp16(Q, K, V, sm_scale, gm_fp16)

    # INT4
    Kp, Ks, Kz = quantize_4bit(K, GROUP_SIZE)
    Vp, Vs, Vz = quantize_4bit(V, GROUP_SIZE)
    K_deq = dequantize_4bit(Kp, Ks, Kz, D, GROUP_SIZE)
    V_deq = dequantize_4bit(Vp, Vs, Vz, D, GROUP_SIZE)
    ref_int4 = naive_attention(Q, K_deq, V_deq, sm_scale)

    out_int4 = launch_int4(Q, Kp, Ks, Kz, Vp, Vs, Vz, sm_scale)
    gm_int4 = compute_global_max_int4(Q, Kp, Ks, Kz, sm_scale, D)
    out_int4_orc = launch_int4(Q, Kp, Ks, Kz, Vp, Vs, Vz, sm_scale, gm_int4)

    def chk(a, b, name, tol=1e-2):
        d = (a.float() - b.float()).abs().max().item()
        status = "PASS" if d < tol else "FAIL"
        print(f"  {name:40s} max_diff={d:.6e}  [{status}]")
        return d < tol

    print(f"Correctness (Lq={Lq}, Lk={Lk}):")
    ok = True
    ok &= chk(ref, out_fp16, "fp16 flash vs naive")
    ok &= chk(ref, out_fp16_orc, "fp16 oracle vs naive")
    ok &= chk(ref_int4, out_int4, "int4 flash vs naive(dequant)")
    ok &= chk(ref_int4, out_int4_orc, "int4 oracle vs naive(dequant)")
    return ok


# ===================================================================
# Benchmark
# ===================================================================

def bench_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "mean_ms": sum(times) / len(times)}


def run_single(B, H, Lq, Lk, D, device, dtype, warmup, iters):
    torch.manual_seed(42)
    sm_scale = 1.0 / math.sqrt(D)

    Q = torch.randn(B, H, Lq, D, device=device, dtype=dtype)
    K = torch.randn(B, H, Lk, D, device=device, dtype=dtype)
    V = torch.randn(B, H, Lk, D, device=device, dtype=dtype)

    Kp, Ks, Kz = quantize_4bit(K, GROUP_SIZE)
    Vp, Vs, Vz = quantize_4bit(V, GROUP_SIZE)

    gm_fp16 = compute_global_max(Q, K, sm_scale)
    gm_int4 = compute_global_max_int4(Q, Kp, Ks, Kz, sm_scale, D)

    res = {}

    print(f"  fp16 flash ...", end="", flush=True)
    res["fp16_flash"] = bench_fn(lambda: launch_fp16(Q, K, V, sm_scale), warmup, iters)
    print(f" {res['fp16_flash']['median_ms']:.4f}ms")

    print(f"  fp16 oracle ...", end="", flush=True)
    res["fp16_oracle"] = bench_fn(lambda: launch_fp16(Q, K, V, sm_scale, gm_fp16), warmup, iters)
    print(f" {res['fp16_oracle']['median_ms']:.4f}ms")

    print(f"  int4 flash ...", end="", flush=True)
    res["int4_flash"] = bench_fn(lambda: launch_int4(Q, Kp, Ks, Kz, Vp, Vs, Vz, sm_scale), warmup, iters)
    print(f" {res['int4_flash']['median_ms']:.4f}ms")

    print(f"  int4 oracle ...", end="", flush=True)
    res["int4_oracle"] = bench_fn(lambda: launch_int4(Q, Kp, Ks, Kz, Vp, Vs, Vz, sm_scale, gm_int4), warmup, iters)
    print(f" {res['int4_oracle']['median_ms']:.4f}ms")

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--draft_lens", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    device = "cuda"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={args.batch_size}, H={args.n_heads}, D={args.head_dim}, dtype={args.dtype}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, GROUP_SIZE={GROUP_SIZE}\n")

    # Correctness
    ok = check_correctness(args.batch_size, args.n_heads, args.draft_lens[0],
                           args.seq_lens[0], args.head_dim, device, dtype)
    if not ok:
        print("CORRECTNESS FAILED — aborting")
        return
    print()

    # Sweep
    all_results = {}
    for dl in args.draft_lens:
        for sl in args.seq_lens:
            key = f"Lq{dl}_Lk{sl}"
            print(f"--- Lq={dl}, Lk={sl} ---")
            all_results[key] = run_single(
                args.batch_size, args.n_heads, dl, sl,
                args.head_dim, device, dtype, args.warmup, args.iters)
            print()

    # Summary
    print("=" * 105)
    print(f"{'Lq':>4} {'Lk':>6} | {'fp16_flash':>10} {'fp16_orc':>10} {'fp16_saved':>10} | "
          f"{'int4_flash':>10} {'int4_orc':>10} {'int4_saved':>10} | {'int4/fp16':>9}")
    print("-" * 105)
    for key, res in all_results.items():
        parts = key.split("_")
        lq = int(parts[0][2:])
        lk = int(parts[1][2:])

        f16f = res["fp16_flash"]["median_ms"]
        f16o = res["fp16_oracle"]["median_ms"]
        f16s = (1 - f16o / f16f) * 100

        i4f = res["int4_flash"]["median_ms"]
        i4o = res["int4_oracle"]["median_ms"]
        i4s = (1 - i4o / i4f) * 100

        # int4 vs fp16 overall speedup
        i4_vs_f16 = f16f / i4f

        print(f"{lq:>4} {lk:>6} | {f16f:>9.4f}ms {f16o:>9.4f}ms {f16s:>8.1f}% | "
              f"{i4f:>9.4f}ms {i4o:>9.4f}ms {i4s:>8.1f}% | {i4_vs_f16:>8.2f}x")
    print("=" * 105)

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
