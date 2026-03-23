"""
Split-K Flash Decoding CUDA Kernels
====================================
Custom CUDA attention kernels for decode (Lq=1, D=128).

Kernels:
  - standard:       Single-kernel split-K with online softmax + correction
  - oracle:         Single-kernel split-K with global max, no correction
  - localmax_async: 2-kernel split-K (independent CTAs, no __syncthreads__)
  - profiled:       Standard with clock64() timestamps
  - profiled_oracle: Oracle with clock64() timestamps

Usage:
  import splitk_kernels as sk

  # Standard split-K attention
  out = sk.attention(Q, K, V, sm_scale, num_splits=8)

  # Oracle (needs pre-computed global max)
  out = sk.attention_oracle(Q, K, V, global_max, sm_scale, num_splits=8)

  # 2-kernel async split-K
  out = sk.attention_localmax_async(Q, K, V, sm_scale, num_splits=8)

  # Occupancy query
  info = sk.query_occupancy("standard", threads=256, smem=4160, grid=256)
"""

import os as _os
import importlib as _imp

import torch  # libc10.so 로드에 필요

# _C.so는 이 패키지 디렉토리 안에 있음 — 직접 로드
_dir = _os.path.dirname(_os.path.abspath(__file__))
_so = [f for f in _os.listdir(_dir) if f.startswith("_C") and f.endswith(".so")]
if _so:
    _spec = _imp.util.spec_from_file_location("_C", _os.path.join(_dir, _so[0]))
    _mod = _imp.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
else:
    raise ImportError(f"_C.so not found in {_dir}. Run: cd {_dir} && pip install -e .")

launch_standard = _mod.launch_standard
launch_oracle = _mod.launch_oracle
launch_localmax_async = _mod.launch_localmax_async
launch_oracle_async = _mod.launch_oracle_async
launch_tc_oracle_async = _mod.launch_tc_oracle_async
launch_profiled = _mod.launch_profiled
launch_profiled_oracle = _mod.launch_profiled_oracle
query_occupancy = _mod.query_occupancy

import math
import torch


def attention(Q, K, V, sm_scale=None, num_splits=8):
    """Standard split-K decode attention (online softmax + correction).

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        V: (B, H, Lk, D) float16
        sm_scale: softmax scale (default: 1/sqrt(D))
        num_splits: number of KV splits (max 32)
    Returns:
        (B, H, Lq, D) float16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    return launch_standard(Q, K, V, sm_scale, num_splits)


def attention_oracle(Q, K, V, global_max, sm_scale=None, num_splits=8):
    """Oracle split-K decode attention (global max, no correction).

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        V: (B, H, Lk, D) float16
        global_max: (B*H,) float32 — pre-computed global max of QK scores
        sm_scale: softmax scale (default: 1/sqrt(D))
        num_splits: number of KV splits (max 32)
    Returns:
        (B, H, Lq, D) float16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    return launch_oracle(Q, K, V, global_max, sm_scale, num_splits)


def attention_localmax_async(Q, K, V, sm_scale=None, num_splits=8):
    """2-kernel async split-K attention (independent CTAs, no __syncthreads__).

    Reads K twice (pass 1: find local max, pass 2: compute PV).
    Higher SM utilization than single-kernel split-K.

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        V: (B, H, Lk, D) float16
        sm_scale: softmax scale (default: 1/sqrt(D))
        num_splits: number of KV splits
    Returns:
        (B, H, Lq, D) float16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    return launch_localmax_async(Q, K, V, sm_scale, num_splits)


def attention_oracle_async(Q, K, V, global_max, sm_scale=None, num_splits=8):
    """Multi-CTA oracle split-K attention (FlashDecoding++ style).

    1-pass over K,V (no local max search). Uses pre-computed global max (φ).
    Simple sum reduction (no correction). High SM utilization even at batch=1.

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        V: (B, H, Lk, D) float16
        global_max: (B*H,) float32 — pre-computed global max or fixed φ
        sm_scale: softmax scale (default: 1/sqrt(D))
        num_splits: number of KV splits
    Returns:
        (B, H, Lq, D) float16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    return launch_oracle_async(Q, K, V, global_max, sm_scale, num_splits)


def attention_tc_oracle_async(Q, K, V, global_max, sm_scale=None, num_splits=8):
    """Multi-CTA oracle with Tensor Core QK (FlashDecoding++ style, TC accelerated).

    Same as oracle_async but QK dot product uses wmma (16 keys per tile).
    PV stays scalar. Best combination of SM utilization + TC acceleration.

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        V: (B, H, Lk, D) float16
        global_max: (B*H,) float32
        sm_scale: softmax scale (default: 1/sqrt(D))
        num_splits: number of KV splits
    Returns:
        (B, H, Lq, D) float16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    return launch_tc_oracle_async(Q, K, V, global_max, sm_scale, num_splits)


def compute_global_max(Q, K, sm_scale=None):
    """Compute per-(batch, head) global max of QK^T scores.

    Used for oracle attention.

    Args:
        Q: (B, H, Lq, D) float16
        K: (B, H, Lk, D) float16
        sm_scale: softmax scale (default: 1/sqrt(D))
    Returns:
        (B*H,) float32
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.shape[-1])
    B, H, Lq, D = Q.shape
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * sm_scale
    return scores.reshape(B * H, Lq, -1).amax(dim=-1).amax(dim=-1).contiguous()
