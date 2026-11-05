"""[specrq] Block Walsh-Hadamard rotation for Drop-by-Drop (arXiv 2606.12876, Alg. 1/2/4).

Per linear layer y = W x (W: d_out x d_in):
    R = blockdiag(H_B, ..., H_B) @ diag(S),  H_B = orthonormal Sylvester Hadamard (H_B = H_B^T, H_B H_B = I)
    W' = W R^T  <=>  row-wise  w' = BlockFWHT(w * S)          (Alg. 1 line 7)
    XX^T' = R XX^T R^T                                        (Alg. 1 line 8)
    inference: x' = R x = BlockFWHT(x * S), y = W' x'         (Alg. 4 lines 10-15)
so W' x' = W R^T R x = W x exactly (up to floating point).
B_H = largest of {256, 128, 64, 32} dividing d_in, else 1 (= no rotation).
The paper does not state the normalization; 1/sqrt(B) is required for W' x' == W x.
"""
from __future__ import annotations

import functools
import zlib
from typing import Optional

import torch

BLOCK_CANDIDATES = (256, 128, 64, 32)


def pick_block_size(d_in: int) -> int:
    for b in BLOCK_CANDIDATES:
        if d_in % b == 0:
            return b
    return 1


@functools.lru_cache(maxsize=None)
def _hadamard_cpu(b: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == b, f"block size {b} must be a power of two"
    return h / (b**0.5)


def hadamard_matrix(b: int, device, dtype) -> torch.Tensor:
    return _hadamard_cpu(b).to(device=device, dtype=dtype)


def block_fwht(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply blockdiag(H_B) along the last dim (orthonormal, symmetric). Computed in fp32 or wider."""
    if block_size <= 1:
        return x
    d = x.shape[-1]
    assert d % block_size == 0, (d, block_size)
    cdtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = hadamard_matrix(block_size, x.device, cdtype)
    y = x.to(cdtype).reshape(*x.shape[:-1], d // block_size, block_size) @ h  # H symmetric -> x_blk @ H = (H x_blk)
    return y.reshape(x.shape).to(x.dtype)


def make_signs(d_in: int, seed: int, device=None) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, (d_in,), generator=g, dtype=torch.int64) * 2 - 1
    return s.to(dtype=torch.float32, device=device)


def layer_seed(base_seed: int, layer_index: int, name: str) -> int:
    return zlib.crc32(f"{base_seed}:{layer_index}:{name}".encode()) & 0x7FFFFFFF


def rotate_weight(w: torch.Tensor, signs: torch.Tensor, block_size: int) -> torch.Tensor:
    """W' = W R^T (row-wise BlockFWHT(w * S)). No-op when block_size <= 1 (paper: B_H = 1 -> no rotation)."""
    if block_size <= 1:
        return w
    return block_fwht(w * signs.to(w.device, w.dtype), block_size)


def rotate_xtx(xtx: torch.Tensor, signs: torch.Tensor, block_size: int) -> torch.Tensor:
    """XX^T' = R XX^T R^T. No-op when block_size <= 1."""
    if block_size <= 1:
        return xtx
    s = signs.to(xtx.device, xtx.dtype)
    a = xtx * s[:, None] * s[None, :]
    a = block_fwht(a, block_size)  # right-multiply by H (columns)
    a = block_fwht(a.transpose(0, 1).contiguous(), block_size).transpose(0, 1)  # left-multiply by H (rows)
    return a.contiguous()


def rotate_input(x: torch.Tensor, signs: Optional[torch.Tensor], block_size: int) -> torch.Tensor:
    """x' = R x = BlockFWHT(x * S) along the last dim (online, Alg. 4)."""
    if signs is None or block_size <= 1:
        return x
    return block_fwht(x * signs.to(x.device, x.dtype), block_size)
