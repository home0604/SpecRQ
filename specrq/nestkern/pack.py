"""Planar packer + prefix-k reference dequant for nested additive VQ weights (DbD / AQLM format).

Source format (DbD QuantizedWeight / AQLM, out_group_size=1):
  codes     [N, K/d, L]        int (int32 in DbD ckpts, int8/int16 in AQLM), values in [0, 2^b)
  codebooks [L, 2^b, 1, d]     fp16/bf16
  scales    [N, 1, 1, 1]       fp16/bf16
  W[n, g*d:(g+1)*d] = scales[n] * sum_l codebooks[l, codes[n, g, l], 0, :]

Planar format "planar_rowmajor_v1" (what the nestkern kernels read):
  codes     [L, N, K/d] uint8  codebook-major; plane l is a dense N x K/d byte matrix, so a prefix-k draft reads
                               only planes 0..k-1 (k*N*K/d bytes)
  codebooks [L, 256, d]        same dtype as source (or cast)
  scales    [N]
Only b=8, d=8 is supported (C1 family).

Kernel byte order ("lane-permuted", applied by `kernel_codes`): the GEMV kernel reads one u32 (4 codes) per lane;
within each block of 128 groups (32 words) byte g of word j holds logical group 128*b + g*w + j (w = words in the
block, 32 except the tail block), so for a fixed byte position the 32 lanes need contiguous x (coalesced 512 B).
PlanarVQ.codes stays in logical order; `kernel_codes` makes the permuted copy the kernels consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FORMAT = "planar_rowmajor_v1"


@dataclass
class PlanarVQ:
    codes: torch.Tensor      # [L, N, K/8] uint8
    codebooks: torch.Tensor  # [L, 256, 8]
    scales: torch.Tensor     # [N]
    N: int
    K: int

    @property
    def L(self):
        return self.codes.shape[0]

    def to(self, device=None, dtype=None):
        return PlanarVQ(self.codes.to(device), self.codebooks.to(device=device, dtype=dtype),
                        self.scales.to(device=device, dtype=dtype), self.N, self.K)

    def nbytes(self, k: int) -> int:
        """Bytes a prefix-k kernel must read (codes planes + codebooks + scales)."""
        return k * self.N * (self.K // 8) + k * 256 * 8 * self.codebooks.element_size() + self.N * self.scales.element_size()


def pack_planar(codes: torch.Tensor, codebooks: torch.Tensor, scales: torch.Tensor, dtype=None) -> PlanarVQ:
    L, C, og, d = codebooks.shape
    if (C, og, d) != (256, 1, 8):
        raise ValueError(f"only 8-bit codebooks with out_group_size=1, d=8 supported; got {tuple(codebooks.shape)}")
    N, G, L2 = codes.shape
    assert L2 == L, (codes.shape, codebooks.shape)
    c = codes.to(torch.int64) % 256  # AQLM stores uint8 data in int8 (negative values); DbD uses int32
    planar = c.permute(2, 0, 1).contiguous().to(torch.uint8)  # [L, N, G]
    cb = codebooks[:, :, 0, :].contiguous()
    sc = scales.reshape(N).contiguous()
    if dtype is not None:
        cb, sc = cb.to(dtype), sc.to(dtype)
    return PlanarVQ(planar, cb, sc, N, G * d)


def lane_perm(G: int) -> torch.Tensor:
    """P with kernel_codes[..., i] = codes[..., P[i]] (see module doc). Requires G % 4 == 0."""
    assert G % 4 == 0, G
    nch = G // 4
    P = torch.empty(G, dtype=torch.long)
    for ch in range(nch):
        b, j = divmod(ch, 32)
        w = min(32, nch - 32 * b)
        for g in range(4):
            P[4 * ch + g] = 128 * b + g * w + j
    return P


def kernel_codes(codes: torch.Tensor) -> torch.Tensor:
    """Logical planar codes [L, N, G] -> lane-permuted copy for nest_gemv."""
    return codes[..., lane_perm(codes.shape[-1]).to(codes.device)].contiguous()


def unpack_planar(p: PlanarVQ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Back to AQLM/DbD layout (codes int32)."""
    return (p.codes.permute(1, 2, 0).to(torch.int32).contiguous(), p.codebooks[:, :, None, :].contiguous(),
            p.scales.view(-1, 1, 1, 1))


def ref_dequant(p: PlanarVQ, k: int, compute_dtype=torch.float32) -> torch.Tensor:
    """W_k [N, K] = scales * sum_{l<k} codebooks[l][codes[l]] (summed in compute_dtype)."""
    assert 1 <= k <= p.L
    G = p.K // 8
    acc = torch.zeros(p.N, G, 8, device=p.codes.device, dtype=compute_dtype)
    for l in range(k):
        acc += p.codebooks[l].to(compute_dtype)[p.codes[l].long()]
    return (acc * p.scales.to(compute_dtype)[:, None, None]).reshape(p.N, p.K)


def ref_matmul(x: torch.Tensor, p: PlanarVQ, k: int) -> torch.Tensor:
    return x.float() @ ref_dequant(p, k).t()
