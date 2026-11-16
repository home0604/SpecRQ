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


def mma_codes(codes: torch.Tensor) -> torch.Tensor:
    """Logical planar codes [L, N, G] -> "mma_tile_v1" bytes [L, N*G] for nest_mma (see csrc/nest_mma.cu header).
    Byte (tile, ks, g, s, j) = code of row tile*16 + g + 8*(j>>1), group ks*8 + 2*s + (j&1). Needs N%16, G%8."""
    L, N, G = codes.shape
    assert N % 16 == 0 and G % 8 == 0, (N, G)
    c = codes.reshape(L, N // 16, 2, 8, G // 8, 4, 2)  # (L, tile, hi, g, ks, s, lo)
    return c.permute(0, 1, 4, 3, 5, 2, 6).reshape(L, N * G).contiguous()


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


# ------------------------------------------------------------------------------------------------ general (nestg)
def _nib_pack(v: torch.Tensor) -> torch.Tensor:
    """[..., 2n] values < 16 -> [..., n] bytes (element 2i in the low nibble)."""
    v = v.to(torch.uint8)
    return (v[..., 0::2] | (v[..., 1::2] << 4)).contiguous()


def _mma_order(codes: torch.Tensor, d: int) -> torch.Tensor:
    """Logical [L, N, G] -> [L, N*G] in nestg mma order: (tile, ks, g, t, j) with n = 4 (d=8) or 2 (d=16) per lane."""
    L, N, G = codes.shape
    if d == 8:
        return mma_codes(codes)  # (tile, ks, g, t, hi, lo): j = hi*2 + lo
    c = codes.reshape(L, N // 16, 2, 8, G // 4, 4)  # (L, tile, hi, g, ks, t)
    return c.permute(0, 1, 4, 3, 5, 2).reshape(L, N * G).contiguous()


def split_planes(c: torch.Tensor, b: int):
    """codes (any int dtype, values < 2^b) -> (byte, nibble values, byte1) sub-plane values (None when unused)."""
    c = c.to(torch.int64)
    p0 = (c & 0xFF).to(torch.uint8) if b >= 8 else None
    pn = ((c >> 8) & 0xF) if b == 12 else ((c & 0xF) if b == 4 else None)
    p1 = ((c >> 8) & 0xFF).to(torch.uint8) if b == 16 else None
    return p0, pn, p1


def kernel_bits(bits) -> int:
    """nestg B template for per-plane bit widths: uniform b -> b; a mix of 8 and 12 -> 12 (8-bit planes have no
    nibble sub-plane); any other mix of 4/8/12 -> 0 ("mix4812": per-plane byte and/or nibble sub-planes)."""
    s = set(bits)
    if len(s) == 1:
        return bits[0]
    if s == {8, 12}:
        return 12
    if s <= {4, 8, 12}:
        return 0
    raise ValueError(f"unsupported per-plane bit mix {bits}")


_QDIG = {16: 1, 8: 2, 4: 3}


def kernel_q(qs) -> int:
    """nestg Q template for per-plane codebook precision: uniform q -> q; otherwise a negative decimal pattern with
    digit l (least significant = plane 0) in {1: bf16, 2: int8, 3: int4}, e.g. (16, 8) -> -21. Trailing planes that
    repeat the last pattern digit are dropped (the kernel repeats the last digit)."""
    qs = list(qs)
    if len(set(qs)) == 1:
        return qs[0]
    while len(qs) > 1 and qs[-1] == qs[-2]:
        qs.pop()
    assert len(qs) <= 9, qs
    return -sum(_QDIG[q] * 10 ** l for l, q in enumerate(qs))


def pack_general(codes: torch.Tensor, bits, d: int) -> dict:
    """Logical codes [L, N, G] (G = K/d; plane l values < 2^bits[l]) -> nestg sub-planes for both paths
    (see csrc/nestg.cu header). bits: int or per-plane list. Nibble sub-planes are stacked only for planes that
    have one; `nib` maps plane -> nibble sub-plane index (-1: none)."""
    L, N, G = codes.shape
    bits = [bits] * L if isinstance(bits, int) else list(bits)
    B = kernel_bits(bits)
    nib = [i for i, b in enumerate(bits) if b in (4, 12)]
    # byte sub-planes: all planes for B >= 8 (as before); B = 0 stacks only the 8/12-bit planes
    byt = [l for l in range(L) if bits[l] in (8, 12)] if B == 0 else (list(range(L)) if B >= 8 else [])
    out = {"nib": [nib.index(l) if l in nib else -1 for l in range(L)], "B": B, "bits": bits,
           "byt": [byt.index(l) if l in byt else -1 for l in range(L)]}
    for path, ordered in (("gemv", codes[..., lane_perm(G).to(codes.device)]), ("mma", _mma_order(codes, d))):
        flat = ordered.reshape(L, -1).to(torch.int64)
        if B == 0:
            p0 = (flat[byt] & 0xFF).to(torch.uint8).contiguous()
        else:
            p0 = (flat & 0xFF).to(torch.uint8).contiguous() if B >= 8 else None
        pn = None
        if nib:
            nv = torch.stack([((flat[l] >> 8) & 0xF) if bits[l] == 12 else (flat[l] & 0xF) for l in nib])
            pn = _nib_pack(nv)
        p1 = ((flat >> 8) & 0xFF).to(torch.uint8).contiguous() if B == 16 else None
        out[path] = {"p0": p0, "pn": pn, "p1": p1}
    return out


def quantize_codebook_list(cbs, q):
    """List of per-plane float codebooks [C_l, d] (C_l may differ) -> (bytes uint8 [sum], scales fp32 [L],
    dequantized list, byte offsets long [L+1] on CPU). q: int or per-plane list (16/8/4)."""
    qs = [q] * len(cbs) if isinstance(q, int) else list(q)
    assert len(qs) == len(cbs), (qs, len(cbs))
    raws, scs, deqs, off = [], [], [], [0]
    for cb, ql in zip(cbs, qs):
        r, s, dq = quantize_codebooks(cb[None], ql)
        raws.append(r)
        scs.append(s)
        deqs.append(dq[0])
        off.append(off[-1] + r.numel())
    return torch.cat(raws), torch.cat(scs), deqs, torch.tensor(off, dtype=torch.long)


def quantize_codebooks(cb: torch.Tensor, q: int):
    """cb float [L, C, d] -> (raw bytes uint8 [L * C * d * q/8], per-plane scale fp32 [L], dequantized fp32 [L, C, d]).
    q=16: bf16 values (scale 1); q=8: symmetric int8, scale = max|v|/127 per plane; q=4: int4 [-8, 7], max|v|/7."""
    L, C, d = cb.shape
    if q == 16:
        v = cb.to(torch.bfloat16)
        return v.contiguous().view(torch.uint8).reshape(-1), torch.ones(L, device=cb.device), v.float()
    qmax = 127 if q == 8 else 7
    s = (cb.abs().amax(dim=(1, 2)) / qmax).clamp_min(1e-12).float()
    iv = torch.round(cb / s[:, None, None]).clamp(-qmax - (1 if q == 4 else 0), qmax).to(torch.int8)
    deq = iv.float() * s[:, None, None]
    if q == 8:
        return iv.contiguous().view(torch.uint8).reshape(-1), s, deq
    u = (iv.to(torch.int16) & 0xF).to(torch.uint8)  # two's complement nibbles
    return _nib_pack(u).reshape(-1), s, deq
