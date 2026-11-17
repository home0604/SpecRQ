"""E7 magicvq: KIVI-style group-wise quantized KV decode attention (Triton split-K) + quantizers, for draft timing.

Storage per layer (post-RoPE K, as KIVI caches it; the tau eval quantized pre-RoPE K - same bytes):
  K codes uint8 [B, H, T, D*b/8] (b = 2 or 4 bit, channel-packed, element c = (8/b)*byte + j), K scale / zero fp16
  [B, H, T/32, D]  (per channel, per group of 32 tokens = KIVI per-channel key quantization, group 32)
  V codes uint8 [B, H, T, D*b/8], V scale / zero fp16 [B, H, T, D/32]  (per token, per group of 32 channels)
  bytes / element = b/8 + 2*2/32 (K: scale+zero per 32 tokens; V: per 32 channels) -> 2-bit: 0.375 B (3 bit), 4-bit: 5 bit
Kernel: same flash-decoding split-K structure / workspace as specrq/kvq `_attn_split` (q RoPE fused, M query rows per
sequence, G q-heads per kv-head stacked), writing partials (O, m, l) that kvq `_attn_combine` (or kvq_hybrid
`_combine2`, for the bf16 residual window) merges. Tile = 32 tokens = one K group.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

import specrq.kvq.kernels as K

GRP = 32


class KIVICache:
    def __init__(self, n_layers, B, Hkv, T, bits, D=128, device="cuda", gen=None):
        assert bits in (2, 4) and T % GRP == 0
        self.n_layers, self.B, self.Hkv, self.T, self.D, self.bits = n_layers, B, Hkv, T, D, bits
        W = D * bits // 8
        mk = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=device, generator=gen)  # noqa: E731
        sc = lambda *s: (torch.rand(*s, device=device, generator=gen) * 0.1 + 0.01).half()  # noqa: E731
        self.kc = [mk(B, Hkv, T, W) for _ in range(n_layers)]
        self.vc = [mk(B, Hkv, T, W) for _ in range(n_layers)]
        self.ks = [sc(B, Hkv, T // GRP, D) for _ in range(n_layers)]
        self.kz = [sc(B, Hkv, T // GRP, D) for _ in range(n_layers)]
        self.vs = [sc(B, Hkv, T, D // GRP) for _ in range(n_layers)]
        self.vz = [sc(B, Hkv, T, D // GRP) for _ in range(n_layers)]

    def bytes_per_token(self):
        return self.n_layers * self.Hkv * 2 * (self.D * self.bits / 8 + 2 * 2 * self.D / GRP)


@triton.jit
def _unpack(x, BN: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr):
    """x uint8 [BN, D*BITS/8] -> float32 [BN, D] codes (element c = (8/BITS)*byte + j)."""
    if BITS == 4:
        lo = (x & 15).to(tl.float32)
        hi = (x >> 4).to(tl.float32)
        return tl.reshape(tl.join(lo, hi), (BN, D))
    else:
        a0 = (x & 3).to(tl.float32)
        a1 = ((x >> 2) & 3).to(tl.float32)
        a2 = ((x >> 4) & 3).to(tl.float32)
        a3 = ((x >> 6) & 3).to(tl.float32)
        return tl.reshape(tl.join(tl.join(a0, a2), tl.join(a1, a3)), (BN, D))


@triton.jit
def _kivi_split(QKV, KC, VC, KS, KZ, VS, VZ, INVF, OP, MP, LP, ctx, chunk, qk_scale, sq_r,
                c_b, c_h, c_t, ks_b, ks_h, ks_g, vs_b, vs_h, vs_t, SPLITS,
                HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
                BITS: tl.constexpr, BN: tl.constexpr):
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    n = ctx + M
    rows = tl.arange(0, BM)
    rm = rows // G
    rg = rows % G
    rvalid = rows < G * M
    dd = tl.arange(0, D)
    qrow = QKV + (b * M + rm)[:, None] * sq_r + (h * G + rg)[:, None] * D
    q = tl.load(qrow + dd[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qrow + ((dd + D // 2) % D)[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    f = tl.load(INVF + (dd % (D // 2)))
    ang = (ctx + rm).to(tl.float32)[:, None] * f[None, :]
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    q = ((q * tl.cos(ang) + sgn[None, :] * qp * tl.sin(ang)) * qk_scale).to(tl.bfloat16)
    lim = ctx + rm
    start = sp * chunk
    end = tl.minimum(start + chunk, n)
    W: tl.constexpr = D * BITS // 8
    ww = tl.arange(0, W)
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for t0 in range(start, end, BN):
        offs = t0 + tl.arange(0, BN)
        tm = offs < end
        kx = tl.load(KC + b * c_b + h * c_h + offs[:, None] * c_t + ww[None, :], mask=tm[:, None], other=0)
        grp = t0 // BN
        ksc = tl.load(KS + b * ks_b + h * ks_h + grp * ks_g + dd).to(tl.float32)
        kzr = tl.load(KZ + b * ks_b + h * ks_h + grp * ks_g + dd).to(tl.float32)
        k = (_unpack(kx, BN, D, BITS) * ksc[None, :] + kzr[None, :]).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k))
        s = tl.where((offs[None, :] <= lim[:, None]) & tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vx = tl.load(VC + b * c_b + h * c_h + offs[:, None] * c_t + ww[None, :], mask=tm[:, None], other=0)
        gi = dd // 32
        vsc = tl.load(VS + b * vs_b + h * vs_h + offs[:, None] * vs_t + gi[None, :], mask=tm[:, None], other=0.0)
        vzr = tl.load(VZ + b * vs_b + h * vs_h + offs[:, None] * vs_t + gi[None, :], mask=tm[:, None], other=0.0)
        v = (_unpack(vx, BN, D, BITS) * vsc.to(tl.float32) + vzr.to(tl.float32)).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (bh * SPLITS + sp) * BM + rows
    tl.store(OP + o[:, None] * D + dd[None, :], acc)
    tl.store(MP + o, m_i)
    tl.store(LP + o, l_i)


def kivi_partials(qkv, c: KIVICache, layer, ctx, M, G, inv_freq, ws, splits, chunk, num_warps=2):
    assert chunk % GRP == 0
    OP, MP, LP, BM = ws
    kc, vc, ks, kz, vs, vz = c.kc[layer], c.vc[layer], c.ks[layer], c.kz[layer], c.vs[layer], c.vz[layer]
    _kivi_split[(c.B * c.Hkv, splits)](qkv, kc, vc, ks, kz, vs, vz, inv_freq, OP, MP, LP, ctx, chunk,
                                       c.D ** -0.5 * K.LOG2E, qkv.stride(0), kc.stride(0), kc.stride(1), kc.stride(2),
                                       ks.stride(0), ks.stride(1), ks.stride(2), vs.stride(0), vs.stride(1),
                                       vs.stride(2), splits, HKV=c.Hkv, G=G, M=M, BM=BM, D=c.D, BITS=c.bits, BN=GRP,
                                       num_warps=num_warps, num_stages=1)
    return splits


def kivi_attention(qkv, c, layer, ctx, M, G, inv_freq, out, ws, splits, chunk):
    s = kivi_partials(qkv, c, layer, ctx, M, G, inv_freq, ws, splits, chunk)
    OP, MP, LP, BM = ws
    K._attn_combine[(c.B * c.Hkv, G * M)](OP, MP, LP, out, s, HKV=c.Hkv, G=G, M=M, BM=BM, D=c.D,
                                          SP2=min(32, max(2, triton.next_power_of_2(s))), num_warps=4)
    return out


def kivi_quant_v(v, bits):
    """v [R, H, D] -> codes uint8 [R, H, D*b/8], scale / zero fp16 [R, H, D/32] (per-token, 32-channel groups)."""
    L = 2 ** bits - 1
    g = v.float().unflatten(-1, (-1, GRP))
    lo, hi = g.amin(-1), g.amax(-1)
    s = ((hi - lo) / L).clamp(min=1e-6)
    q = torch.round((g - lo[..., None]) / s[..., None]).clamp(0, L).to(torch.uint8).flatten(-2)
    q = pack(q, bits)
    return q, s.half(), lo.half()


def kivi_quant_k_group(k, bits):
    """k [B, H, 32, D] (one full token group) -> codes [B, H, 32, D*b/8], scale / zero [B, H, D] (per channel)."""
    L = 2 ** bits - 1
    kf = k.float()
    lo, hi = kf.amin(2), kf.amax(2)
    s = ((hi - lo) / L).clamp(min=1e-6)
    q = torch.round((kf - lo[:, :, None]) / s[:, :, None]).clamp(0, L).to(torch.uint8)
    return pack(q, bits), s.half(), lo.half()


def pack(q, bits):
    per = 8 // bits
    q = q.unflatten(-1, (-1, per))
    out = q[..., 0].clone()
    for j in range(1, per):
        out |= q[..., j] << (bits * j)
    return out


@torch.no_grad()
def check(B=2, ctx=700, M=3, bits=2):
    torch.manual_seed(0)
    Hkv, G, D = 8, 4, 128
    inv = K.inv_freq_llama(D)
    T = ((ctx + M + 8 + GRP - 1) // GRP) * GRP
    kvf = torch.randn(2, B, Hkv, T, D, device="cuda")
    c = KIVICache(1, B, Hkv, T, bits)
    # quantize with the reference quantizers into the cache
    for g0 in range(0, T, GRP):
        q, s, z = kivi_quant_k_group(kvf[0, :, :, g0:g0 + GRP], bits)
        c.kc[0][:, :, g0:g0 + GRP], c.ks[0][:, :, g0 // GRP], c.kz[0][:, :, g0 // GRP] = q, s, z
    q, s, z = kivi_quant_v(kvf[1].reshape(-1, T, D), bits)
    c.vc[0].copy_(q.view(B, Hkv, T, -1)); c.vs[0].copy_(s.view(B, Hkv, T, -1)); c.vz[0].copy_(z.view(B, Hkv, T, -1))
    # dequant reference
    L = 2 ** bits - 1

    def unpack_ref(x):
        per = 8 // bits
        return torch.stack([(x >> (bits * j)) & L for j in range(per)], -1).flatten(-2).float()
    Kd = unpack_ref(c.kc[0]) * c.ks[0].float().repeat_interleave(GRP, 2) + c.kz[0].float().repeat_interleave(GRP, 2)
    Vd = (unpack_ref(c.vc[0]).unflatten(-1, (-1, GRP)) * c.vs[0].float()[..., None] + c.vz[0].float()[..., None]).flatten(-2)
    qkv = torch.randn(B * M, (Hkv * G + 2 * Hkv) * D, device="cuda").to(torch.bfloat16)
    out = torch.empty(B * M, Hkv * G * D, device="cuda", dtype=torch.bfloat16)
    sp, ch = K.split_plan(B, Hkv, ctx + M, 64, 256)
    ws = K.attn_workspace(B, Hkv, G, M, sp)
    kivi_attention(qkv, c, 0, ctx, M, G, inv, out, ws, sp, ch)
    ref = K.ref_attention(qkv, Kd.to(torch.bfloat16), Vd.to(torch.bfloat16), ctx, M, Hkv, G, inv)
    return (out.float() - ref.float()).abs().max().item(), ref.float().abs().max().item()
