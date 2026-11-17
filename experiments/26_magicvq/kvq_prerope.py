"""E7 magicvq: pre-RoPE VQ-KV decode attention (Triton split-K), i.e. the kernel the tau eval's VQ format needs.

Same flash-decoding structure as specrq/kvq `_attn_split` MODE 0 (copied, read-only reuse), but the codes hold
channel-STANDARDIZED, PRE-RoPE K (and standardized V) in the Meta RoPE-pair layout (channels (i, i + 64) adjacent):
per key tile  k = dequant(codes) * sd_k + mu_k  ->  RoPE at the key's position (pairs (2i, 2i+1), angle pos * f_i,
computed in-kernel) ; v = dequant(codes) * sd_v + mu_v. q is read from the fused QKV row in HF order, RoPE'd and
permuted to the Meta pair layout. Uniform 8-bit-code stages only (d8 K256 x S, d16 K256 x S).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import specrq.kvq.kernels as K


@triton.jit
def _deq(P, CB, b, h, offs_t, tmask, s_s, s_b, s_h, s_t, scb_h, scb_s, D: tl.constexpr, DSUB: tl.constexpr,
         KST: tl.constexpr):
    jj = tl.arange(0, D)
    base = P + b * s_b + h * s_h + offs_t[:, None] * s_t + (jj // DSUB)[None, :]
    cbb = CB + h * scb_h + (jj % DSUB)[None, :]
    c = tl.load(base, mask=tmask[:, None], other=0).to(tl.int32)
    acc = tl.load(cbb + c * DSUB).to(tl.float32)
    for s in tl.static_range(1, KST):
        c = tl.load(base + s * s_s, mask=tmask[:, None], other=0).to(tl.int32)
        acc += tl.load(cbb + s * scb_s + c * DSUB).to(tl.float32)
    return acc


@triton.jit
def _deq_gs(P, CB, GN, b, h, offs_t, tmask, s_s, s_b, s_h, s_t, scb_h, scb_s, sg_h, sg_s, D: tl.constexpr,
            DSUB: tl.constexpr, KST: tl.constexpr, C: tl.constexpr):
    """GSRQ: per-sub-vector-position codebooks [.., NSV, C, DSUB] of unit shapes x gains [.., NSV, C]."""
    jj = tl.arange(0, D)
    pos = jj // DSUB
    base = P + b * s_b + h * s_h + offs_t[:, None] * s_t + pos[None, :]
    cbb = CB + h * scb_h + (pos * (C * DSUB) + jj % DSUB)[None, :]
    gb = GN + h * sg_h + (pos * C)[None, :]
    c = tl.load(base, mask=tmask[:, None], other=0).to(tl.int32)
    acc = tl.load(cbb + c * DSUB).to(tl.float32) * tl.load(gb + c).to(tl.float32)
    for s in tl.static_range(1, KST):
        c = tl.load(base + s * s_s, mask=tmask[:, None], other=0).to(tl.int32)
        acc += tl.load(cbb + s * scb_s + c * DSUB).to(tl.float32) * tl.load(gb + s * sg_s + c).to(tl.float32)
    return acc


@triton.jit
def _attn_split_pre(QKV, KC, VC, CBK, CBV, MUK, SDK, MUV, SDV, INVF, OP, MP, LP, ctx, chunk, qk_scale,
                    sq_r, s_s, s_b, s_h, s_t, scb_h, scb_s, SPLITS, GNK, GNV, sg_h, sg_s,
                    HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
                    DSUB: tl.constexpr, KST: tl.constexpr, BN: tl.constexpr, GS: tl.constexpr, C: tl.constexpr):
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
    # meta index dd -> HF channel: even 2i -> i, odd 2i+1 -> i + 64 ; frequency index i = dd // 2
    hf = (dd // 2) + (dd % 2) * (D // 2)
    hfp = (hf + D // 2) % D
    fi = tl.load(INVF + dd // 2)
    qrow = QKV + (b * M + rm)[:, None] * sq_r + (h * G + rg)[:, None] * D
    q = tl.load(qrow + hf[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qrow + hfp[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    ang = (ctx + rm).to(tl.float32)[:, None] * fi[None, :]
    sgn = tl.where(dd % 2 == 0, -1.0, 1.0)  # meta-even = HF first half (rotate-half: -x2), odd = second half (+x1)
    q = (q * tl.cos(ang) + sgn[None, :] * qp * tl.sin(ang)) * qk_scale
    q = q.to(tl.bfloat16)
    muk = tl.load(MUK + h * D + dd)
    sdk = tl.load(SDK + h * D + dd)
    muv = tl.load(MUV + h * D + dd)
    sdv = tl.load(SDV + h * D + dd)
    lim = ctx + rm
    start = sp * chunk
    end = tl.minimum(start + chunk, n)
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for t0 in range(start, end, BN):
        offs = t0 + tl.arange(0, BN)
        tm = offs < end
        if GS:
            k = _deq_gs(KC, CBK, GNK, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, sg_h, sg_s, D, DSUB, KST, C)
        else:
            k = _deq(KC, CBK, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, D, DSUB, KST)
            k = k * sdk[None, :] + muk[None, :]
        # RoPE on pairs (2i, 2i+1): partner via reshape/split
        k2 = tl.reshape(k, (BN, D // 2, 2))
        ka, kb = tl.split(k2)
        th = offs.to(tl.float32)[:, None] * tl.load(INVF + tl.arange(0, D // 2))[None, :]
        c, s_ = tl.cos(th), tl.sin(th)
        k = tl.reshape(tl.join(ka * c - kb * s_, kb * c + ka * s_), (BN, D)).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k))
        sc = tl.where((offs[None, :] <= lim[:, None]) & tm[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(sc - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if GS:
            v = _deq_gs(VC, CBV, GNV, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, sg_h, sg_s, D, DSUB, KST,
                        C).to(tl.bfloat16)
        else:
            v = _deq(VC, CBV, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, D, DSUB, KST)
            v = (v * sdv[None, :] + muv[None, :]).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (bh * SPLITS + sp) * BM + rows
    tl.store(OP + o[:, None] * D + dd[None, :], acc)
    tl.store(MP + o, m_i)
    tl.store(LP + o, l_i)


class Affine:
    """per (layer, K/V, kv-head, channel) mean / std in Meta layout, fp32 [Hkv, D] (synthetic or from a codebook file)."""

    def __init__(self, n_layers, Hkv, D=128, device="cuda", gen=None):
        self.mu = [[torch.randn(Hkv, D, device=device, generator=gen) * 0.1 for _ in range(2)] for _ in range(n_layers)]
        self.sd = [[torch.rand(Hkv, D, device=device, generator=gen) + 0.5 for _ in range(2)] for _ in range(n_layers)]


class GSBooks:
    """GSRQ per-sub-vector-position codebooks for a KVCache (codes reused): shapes bf16 [L][2, Hkv, S, NSV, C, d],
    gains fp16 [L][2, Hkv, S, NSV, C] (synthetic values; timing)."""

    def __init__(self, cache, C=256, gen=None):
        L, Hkv, S, nsv, d = cache.n_layers, cache.Hkv, cache.S, cache.nsv, cache.d
        self.C = C
        self.cb = [torch.nn.functional.normalize(torch.randn(2, Hkv, S, nsv, C, d, device="cuda", generator=gen), dim=-1)
                   .to(torch.bfloat16) for _ in range(L)]
        self.g = [(torch.rand(2, Hkv, S, nsv, C, device="cuda", generator=gen) + 0.1).half() for _ in range(L)]


def prerope_partials(qkv, cache, aff, layer, ctx, M, G, inv_freq, ws, splits, chunk, kst, BN=64, num_warps=4):
    """writes split-K partials (meta-layout output channels) into ws; returns the split count. aff: Affine
    (standardized nested VQ) or GSBooks (GSRQ: per-position shape codebooks x gains, no standardization)."""
    assert cache.mode == "vq" and cache.uniform
    B, Hkv, D = cache.B, cache.Hkv, cache.D
    OP, MP, LP, BM = ws
    Kc, Vc = cache.codes[layer][0], cache.codes[layer][1]
    if isinstance(aff, GSBooks):
        cb, gn = aff.cb[layer], aff.g[layer]
        CBK, CBV, GNK, GNV = cb[0], cb[1], gn[0], gn[1]
        mu = sd = torch.empty(1, device=qkv.device)
        args = (mu, sd, mu, sd)
        gs, C = True, aff.C
        sg_h, sg_s = GNK.stride(0), GNK.stride(1)
    else:
        CBK, CBV = cache.cb[layer][0], cache.cb[layer][1]
        GNK = GNV = CBK
        args = (aff.mu[layer][0], aff.sd[layer][0], aff.mu[layer][1], aff.sd[layer][1])
        gs, C, sg_h, sg_s = False, 256, 0, 0
    _attn_split_pre[(B * Hkv, splits)](qkv, Kc, Vc, CBK, CBV, *args, inv_freq, OP, MP, LP, ctx, chunk,
                                       D ** -0.5 * K.LOG2E,
                                       qkv.stride(0), Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3),
                                       CBK.stride(0), CBK.stride(1), splits, GNK, GNV, sg_h, sg_s,
                                       HKV=Hkv, G=G, M=M, BM=BM, D=D,
                                       DSUB=cache.d, KST=kst, BN=BN, GS=gs, C=C, num_warps=num_warps, num_stages=1)
    return splits


def gs_encode_step(x, books, layer, kv, kst, codes_out):
    """Unfused torch GSRQ encoder of R new rows (timing of the per-token encode cost): x [R, Hkv*128] (meta layout,
    pre-RoPE) -> codes [P, R] per stage handed to codes_out(stage, codes) (greedy residual, argmax 2 g x.s - g^2)."""
    cb, gn = books.cb[layer][kv], books.g[layer][kv]  # [Hkv, S, nsv, C, d], [Hkv, S, nsv, C]
    Hkv, S, nsv, C, d = cb.shape
    r = x.float().view(-1, Hkv * nsv, d).transpose(0, 1)  # [P, R, d]
    for s in range(kst):
        Sh = cb[:, s].reshape(Hkv * nsv, C, d).float()
        g = gn[:, s].reshape(Hkv * nsv, C).float()
        sc = 2 * torch.bmm(r, Sh.transpose(1, 2)) * g[:, None] - (g * g)[:, None]
        c = sc.argmax(-1)  # [P, R]
        codes_out(s, c)
        r = r - torch.gather(Sh, 1, c[..., None].expand(-1, -1, d)) * torch.gather(g, 1, c)[..., None]


def prerope_attention(qkv, cache, aff, layer, ctx, M, G, inv_freq, out, ws, splits, chunk, kst):
    """Output channels are in Meta layout for V (the following o_proj columns can be permuted offline; timing-equal)."""
    s = prerope_partials(qkv, cache, aff, layer, ctx, M, G, inv_freq, ws, splits, chunk, kst)
    OP, MP, LP, BM = ws
    K._attn_combine[(cache.B * cache.Hkv, G * M)](OP, MP, LP, out, s, HKV=cache.Hkv, G=G, M=M, BM=BM, D=cache.D,
                                                  SP2=min(32, max(2, triton.next_power_of_2(s))), num_warps=4)
    return out


@torch.no_grad()
def check(B=2, ctx=700, M=3, kst=2, d=8):
    torch.manual_seed(0)
    Hkv, G, D = 8, 4, 128
    inv = K.inv_freq_llama(D)
    vqc = K.KVCache(1, B, Hkv, ctx + M + 8, mode="vq", d=d, bits=[8, 8])
    aff = Affine(1, Hkv, D)
    qkv = torch.randn(B * M, (Hkv * G + 2 * Hkv) * D, device="cuda").to(torch.bfloat16)
    out = torch.empty(B * M, Hkv * G * D, device="cuda", dtype=torch.bfloat16)
    sp, ch = K.split_plan(B, Hkv, ctx + M, 64, 256)
    ws = K.attn_workspace(B, Hkv, G, M, sp)
    prerope_attention(qkv, vqc, aff, 0, ctx, M, G, inv, out, ws, sp, ch, kst)
    # reference
    Kq, Vq = K.ref_dequant_kv(vqc, 0, kst)  # bf16 [B,Hkv,T,D] (meta layout, standardized)
    Kf = Kq.float() * aff.sd[0][0][None, :, None] + aff.mu[0][0][None, :, None]
    Vf = Vq.float() * aff.sd[0][1][None, :, None] + aff.mu[0][1][None, :, None]
    T = Kf.shape[2]
    hf_from_meta = lambda x: x.unflatten(-1, (64, 2)).transpose(-1, -2).flatten(-2)  # noqa: E731
    Kh = K.ref_rope(hf_from_meta(Kf), torch.arange(T, device="cuda")[None, None], inv)  # HF layout, RoPE'd
    q = qkv[:, :Hkv * G * D].view(B, M, Hkv, G, D).float()
    ref = torch.empty(B, M, Hkv, G, D, device="cuda")
    for b in range(B):
        for m in range(M):
            qr = K.ref_rope(q[b, m], torch.tensor(ctx + m, device="cuda"), inv)
            s = torch.einsum("hgd,htd->hgt", qr, Kh[b, :, :ctx + m + 1]) / D ** 0.5
            ref[b, m] = torch.einsum("hgt,htd->hgd", s.softmax(-1), Vf[b, :, :ctx + m + 1])  # meta-layout channels
    err = (out.float().view_as(ref) - ref).abs().max().item()
    return err, ref.abs().max().item()


@torch.no_grad()
def check_gs(B=2, ctx=500, M=3, kst=2, d=8):
    torch.manual_seed(0)
    Hkv, G, D = 8, 4, 128
    inv = K.inv_freq_llama(D)
    vqc = K.KVCache(1, B, Hkv, ctx + M + 8, mode="vq", d=d, bits=[8, 8])
    bk = GSBooks(vqc)
    qkv = torch.randn(B * M, (Hkv * G + 2 * Hkv) * D, device="cuda").to(torch.bfloat16)
    out = torch.empty(B * M, Hkv * G * D, device="cuda", dtype=torch.bfloat16)
    sp, ch = K.split_plan(B, Hkv, ctx + M, 64, 256)
    ws = K.attn_workspace(B, Hkv, G, M, sp)
    prerope_attention(qkv, vqc, bk, 0, ctx, M, G, inv, out, ws, sp, ch, kst)
    codes = vqc.codes[0]  # [2, S, B, H, T, nsv]
    rec = []
    for kv in range(2):
        acc = 0
        for s in range(kst):
            c = codes[kv, s].long()  # [B,H,T,nsv]
            cb = bk.cb[0][kv, :, s].float()  # [H, nsv, C, d]
            g = bk.g[0][kv, :, s].float()  # [H, nsv, C]
            Hh = torch.arange(Hkv, device="cuda")[None, :, None, None]
            pp = torch.arange(c.shape[-1], device="cuda")[None, None, None, :]
            acc = acc + (cb[Hh, pp, c] * g[Hh, pp, c][..., None]).to(torch.bfloat16).float()
        rec.append(acc.flatten(-2))  # [B,H,T,128] meta
    Kf, Vf = rec
    T = Kf.shape[2]
    hf_from_meta = lambda x: x.unflatten(-1, (64, 2)).transpose(-1, -2).flatten(-2)  # noqa: E731
    Kh = K.ref_rope(hf_from_meta(Kf), torch.arange(T, device="cuda")[None, None], inv)
    q = qkv[:, :Hkv * G * D].view(B, M, Hkv, G, D).float()
    ref = torch.empty(B, M, Hkv, G, D, device="cuda")
    for b in range(B):
        for m in range(M):
            qr = K.ref_rope(q[b, m], torch.tensor(ctx + m, device="cuda"), inv)
            s_ = torch.einsum("hgd,htd->hgt", qr, Kh[b, :, :ctx + m + 1]) / D ** 0.5
            ref[b, m] = torch.einsum("hgt,htd->hgd", s_.softmax(-1), Vf[b, :, :ctx + m + 1])
    return (out.float().view_as(ref) - ref).abs().max().item(), ref.abs().max().item()
