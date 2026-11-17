"""E7 magicvq: hybrid draft attention = VQ part (specrq/kvq kernels, whole context) + bf16 part (16 sinks + recent
window of R tokens, kvq Triton split-K kernel in bf16 mode), merged by a two-set split-K log-sum-exp combine.

specrq/kvq is used read-only: vq_attention() is called with its final _attn_combine launch intercepted (the partials
O / m / l and the effective split count stay in the workspace), then the window partials are written to a second
workspace and _combine2 merges both. Timing harness use: the window kernel's q-RoPE angle uses the window-local
position (kernel has no position-offset argument); the cost is identical to a correct offset (numerics not checked).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

import specrq.kvq.kernels as K


class _Grab:
    splits = None

    def __getitem__(self, grid):
        def f(OP, MP, LP, out, splits, **kw):
            self.splits = splits
        return f


@triton.jit
def _combine2(OP1, MP1, LP1, S1, OP2, MP2, LP2, S2, OUT, HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr,
              BM1: tl.constexpr, BM2: tl.constexpr, D: tl.constexpr, SP2: tl.constexpr):
    bh = tl.program_id(0)
    r = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    dd = tl.arange(0, D)
    mx = tl.max(tl.full([SP2], float("-inf"), tl.float32), 0)
    for c0 in range(0, S1, SP2):
        sp = c0 + tl.arange(0, SP2)
        mx = tl.maximum(mx, tl.max(tl.load(MP1 + (bh * S1 + sp) * BM1 + r, mask=sp < S1, other=float("-inf")), 0))
    for c0 in range(0, S2, SP2):
        sp = c0 + tl.arange(0, SP2)
        mx = tl.maximum(mx, tl.max(tl.load(MP2 + (bh * S2 + sp) * BM2 + r, mask=sp < S2, other=float("-inf")), 0))
    acc = tl.zeros([D], dtype=tl.float32)
    den = tl.sum(tl.zeros([SP2], dtype=tl.float32), 0)
    for c0 in range(0, S1, SP2):
        sp = c0 + tl.arange(0, SP2)
        spm = sp < S1
        o = (bh * S1 + sp) * BM1 + r
        m = tl.load(MP1 + o, mask=spm, other=float("-inf"))
        w = tl.where(spm & (m > float("-inf")), tl.exp2(m - mx), 0.0)
        acc += tl.sum(tl.load(OP1 + o[:, None] * D + dd[None, :], mask=spm[:, None], other=0.0) * w[:, None], 0)
        den += tl.sum(tl.load(LP1 + o, mask=spm, other=0.0) * w, 0)
    for c0 in range(0, S2, SP2):
        sp = c0 + tl.arange(0, SP2)
        spm = sp < S2
        o = (bh * S2 + sp) * BM2 + r
        m = tl.load(MP2 + o, mask=spm, other=float("-inf"))
        w = tl.where(spm & (m > float("-inf")), tl.exp2(m - mx), 0.0)
        acc += tl.sum(tl.load(OP2 + o[:, None] * D + dd[None, :], mask=spm[:, None], other=0.0) * w[:, None], 0)
        den += tl.sum(tl.load(LP2 + o, mask=spm, other=0.0) * w, 0)
    rm = r // G
    rg = r % G
    tl.store(OUT + (b * M + rm) * (HKV * G * D) + (h * G + rg) * D + dd, (acc / den).to(tl.bfloat16))


def hybrid_attention(qkv, vqc, wc, layer, ctx, wlen, M, G, inv_freq, out, ws1, ws2, splits, chunk, wsplits, wchunk,
                     kst, backend, aff=None):
    """vqc: KVCache vq (whole context, positions [0, ctx+M)); wc: KVCache bf16 window (wlen + M tokens).
    backend 'pre': pre-RoPE standardized VQ part (kvq_prerope, aff = Affine)."""
    if backend == "pre":
        from kvq_prerope import prerope_partials
        s1 = prerope_partials(qkv, vqc, aff, layer, ctx, M, G, inv_freq, ws1, splits, chunk, kst)
    elif backend == "kivi":
        from kivi_attn import kivi_partials
        s1 = kivi_partials(qkv, vqc, layer, ctx, M, G, inv_freq, ws1, splits, chunk)
    else:
        grab = _Grab()
        orig = K._attn_combine
        K._attn_combine = grab
        try:
            K.vq_attention(qkv, vqc, layer, ctx, M, G, inv_freq, out, ws1, splits, chunk, kst, backend=backend)
        finally:
            K._attn_combine = orig
        s1 = grab.splits
    B, Hkv, D = wc.B, wc.Hkv, wc.D
    Kw, Vw = wc.data[layer][0], wc.data[layer][1]
    OP2, MP2, LP2, BM2 = ws2
    K._attn_split[(B * Hkv, wsplits)](qkv, Kw, Vw, Kw, Vw, Kw, Vw, inv_freq, OP2, MP2, LP2, wlen, wchunk,
                                      D ** -0.5 * K.LOG2E, qkv.stride(0), 0, Kw.stride(0), Kw.stride(1), Kw.stride(2),
                                      0, 0, 0, 0, wsplits, HKV=Hkv, G=G, M=M, BM=BM2, D=D, DSUB=8, NSV=D // 8, KST=1,
                                      BN=64, MODE=1, num_warps=4, num_stages=1)
    OP1, MP1, LP1, BM1 = ws1
    _combine2[(B * Hkv, G * M)](OP1, MP1, LP1, s1, OP2, MP2, LP2, wsplits, out, HKV=Hkv, G=G, M=M, BM1=BM1,
                                BM2=BM2, D=D, SP2=min(32, max(2, triton.next_power_of_2(max(s1, wsplits)))),
                                num_warps=4)
    return out


@torch.no_grad()
def check(B=2, ctx=1000, wlen=80, M=3, kst=2, backend="mmaw"):
    """Numerical check of the merge: hybrid(VQ ctx, window) vs a reference softmax over the concatenated keys,
    using a window cache filled with the reference RoPE at window-local positions (matches the kernel's angle)."""
    torch.manual_seed(0)
    Hkv, G_, D = 8, 4, 128
    inv = K.inv_freq_llama(D)
    vqc = K.KVCache(1, B, Hkv, ctx + M + 8, mode="vq", d=8, bits=[8, 8])
    wc = K.KVCache(1, B, Hkv, wlen + M + 8, mode="bf16")
    qkv = torch.randn(B * M, (Hkv * G_ + 2 * Hkv) * D, device="cuda").to(torch.bfloat16)
    out = torch.empty(B * M, Hkv * G_ * D, device="cuda", dtype=torch.bfloat16)
    sp, ch = K.split_plan(B, Hkv, ctx + M, 128, 256)
    ws1 = K.attn_workspace(B, Hkv, G_, M, sp * 16)
    wsp, wch = K.split_plan(B, Hkv, wlen + M, 64, 64)
    ws2 = K.attn_workspace(B, Hkv, G_, M, wsp)
    hybrid_attention(qkv, vqc, wc, 0, ctx, wlen, M, G_, inv, out, ws1, ws2, sp, ch, wsp, wch, kst, backend)
    # reference: VQ part (dequantized, includes the new rows' positions [0, ctx+M)) + window part [0, wlen+M)
    Kq, Vq = K.ref_dequant_kv(vqc, 0, kst)
    q = qkv[:, :Hkv * G_ * D].view(B, M, Hkv, G_, D).float()
    ref = torch.empty_like(out).view(B, M, Hkv, G_, D)
    for b in range(B):
        for m in range(M):
            qa = K.ref_rope(q[b, m].reshape(-1, D), torch.tensor([ctx + m], device="cuda"), inv).view(Hkv, G_, D)
            qw = K.ref_rope(q[b, m].reshape(-1, D), torch.tensor([wlen + m], device="cuda"), inv).view(Hkv, G_, D)
            ka, va = Kq[b, :, :ctx + m + 1].float(), Vq[b, :, :ctx + m + 1].float()
            kw, vw = wc.data[0][0][b, :, :wlen + m + 1].float(), wc.data[0][1][b, :, :wlen + m + 1].float()
            sa = torch.einsum("hgd,htd->hgt", qa.bfloat16().float(), ka) / D ** 0.5
            sw = torch.einsum("hgd,htd->hgt", qw.bfloat16().float(), kw) / D ** 0.5
            p = torch.cat([sa, sw], -1).softmax(-1)
            ref[b, m] = (torch.einsum("hgt,htd->hgd", p[..., :sa.shape[-1]], va)
                         + torch.einsum("hgt,htd->hgd", p[..., sa.shape[-1]:], vw)).bfloat16()
    err = (out.float() - ref.view_as(out).float()).abs().max().item()
    return err, ref.float().abs().max().item()
