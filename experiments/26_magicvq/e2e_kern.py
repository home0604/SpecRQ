"""E7 magicvq v5 (fully real e2e): ragged, CUDA-graph-safe attention / cache kernels for the real decode loop.

All lengths / positions are int32 DEVICE tensors (per sequence or per (sequence, kv-head)), so one captured graph
serves every round while caches grow and sequences advance by different accepted counts.

  bf_partial : split-K flash-decoding partials over a bf16 cache [B, Hkv, T, D] (post-RoPE K, V), keys = an optional
               per-(b, h) index list IDX (e.g. SnapKV-selected positions, attention sinks) followed by the contiguous
               position range [LO[b], HI[b]); causal among the M new rows (key pos <= QPOS[b] + m); q RoPE fused at
               QPOS[b] + m. Same structure as specrq/kvq _attn_split MODE 1 (the 'triton' backend of v3/v4b timing).
  lut_rag    : VQ partials (CUDA, csrc/vqattn_lut_rag.cu: ragged copy of the kvq LUT kernel, chnorm folded).
  combine    : log-sum-exp merge of all SPT partial splits of one (b, kv-head, row) (the bf16 and VQ partial kernels
               write disjoint split ranges of one workspace).
  append_kv  : RoPE + write of the M new rows' K / V into the bf16 cache at QPOS[b] + m.
  rq_encode  : greedy residual VQ encode (fp32, exact argmin) of cached post-RoPE K / V (static per-channel
               standardization, calib --chnorm) at source positions SRC[b, h, j] (compact mode) or LO[b] + j (range
               mode), codes written to planes [2, S, B, Hkv, T, 16] at j (compact) or the source position (range).
specrq/kvq is not modified.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _bf_partial(QKV, KC, VC, INVF, OP, MP, LP, IDX, NIDX, LO, HI, QPOS, chunk, qk_scale, sq_r, s_b, s_h, s_t,
                i_bh, SPT, OFF, HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr,
                D: tl.constexpr, BN: tl.constexpr, HAS_IDX: tl.constexpr):
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    if HAS_IDX:
        nidx = tl.load(NIDX + bh)
    else:
        nidx = 0
    lo = tl.load(LO + b)
    hi = tl.load(HI + b)
    qpos = tl.load(QPOS + b)
    nv = nidx + hi - lo
    rows = tl.arange(0, BM)
    rm = rows // G
    rg = rows % G
    rvalid = rows < G * M
    dd = tl.arange(0, D)
    qrow = QKV + (b * M + rm)[:, None] * sq_r + (h * G + rg)[:, None] * D
    q = tl.load(qrow + dd[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qrow + ((dd + D // 2) % D)[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    f = tl.load(INVF + (dd % (D // 2)))
    ang = (qpos + rm).to(tl.float32)[:, None] * f[None, :]
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    q = ((q * tl.cos(ang) + sgn[None, :] * qp * tl.sin(ang)) * qk_scale).to(tl.bfloat16)
    lim = qpos + rm
    start = sp * chunk
    end = tl.minimum(start + chunk, nv)
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    kb = KC + b * s_b + h * s_h
    vb = VC + b * s_b + h * s_h
    for t0 in range(start, end, BN):
        v = t0 + tl.arange(0, BN)
        vm = v < end
        if HAS_IDX:
            ii = tl.load(IDX + bh * i_bh + v, mask=vm & (v < nidx), other=0)
            pos = tl.where(v < nidx, ii, lo + v - nidx)
        else:
            pos = lo + v
        k = tl.load(kb + pos[:, None] * s_t + dd[None, :], mask=vm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = tl.where((pos[None, :] <= lim[:, None]) & vm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        vv = tl.load(vb + pos[:, None] * s_t + dd[None, :], mask=vm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vv)
        m_i = m_new
    o = (bh * SPT + OFF + sp) * BM + rows
    tl.store(OP + o[:, None] * D + dd[None, :], acc)
    tl.store(MP + o, m_i)
    tl.store(LP + o, l_i)


@triton.jit
def _combine(OP, MP, LP, OUT, SPT, HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr,
             D: tl.constexpr, SP2: tl.constexpr):
    bh = tl.program_id(0)
    r = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    dd = tl.arange(0, D)
    mx = tl.max(tl.full([SP2], float("-inf"), tl.float32), 0)
    for c0 in range(0, SPT, SP2):
        sp = c0 + tl.arange(0, SP2)
        m = tl.load(MP + (bh * SPT + sp) * BM + r, mask=sp < SPT, other=float("-inf"))
        mx = tl.maximum(mx, tl.max(m, 0))
    acc = tl.zeros([D], dtype=tl.float32)
    den = tl.sum(tl.zeros([SP2], dtype=tl.float32), 0)
    for c0 in range(0, SPT, SP2):
        sp = c0 + tl.arange(0, SP2)
        spm = sp < SPT
        o = (bh * SPT + sp) * BM + r
        m = tl.load(MP + o, mask=spm, other=float("-inf"))
        w = tl.where(spm & (m > float("-inf")), tl.exp2(m - mx), 0.0)
        acc += tl.sum(tl.load(OP + o[:, None] * D + dd[None, :], mask=spm[:, None], other=0.0) * w[:, None], 0)
        den += tl.sum(tl.load(LP + o, mask=spm, other=0.0) * w, 0)
    rm = r // G
    rg = r % G
    tl.store(OUT + (b * M + rm) * (HKV * G * D) + (h * G + rg) * D + dd, (acc / den).to(tl.bfloat16))


@triton.jit
def _append(QKV, KVC, INVF, QPOS, sq_r, s_kv, s_b, s_h, s_t, HKV: tl.constexpr, HQ: tl.constexpr, M: tl.constexpr,
            D: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)
    b = r // M
    pos = tl.load(QPOS + b) + r % M
    dd = tl.arange(0, D)
    src = QKV + r * sq_r + (HQ + kv * HKV + h) * D
    x = tl.load(src + dd).to(tl.float32)
    xp = tl.load(src + (dd + D // 2) % D).to(tl.float32)
    ang = pos.to(tl.float32) * tl.load(INVF + dd % (D // 2))
    isk = (kv == 0).to(tl.float32)
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    x = x * (isk * tl.cos(ang) + (1.0 - isk)) + isk * sgn * xp * tl.sin(ang)
    tl.store(KVC + kv * s_kv + b * s_b + h * s_h + pos * s_t + dd, x.to(tl.bfloat16))


@triton.jit
def _rq_encode(KVC, CODES, CB, MU, SD, SRC, LO, CNT, NJ, s_kv, s_b, s_h, s_t, src_bh, c_kv, c_s, c_b, c_h, c_t,
               cb_kv, cb_h, cb_s, mu_kv, HKV: tl.constexpr, D: tl.constexpr, DSUB: tl.constexpr, NSV: tl.constexpr,
               C: tl.constexpr, S: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr, COMPACT: tl.constexpr):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)
    b = pid // NJ
    j = pid % NJ
    if j < tl.load(CNT + b * HKV + h):
        if COMPACT:
            pos = tl.load(SRC + (b * HKV + h) * src_bh + j)
            dpos = j
        else:
            pos = tl.load(LO + b) + j
            dpos = pos
        ri = tl.arange(0, XR)
        ci = tl.arange(0, XC)
        xm = (ri[:, None] < NSV) & (ci[None, :] < DSUB)
        idx = ri[:, None] * DSUB + ci[None, :]
        x = tl.load(KVC + kv * s_kv + b * s_b + h * s_h + pos * s_t + idx, mask=xm, other=0.0).to(tl.float32)
        mu = tl.load(MU + kv * mu_kv + h * D + idx, mask=xm, other=0.0)
        sd = tl.load(SD + kv * mu_kv + h * D + idx, mask=xm, other=1.0)
        x = tl.where(xm, (x - mu) / sd, 0.0)
        cbase = CB + kv * cb_kv + h * cb_h
        cidx = tl.arange(0, C)
        dst = CODES + kv * c_kv + b * c_b + h * c_h + dpos * c_t + ri
        for s in tl.static_range(S):
            cbp = cbase + s * cb_s
            cbt = tl.load(cbp + cidx[None, :] * DSUB + ci[:, None], mask=(ci[:, None] < DSUB), other=0.0)
            cn = tl.sum(cbt * cbt, 0)
            dots = tl.dot(x, cbt, input_precision="ieee")
            code = tl.argmin(cn[None, :] - 2.0 * dots, 1)
            sel = tl.load(cbp + code[:, None] * DSUB + ci[None, :], mask=xm, other=0.0)
            x = x - sel
            tl.store(dst + s * c_s, code.to(tl.uint8), mask=ri < NSV)


# ------------------------------------------------------------------------------------------------ python wrappers
def workspace(B, Hkv, G, M, spt, D=128, device="cuda"):
    BM = max(16, triton.next_power_of_2(G * M))
    n = B * Hkv * spt * BM
    return (torch.empty(n, D, dtype=torch.float32, device=device), torch.empty(n, dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device), BM)


def bf_partial(qkv, kc, vc, inv, ws, spt, off, splits, chunk, lo, hi, qpos, M, G, idx=None, nidx=None, BN=64):
    """kc / vc: [B, Hkv, T, D] bf16 views; lo / hi / qpos int32 [B]; idx int32 [B, Hkv, n] + nidx int32 [B*Hkv]."""
    B, Hkv, T, D = kc.shape
    OP, MP, LP, BM = ws
    has = idx is not None
    _bf_partial[(B * Hkv, splits)](qkv, kc, vc, inv, OP, MP, LP, idx if has else lo, nidx if has else lo, lo, hi, qpos,
                                   chunk, D ** -0.5 * LOG2E, qkv.stride(0), kc.stride(0), kc.stride(1), kc.stride(2),
                                   idx.stride(1) if has else 0, spt, off, HKV=Hkv, G=G, M=M, BM=BM, D=D, BN=BN,
                                   HAS_IDX=has, num_warps=4, num_stages=1)


def combine(ws, spt, out, B, Hkv, G, M, D=128):
    OP, MP, LP, BM = ws
    _combine[(B * Hkv, G * M)](OP, MP, LP, out, spt, HKV=Hkv, G=G, M=M, BM=BM, D=D,
                               SP2=min(32, max(2, triton.next_power_of_2(spt))), num_warps=4)
    return out


def append_kv(qkv, kvc, inv, qpos, M, Hq):
    """kvc: [2, B, Hkv, T, D] bf16 (one layer)."""
    _, B, Hkv, T, D = kvc.shape
    _append[(qkv.shape[0], Hkv, 2)](qkv, kvc, inv, qpos, qkv.stride(0), kvc.stride(0), kvc.stride(1), kvc.stride(2),
                                    kvc.stride(3),
                                    HKV=Hkv, HQ=Hq, M=M, D=D, num_warps=1)


def rq_encode(kvc, codes, cb, mu, sd, cnt, nj, src=None, lo=None):
    """kvc [2, B, Hkv, T, D] bf16 (one layer); codes [2, S, B, Hkv, Tc, 16] uint8; cb fp32 [2, Hkv, S, 256, 8];
    mu / sd fp32 [2, Hkv, 128]; cnt int32 [B*Hkv]; compact (src int32 [B, Hkv, n]) or range (lo int32 [B])."""
    _, B, Hkv, T, D = kvc.shape
    S = codes.shape[1]
    comp = src is not None
    _rq_encode[(B * nj, Hkv, 2)](kvc, codes, cb, mu, sd, src if comp else cnt, lo if lo is not None else cnt, cnt,
                                 nj, kvc.stride(0), kvc.stride(1), kvc.stride(2), kvc.stride(3),
                                 src.stride(1) if comp else 0,
                                 codes.stride(0), codes.stride(1), codes.stride(2), codes.stride(3), codes.stride(4),
                                 cb.stride(0), cb.stride(1), cb.stride(2), mu.stride(0), HKV=Hkv, D=D, DSUB=8, NSV=16,
                                 C=256, S=S, XR=16, XC=16, COMPACT=comp, num_warps=4)


_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("CC", "/usr/bin/gcc-11")
        os.environ.setdefault("CXX", "/usr/bin/g++-11")
        os.environ.setdefault("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11")
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _EXT = load(name="mv_lut_rag", sources=[str(Path(__file__).parent / "csrc" / "vqattn_lut_rag.cu")],
                    extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _EXT


def lut_partial(qkv, planes_k, planes_v, cbk, cbv, inv, ws, spt, off, splits, chunk, lo, hi, qpos, sdk, muk, sdv, muv,
                B, Hkv, kst):
    """VQ partials: writes splits * 16 slots at `off`. planes [B, Hkv, T, 16]; cb bf16 [Hkv, 256, 8] per stage;
    lo / hi int32 [B*Hkv] (token index range of the planes); qpos int32 [B]; sd / mu fp32 [Hkv, 128]."""
    OP, MP, LP, BM = ws
    ext().lut_rag(qkv, planes_k, planes_v, cbk, cbv, inv, OP, MP, LP, lo, hi, qpos, sdk, muk, sdv, muv, chunk, splits,
                  B, Hkv, BM, spt, off, kst, D_SCALE)


D_SCALE = 128 ** -0.5 * LOG2E
