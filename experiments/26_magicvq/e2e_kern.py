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
def _combine_ns(OP, MP, LP, OUT, SPT, NS, HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr,
                D: tl.constexpr, SP2: tl.constexpr):
    """_combine over the first NS of SPT slots (slot stride SPT); MV_LUT_VARIANT (p3) path only."""
    bh = tl.program_id(0)
    r = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    dd = tl.arange(0, D)
    mx = tl.max(tl.full([SP2], float("-inf"), tl.float32), 0)
    for c0 in range(0, NS, SP2):
        sp = c0 + tl.arange(0, SP2)
        m = tl.load(MP + (bh * SPT + sp) * BM + r, mask=sp < NS, other=float("-inf"))
        mx = tl.maximum(mx, tl.max(m, 0))
    acc = tl.zeros([D], dtype=tl.float32)
    den = tl.sum(tl.zeros([SP2], dtype=tl.float32), 0)
    for c0 in range(0, NS, SP2):
        sp = c0 + tl.arange(0, SP2)
        spm = sp < NS
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
class _WS(tuple):
    """workspace tuple (OP, MP, LP, BM) used only when MV_LUT_VARIANT is set: `ns` = number of slots in use per
    (b, kv-head) (set by lut_partial for the p3 one-slot-per-split contract), None = all SPT."""
    ns = None


def workspace(B, Hkv, G, M, spt, D=128, device="cuda"):
    BM = max(16, triton.next_power_of_2(G * M))
    n = B * Hkv * spt * BM
    ws = (torch.empty(n, D, dtype=torch.float32, device=device), torch.empty(n, dtype=torch.float32, device=device),
          torch.empty(n, dtype=torch.float32, device=device), BM)
    return _WS(ws) if LUT_VARIANT else ws


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
    ns = getattr(ws, "ns", None)
    if ns is not None and ns != spt:  # MV_LUT_VARIANT p3: only slots [0, ns) of each (b, kv-head) are written
        _combine_ns[(B * Hkv, G * M)](OP, MP, LP, out, spt, ns, HKV=Hkv, G=G, M=M, BM=BM, D=D,
                                      SP2=min(32, max(2, triton.next_power_of_2(ns))), num_warps=4)
        return out
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


# E10 phase 1 lazy online max in the VQ LUT kernel (exact; rescale only when a score exceeds the reference by > 8).
# Off by default (original online-max path); e2e_real --lazy_max or MV_LUT_LAZY=1 turns it on.
LAZY_MAX = os.environ.get("MV_LUT_LAZY", "0") == "1"


def lut_partial(qkv, planes_k, planes_v, cbk, cbv, inv, ws, spt, off, splits, chunk, lo, hi, qpos, sdk, muk, sdv, muv,
                B, Hkv, kst, lazy=None):
    """VQ partials: writes splits * 16 slots at `off`. planes [B, Hkv, T, 16]; cb bf16 [Hkv, 256, 8] per stage;
    lo / hi int32 [B*Hkv] (token index range of the planes); qpos int32 [B]; sd / mu fp32 [Hkv, 128]."""
    OP, MP, LP, BM = ws
    if LUT_VARIANT:
        # p3: one partial slot per split at off + sp. The VQ range must be the last slot range of the workspace
        # (e2e_real Attn: bf16 [0, bs), VQ [bs, bs + 16 vs)); the combine then reads only [0, off + splits).
        assert off + 16 * splits == spt or off + splits == spt, (off, splits, spt)
        assert isinstance(ws, _WS), "workspace() was created before MV_LUT_VARIANT was set"
        ws.ns = off + splits
        p3_ext().lut_p3(_p3_vid(kst), qkv, planes_k, planes_v, cbk, cbv, inv, OP, MP, LP, lo, hi, qpos, sdk, muk, sdv,
                        muv, chunk, splits, B, Hkv, BM, spt, off, D_SCALE)
        return
    ext().lut_rag(qkv, planes_k, planes_v, cbk, cbv, inv, OP, MP, LP, lo, hi, qpos, sdk, muk, sdv, muv, chunk, splits,
                  B, Hkv, BM, spt, off, kst, D_SCALE, int(LAZY_MAX if lazy is None else lazy))


D_SCALE = 128 ** -0.5 * LOG2E

# ---------------------------------------------------------------------------------------------------------------
# E10 phase 3 -> e2e (opt-in): MV_LUT_VARIANT selects a phase-3 LUT kernel variant (experiments/32_kernel_tiling/
# csrc/lut_rag_p3.cuh via csrc/vqattn_lut_p3*.cu, extension mv_lut_p3_sm80_89) for the VQ partials. Unset / "" / "0"
# = the original kernel (lut_rag) and combine, unchanged. Values:
#   a100 | 4090      presets (A100 kernel-only best: VM1 tensor-core P.V with 8 V-codebook replicas; 4090 best)
#   auto             preset by device capability (8.0 -> a100, 8.9 -> 4090)
#   p3:k1=<v>,k2=<v> explicit, <v> = w{NW}_v{VM}_r{VREP}_h{HS} (must be in csrc/vqattn_lut_p3_vars.h)
# If the chosen variant is not launchable on the device (smem; e.g. VREP 8 at KST 2 on the 4090, ~141 KB), the first
# launchable entry of _P3_FALLBACK[kst] is used and logged. Resolved choice: LUT_VARIANT_INFO (per kst).
# p3 kernels always use the lazy online max (LAZY_MAX / --lazy_max are ignored on this path).
_P3_PRESETS = {"a100": {1: "w8_v1_r8_h0", 2: "w8_v1_r8_h0"}, "4090": {1: "w8_v1_r1_h1", 2: "w8_v2_r1_h0"}}
_P3_FALLBACK = {1: ["w8_v1_r8_h0", "w8_v1_r4_h0", "w8_v1_r2_h1", "w8_v1_r1_h1", "w8_v1_r1_h0"],
                2: ["w8_v1_r8_h0", "w8_v1_r4_h0", "w8_v1_r2_h0", "w8_v2_r1_h0", "w8_v1_r1_h0"]}
LUT_VARIANT = os.environ.get("MV_LUT_VARIANT", "").strip()
LUT_VARIANT = None if LUT_VARIANT in ("", "0", "off", "none") else LUT_VARIANT
LUT_VARIANT_INFO = {}  # kst -> {spec, requested, used, fallback, device, cc, occupancy}
_P3 = None
_P3_VID = {}
_P3_LOGGED = set()


def p3_ext():
    global _P3
    if _P3 is None:
        from torch.utils.cpp_extension import load
        for k, v in (("CC", "/usr/bin/gcc-11"), ("CXX", "/usr/bin/g++-11"),
                     ("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11"), ("CUDA_HOME", "/usr/local/cuda-12.1")):
            if Path(v.split()[-1]).exists():
                os.environ.setdefault(k, v)
        here = Path(__file__).resolve().parent
        old = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.9"  # A100 + 4090 in one binary (VM 1 needs sm_80+)
        try:
            _P3 = load(name="mv_lut_p3_sm80_89",
                       sources=[str(here / "csrc" / f) for f in ("vqattn_lut_p3.cu", "vqattn_lut_p3_k1.cu",
                                                                 "vqattn_lut_p3_k2.cu")],
                       extra_include_paths=[str(here.parent / "32_kernel_tiling" / "csrc")],
                       extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
        finally:
            if old is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = old
    return _P3


def _p3_request(kst):
    """variant name (without the k{kst}_ prefix) asked for by MV_LUT_VARIANT; p3: entries missing a kst -> auto"""
    auto = "a100" if torch.cuda.get_device_capability() == (8, 0) else "4090"
    spec = auto if LUT_VARIANT == "auto" else LUT_VARIANT
    if spec in _P3_PRESETS:
        return _P3_PRESETS[spec][kst]
    assert spec.startswith("p3:"), f"MV_LUT_VARIANT={LUT_VARIANT!r}: use a100 | 4090 | auto | p3:k1=..,k2=.."
    kv = dict(x.split("=") for x in spec[3:].split(",") if x)
    return kv.get(f"k{kst}", _P3_PRESETS[auto][kst])


def _p3_vid(kst):
    if kst in _P3_VID:
        return _P3_VID[kst]
    e = p3_ext()
    names = {n: i for i, (n, _) in enumerate(e.variants())}
    req = _p3_request(kst)
    order = [req] + [v for v in _P3_FALLBACK[kst] if v != req]
    used, occ = None, {}
    for v in order:
        n = f"k{kst}_{v}"
        assert n in names, f"p3 variant {n} not compiled (csrc/vqattn_lut_p3_vars.h)"
        o = e.occupancy(names[n])
        occ[n] = {"blocks_per_sm": o[0], "dyn_smem": o[1], "regs": o[2], "local_bytes": o[3]}
        if o[0] >= 1:
            used = n
            break
    assert used is not None, f"no launchable p3 variant for kst {kst}: {occ}"
    info = {"spec": LUT_VARIANT, "requested": f"k{kst}_{req}", "used": used, "fallback": used != f"k{kst}_{req}",
            "device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()), "occupancy": occ}
    LUT_VARIANT_INFO[kst] = info
    msg = (f"[e2e_kern] MV_LUT_VARIANT={LUT_VARIANT}: kst {kst} -> {used}"
           + (f" (FALLBACK: {info['requested']} not launchable on {info['device']}: {occ})" if info["fallback"] else ""))
    if msg not in _P3_LOGGED:
        _P3_LOGGED.add(msg)
        print(msg, flush=True)
    _P3_VID[kst] = names[used]
    return names[used]
