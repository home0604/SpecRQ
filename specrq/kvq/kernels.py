"""E6 kvregime: Triton kernels for a nested (residual) VQ KV cache at decode time.

KV format (per layer): every kv-head vector (D = 128, post-RoPE K, plain V) is split into NSV = D / d sub-vectors;
stage s holds one code (C = 256 codewords, uint8) per sub-vector; the reconstruction of a prefix of k stages is
sum_{s<k} cb[kv, h, s, code_s]. One codebook per (layer, K/V, kv-head, stage), shared over the NSV sub-vector
positions of the head (C x d bf16 = 4 KiB for d8, 8 KiB for d16).
  codes : uint8 [2 (K,V), S, B, Hkv, T, NSV]   (stage-major: a k-stage draft reads only the first k planes)
  cb    : bf16  [2, Hkv, S, C, d]
Rate: log2(C) / d bit per element per stage (d8 K256: 1 bit; d16 K256: 0.5 bit).

Kernels
  vq_attention : flash-decoding (split-K over context) GQA attention of M new query rows per sequence (G query heads per
                 kv-head stacked as G*M rows), reading the first `kst` stages of the codes and dequantizing on the fly
                 (codebook gathers are L1-resident global loads: Triton has no explicit shared-memory gather), RoPE of q
                 fused into the q load (q read straight from the fused QKV projection output). MODE 1/2/3 read a dense
                 bf16 / INT4 / INT2 (per token-head fp16 scale) cache with the same kernel structure (references).
  vq_encode    : append of the M new tokens: K RoPE + greedy residual nearest-codeword search (fp32, exact) for the
                 first `senc` stages of K and V, codes written in place at positions ctx .. ctx+M-1.
  vq_dequant_weight : weight-VQ dequantization (logical codes [L, N, K/d] uint8, float codebooks [L, 256, d],
                 row scales) -> bf16 W [N, K], used by the dequant + cuBLAS weight path.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------------------------------------- tiles
@triton.jit
def _kv_tile(P, CB, SC, b, h, offs_t, tmask, s_s, s_b, s_h, s_t, scb_h, scb_s, ss_b, ss_h,
             D: tl.constexpr, DSUB: tl.constexpr, NSV: tl.constexpr, KST: tl.constexpr, BN: tl.constexpr,
             MODE: tl.constexpr):
    if MODE == 0:  # nested VQ codes
        # 2-D gather: column j of the tile reads code (t, j // DSUB) (same byte for DSUB lanes, L1 hit) and codeword
        # element j % DSUB
        jj = tl.arange(0, D)
        base = P + b * s_b + h * s_h + offs_t[:, None] * s_t + (jj // DSUB)[None, :]
        cbb = CB + h * scb_h + (jj % DSUB)[None, :]
        c = tl.load(base, mask=tmask[:, None], other=0).to(tl.int32)
        acc = tl.load(cbb + c * DSUB).to(tl.float32)
        for s in tl.static_range(1, KST):
            c = tl.load(base + s * s_s, mask=tmask[:, None], other=0).to(tl.int32)
            acc += tl.load(cbb + s * scb_s + c * DSUB).to(tl.float32)
        t = acc.to(tl.bfloat16)
    elif MODE == 1:  # dense bf16 [B, Hkv, T, D]
        t = tl.load(P + b * s_b + h * s_h + offs_t[:, None] * s_t + tl.arange(0, D)[None, :], mask=tmask[:, None],
                    other=0.0)
    elif MODE == 2:  # INT4 [B, Hkv, T, D/2] bytes, lo nibble = even element; value (q - 8) * scale
        x = tl.load(P + b * s_b + h * s_h + offs_t[:, None] * s_t + tl.arange(0, D // 2)[None, :],
                    mask=tmask[:, None], other=0)
        lo = (x & 15).to(tl.float32) - 8.0
        hi = (x >> 4).to(tl.float32) - 8.0
        v = tl.reshape(tl.join(lo, hi), (BN, D))
        sc = tl.load(SC + b * ss_b + h * ss_h + offs_t, mask=tmask, other=0.0).to(tl.float32)
        t = (v * sc[:, None]).to(tl.bfloat16)
    else:  # INT2 [B, Hkv, T, D/4] bytes, element 4j+i in bits 2i..2i+1; value (q - 1.5) * scale
        x = tl.load(P + b * s_b + h * s_h + offs_t[:, None] * s_t + tl.arange(0, D // 4)[None, :],
                    mask=tmask[:, None], other=0)
        a0 = (x & 3).to(tl.float32) - 1.5
        a1 = ((x >> 2) & 3).to(tl.float32) - 1.5
        a2 = ((x >> 4) & 3).to(tl.float32) - 1.5
        a3 = ((x >> 6) & 3).to(tl.float32) - 1.5
        v = tl.reshape(tl.join(tl.join(a0, a2), tl.join(a1, a3)), (BN, D))
        sc = tl.load(SC + b * ss_b + h * ss_h + offs_t, mask=tmask, other=0.0).to(tl.float32)
        t = (v * sc[:, None]).to(tl.bfloat16)
    return t


# ------------------------------------------------------------------------------------------------------- attention
@triton.jit
def _attn_split(QKV, K, V, CBK, CBV, SCK, SCV, INVF, OP, MP, LP,
                ctx, chunk, qk_scale,
                sq_r, s_s, s_b, s_h, s_t, scb_h, scb_s, ss_b, ss_h, SPLITS,
                HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
                DSUB: tl.constexpr, NSV: tl.constexpr, KST: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr):
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
    # q rows straight from the fused QKV output [B*M, (H + 2 Hkv) * D]; RoPE (rotate-half) in fp32
    qrow = QKV + (b * M + rm)[:, None] * sq_r + (h * G + rg)[:, None] * D
    q = tl.load(qrow + dd[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qrow + ((dd + D // 2) % D)[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    f = tl.load(INVF + (dd % (D // 2)))
    ang = (ctx + rm).to(tl.float32)[:, None] * f[None, :]
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    q = (q * tl.cos(ang) + sgn[None, :] * qp * tl.sin(ang)) * qk_scale  # qk_scale = D^-0.5 * log2(e)
    q = q.to(tl.bfloat16)
    lim = ctx + rm  # last visible key position of each row (causal among the M new rows)
    start = sp * chunk
    end = tl.minimum(start + chunk, n)
    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for t0 in range(start, end, BN):
        offs = t0 + tl.arange(0, BN)
        tm = offs < end
        k = _kv_tile(K, CBK, SCK, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, ss_b, ss_h,
                     D, DSUB, NSV, KST, BN, MODE)
        s = tl.dot(q, tl.trans(k))
        s = tl.where((offs[None, :] <= lim[:, None]) & tm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = _kv_tile(V, CBV, SCV, b, h, offs, tm, s_s, s_b, s_h, s_t, scb_h, scb_s, ss_b, ss_h,
                     D, DSUB, NSV, KST, BN, MODE)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = (bh * SPLITS + sp) * BM + rows
    tl.store(OP + o[:, None] * D + dd[None, :], acc)
    tl.store(MP + o, m_i)
    tl.store(LP + o, l_i)


@triton.jit
def _attn_combine(OP, MP, LP, OUT, SPLITS, HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr,
                  D: tl.constexpr, SP2: tl.constexpr):
    """Merge SPLITS partials (O unnormalized, m, l; exp2 domain) of one (b, kv-head, row); chunks of SP2 splits."""
    bh = tl.program_id(0)
    r = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    dd = tl.arange(0, D)
    mx = tl.max(tl.full([SP2], float("-inf"), tl.float32), 0)
    for c0 in range(0, SPLITS, SP2):
        sp = c0 + tl.arange(0, SP2)
        m = tl.load(MP + (bh * SPLITS + sp) * BM + r, mask=sp < SPLITS, other=float("-inf"))
        mx = tl.maximum(mx, tl.max(m, 0))
    acc = tl.zeros([D], dtype=tl.float32)
    den = tl.sum(tl.zeros([SP2], dtype=tl.float32), 0)
    for c0 in range(0, SPLITS, SP2):
        sp = c0 + tl.arange(0, SP2)
        spm = sp < SPLITS
        o = (bh * SPLITS + sp) * BM + r
        m = tl.load(MP + o, mask=spm, other=float("-inf"))
        l = tl.load(LP + o, mask=spm, other=0.0)
        w = tl.where(spm & (m > float("-inf")), tl.exp2(m - mx), 0.0)
        a = tl.load(OP + o[:, None] * D + dd[None, :], mask=spm[:, None], other=0.0)
        acc += tl.sum(a * w[:, None], 0)
        den += tl.sum(l * w, 0)
    out = acc / den
    rm = r // G
    rg = r % G
    tl.store(OUT + (b * M + rm) * (HKV * G * D) + (h * G + rg) * D + dd, out.to(tl.bfloat16))


FORMATS = {  # name -> (d, per-stage code bits); fmt id of the LUT kernel (None: not supported there)
    "d8S2": (8, [8, 8]), "d16S4": (16, [8, 8, 8, 8]), "d8K16x4": (8, [4, 4, 4, 4]), "d8K16x2K256": (8, [4, 4, 8])}
LUT_FMT = {(8, (8, 8)): 0, (8, (4, 4, 4, 4)): 1, (8, (4, 4, 8)): 2}


class KVCache:
    """One layer-stack of KV storage. mode 'vq' (d, per-stage bits; uniform 8-bit stages use the [2,S,B,H,T,NSV]
    layout of the v1 kernels, other mixes per-stage planes [2,B,H,T,W_s] with 4-bit codes nibble-packed),
    'bf16', 'int4', 'int2'. planes(layer) -> (K planes, V planes, K codebooks, V codebooks), one entry per stage:
    planes [B,H,T,W_s] (token-contiguous), codebooks [H,C_s,d]."""

    def __init__(self, n_layers, B, Hkv, T, D=128, mode="vq", d=8, S=2, C=256, device="cuda", gen=None, bits=None):
        self.n_layers, self.B, self.Hkv, self.T, self.D, self.mode = n_layers, B, Hkv, T, D, mode
        self.bits = list(bits) if bits is not None else [8] * S
        self.d, self.S = d, len(self.bits)
        self.C = 2 ** self.bits[0] if len(set(self.bits)) == 1 else None
        self.nsv = D // d
        self.uniform = all(b == 8 for b in self.bits)
        self.widths = [self.nsv * b // 8 for b in self.bits]
        self.codes, self.cb, self.data, self.scale, self._planes = [], [], [], [], []
        for _ in range(n_layers):
            if mode == "vq":
                amp = [0.5 ** s for s in range(self.S)]  # residual structure: stage codebooks shrink
                if self.uniform:
                    S_, C_ = self.S, 256
                    codes = torch.randint(0, C_, (2, S_, B, Hkv, T, self.nsv), dtype=torch.uint8, device=device,
                                          generator=gen)
                    sc = torch.tensor(amp, device=device)[None, None, :, None, None]
                    cb = (torch.randn(2, Hkv, S_, C_, d, device=device, generator=gen) * sc).to(torch.bfloat16)
                    self.codes.append(codes)
                    self.cb.append(cb)
                    self._planes.append(([codes[0, s] for s in range(S_)], [codes[1, s] for s in range(S_)],
                                         [cb[0, :, s] for s in range(S_)], [cb[1, :, s] for s in range(S_)]))
                else:
                    assert d == 8, "mixed / 4-bit stages: d = 8 only"
                    pl = [torch.randint(0, 256, (2, B, Hkv, T, w), dtype=torch.uint8, device=device, generator=gen)
                          for w in self.widths]
                    cbs = [(torch.randn(2, Hkv, 2 ** b, d, device=device, generator=gen) * a).to(torch.bfloat16)
                           for b, a in zip(self.bits, amp)]
                    self.codes.append(pl)
                    self.cb.append(cbs)
                    self._planes.append(([p[0] for p in pl], [p[1] for p in pl], [c[0] for c in cbs],
                                         [c[1] for c in cbs]))
            elif mode == "bf16":
                self.data.append(torch.randn(2, B, Hkv, T, D, device=device, generator=gen).to(torch.bfloat16))
            else:
                w = D // 2 if mode == "int4" else D // 4
                self.data.append(torch.randint(0, 256, (2, B, Hkv, T, w), dtype=torch.uint8, device=device,
                                               generator=gen))
                self.scale.append((torch.rand(2, B, Hkv, T, device=device, generator=gen) * 0.2 + 0.05).half())

    def planes(self, layer):
        return self._planes[layer]

    def lut_fmt(self):
        return LUT_FMT.get((self.d, tuple(self.bits)))

    def nbytes_per_token(self):
        per = {"vq": 2 * sum(self.widths) if self.mode == "vq" else 0, "bf16": 2 * self.D * 2,
               "int4": 2 * (self.D // 2 + 2), "int2": 2 * (self.D // 4 + 2)}[self.mode]
        return per * self.Hkv * self.n_layers

    def bytes_read(self, layer_tokens, kst=None):
        """bytes of KV payload read by one attention call over `layer_tokens` tokens (all B, one layer)."""
        if self.mode == "vq":
            return 2 * sum(self.widths[:(kst or self.S)]) * self.Hkv * layer_tokens
        return self.nbytes_per_token() // self.n_layers * layer_tokens


def attn_workspace(B, Hkv, G, M, splits, D=128, device="cuda"):
    BM = max(16, triton.next_power_of_2(G * M))
    n = B * Hkv * splits * BM
    return (torch.empty(n, D, dtype=torch.float32, device=device), torch.empty(n, dtype=torch.float32, device=device),
            torch.empty(n, dtype=torch.float32, device=device), BM)


_EXT = None


def ext():
    """CUDA VQ attention (codebooks in shared memory); build flags per CLAUDE.md gotchas."""
    global _EXT
    if _EXT is None:
        import os
        from pathlib import Path
        from torch.utils.cpp_extension import load
        os.environ.setdefault("CC", "/usr/bin/gcc-11")
        os.environ.setdefault("CXX", "/usr/bin/g++-11")
        os.environ.setdefault("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11")
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.1")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        _EXT = load(name="kvq_vqattn", sources=[str(Path(__file__).parent / "csrc" / "vqattn.cu")],
                    extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _EXT


_EXT_MMA = None


def ext_mma():
    """Tensor-core (WMMA) VQ attention."""
    global _EXT_MMA
    if _EXT_MMA is None:
        from pathlib import Path
        from torch.utils.cpp_extension import load
        ext()  # sets the build environment
        _EXT_MMA = load(name="kvq_vqattn_mma", sources=[str(Path(__file__).parent / "csrc" / "vqattn_mma.cu")],
                        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _EXT_MMA


_EXT_LUT = None


def ext_lut():
    """LUT-score / FMA-value VQ attention (vqattn_lut.cu)."""
    global _EXT_LUT
    if _EXT_LUT is None:
        from pathlib import Path
        from torch.utils.cpp_extension import load
        ext()
        _EXT_LUT = load(name="kvq_vqattn_lut", sources=[str(Path(__file__).parent / "csrc" / "vqattn_lut.cu")],
                        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17"], verbose=False)
    return _EXT_LUT


def lut_ok(cache, kst, M, smem_max=101376):
    """True if the LUT kernel can run (format supported, shared memory fits)."""
    fmt = cache.lut_fmt() if cache.mode == "vq" else None
    if fmt is None:
        return False
    return ext_lut().vq_attn_lut_plan(fmt, kst, M)[0] <= smem_max


def vq_attention(qkv, cache: KVCache, layer, ctx, M, G, inv_freq, out, ws, splits, chunk, kst=None, BN=64,
                 num_warps=4, num_stages=1, backend="cuda"):
    """qkv [B*M, (Hkv*G + 2*Hkv)*D] bf16 (pre-RoPE q); attends to positions [0, ctx+M) of `layer`. Writes out
    [B*M, Hkv*G*D] bf16. kst = number of VQ stages read (default all)."""
    B, Hkv, D = cache.B, cache.Hkv, cache.D
    OP, MP, LP, BM = ws
    mode = {"vq": 0, "bf16": 1, "int4": 2, "int2": 3}[cache.mode]
    if mode == 0:
        K, V = cache.codes[layer][0], cache.codes[layer][1]
        CBK, CBV = cache.cb[layer][0], cache.cb[layer][1]
        SCK = SCV = CBK
        s_s, s_b, s_h, s_t = K.stride(0), K.stride(1), K.stride(2), K.stride(3)
        scb_h, scb_s = CBK.stride(0), CBK.stride(1)
        ss_b = ss_h = 0
        kst = kst or cache.S
    else:
        K, V = cache.data[layer][0], cache.data[layer][1]
        CBK = CBV = K
        SCK, SCV = (cache.scale[layer][0], cache.scale[layer][1]) if mode >= 2 else (K, V)
        s_s, s_b, s_h, s_t = 0, K.stride(0), K.stride(1), K.stride(2)
        scb_h = scb_s = 0
        ss_b, ss_h = (SCK.stride(0), SCK.stride(1)) if mode >= 2 else (0, 0)
        kst = 1
    qk_scale = D ** -0.5 * LOG2E
    if mode == 0 and backend == "lut":
        fmt = cache.lut_fmt()
        assert fmt is not None, ("LUT kernel: unsupported format", cache.d, cache.bits)
        nw = 16 if M == 1 else 4
        assert OP.shape[0] >= B * Hkv * splits * nw * BM, "workspace: attn_workspace(..., splits * 16)"
        kpl, vpl, kcb, vcb = cache.planes(layer)
        ext_lut().vq_attn_lut(qkv, kpl, vpl, kcb, vcb, inv_freq, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, M, fmt,
                              kst, qk_scale)
        splits = splits * nw
    elif mode == 0 and backend == "mmawp":
        fmt = cache.lut_fmt()
        assert fmt is not None, ("planar warp-split kernel: unsupported format", cache.d, cache.bits)
        nw = 4
        assert OP.shape[0] >= B * Hkv * splits * nw * BM, "workspace"
        kpl, vpl, kcb, vcb = cache.planes(layer)
        ext_mma().vq_attn_wp(qkv, kpl, vpl, kcb, vcb, inv_freq, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, M, fmt,
                             kst, qk_scale)
        splits = splits * nw
    elif mode == 0 and not cache.uniform:
        raise ValueError(f"backend {backend} supports uniform 8-bit stages only")
    elif mode == 0 and backend == "mmaw":
        nw = 2 if (cache.d == 16 and kst == 4) else 4
        assert OP.shape[0] >= B * Hkv * splits * nw * BM, "workspace: attn_workspace(..., splits * 4)"
        ext_mma().vq_attn_w(qkv, K, V, CBK, CBV, inv_freq, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, M, cache.d,
                            kst, s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale)
        splits = splits * nw
    elif mode == 0 and backend == "mma":
        assert chunk % 32 == 0, chunk
        ext_mma().vq_attn_mma(qkv, K, V, CBK, CBV, inv_freq, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, M, cache.d,
                              kst, s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale)
    elif mode == 0 and backend == "cuda":
        assert chunk % 128 == 0, chunk
        ext().vq_attn_split(qkv, K, V, CBK, CBV, inv_freq, OP, MP, LP, ctx, chunk, splits, B, Hkv, BM, M, cache.d,
                            kst, s_s, s_b, s_h, s_t, scb_h, scb_s, qk_scale)
    else:
        _attn_split[(B * Hkv, splits)](qkv, K, V, CBK, CBV, SCK, SCV, inv_freq, OP, MP, LP, ctx, chunk, qk_scale,
                                   qkv.stride(0), s_s, s_b, s_h, s_t, scb_h, scb_s, ss_b, ss_h, splits,
                                   HKV=Hkv, G=G, M=M, BM=BM, D=D, DSUB=cache.d, NSV=cache.nsv, KST=kst, BN=BN,
                                   MODE=mode, num_warps=num_warps, num_stages=num_stages)
    _attn_combine[(B * Hkv, G * M)](OP, MP, LP, out, splits, HKV=Hkv, G=G, M=M, BM=BM, D=D,
                                    SP2=min(32, max(2, triton.next_power_of_2(splits))), num_warps=4)
    return out


def split_plan(B, Hkv, n, BN=64, target_programs=1024, max_chunk=None):
    """(splits, chunk): chunk a multiple of BN, about target_programs programs in total."""
    want = max(1, -(-target_programs // (B * Hkv)))
    chunk = max(BN, -(-n // want))
    chunk = -(-chunk // BN) * BN
    if max_chunk:
        chunk = min(chunk, max_chunk)
    return -(-n // chunk), chunk


# ---------------------------------------------------------------------------------------------------------- encode
@triton.jit
def _encode(QKV, CODES, CB, INVF, ctx, sq_r, c_kv, c_s, c_b, c_h, c_t, cb_kv, cb_h, cb_s,
            HKV: tl.constexpr, HQ: tl.constexpr, M: tl.constexpr, D: tl.constexpr, DSUB: tl.constexpr,
            NSV: tl.constexpr, C: tl.constexpr, SENC: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)
    b = r // M
    m = r % M
    pos = ctx + m
    ri = tl.arange(0, XR)
    ci = tl.arange(0, XC)
    xm = (ri[:, None] < NSV) & (ci[None, :] < DSUB)
    idx = ri[:, None] * DSUB + ci[None, :]
    src = QKV + r * sq_r + (HQ + kv * HKV + h) * D
    x = tl.load(src + idx, mask=xm, other=0.0).to(tl.float32)
    xp = tl.load(src + (idx + D // 2) % D, mask=xm, other=0.0).to(tl.float32)
    f = tl.load(INVF + idx % (D // 2), mask=xm, other=0.0)
    ang = pos.to(tl.float32) * f
    isk = (kv == 0).to(tl.float32)
    sgn = tl.where(idx < D // 2, -1.0, 1.0)
    # K: RoPE (rotate-half); V: identity (cos -> 1, sin -> 0)
    x = x * (isk * tl.cos(ang) + (1.0 - isk)) + isk * sgn * xp * tl.sin(ang)
    x = tl.where(xm, x, 0.0)
    cbase = CB + kv * cb_kv + h * cb_h
    cidx = tl.arange(0, C)
    dst = CODES + kv * c_kv + b * c_b + h * c_h + pos * c_t + ri
    for s in tl.static_range(SENC):
        cbp = cbase + s * cb_s
        cbt = tl.load(cbp + cidx[None, :] * DSUB + ci[:, None], mask=(ci[:, None] < DSUB), other=0.0).to(tl.float32)
        cn = tl.sum(cbt * cbt, 0)
        dots = tl.dot(x, cbt, input_precision="ieee")
        dist = cn[None, :] - 2.0 * dots
        code = tl.argmin(dist, 1)
        sel = tl.load(cbp + code[:, None] * DSUB + ci[None, :], mask=xm, other=0.0).to(tl.float32)
        x = x - sel
        tl.store(dst + s * c_s, code.to(tl.uint8), mask=ri < NSV)


def vq_encode(qkv, cache: KVCache, layer, ctx, M, Hq, inv_freq, senc=None):
    """Encode the K (with RoPE) and V of the B*M new rows of qkv into the codes of `layer` at ctx .. ctx+M-1."""
    senc = senc or cache.S
    if not cache.uniform:
        return _vq_encode_planar(qkv, cache, layer, ctx, M, Hq, inv_freq, senc)
    codes, cb = cache.codes[layer], cache.cb[layer]
    nsv, d = cache.nsv, cache.d
    _encode[(qkv.shape[0], cache.Hkv, 2)](qkv, codes, cb, inv_freq, ctx, qkv.stride(0), codes.stride(0),
                                          codes.stride(1), codes.stride(2), codes.stride(3), codes.stride(4),
                                          cb.stride(0), cb.stride(1), cb.stride(2),
                                          HKV=cache.Hkv, HQ=Hq, M=M, D=cache.D, DSUB=d, NSV=nsv, C=256, SENC=senc,
                                          XR=max(16, nsv), XC=max(16, d), num_warps=4)


@triton.jit
def _enc_stage(x, cbp, dst, ri, ci, xm, C: tl.constexpr, NIB: tl.constexpr, DSUB: tl.constexpr, NSV: tl.constexpr,
               XR: tl.constexpr):
    cidx = tl.arange(0, C)
    cbt = tl.load(cbp + cidx[None, :] * DSUB + ci[:, None], mask=(ci[:, None] < DSUB), other=0.0).to(tl.float32)
    cn = tl.sum(cbt * cbt, 0)
    dots = tl.dot(x, cbt, input_precision="ieee")
    code = tl.argmin(cn[None, :] - 2.0 * dots, 1)
    sel = tl.load(cbp + code[:, None] * DSUB + ci[None, :], mask=xm, other=0.0).to(tl.float32)
    if NIB:  # two sub-vectors per byte, low nibble = even sub-vector
        c2 = tl.reshape(code, (XR // 2, 2))
        pk = tl.sum(c2 * (1 + 15 * tl.arange(0, 2))[None, :], 1)
        tl.store(dst + tl.arange(0, XR // 2), pk.to(tl.uint8), mask=tl.arange(0, XR // 2) < NSV // 2)
    else:
        tl.store(dst + ri, code.to(tl.uint8), mask=ri < NSV)
    return x - sel


@triton.jit
def _encode_p(QKV, P0, P1, P2, P3, CB0, CB1, CB2, CB3, INVF, ctx, sq_r,
              pkv0, pb0, ph0, pkv1, pb1, ph1, pkv2, pb2, ph2, pkv3, pb3, ph3, ckv0, ch0, ckv1, ch1, ckv2, ch2, ckv3,
              ch3, HKV: tl.constexpr, HQ: tl.constexpr, M: tl.constexpr, D: tl.constexpr, DSUB: tl.constexpr,
              NSV: tl.constexpr, SENC: tl.constexpr, XR: tl.constexpr, XC: tl.constexpr,
              B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr):
    """Planar-format encode (per-stage code bits B_s in {4, 8}); otherwise as _encode."""
    r = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)
    b = r // M
    pos = ctx + r % M
    ri = tl.arange(0, XR)
    ci = tl.arange(0, XC)
    xm = (ri[:, None] < NSV) & (ci[None, :] < DSUB)
    idx = ri[:, None] * DSUB + ci[None, :]
    src = QKV + r * sq_r + (HQ + kv * HKV + h) * D
    x = tl.load(src + idx, mask=xm, other=0.0).to(tl.float32)
    xp = tl.load(src + (idx + D // 2) % D, mask=xm, other=0.0).to(tl.float32)
    f = tl.load(INVF + idx % (D // 2), mask=xm, other=0.0)
    ang = pos.to(tl.float32) * f
    isk = (kv == 0).to(tl.float32)
    sgn = tl.where(idx < D // 2, -1.0, 1.0)
    x = x * (isk * tl.cos(ang) + (1.0 - isk)) + isk * sgn * xp * tl.sin(ang)
    x = tl.where(xm, x, 0.0)
    x = _enc_stage(x, CB0 + kv * ckv0 + h * ch0, P0 + kv * pkv0 + b * pb0 + h * ph0 + pos * (NSV * B0 // 8),
                   ri, ci, xm, 2 ** B0, B0 == 4, DSUB, NSV, XR)
    if SENC > 1:
        x = _enc_stage(x, CB1 + kv * ckv1 + h * ch1, P1 + kv * pkv1 + b * pb1 + h * ph1 + pos * (NSV * B1 // 8),
                       ri, ci, xm, 2 ** B1, B1 == 4, DSUB, NSV, XR)
    if SENC > 2:
        x = _enc_stage(x, CB2 + kv * ckv2 + h * ch2, P2 + kv * pkv2 + b * pb2 + h * ph2 + pos * (NSV * B2 // 8),
                       ri, ci, xm, 2 ** B2, B2 == 4, DSUB, NSV, XR)
    if SENC > 3:
        x = _enc_stage(x, CB3 + kv * ckv3 + h * ch3, P3 + kv * pkv3 + b * pb3 + h * ph3 + pos * (NSV * B3 // 8),
                       ri, ci, xm, 2 ** B3, B3 == 4, DSUB, NSV, XR)


def _vq_encode_planar(qkv, cache, layer, ctx, M, Hq, inv_freq, senc):
    pl, cbs = cache.codes[layer], cache.cb[layer]
    S = cache.S
    P = [pl[min(s, S - 1)] for s in range(4)]
    CB = [cbs[min(s, S - 1)] for s in range(4)]
    bits = [cache.bits[min(s, S - 1)] for s in range(4)]
    st = []
    for p_ in P:
        st += [p_.stride(0), p_.stride(1), p_.stride(2)]
    for c_ in CB:
        st += [c_.stride(0), c_.stride(1)]
    _encode_p[(qkv.shape[0], cache.Hkv, 2)](qkv, *P, *CB, inv_freq, ctx, qkv.stride(0), *st,
                                            HKV=cache.Hkv, HQ=Hq, M=M, D=cache.D, DSUB=cache.d, NSV=cache.nsv,
                                            SENC=senc, XR=max(16, cache.nsv), XC=max(16, cache.d),
                                            B0=bits[0], B1=bits[1], B2=bits[2], B3=bits[3], num_warps=4)


@triton.jit
def _append_dense(QKV, DATA, INVF, ctx, sq_r, d_kv, d_b, d_h, d_t, HKV: tl.constexpr, HQ: tl.constexpr,
                  M: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0)
    h = tl.program_id(1)
    kv = tl.program_id(2)
    b = r // M
    pos = ctx + r % M
    dd = tl.arange(0, D)
    src = QKV + r * sq_r + (HQ + kv * HKV + h) * D
    x = tl.load(src + dd).to(tl.float32)
    xp = tl.load(src + (dd + D // 2) % D).to(tl.float32)
    ang = pos.to(tl.float32) * tl.load(INVF + dd % (D // 2))
    isk = (kv == 0).to(tl.float32)
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    x = x * (isk * tl.cos(ang) + (1.0 - isk)) + isk * sgn * xp * tl.sin(ang)
    tl.store(DATA + kv * d_kv + b * d_b + h * d_h + pos * d_t + dd, x.to(tl.bfloat16))


def dense_append(qkv, cache: KVCache, layer, ctx, M, Hq, inv_freq):
    data = cache.data[layer]
    _append_dense[(qkv.shape[0], cache.Hkv, 2)](qkv, data, inv_freq, ctx, qkv.stride(0), data.stride(0),
                                                data.stride(1), data.stride(2), data.stride(3),
                                                HKV=cache.Hkv, HQ=Hq, M=M, D=cache.D, num_warps=1)


# ------------------------------------------------------------------------------------------------- weight dequant
@triton.jit
def _wdeq(CODES, CB, SC, W, N, G, c_l, KW: tl.constexpr, DSUB: tl.constexpr, C: tl.constexpr, BNR: tl.constexpr,
          BG: tl.constexpr):
    pn = tl.program_id(0)
    pg = tl.program_id(1)
    rn = pn * BNR + tl.arange(0, BNR)
    gg = pg * BG + tl.arange(0, BG)
    m2 = (rn[:, None] < N) & (gg[None, :] < G)
    e = tl.arange(0, DSUB)
    acc = tl.zeros([BNR, BG, DSUB], dtype=tl.float32)
    for l in tl.static_range(KW):
        c = tl.load(CODES + l * c_l + rn[:, None] * G + gg[None, :], mask=m2, other=0).to(tl.int32)
        acc += tl.load(CB + l * C * DSUB + c[:, :, None] * DSUB + e[None, None, :]).to(tl.float32)
    sc = tl.load(SC + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc = acc * sc[:, None, None]
    tl.store(W + rn[:, None, None] * (G * DSUB) + gg[None, :, None] * DSUB + e[None, None, :], acc.to(tl.bfloat16),
             mask=m2[:, :, None])


_WDEQ_CFG = {}
_WDEQ_CANDS = ((8, 64, 4), (32, 32, 4), (16, 128, 8), (64, 16, 4))


def vq_dequant_weight(codes, cb, scales, k, out, d=8):
    """codes uint8 [L, N, K/d] (logical layout), cb [L, 256, d] (bf16 or fp32), scales [N] -> out bf16 [N, K]
    (prefix k). Tile config autotuned once per (N, K, k) outside graph capture."""
    L, N, G = codes.shape

    def run(cfg):
        BNR, BG, nw = cfg
        _wdeq[(triton.cdiv(N, BNR), triton.cdiv(G, BG))](codes, cb, scales, out, N, G, codes.stride(0), KW=int(k),
                                                         DSUB=d, C=cb.shape[1], BNR=BNR, BG=BG, num_warps=nw)
    key = (N, G, int(k), d)
    if key not in _WDEQ_CFG:
        if torch.cuda.is_current_stream_capturing():
            _WDEQ_CFG[key] = _WDEQ_CANDS[1]
        else:
            ts = {}
            for cfg in _WDEQ_CANDS:
                run(cfg)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                for _ in range(5):
                    run(cfg)
                e.record()
                torch.cuda.synchronize()
                ts[cfg] = s.elapsed_time(e)
            _WDEQ_CFG[key] = min(ts, key=ts.get)
    run(_WDEQ_CFG[key])
    return out


# ----------------------------------------------------------------------------------------------------- references
def ref_rope(x, pos, inv_freq):
    """x [..., D] fp32 at integer positions pos (broadcast over leading dims except last)."""
    D = x.shape[-1]
    ang = pos[..., None].float() * inv_freq.repeat(2)
    rot = torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1)
    return x * ang.cos() + rot * ang.sin()


def ref_dequant_kv(cache: KVCache, layer, kst=None):
    """-> K, V bf16 [B, Hkv, T, D] dequantized (VQ: sum of the first kst stages in fp32, rounded to bf16 as in the
    kernels; INT: (q - z) * scale)."""
    if cache.mode == "vq":
        kst = kst or cache.S
        kpl, vpl, kcb, vcb = cache.planes(layer)
        out = []
        for planes, cbs in ((kpl, kcb), (vpl, vcb)):
            acc = 0
            for s in range(kst):
                c = planes[s].long()  # [B,H,T,W]
                if cache.bits[s] == 4:
                    c = torch.stack([c & 15, c >> 4], -1).flatten(-2)  # [B,H,T,NSV]
                tab = cbs[s].float()  # [H,C,d]
                hidx = torch.arange(cache.Hkv, device=c.device)[None, :, None, None]
                acc = acc + tab[hidx, c]  # [B,H,T,NSV,d]
            out.append(acc.reshape(*acc.shape[:3], cache.D).to(torch.bfloat16))
        return out
    if cache.mode == "bf16":
        return cache.data[layer][0], cache.data[layer][1]
    out = []
    for kv in range(2):
        x = cache.data[layer][kv].long()
        if cache.mode == "int4":
            v = torch.stack([(x & 15) - 8.0, (x >> 4) - 8.0], -1).flatten(-2)
        else:
            v = torch.stack([(x & 3), (x >> 2) & 3, (x >> 4) & 3, (x >> 6) & 3], -1).float() - 1.5
            v = v.flatten(-2)
        out.append((v.float() * cache.scale[layer][kv].float()[..., None]).to(torch.bfloat16))
    return out


def ref_attention(qkv, K, V, ctx, M, Hkv, G, inv_freq):
    """Dequantized-cache reference: fp32 RoPE(q) (rounded to bf16 like the kernel), SDPA in fp32 over [0, ctx+M)
    with causal mask among the new rows. Returns [B*M, Hq*D] fp32."""
    D = K.shape[-1]
    BM_ = qkv.shape[0]
    B = BM_ // M
    Hq = Hkv * G
    q = qkv[:, :Hq * D].float().view(B, M, Hq, D)
    pos = (ctx + torch.arange(M, device=qkv.device))[None, :, None].expand(B, M, Hq)
    q = ref_rope(q, pos, inv_freq).to(torch.bfloat16).float()
    n = ctx + M
    k = K[:, :, :n].float().repeat_interleave(G, 1)  # [B,Hq,n,D]
    v = V[:, :, :n].float().repeat_interleave(G, 1)
    s = torch.einsum("bmhd,bhtd->bhmt", q, k) * D ** -0.5
    allowed = torch.arange(n, device=qkv.device)[None, :] <= (ctx + torch.arange(M, device=qkv.device))[:, None]
    s = s.masked_fill(~allowed[None, None], float("-inf"))
    o = torch.einsum("bhmt,bhtd->bmhd", torch.softmax(s, -1), v)
    return o.reshape(B * M, Hq * D)


def ref_encode(x, cb, senc):
    """x fp32 [N, NSV, d] (one kv-head), cb [S, C, d] -> codes [senc, N, NSV] (greedy residual, exact fp32)."""
    out = []
    r = x.clone()
    for s in range(senc):
        c = cb[s].float()
        dist = (c * c).sum(-1)[None, None] - 2 * torch.einsum("nkd,cd->nkc", r, c)
        code = dist.argmin(-1)
        out.append(code)
        r = r - c[code]
    return torch.stack(out)


def inv_freq_llama(D=128, theta=500000.0, device="cuda"):
    return (1.0 / theta ** (torch.arange(0, D, 2, device=device).float() / D)).contiguous()


def splits_for(B, Hkv, n, BN=64):
    return split_plan(B, Hkv, n, BN)


__all__ = ["FORMATS", "LUT_FMT", "lut_ok", "ext_lut", "KVCache", "vq_attention", "vq_encode", "dense_append", "vq_dequant_weight", "attn_workspace",
           "split_plan", "ref_dequant_kv", "ref_attention", "ref_encode", "ref_rope", "inv_freq_llama", "math"]
