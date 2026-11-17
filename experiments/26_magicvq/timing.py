"""E7 magicvq step 2: measured decode-step parts of a SYNTHETIC bf16 Llama-3.1-8B (real shapes, random weights; timing
only) for MagicDec-style self-speculation on one RTX 4090.

Per (B, ctx): one step of B sequences x M new rows (embedding -> 32 x [RMSNorm, fused QKV, KV append, attention,
o_proj, RMSNorm, fused gate|up, SiLU*mul, down] -> norm -> full-vocab lm_head -> argmax), bf16 cuBLAS weights,
CUDA graph of `iters` steps, `reps` replays, median / min / max.
  target : bf16 KV over ctx, M = 1 (AR) and M = gamma + 1 (verify); attention = the faster of
           (i) SDPA flash (GQA folded into the query rows; M > 1 without the intra-block causal mask = cost proxy) and
           (ii) the specrq/kvq Triton split-K flash-decoding kernel (bf16 mode), split count tuned.
  evict<W>: same step with a bf16 cache of W tokens (MagicDec / StreamingLLM draft; window bookkeeping not timed).
  vq<fmt>  : draft step reading a VQ cache of the whole ctx (specrq/kvq kernels: best of LUT / WMMA / CUDA backends x
           split count) + vq_encode of its own new token; plus enc(M): encoding the gamma + 1 verified tokens into the
           draft cache once per round. NOTE the kvq kernels encode post-RoPE K; the tau eval quantizes pre-RoPE K
           (identical bytes; a pre-RoPE kernel would add RoPE of dequantized K: not timed).
  int4/int2: kvq per-token-head scalar KV attention (bandwidth proxy for a KIVI-like draft; no encode timed).
Usage: python experiments/26_magicvq/timing.py --grid 1x4096 1x8192 ... --run v1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from specrq.kvq.kernels import (KVCache, attn_workspace, dense_append, inv_freq_llama, lut_ok, split_plan,  # noqa: E402
                                vq_attention, vq_encode)
from specrq.nestkern.gpucheck import cotenants  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kvq_hybrid import hybrid_attention  # noqa: E402
from kvq_prerope import Affine, GSBooks, gs_encode_step, prerope_attention  # noqa: E402

NL, HID, H, HKV, D, INTER, VOCAB = 32, 4096, 32, 8, 128, 14336, 128256
G = H // HKV
QKVN = (H + 2 * HKV) * D
EPS = 1e-5
GSFMTS = {  # GSRQ (per-sub-vector-position shape codebooks x gains, pre-RoPE): name -> (KVCache kwargs, kst)
    "gs2_D8C256x2": (dict(d=8, bits=[8, 8]), 2),
    "gs1_D8C256": (dict(d=8, bits=[8, 8]), 1),
    "gs1_D16C256x2": (dict(d=16, bits=[8, 8]), 2),
    "gs05_D16C256": (dict(d=16, bits=[8, 8]), 1),
}
VQFMTS = {  # name -> (KVCache kwargs, kst)
    "vq2_d8K256x2": (dict(d=8, bits=[8, 8]), 2),
    "vq1_d8K256": (dict(d=8, bits=[8, 8]), 1),
    "vq1_d16K256x2": (dict(d=16, bits=[8, 8]), 2),
    "vq05_d16K256": (dict(d=16, bits=[8, 8]), 1),
    "vq05_d8K16": (dict(d=8, bits=[4, 4, 4, 4]), 1),
}


def gtime(fn, iters, reps):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    del g
    return {"us": statistics.median(ts), "min": min(ts), "max": max(ts), "n": reps, "iters": iters}


class Model:
    def __init__(self):
        bf = torch.bfloat16
        mk = lambda *s: torch.empty(*s, device="cuda", dtype=bf).normal_(0, 0.02)  # noqa: E731
        self.embed, self.lm_head = mk(VOCAB, HID), mk(VOCAB, HID)
        self.norm = torch.ones(HID, device="cuda", dtype=bf)
        self.layers = [{"qkv": mk(QKVN, HID), "o": mk(HID, H * D), "gu": mk(2 * INTER, HID), "down": mk(HID, INTER),
                        "n1": torch.ones(HID, device="cuda", dtype=bf), "n2": torch.ones(HID, device="cuda", dtype=bf)}
                       for _ in range(NL)]
        self.inv = inv_freq_llama(D)


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


class Step:
    """kv: KVCache (target bf16 / evict bf16 of W tokens / vq / int); attends to positions [0, ctx + M)."""

    def __init__(self, m, kv, B, ctx, M, backend, splits=1, chunk=64, kst=None, senc=None, append=True, wc=None,
                 wlen=0, aff=None):
        self.aff = aff
        self.m, self.kv, self.B, self.ctx, self.M = m, kv, B, ctx, M
        self.wc, self.wlen = wc, wlen  # hybrid: bf16 window cache (sinks + recent R) of wlen tokens
        if wc is not None:
            self.wsplits, self.wchunk = split_plan(B, HKV, wlen + M, 64, 256)
            self.ws2 = attn_workspace(B, HKV, G, M, self.wsplits)
        self.R = B * M
        self.backend, self.splits, self.chunk, self.kst, self.senc, self.append = backend, splits, chunk, kst, senc, append
        self.tokens = torch.randint(0, VOCAB, (self.R,), device="cuda")
        self.ws = attn_workspace(B, HKV, G, M, max(1, splits) * 16)
        self.out = torch.empty(self.R, H * D, device="cuda", dtype=torch.bfloat16)
        self.qkv_d = torch.randn(self.R, QKVN, device="cuda").to(torch.bfloat16)
        pos = torch.arange(ctx, ctx + M, device="cuda").float()
        ang = pos[:, None] * m.inv[None]  # [M, D/2]
        self.cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.bfloat16)  # [M, D]
        self.sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.bfloat16)

    def _append(self, qkv, li):
        if not self.append:
            return
        if self.wc is not None:  # new token enters the bf16 window; the token leaving it is VQ-encoded
            dense_append(qkv, self.wc, li, self.wlen, self.M, H, self.m.inv)
        if self.kv.mode == "vq" and isinstance(self.aff, GSBooks):
            B, M, nsv = self.B, self.M, self.kv.nsv
            for kv in range(2):
                tgt = self.kv.codes[li][kv]  # [S, B, Hkv, T, nsv]

                def put(s, c, tgt=tgt):
                    tgt[s, :, :, self.ctx:self.ctx + M].copy_(c.t().reshape(B, M, HKV, nsv).permute(0, 2, 1, 3))
                gs_encode_step(qkv[:, (H + kv * HKV) * D:(H + (kv + 1) * HKV) * D], self.aff, li, kv, self.senc, put)
        elif self.kv.mode == "vq":
            vq_encode(qkv, self.kv, li, self.ctx, self.M, H, self.m.inv, self.senc)
        elif self.kv.mode == "bf16":
            dense_append(qkv, self.kv, li, self.ctx, self.M, H, self.m.inv)

    def _attn(self, qkv, li):
        if self.wc is not None:
            return hybrid_attention(qkv, self.kv, self.wc, li, self.ctx, self.wlen, self.M, G, self.m.inv, self.out,
                                    self.ws, self.ws2, self.splits, self.chunk, self.wsplits, self.wchunk, self.kst,
                                    self.backend, self.aff)
        if self.backend == "pre":
            return prerope_attention(qkv, self.kv, self.aff, li, self.ctx, self.M, G, self.m.inv, self.out, self.ws,
                                     self.splits, self.chunk, self.kst)
        if self.backend == "sdpa":
            B, M = self.B, self.M
            q = qkv[:, :H * D].view(B, M, H, D)
            q1, q2 = q[..., :D // 2], q[..., D // 2:]
            qr = q * self.cos[None, :, None] + torch.cat([-q2, q1], -1) * self.sin[None, :, None]
            qr = qr.view(B, M, HKV, G, D).permute(0, 2, 3, 1, 4).reshape(B, HKV, G * M, D)
            n = self.ctx + M
            Kc, Vc = self.kv.data[li][0][:, :, :n], self.kv.data[li][1][:, :, :n]
            o = F.scaled_dot_product_attention(qr, Kc, Vc)
            self.out.view(B, M, HKV, G, D).copy_(o.view(B, HKV, G, M, D).permute(0, 3, 1, 2, 4))
            return self.out
        vq_attention(qkv, self.kv, li, self.ctx, self.M, G, self.m.inv, self.out, self.ws, self.splits, self.chunk,
                     self.kst, backend=self.backend)
        return self.out

    def full(self):
        m = self.m
        h = F.embedding(self.tokens, m.embed)
        for li, L in enumerate(m.layers):
            qkv = torch.mm(rms(h, L["n1"]), L["qkv"].t())
            self._append(qkv, li)
            o = self._attn(qkv, li)
            h = h + torch.mm(o, L["o"].t())
            gu = torch.mm(rms(h, L["n2"]), L["gu"].t())
            h = h + torch.mm(F.silu(gu[:, :INTER]) * gu[:, INTER:], L["down"].t())
        return torch.argmax(torch.mm(rms(h, m.norm), m.lm_head.t()), -1)

    def attention(self):
        for li in range(NL):
            self._attn(self.qkv_d, li)

    def encode(self):
        for li in range(NL):
            self._append(self.qkv_d, li)


def backends(kv, pre=False):
    if pre:
        return ("pre",)
    if kv.mode == "bf16":
        return ("sdpa", "triton")
    if kv.mode != "vq":
        return ("triton",)
    return ("lut", "mmaw", "mma", "cuda") if kv.uniform else ("lut", "mmawp")


def tune(m, kv, B, ctx, M, kst, iters, reps, aff=None):
    """attention-only (32 layers) over backends x split plans; returns best (backend, splits, chunk) and all."""
    n = ctx + M
    res = {}
    for be in backends(kv, aff is not None):
        if be == "lut" and not lut_ok(kv, kst, M):
            continue
        if be == "sdpa":
            st = Step(m, kv, B, ctx, M, "sdpa")
            res["sdpa"] = {"backend": "sdpa", "splits": 1, "chunk": n, **gtime(st.attention, iters, reps)}
            del st
            continue
        seen = set()
        for tp in (64, 128, 256, 512, 1024, 2048):
            sp, ch = split_plan(B, HKV, n, 64 if be == "pre" else 128, tp)
            if (sp, ch) in seen:
                continue
            seen.add((sp, ch))
            try:
                st = Step(m, kv, B, ctx, M, be, sp, ch, kst, aff=aff)
                res[f"{be}_tp{tp}"] = {"backend": be, "splits": sp, "chunk": ch, **gtime(st.attention, iters, reps)}
                del st
            except Exception as e:  # noqa: BLE001
                res[f"{be}_tp{tp}"] = {"backend": be, "error": repr(e)[:200]}
    ok = [r for r in res.values() if "us" in r]
    best = min(ok, key=lambda r: r["us"])
    return best, res


def measure(m, kv, B, ctx, M, iters, reps, kst=None, senc=None, comps=True, enc_Ms=(), aff=None):
    """aff: None (kvq kernels), Affine (pre-RoPE standardized nested VQ), GSBooks (GSRQ)."""
    best, tr = tune(m, kv, B, ctx, M, kst, iters, reps, aff)
    st = Step(m, kv, B, ctx, M, best["backend"], best["splits"], best["chunk"], kst, senc, aff=aff)
    r = {"B": B, "ctx": ctx, "M": M, "kst": kst, "attn_backend": best["backend"], "splits": best["splits"],
         "chunk": best["chunk"], "attn_tune": tr, "full": gtime(st.full, iters, reps)}
    if comps:
        r["attention"] = gtime(st.attention, iters, reps)
        if kv.mode in ("vq", "bf16"):
            r["append"] = gtime(st.encode, iters, reps)
    del st
    for Me in enc_Ms:  # encode of Me verified tokens into the (VQ) draft cache, 32 layers
        se = Step(m, kv, B, ctx, Me, "triton", kst=kst, senc=senc, aff=aff)
        r[f"enc_M{Me}"] = gtime(se.encode, iters, reps)
        del se
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", default=["1x4096", "1x8192", "1x16384", "1x32768", "4x4096", "4x8192",
                                                  "8x4096", "16x2048"])
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--vq", nargs="+", default=list(VQFMTS))
    ap.add_argument("--int", nargs="+", default=["int4", "int2"])
    ap.add_argument("--hyb_vq", nargs="+", default=["vq2_d8K256x2", "vq1_d8K256", "vq05_d16K256"])
    ap.add_argument("--windows", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--int_windows", type=int, nargs="+", default=[128])
    ap.add_argument("--gs", nargs="+", default=list(GSFMTS))
    ap.add_argument("--gs_hyb", nargs="+", default=["gs2_D8C256x2", "gs1_D8C256", "gs05_D16C256"])
    ap.add_argument("--pre_vq", nargs="+", default=["vq2_d8K256x2", "vq1_d8K256", "vq1_d16K256x2", "vq05_d16K256"],
                    help="also time the pre-RoPE standardized VQ kernel (the format the tau eval uses)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)  # noqa: E731
    co = cotenants()
    log(f"co-tenants at start: {co or 'none'}")
    m = Model()
    torch.cuda.synchronize()
    wbytes = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    log(f"weights {wbytes / 2**30:.2f} GiB, free {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    kvtok = NL * 2 * HKV * D * 2  # bytes per token bf16
    out_dir = "/data/lsh/results_accept/magicvq/timing"
    os.makedirs(out_dir, exist_ok=True)
    out = {"config": vars(a), "weights_GiB": wbytes / 2**30, "free_after_weights_GiB": free / 2**30,
           "cotenants_start": co, "fit": [], "records": []}
    Ms = [g + 1 for g in a.gammas]
    margin = 1.5 * 2**30
    for g in a.grid:
        B, ctx = (int(x) for x in g.split("x"))
        maxM = max(Ms)
        need_t = B * (ctx + maxM) * kvtok
        need_vq = B * (ctx + maxM) * NL * 2 * HKV * 32  # largest VQ draft cache (2 stages x 16 B per head-token)
        free = torch.cuda.mem_get_info()[0]
        fits = need_t + need_vq + margin <= free
        out["fit"].append({"B": B, "ctx": ctx, "target_kv_GiB": need_t / 2**30, "vq_draft_kv_GiB": need_vq / 2**30,
                           "free_GiB": free / 2**30, "fits": fits})
        log(f"[fit] B={B} ctx={ctx} target KV {need_t / 2**30:.2f} GiB + VQ {need_vq / 2**30:.2f} GiB, "
            f"free {free / 2**30:.2f} GiB -> {fits}")
        if not fits:
            continue
        recs = {"B": B, "ctx": ctx}
        kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode="bf16")
        recs["target"] = {M: measure(m, kv, B, ctx, M, a.iters, a.reps) for M in [1] + Ms}
        log(f"[target] B={B} ctx={ctx} " + " ".join(f"M{M}={r['full']['us'] / 1e3:.2f}ms({r['attn_backend']})"
                                                    for M, r in recs["target"].items()))
        del kv
        torch.cuda.empty_cache()
        recs["evict"] = {}
        for W in a.budgets:
            if W >= ctx:
                continue
            kv = KVCache(NL, B, HKV, W + 8, mode="bf16")
            recs["evict"][W] = measure(m, kv, B, W - 1, 1, a.iters, a.reps)
            del kv
        log(f"[evict] B={B} " + " ".join(f"W{W}={r['full']['us'] / 1e3:.2f}ms" for W, r in recs["evict"].items()))
        recs["vq"] = {}
        for name in a.vq:
            kw, kst = VQFMTS[name]
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode="vq", **kw)
            recs["vq"][name] = measure(m, kv, B, ctx, 1, a.iters, a.reps, kst=kst, senc=kst, enc_Ms=Ms)
            del kv
            torch.cuda.empty_cache()
        log(f"[vq] B={B} ctx={ctx} " + " ".join(
            f"{n}={r['full']['us'] / 1e3:.2f}ms(att {r['attention']['us'] / 1e3:.2f},{r['attn_backend']})"
            for n, r in recs["vq"].items()))
        recs["vq_pre"], recs["hyb_pre"] = {}, {}
        aff = Affine(NL, HKV, D)
        for name in a.pre_vq:
            kw, kst = VQFMTS[name]
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode="vq", **kw)
            r = measure(m, kv, B, ctx, 1, a.iters, a.reps, kst=kst, senc=kst, enc_Ms=Ms, aff=aff)
            recs["vq_pre"][name] = r
            if name in a.hyb_vq:
                for R in a.windows:
                    wc = KVCache(NL, B, HKV, R + 16 + 8, mode="bf16")
                    st = Step(m, kv, B, ctx, 1, "pre", r["splits"], r["chunk"], kst, kst, wc=wc, wlen=R + 16, aff=aff)
                    recs["hyb_pre"][f"{name}+r{R}"] = {"full": gtime(st.full, a.iters, a.reps),
                                                       "attention": gtime(st.attention, a.iters, a.reps),
                                                       "attn_backend": "pre", "wlen": R + 16}
                    del st, wc
            del kv
            torch.cuda.empty_cache()
        del aff
        log(f"[vq_pre] B={B} ctx={ctx} " + " ".join(
            f"{n}={r['full']['us'] / 1e3:.2f}ms(att {r['attention']['us'] / 1e3:.2f})" for n, r in recs["vq_pre"].items())
            + " | hyb " + " ".join(f"{n}={r['full']['us'] / 1e3:.2f}" for n, r in recs["hyb_pre"].items()))
        recs["gs"], recs["hyb_gs"] = {}, {}
        for name in a.gs:
            kw, kst = GSFMTS[name]
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode="vq", **kw)
            bk = GSBooks(kv)
            r = measure(m, kv, B, ctx, 1, a.iters, a.reps, kst=kst, senc=kst, enc_Ms=Ms, aff=bk)
            recs["gs"][name] = r
            if name in a.gs_hyb:
                for R in a.windows:
                    wc = KVCache(NL, B, HKV, R + 16 + 8, mode="bf16")
                    st = Step(m, kv, B, ctx, 1, "pre", r["splits"], r["chunk"], kst, kst, wc=wc, wlen=R + 16, aff=bk)
                    recs["hyb_gs"][f"{name}+r{R}"] = {"full": gtime(st.full, a.iters, a.reps),
                                                      "attention": gtime(st.attention, a.iters, a.reps),
                                                      "attn_backend": "pre-gs", "wlen": R + 16}
                    del st, wc
            del kv, bk
            torch.cuda.empty_cache()
        log(f"[gs] B={B} ctx={ctx} " + " ".join(
            f"{n}={r['full']['us'] / 1e3:.2f}ms(att {r['attention']['us'] / 1e3:.2f} enc1tok {r['append']['us'] / 1e3:.2f})"
            for n, r in recs["gs"].items()) + " | hyb " + " ".join(f"{n}={r['full']['us'] / 1e3:.2f}"
                                                                  for n, r in recs["hyb_gs"].items()))
        recs["hyb"] = {}
        for name in a.hyb_vq:
            kw, kst = VQFMTS[name]
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode="vq", **kw)
            be = recs["vq"][name]
            for R in a.windows:
                wc = KVCache(NL, B, HKV, R + 16 + 8, mode="bf16")
                st = Step(m, kv, B, ctx, 1, be["attn_backend"], be["splits"], be["chunk"], kst, kst, wc=wc, wlen=R + 16)
                recs["hyb"][f"{name}+r{R}"] = {"full": gtime(st.full, a.iters, a.reps),
                                               "attention": gtime(st.attention, a.iters, a.reps),
                                               "append": gtime(st.encode, a.iters, a.reps),
                                               "attn_backend": be["attn_backend"], "wlen": R + 16}
                del st, wc
            del kv
            torch.cuda.empty_cache()
        for mode in a.int:
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode=mode)
            for R in a.int_windows:
                wc = KVCache(NL, B, HKV, R + 16 + 8, mode="bf16")
                sp, ch = split_plan(B, HKV, ctx + 1, 128, 1024)
                st = Step(m, kv, B, ctx, 1, "triton", sp, ch, None, None, wc=wc, wlen=R + 16)
                recs["hyb"][f"{mode}+r{R}"] = {"full": gtime(st.full, a.iters, a.reps),
                                               "attention": gtime(st.attention, a.iters, a.reps),
                                               "attn_backend": "triton", "wlen": R + 16}
                del st, wc
            del kv
            torch.cuda.empty_cache()
        log(f"[hyb] B={B} ctx={ctx} " + " ".join(f"{n}={r['full']['us'] / 1e3:.2f}ms" for n, r in recs["hyb"].items()))
        recs["int"] = {}
        for mode in a.int:
            kv = KVCache(NL, B, HKV, ctx + maxM + 8, mode=mode)
            recs["int"][mode] = measure(m, kv, B, ctx, 1, a.iters, a.reps)
            del kv
            torch.cuda.empty_cache()
        log(f"[int] B={B} ctx={ctx} " + " ".join(f"{n}={r['full']['us'] / 1e3:.2f}ms" for n, r in recs["int"].items()))
        c = cotenants()
        recs["cotenants_after"] = c
        if c:
            log(f"[warn] co-tenant appeared: {c}")
        out["records"].append(recs)
        json.dump(out, open(os.path.join(out_dir, f"timing_{a.run}.json.partial"), "w"))
    out["cotenants_end"] = cotenants()
    json.dump(out, open(os.path.join(out_dir, f"timing_{a.run}.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
