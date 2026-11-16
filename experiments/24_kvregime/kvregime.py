"""E6 kvregime: decode cost structure of a SYNTHETIC Llama-3.1-8B (shapes only; all tensors random, timing only, no
quality) with 2-bit nested VQ weights (C1-style d8 K256 L2, nestg GenLinear, Hadamard B=256) and a nested VQ KV cache,
at batch B x context ctx, batch of B sequences each with M new rows (M = 1 AR / draft step, M = gamma+1 verify).

Per layer (fused QKV [6144x4096] and gate|up [28672x4096] linears, glue kernels from specrq.nestkern):
  add_rms_rot -> QKV (W prefix kw) -> KV encode (senc stages, K RoPE fused) -> VQ attention (kst stages, q RoPE fused,
  flash-decoding split-K) -> o_proj (FWHT + W) -> add_rms_rot -> gate|up -> act_rot -> down -> ...; final norm,
  full-vocab lm_head (R rows), argmax.
Weights: per (shape, rows R, kw) the faster of nestg MMA (16-row chunks for R > 16) and dequant-to-bf16 (Triton) +
cuBLAS, chosen by a 32-layer microbenchmark (both reported). The dequant path reads the nestg MMA byte planes as if they
were logical codes (same bytes, permuted values: timing-equivalent; the dequant kernel itself is checked on logical
codes in check_kernels.py).
KV formats: vq d8 S2 (2 bit, stage = 1 bit), vq d16 S4 (2 bit, stage = 0.5 bit), bf16 (where it fits);
INT4 / INT2 scalar-KV attention kernels timed as bandwidth proxies (attention only, no append).
Timing: CUDA graphs of `iters` steps, `reps` replays, median (and range); components (weights / attention / encode /
fixed glue+lm_head) timed as separate 32-layer graphs; the full step graph is the T used for c.
Usage: python experiments/24_kvregime/kvregime.py --grid 1x2048 8x8192 ... --run NAME
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from specrq.infer.llama_cost import req_alpha, rss_gb  # noqa: E402
from specrq.infer.model import GenLinear  # noqa: E402
from specrq.kvq.kernels import (FORMATS, KVCache, attn_workspace, dense_append, inv_freq_llama, lut_ok,  # noqa: E402
                                split_plan,
                                vq_attention, vq_dequant_weight, vq_encode)
from specrq.nestkern.gpucheck import cotenants  # noqa: E402
from specrq.nestkern.ops import act_rot, add_rms_rot, block_fwht_signed  # noqa: E402

NL, HID, H, HKV, D, INTER, VOCAB = 32, 4096, 32, 8, 128, 14336, 128256
G = H // HKV
EPS = 1e-5
VD = 32768
SHAPES = {"qkv": ((H + 2 * HKV) * D, HID), "o": (HID, H * D), "gu": (2 * INTER, HID), "down": (HID, INTER)}


def gtime(fn, iters, reps):
    """median / min / max us per call of fn over `reps` replays of a CUDA graph holding `iters` calls."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    g.replay()  # warm replay (excluded)
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
    return {"us": statistics.median(ts), "min": min(ts), "max": max(ts)}


class WLin:
    """GenLinear (nestg) + dequant path on the same bytes."""

    def __init__(self, gl: GenLinear, scratch, gen):
        self.gl, self.N, self.K = gl, gl.N, gl.K
        gv = gl.g
        self.codes = gv.planes["mma"]["p0"].view(gv.L, gv.N, gv.K // gv.d)
        self.cb = (torch.randn(gv.L, 256, gv.d, device="cuda", generator=gen) * 0.01).to(torch.bfloat16)
        self.scales, self.scratch = gv.scales, scratch

    def __call__(self, x, kw, path, rotated=True):
        if not rotated:
            x = block_fwht_signed(x, self.gl.signs, self.gl.hb)
        if path == "nest":
            return self.gl(x, kw, rotated=True)
        W = self.scratch[:self.N * self.K].view(self.N, self.K)
        vq_dequant_weight(self.codes, self.cb, self.scales, kw, W)
        return torch.mm(x, W.t())


class Model:
    def __init__(self, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gen = gen
        bf = torch.bfloat16
        self.embed = torch.randn(VOCAB, HID, device="cuda", generator=gen, dtype=bf) * 0.02
        self.lm_head = torch.randn(VOCAB, HID, device="cuda", generator=gen, dtype=bf) * 0.02
        # FR-Spec draft vocabulary (V_d = 32768 rows; synthetic: a separate random slice, same cost as a gathered one)
        self.lm_head_d = torch.randn(VD, HID, device="cuda", generator=gen, dtype=bf) * 0.02
        self.norm = torch.ones(HID, device="cuda", dtype=bf)
        self.scratch = torch.empty(max(n * k for n, k in SHAPES.values()), device="cuda", dtype=bf)
        self.layers = []
        for i in range(NL):
            lay = {"in_norm": torch.ones(HID, device="cuda", dtype=bf),
                   "post_norm": torch.ones(HID, device="cuda", dtype=bf)}
            for n, (N, K) in SHAPES.items():
                gl = GenLinear.synthetic(N, K, torch.device("cuda"), bf, 8, [8, 8], 16, 256, gen=gen)
                lay[n] = WLin(gl, self.scratch, gen)
            lay["s_qkv"] = lay["qkv"].gl.signs[None].contiguous()
            lay["s_gu"] = lay["gu"].gl.signs[None].contiguous()
            self.layers.append(lay)
        self.inv = inv_freq_llama(D)
        self.paths = {}  # (shape, R, kw) -> "nest" | "deq"

    def lin(self, lay, name, x, kw, R, rotated=True):
        return lay[name](x, kw, self.paths.get((name, R, kw), "nest"), rotated)


class Step:
    """One decode step of B sequences x M rows. kv: KVCache; kw: weight planes; kst: KV stages read; senc: stages
    encoded for the new rows. Buffers are static (graph-capturable)."""

    def __init__(self, m: Model, kv: KVCache, B, ctx, M, kw, kst, senc, splits, chunk, backend="mma", vd=False):
        self.backend, self.vd = backend, vd
        self.m, self.kv, self.B, self.ctx, self.M, self.kw, self.kst, self.senc = m, kv, B, ctx, M, kw, kst, senc
        self.R = R = B * M
        self.splits, self.chunk = splits, chunk
        gen = m.gen
        self.tokens = torch.randint(0, VOCAB, (R,), device="cuda", generator=gen)
        self.ws = attn_workspace(B, HKV, G, M, splits * 16)
        self.attn_out = torch.empty(R, H * D, device="cuda", dtype=torch.bfloat16)
        # dummy inputs for the component graphs
        self.qkv_d = torch.randn(R, SHAPES["qkv"][0], device="cuda", generator=gen).to(torch.bfloat16)
        self.x_d = torch.randn(R, HID, device="cuda", generator=gen).to(torch.bfloat16)
        self.a_d = torch.randn(R, INTER, device="cuda", generator=gen).to(torch.bfloat16)
        self.gu_d = torch.randn(R, 2 * INTER, device="cuda", generator=gen).to(torch.bfloat16)

    # ---- pieces
    def _kv_append(self, qkv, li):
        if self.kv.mode == "vq":
            vq_encode(qkv, self.kv, li, self.ctx, self.M, H, self.m.inv, self.senc)
        elif self.kv.mode == "bf16":
            dense_append(qkv, self.kv, li, self.ctx, self.M, H, self.m.inv)

    def _attn(self, qkv, li):
        vq_attention(qkv, self.kv, li, self.ctx, self.M, G, self.m.inv, self.attn_out, self.ws, self.splits,
                     self.chunk, self.kst, backend=self.backend)
        return self.attn_out

    def full(self):
        m, R, kw = self.m, self.R, self.kw
        h = torch.nn.functional.embedding(self.tokens, m.embed)
        _, xs = add_rms_rot(h, None, m.layers[0]["in_norm"], m.layers[0]["s_qkv"], 256, EPS)
        for li, lay in enumerate(m.layers):
            qkv = m.lin(lay, "qkv", xs[0], kw, R)
            self._kv_append(qkv, li)
            o = self._attn(qkv, li)
            h, xs = add_rms_rot(h, m.lin(lay, "o", o, kw, R, rotated=False), lay["post_norm"], lay["s_gu"], 256, EPS)
            gu = m.lin(lay, "gu", xs[0], kw, R)
            a = act_rot(gu[:, :INTER], gu[:, INTER:], lay["down"].gl.signs, 256)
            dn = m.lin(lay, "down", a, kw, R)
            if li + 1 < NL:
                nx = m.layers[li + 1]
                h, xs = add_rms_rot(h, dn, nx["in_norm"], nx["s_qkv"], 256, EPS)
            else:
                h, x = add_rms_rot(h, dn, m.norm, None, 128, EPS)
        return torch.argmax(torch.nn.functional.linear(x, m.lm_head_d if self.vd else m.lm_head), -1)

    def weights(self):
        m, R, kw = self.m, self.R, self.kw
        for lay in m.layers:
            m.lin(lay, "qkv", self.x_d, kw, R)
            m.lin(lay, "o", self.x_d, kw, R, rotated=False)
            m.lin(lay, "gu", self.x_d, kw, R)
            m.lin(lay, "down", self.a_d, kw, R)

    def attention(self):
        for li in range(NL):
            self._attn(self.qkv_d, li)

    def encode(self):
        for li in range(NL):
            self._kv_append(self.qkv_d, li)

    def fixed(self):
        m = self.m
        h = torch.nn.functional.embedding(self.tokens, m.embed)
        _, xs = add_rms_rot(h, None, m.layers[0]["in_norm"], m.layers[0]["s_qkv"], 256, EPS)
        for li, lay in enumerate(m.layers):
            h, xs = add_rms_rot(h, self.x_d, lay["post_norm"], lay["s_gu"], 256, EPS)
            act_rot(self.gu_d[:, :INTER], self.gu_d[:, INTER:], lay["down"].gl.signs, 256)
            h, x = add_rms_rot(h, self.x_d, m.norm, None, 128, EPS) if li + 1 == NL else \
                add_rms_rot(h, self.x_d, m.layers[li + 1]["in_norm"], m.layers[li + 1]["s_qkv"], 256, EPS)
        return torch.argmax(torch.nn.functional.linear(x, m.lm_head_d if self.vd else m.lm_head), -1)


def choose_paths(m: Model, Rs, kws, iters, reps, log):
    """32-layer microbenchmark per (shape, R, kw): nestg vs dequant+cuBLAS (FWHT excluded: inputs pre-rotated)."""
    out = []
    for R in Rs:
        x = {n: torch.randn(R, K, device="cuda", generator=m.gen).to(torch.bfloat16) for n, (N, K) in SHAPES.items()}
        for kw in kws:
            for n in SHAPES:
                t = {}
                for path in ("nest", "deq"):
                    if path == "deq" and R < 16:
                        continue
                    t[path] = gtime(lambda: [lay[n](x[n], kw, path) for lay in m.layers], iters, reps)["us"]
                best = min(t, key=t.get)
                m.paths[(n, R, kw)] = best
                N, K = SHAPES[n]
                rec = {"shape": n, "N": N, "K": K, "R": R, "kw": kw, "us_32layers": t, "best": best,
                       "best_GBps": NL * (N * K * kw // 4) / (t[best] * 1e-6) / 1e9}
                out.append(rec)
                log(f"[w] {n} R={R} kw={kw} " + " ".join(f"{p}={v:.0f}us" for p, v in t.items()) + f" -> {best}")
    return out


def attn_backends(kv):
    if kv.mode != "vq":
        return ("triton",)
    return ("lut", "mmaw", "mma", "cuda") if kv.uniform else ("lut", "mmawp")


def tune_splits(m, kv, B, ctx, M, kst, iters, reps):
    """Attention backend (VQ: LUT, CUDA warp-split/block WMMA, CUDA scalar; others: Triton) x split count."""
    n = ctx + M
    res = {}
    for be in attn_backends(kv):
        if be == "lut" and not lut_ok(kv, kst, M):
            continue
        seen = set()
        for tp in (64, 128, 256, 512, 1024, 2048):
            sp, ch = split_plan(B, HKV, n, 128, tp)
            if (sp, ch) in seen:
                continue
            seen.add((sp, ch))
            st = Step(m, kv, B, ctx, M, 2, kst, kst, sp, ch, be)
            res[f"{be}_tp{tp}"] = {"backend": be, "splits": sp, "chunk": ch, **gtime(st.attention, iters, reps)}
            del st
    best = min(res.values(), key=lambda r: r["us"])
    return best["splits"], best["chunk"], best["backend"], res


def measure(m, kv, B, ctx, M, kw, kst, senc, iters, reps, comps=True, vd_too=False):
    sp, ch, be, tune = tune_splits(m, kv, B, ctx, M, kst, iters, reps)
    st = Step(m, kv, B, ctx, M, kw, kst, senc, sp, ch, be)
    r = {"B": B, "ctx": ctx, "M": M, "R": B * M, "kw": kw, "kst": kst, "senc": senc, "splits": sp, "chunk": ch,
         "attn_backend": be,
         "split_tune": tune, "full": gtime(st.full, iters, reps)}
    if comps:
        r["weights"] = gtime(st.weights, iters, reps)
        r["attention"] = gtime(st.attention, iters, reps)
        if kv.mode in ("vq", "bf16"):
            r["encode"] = gtime(st.encode, iters, reps)
        r["fixed"] = gtime(st.fixed, iters, reps)
        r["sum_components_us"] = sum(r[c]["us"] for c in ("weights", "attention", "encode", "fixed") if c in r)
    if vd_too:  # FR-Spec draft lm_head (V_d = 32768)
        st.vd = True
        r["full_vd"] = gtime(st.full, iters, reps)
        if comps:
            r["fixed_vd"] = gtime(st.fixed, iters, reps)
    n = ctx + M
    kv_bytes = NL * kv.bytes_read(B * n, kst if kv.mode == "vq" else None)
    r["kv_bytes_read"] = kv_bytes
    if "attention" in r:
        r["attn_GBps"] = kv_bytes / (r["attention"]["us"] * 1e-6) / 1e9
    del st
    torch.cuda.empty_cache()
    c = cotenants()
    if c:
        DIRTY.extend(c)
    return r


DIRTY = []


def wait_alone(log, poll_s=60):
    """Block until no other compute process is on our GPU; returns seconds waited."""
    t0 = time.time()
    while True:
        c = cotenants()
        if not c:
            return time.time() - t0
        log(f"[wait] co-tenant {c}; waiting")
        time.sleep(poll_s)


def derive(T_ar, T_d, T_v):
    """[I] computed from measured step times: c, break-even tau, ceiling at tau = gamma + 1 (overhead ignored)."""
    out = {"c": T_d / T_ar}
    for g, Tv in T_v.items():
        R = g * T_d + Tv
        out[f"g{g}"] = {"round_us": R, "tau_breakeven": R / T_ar, "alpha_iid_breakeven": req_alpha(R / T_ar, g),
                        "ceiling_speedup": (g + 1) * T_ar / R}
    return out


def kv_fits(B, ctx, per_token_bytes, free_bytes, margin=1.2 * 2**30):
    need = B * (ctx + 8) * per_token_bytes
    return need + margin <= free_bytes, need


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", default=[f"{b}x{c}" for b in (1, 8, 16, 32, 64) for c in (2048, 8192, 32768)])
    ap.add_argument("--Ms", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--formats", nargs="+", default=["d8S2", "d8K16x4", "d8K16x2K256"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--jsonl", default=None, help="append per-(B,ctx) records here as they finish")
    ap.add_argument("--reuse", default=None, help="jsonl of an earlier run: co-tenant-free (B, ctx) records are reused")
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--no-vd", dest="vd", action="store_false", help="skip the FR-Spec V_d draft lm_head timings")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)  # noqa: E731
    co0 = cotenants() or "none"
    log(f"co-tenants at start: {co0}")
    m = Model()
    torch.cuda.synchronize()
    wbytes = torch.cuda.memory_allocated()
    log(f"model built: {wbytes / 2**30:.2f} GiB allocated, rss {rss_gb():.1f} GB")
    grid = [tuple(int(v) for v in g.split("x")) for g in a.grid]
    Rs = sorted({b * mm for b, _ in grid for mm in [1] + a.Ms})
    for attempt in range(a.max_attempts):
        wait_alone(log)
        wpaths = choose_paths(m, Rs, [1, 2], a.iters, a.reps, log)
        c = cotenants()
        if not c:
            break
        log(f"[co-tenant] during weight-path microbench: {c} -> retry")
    total = torch.cuda.get_device_properties(0).total_memory
    res = {"weights_paths": wpaths, "weight_bytes_alloc": wbytes, "grid": [], "fit": []}
    fmts = {"bf16": ("bf16", 8, None), "int4": ("int4", 8, None), "int2": ("int2", 8, None)}
    fmts.update({k: ("vq", d, bits) for k, (d, bits) in FORMATS.items()})

    def point(B, ctx):
        rec = {"B": B, "ctx": ctx, "co_tenants_start": cotenants() or "none", "formats": {}}
        for fname in a.formats:
            mode, d, bits = fmts[fname]
            S = len(bits) if bits else 1
            gc.collect()
            torch.cuda.empty_cache()
            free = total - torch.cuda.memory_allocated() - 300 * 2**20
            per_tok = {"vq": 2 * sum(D // d * b // 8 for b in bits) if bits else 0, "bf16": 4 * D,
                       "int4": 2 * (D // 2 + 2), "int2": 2 * (D // 4 + 2)}[mode] * HKV * NL
            ok, need = kv_fits(B, ctx, per_tok, free)
            res["fit"].append({"B": B, "ctx": ctx, "format": fname, "kv_GiB": need / 2**30, "free_GiB": free / 2**30,
                               "fits": ok})
            if not ok:
                log(f"[skip] B={B} ctx={ctx} {fname}: KV {need / 2**30:.1f} GiB > free {free / 2**30:.1f} GiB")
                continue
            try:
                kv = KVCache(NL, B, HKV, ctx + 8, D, mode, d, gen=m.gen, bits=bits)
            except torch.cuda.OutOfMemoryError as e:
                res["fit"][-1]["fits"] = False
                res["fit"][-1]["oom"] = str(e)[:200]
                log(f"[oom] B={B} ctx={ctx} {fname}")
                continue
            fr = {}
            try:
                if mode == "vq":
                    # drafts: weight prefix kw in {1, 2} x KV stage prefix k < S (+ W1 with the full KV);
                    # name d_w{kw}_kv{bits}: bits = KV index bits/element read by the draft (e.g. 0.5, 1)
                    runs = {"ar": (1, 2, S, S)}
                    for k in range(1, S):
                        kb = sum(bits[:k]) / d
                        for kw in (1, 2):
                            runs[f"d_w{kw}_kv{kb:g}b"] = (1, kw, k, k)
                    runs["d_w1_kvall"] = (1, 1, S, S)
                    for M in a.Ms:
                        runs[f"v{M}"] = (M, 2, S, S)
                    for name, (M, kw, kst, senc) in runs.items():
                        fr[name] = measure(m, kv, B, ctx, M, kw, kst, senc, a.iters, a.reps,
                                           vd_too=name.startswith("d_") and a.vd)
                        log(f"B={B} ctx={ctx} {fname} {name}: full {fr[name]['full']['us']:.0f}us "
                            f"w {fr[name]['weights']['us']:.0f} attn {fr[name]['attention']['us']:.0f} "
                            f"enc {fr[name]['encode']['us']:.0f} fix {fr[name]['fixed']['us']:.0f} "
                            f"attn {fr[name]['attn_GBps']:.0f} GB/s {fr[name]['attn_backend']}"
                            + (f" | vd {fr[name]['full_vd']['us']:.0f}us" if "full_vd" in fr[name] else ""))
                    T_ar = fr["ar"]["full"]["us"]
                    Tv = {M - 1: fr[f"v{M}"]["full"]["us"] for M in a.Ms}
                    fr["derived"] = {k: derive(T_ar, fr[k]["full"]["us"], Tv) for k in fr if k.startswith("d_")}
                    fr["derived_vd"] = {k: derive(T_ar, fr[k]["full_vd"]["us"], Tv) for k in fr
                                        if k.startswith("d_") and "full_vd" in fr[k]}
                elif mode == "bf16":
                    runs = {"ar": (1, 2, 1, 1), "d_w1": (1, 1, 1, 1)}
                    for M in a.Ms:
                        runs[f"v{M}"] = (M, 2, 1, 1)
                    for name, (M, kw, kst, senc) in runs.items():
                        fr[name] = measure(m, kv, B, ctx, M, kw, kst, senc, a.iters, a.reps)
                        log(f"B={B} ctx={ctx} {fname} {name}: full {fr[name]['full']['us']:.0f}us "
                            f"attn {fr[name]['attention']['us']:.0f} ({fr[name]['attn_GBps']:.0f} GB/s)")
                    T_ar = fr["ar"]["full"]["us"]
                    Tv = {M - 1: fr[f"v{M}"]["full"]["us"] for M in a.Ms}
                    fr["derived"] = {"d_w1": derive(T_ar, fr["d_w1"]["full"]["us"], Tv)}
                else:  # scalar proxies: attention only
                    for M in [1] + a.Ms:
                        sp, ch, be, tune = tune_splits(m, kv, B, ctx, M, 1, a.iters, a.reps)
                        st = Step(m, kv, B, ctx, M, 2, 1, 1, sp, ch, be)
                        t = gtime(st.attention, a.iters, a.reps)
                        kvb = NL * kv.bytes_read(B * (ctx + M))
                        fr[f"attn_M{M}"] = {"attention": t, "splits": sp, "chunk": ch, "kv_bytes_read": kvb,
                                            "attn_GBps": kvb / (t["us"] * 1e-6) / 1e9}
                        log(f"B={B} ctx={ctx} {fname} attn M={M}: {t['us']:.0f}us "
                            f"({fr[f'attn_M{M}']['attn_GBps']:.0f} GB/s)")
                        del st
            except torch.cuda.OutOfMemoryError as e:
                fr["oom"] = str(e)[:300]
                log(f"[oom-run] B={B} ctx={ctx} {fname}: {str(e)[:120]}")
            rec["formats"][fname] = fr
            del kv
            gc.collect()
            torch.cuda.empty_cache()
        rec["co_tenants_end"] = cotenants() or "none"
        return rec

    reuse = {}
    if a.reuse:
        for line in open(a.reuse):
            r = json.loads(line)
            if r["co_tenants_start"] == "none" and r["co_tenants_end"] == "none" and not r.get("dirty"):
                reuse[(r["B"], r["ctx"])] = r
    for B, ctx in grid:
        if (B, ctx) in reuse:
            rec = dict(reuse[(B, ctx)], reused_from=a.reuse)
            log(f"[reuse] B={B} ctx={ctx} from {a.reuse}")
        else:
            for attempt in range(a.max_attempts):
                waited = wait_alone(log)
                DIRTY.clear()
                res["fit"] = [f for f in res["fit"] if (f["B"], f["ctx"]) != (B, ctx)]
                rec = point(B, ctx)
                rec["attempt"], rec["waited_s"] = attempt, waited
                rec["dirty"] = sorted({str(x) for x in DIRTY})
                if not rec["dirty"] and rec["co_tenants_start"] == "none" and rec["co_tenants_end"] == "none":
                    break
                log(f"[co-tenant] B={B} ctx={ctx} attempt {attempt}: {rec['dirty']} -> retry")
        res["grid"].append(rec)
        if a.jsonl:
            with open(a.jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
    res["peak_gpu_mem_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    res["peak_rss_gb"] = rss_gb()
    if a.no_save:
        log("no-save: done")
        return
    from tools.results import save_result
    p = save_result("E6", a.run, config={
        "setting": "SYNTHETIC Llama-3.1-8B shapes (all tensors random; timing only, no quality); fused QKV and gate|up "
                   "linears; nested VQ weights d8 K256 L2 q16 (nestg) + Hadamard B=256; nested VQ KV cache "
                   "(per layer/K-V/kv-head/stage codebook C=256, shared over sub-vectors); greedy residual encode",
        "grid": a.grid, "Ms": a.Ms, "iters": a.iters, "reps": a.reps, "formats": a.formats,
        "co_tenants_start": co0, "co_tenants_end": cotenants() or "none",
        "cpu_pin": os.environ.get("SR_PIN", "taskset -c 0-15,32-47, nice 10"),
        "weight_path_rule": "per (shape, R, kw) faster of nestg MMA and dequant+cuBLAS (32-layer microbench)",
        "derived_note": "c, tau_breakeven, ceiling are [I] computed from measured full-step graphs; overhead "
                        "(accept/rollback/bookkeeping) not included; target/verify use the full-vocab lm_head; drafts are timed with "
                        "the full lm_head (derived) and with an FR-Spec V_d=32768 lm_head (derived_vd)",
        "kv_formats": {f: FORMATS.get(f) for f in a.formats}},
        metrics=res, model_dir="/data/models/llama3.1_8b_instruct")
    log(f"saved {p}")


if __name__ == "__main__":
    main()
