"""E7 magicvq round 3: decode-step timing of bf16 Llama-3.1-8B self-speculation on 4 RTX 4090 (GPUs 4-7), synthetic
weights of the real shapes (timing only), in the KV-dominated regime (B up to 128, ctx up to 64k where KV fits).

Deployments
  tp : tensor parallel over 4 ranks (torchrun, NCCL with P2P disabled -> via host/SHM; PCIe/SYS topology): per rank
       8 q-heads / 2 kv-heads, MLP 3584, lm_head vocab/4; 2 all_reduces per layer (+1 small for the global argmax).
       Every rank holds 1/4 of the weights and the KV of its 2 kv-heads for all 32 layers.
  pp : pipeline layer split (8 layers per GPU): this process times ONE stage (8 layers, all heads) on its GPU; the
       driver runs it on GPUs 4-7 in parallel and sums the 4 stages + lm_head + 3 activation hops (measured with
       --mode hops). Sequential stages = latency of one decode step on a hypothetical 96 GB device (no overlap).
Per (B, ctx) and step kind (CUDA graph of `iters` steps, `reps` replays, median/min/max, warm replay excluded):
  target M=1 and verify M=gamma+1 (bf16 KV; attention = faster of kvq Triton split-K and SDPA-flash GQA-folded)
  drafts (M=1, own-token KV append / encode included):
    evict<W>       StreamingLLM / SnapKV: bf16 attention over W tokens (SnapKV's appended generated tokens ignored)
    rq<b>[+r<R>]   plain RQ post-RoPE (specrq/kvq LUT/WMMA kernels; d8K256 1/2 bit, d16K256 0.5 bit) [+ bf16 window]
    kivi<b>+r<R>   KIVI g32 (kivi_attn.py, group-wise scales, R-token bf16 residual window; V token quant + amortized
                   K group quant per step); R >= 32 (a partial K group is always bf16 in KIVI)
    gs<b>[+r<R>]   GSRQ pre-RoPE per-position codebooks (kvq_prerope.py GS kernel; unoptimized) + torch encoder
    cvq1           CommVQ 1-bit official Triton kernels + official encoders (pp only; see commvq_time.py)
    hyb <rq><W>    (v4b) SnapKV+RQ hybrid: bf16 attention over W tokens (SnapKV-selected set incl. its 32-token window)
                   + RQ LUT/WMMA kernel over the other ctx - W tokens, merged by the two-set LSE combine (kvq_hybrid).
                   Own generated tokens go to the bf16 part -> no per-round encode (enc = 0); the bf16 part is timed at
                   a fixed W tokens (generated tokens appended during decoding ignored, as for evict<W> / SnapKV).
  enc_M<g+1>: encoding the gamma+1 verified tokens into the draft cache once per round (VQ / KIVI / GSRQ).
Usage (driver): tools/run_tmux.sh ... python experiments/26_magicvq/mgpu_bench.py --mode tp --run v1  (via torchrun)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from specrq.kvq.kernels import (KVCache, attn_workspace, dense_append, inv_freq_llama, lut_ok, split_plan,  # noqa: E402
                                vq_attention, vq_encode)
from specrq.nestkern.gpucheck import cotenants  # noqa: E402
from kvq_hybrid import hybrid_attention  # noqa: E402
from kivi_attn import GRP, KIVICache, kivi_attention, kivi_quant_k_group, kivi_quant_v  # noqa: E402
from kvq_prerope import GSBooks, gs_encode_step, prerope_attention  # noqa: E402

HID, D, VOCAB, EPS = 4096, 128, 128256, 1e-5
OUT = "/data/lsh/results_accept/magicvq/mgpu"
DIST = None


def barrier():
    if DIST is not None:
        DIST.barrier()


def gtime(fn, iters, reps):
    fn()
    torch.cuda.synchronize()
    barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    barrier()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        barrier()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    del g
    if DIST is not None:  # global step time = slowest rank
        t = torch.tensor(ts, device="cuda")
        DIST.all_reduce(t, op=DIST.ReduceOp.MAX)
        ts = t.tolist()
    return {"us": statistics.median(ts), "min": min(ts), "max": max(ts), "reps": reps, "iters": iters}


def nvsmi_apps():
    """raw `nvidia-smi --query-compute-apps` rows (all GPUs) with the GPU index, for the co-tenant record."""
    import subprocess
    idx = dict(r.split(", ") [::-1] for r in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                                           capture_output=True, text=True).stdout.strip().splitlines())
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "apps": [{"pid": int(p), "mem": mem, "gpu": idx.get(u.strip())} for p, mem, u in
                     (x.split(", ") for x in rows if x.strip())]}


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


class Model:
    """synthetic shard: nl layers, hq q-heads, hkv kv-heads, inter MLP width, vocab rows of lm_head."""

    def __init__(self, nl, hq, hkv, inter, vocab, lm_head=True):
        bf = torch.bfloat16
        mk = lambda *s: torch.empty(*s, device="cuda", dtype=bf).normal_(0, 0.02)  # noqa: E731
        self.nl, self.hq, self.hkv, self.inter = nl, hq, hkv, inter
        self.G = hq // hkv
        self.qkvn = (hq + 2 * hkv) * D
        self.embed = mk(VOCAB, HID)
        self.lm_head = mk(vocab, HID) if lm_head else None
        self.norm = torch.ones(HID, device="cuda", dtype=bf)
        self.layers = [{"qkv": mk(self.qkvn, HID), "o": mk(HID, hq * D), "gu": mk(2 * inter, HID),
                        "down": mk(HID, inter), "n1": torch.ones(HID, device="cuda", dtype=bf),
                        "n2": torch.ones(HID, device="cuda", dtype=bf)} for _ in range(nl)]
        self.inv = inv_freq_llama(D)


def allreduce(x):
    if DIST is not None:
        DIST.all_reduce(x)
    return x


class Step:
    """one decode / verify / draft step of B sequences x M rows over the model shard.
    kind: 'bf16' (kv: KVCache bf16; backend sdpa|triton), 'vq' (kvq), 'kivi', 'gs', 'cvq'; wc: bf16 window cache
    (hybrids); enc: callable(qkv, layer) for the draft's own-token encode (None -> standard append)."""

    def __init__(self, m: Model, kind, kv, B, ctx, M, backend="triton", splits=1, chunk=64, kst=None, wc=None,
                 wlen=0, aux=None, lm=True):
        self.m, self.kind, self.kv, self.B, self.ctx, self.M = m, kind, kv, B, ctx, M
        self.R = B * M
        self.backend, self.splits, self.chunk, self.kst, self.wc, self.wlen, self.aux, self.lm = \
            backend, splits, chunk, kst, wc, wlen, aux, lm
        self.tokens = torch.randint(0, VOCAB, (self.R,), device="cuda")
        self.ws = attn_workspace(B, m.hkv, m.G, M, max(1, splits) * 16)
        if wc is not None:
            self.wsplits, self.wchunk = split_plan(B, m.hkv, wlen + M, 64, 256)
            self.ws2 = attn_workspace(B, m.hkv, m.G, M, self.wsplits)
        self.out = torch.empty(self.R, m.hq * D, device="cuda", dtype=torch.bfloat16)
        self.qkv_d = torch.randn(self.R, m.qkvn, device="cuda").to(torch.bfloat16)
        pos = torch.arange(ctx, ctx + M, device="cuda").float()
        ang = pos[:, None] * m.inv[None]
        self.cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.bfloat16)
        self.sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.bfloat16)
        self.amax = torch.zeros(self.R, device="cuda")

    # ---------------- per-layer pieces
    def append(self, qkv, li):
        m, k = self.m, self.kind
        if self.wc is not None:  # own token -> bf16 window
            dense_append(qkv, self.wc, li, self.wlen, self.M, m.hq, m.inv)
        if k == "bf16":
            if self.wc is None:
                dense_append(qkv, self.kv, li, self.ctx, self.M, m.hq, m.inv)
        elif k == "vq":
            if self.wc is None:
                vq_encode(qkv, self.kv, li, self.ctx, self.M, m.hq, m.inv, self.kst)
        elif k == "gs":
            if self.wc is None:
                self._gs_enc(qkv, li)
        elif k == "kivi":  # V of the new token quantized per token; K enters the bf16 residual (window cache)
            self.aux["vq"](qkv[:, (m.hq + m.hkv) * D:].view(self.R, m.hkv, D))
        elif k == "cvq":
            self.aux.encode()

    def _gs_enc(self, qkv, li):
        m, B, M, nsv = self.m, self.B, self.M, self.kv.nsv
        for kvi in range(2):
            tgt = self.kv.codes[li][kvi]

            def put(s, c, tgt=tgt):
                tgt[s, :, :, self.ctx:self.ctx + M].copy_(c.t().reshape(B, M, m.hkv, nsv).permute(0, 2, 1, 3))
            gs_encode_step(qkv[:, (m.hq + kvi * m.hkv) * D:(m.hq + (kvi + 1) * m.hkv) * D], self.aux, li, kvi,
                           self.kst, put)

    def attn(self, qkv, li):
        m, k = self.m, self.kind
        if k == "cvq":
            return self.aux.attention()
        if self.wc is not None:
            be = {"vq": self.backend, "gs": "pre", "kivi": "kivi"}[k]
            return hybrid_attention(qkv, self.kv, self.wc, li, self.ctx, self.wlen, self.M, m.G, m.inv, self.out,
                                    self.ws, self.ws2, self.splits, self.chunk, self.wsplits, self.wchunk, self.kst,
                                    be, self.aux if k == "gs" else None)
        if k == "gs":
            return prerope_attention(qkv, self.kv, self.aux, li, self.ctx, self.M, m.G, m.inv, self.out, self.ws,
                                     self.splits, self.chunk, self.kst)
        if k == "kivi":
            return kivi_attention(qkv, self.kv, li, self.ctx, self.M, m.G, m.inv, self.out, self.ws, self.splits,
                                  self.chunk)
        if self.backend == "sdpa":
            B, M = self.B, self.M
            q = qkv[:, :m.hq * D].view(B, M, m.hq, D)
            q1, q2 = q[..., :D // 2], q[..., D // 2:]
            qr = q * self.cos[None, :, None] + torch.cat([-q2, q1], -1) * self.sin[None, :, None]
            qr = qr.view(B, M, m.hkv, m.G, D).permute(0, 2, 3, 1, 4).reshape(B, m.hkv, m.G * M, D)
            n = self.ctx + M
            o = F.scaled_dot_product_attention(qr, self.kv.data[li][0][:, :, :n], self.kv.data[li][1][:, :, :n])
            self.out.view(B, M, m.hkv, m.G, D).copy_(o.view(B, m.hkv, m.G, M, D).permute(0, 3, 1, 2, 4))
            return self.out
        vq_attention(qkv, self.kv, li, self.ctx, self.M, m.G, m.inv, self.out, self.ws, self.splits, self.chunk,
                     self.kst, backend=self.backend)
        return self.out

    def full(self):
        m = self.m
        h = F.embedding(self.tokens, m.embed)
        for li, L in enumerate(m.layers):
            qkv = torch.mm(rms(h, L["n1"]), L["qkv"].t())
            self.append(qkv, li)
            o = self.attn(qkv, li if self.kind != "cvq" else 0)
            if self.kind == "cvq":
                o = o.transpose(1, 2).reshape(self.R, -1)
            h = h + allreduce(torch.mm(o, L["o"].t()))
            gu = torch.mm(rms(h, L["n2"]), L["gu"].t())
            h = h + allreduce(torch.mm(F.silu(gu[:, :m.inter]) * gu[:, m.inter:], L["down"].t()))
        if self.lm and m.lm_head is not None:
            mx = torch.mm(rms(h, m.norm), m.lm_head.t()).max(-1).values.float()
            self.amax.copy_(allreduce(mx))
        return h

    def attention_only(self):
        for li in range(self.m.nl):
            self.attn(self.qkv_d, li if self.kind != "cvq" else 0)


def backends(kind):
    return {"bf16": ("sdpa", "triton"), "vq": ("lut", "mmaw", "mma", "cuda"), "kivi": ("kivi",), "gs": ("pre",),
            "cvq": ("cvq",)}[kind]


def tune(m, kind, kv, B, ctx, M, kst, iters, reps, aux=None):
    n, res = ctx + M, {}
    for be in backends(kind):
        if kind == "vq" and be == "lut" and not lut_ok(kv, kst, M):
            continue
        if be in ("sdpa", "cvq"):
            st = Step(m, kind, kv, B, ctx, M, be, aux=aux)
            res[be] = {"backend": be, "splits": 1, "chunk": n, **gtime(st.attention_only, iters, reps)}
            del st
            continue
        seen = set()
        bn = GRP if kind == "kivi" else 64 if kind == "gs" else 128
        for tp in (128, 512, 2048):
            sp, ch = split_plan(B, m.hkv, n, bn, tp)
            if (sp, ch) in seen:
                continue
            seen.add((sp, ch))
            try:
                st = Step(m, kind, kv, B, ctx, M, be, sp, ch, kst, aux=aux)
                res[f"{be}_tp{tp}"] = {"backend": be, "splits": sp, "chunk": ch, **gtime(st.attention_only, iters, reps)}
                del st
            except Exception as e:  # noqa: BLE001
                res[f"{be}_tp{tp}"] = {"backend": be, "error": repr(e)[:200]}
    ok = [r for r in res.values() if "us" in r]
    return min(ok, key=lambda r: r["us"]), res


def measure(m, kind, kv, B, ctx, M, a, kst=None, aux=None, windows=(None,), enc_Ms=(), lm=True):
    best, tr = tune(m, kind, kv, B, ctx, M, kst, a.iters, a.reps, aux)
    out = {}
    for R in windows:
        wc = KVCache(m.nl, B, m.hkv, R + 16 + 8, mode="bf16") if R is not None else None
        st = Step(m, kind, kv, B, ctx, M, best["backend"], best["splits"], best["chunk"], kst, wc,
                  (R + 16) if R is not None else 0, aux, lm)
        out["base" if R is None else f"r{R}"] = {"full": gtime(st.full, a.iters, a.reps), "backend": best["backend"],
                                                 "splits": best["splits"]}
        del st, wc
    out["attn_tune"] = tr
    for Me in enc_Ms:  # encode of Me verified tokens into the draft cache, all layers
        st = Step(m, kind, kv, B, ctx, Me, "triton", kst=kst, aux=aux)
        if kind == "kivi":
            fn = lambda st=st: [aux["vq"](st.qkv_d[:, (m.hq + m.hkv) * D:].view(st.R, m.hkv, D))  # noqa: E731
                                for _ in range(m.nl)]
        elif kind == "gs":
            fn = lambda st=st: [st._gs_enc(st.qkv_d, li) for li in range(m.nl)]  # noqa: E731
        elif kind == "cvq":
            fn = lambda: [aux.encode() for _ in range(m.nl)]  # noqa: E731
        else:
            fn = lambda st=st: [vq_encode(st.qkv_d, kv, li, ctx, Me, m.hq, m.inv, kst) for li in range(m.nl)]  # noqa: E731
        out[f"enc_M{Me}"] = gtime(fn, a.iters, a.reps)
        del st
    torch.cuda.empty_cache()
    return out


def kivi_aux(m, B, bits):
    """per-step KIVI encode: V token quant (B*M rows); K group quant of one 32-token group per layer, amortized /32."""
    kgrp = torch.randn(B, m.hkv, GRP, D, device="cuda").to(torch.bfloat16)

    def vq(v):
        kivi_quant_v(v, bits)
    return {"vq": vq, "kgrp": kgrp, "kq": lambda: kivi_quant_k_group(kgrp, bits)}


def main():
    global DIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tp", "pp", "hops"], required=True)
    ap.add_argument("--grid", nargs="+", default=[f"{b}x{c}" for b in (1, 8, 16, 32, 64, 128)
                                                  for c in (8192, 16384, 32768, 65536)])
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    ap.add_argument("--rq", nargs="+", default=["rq1", "rq2", "rq05"])
    ap.add_argument("--rq_windows", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--kivi", type=int, nargs="*", default=[2, 4])
    ap.add_argument("--kivi_windows", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--gs", nargs="*", default=["gs1", "gs2", "gs05"])
    ap.add_argument("--hyb_rq", nargs="*", default=[], help="v4b: SnapKV+RQ hybrid drafts, e.g. rq2")
    ap.add_argument("--hyb_windows", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--gs_windows", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--cvq", action="store_true")
    ap.add_argument("--no_lm", action="store_true", help="pp: stage without lm_head (only the last stage has it)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--margin_gib", type=float, default=2.5)
    ap.add_argument("--tag", "--run_name", dest="run", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rank, world = 0, 1
    if a.mode == "tp":
        import torch.distributed as dist
        dist.init_process_group("nccl")
        DIST = dist
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    log = (lambda s: print(time.strftime("%H:%M:%S"), f"[r{rank}]", s, flush=True)) if rank == 0 else (lambda s: None)
    own = {os.getpid()}
    if DIST is not None:  # exclude this job's own ranks from the co-tenant check
        lst = [None] * world
        DIST.all_gather_object(lst, os.getpid())
        own = set(lst)
    cot = lambda: [c for c in cotenants() if c[0] not in own]  # noqa: E731
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.mode}_{a.run}" + ("" if a.mode == "tp" else f"_gpu{os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if a.mode == "hops":
        res = {}
        for R in (1, 8, 16, 32, 64, 128, 256, 512, 1152):
            x = torch.randn(R, HID, device="cuda:0").to(torch.bfloat16)
            y = torch.empty(R, HID, device="cuda:1", dtype=torch.bfloat16)
            for _ in range(5):
                y.copy_(x)
            torch.cuda.synchronize("cuda:0"); torch.cuda.synchronize("cuda:1")
            ts = []
            for _ in range(50):
                t = time.perf_counter()
                y.copy_(x)
                torch.cuda.synchronize("cuda:1")
                ts.append((time.perf_counter() - t) * 1e6)
            res[R] = {"us": statistics.median(ts), "min": min(ts), "max": max(ts)}
            print("hop", R, res[R], flush=True)
        json.dump({"hops": res, "cotenants": cotenants(), "visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
                  open(f"{OUT}/hops_{a.run}.json", "w"), indent=1)
        return
    if a.mode == "tp":
        m = Model(32, 32 // world, 8 // world, 14336 // world, VOCAB // world)
    else:
        m = Model(8, 32, 8, 14336, VOCAB, lm_head=not a.no_lm)
    co0 = cot()
    nv0 = nvsmi_apps()
    torch.cuda.synchronize()
    free = torch.cuda.mem_get_info()[0]
    log(f"model shard built: {torch.cuda.memory_allocated() / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB, "
        f"co-tenants {co0 or 'none'}")
    kvtok = m.nl * 2 * m.hkv * D * 2
    Ms = [g + 1 for g in a.gammas]
    out = {"config": vars(a), "world": world, "shard": {"nl": m.nl, "hq": m.hq, "hkv": m.hkv, "inter": m.inter},
           "cotenants_start": co0, "nvsmi_apps_start": nv0, "fit": [], "records": []}
    path = f"{OUT}/bench_{tag}.json"
    for gspec in a.grid:
        B, ctx = (int(x) for x in gspec.split("x"))
        T = ((ctx + max(Ms) + 8 + GRP - 1) // GRP) * GRP
        need = B * T * kvtok
        free = torch.cuda.mem_get_info()[0]
        fits = need + a.margin_gib * 2**30 <= free  # drafts are allocated after the target cache is freed
        if DIST is not None:
            f = torch.tensor([1 if fits else 0], device="cuda")
            DIST.all_reduce(f, op=DIST.ReduceOp.MIN)
            fits = bool(f.item())
        out["fit"].append({"B": B, "ctx": ctx, "target_kv_GiB_per_gpu": need / 2**30, "free_GiB": free / 2**30,
                           "fits": fits})
        log(f"[fit] B={B} ctx={ctx} target KV/GPU {need / 2**30:.2f} GiB free {free / 2**30:.2f} -> {fits}")
        if not fits:
            continue
        rec = {"B": B, "ctx": ctx, "co_before": cot()}
        t0 = time.time()
        kv = KVCache(m.nl, B, m.hkv, T, mode="bf16")
        rec["target"] = {M: measure(m, "bf16", kv, B, ctx, M, a)["base"] | {} for M in [1] + Ms}
        del kv
        torch.cuda.empty_cache()
        log(f"[target] B={B} ctx={ctx} " + " ".join(f"M{M}={r['full']['us'] / 1e3:.2f}ms" for M, r in rec["target"].items()))
        # eviction drafts (bf16 window of W tokens; W <= ctx)
        rec["evict"] = {}
        for W in a.budgets:
            if W >= ctx:
                continue
            kv = KVCache(m.nl, B, m.hkv, W + 8, mode="bf16")
            rec["evict"][W] = measure(m, "bf16", kv, B, W - 1, 1, a)["base"]
            del kv
        log(f"[evict] " + " ".join(f"W{W}={r['full']['us'] / 1e3:.2f}" for W, r in rec["evict"].items()))
        # plain RQ post-RoPE (kvq)
        rec["rq"] = {}
        for name in a.rq:
            kw, kst = {"rq1": (dict(d=8, bits=[8, 8]), 1), "rq2": (dict(d=8, bits=[8, 8]), 2),
                       "rq05": (dict(d=16, bits=[8, 8]), 1)}[name]
            kv = KVCache(m.nl, B, m.hkv, T, mode="vq", **kw)
            rec["rq"][name] = measure(m, "vq", kv, B, ctx, 1, a, kst=kst, windows=[None] + a.rq_windows, enc_Ms=Ms)
            del kv
            torch.cuda.empty_cache()
        log(f"[rq] " + " ".join(f"{n}:{r['base']['full']['us'] / 1e3:.2f}/{r['r128']['full']['us'] / 1e3:.2f}"
                                 for n, r in rec["rq"].items()))
        # KIVI g32 + bf16 residual
        rec["kivi"] = {}
        for bits in a.kivi:
            kv = KIVICache(m.nl, B, m.hkv, T, bits)
            aux = kivi_aux(m, B, bits)
            r = measure(m, "kivi", kv, B, ctx, 1, a, aux=aux, windows=a.kivi_windows, enc_Ms=Ms)
            r["kgroup_quant_per_layer"] = gtime(lambda: [aux["kq"]() for _ in range(m.nl)], a.iters, a.reps)
            rec["kivi"][bits] = r
            del kv, aux
            torch.cuda.empty_cache()
        log(f"[kivi] " + " ".join(f"{b}b:" + "/".join(f"{r[k]['full']['us'] / 1e3:.2f}" for k in r if k.startswith("r"))
                                   for b, r in rec["kivi"].items()))
        # GSRQ (unoptimized pre-RoPE GS kernel)
        rec["gs"] = {}
        for name in a.gs:
            kw, kst = {"gs1": (dict(d=8, bits=[8, 8]), 1), "gs2": (dict(d=8, bits=[8, 8]), 2),
                       "gs05": (dict(d=16, bits=[8, 8]), 1)}[name]
            kv = KVCache(m.nl, B, m.hkv, T, mode="vq", **kw)
            bk = GSBooks(kv)
            rec["gs"][name] = measure(m, "gs", kv, B, ctx, 1, a, kst=kst, aux=bk, windows=[None] + a.gs_windows,
                                      enc_Ms=Ms)
            del kv, bk
            torch.cuda.empty_cache()
        log(f"[gs] " + " ".join(f"{n}:{r['base']['full']['us'] / 1e3:.2f}" for n, r in rec["gs"].items()))
        if a.cvq and a.mode == "pp":
            from commvq_time import CommVQLayerState
            try:
                cs = CommVQLayerState(B, ctx)
                rec["cvq"] = {"cvq1": measure(m, "cvq", None, B, ctx, 1, a, aux=cs, enc_Ms=[1])}
                rec["cvq"]["note"] = "official kernels; one layer state reused by the 8 layers of the stage"
                log(f"[cvq] {rec['cvq']['cvq1']['base']['full']['us'] / 1e3:.2f} ms")
                del cs
            except Exception as e:  # noqa: BLE001
                rec["cvq"] = {"error": repr(e)[:300]}
                log(f"[cvq] error {e!r}"[:300])
            torch.cuda.empty_cache()
        # SnapKV+RQ hybrid (v4b): RQ over ctx - W tokens + bf16 W tokens, LSE merge; no per-round encode
        if a.hyb_rq:
            rec["hyb"] = {}
        for name in a.hyb_rq:
            kw, kst = {"rq1": (dict(d=8, bits=[8, 8]), 1), "rq2": (dict(d=8, bits=[8, 8]), 2),
                       "rq05": (dict(d=16, bits=[8, 8]), 1)}[name]
            kv = KVCache(m.nl, B, m.hkv, T, mode="vq", **kw)
            rec["hyb"][name] = {}
            for W in a.hyb_windows:
                if W >= ctx:
                    continue
                r = measure(m, "vq", kv, B, ctx - W, 1, a, kst=kst, windows=[W - 16])  # window cache wlen = W
                rec["hyb"][name][W] = {"full": r[f"r{W - 16}"]["full"], "backend": r[f"r{W - 16}"]["backend"],
                                       "splits": r[f"r{W - 16}"]["splits"], "vq_tokens": ctx - W, "bf16_tokens": W,
                                       "attn_tune": r["attn_tune"]}
            del kv
            torch.cuda.empty_cache()
        if a.hyb_rq:
            log(f"[hyb] " + " ".join(f"{n}w{W}:{r['full']['us'] / 1e3:.2f}" for n, x in rec["hyb"].items()
                                     for W, r in x.items()))
        rec["co_after"] = cot()
        rec["wall_s"] = time.time() - t0
        out["records"].append(rec)
        if rank == 0:
            json.dump(out, open(path + ".partial", "w"))
    out["cotenants_end"] = cot()
    out["nvsmi_apps_end"] = nvsmi_apps()
    if rank == 0 or a.mode == "pp":
        json.dump(out, open(path, "w"), indent=1)
    log("done")
    if DIST is not None:
        DIST.barrier()


if __name__ == "__main__":
    main()
