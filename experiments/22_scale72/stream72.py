"""E4 scale72: SYNTHETIC Qwen2.5-72B 4-bit target on ONE RTX 4090 with layer-wise weight STREAMING (batch regime).
GPU-resident: stages 1-2 codes (C1 nest_mma layout, 16.35 GiB), lm_head bf16, norms, Hadamard signs, KV cache;
embedding on the CPU (lookup on the host, H2D of the hidden rows). Stages 3-4 codes (2 planes, 17.55 GB) live in
pinned host RAM, one contiguous chunk per layer (219 MB), streamed into NBUF device slots on a separate copy stream
(event-ordered: layer i computes after its chunk arrived; the copy of layer i+NBUF waits until layer i's compute
released the slot). All weights random (timing only). Batched forward: B sequences x M tokens, equal context.

4-bit linear = GPU k2 part (resident planes 1-2) + nest_mma over the streamed planes 3-4 (same per-row scale,
separate codebooks), summed; rows > 16 are processed in chunks of 16 (nest_mma limit), as in NestLinear.
Measured (eager, device sync around each call, warm-up excluded, median of reps):
  T4(B, M)        : full 4-bit forward incl. embedding round trip, lm_head, argmax, with copy/compute overlap
  T4_compute(B,M) : same forward, streamed planes NOT copied (stale slot data) -> compute-bound time
  T_copy          : the 80 layer chunks copied alone on the copy stream -> streaming-bound time
  T2(B)           : k2 draft / 2-bit AR step, M=1, GPU-resident, CUDA graph (+ host embedding round trip)
Usage: python experiments/22_scale72/stream72.py --Bs 1 4 16 --Ms 1 2 5 9 --run qwen72b_synth_stream_v1
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "14_q32b_upper"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from q32b_upper import gpu_util  # noqa: E402
from q72_synth import shapes_of, synth_model  # noqa: E402
from specrq.infer.decode import _Graph  # noqa: E402
from specrq.nestkern.gpucheck import cotenants  # noqa: E402
from specrq.nestkern.ops import act_rot, add_rms_rot, nest_mma  # noqa: E402
from specrq.nestkern.pack import mma_codes  # noqa: E402

MODEL = "/data/models/qwen2.5_72b_instruct"
NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GiB = 2 ** 30


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def mma_rows(x, mc, cb, sc, k):
    if x.shape[0] <= 16:
        return nest_mma(x, mc, cb, sc, k)
    return torch.cat([nest_mma(x[i:i + 16], mc, cb, sc, k) for i in range(0, x.shape[0], 16)])


class Streamer:
    def __init__(self, m, nbuf, gen):
        self.m, self.nbuf = m, nbuf
        shp = shapes_of(m.cfg)
        self.meta, off = {}, 0
        for n in NAMES:
            N, K = shp[n]
            nb = 2 * N * K // 8
            cb = (torch.randn(2, 256, 8, device="cuda", generator=gen) * (3.0 / (K * 4) ** 0.5)).to(torch.bfloat16)
            self.meta[n] = (N, K, off, nb, cb)
            off += nb
        self.LB = off
        nl = m.n_layers
        self.cbs = [{n: self.meta[n][4] for n in NAMES} for _ in range(nl)]
        t0 = time.time()
        self.host = torch.empty(nl * self.LB, dtype=torch.uint8, pin_memory=True)
        dev = torch.empty(self.LB, dtype=torch.uint8, device="cuda")
        for n in NAMES:  # random codes in the real mma layout for layer 0, replicated to all layers
            N, K, o, nb, _ = self.meta[n]
            c = torch.randint(0, 256, (2, N, K // 8), device="cuda", dtype=torch.uint8, generator=gen)
            dev[o:o + nb].copy_(mma_codes(c).reshape(-1))
            del c
        for i in range(nl):
            self.host[i * self.LB:(i + 1) * self.LB].copy_(dev)
        del dev
        torch.cuda.synchronize()
        self.fill_s = time.time() - t0
        self.slots = [torch.empty(self.LB, dtype=torch.uint8, device="cuda") for _ in range(nbuf)]
        self.cs = torch.cuda.Stream()
        self.ev_copied = [torch.cuda.Event() for _ in range(nl)]
        self.ev_free = [torch.cuda.Event() for _ in range(nbuf)]
        self.mode = "overlap"

    def issue(self, j):
        s = j % self.nbuf
        with torch.cuda.stream(self.cs):
            if j >= self.nbuf:
                self.cs.wait_event(self.ev_free[s])
            self.slots[s].copy_(self.host[j * self.LB:(j + 1) * self.LB], non_blocking=True)
            self.ev_copied[j].record(self.cs)

    def begin(self):
        if self.mode == "overlap":
            for j in range(min(self.nbuf, self.m.n_layers)):
                self.issue(j)

    def acquire(self, i):
        if self.mode == "overlap":
            torch.cuda.current_stream().wait_event(self.ev_copied[i])

    def release(self, i):
        if self.mode == "overlap":
            self.ev_free[i % self.nbuf].record(torch.cuda.current_stream())
            if i + self.nbuf < self.m.n_layers:
                self.issue(i + self.nbuf)

    def codes(self, i, n):
        N, K, o, nb, _ = self.meta[n]
        return self.slots[i % self.nbuf][o:o + nb].view(2, N * K // 8)

    def copy_only(self):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for j in range(self.m.n_layers):
            s = j % self.nbuf
            with torch.cuda.stream(self.cs):
                self.slots[s].copy_(self.host[j * self.LB:(j + 1) * self.LB], non_blocking=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t


def rope(q, k, cos, sin):
    def rh(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), -1)
    return q * cos + rh(q) * sin, k * cos + rh(k) * sin


class BatchRunner:
    """Batched forward over B sequences x M tokens (rows ordered (b, m)), equal context for all sequences."""

    def __init__(self, m, st, B, T):
        self.m, self.st, self.B, self.T = m, st, B, T
        shp = (m.n_layers, B, m.Hkv, T, m.D)
        self.kc = torch.zeros(shp, device="cuda", dtype=m.dtype)
        self.vc = torch.zeros(shp, device="cuda", dtype=m.dtype)
        if not hasattr(m, "fused_ok"):
            m._setup_fused()
        assert m.fused_ok

    def lin(self, i, n, x, k):
        y = self.m.layers[i][n](x, 2 if k >= 2 else k, rotated=True)
        if k == 4:
            N, K, o, nb, cb = self.st.meta[n]
            y = y + mma_rows(x, self.st.codes(i, n), self.st.cbs[i][n], self.m.layers[i][n].sc, 2)
        return y

    def forward(self, h, pos_t, M, k):
        """h [B*M, H] bf16 (embedded rows), pos_t 0-d long device tensor (position of the first token of every seq)."""
        m, B, T = self.m, self.B, self.T
        H, Hkv, D = m.H, m.Hkv, m.D
        G = H // Hkv
        R = B * M
        positions = pos_t + torch.arange(M, device="cuda")
        ct, stt = m._rope_tables(T)
        cos = ct[positions][None, :, None, :]  # [1, M, 1, D]
        sin = stt[positions][None, :, None, :]
        allowed = torch.arange(T, device="cuda")[None, :] <= positions[:, None]  # [M, T]
        mask = allowed.repeat(G, 1)[None, None]  # [1, 1, G*M, T]
        if k == 4:
            self.st.begin()
        l0 = m.layers[0]
        C, sg = l0["g_qkv"]
        _, xs = add_rms_rot(h, None, l0["in_norm"], sg, C, m.eps)
        for i, lay in enumerate(m.layers):
            if k == 4:
                self.st.acquire(i)
            q = self.lin(i, "q_proj", xs[0], k).view(B, M, H, D)
            kk = self.lin(i, "k_proj", xs[1], k).view(B, M, Hkv, D)
            v = self.lin(i, "v_proj", xs[2], k).view(B, M, Hkv, D)
            q, kk = rope(q, kk, cos, sin)
            self.kc[i].index_copy_(2, positions, kk.permute(0, 2, 1, 3))
            self.vc[i].index_copy_(2, positions, v.permute(0, 2, 1, 3))
            qh = q.view(B, M, Hkv, G, D).permute(0, 2, 3, 1, 4).reshape(B, Hkv, G * M, D)
            o = F.scaled_dot_product_attention(qh, self.kc[i], self.vc[i], attn_mask=mask, scale=D ** -0.5)
            o = o.view(B, Hkv, G, M, D).permute(0, 3, 1, 2, 4).reshape(R, H * D)
            C, sg = lay["g_gu"]
            h, xs = add_rms_rot(h, self.lin(i, "o_proj", o.contiguous(), k), lay["post_norm"], sg, C, m.eps)
            Cd, sd = lay["g_down"]
            a = act_rot(self.lin(i, "gate_proj", xs[0], k), self.lin(i, "up_proj", xs[1], k), sd, Cd)
            dn = self.lin(i, "down_proj", a, k)
            if k == 4:
                self.st.release(i)
            if i + 1 < m.n_layers:
                C, sg = m.layers[i + 1]["g_qkv"]
                h, xs = add_rms_rot(h, dn, m.layers[i + 1]["in_norm"], sg, C, m.eps)
            else:
                h, x = add_rms_rot(h, dn, m.norm, None, 128, m.eps)
        return F.linear(x, m.lm_head).argmax(-1)


def embed_rows(m, tok_dev, hbuf_host, hdev):
    """host embedding: D2H token ids, CPU gather (bf16), H2D of the rows into hdev."""
    ids = tok_dev.to("cpu")
    torch.index_select(m.embed, 0, ids, out=hbuf_host[: ids.numel()])
    hdev[: ids.numel()].copy_(hbuf_host[: ids.numel()], non_blocking=True)
    return hdev[: ids.numel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Bs", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 5, 9])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--nbuf", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reps2", type=int, default=20)
    ap.add_argument("--free-gib", type=float, default=0.8, help="headroom required to try a B (KV + activations)")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    e0 = {"co_tenants": cotenants() or "none", "gpu_util": gpu_util(), "loadavg": os.getloadavg()}
    print("[stream] start", e0, flush=True)
    gen = torch.Generator(device="cuda").manual_seed(1)
    m = synth_model("c1", MODEL, embed_dev="cpu", verbose=False)
    st = Streamer(m, a.nbuf, gen)
    torch.cuda.synchronize()
    tot = torch.cuda.get_device_properties(0).total_memory
    load = {"gpu_alloc_gib_after_load": torch.cuda.memory_allocated() / GiB, "layer_chunk_bytes": st.LB,
            "host_pinned_bytes": st.host.numel(), "slot_bytes_total": st.LB * a.nbuf, "fill_s": st.fill_s,
            "rss_gib": int(open('/proc/self/statm').read().split()[1]) * 4096 / GiB, "gpu_total_gib": tot / GiB}
    print("[load]", load, flush=True)
    res = {"env_start": e0, "load": load, "copy_only": None, "rows": [], "t2": [], "skipped": []}
    cps = [st.copy_only() for _ in range(a.reps + 1)][1:]
    res["copy_only"] = {"s": med(cps), "GBps": st.host.numel() / statistics.median(cps) / 1e9}
    print("[copy only]", res["copy_only"], flush=True)
    Mmax = max(a.Ms)
    for B in a.Bs:
        T = (a.ctx + Mmax + 16 + 15) // 16 * 16
        kv_bytes = 2 * m.n_layers * B * m.Hkv * T * m.D * 2
        free = (tot - torch.cuda.memory_reserved()) / GiB
        need = kv_bytes / GiB + a.free_gib
        if need > free:
            res["skipped"].append({"B": B, "kv_gib": kv_bytes / GiB, "free_gib": free, "need_gib": need})
            print(f"[skip] B={B}: KV {kv_bytes / GiB:.2f} GiB + headroom > free {free:.2f} GiB", flush=True)
            continue
        try:
            br = BatchRunner(m, st, B, T)
            hb_host = torch.empty(B * Mmax, m.cfg.hidden_size, dtype=torch.bfloat16, pin_memory=True)
            hdev = torch.empty(B * Mmax, m.cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
            pos_t = torch.tensor(a.ctx, device="cuda")
            for M in a.Ms:
                tok = torch.randint(0, m.cfg.vocab_size, (B * M,), device="cuda", generator=gen)

                def f4(mode):
                    st.mode = mode
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    h = embed_rows(m, tok, hb_host, hdev)
                    out = br.forward(h, pos_t, M, 4)
                    torch.cuda.synchronize()
                    return time.perf_counter() - t
                f4("overlap"), f4("compute")  # warm-up
                ov = [f4("overlap") for _ in range(a.reps)]
                co = [f4("compute") for _ in range(a.reps)]
                r = {"B": B, "M": M, "rows": B * M, "T4_ms": med([1e3 * x for x in ov]),
                     "T4_compute_ms": med([1e3 * x for x in co]),
                     "T_copy_ms": 1e3 * res["copy_only"]["s"]["median"]}
                tc, tcopy, t4 = r["T4_compute_ms"]["median"], r["T_copy_ms"], r["T4_ms"]["median"]
                r["bound"] = "streaming" if tcopy > tc else "compute"
                r["overlap_eff"] = max(tc, tcopy) / t4
                res["rows"].append(r)
                print(f"[T4] B={B} M={M} rows={B * M}: T4 {t4:.1f} ms (compute-only {tc:.1f}, copy-only {tcopy:.1f}) "
                      f"-> {r['bound']}-bound; peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB", flush=True)
            # k2 draft / 2-bit AR step (M=1), CUDA graph over the GPU part + host embedding round trip
            st.mode = "compute"
            tok1 = torch.randint(0, m.cfg.vocab_size, (B,), device="cuda", generator=gen)
            hstat = torch.empty(B, m.cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
            outb = torch.zeros(B, dtype=torch.long, device="cuda")
            for k in (2, 1):
                def step():
                    outb.copy_(br.forward(hstat, pos_t, 1, k))
                gph = _Graph(step)

                def f2():
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    h = embed_rows(m, tok1 if outb is None else outb, hb_host, hdev)
                    hstat.copy_(h)
                    gph.replay()
                    torch.cuda.synchronize()
                    return time.perf_counter() - t
                for _ in range(3):
                    f2()
                ts = [f2() for _ in range(a.reps2)]
                r2 = {"B": B, "k": k, "step_ms": med([1e3 * x for x in ts])}
                res["t2"].append(r2)
                print(f"[T{k}] B={B}: {r2['step_ms']['median']:.2f} ms/step ({B / r2['step_ms']['median'] * 1e3:.1f} tok/s)",
                      flush=True)
                del gph
            del br, hb_host, hdev
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as e:
            res["skipped"].append({"B": B, "oom": str(e)[:200]})
            print(f"[skip] B={B}: OOM {str(e)[:120]}", flush=True)
            torch.cuda.empty_cache()
    # spec round / throughput from measured parts (forced acceptance tau as a parameter; no tau measured here)
    summ = []
    t2 = {(r["B"], r["k"]): r["step_ms"]["median"] for r in res["t2"]}
    t4 = {(r["B"], r["M"]): r["T4_ms"]["median"] for r in res["rows"]}
    for (B, M), T4 in t4.items():
        if M == 1 or (B, 2) not in t2:
            continue
        g = M - 1
        R = g * t2[(B, 2)] + T4
        row = {"B": B, "gamma": g, "round_ms": R, "T_draft_k2_ms": t2[(B, 2)], "T4_verify_ms": T4,
               "AR4_stream_tok_s": B / t4[(B, 1)] * 1e3 if (B, 1) in t4 else None,
               "AR2_gpu_tok_s": B / t2[(B, 2)] * 1e3,
               "tau_breakeven_vs_AR4_stream": R / t4[(B, 1)] if (B, 1) in t4 else None,
               "tau_breakeven_vs_AR2_gpu": R / t2[(B, 2)],
               "spec_tok_s_at_tau": {f"{tau}": B * tau / R * 1e3 for tau in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0) if tau <= g + 1}}
        summ.append(row)
        print("[spec]", row, flush=True)
    res["summary"] = summ
    res["peak_gib"] = torch.cuda.max_memory_allocated() / GiB
    res["env_end"] = {"co_tenants": cotenants() or "none", "gpu_util": gpu_util(), "loadavg": os.getloadavg()}
    print("[stream] end", res["env_end"], flush=True)
    if a.run:
        from tools.results import save_result
        print(save_result("E4", a.run, config={
            "setting": "SYNTHETIC Qwen2.5-72B (all weights random), nested d8 K256 C1 kernels; GPU-resident stages 1-2 "
                       "+ lm_head bf16 + KV bf16, embedding on CPU (host gather + H2D); stages 3-4 (2 planes) in pinned "
                       "host RAM streamed per layer into NBUF device slots on a copy stream with event ordering",
            "linear_4bit": "GPU k2 part + nest_mma over streamed planes 3-4 (same scale), rows > 16 in chunks of 16",
            "ctx": a.ctx, "nbuf": a.nbuf, "Bs": a.Bs, "Ms": a.Ms, "reps": a.reps, "reps2": a.reps2,
            "timing": "eager 4-bit forward with device sync around it (cold start: first NBUF layer chunks are fetched "
                      "inside the timed forward); draft/AR2 = CUDA-graph step + host embedding round trip",
            "spec_model": "round = gamma * T_draft_k2(B) + T4(B, gamma+1); throughput = B * tau / round (tau assumed)",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "co_tenant_start": e0["co_tenants"],
            "co_tenant_end": res["env_end"]["co_tenants"], "host_loadavg_start": e0["loadavg"],
            "host_loadavg_end": res["env_end"]["loadavg"], "script": "experiments/22_scale72/stream72.py"},
            metrics=res, model_dir=MODEL))


if __name__ == "__main__":
    main()
