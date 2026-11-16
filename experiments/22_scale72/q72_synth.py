"""E4 scale72: SYNTHETIC Qwen2.5-72B-Instruct speed (timing only, no quality). Shapes from config.json of
/data/models/qwen2.5_72b_instruct; ALL tensors random (embed / norms / lm_head too: the safetensors are not needed).
Nested d8 K256 structure: stages 1-2 (2 bit) on the GPU, stages 3-4 (2 more bit, 4-bit target) on the CPU.

Modes
  kern  : per-linear kernel choice for the 72B shapes: C1 (NestLinear, nest_mma; no GEMV path since K > 4096) vs
          nestg (GenLinear d8 bits [8,8] q16, M=1 path auto gemv/mma), k = 1, 2, M = 1..5, CUDA graph, 1 layer.
  gpu   : full 80-layer GPU-resident model with stages 1-2 (L=2 planes) on the chosen kernel:
          AR tok/s k2 and k1 (full vocab, and k1 with FR-Spec V_d), forward graph T(M) for k1, k2, M=1..5.
  split : same GPU model + CPU residual planes (2 per linear, 16.35 GiB of codes, first-touched on this NUMA node)
          -> REAL split forward for k=4 (eager; per linear: GPU k2 part on device, D2H of the rotated input, CPU LUT
          kernels for planes 3 and 4 (perm_i8_tile, int8 LUT), H2D add). Measures offloaded 4-bit AR tok/s,
          eager split forward T(M) M=1..5, k2 draft-step graph, and spec rounds (gamma k2 draft steps in a CUDA graph
          + one eager k4 verify of gamma+1 rows, forced acceptance -> round time only).
Usage: python experiments/22_scale72/q72_synth.py --mode kern|gpu|split [--impl c1|nestg] --run NAME
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "14_q32b_upper"))
sys.path.insert(0, str(REPO / "experiments" / "18_offload"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from q32b_upper import ar_rows, graph_us, gpu_util, med  # noqa: E402
from specrq.infer.decode import _Graph  # noqa: E402
from specrq.infer.model import GenLinear, NestLinear, NestModel, _rope_inv_freq  # noqa: E402
from specrq.nestkern.gpucheck import cotenants  # noqa: E402
from specrq.nestkern.ops import block_fwht_signed  # noqa: E402

MODEL = "/data/models/qwen2.5_72b_instruct"


def shapes_of(cfg):
    H, Hkv, hid, inter = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size, cfg.intermediate_size
    D = hid // H
    return {"q_proj": (H * D, hid), "k_proj": (Hkv * D, hid), "v_proj": (Hkv * D, hid), "o_proj": (hid, H * D),
            "gate_proj": (inter, hid), "up_proj": (inter, hid), "down_proj": (hid, inter)}


def make_linear(impl, N, K, bias, gen, dtype=torch.bfloat16):
    if impl == "c1":
        return NestLinear.synthetic(N, K, "cuda", dtype, 2, 256, bias=bias, gen=gen)
    return GenLinear.synthetic(N, K, torch.device("cuda"), dtype, 8, [8, 8], 16, 256, bias=bias, gen=gen)


def synth_model(impl, model_path=MODEL, embed_dev="cuda", seed=0, verbose=True):
    """NestModel with random everything (cf. NestModel.synthetic, which reads embed/norms from the safetensors)."""
    from transformers import AutoConfig
    m = NestModel.__new__(NestModel)
    m.cfg = cfg = AutoConfig.from_pretrained(model_path)
    m.device, m.dtype = torch.device("cuda"), torch.bfloat16
    m.n_layers, m.H, m.Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    m.D = cfg.hidden_size // cfg.num_attention_heads
    m.eps = cfg.rms_norm_eps
    gen = torch.Generator(device="cuda").manual_seed(seed)
    V, hid = cfg.vocab_size, cfg.hidden_size
    m.embed = (torch.randn(V, hid, device="cuda", generator=gen, dtype=torch.bfloat16) * 0.02).to(embed_dev)
    m.lm_head = torch.randn(V, hid, device="cuda", generator=gen, dtype=torch.bfloat16) * 0.02
    m.norm = torch.ones(hid, device="cuda", dtype=torch.bfloat16)
    shp = shapes_of(cfg)
    m.layers = []
    for i in range(m.n_layers):
        d = {"in_norm": torch.ones(hid, device="cuda", dtype=torch.bfloat16),
             "post_norm": torch.ones(hid, device="cuda", dtype=torch.bfloat16)}
        for n, (N, K) in shp.items():
            d[n] = make_linear(impl, N, K, cfg.model_type == "qwen2" and n in ("q_proj", "k_proj", "v_proj"), gen)
        m.layers.append(d)
        if verbose and i % 16 == 0:
            print(f"[q72] synthetic layer {i}/{m.n_layers} alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
    m.L = 2
    m.inv_freq, m.rope_scaling = _rope_inv_freq(cfg, m.device)
    m.use_compile, m.decode_attn, m._rope_len, m.fused_glue = True, "fp32", 0, True
    m.vq = None if impl == "c1" else {"d": 8, "bits": [8, 8], "q": 16}
    m.ckpt_dir = f"synthetic-all-random:{impl} d8 K256 L2(GPU planes) hadamard B=256 seed={seed}"
    m.model_path = model_path
    return m


def env_rec():
    return {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "co_tenants": cotenants() or "none",
            "gpu_util": gpu_util(), "loadavg": os.getloadavg()}


# ------------------------------------------------------------------------------------------------- mode kern
def mode_kern(a):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for n, (N, K) in shapes_of(cfg).items():
        for impl in ("c1", "nestg"):
            lin = make_linear(impl, N, K, False, gen)
            for k in (1, 2):
                for M in (1, 2, 3, 4, 5):
                    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
                    if impl == "nestg" and M == 1:
                        lin(x, k, rotated=True)  # auto-pick gemv/mma outside capture
                    r = {"lin": n, "N": N, "K": K, "impl": impl, "k": k, "M": M,
                         "us": graph_us(lambda: lin(x, k, rotated=True), iters=50, reps=5)}
                    if impl == "nestg" and M == 1:
                        r["m1_path"] = lin.m1.get(k)
                    rows.append(r)
            del lin
            torch.cuda.empty_cache()
        print(f"[kern] {n} " + " ".join(f"{r['impl']}k{r['k']}M{r['M']}={r['us']['median']:.1f}" for r in rows
                                         if r["lin"] == n), flush=True)
    summ = {}
    for impl in ("c1", "nestg"):
        for k in (1, 2):
            for M in (1, 2, 3, 4, 5):
                summ[f"{impl}_k{k}_M{M}_layer_us"] = sum(r["us"]["median"] for r in rows
                                                         if r["impl"] == impl and r["k"] == k and r["M"] == M)
    print("[kern summary per layer, 7 linears]", summ, flush=True)
    return {"rows": rows, "per_layer_sum_us": summ}


# ------------------------------------------------------------------------------------------------- mode gpu
def mode_gpu(a, m):
    res = {"ar": [], "fwd": []}
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(0, m.cfg.vocab_size, (a.P,), device="cuda", generator=gen)
    T = a.P + a.n_new + 24
    for k, vd in (() if a.skip_ar else ((2, None), (1, None), (1, a.vd))):
        res["ar"].append(ar_rows(m, prompt, T, k, vd, a.n_new, a.reps))
        print("[ar]", res["ar"][-1], flush=True)
    cache = m.new_cache(T)
    m.prefill(prompt, cache, 2)
    pos = torch.tensor(a.P, device="cuda")
    toks = torch.randint(0, m.cfg.vocab_size, (8,), device="cuda", generator=gen)
    for k in (1, 2):
        for M in (1, 2, 3, 4, 5):
            r = {"k": k, "vd": None, "M": M,
                 "us": graph_us(lambda: m.forward(toks[:M], pos, cache, k, last_only=(M == 1)).argmax(-1),
                                iters=a.giters)}
            res["fwd"].append(r)
            print("[fwd]", r, flush=True)
    r = {"k": 1, "vd": a.vd, "M": 1,
         "us": graph_us(lambda: m.to_token_ids(m.forward(toks[:1], pos, cache, 1, last_only=True,
                                                         vd=a.vd).argmax(-1), a.vd), iters=a.giters)}
    res["fwd"].append(r)
    print("[fwd]", r, flush=True)
    del cache
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------------------------------------- mode split
class CPUPlanes:
    """Residual planes 3..L of all linears in host memory (first-touched by the CPU kernel threads), int8-LUT kernels
    of experiments/18_offload/cpu_kernels.cpp. Codes are random; each linear gets its own [256, 8] codebook per plane
    and per-row scale (as the C1 format)."""

    def __init__(self, m, n_planes, lib, rng):
        import bench_cpu as B
        self.lib, self.B, self.np = lib, B, n_planes
        self.lin = {}
        self.bytes = 0
        maxK = max(K for _, K in shapes_of(m.cfg).values())
        maxN = max(N for N, _ in shapes_of(m.cfg).values())
        Mmax = 16
        self.lutq = B.aligned(Mmax * (maxK // 8) * 256, np.int8, lib)
        self.ls = B.aligned(Mmax * (maxK // 256) * 4, np.float32, lib)
        self.yp = B.aligned(Mmax * maxN * 4, np.float32, lib)
        self.ysum = B.aligned(Mmax * maxN * 4, np.float32, lib)
        self.xh = torch.empty(Mmax, maxK, dtype=torch.bfloat16, pin_memory=True)
        self.xf = B.aligned(Mmax * maxK * 4, np.float32, lib)
        self.yh = torch.empty(Mmax * maxN, dtype=torch.bfloat16, pin_memory=True)
        for i, lay in enumerate(m.layers):
            for n, (N, K) in shapes_of(m.cfg).items():
                planes = []
                for p in range(n_planes):
                    c = B.aligned(N * K // 8, np.uint8, lib)
                    c[:] = rng.integers(0, 256, c.shape[0], dtype=np.uint8) if i < 1 else self.lin[(0, n)][p][0][:]
                    cb = (rng.standard_normal((256, 8)).astype(np.float32) * 0.02)
                    planes.append((c, np.ascontiguousarray(cb.T), (rng.uniform(0.5, 1.5, N)).astype(np.float32)))
                    self.bytes += c.nbytes
                self.lin[(i, n)] = planes
        self.t_cpu = 0.0
        self.t_xfer = 0.0
        self.calls = 0

    def apply(self, i, n, x_rot):
        """x_rot [M, K] bf16 (device, already Hadamard-rotated) -> sum over CPU planes of x W_p^T, [M, N] bf16 device."""
        B, lib = self.B, self.lib
        M, K = x_rot.shape
        planes = self.lin[(i, n)]
        N = planes[0][2].shape[0]
        t0 = time.perf_counter()
        self.xh[:M, :K].copy_(x_rot)  # D2H (synchronous: waits for the producer kernel)
        xf = self.xf[: M * K].reshape(M, K)
        xf[:] = self.xh[:M, :K].float().numpy()
        t1 = time.perf_counter()
        ys = self.ysum[: M * N].reshape(M, N)
        for p, (c, cbT, sc) in enumerate(planes):
            lib.lut_build_i8(B.ptr(xf), B.ptr(cbT), B.ptr(self.lutq), B.ptr(self.ls), M, K)
            yp = self.yp[: M * N].reshape(M, N) if p else ys
            lib.gemv_perm_i8t(B.ptr(c), B.ptr(self.lutq), B.ptr(self.ls), B.ptr(sc), B.ptr(yp), M, K, N, 0)
            if p:
                ys += yp
        t2 = time.perf_counter()
        yh = self.yh[: M * N].view(M, N)
        yh.copy_(torch.from_numpy(ys))
        out = yh.to("cuda", non_blocking=True)
        t3 = time.perf_counter()
        self.t_cpu += t2 - t1
        self.t_xfer += (t1 - t0) + (t3 - t2)
        self.calls += 1
        return out


class SplitLinear:
    """k <= 2: the GPU linear (graph-capturable). k > 2: GPU stages 1-2 + CPU planes 3..k (eager, host sync)."""

    def __init__(self, gpu_lin, cpu, i, n):
        self.g, self.cpu, self.i, self.n = gpu_lin, cpu, i, n
        self.N, self.K, self.L = gpu_lin.N, gpu_lin.K, 2 + cpu.np
        self.signs, self.hb, self.bias = gpu_lin.signs, gpu_lin.hb, gpu_lin.bias

    def __call__(self, x, k, use_draft=False, rotated=False):
        if k <= 2:
            return self.g(x, k, use_draft, rotated)
        if self.signs is not None and not rotated:
            x = block_fwht_signed(x, self.signs, self.hb)
        y = self.g(x, 2, use_draft, rotated=True)  # async on the GPU while the CPU works below
        return y + self.cpu.apply(self.i, self.n, x)

    def nbytes(self, k):
        return self.g.nbytes(min(k, 2))


def mode_split(a, m):
    import bench_cpu as B
    lib = B.build()
    rng = np.random.default_rng(0)
    t0 = time.time()
    cpu = CPUPlanes(m, 2, lib, rng)
    print(f"[split] CPU planes: {cpu.bytes / 2**30:.2f} GiB codes in {time.time() - t0:.0f}s; "
          f"RSS {int(open('/proc/self/statm').read().split()[1]) * 4096 / 2**30:.2f} GiB", flush=True)
    for i, lay in enumerate(m.layers):
        for n in shapes_of(m.cfg):
            lay[n] = SplitLinear(lay[n], cpu, i, n)
    res = {"cpu_code_bytes": cpu.bytes, "omp_threads": os.environ.get("OMP_NUM_THREADS"), "loadavg_start": os.getloadavg()}
    gen = torch.Generator(device="cuda").manual_seed(0)
    P = a.P
    prompt = torch.randint(0, m.cfg.vocab_size, (P,), device="cuda", generator=gen)
    T = P + max(a.n_off, 64) + 24
    cache = m.new_cache(T)
    m.prefill(prompt, cache, 2)  # prompt KV from the GPU part (timing only)
    pos = torch.tensor(P, device="cuda")
    toks = torch.randint(0, m.cfg.vocab_size, (8,), device="cuda", generator=gen)

    def eager_fwd(M, k):
        torch.cuda.synchronize()
        cpu.t_cpu = cpu.t_xfer = 0.0
        cpu.calls = 0
        t = time.perf_counter()
        y = m.forward(toks[:M], pos, cache, k, last_only=(M == 1)).argmax(-1)
        torch.cuda.synchronize()
        return time.perf_counter() - t, cpu.t_cpu, cpu.t_xfer, cpu.calls

    # warm-up (compile of the glue for each M)
    for M in (1, 2, 3, 4, 5):
        eager_fwd(M, 4)
        eager_fwd(M, 2)
    res["fwd_eager"] = []
    for k in (2, 4):
        for M in (1, 2, 3, 4, 5):
            rs = [eager_fwd(M, k) for _ in range(a.reps)]
            r = {"k": k, "M": M, "wall_ms": med([1e3 * x[0] for x in rs]), "cpu_kernel_ms": med([1e3 * x[1] for x in rs]),
                 "host_xfer_ms": med([1e3 * x[2] for x in rs]), "cpu_calls": rs[0][3]}
            res["fwd_eager"].append(r)
            print("[split fwd eager]", r, flush=True)
    # graph-captured k2 forward for reference (same model object; SplitLinear k<=2 path is capturable)
    res["fwd_graph_k2"] = []
    for M in (1, 2, 3, 4, 5):
        r = {"k": 2, "M": M, "us": graph_us(lambda: m.forward(toks[:M], pos, cache, 2, last_only=(M == 1)).argmax(-1),
                                            iters=a.giters)}
        res["fwd_graph_k2"].append(r)
        print("[split fwd graph k2]", r, flush=True)

    # offloaded 4-bit AR: eager M=1 k=4 steps, argmax fed back on device
    def ar_off(n):
        tok = toks[:1].clone()
        p = pos.clone()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            tok = m.forward(tok, p, cache, 4, last_only=True).argmax(-1)
            p = p + 1
        torch.cuda.synchronize()
        return time.perf_counter() - t
    ar_off(2)
    st = [ar_off(a.n_off) for _ in range(a.reps)]
    res["ar_offloaded_k4"] = {"n_tokens": a.n_off, "step_ms": med([1e3 * s / a.n_off for s in st]),
                              "tokens_per_s": med([a.n_off / s for s in st])}
    print("[ar offloaded k4]", res["ar_offloaded_k4"], flush=True)

    # spec rounds: gamma k2 draft steps (one CUDA graph) + eager k4 verify of gamma+1 rows (forced acceptance)
    res["spec_round"] = []
    for g in a.gammas:
        dtok = torch.zeros(1, dtype=torch.long, device="cuda")
        dpos = torch.zeros((), dtype=torch.long, device="cuda")
        buf = torch.zeros(g + 1, dtype=torch.long, device="cuda")
        idx = torch.zeros((), dtype=torch.long, device="cuda")

        def draft():
            for j in range(g):
                nxt = m.forward(dtok, dpos, cache, 2, last_only=True).argmax(-1)
                buf[j + 1:j + 2].copy_(nxt)
                dtok.copy_(nxt)
                dpos.add_(1)
        gd = _Graph(draft)

        def round_once():
            dtok.copy_(toks[:1])
            dpos.copy_(pos)
            buf[:1].copy_(toks[:1])
            torch.cuda.synchronize()
            t = time.perf_counter()
            gd.replay()
            torch.cuda.synchronize()
            td = time.perf_counter() - t
            v = m.forward(buf, pos, cache, 4).argmax(-1)
            nacc = int(v[0])  # host read (acceptance decision point)
            torch.cuda.synchronize()
            return time.perf_counter() - t, td
        round_once()
        rs = [round_once() for _ in range(a.reps)]
        r = {"gamma": g, "round_ms": med([1e3 * x[0] for x in rs]), "draft_ms": med([1e3 * x[1] for x in rs]),
             "verify_ms": med([1e3 * (x[0] - x[1]) for x in rs])}
        res["spec_round"].append(r)
        print("[spec round]", r, flush=True)
        del gd
    res["loadavg_end"] = os.getloadavg()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kern", "gpu", "split"], required=True)
    ap.add_argument("--impl", choices=["c1", "nestg"], default="c1")
    ap.add_argument("--embed", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--n-new", type=int, default=256)
    ap.add_argument("--n-off", type=int, default=16, help="split mode: offloaded AR tokens per repeat")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--giters", type=int, default=4, help="forward calls per captured graph (graph_us)")
    ap.add_argument("--skip-ar", action="store_true")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    e0 = env_rec()
    print("[q72] start", e0, flush=True)
    torch.cuda.reset_peak_memory_stats()
    res = {"env_start": e0}
    if a.mode == "kern":
        res["kern"] = mode_kern(a)
    else:
        t0 = time.time()
        m = synth_model(a.impl, embed_dev=a.embed)
        if a.mode == "gpu":
            from specrq.infer.frspec import default_freq_dir, load_subset
            m.set_draft_vocab(load_subset(default_freq_dir(m.cfg), a.vd, None))
        torch.cuda.synchronize()
        res["load"] = {"load_s": time.time() - t0, "gpu_mem_alloc_gib": torch.cuda.memory_allocated() / 2**30,
                       "linear_bytes": {f"k{k}": m.linear_bytes(k) for k in (1, 2)},
                       "embed_bytes": m.embed.numel() * 2, "lm_head_bytes": m.lm_head.numel() * 2, "embed_dev": a.embed}
        print("[load]", res["load"], flush=True)
        res[a.mode] = mode_gpu(a, m) if a.mode == "gpu" else mode_split(a, m)
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    res["env_end"] = env_rec()
    print("[q72] end", res["env_end"], "peak", res["peak_mem_gib"], flush=True)
    if a.run:
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": "SYNTHETIC Qwen2.5-72B-Instruct shapes (config.json), ALL weights random (embed/norms/lm_head "
                       "too); nested d8 K256, stages 1-2 on GPU (L=2 planes), Hadamard B=256, qkv bias; timing only",
            "mode": a.mode, "impl": a.impl, "embed": a.embed, "P": a.P, "n_new": a.n_new, "n_off": a.n_off,
            "reps": a.reps, "giters": a.giters, "skip_ar": a.skip_ar, "gammas": a.gammas, "vd": a.vd, "dtype": "bf16", "decode_attn": "fp32 compiled",
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
            "co_tenant_start": e0["co_tenants"], "co_tenant_end": res["env_end"]["co_tenants"],
            "script": "experiments/22_scale72/q72_synth.py"}, metrics=res, model_dir=MODEL))


if __name__ == "__main__":
    main()
