"""E0 baseline harness: time shape x M x kernel for weight-only quantized GEMV/GEMM (Notion 03 E0).

Modes (--mode, stored per row as "mode"):
  flush       : (default, the E0 v1 protocol) per-call CUDA events, L2 flushed before every call
  warm        : per-call CUDA events, no flush (L2 warm when the weight fits in the 72 MB L2)
  graph       : `graph_iters` back-to-back calls captured in one CUDA graph, time = replay / graph_iters
                (removes host launch gaps and the ~1 us event quantization; L2 warm across calls)
  graph_flush : graph of graph_iters x (flush + call) minus graph of graph_iters x flush (cold L2, no launch gaps;
                noisier because it is a difference)

flush-mode protocol: warm-up calls excluded (includes Triton autotune / JIT build); before every timed call the L2 is flushed
by writing a 256 MB buffer; each call is bracketed by CUDA events; `iters` calls per round, `rounds` rounds;
report the median of round medians and the [min, max] of round medians. Times include host launch gaps when the
path is host-bound (AQLM CUDA/Triton loop over the M rows on the host).

Usage:
  python -m specrq.e0.bench --run <name> [--models qwen llama] [--ms 1 2 3 4 5 8 16] [--kernels ...] [--iters 100]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
import traceback

import torch

from specrq.e0 import kernels as K

SHAPES = {  # (N_out, K_in)
    "qwen2.5-0.5b": {"q_o": (896, 896), "k_v": (128, 896), "gate_up": (4864, 896), "down": (896, 4864),
                     "qkv": (1152, 896), "gate_up_fused": (9728, 896)},
    "llama3.1-8b": {"q_o": (4096, 4096), "k_v": (1024, 4096), "gate_up": (14336, 4096), "down": (4096, 14336),
                    "qkv": (6144, 4096), "gate_up_fused": (28672, 4096)},
}
MODEL_ALIASES = {"qwen": "qwen2.5-0.5b", "llama": "llama3.1-8b"}
FLUSH_BYTES = 256 << 20


class Flusher:
    """L2 eviction: 'write' zero-fills a 256 MB buffer (E0 v1; leaves ~72 MB of dirty lines whose write-back overlaps
    the next kernel); 'read' sums the buffer (evicts L2 with clean lines, closer to decoding where L2 holds the
    previous layer's clean weights)."""

    def __init__(self, buf, kind):
        self.buf, self.kind = buf, kind
        self.out = torch.empty((), device=buf.device, dtype=torch.int64)

    def zero_(self):
        if self.kind == "write":
            self.buf.zero_()
        else:
            self.out = self.buf.sum(dtype=torch.int64)


def time_fn(fn, x, flush, iters, rounds, warmup=10):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in ev:
            if flush is not None:
                flush.zero_()
            s.record()
            fn(x)
            e.record()
        torch.cuda.synchronize()
        meds.append(statistics.median(s.elapsed_time(e) for s, e in ev) * 1e3)  # us
    return statistics.median(meds), min(meds), max(meds)


def _graph_time(body, reps, rounds):
    """Capture body() once into a CUDA graph; return per-round median replay time in us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    g.replay()
    torch.cuda.synchronize()
    meds = []
    for _ in range(rounds):
        ts = []
        for _ in range(reps):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            g.replay()
            en.record()
            torch.cuda.synchronize()
            ts.append(st.elapsed_time(en) * 1e3)
        meds.append(statistics.median(ts))
    del g
    return meds


def time_graph(fn, x, flush, graph_iters, rounds, with_flush, reps=10, warmup=10):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    def body():
        for _ in range(graph_iters):
            if with_flush:
                flush.zero_()
            fn(x)
    meds = _graph_time(body, reps, rounds)
    if with_flush:
        def fbody():
            for _ in range(graph_iters):
                flush.zero_()
        fmeds = _graph_time(fbody, reps, rounds)
        meds = [m - f for m, f in zip(meds, fmeds)]
    meds = [m / graph_iters for m in meds]
    return statistics.median(meds), min(meds), max(meds)


def time_mode(mode, fn, x, flush, a):
    if mode == "flush":
        return time_fn(fn, x, flush, a.iters, a.rounds)
    if mode == "warm":
        return time_fn(fn, x, None, a.iters, a.rounds)
    if mode in ("graph", "graph_flush"):
        return time_graph(fn, x, flush, a.graph_iters, a.rounds, mode == "graph_flush")
    raise ValueError(mode)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", nargs="+", default=["qwen", "llama"])
    ap.add_argument("--layers", nargs="+", default=None, help="subset of layer names")
    ap.add_argument("--ms", nargs="+", type=int, default=[1, 2, 3, 4, 5, 8, 16])
    ap.add_argument("--kernels", nargs="+", default=[k for k in K.KERNELS if k != "nest_gemv"])
    ap.add_argument("--cfgs", nargs="+", default=None, help="restrict configs (e.g. A1x8 A2x8 C1k1)")
    ap.add_argument("--mode", default="flush", choices=["flush", "warm", "graph", "graph_flush"])
    ap.add_argument("--graph-iters", type=int, default=20)
    ap.add_argument("--flush-kind", default="write", choices=["write", "read"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    dev = "cuda"
    flush = Flusher(torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device=dev), a.flush_kind)
    rows, unsupported, errors = [], [], []
    t0 = time.time()
    for mkey in a.models:
        model = MODEL_ALIASES.get(mkey, mkey)
        for lname, (N, Kin) in SHAPES[model].items():
            if a.layers and lname not in a.layers:
                continue
            for kern in a.kernels:
                for cfg in K.KERNELS[kern][1]:
                    if a.cfgs and cfg is not None and cfg not in a.cfgs:
                        continue
                    tag = f"{model}/{lname} {N}x{Kin} {kern}:{cfg}"
                    try:
                        case = K.build(kern, N, Kin, cfg)
                    except K.Unsupported as e:
                        unsupported.append({"model": model, "layer": lname, "kernel": kern, "cfg": cfg, "reason": str(e)})
                        continue
                    except Exception as e:
                        errors.append({"model": model, "layer": lname, "kernel": kern, "cfg": cfg, "stage": "build",
                                       "error": f"{type(e).__name__}: {e}"})
                        print("[build error]", tag, traceback.format_exc(limit=3), file=sys.stderr)
                        continue
                    for M in a.ms:
                        if case.max_m is not None and M > case.max_m:
                            unsupported.append({"model": model, "layer": lname, "kernel": kern, "cfg": cfg, "M": M,
                                                "reason": f"kernel supports M<={case.max_m}"})
                            continue
                        x = torch.randn(M, Kin, device=dev).half()
                        try:
                            y = case.fn(x)
                            yr = case.ref(x)
                            rel = ((y.float() - yr).norm() / yr.norm()).item()
                            med, lo, hi = time_mode(a.mode, case.fn, x, flush, a)
                        except Exception as e:
                            errors.append({"model": model, "layer": lname, "kernel": kern, "cfg": cfg, "M": M,
                                           "stage": "run", "error": f"{type(e).__name__}: {str(e)[:400]}"})
                            print("[run error]", tag, M, traceback.format_exc(limit=3), file=sys.stderr)
                            break
                        act = 2 * M * Kin + 2 * M * N
                        rows.append({"model": model, "layer": lname, "N": N, "K": Kin, "M": M, "kernel": kern,
                                     "cfg": cfg, "mode": a.mode, "flush_kind": a.flush_kind, "t_us": med, "t_us_min": lo, "t_us_max": hi, "rel_err": rel,
                                     "bytes_w": case.bytes_w, "gbps": (case.bytes_w + act) / (med * 1e-6) / 1e9,
                                     **{k: v for k, v in case.info.items() if k in ("bpw", "code_dtype", "path", "k")}})
                        print(f"[{a.mode}/{a.flush_kind}] {tag} M={M:2d} t={med:9.2f}us [{lo:.2f},{hi:.2f}] "
                              f"{rows[-1]['gbps']:7.1f}GB/s rel={rel:.2e}", flush=True)
                    del case
                    torch.cuda.empty_cache()
    # T(M)/T(1) and speed vs fp16 at same (model, layer, M)
    t1 = {(r["model"], r["layer"], r["kernel"], r["cfg"]): r["t_us"] for r in rows if r["M"] == 1}
    fp = {(r["model"], r["layer"], r["M"]): r["t_us"] for r in rows if r["kernel"] == "fp16"}
    for r in rows:
        b = t1.get((r["model"], r["layer"], r["kernel"], r["cfg"]))
        r["t_over_t1"] = r["t_us"] / b if b else None
        f = fp.get((r["model"], r["layer"], r["M"]))
        r["speedup_vs_fp16"] = f / r["t_us"] if f else None
    wall = time.time() - t0
    print(f"done: {len(rows)} rows, {len(unsupported)} unsupported, {len(errors)} errors, {wall:.0f}s")
    for u in {(u["kernel"], u["cfg"], u["reason"]) for u in unsupported}:
        print("[unsupported]", *u)
    if not a.no_save:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
        from tools.results import save_result
        p = save_result("E0", a.run,
                        config={"setting": "weight-only GEMV/GEMM microbench, random weights", "models": a.models,
                                "ms": a.ms, "kernels": a.kernels, "cfgs": a.cfgs, "mode": a.mode, "flush_kind": a.flush_kind,
                                "iters": a.iters, "rounds": a.rounds, "graph_iters": a.graph_iters,
                                "flush_bytes": FLUSH_BYTES, "dtype": "fp16", "vq_configs": K.VQ_CONFIGS,
                                "aqlm_src": "/data/lsh/src/AQLM@e79a896", "marlin_src": "/data/lsh/src/marlin@1f25790"},
                        metrics={"rows": rows, "unsupported": unsupported, "errors": errors, "wall_s": wall})
        print("saved", p)


if __name__ == "__main__":
    main()
