"""GPU side of the offload micro-benchmark (run on a GPU attached to the same NUMA node as the CPU worker).

Parts
  pcie : activation round trips between GPU and pinned host memory, per layer stage of a split linear
         (D2H of the stage input, host waits on an event, H2D of the residual output, GPU consumes it).
         Sizes per token row (bf16): qkv (5120 -> 7168), o (5120 -> 5120), gate+up (5120 -> 55296),
         down (27648 -> 5120). Also plain one-way copies.
  fwd  : synthetic Qwen2.5-32B d8K256L2 (same NestModel.synthetic as experiments/14_q32b_upper) forward graph
         for k=1 (prefix, = the GPU part of a split forward) and k=2 at M=1..5, full-vocab lm_head; k=1 M=1 with V_d.
Usage: CUDA_VISIBLE_DEVICES=4 taskset -c 16-31 python experiments/18_offload/bench_gpu.py --part pcie fwd --out X.json
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


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


STAGES = {"qkv": (5120, 7168), "o": (5120, 5120), "gateup": (5120, 55296), "down": (27648, 5120)}


def part_pcie(Ms, iters=200, reps=5):
    dev = torch.device("cuda")
    out = {"stages": {}, "oneway": {}}
    # warm-up: ~2 s of copies + kernels so the GPU / PCIe link leave their idle clock states
    wx = torch.randn(1 << 20, device=dev)
    wh = torch.empty(1 << 20, pin_memory=True)
    t_end = time.time() + 2.0
    while time.time() < t_end:
        wx.add_(1)
        wh.copy_(wx, non_blocking=True)
        torch.cuda.synchronize()
    for M in Ms:
        for st, (din, dout) in STAGES.items():
            gx = torch.randn(M, din, device=dev, dtype=torch.bfloat16)
            gy = torch.empty(M, dout, device=dev, dtype=torch.bfloat16)
            hx = torch.empty(M, din, dtype=torch.bfloat16, pin_memory=True)
            hy = torch.randn(M, dout, dtype=torch.bfloat16).pin_memory()
            ev = torch.cuda.Event()
            s = torch.cuda.current_stream()

            def once():
                gx.add_(1)  # a GPU producer kernel before the copy, like the real stage
                hx.copy_(gx, non_blocking=True)
                ev.record(s)
                ev.synchronize()  # host sees the activation
                gy.copy_(hy, non_blocking=True)
                gy.add_(1)  # GPU consumer
            for _ in range(20):
                once()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                for _ in range(iters):
                    once()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) / iters * 1e6)
            # GPU-only reference (same two kernels, no copies / host sync)
            g = []
            for _ in range(reps):
                t = time.perf_counter()
                for _ in range(iters):
                    gx.add_(1)
                    gy.add_(1)
                torch.cuda.synchronize()
                g.append((time.perf_counter() - t) / iters * 1e6)
            r = {"roundtrip_us": med(ts), "gpu_kernels_only_us": med(g),
                 "bytes_d2h": M * din * 2, "bytes_h2d": M * dout * 2}
            out["stages"][f"{st}_M{M}"] = r
            print(f"[pcie] {st} M={M}: round trip {r['roundtrip_us']['median']:.1f} us "
                  f"(kernels only {r['gpu_kernels_only_us']['median']:.1f} us)", flush=True)
    for nbytes in (10240, 102400, 1 << 20, 16 << 20):
        n = nbytes // 2
        g = torch.empty(n, device="cuda", dtype=torch.bfloat16)
        h = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        r = {}
        for tag, f in (("h2d", lambda: g.copy_(h, non_blocking=True)), ("d2h", lambda: h.copy_(g, non_blocking=True))):
            for _ in range(10):
                f()
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                for _ in range(50):
                    f()
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) / 50 * 1e6)
            r[tag + "_us"] = med(ts)
            r[tag + "_GBps"] = nbytes / (r[tag + "_us"]["median"] * 1e-6) / 1e9
        out["oneway"][str(nbytes)] = r
        print(f"[pcie] oneway {nbytes} B: {r}", flush=True)
    return out


def part_fwd(model, vd):
    sys.path.insert(0, str(REPO / "experiments/14_q32b_upper"))
    from q32b_upper import graph_us  # noqa: E402
    from specrq.infer.model import NestModel  # noqa: E402
    from specrq.infer.frspec import default_freq_dir, load_subset  # noqa: E402
    t0 = time.time()
    m = NestModel.synthetic(model, L=2, hadamard=True, verbose=False)
    m.set_draft_vocab(load_subset(default_freq_dir(m.cfg), vd, None))
    torch.cuda.synchronize()
    res = {"load_s": time.time() - t0, "linear_bytes": {f"k{k}": m.linear_bytes(k) for k in (1, 2)}, "rows": []}
    gen = torch.Generator(device="cuda").manual_seed(0)
    P = 128
    prompt = torch.randint(0, m.cfg.vocab_size, (P,), device="cuda", generator=gen)
    cache = m.new_cache(P + 64)
    m.prefill(prompt, cache, 2)
    pos = torch.tensor(P, device="cuda")
    toks = torch.randint(0, m.cfg.vocab_size, (8,), device="cuda", generator=gen)
    for k in (1, 2):
        for M in (1, 2, 3, 4, 5):
            r = {"k": k, "vd": None, "M": M,
                 "us": graph_us(lambda: m.forward(toks[:M], pos, cache, k, last_only=(M == 1)).argmax(-1))}
            res["rows"].append(r)
            print("[fwd]", r, flush=True)
    r = {"k": 1, "vd": vd, "M": 1,
         "us": graph_us(lambda: m.to_token_ids(m.forward(toks[:1], pos, cache, 1, last_only=True, vd=vd).argmax(-1), vd))}
    res["rows"].append(r)
    print("[fwd]", r, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["pcie", "fwd"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16])
    ap.add_argument("--model", default="/data/models/qwen2.5_32b_instruct")
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "affinity": sorted(os.sched_getaffinity(0))}
    if "pcie" in a.part:
        res["pcie"] = part_pcie(a.Ms)
    if "fwd" in a.part:
        res["fwd"] = part_fwd(a.model, a.vd)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
