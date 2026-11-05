"""P3: decompose the verify-linear overhead (Llama-3.1-8B shapes, C1 k=5, bf16) into
  (a) gemv (M=1) vs mma (M>=2) kernel choice, (b) deterministic vs atomic split-K, (c) auto vs best tile config.
Per shape: q/o 4096x4096, k/v 1024x4096, gate/up 14336x4096, down 4096x14336 (7 linears / layer, unfused).
Timing: graph_flush/read (cold L2, CUDA graph, E0 protocol). Layer sum = q+k+v+o+gate+up+down.
Usage: python -m specrq.infer.p3_linear --run NAME [--k 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.e0.bench import FLUSH_BYTES, Flusher, time_graph
from specrq.nestkern.ops import nest_gemv, nest_mma
from specrq.nestkern.pack import kernel_codes, mma_codes
from specrq.nestkern.test_gemv import rand_planar

REPO = Path(__file__).resolve().parents[2]
SHAPES = {"q": (4096, 4096), "k": (1024, 4096), "v": (1024, 4096), "o": (4096, 4096), "gate": (14336, 4096),
          "up": (14336, 4096), "down": (4096, 14336)}
CFGS = list(range(1, 10))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 3])
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    fl = Flusher(torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda"), "read")
    t = lambda fn, x: time_graph(fn, x, fl, 20, 3, True, reps=5)[0]
    rows = []
    for name, (N, K) in SHAPES.items():
        p = rand_planar(N, K, dtype=torch.bfloat16)
        mc = mma_codes(p.codes)
        kc = kernel_codes(p.codes) if K <= 4096 else None
        for k in a.k:
            for M in a.ms:
                x = torch.randn(M, K, device="cuda").bfloat16()
                r = {"layer": name, "N": N, "K": K, "k": k, "M": M}
                if M == 1 and kc is not None:
                    r["gemv_us"] = t(lambda x: nest_gemv(x, kc, p.codebooks, p.scales, k), x)
                for det in (True, False):
                    tag = "det" if det else "atomic"
                    r[f"mma_auto_{tag}_us"] = t(lambda x: nest_mma(x, mc, p.codebooks, p.scales, k, 0, 0, det), x)
                    best = None
                    for cfg in CFGS:
                        us = t(lambda x: nest_mma(x, mc, p.codebooks, p.scales, k, 0, cfg, det), x)
                        if best is None or us < best[0]:
                            best = (us, cfg)
                    r[f"mma_best_{tag}_us"], r[f"mma_best_{tag}_cfg"] = best
                rows.append(r)
                print(r, flush=True)
        del p, mc, kc
        torch.cuda.empty_cache()
    summary = []
    for k in a.k:
        for M in a.ms:
            sel = [r for r in rows if r["k"] == k and r["M"] == M]
            s = {"k": k, "M": M}
            for key in ("mma_auto_det_us", "mma_auto_atomic_us", "mma_best_det_us", "mma_best_atomic_us"):
                s[key] = sum(r[key] for r in sel)
            if M == 1:
                s["dispatch_m1_us"] = sum(r.get("gemv_us", r["mma_auto_det_us"]) for r in sel)
            summary.append(s)
            print("[layer]", s, flush=True)
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.run, config={
            "setting": "P3 verify-linear decomposition, Llama-3.1-8B shapes (7 unfused linears), random C1, bf16",
            "mode": "graph_flush", "flush_kind": "read", "ks": a.k, "ms": a.ms, "cfgs": CFGS},
            metrics={"rows": rows, "layer_sum": summary}))


if __name__ == "__main__":
    main()
