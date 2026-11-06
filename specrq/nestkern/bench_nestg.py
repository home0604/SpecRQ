"""nestg (general) vs specialized C1 kernels on Llama-3.1-8B shapes (d=8, b=8, q=bf16), and nestg timing for the
sweep configurations: M=1 GEMV and M in {2, 5, 8} MMA, graph_flush/read. Records the launch plan (smem bytes,
smem/global fallback, blocks/SM) for each configuration and prefix k.
Usage: python -m specrq.nestkern.bench_nestg [--mode c1|configs] [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.e0.bench import FLUSH_BYTES, Flusher, time_graph
from specrq.nestkern.ops import GenVQ, nest_gemv, nest_mma
from specrq.nestkern.pack import kernel_codes, mma_codes

REPO = Path(__file__).resolve().parents[2]
LLAMA = {"q": (4096, 4096), "k": (1024, 4096), "o": (4096, 4096), "gate": (14336, 4096), "down": (4096, 14336)}
LAYER_COUNT = {"q": 1, "k": 2, "o": 1, "gate": 2, "down": 1}  # k_proj and v_proj, gate and up share shapes
CONFIGS = [(8, 8, 5), (16, 8, 10), (8, 4, 10), (8, 12, 5), (8, 12, 3), (16, 12, 6), (16, 12, 7)]  # (d, b, L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="c1", choices=["c1", "configs"])
    ap.add_argument("--qs", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    from specrq.nestkern.gpucheck import cotenants, require_alone
    cotenant_start = require_alone()
    fl = Flusher(torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda"), "read")
    t = lambda fn, x: time_graph(fn, x, fl, 20, 3, True, reps=5)[0]
    rows = []
    cfgs = [(8, 8, 5)] if a.mode == "c1" else CONFIGS
    qs = [16] if a.mode == "c1" else a.qs
    for d, b, L in cfgs:
        for q in qs:
            for name, (N, K) in LLAMA.items():
                codes = torch.randint(0, 2 ** b, (L, N, K // d), device="cuda")
                cbf = torch.randn(L, 2 ** b, d, device="cuda") * (3.0 / (K * L) ** 0.5)
                g = GenVQ(codes, cbf, torch.rand(N, device="cuda") + 0.5, b, d, q)
                spec = None
                if a.mode == "c1":
                    cb = torch.stack(g.cb_deq).to(torch.bfloat16).contiguous()
                    spec = (kernel_codes(codes.to(torch.uint8)) if K <= 4096 else None, mma_codes(codes.to(torch.uint8)), cb)
                for k in range(1, L + 1):
                    for M in (1, 2, 5, 8):
                        x = torch.randn(M, K, device="cuda").bfloat16()
                        r = {"d": d, "b": b, "L": L, "q": q, "layer": name, "N": N, "K": K, "k": k, "M": M}
                        if M == 1:
                            r["g_gemv_us"] = t(lambda x: g.gemv(x, k)[0], x)
                            r["plan_gemv"] = g.gemv(x, k, dry_run=True)[1]
                        r["g_mma_us"] = t(lambda x: g.mma(x, k)[0], x)
                        r["plan_mma"] = g.mma(x, k, dry_run=True)[1]
                        if spec is not None:
                            if M == 1 and spec[0] is not None:
                                r["c1_gemv_us"] = t(lambda x: nest_gemv(x, spec[0], spec[2], g.scales, k), x)
                            r["c1_mma_us"] = t(lambda x: nest_mma(x, spec[1], spec[2], g.scales, k), x)
                        rows.append(r)
                        print({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in r.items()
                               if not kk.startswith("plan")}, r.get("plan_mma", {}).get("use_smem"), flush=True)
                del g, codes, cbf, spec
                torch.cuda.empty_cache()
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"co_tenant_start": cotenant_start, "co_tenant_end": cotenants() or "none","setting": f"nestg bench mode={a.mode}, Llama-3.1-8B shapes, bf16",
                                                          "mode": "graph_flush", "flush_kind": "read",
                                                          "configs": cfgs, "qs": qs, "layer_count": LAYER_COUNT,
                                                          "smem_max": 96 * 1024},
                                   metrics={"rows": rows}))


if __name__ == "__main__":
    main()
