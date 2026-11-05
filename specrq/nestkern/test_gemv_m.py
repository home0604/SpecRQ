"""P4: small-M GEMV (nest_gemv_m, CUDA cores, M=1..8) correctness vs fp32 prefix-k reference and timing vs nest_mma
(det, auto cfg) and nest_gemv (M=1) on Qwen2.5-0.5B and Llama-3.1-8B shapes, bf16, graph_flush/read.
Pass rule as nest_mma: rel < 1e-3 or rel < 1.5 * floor(weight sum rounded once + output rounding).
Usage: python -m specrq.nestkern.test_gemv_m [--save-correct RUN] [--save-time RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.e0.bench import FLUSH_BYTES, Flusher, time_graph
from specrq.nestkern.ops import nest_gemv, nest_gemv_m, nest_mma
from specrq.nestkern.pack import kernel_codes, mma_codes, ref_matmul
from specrq.nestkern.test_gemv import rand_planar
from specrq.nestkern.test_mma import floor_ref, relerr

REPO = Path(__file__).resolve().parents[2]
SHAPES = {"qwen/q_o": (896, 896), "qwen/k_v": (128, 896), "qwen/gate_up": (4864, 896), "qwen/down": (896, 4864),
          "llama/q_o": (4096, 4096), "llama/k_v": (1024, 4096), "llama/gate_up": (14336, 4096),
          "llama/down": (4096, 14336)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-correct", default=None)
    ap.add_argument("--save-time", default=None)
    ap.add_argument("--ms", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    corr, tim = [], []
    fl = Flusher(torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda"), "read")
    t = lambda fn, x: time_graph(fn, x, fl, 20, 3, True, reps=5)[0]
    for dt in (torch.bfloat16, torch.float16):
        for name, (N, K) in SHAPES.items():
            p = rand_planar(N, K, dtype=dt)
            kc, mc = kernel_codes(p.codes), mma_codes(p.codes)
            for M in a.ms:
                x = torch.randn(M, K, device="cuda").to(dt)
                for k in (1, 2, 3, 5):
                    yr = ref_matmul(x, p, k)
                    fl_ = relerr(floor_ref(x, p, k, dt), yr)
                    e = relerr(nest_gemv_m(x, kc, p.codebooks, p.scales, k), yr)
                    corr.append({"shape": name, "N": N, "K": K, "M": M, "k": k, "dtype": str(dt), "rel_err": e,
                                 "floor": fl_, "pass": e < 1e-3 or e < 1.5 * fl_})
                    if dt == torch.bfloat16 and k in (1, 3, 5):
                        r = {"shape": name, "N": N, "K": K, "M": M, "k": k,
                             "gemv_m_us": t(lambda x: nest_gemv_m(x, kc, p.codebooks, p.scales, k), x),
                             "mma_det_us": t(lambda x: nest_mma(x, mc, p.codebooks, p.scales, k), x)}
                        if M == 1 and K <= 4096:
                            r["gemv_us"] = t(lambda x: nest_gemv(x, kc, p.codebooks, p.scales, k), x)
                        tim.append(r)
                        print(r, flush=True)
            del p, kc, mc
    bad = [c for c in corr if not c["pass"]]
    print("correctness:", "PASS" if not bad else f"FAIL {len(bad)}", bad[:3], "max rel/floor",
          max(c["rel_err"] / c["floor"] for c in corr))
    sys.path.insert(0, str(REPO))
    from tools.results import save_result
    if a.save_correct:
        print("saved", save_result("E0", a.save_correct, config={"setting": "P4 nest_gemv_m correctness",
                                                                   "shapes": SHAPES, "ms": a.ms},
                                   metrics={"rows": corr, "pass": not bad}))
    if a.save_time:
        print("saved", save_result("E0", a.save_time, config={"setting": "P4 nest_gemv_m vs nest_mma(det, auto cfg) vs "
                                                                "nest_gemv, bf16", "mode": "graph_flush",
                                                                "flush_kind": "read", "shapes": SHAPES, "ms": a.ms},
                                   metrics={"rows": tim}))


if __name__ == "__main__":
    main()
