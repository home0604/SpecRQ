"""Online Hadamard input rotation cost: DbD torch rotate_input vs nestkern CUDA block_fwht_signed.

M x K activations (M in {1..16}), block size B in {32, 64, 128, 256}; correctness vs the DbD function.
Timing: graph_flush/read (cold L2, CUDA graph) for `torch_cachedH` (DbD math with H cached on the GPU) and
`cuda_fwht`. The DbD function itself (`torch_dbd`) copies H host->device on every call, which cannot be captured in a
CUDA graph, so it is timed with per-call events + read flush (mode flush/read; includes that copy and launch gaps).
Usage: python -m specrq.nestkern.bench_fwht --run RUN [--dtype bf16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "specrq" / "dbd"))
from src.hadamard import hadamard_matrix, make_signs, rotate_input  # noqa: E402

from specrq.e0.bench import FLUSH_BYTES, Flusher, time_fn, time_graph  # noqa: E402
from specrq.nestkern.ops import block_fwht_signed  # noqa: E402

KS = [896, 4096, 4864, 14336]
MS = [1, 2, 3, 4, 5, 8, 16]
BS = [32, 64, 128, 256]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    fl = Flusher(torch.zeros(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda"), "read")
    rows = []
    for K in KS:
        s = make_signs(K, 0, "cuda")
        for B in BS:
            if K % B:
                continue
            H = hadamard_matrix(B, "cuda", torch.float32)

            def cached(x, s=s, H=H, B=B):
                return ((x.float() * s).reshape(*x.shape[:-1], -1, B) @ H).reshape(x.shape).to(x.dtype)
            for M in MS:
                x = torch.randn(M, K, device="cuda").to(dt)
                ref = rotate_input(x.float(), s, B)
                r = {"K": K, "B": B, "M": M, "dtype": a.dtype}
                for name, fn, graph in (("torch_dbd", lambda x: rotate_input(x, s, B), False),
                                        ("torch_cachedH", cached, True),
                                        ("cuda_fwht", lambda x: block_fwht_signed(x, s, B), True)):
                    y = fn(x)
                    r[f"{name}_rel"] = ((y.float() - ref).norm() / ref.norm()).item()
                    r[f"{name}_us"] = (time_graph(fn, x, fl, 20, a.rounds, True, reps=5) if graph
                                       else time_fn(fn, x, fl, 100, a.rounds))[0]
                rows.append(r)
                print(f"K={K:5d} B={B:3d} M={M:2d} dbd(flush/read) {r['torch_dbd_us']:6.2f}us | cachedH {r['torch_cachedH_us']:6.2f}us"
                      f" | cuda {r['cuda_fwht_us']:6.2f}us (rel {r['cuda_fwht_rel']:.1e})", flush=True)
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.run, config={"setting": "online Hadamard input rotation cost",
                                                         "mode": "graph_flush (torch_cachedH, cuda_fwht); flush (torch_dbd)", "flush_kind": "read",
                                                         "dtype": a.dtype, "ks": KS, "ms": MS, "bs": BS},
                                   metrics={"rows": rows}))


if __name__ == "__main__":
    main()
