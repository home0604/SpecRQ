"""E4: per-operation breakdown of one batch-1 M=1 decode step (CUDA graph, as ARDecoder) on the synthetic C1 model
(q72_synth.synth_model: all weights random, nested d8 K256 L=2 GPU planes, Hadamard B=256).
Run under nsys (CUDA-graph node tracing); each config is one NVTX range "cfg:<name>" holding --steps graph replays,
each replay followed by a device sync (so each step's kernels are isolated); the un-synced pipelined step time
(ARDecoder style, = the AR measurement) is measured before the profiled replays and printed as WALL lines.
  nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi -t cuda,nvtx -o OUT \
      python experiments/22_scale72/prof_synth.py --model M --configs k1 k1v k2
Configs: k1 / k2 = prefix planes with the full-vocab lm_head; k1v = k1 with FR-Spec V_d (32768).
Analyse with prof_analyze.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from q72_synth import synth_model  # noqa: E402
from specrq.infer.decode import ARDecoder  # noqa: E402
from specrq.nestkern.gpucheck import cotenants  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="+", default=["k1", "k1v", "k2"])
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--pre", type=int, default=128, help="decode steps before profiling (ctx ~ P + pre)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--meta", required=True, help="json written with wall step times and config")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    co0 = cotenants() or "none"
    m = synth_model("c1", model_path=a.model, verbose=False)
    from specrq.infer.frspec import default_freq_dir, load_subset
    m.set_draft_vocab(load_subset(default_freq_dir(m.cfg), a.vd, None))
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(0, m.cfg.vocab_size, (a.P,), device="cuda", generator=gen)
    T = a.P + a.pre + 3 * a.steps + 64
    meta = {"model": a.model, "P": a.P, "pre": a.pre, "steps": a.steps, "vd": a.vd, "co_tenants_start": co0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "n_layers": m.n_layers,
            "linear_bytes": {f"k{k}": m.linear_bytes(k) for k in (1, 2)},
            "lm_head_bytes": m.lm_head.numel() * 2, "vd_bytes": m.dvocab[a.vd][1].numel() * 2,
            "gpu_mem_alloc_gib": torch.cuda.memory_allocated() / 2**30, "cfg": {}}
    decs = {}
    for c in a.configs:
        k = 1 if c.startswith("k1") else 2
        vd = a.vd if c.endswith("v") else None
        ar = ARDecoder(m, T, k=k, vd=vd)
        ar.generate(prompt, 16)  # warm-up + capture
        _, st = ar.generate(prompt, a.pre)  # ctx -> P + pre; pipelined wall time of these steps
        ws = []
        for _ in range(3):  # pipelined wall per step at ctx P+pre .. P+pre+steps (pos advances; reset after)
            pos0 = int(ar.pos)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(a.steps):
                ar.graph.replay()
            torch.cuda.synchronize()
            ws.append((time.perf_counter() - t) / a.steps * 1e6)
            ar.pos.fill_(pos0)
        meta["cfg"][c] = {"k": k, "vd": vd, "wall_step_us_runs": ws, "wall_step_us": sorted(ws)[1],
                          "ar_generate_step_us": 1e6 * st["t_decode_s"] / st["n_decode_tokens"], "ctx": int(ar.pos)}
        print(f"WALL {c}: {meta['cfg'][c]}", flush=True)
        decs[c] = ar
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for c, ar in decs.items():
        torch.cuda.nvtx.range_push(f"cfg:{c}")
        for i in range(a.steps):
            torch.cuda.nvtx.range_push(f"step:{c}:{i}")
            ar.graph.replay()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    meta["co_tenants_end"] = cotenants() or "none"
    json.dump(meta, open(a.meta, "w"), indent=1)
    print("wrote", a.meta, flush=True)


if __name__ == "__main__":
    main()
