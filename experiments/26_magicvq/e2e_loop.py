"""E7 magicvq round 3: end-to-end speculative decoding LOOP on TP=4 (GPUs 4-7) with real kernels and real loop control,
synthetic weights, and per-sequence acceptance REPLAYED from measured greedy match traces (tau_eval --dump_matches).

Per round (all B sequences in lockstep, as in MagicDec's batched self-speculation):
  gamma x draft step graph (M=1)  ->  verify step graph (M=gamma+1)  ->  host reads a GPU acceptance tensor
  (synchronizing .tolist(), as a real greedy compare would)  ->  accepted counts from the replayed traces ->
  encode of the gamma+1 verified tokens into the draft cache (VQ / KIVI drafts; graph)  ->  next round.
AR baseline loop: target step graph (M=1) + per-step host read of the argmax tensor.
Throughput = generated tokens (sum over sequences, accepted + 1 per round) / wall time; reported spec / AR.
Simplifications [stated in the report]: synthetic weights (shapes real); fixed cache length (graphs captured at ctx,
caches do not grow during the timed window); acceptance replayed from traces (sequence b uses trace b mod n_traces,
starting at a random offset; traces wrap).
Usage: torchrun --nproc_per_node 4 experiments/26_magicvq/e2e_loop.py --B 16 --ctx 32768 --draft evict512 \
          --tau_draft snapkv512 --gamma 2 --traces <tau json> --tag v1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import mgpu_bench as MB  # noqa: E402
from specrq.kvq.kernels import KVCache, vq_encode  # noqa: E402
from kivi_attn import GRP, KIVICache  # noqa: E402
from kvq_prerope import GSBooks  # noqa: E402


def graph(fn):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return g


def build_draft(m, name, B, ctx, T, a):
    """-> (Step for M=1, encode fn for gamma+1 rows or None)."""
    G1 = a.gamma + 1
    if name.startswith("evict"):
        W = int(name[5:])
        kv = KVCache(m.nl, B, m.hkv, W + 8, mode="bf16")
        best, _ = MB.tune(m, "bf16", kv, B, W - 1, 1, None, 2, 3)
        return MB.Step(m, "bf16", kv, B, W - 1, 1, best["backend"], best["splits"], best["chunk"]), None
    if name.startswith("hyb"):  # v4b: hyb<W>rq2 = RQ over ctx - W tokens + bf16 W tokens (LSE merge), no encode
        import re
        mm = re.fullmatch(r"hyb(\d+)(rq1|rq2|rq05)", name)
        W, base = int(mm.group(1)), mm.group(2)
        kw, kst = {"rq1": (dict(d=8, bits=[8, 8]), 1), "rq2": (dict(d=8, bits=[8, 8]), 2),
                   "rq05": (dict(d=16, bits=[8, 8]), 1)}[base]
        kv = KVCache(m.nl, B, m.hkv, T, mode="vq", **kw)
        best, _ = MB.tune(m, "vq", kv, B, ctx - W, 1, kst, 2, 3)
        wc = KVCache(m.nl, B, m.hkv, W + 8, mode="bf16")
        return MB.Step(m, "vq", kv, B, ctx - W, 1, best["backend"], best["splits"], best["chunk"], kst, wc, W), None
    base, _, r = name.partition("+r")
    R = int(r) if r else None
    if base.startswith("rq"):
        kw, kst = {"rq1": (dict(d=8, bits=[8, 8]), 1), "rq2": (dict(d=8, bits=[8, 8]), 2),
                   "rq05": (dict(d=16, bits=[8, 8]), 1)}[base]
        kv = KVCache(m.nl, B, m.hkv, T, mode="vq", **kw)
        best, _ = MB.tune(m, "vq", kv, B, ctx, 1, kst, 2, 3)
        wc = KVCache(m.nl, B, m.hkv, R + 24, mode="bf16") if R else None
        st = MB.Step(m, "vq", kv, B, ctx, 1, best["backend"], best["splits"], best["chunk"], kst, wc,
                     (R + 16) if R else 0)
        se = MB.Step(m, "vq", kv, B, ctx, G1, "triton", kst=kst)
        return st, (lambda: [vq_encode(se.qkv_d, kv, li, ctx, G1, m.hq, m.inv, kst) for li in range(m.nl)])
    if base.startswith("kivi"):
        bits = int(base[4:])
        kv = KIVICache(m.nl, B, m.hkv, T, bits)
        aux = MB.kivi_aux(m, B, bits)
        best, _ = MB.tune(m, "kivi", kv, B, ctx, 1, None, 2, 3, aux)
        wc = KVCache(m.nl, B, m.hkv, R + 24, mode="bf16")
        st = MB.Step(m, "kivi", kv, B, ctx, 1, best["backend"], best["splits"], best["chunk"], None, wc, R + 16, aux)
        se = MB.Step(m, "kivi", kv, B, ctx, G1, "triton", aux=aux)
        cnt = {"n": 0}

        def enc():
            for _ in range(m.nl):
                aux["vq"](se.qkv_d[:, (m.hq + m.hkv) * MB.D:].view(se.R, m.hkv, MB.D))
        return st, enc
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--tau_draft", required=True)
    ap.add_argument("--gamma", type=int, required=True)
    ap.add_argument("--traces", required=True, help="tau_eval json written with --dump_matches")
    ap.add_argument("--set", default="pg19")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    MB.DIST = dist
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    lst = [None] * world
    dist.all_gather_object(lst, os.getpid())
    own = set(lst)
    cot = lambda: [c for c in MB.cotenants() if c[0] not in own]  # noqa: E731
    co0 = cot()
    m = MB.Model(32, 32 // world, 8 // world, 14336 // world, MB.VOCAB // world)
    B, ctx, g = a.B, a.ctx, a.gamma
    T = ((ctx + g + 9 + GRP - 1) // GRP) * GRP
    kvt = KVCache(m.nl, B, m.hkv, T, mode="bf16")
    bt, _ = MB.tune(m, "bf16", kvt, B, ctx, 1, None, 2, 3)
    bv, _ = MB.tune(m, "bf16", kvt, B, ctx, g + 1, None, 2, 3)
    s_ar = MB.Step(m, "bf16", kvt, B, ctx, 1, bt["backend"], bt["splits"], bt["chunk"])
    s_v = MB.Step(m, "bf16", kvt, B, ctx, g + 1, bv["backend"], bv["splits"], bv["chunk"])
    s_d, enc = build_draft(m, a.draft, B, ctx, T, a)
    torch.cuda.empty_cache()  # v4b: free tuning leftovers before graph capture (hyb B16x32k OOM otherwise)
    g_ar, g_v, g_d = graph(s_ar.full), graph(s_v.full), graph(s_d.full)
    g_e = graph(enc) if enc is not None else None
    # acceptance traces
    tj = json.load(open(a.traces))
    traces = [r["drafts"][a.tau_draft]["matches"] for r in tj["per_prompt"] if r["set"] == a.set]
    rng = random.Random(0)
    flag = torch.zeros(B, device="cuda")

    def accepted(tr, pos):
        n = len(tr[0])
        k = 0
        while k < g and tr[k][(pos + k) % n]:
            k += 1
        return k
    res = {"spec": [], "ar": []}
    for rep in range(a.reps):
        # --- spec loop
        seqs = [(traces[b % len(traces)], rng.randrange(len(traces[b % len(traces)][0]))) for b in range(B)]
        pos = [p for _, p in seqs]
        toks = 0
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.rounds):
            for _ in range(g):
                g_d.replay()
            g_v.replay()
            flag.tolist()  # host sync: greedy compare result read back (as in a real loop)
            for b in range(B):
                k = accepted(seqs[b][0], pos[b])
                pos[b] += k + 1
                toks += k + 1
            if g_e is not None:
                g_e.replay()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res["spec"].append({"tokens": toks, "s": dt, "tok_per_s": toks / dt})
        # --- AR loop
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        steps = a.rounds * 2
        for _ in range(steps):
            g_ar.replay()
            flag.tolist()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res["ar"].append({"tokens": steps * B, "s": dt, "tok_per_s": steps * B / dt})
    # global (slowest rank) wall times
    t = torch.tensor([[x["s"] for x in res["spec"]], [x["s"] for x in res["ar"]]], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    spec_tps = [x["tokens"] / s for x, s in zip(res["spec"], t[0].tolist())]
    ar_tps = [x["tokens"] / s for x, s in zip(res["ar"], t[1].tolist())]
    tau_meas = res["spec"][0]["tokens"] / (a.rounds * B)
    out = {"args": vars(a), "spec_tok_per_s": spec_tps, "ar_tok_per_s": ar_tps,
           "speedup_e2e_median": statistics.median(spec_tps) / statistics.median(ar_tps),
           "tau_replayed": tau_meas, "cotenants_start": co0, "cotenants_end": cot(),
           "draft_backend": s_d.backend, "target_backend": bt["backend"]}
    if rank == 0:
        os.makedirs(MB.OUT, exist_ok=True)
        fn = f"{MB.OUT}/e2e_{a.tag}_B{B}_ctx{ctx}_{a.draft}_g{g}_{a.set}.json"
        json.dump(out, open(fn, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "args"}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
