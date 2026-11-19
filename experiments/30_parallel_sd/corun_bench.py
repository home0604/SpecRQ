"""E9 parallel draft || verify feasibility microbench (PEARL-style parallel self-speculation, same GPUs).

Question: when one (or gamma) draft step(s) and one verify step run CONCURRENTLY on the same 4 GPUs (TP=4, two CUDA
streams in each rank process), how much does each slow down, and how does that depend on the draft KV format?

Setup = experiments/26_magicvq/e2e_real.py (imported, not modified): bf16 Llama-3.1-8B-Instruct, TP=4 on GPUs 4-7,
real syn_quote prompts, real prefill, real draft-cache construction, Triton split-K bf16 attention + CUDA LUT VQ
attention, CUDA graphs. Output correctness is NOT checked (draft and verify write the same cache rows concurrently).

Concurrency: draft-side graphs are captured on NCCL communicator gD, verify-side graphs on communicator gV
(dist.new_group over all ranks = separate NCCL communicators), so collectives of the two streams never share a
communicator. e2e_real's module-level `dist` is replaced by a proxy whose all_reduce uses the group selected at
capture time (CURG). Graph pools: one for draft-side graphs, one for verify-side graphs (never replayed concurrently
within a side).

Timing of one measurement (a set of (stream, [graphs])): sync + barrier, a GPU sleep on the current stream (so all
graph launches are queued before the GPU starts), event e0, every stream waits e0, replays its graphs, records its end
event. Per-stream time = e0 -> end; co-run time = max over streams. Per rank, then MAX over ranks. 3 blocks x N iters
after warm-up; median over all iters + per-block medians (spread).

Measurements per draft config and gamma:
  iso   : d1 (one draft step), dG (gamma draft steps back to back, one graph), vG (verify gamma+1 rows)
  corun : d1 || vG and dG || vG, priorities: equal / verify-high / draft-high
  parts : components alone and co-run with the full other side (gamma = 4 only):
          draft {comm (all_reduces only), gemm (forward w/o comm and w/o attention), attn (all-layer attention)}
          verify {comm, gemm, attn}
Draft 'kvfree' = forward with comm, no attention (weights + all_reduce floor).
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
sys.path.insert(0, str(REPO / "experiments" / "26_magicvq"))

import torch  # noqa: E402
import torch.distributed as tdist  # noqa: E402

import e2e_real as R  # noqa: E402

CURG = None  # None = default group; "none" = no communication (component graphs)


class _DistProxy:
    """replaces e2e_real.dist: all_reduce goes to the group chosen at capture time."""
    ReduceOp = tdist.ReduceOp

    def __getattr__(self, k):
        return getattr(tdist, k)

    def all_reduce(self, x, op=tdist.ReduceOp.SUM, group=None):
        if CURG == "none":
            return None
        return tdist.all_reduce(x, op=op, group=group if group is not None else CURG)


R.dist = _DistProxy()


def log(*a):
    R.log(*a)


def capture(fn, pool, group, stream):
    """warm-up + capture on `stream` with collectives on `group`."""
    global CURG
    CURG = group
    torch.cuda.empty_cache()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    tdist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool, stream=stream):
        fn()
    torch.cuda.synchronize()
    tdist.barrier()
    CURG = None
    g._keep_fn = fn  # keep the closure (and the buffers it owns) alive as long as the graph
    if os.environ.get("E9_DEBUG"):
        g.replay()
        torch.cuda.synchronize()
        log("  captured+replayed ok:", getattr(fn, "__name__", "?"), group if isinstance(group, str) else id(group))
    return g


def gmax(x):
    t = torch.tensor(x, device="cuda", dtype=torch.float64)
    tdist.all_reduce(t, op=tdist.ReduceOp.MAX)
    return t.tolist()


class Timer:
    def __init__(self, sleep_ms, blocks, iters, warm):
        self.blocks, self.iters, self.warm = blocks, iters, warm
        # calibrate torch.cuda._sleep cycles per ms
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(10_000_000)
        e.record()
        torch.cuda.synchronize()
        self.cyc = int(10_000_000 / s.elapsed_time(e) * sleep_ms)
        self.sleep_ms = sleep_ms
        self.launch_ms = []

    def once(self, sides):
        """sides: list of (stream, [graph, ...]); -> per-side ms (this rank)."""
        torch.cuda.synchronize()
        tdist.barrier()
        torch.cuda.synchronize()
        cur = torch.cuda.current_stream()
        torch.cuda._sleep(self.cyc)
        e0 = torch.cuda.Event(enable_timing=True)
        e0.record(cur)
        ends = []
        h0 = time.perf_counter()
        for s, gs in sides:
            s.wait_event(e0)
            with torch.cuda.stream(s):
                for g in gs:
                    g.replay()
                e = torch.cuda.Event(enable_timing=True)
                e.record(s)
            ends.append(e)
        self.launch_ms.append((time.perf_counter() - h0) * 1e3)
        torch.cuda.synchronize()
        return [e0.elapsed_time(e) for e in ends]

    def run(self, sides):
        for _ in range(self.warm):
            self.once(sides)
        per = []  # [iter][side] (max over ranks)
        blocks = []
        for _ in range(self.blocks):
            bl = [gmax(self.once(sides)) for _ in range(self.iters)]
            per += bl
            blocks.append(bl)
        n = len(sides)
        med = [statistics.median(x[i] for x in per) for i in range(n)]
        tot = [max(x) for x in per]
        return {"side_ms": med, "side_min": [min(x[i] for x in per) for i in range(n)],
                "side_max": [max(x[i] for x in per) for i in range(n)],
                "total_ms": statistics.median(tot), "total_min": min(tot), "total_max": max(tot),
                "block_total_med": [statistics.median(max(x) for x in bl) for bl in blocks],
                "block_side_med": [[statistics.median(x[i] for x in bl) for i in range(n)] for bl in blocks]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="syn_quote")
    ap.add_argument("--ctx", type=int, default=32768, help="prompt file context")
    ap.add_argument("--trunc", type=int, default=0, help="truncate prompts to this many tokens (timing only)")
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--drafts", nargs="+",
                    default=["kvfree", "snapkv4096", "hyb512pL2@t2w", "hyb512pL1@t1w", "vqpL2+r128@t2w",
                             "vqpL1+r128@t2w"])
    ap.add_argument("--gammas", nargs="+", type=int, default=[4, 6])
    ap.add_argument("--parts_gamma", type=int, default=4)
    ap.add_argument("--no_parts", action="store_true")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--sleep_ms", type=float, default=8.0)
    ap.add_argument("--gen_room", type=int, default=64, help="cache slots beyond the prompt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tdist.init_process_group("nccl")
    R.RANK, R.WORLD = tdist.get_rank(), tdist.get_world_size()
    torch.cuda.set_device(R.RANK)
    W = R.WORLD
    lst = [None] * W
    tdist.all_gather_object(lst, os.getpid())
    own = set(lst)
    rec = {"args": vars(a), "world": W, "co_start": R.cocheck(own),
           "env": {k: os.environ.get(k) for k in ("NCCL_P2P_DISABLE", "CUDA_VISIBLE_DEVICES", "NCCL_ALGO",
                                                   "NCCL_PROTO", "CUDA_DEVICE_MAX_CONNECTIONS")}}
    log("co-tenant at start:", rec["co_start"]["others"] or "none")
    gD = tdist.new_group(list(range(W)))
    gV = tdist.new_group(list(range(W)))
    sD_lo, sV_lo = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=0)
    sD_hi, sV_hi = torch.cuda.Stream(priority=-1), torch.cuda.Stream(priority=-1)
    rec["stream_priority"] = {"sD_lo": sD_lo.priority, "sV_lo": sV_lo.priority, "sD_hi": sD_hi.priority,
                              "sV_hi": sV_hi.priority}
    for g_, s_ in ((gD, sD_lo), (gV, sV_lo)):  # init the communicators
        x = torch.ones(8, device="cuda")
        with torch.cuda.stream(s_):
            tdist.all_reduce(x, group=g_)
        torch.cuda.synchronize()

    prompts = json.load(open(f"{R.OUT}/prompts_{a.task}_ctx{a.ctx}.json"))[:a.B]
    if a.trunc:
        for p in prompts:
            p["ids"] = p["ids"][:a.trunc]
            p["P"] = len(p["ids"])
    m = R.Shard()
    P_list = [p["P"] for p in prompts]
    gmx = max(a.gammas)
    T = -(-(max(P_list) + a.gen_room + gmx + 2) // 64) * 64
    st = R.State(m, a.B, T, a.gen_room, gmx)
    log(f"T={T} cache {st.tc.numel() * 2 / 2**30:.2f} GiB, free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    qlast = [[None] * a.B for _ in range(R.NL)]
    g0 = torch.zeros(a.B, dtype=torch.int64, device="cuda")
    t0 = time.time()
    for b, p in enumerate(prompts):
        g0[b] = R.prefill(m, st.tc, b, torch.tensor(p["ids"], device="cuda"), qlast)
        torch.cuda.empty_cache()
    st.set_prompts(P_list, g0)
    log(f"prefill {time.time() - t0:.0f}s, P mean {sum(P_list) / len(P_list):.0f}")
    rec["P"] = P_list
    cbd = f"{R.OUTROOT}/cb"
    cbs = {"t2w": R.CB(m, f"{cbd}/cb_d8K256x2_post_cn_t2w.pt"), "t1w": R.CB(m, f"{cbd}/cb_d8K256x2_post_cn_t1w.pt")}
    B, hq, hkv, D = a.B, m.hq, m.hkv, R.D
    timer = Timer(a.sleep_ms, a.blocks, a.iters, a.warm)
    poolV, poolS = torch.cuda.graph_pool_handle(), torch.cuda.graph_pool_handle()
    st.reset()

    def fwd_noattn(tokens):
        return R.forward(m, tokens, lambda qkv, l: qkv[:, :hq * D])

    def comm_graph_fn(M):  # the collectives of one forward with M rows / seq: embed + 2 / layer + argmax
        bufs = [torch.zeros(B * M, R.HID, dtype=torch.bfloat16, device="cuda") for _ in range(1 + 2 * R.NL)]
        key = torch.zeros(B * M, dtype=torch.int64, device="cuda")

        def f():
            for x in bufs:
                R.dist.all_reduce(x)
            R.dist.all_reduce(key, op=tdist.ReduceOp.MAX)
        return f

    # ---------------- verify side
    V = {}
    for g in a.gammas:
        G1 = g + 1
        vplan, vtune = R.tune_plan(m, st, G1, None)
        st.reset()
        va = R.Attn(m, st, G1, 0, None, vplan)
        dtok = torch.zeros(B, G1, dtype=torch.int64, device="cuda")
        targ = torch.zeros(B, G1, dtype=torch.int64, device="cuda")
        qkv_r = torch.randn(B * G1, (hq + 2 * hkv) * D, device="cuda").to(torch.bfloat16) * 0.1

        def vfn(va=va, dtok=dtok, targ=targ):
            va.prep(st.length)
            targ.copy_(R.forward(m, dtok.reshape(-1), va).view(B, G1))

        ent = {"plan": vplan, "tune": vtune, "full": capture(vfn, poolV, gV, sV_lo)}
        if not a.no_parts and g == a.parts_gamma:
            ent["comm"] = capture(comm_graph_fn(G1), poolV, gV, sV_lo)
            ent["gemm"] = capture(lambda dtok=dtok: fwd_noattn(dtok.reshape(-1)), poolV, "none", sV_lo)

            def vattn(va=va, q=qkv_r):
                va.prep(st.length)
                for l in range(R.NL):
                    va(q, l)
            ent["attn"] = capture(vattn, poolV, "none", sV_lo)
        V[g] = ent
        log(f"verify g{g}: plan {vplan}")
    rec["verify"] = {g: {"plan": v["plan"], "tune": v["tune"]} for g, v in V.items()}
    rec["drafts"] = {}

    def M(name, sides):
        r = timer.run(sides)
        log(f"  {name}: total {r['total_ms']:.2f} ms sides {['%.2f' % x for x in r['side_ms']]}")
        return r

    # ---------------- draft side
    shared_parts = {}
    dtok0 = torch.zeros(B, dtype=torch.int64, device="cuda")  # persistent input of the shared draft-part graphs
    for dn in a.drafts:
        tc0 = time.time()
        poolD = torch.cuda.graph_pool_handle()  # per draft: freed with the draft's graphs (OOM at 32k otherwise)
        co_b = R.cocheck(own) if R.RANK == 0 else None
        name, _, cbtag = dn.partition("@")
        kvfree = name == "kvfree"
        d = None if kvfree else R.Draft(name, m, st, cbs.get(cbtag) or cbs["t2w"])
        if d is not None:
            d.build(st.tc, qlast)
            torch.cuda.synchronize()
            dplan, dtune = R.tune_plan(m, st, 1, d)
        else:
            dplan, dtune = None, None
        st.reset()
        dtok = torch.zeros(B, gmx + 1, dtype=torch.int64, device="cuda")
        dsteps = [R.Attn(m, st, 1, j, d, dplan) for j in range(gmx)] if d is not None else None

        def dfn(j):  # draft step j (qpos = len + j); gamma steps = replay of steps 0..gamma-1 back to back
            def f():
                if kvfree:
                    dtok[:, j + 1].copy_(fwd_noattn(dtok[:, j]))
                else:
                    a_ = dsteps[j]
                    a_.prep(st.length)
                    dtok[:, j + 1].copy_(R.forward(m, dtok[:, j], a_))
            return f
        torch.cuda.empty_cache()
        steps = [capture(dfn(j), poolD, gD, sD_lo) for j in range(gmx)]
        gr = {n: steps[:n] for n in [1] + list(a.gammas)}
        log(f"  draft graphs captured, free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
        res = {"plan": dplan, "tune": dtune, "iso": {}, "corun": {}, "parts": {}}
        dparts = {}
        log(f"[{dn}] plan {dplan}")
        res["iso"]["d1"] = M("iso d1", [(sD_lo, gr[1])])
        for g in a.gammas:
            res["iso"][f"d{g}"] = M(f"iso d{g}", [(sD_lo, gr[g])])
            res["iso"][f"v{g}"] = M(f"iso v{g}", [(sV_lo, [V[g]["full"]])])
            for n in (1, g):
                for pri, sd, sv in (("eq", sD_lo, sV_lo), ("vhi", sD_lo, sV_hi), ("dhi", sD_hi, sV_lo)):
                    res["corun"][f"d{n}|v{g}|{pri}"] = M(f"corun d{n}||v{g} {pri}", [(sd, gr[n]), (sv, [V[g]["full"]])])
        if not a.no_parts:
            g = a.parts_gamma
            P_ = res["parts"]
            # draft components (comm / gemm shared across drafts: measured once, under the first draft)
            if not shared_parts:
                shared_parts["comm"] = capture(comm_graph_fn(1), poolS, gD, sD_lo)
                shared_parts["gemm"] = capture(lambda: fwd_noattn(dtok0), poolS, "none", sD_lo)
            dparts = dict(shared_parts)
            if d is not None:
                qkv_d = torch.randn(B, (hq + 2 * hkv) * D, device="cuda").to(torch.bfloat16) * 0.1

                def dattn(q=qkv_d):
                    a_ = dsteps[0]
                    a_.prep(st.length)
                    for l in range(R.NL):
                        a_(q, l)
                dparts["attn"] = capture(dattn, poolD, "none", sD_lo)
            for k, gk in dparts.items():
                P_[f"d_{k}"] = M(f"iso d_{k}", [(sD_lo, [gk])])
                P_[f"d_{k}|v{g}"] = M(f"corun d_{k}||v{g}", [(sD_lo, [gk]), (sV_lo, [V[g]["full"]])])
            for k in ("comm", "gemm", "attn"):
                if dn == a.drafts[0]:
                    P_[f"v_{k}"] = M(f"iso v_{k}", [(sV_lo, [V[g][k]])])
                P_[f"d1|v_{k}"] = M(f"corun d1||v_{k}", [(sD_lo, gr[1]), (sV_lo, [V[g][k]])])
            if d is not None:
                del dparts["attn"]
        res["co_before"], res["co_after"] = co_b, R.cocheck(own) if R.RANK == 0 else None
        res["wall_s"] = time.time() - tc0
        rec["drafts"][dn] = res
        del gr, steps
        if d is not None:
            d.free()
        del d, dsteps, dparts
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if R.RANK == 0:
            json.dump(rec, open(a.out + ".partial", "w"), indent=1)
        log(f"[{dn}] done {time.time() - tc0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    rec["co_end"] = R.cocheck(own)
    rec["cotenant_any"] = bool(rec["co_start"]["cotenant"] or rec["co_end"]["cotenant"] or any(
        (x.get("co_before") or {}).get("cotenant") or (x.get("co_after") or {}).get("cotenant")
        for x in rec["drafts"].values()))
    rec["launch_ms"] = {"median": statistics.median(timer.launch_ms), "max": max(timer.launch_ms),
                        "sleep_ms": timer.sleep_ms}
    if R.RANK == 0:
        json.dump(rec, open(a.out, "w"), indent=1)
        os.remove(a.out + ".partial")
    log("done; co-tenant at end:", rec["co_end"]["others"] or "none", "launch ms", rec["launch_ms"])
    # destroy_process_group() hung here in the smoke test (graphs still hold collectives of gD / gV): exit directly
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
