"""E9 diagnostic: do two CUDA streams (CUDA graphs, separate NCCL communicators) really run concurrently on TP=4 4090s?

Pairs of synthetic graphs, alone and co-run (same Timer as corun_bench):
  sleep   : torch.cuda._sleep for ~S ms (occupies 1 block: pure latency)  -> co-run must be max, not sum
  gemv    : bf16 [B, 4096] x [4096, N]^T chain reading ~R MB of weights   (HBM-bandwidth bound)
  comm    : 65 bf16 all_reduce [B, 4096] + 1 int64 (one decode forward's collectives)
  comm7   : same with 7 rows / sequence (verify gamma 6)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.distributed as tdist  # noqa: E402

import corun_bench as CB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tdist.init_process_group("nccl")
    rank, W = tdist.get_rank(), tdist.get_world_size()
    CB.R.RANK, CB.R.WORLD = rank, W
    torch.cuda.set_device(rank)
    lst = [None] * W
    tdist.all_gather_object(lst, os.getpid())
    own = set(lst)
    rec = {"args": vars(a), "co_start": CB.R.cocheck(own)}
    gD, gV = tdist.new_group(list(range(W))), tdist.new_group(list(range(W)))
    sD, sV = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=0)
    sVh = torch.cuda.Stream(priority=-1)
    for g_, s_ in ((gD, sD), (gV, sV)):
        x = torch.ones(8, device="cuda")
        with torch.cuda.stream(s_):
            tdist.all_reduce(x, group=g_)
        torch.cuda.synchronize()
    B = a.B
    timer = CB.Timer(8.0, 3, 10, 2)
    pool = {"D": torch.cuda.graph_pool_handle(), "V": torch.cuda.graph_pool_handle()}
    cyc = timer.cyc  # cycles per 8 ms

    def mk(kind, side):
        g_ = gD if side == "D" else gV
        s_ = sD if side == "D" else sV
        if kind == "sleep":
            f = lambda: torch.cuda._sleep(cyc)  # noqa: E731
            return CB.capture(f, pool[side], "none", s_)
        if kind == "gemv":  # ~ 32 layers x (qkv+o+gu+down)/4 = one TP4 rank's decode weights (~3.5 GB)
            ws = [torch.randn(n, 4096, device="cuda", dtype=torch.bfloat16) for n in (1536, 4096, 7168)]
            wd = torch.randn(4096, 3584, device="cuda", dtype=torch.bfloat16)
            x = torch.randn(B, 4096, device="cuda", dtype=torch.bfloat16)

            def f():
                for _ in range(32):
                    y = x @ ws[0].t()
                    y = x @ ws[1].t()[:, :1024]
                    y = x @ ws[2].t()
                    y = y[:, :3584] @ wd.t()
            f.keep = (ws, wd, x)
            return CB.capture(f, pool[side], "none", s_)
        M = 7 if kind == "comm7" else 1
        bufs = [torch.zeros(B * M, 4096, dtype=torch.bfloat16, device="cuda") for _ in range(65)]
        key = torch.zeros(B * M, dtype=torch.int64, device="cuda")

        def f():
            for x in bufs:
                tdist.all_reduce(x, group=g_)
            tdist.all_reduce(key, op=tdist.ReduceOp.MAX, group=g_)
        return CB.capture(f, pool[side], g_, s_)
    kinds = ["sleep", "gemv", "comm", "comm7"]
    G = {(k, s): mk(k, s) for k in kinds for s in "DV"}
    res = {}
    for k in kinds:
        res[f"iso {k}"] = timer.run([(sD, [G[(k, 'D')]])])
    pairs = [("sleep", "sleep"), ("gemv", "gemv"), ("comm", "comm"), ("comm", "comm7"), ("gemv", "comm7"),
             ("comm", "gemv"), ("sleep", "gemv"), ("sleep", "comm7"), ("gemv", "sleep")]
    for x, y in pairs:
        r = timer.run([(sD, [G[(x, 'D')]]), (sV, [G[(y, 'V')]])])
        res[f"{x}|{y}"] = r
        r2 = timer.run([(sD, [G[(x, 'D')]]), (sVh, [G[(y, 'V')]])])
        res[f"{x}|{y} vhi"] = r2
        CB.log(f"{x:6s}|{y:6s}: iso {res[f'iso {x}']['total_ms']:.2f} / {res[f'iso {y}']['total_ms']:.2f}  corun total "
               f"{r['total_ms']:.2f} sides {['%.2f' % v for v in r['side_ms']]}  vhi sides "
               f"{['%.2f' % v for v in r2['side_ms']]}")
    rec["res"] = res
    rec["co_end"] = CB.R.cocheck(own)
    rec["env"] = {k: os.environ.get(k) for k in ("NCCL_P2P_DISABLE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS")}
    if rank == 0:
        json.dump(rec, open(a.out, "w"), indent=1)
    CB.log("done; co-tenant end:", rec["co_end"]["others"] or "none")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
