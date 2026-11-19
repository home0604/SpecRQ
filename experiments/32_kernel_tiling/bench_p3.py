"""E10 phase 3 timing: A100-oriented LUT kernel variants (csrc/lut_rag_p3.cuh) next to the unmodified original
(phase-0 extension, 'orig_k{1,2}') and the phase-1 best struct (p1_ext, 'p1:k{1,2}_w8_b1_r1_p1_l0_s2'). Same harness
as bench_p1: CUDA graphs over R code copies (>= l2_mult x L2) so codes stream from DRAM; t_tot = partial + Triton
combine. Per (impl, kst, n, bh) the measured chunks = phase-0 chunk set (<= n, plus one chunk >= n) + split_plan
candidates + the wave-aware rule pick + explicit --splits; oracle = min over them (done in the analysis).

Timing: per point a warm-up of --warm_ms of graph replays (the A100 shows slow, bimodal first replays), then --reps
reps of m replays (m chosen for ~3 ms per rep); median, min and max (us per call) are recorded.
Co-tenant check: nvidia-smi compute apps before and after the sweep are stored (timing must run on an empty GPU).
Usage: python bench_p3.py --out raw.json --variants <p3 names> [--orig] [--p1] [--kst ..] [--n ..] [--bh ..] [--splits ..]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

import bench_p0 as BP
import p3_ext as PX
from bench_p1 import ceil64, rule_splits
from specrq.kvq.kernels import split_plan

D, G, C = 128, 4, 256
P1_STRUCT = {1: (1, 8, 1, 1, 1, 0, 2), 2: (2, 8, 1, 1, 1, 0, 2)}


def gtime(fn, reps, warm_ms, target_ms=3.0):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay()
    torch.cuda.synchronize()
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    one = max(e0.elapsed_time(e1), 1e-3)
    for _ in range(max(1, math.ceil(warm_ms / one))):
        g.replay()
    torch.cuda.synchronize()
    m = max(1, min(200, math.ceil(target_ms / one)))
    ts = []
    for _ in range(reps):
        e0.record()
        for _ in range(m):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / m)
    del g
    return statistics.median(ts), min(ts), max(ts), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--orig", action="store_true")
    ap.add_argument("--orig_lazy", action="store_true", help="also the original with its lazy-max flag on")
    ap.add_argument("--p1", action="store_true", help="also time the phase-1 best struct (p1_ext)")
    ap.add_argument("--kst", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--bh", type=int, nargs="+", default=[8, 32, 256])
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--splits", type=int, nargs="*", default=[], help="extra splits candidates")
    ap.add_argument("--only_splits", action="store_true", help="measure only --splits (no chunk set / rule)")
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warm_ms", type=float, default=50.0)
    ap.add_argument("--ws_cap_gb", type=float, default=3.0)
    ap.add_argument("--l2_mult", type=float, default=3.0)
    ap.add_argument("--max_copies", type=int, default=256)
    ap.add_argument("--no_part", action="store_true", help="time only t_tot (halves the sweep time)")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    apps_before = BP.nvsmi_apps()
    uuid = BP.my_gpu_uuid()
    t_build = time.time()
    E = PX.build([PX.parse(v) for v in a.variants]) if a.variants else None
    ext0 = BP.build_ext() if a.orig else None
    E1 = None
    if a.p1:
        import p1_ext
        E1 = p1_ext.build([P1_STRUCT[k] for k in a.kst])
    t_build = time.time() - t_build
    lim = E.device_limits() if E else dict(zip(
        ["sm_count", "smem_per_sm", "smem_per_block_optin", "reserved_smem_per_block", "regs_per_sm", "threads_per_sm",
         "blocks_per_sm", "l2_bytes", "cc_major", "cc_minor"], ext0.device_limits()))
    sms, l2 = lim["sm_count"], lim["l2_bytes"]
    V = {}
    if ext0 is not None:
        for k in a.kst:
            o = dict(zip(["blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads_per_block",
                          "threads", "vcb_bytes", "lut_bytes", "warp_off", "tail_bytes", "warp_bytes"],
                         ext0.occupancy(k)))
            o["slots_per_split"] = 16
            V[f"orig_k{k}"] = dict(kst=k, impl="orig", occ=o, lazy=0)
            if a.orig_lazy:
                V[f"origlazy_k{k}"] = dict(kst=k, impl="orig", occ=o, lazy=1)
    if E1 is not None:
        for nm in E1.names:
            V["p1:" + nm] = dict(kst=E1.fields[nm]["kst"], impl="p1", occ=E1.occ[nm], p1name=nm)
    if E is not None:
        for nm in E.names:
            V[nm] = dict(**E.fields[nm], impl="p3", occ=E.occ[nm])
    for nm, v in V.items():
        print(nm, v["occ"], flush=True)
    skipped = [nm for nm, v in V.items() if v["occ"]["blocks_per_sm"] < 1]
    for nm in skipped:
        print("NOT LAUNCHABLE (smem):", nm, flush=True)
        V.pop(nm)

    x = torch.empty(512 * 2 ** 20 // 4, dtype=torch.float32, device=dev).uniform_()
    r = torch.empty((), device=dev)
    tm, _, _, _ = gtime(lambda: torch.sum(x, dim=0, out=r), 7, a.warm_ms)
    bw_ref = x.numel() * 4 / (tm * 1e-3) / 1e9
    del x
    print(f"reference read BW: {bw_ref:.1f} GB/s", flush=True)

    inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
    rows = []
    t_start = time.time()
    for bh in a.bh:
        B, Hkv = bh // a.hkv, a.hkv
        Hq = Hkv * G
        qkv = (torch.randn(B, (Hq + 2 * Hkv) * D, device=dev) * 0.5).to(torch.bfloat16)
        qpos = torch.full((B,), 200000, device=dev, dtype=torch.int32)
        mu = torch.randn(2, Hkv, D, device=dev) * 0.3
        sd = torch.rand(2, Hkv, D, device=dev) + 0.5
        lo = torch.zeros(bh, device=dev, dtype=torch.int32)
        for n in a.n:
            hi = torch.full((bh,), n, device=dev, dtype=torch.int32)
            for kst in a.kst:
                vk = [nm for nm, v in V.items() if v["kst"] == kst]
                if not vk:
                    continue
                bytes_codes = 2 * kst * bh * n * 16
                R = max(1, min(a.max_copies, math.ceil(a.l2_mult * l2 / bytes_codes)))
                planes = [torch.randint(0, 256, (2, kst, B, Hkv, n, 16), dtype=torch.uint8, device=dev)
                          for _ in range(R)]
                cb = (torch.randn(2, Hkv, kst, C, 8, device=dev) *
                      torch.tensor([1.0, 0.4], device=dev)[None, None, :kst, None, None]).to(torch.bfloat16)
                base = set()
                if not a.only_splits:
                    base = set(c for c in a.chunks if c < n) | {min([c for c in a.chunks if c >= n] + [n])}
                    for tp in (128, 256, 512, 1024, 2048):
                        base.add(split_plan(B, Hkv, n, 64, tp)[1])
                for s_ in a.splits:
                    base.add(ceil64(-(-n // s_)))
                for nm in vk:
                    v = V[nm]
                    occ = v["occ"]
                    slots = occ["slots_per_split"]
                    rs = rule_splits(n, bh, sms, occ["blocks_per_sm"], slots)
                    rule_chunk = ceil64(-(-n // rs))
                    cands = set(base) | ({rule_chunk} if not a.only_splits else set())
                    seen = set()
                    for chunk in sorted(cands):
                        splits = -(-n // chunk)
                        if splits in seen:
                            continue
                        seen.add(splits)
                        spt = slots * splits
                        if bh * spt * 16 * (D + 2) * 4 > a.ws_cap_gb * 2 ** 30:
                            continue
                        ws = BP.EK.workspace(B, Hkv, G, 1, spt)
                        out = torch.empty(B, Hq * D, dtype=torch.bfloat16, device=dev)

                        def call(p, h_, nm=nm, chunk=chunk, splits=splits, spt=spt, ws=ws):
                            args = (qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)],
                                    [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv,
                                    ws[0], ws[1], ws[2], lo, h_, qpos, sd[0], mu[0], sd[1], mu[1], chunk, splits,
                                    B, Hkv, ws[3], spt, 0)
                            imp = V[nm]["impl"]
                            if imp == "orig":
                                ext0.lut_rag(*args, kst, BP.D_SCALE, V[nm]["lazy"])
                            elif imp == "p1":
                                E1.run(V[nm]["p1name"], *args, BP.D_SCALE)
                            else:
                                E.run(nm, *args, BP.D_SCALE)

                        def part(call=call):
                            for p in planes:
                                call(p, hi)

                        def tot(call=call, ws=ws, spt=spt, out=out):
                            for p in planes:
                                call(p, hi)
                                BP.EK.combine(ws, spt, out, B, Hkv, G, 1)

                        res = {}
                        fns = [("tot", tot)] + ([] if a.no_part else [("part", part)])
                        for fnm, f in fns:
                            md, mn, mx, m = gtime(f, a.reps, a.warm_ms)
                            res[fnm] = (md / R * 1e3, mn / R * 1e3, mx / R * 1e3)
                        nblk = bh * splits
                        row = dict(variant=nm, impl=v["impl"], kst=kst, n=n, bh=bh, B=B, Hkv=Hkv, chunk=chunk,
                                   splits=splits, grid=nblk, copies=R, slots=slots, bps=occ["blocks_per_sm"],
                                   regs=occ["regs"], waves=math.ceil(nblk / (sms * occ["blocks_per_sm"])),
                                   rule_splits=rs, is_rule=chunk == rule_chunk,
                                   t_tot_us=res["tot"][0], t_tot_min=res["tot"][1], t_tot_max=res["tot"][2],
                                   t_part_us=res["part"][0] if "part" in res else None,
                                   t_part_min=res["part"][1] if "part" in res else None,
                                   bytes_codes=bytes_codes,
                                   code_GBps=bytes_codes / (res["tot"][0] * 1e-6) / 1e9)
                        rows.append(row)
                        print(f"{nm:30s} k{kst} n{n:6d} bh{bh:3d} sp{splits:4d} | tot {row['t_tot_us']:8.1f} "
                              f"(min {row['t_tot_min']:8.1f}) part {row['t_part_us'] or 0:8.1f} | "
                              f"{row['code_GBps']:6.1f} GB/s{' R' if row['is_rule'] else ''}", flush=True)
                        del ws, out
                del planes
                torch.cuda.empty_cache()
    apps_after = BP.nvsmi_apps()
    rec = dict(device=torch.cuda.get_device_name(), gpu_uuid=uuid, limits=lim, variants=V, skipped=skipped,
               bw_ref_GBps=bw_ref, torch=torch.__version__, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               arch=os.environ.get("TORCH_CUDA_ARCH_LIST"), build_s=t_build, sweep_s=time.time() - t_start,
               args=vars(a), apps_before=apps_before, apps_after=apps_after,
               cotenant_note="apps_* = nvidia-smi compute apps (all GPUs); our GPU uuid is gpu_uuid",
               kernel_src="experiments/32_kernel_tiling/csrc/lut_rag_p3.cuh (+ orig 26_magicvq/csrc/vqattn_lut_rag.cu, "
                          "p1 csrc/lut_rag_p1.cuh)",
               commit=subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out, "apps_after", apps_after, flush=True)


if __name__ == "__main__":
    main()
