"""E10 phase 2: new structural axes (LUT source, stream-K decomposition, cp.async code pipeline, fused combine) of
the LUT VQ-KV draft-attention kernel (csrc/lut_rag_p2.cuh), next to the phase-1 global struct (p1 extension,
'p1:w8_b1_r1_p1_l0_s2') and the unmodified original ('orig'). Same harness / timing as bench_p1: CUDA graphs over
R code copies >= 3x L2, median of reps, t_tot = everything one attention call needs (LUT build kernel if LS 1,
partial kernel, Triton combine if FUSE 0).

Runtime knob per (variant, condition), oracle = min over the measured knobs:
  DEC 0 / p1 / orig : chunk in phase-1 set (phase-0 chunks < n + one >= n, split_plan candidates, wave rule pick)
  DEC 1             : P (block count) in {SMs*bps*m, m=1..4} U {bh * splits of the DEC-0 chunk set}
Usage: python bench_p2.py --out raw.json --set screen [--variants ...] [--p1] [--orig] [--kst ..] [--n ..] [--bh ..]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

import bench_p0 as BP
import bench_p1 as B1
import p1_ext as P1X
import p2_common as PC
import p2_ext as PX
import p2_sets
from specrq.kvq.kernels import split_plan

D, G, C = 128, 4, 256
P1_STRUCT = "w8_b1_r1_p1_l0_s2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default=None)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--p1", action="store_true", help="also time the phase-1 global struct (p1 extension)")
    ap.add_argument("--orig", action="store_true", help="also time the unmodified original kernel")
    ap.add_argument("--kst", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--bh", type=int, nargs="+", default=[8, 32, 256])
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--pmult", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--knobs", default=None, help="json {variant|cond key: [knobs]} to restrict knobs (rep runs)")
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--ws_cap_gb", type=float, default=3.0)
    ap.add_argument("--l2_mult", type=float, default=3.0)
    ap.add_argument("--max_copies", type=int, default=256)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    apps_before = BP.nvsmi_apps()
    uuid = BP.my_gpu_uuid()
    t_build = time.time()
    vs = [PX.parse(v) for v in a.variants] + (p2_sets.SETS[a.set]() if a.set else [])
    E = PX.build(vs) if vs else None
    E1 = P1X.build(__import__("p1_sets").final()) if a.p1 else None
    ext0 = PC.build_orig() if a.orig else None
    t_build = time.time() - t_build
    lim = E.device_limits() if E else E1.device_limits()
    sms, l2 = lim["sm_count"], lim["l2_bytes"]
    V = {}
    if ext0 is not None:
        for k in a.kst:
            o = dict(zip(["blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads_per_block",
                          "threads", "vcb_bytes", "lut_bytes", "warp_off", "tail_bytes", "warp_bytes"],
                         ext0.occupancy(k)))
            V[f"orig_k{k}"] = dict(kind="orig", kst=k, dec=0, fuse=0, ls=0, occ=o, slots=16)
    if E1 is not None:
        for k in a.kst:
            nm = f"k{k}_{P1_STRUCT}"
            V[f"p1:{nm}"] = dict(kind="p1", kst=k, dec=0, fuse=0, ls=0, occ=E1.occ[nm], slots=1, ename=nm)
    if E is not None:
        for nm in E.names:
            if E.fields[nm]["kst"] in a.kst:
                V[nm] = dict(kind="p2", **E.fields[nm], occ=E.occ[nm], slots=1)
    for nm, v in V.items():
        print(nm, v["occ"], flush=True)
    for nm in [nm for nm, v in V.items() if v["occ"]["blocks_per_sm"] < 1]:
        print("NOT LAUNCHABLE (smem):", nm, flush=True)
        V.pop(nm)
    knobs_sel = json.load(open(a.knobs)) if a.knobs else None

    x = torch.empty(512 * 2 ** 20 // 4, dtype=torch.float32, device=dev).uniform_()
    r = torch.empty((), device=dev)
    tm, _, _, _ = BP.gtime(lambda: torch.sum(x, dim=0, out=r), 7)
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
            lens = [n] * bh
            for kst in a.kst:
                vk = [nm for nm, v in V.items() if v["kst"] == kst]
                if not vk:
                    continue
                bufs = PC.Bufs(B, Hkv, kst)
                bytes_codes = 2 * kst * bh * n * 16
                R = max(1, min(a.max_copies, math.ceil(a.l2_mult * l2 / bytes_codes)))
                planes = [torch.randint(0, 256, (2, kst, B, Hkv, n, 16), dtype=torch.uint8, device=dev)
                          for _ in range(R)]
                cb = (torch.randn(2, Hkv, kst, C, 8, device=dev) *
                      torch.tensor([1.0, 0.4], device=dev)[None, None, :kst, None, None]).to(torch.bfloat16)
                base = set(c for c in a.chunks if c < n) | {min([c for c in a.chunks if c >= n] + [n])}
                for tp in (128, 256, 512, 1024, 2048):
                    base.add(split_plan(B, Hkv, n, 64, tp)[1])
                ckey = f"k{kst}_n{n}_bh{bh}"
                for nm in vk:
                    v = V[nm]
                    occ = v["occ"]
                    bps = occ["blocks_per_sm"]
                    if v["dec"] == 0:
                        rs = B1.rule_splits(n, bh, sms, bps, v["slots"])
                        cands = sorted(base | {B1.ceil64(-(-n // rs))})
                    else:
                        cands = sorted({sms * bps * m for m in a.pmult} | {bh * (-(-n // c)) for c in base})
                    if knobs_sel is not None:
                        cands = knobs_sel.get(f"{nm}|{ckey}", [])
                    seen = set()
                    for kb in cands:
                        if v["kind"] == "p2":
                            pl = PC.plan(v["dec"], lens, kb)
                        else:
                            sp = -(-n // kb)
                            pl = dict(chunk=kb, splits=sp, nblk=bh * sp, spt=v["slots"] * sp, cu=None)
                        sig = (pl["chunk"], pl["nblk"])
                        if sig in seen:
                            continue
                        seen.add(sig)
                        if bh * pl["spt"] * 16 * (D + 2) * 4 > a.ws_cap_gb * 2 ** 30:
                            continue
                        ws = PC.workspace(bh, pl["spt"])
                        cu_t = (torch.tensor(pl["cu"], dtype=torch.int32, device=dev) if pl["cu"] is not None
                                else lo)

                        def one(p, nm=nm, v=v, pl=pl, ws=ws, cu_t=cu_t):
                            if v["kind"] == "p2":
                                PC.call(E, nm, qkv, p, cb, inv, ws, lo, hi, qpos, sd, mu, pl, cu_t, bufs, B, Hkv, kst,
                                        BP.D_SCALE)
                            else:
                                args = (qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)],
                                        [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv,
                                        ws[0], ws[1], ws[2], lo, hi, qpos, sd[0], mu[0], sd[1], mu[1], pl["chunk"],
                                        pl["splits"], B, Hkv, ws[3], pl["spt"], 0)
                                if v["kind"] == "orig":
                                    ext0.lut_rag(*args, kst, BP.D_SCALE)
                                else:
                                    E1.run(v["ename"], *args, BP.D_SCALE)
                            if not v["fuse"]:
                                BP.EK.combine(ws, pl["spt"], bufs.out, B, Hkv, G, 1)

                        def tot(one=one):
                            for p in planes:
                                one(p)

                        md, mn, mx, m = BP.gtime(tot, a.reps)
                        t = (md / R * 1e3, mn / R * 1e3, mx / R * 1e3)
                        row = dict(variant=nm, kind=v["kind"], kst=kst, n=n, bh=bh, B=B, Hkv=Hkv,
                                   nw=v.get("nw", 16 if v["kind"] == "orig" else 8), ls=v["ls"], dec=v["dec"],
                                   pipe=v.get("pipe", 0), fuse=v["fuse"], bps=bps, regs=occ["regs"], knob=kb,
                                   chunk=pl["chunk"], splits=pl["splits"], nblk=pl["nblk"], spt=pl["spt"], copies=R,
                                   waves=math.ceil(pl["nblk"] / (sms * bps)),
                                   t_tot_us=t[0], t_tot_min=t[1], t_tot_max=t[2], bytes_codes=bytes_codes,
                                   code_GBps=bytes_codes / (t[0] * 1e-6) / 1e9)
                        rows.append(row)
                        print(f"{nm:28s} n{n:6d} bh{bh:3d} knob{kb:6d} chunk{pl['chunk']:6d} nblk{pl['nblk']:5d} | "
                              f"tot {t[0]:8.1f} [{t[1]:.1f},{t[2]:.1f}] | {row['code_GBps']:6.1f} GB/s", flush=True)
                        del ws
                del planes, bufs
                torch.cuda.empty_cache()
    apps_after = BP.nvsmi_apps()
    rec = dict(device=torch.cuda.get_device_name(), gpu_uuid=uuid, limits=lim,
               variants={k: {kk: vv for kk, vv in v.items()} for k, v in V.items()}, bw_ref_GBps=bw_ref,
               torch=torch.__version__, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               arch=os.environ.get("TORCH_CUDA_ARCH_LIST"), build_s=t_build, sweep_s=time.time() - t_start,
               args=vars(a), apps_before=apps_before, apps_after=apps_after,
               kernel_src="experiments/32_kernel_tiling/csrc/lut_rag_p2.cuh (+ p1: lut_rag_p1.cuh, orig: "
                          "26_magicvq/csrc/vqattn_lut_rag.cu)",
               commit=subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out, "apps_after", apps_after, flush=True)


if __name__ == "__main__":
    main()
