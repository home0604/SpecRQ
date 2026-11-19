"""E10 phase 1: structural-variant sweep of the LUT VQ-KV draft-attention kernel (csrc/lut_rag_p1.cuh) next to the
unmodified original (phase-0 extension, variant name 'orig'), same harness / timing as bench_p0 (CUDA graphs over R
code copies >= 3x L2, median of reps; t_tot = partial + Triton combine over slots_per_split * splits partial slots).

Per (variant, kst, n, bh) the measured chunks are: the phase-0 chunk set (<= n, plus one chunk >= n) + split_plan
candidates + the wave-aware rule pick recomputed with the variant's blocks/SM and partial slots per split:
   s* = argmin_s ceil(bh*s / (SMs*bps)) * (bps*ceil(n/s) + F) + CS * s * slots/16      (F=1500, CS=200 from p0)
Usage: python bench_p1.py --out raw.json --variants k1_w16_b1_r0_p0_l0_s0 ... [--orig] [--kst ..] [--n ..] [--bh ..]
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
import p1_ext as PX
from specrq.kvq.kernels import split_plan

D, G, C = 128, 4, 256
F_TOK, CS_TOK = 1500, 200


def rule_splits(n, bh, sms, bps, slots):
    best = None
    for s in range(1, max(1, -(-n // 64)) + 1):
        cost = math.ceil(bh * s / (sms * bps)) * (bps * math.ceil(n / s) + F_TOK) + CS_TOK * s * slots / 16
        if best is None or cost < best[0]:
            best = (cost, s)
    return best[1]


def ceil64(x):
    return -(-x // 64) * 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--orig", action="store_true", help="also time the unmodified original kernel")
    ap.add_argument("--kst", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072])
    ap.add_argument("--bh", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--splits", type=int, nargs="*", default=[], help="extra splits candidates (waveq probe)")
    ap.add_argument("--no_fixed", action="store_true")
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
    E = PX.build([PX.parse(v) for v in a.variants]) if a.variants else None
    ext0 = BP.build_ext() if a.orig else None
    t_build = time.time() - t_build
    lim = (E.device_limits() if E else dict(zip(
        ["sm_count", "smem_per_sm", "smem_per_block_optin", "reserved_smem_per_block", "regs_per_sm", "threads_per_sm",
         "blocks_per_sm", "l2_bytes", "cc_major", "cc_minor"], ext0.device_limits())))
    sms, l2 = lim["sm_count"], lim["l2_bytes"]
    V = {}
    if ext0 is not None:
        for k in a.kst:
            o = dict(zip(["blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads_per_block",
                          "threads", "vcb_bytes", "lut_bytes", "warp_off", "tail_bytes", "warp_bytes"],
                         ext0.occupancy(k)))
            o["slots_per_split"] = 16
            V[f"orig_k{k}"] = dict(kst=k, nw=16, minb=0, red=0, pf=0, lutm=0, sb=0, occ=o, orig=True)
    if E is not None:
        for nm in E.names:
            V[nm] = dict(**E.fields[nm], occ=E.occ[nm], orig=False)
    for nm, v in V.items():
        print(nm, v["occ"], flush=True)
    bad = [nm for nm, v in V.items() if v["occ"]["blocks_per_sm"] < 1]
    for nm in bad:
        print("NOT LAUNCHABLE (smem):", nm, flush=True)
        V.pop(nm)

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
                    cands = set(base) | {rule_chunk}
                    seen = set()
                    for chunk in sorted(cands):
                        splits = -(-n // chunk)
                        if splits in seen:
                            continue
                        seen.add(splits)
                        spt = slots * splits
                        ws_bytes = bh * spt * 16 * (D + 2) * 4
                        if ws_bytes > a.ws_cap_gb * 2 ** 30:
                            continue
                        ws = BP.EK.workspace(B, Hkv, G, 1, spt)
                        out = torch.empty(B, Hq * D, dtype=torch.bfloat16, device=dev)

                        def call(p, h_, nm=nm, chunk=chunk, splits=splits, spt=spt, ws=ws):
                            args = (qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)],
                                    [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv,
                                    ws[0], ws[1], ws[2], lo, h_, qpos, sd[0], mu[0], sd[1], mu[1], chunk, splits,
                                    B, Hkv, ws[3], spt, 0)
                            if V[nm]["orig"]:
                                ext0.lut_rag(*args, kst, BP.D_SCALE)
                            else:
                                E.run(nm, *args, BP.D_SCALE)

                        def part(h_=hi, call=call):
                            for p in planes:
                                call(p, h_)

                        def tot(call=call, ws=ws, spt=spt, out=out):
                            for p in planes:
                                call(p, hi)
                                BP.EK.combine(ws, spt, out, B, Hkv, G, 1)

                        res = {}
                        fns = [("tot", tot), ("part", part)] + ([] if a.no_fixed else [("fixed", lambda: part(lo))])
                        for fnm, f in fns:
                            md, mn, mx, m = BP.gtime(f, a.reps)
                            res[fnm] = (md / R * 1e3, mn / R * 1e3, mx / R * 1e3)
                        nblk = bh * splits
                        bps = occ["blocks_per_sm"]
                        row = dict(variant=nm, kst=kst, nw=v["nw"], minb=v["minb"], red=v["red"], pf=v["pf"],
                                   lutm=v["lutm"], sb=v["sb"], orig=v["orig"], bps=bps, regs=occ["regs"],
                                   n=n, bh=bh, B=B, Hkv=Hkv, chunk=chunk, splits=splits, grid=nblk, copies=R,
                                   slots=slots, waves=math.ceil(nblk / (sms * bps)), rule_splits=rs,
                                   is_rule=chunk == rule_chunk,
                                   t_tot_us=res["tot"][0], t_tot_min=res["tot"][1], t_tot_max=res["tot"][2],
                                   t_part_us=res["part"][0], t_part_min=res["part"][1], t_part_max=res["part"][2],
                                   t_fixed_us=res["fixed"][0] if "fixed" in res else None,
                                   bytes_codes=bytes_codes,
                                   code_GBps=bytes_codes / (res["part"][0] * 1e-6) / 1e9)
                        rows.append(row)
                        print(f"{nm:24s} n{n:6d} bh{bh:3d} chunk{chunk:6d} sp{splits:4d} | tot {row['t_tot_us']:8.1f} "
                              f"part {row['t_part_us']:8.1f} fixed {row['t_fixed_us'] or 0:7.1f} | "
                              f"{row['code_GBps']:6.1f} GB/s{' R' if row['is_rule'] else ''}", flush=True)
                        del ws, out
                del planes
                torch.cuda.empty_cache()
    apps_after = BP.nvsmi_apps()
    rec = dict(device=torch.cuda.get_device_name(), gpu_uuid=uuid, limits=lim, variants=V, bw_ref_GBps=bw_ref,
               torch=torch.__version__, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               arch=os.environ.get("TORCH_CUDA_ARCH_LIST"), build_s=t_build, sweep_s=time.time() - t_start,
               args=vars(a), apps_before=apps_before, apps_after=apps_after,
               kernel_src="experiments/32_kernel_tiling/csrc/lut_rag_p1.cuh (+ orig: 26_magicvq/csrc/vqattn_lut_rag.cu)",
               rule=dict(F_TOK=F_TOK, CS_TOK=CS_TOK),
               commit=subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
               rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out, "apps_after", apps_after, flush=True)


if __name__ == "__main__":
    main()
