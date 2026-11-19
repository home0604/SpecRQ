"""E10 phase 0: runtime-knob (chunk / splits) sweep of the unmodified E7 VQ-KV LUT draft-attention kernel
(experiments/26_magicvq/csrc/vqattn_lut_rag.cu) over KST x n x (B*Hkv), standalone, CUDA-graph timing.

Per config (kst, n, bh, chunk), splits = ceil(n / chunk), grid = (bh, splits), 512 threads/block:
  t_part  : lut_rag partial kernel alone
  t_comb  : log-sum-exp combine (Triton, e2e_kern._combine) alone over the 16 * splits partial slots
  t_tot   : partial + combine (one decode attention call of the VQ part; the real path adds the bf16 sinks/window part)
  t_fixed : partial kernel with an empty token range (HI = LO): q RoPE + LUT build + partial store only, same grid
Each graph cycles over R independent code-plane copies (R * bytes >= ~3x L2) so the codes are not L2-resident across
calls, as across the 32 layers of the real decode. Times = median of `reps` reps (us per call), min/max kept.

Portable: arch / SM count / L2 from the device; extension built for the current device's compute capability.
Usage: python bench_p0.py --out raw.json [--kst 1 2] [--n ...] [--bh ...] [--chunks ...] [--hkv 2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MAGIC = HERE.parent / "26_magicvq"
sys.path.insert(0, str(MAGIC))
sys.path.insert(0, str(HERE.parent.parent))
import e2e_kern as EK  # noqa: E402
from specrq.kvq.kernels import split_plan  # noqa: E402

D, G, NSV, C = 128, 4, 16, 256
NW = 16
D_SCALE = 128 ** -0.5 * 1.4426950408889634


def build_ext():
    from torch.utils.cpp_extension import load
    cap = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    for k, v in (("CC", "/usr/bin/gcc-11"), ("CXX", "/usr/bin/g++-11"), ("NVCC_PREPEND_FLAGS", "-ccbin /usr/bin/g++-11"),
                 ("CUDA_HOME", "/usr/local/cuda-12.1")):
        if Path(v.split()[-1]).exists():
            os.environ.setdefault(k, v)
    arch = os.environ["TORCH_CUDA_ARCH_LIST"].replace(".", "")
    return load(name=f"e10_lut_rag_occ_sm{arch}", sources=[str(HERE / "csrc" / "lut_rag_occ.cu")],
                extra_include_paths=[str(MAGIC / "csrc")],
                extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++17", "-Xptxas=-v"], verbose=False)


def nvsmi_apps():
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return [f"error {e}"]


def my_gpu_uuid():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    return out


def gtime(fn, reps, target_ms=3.0):
    """CUDA-graph timing: capture fn once (fn may launch several kernels), replay m times per rep; returns ms per
    replay (median, min, max over reps)."""
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
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
    one = max(e0.elapsed_time(e1), 1e-3)
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
    ap.add_argument("--kst", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n", type=int, nargs="+", default=[4096, 16384, 32768, 65536, 131072])
    ap.add_argument("--bh", type=int, nargs="+", default=[8, 32, 64, 128, 256])
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--splits", type=int, nargs="*", default=[],
                    help="extra candidates chunk = ceil(n / s) rounded up to 64 (wave-quantization probe)")
    ap.add_argument("--hkv", type=int, default=2)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--ws_cap_gb", type=float, default=3.0)
    ap.add_argument("--l2_mult", type=float, default=3.0)
    ap.add_argument("--max_copies", type=int, default=256)
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    apps_before = nvsmi_apps()
    uuid = my_gpu_uuid()
    t_build = time.time()
    ext = build_ext()
    t_build = time.time() - t_build
    lim = ext.device_limits()
    limits = dict(zip(["sm_count", "smem_per_sm", "smem_per_block_optin", "reserved_smem_per_block", "regs_per_sm",
                       "threads_per_sm", "blocks_per_sm", "l2_bytes", "cc_major", "cc_minor"], lim))
    occ = {}
    for k in a.kst:
        o = ext.occupancy(k)
        occ[k] = dict(zip(["blocks_per_sm", "dyn_smem", "regs", "local_bytes", "static_smem", "max_threads_per_block",
                           "threads", "vcb_bytes", "lut_bytes", "warp_off", "tail_bytes", "warp_bytes"], o))
    print("device", torch.cuda.get_device_name(), limits, flush=True)
    print("occupancy", occ, flush=True)
    sms = limits["sm_count"]
    l2 = limits["l2_bytes"]

    # HBM read reference: large fp32 sum (read-only stream)
    x = torch.empty(512 * 2 ** 20 // 4, dtype=torch.float32, device=dev).uniform_()
    r = torch.empty((), device=dev)
    tm, _, _, _ = gtime(lambda: torch.sum(x, dim=0, out=r), 7)
    bw_ref = x.numel() * 4 / (tm * 1e-3) / 1e9
    del x
    print(f"reference read BW (torch.sum 512 MiB fp32): {bw_ref:.1f} GB/s", flush=True)

    inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
    rows = []
    for bh in a.bh:
        assert bh % a.hkv == 0
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
                bytes_codes = 2 * kst * bh * n * 16
                R = max(1, min(a.max_copies, math.ceil(a.l2_mult * l2 / bytes_codes)))
                try:
                    planes = [torch.randint(0, 256, (2, kst, B, Hkv, n, 16), dtype=torch.uint8, device=dev)
                              for _ in range(R)]
                except torch.cuda.OutOfMemoryError:
                    print(f"skip OOM planes bh{bh} n{n} kst{kst}", flush=True)
                    continue
                cb = (torch.randn(2, Hkv, kst, C, 8, device=dev) *
                      torch.tensor([1.0, 0.4], device=dev)[None, None, :kst, None, None]).to(torch.bfloat16)
                cands = set(a.chunks)
                sp_c = {}
                for tp in (128, 256, 512, 1024, 2048):
                    s_, c_ = split_plan(B, Hkv, n, 64, tp)
                    cands.add(c_)
                    sp_c.setdefault(c_, []).append(tp)
                for s_ in a.splits:
                    cands.add(-(-(-(-n // s_)) // 64) * 64)
                for chunk in sorted(cands):
                    splits = -(-n // chunk)
                    if chunk > n and chunk != min(c for c in cands if c >= n):
                        continue  # splits = 1 for every chunk >= n: keep only the smallest such chunk
                    spt = 16 * splits
                    ws_bytes = bh * spt * 16 * (D + 2) * 4
                    if ws_bytes > a.ws_cap_gb * 2 ** 30:
                        print(f"skip ws {ws_bytes / 2**30:.1f} GiB bh{bh} n{n} kst{kst} chunk{chunk}", flush=True)
                        continue
                    ws = EK.workspace(B, Hkv, G, 1, spt)
                    out = torch.empty(B, Hq * D, dtype=torch.bfloat16, device=dev)

                    def part(h_=hi):
                        for p in planes:
                            ext.lut_rag(qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)],
                                        [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv,
                                        ws[0], ws[1], ws[2], lo, h_, qpos, sd[0], mu[0], sd[1], mu[1], chunk, splits,
                                        B, Hkv, ws[3], spt, 0, kst, D_SCALE)

                    def comb():
                        for _ in planes:
                            EK.combine(ws, spt, out, B, Hkv, G, 1)

                    def tot():
                        for p in planes:
                            ext.lut_rag(qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)],
                                        [cb[0, :, s] for s in range(kst)], [cb[1, :, s] for s in range(kst)], inv,
                                        ws[0], ws[1], ws[2], lo, hi, qpos, sd[0], mu[0], sd[1], mu[1], chunk, splits,
                                        B, Hkv, ws[3], spt, 0, kst, D_SCALE)
                            EK.combine(ws, spt, out, B, Hkv, G, 1)

                    res = {}
                    for nm, f in (("part", part), ("comb", comb), ("tot", tot), ("fixed", lambda: part(lo))):
                        md, mn, mx, m = gtime(f, a.reps)
                        res[nm] = (md / R * 1e3, mn / R * 1e3, mx / R * 1e3)
                    nblk = bh * splits
                    part_bytes = nblk * NW * G * (D + 2) * 4
                    row = dict(kst=kst, n=n, bh=bh, B=B, Hkv=Hkv, chunk=chunk, splits=splits, grid=nblk, copies=R,
                               waves=math.ceil(nblk / (sms * occ[kst]["blocks_per_sm"])),
                               wave_eff=nblk / (math.ceil(nblk / (sms * occ[kst]["blocks_per_sm"])) * sms *
                                                occ[kst]["blocks_per_sm"]),
                               split_plan_tp=sp_c.get(chunk, []),
                               t_part_us=res["part"][0], t_part_min=res["part"][1], t_part_max=res["part"][2],
                               t_comb_us=res["comb"][0], t_comb_min=res["comb"][1], t_comb_max=res["comb"][2],
                               t_tot_us=res["tot"][0], t_tot_min=res["tot"][1], t_tot_max=res["tot"][2],
                               t_fixed_us=res["fixed"][0], t_fixed_min=res["fixed"][1], t_fixed_max=res["fixed"][2],
                               bytes_codes=bytes_codes, bytes_partials=part_bytes,
                               code_GBps=bytes_codes / (res["part"][0] * 1e-6) / 1e9,
                               eff_GBps=(bytes_codes + 2 * part_bytes) / (res["tot"][0] * 1e-6) / 1e9)
                    rows.append(row)
                    print(f"kst{kst} n{n:6d} bh{bh:3d} chunk{chunk:5d} sp{splits:4d} grid{nblk:6d} R{R:2d} | part "
                          f"{row['t_part_us']:8.1f} comb {row['t_comb_us']:7.1f} tot {row['t_tot_us']:8.1f} fixed "
                          f"{row['t_fixed_us']:7.1f} us | {row['code_GBps']:6.1f} GB/s", flush=True)
                    del ws, out
                del planes
                torch.cuda.empty_cache()
    apps_after = nvsmi_apps()
    rec = dict(device=torch.cuda.get_device_name(), gpu_uuid=uuid, limits=limits, occupancy=occ,
               bw_ref_GBps=bw_ref, torch=torch.__version__, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               arch=os.environ.get("TORCH_CUDA_ARCH_LIST"), build_s=t_build, args=vars(a),
               apps_before=apps_before, apps_after=apps_after,
               kernel_src="experiments/26_magicvq/csrc/vqattn_lut_rag.cu",
               commit=subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
               rows=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out, "apps_after", apps_after, flush=True)


if __name__ == "__main__":
    main()
