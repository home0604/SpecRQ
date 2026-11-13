"""CPU micro-benchmarks for the offload regime (residual plane on CPU).

Run pinned to one NUMA node (first-touch allocation => local memory), e.g.
  OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores nice -n 10 taskset -c 16-31,48-63 \
      python experiments/18_offload/bench_cpu.py --part all --out /data/lsh/temp/tmp/offload/cpu_t16.json

Parts
  stream : read bandwidth (AVX-512 sum over a 4 GiB fp32 array, first-touched by the same threads)
  check  : correctness of the LUT kernels vs numpy on a small shape
  sweep  : one decode step of a residual plane (d8 K256, 1 plane) over all 64 layers x 7 linears of Qwen2.5-32B,
           per M; fp32-gather kernel (exact) and int8-vpermi2b kernel (approximate LUT)
  torch  : dequant -> bf16 matmul via torch (oneDNN/AMX) for reference, a few linears
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBDIR = "/data/lsh/temp/tmp/offload"
LIB = f"{LIBDIR}/libcpuk.so"

# Qwen2.5-32B linear shapes (in, out)
QWEN32B = {"q": (5120, 5120), "k": (5120, 1024), "v": (5120, 1024), "o": (5120, 5120),
           "gate": (5120, 27648), "up": (5120, 27648), "down": (27648, 5120)}
N_LAYERS = 64


def build():
    os.makedirs(LIBDIR, exist_ok=True)
    src = f"{HERE}/cpu_kernels.cpp"
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(src):
        subprocess.run(["g++-11", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", src, "-o", LIB], check=True)
    lib = C.CDLL(LIB)
    P, I, L, D = C.c_void_p, C.c_int, C.c_int64, C.c_double
    lib.first_touch.argtypes = [P, L]
    lib.stream_read.argtypes = [P, L]
    lib.stream_read.restype = D
    lib.lut_build_f32.argtypes = [P, P, P, I, I]
    lib.gemv_gather_f32.argtypes = [P, P, P, P, I, I, I, I]
    lib.lut_build_i8.argtypes = [P, P, P, P, I, I]
    lib.gemv_perm_i8.argtypes = [P, P, P, P, P, I, I, I]
    lib.gemv_perm_i8s.argtypes = [P, P, P, P, P, I, I, I]
    lib.gemv_perm_i8t.argtypes = [P, P, P, P, P, I, I, I, I]
    lib.sweep.argtypes = [I, I, P, P, P, P, P, P, P, P, P, P, I, I, P, P]
    return lib


def aligned(n_bytes, dtype, touch_lib=None):
    raw = np.empty(n_bytes + 4096, dtype=np.uint8)
    off = (-raw.ctypes.data) % 4096
    a = raw[off:off + n_bytes]
    if touch_lib is not None:
        touch_lib.first_touch(a.ctypes.data, n_bytes)
    return a.view(dtype)


def ptr(a):
    return a.ctypes.data


def stats(v):
    v = sorted(v)
    return {"median": float(np.median(v)), "min": float(v[0]), "max": float(v[-1]), "n": len(v)}


def part_stream(lib, reps, gib=4):
    n = gib * 2 ** 30 // 4
    a = aligned(n * 4, np.float32, lib)
    a[:] = 1.0  # serial write would be slow; first_touch already placed pages, fill is fine
    lib.stream_read(ptr(a), n)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        lib.stream_read(ptr(a), n)
        ts.append(time.perf_counter() - t)
    s = stats(ts)
    return {"bytes": n * 4, "s": s, "GBps": {k: (n * 4 / s[k] / 1e9 if k in ("median",) else None) for k in ("median",)},
            "GBps_best": n * 4 / s["min"] / 1e9, "GBps_worst": n * 4 / s["max"] / 1e9}


def make_cb(rng):
    cb = rng.standard_normal((256, 8)).astype(np.float32) * 0.02
    return cb, np.ascontiguousarray(cb.T)


def part_check(lib, rng):
    out = {}
    for (inn, o, M) in [(512, 128, 1), (512, 256, 3), (1024, 512, 5), (2048, 64, 8)]:
        G = inn // 8
        cb, cbT = make_cb(rng)
        codes_og = rng.integers(0, 256, (o, G), dtype=np.uint8)
        scale = rng.uniform(0.5, 1.5, o).astype(np.float32)
        x = rng.standard_normal((M, inn)).astype(np.float32)
        W = (cb[codes_og].reshape(o, inn)) * scale[:, None]
        ref = x @ W.T
        blk = aligned(o * G, np.uint8)
        blk[:] = codes_og.reshape(o // 64, 64, G).transpose(0, 2, 1).reshape(-1)
        lut = aligned(M * G * 256 * 4, np.float32)
        y = np.zeros((M, o), np.float32)
        lib.lut_build_f32(ptr(x), ptr(cbT), ptr(lut), M, inn)
        lib.gemv_gather_f32(ptr(blk), ptr(lut), ptr(scale), ptr(y), M, inn, o, 32)
        e_f = float(np.abs(y - ref).max() / np.abs(ref).max())
        lutq = aligned(M * G * 256, np.int8)
        ls = np.zeros((M, G // 32), np.float32)
        y2 = np.zeros((M, o), np.float32)
        lib.lut_build_i8(ptr(x), ptr(cbT), ptr(lutq), ptr(ls), M, inn)
        lib.gemv_perm_i8(ptr(blk), ptr(lutq), ptr(ls), ptr(scale), ptr(y2), M, inn, o)
        y3 = np.zeros((M, o), np.float32)
        lib.gemv_perm_i8s(ptr(blk), ptr(lutq), ptr(ls), ptr(scale), ptr(y3), M, inn, o)
        same_s = bool(np.allclose(y2, y3, rtol=1e-5, atol=1e-6))
        for obt in (1, 2, 4):
            if (o // 64) % obt == 0:
                y4 = np.zeros((M, o), np.float32)
                lib.gemv_perm_i8t(ptr(blk), ptr(lutq), ptr(ls), ptr(scale), ptr(y4), M, inn, o, obt)
                same_s = same_s and bool(np.allclose(y2, y4, rtol=1e-5, atol=1e-6))
        e_q = float(np.abs(y2 - ref).max() / np.abs(ref).max())
        rel_q = float(np.linalg.norm(y2 - ref) / np.linalg.norm(ref))
        out[f"in{inn}_out{o}_M{M}"] = {"f32_maxerr_rel": e_f, "i8_maxerr_rel": e_q, "i8_rel_l2": rel_q,
                                        "i8_shift_and_tile_equal_i8": same_s}
    return out


def part_sweep(lib, rng, Ms, reps, gtile, kinds, obt=0):
    names = list(QWEN32B)
    # 64 layers x 7 linears of codes, each first-touched on this node
    codes, ins, outs, lin_names = [], [], [], []
    for _ in range(N_LAYERS):
        for nm in names:
            inn, o = QWEN32B[nm]
            a = aligned(o * inn // 8, np.uint8, lib)
            a[:] = rng.integers(0, 256, a.shape[0], dtype=np.uint8) if len(codes) < 7 else codes[len(codes) % 7][:]
            codes.append(a)
            ins.append(inn)
            outs.append(o)
            lin_names.append(nm)
    n = len(codes)
    total_code_bytes = sum(c.nbytes for c in codes)
    Mmax = max(Ms)
    max_in = max(ins)
    max_out = max(outs)
    _, cbT = make_cb(rng)
    x = rng.standard_normal((Mmax, max_in)).astype(np.float32)
    lut = aligned(Mmax * (max_in // 8) * 256 * 4, np.float32, lib)
    lutq = aligned(Mmax * (max_in // 8) * 256, np.int8, lib)
    ls = aligned(Mmax * (max_in // 256) * 4, np.float32, lib)
    scale = np.ones(max_out, np.float32)
    y = aligned(Mmax * max_out * 4, np.float32, lib)
    cp = (C.c_void_p * n)(*[ptr(c) for c in codes])
    ia = (C.c_int * n)(*ins)
    oa = (C.c_int * n)(*outs)
    tl = np.zeros(n, np.float64)
    tg = np.zeros(n, np.float64)
    res = {"total_code_bytes": total_code_bytes, "n_linears": n, "gtile": gtile, "obt": obt, "runs": []}
    for kind, kname in [(0, "gather_f32"), (1, "perm_i8"), (2, "perm_i8_shift"), (3, "perm_i8_tile")]:
        if kind not in kinds:
            continue
        for M in Ms:
            # x must be [M, in] contiguous per call: use the first M rows with row stride = in -> pass a view
            # sweep reads x with stride ins[i]; all linears of one input size share a buffer -> run two sub-sweeps
            # would complicate; instead use one buffer of the largest in and stride per call (row stride = in).
            xb = np.ascontiguousarray(x[:M, :max_in]).reshape(-1)  # values irrelevant for timing
            walls, per_lut, per_gemv = [], [], []
            for r in range(reps + 1):
                t = time.perf_counter()
                lib.sweep(kind, n, cp, ia, oa, ptr(xb), ptr(cbT), ptr(lut), ptr(lutq), ptr(ls), ptr(scale), ptr(y),
                          M, gtile if kind == 0 else obt, ptr(tl), ptr(tg))
                w = time.perf_counter() - t
                if r == 0:
                    continue  # warm-up
                walls.append(w)
                per_lut.append(tl.copy())
                per_gemv.append(tg.copy())
            wall = stats(walls)
            by = {}
            for nm in names:
                idx = [i for i in range(n) if lin_names[i] == nm]
                lt = [float(np.sum(p[idx])) for p in per_lut]
                gt = [float(np.sum(p[idx])) for p in per_gemv]
                b = sum(codes[i].nbytes for i in idx)
                by[nm] = {"lut_s_64L": stats(lt), "gemv_s_64L": stats(gt),
                          "per_call_us": float(np.median([a + c for a, c in zip(lt, gt)]) / N_LAYERS * 1e6),
                          "gemv_GBps": b / float(np.median(gt)) / 1e9}
            row = {"kernel": kname, "M": M, "wall_s": wall, "GBps_codes": total_code_bytes / wall["median"] / 1e9,
                   "lut_s": stats([float(p.sum()) for p in per_lut]), "gemv_s": stats([float(p.sum()) for p in per_gemv]),
                   "by_linear": by}
            res["runs"].append(row)
            print(f"[sweep] {kname} M={M}: wall {wall['median']*1e3:.2f} ms (min {wall['min']*1e3:.2f}, max "
                  f"{wall['max']*1e3:.2f}), lut {row['lut_s']['median']*1e3:.2f} ms, gemv {row['gemv_s']['median']*1e3:.2f}"
                  f" ms, {row['GBps_codes']:.1f} GB/s codes", flush=True)
    return res


def part_torch(rng, Ms, reps, nthreads):
    import torch
    torch.set_num_threads(nthreads)
    out = {}
    for nm in ("q", "down"):
        inn, o = QWEN32B[nm]
        cb = torch.randn(256, 8, dtype=torch.bfloat16) * 0.02
        codes = torch.from_numpy(rng.integers(0, 256, (o, inn // 8), dtype=np.uint8)).long()
        Wd = cb[codes].reshape(o, inn).contiguous()
        for M in Ms:
            x = torch.randn(M, inn, dtype=torch.bfloat16)
            def deq_mm():
                W = cb[codes].reshape(o, inn)
                return x @ W.T
            def mm():
                return x @ Wd.T
            r = {}
            for tag, f in (("dequant_mm", deq_mm), ("dense_bf16_mm", mm)):
                f()
                ts = []
                for _ in range(reps):
                    t = time.perf_counter()
                    f()
                    ts.append(time.perf_counter() - t)
                r[tag] = stats(ts)
            r["dense_bf16_GBps"] = Wd.numel() * 2 / r["dense_bf16_mm"]["median"] / 1e9
            out[f"{nm}_M{M}"] = r
            print(f"[torch] {nm} M={M}: dequant+mm {r['dequant_mm']['median']*1e3:.2f} ms, dense bf16 mm "
                  f"{r['dense_bf16_mm']['median']*1e3:.2f} ms ({r['dense_bf16_GBps']:.1f} GB/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", nargs="+", default=["stream", "check", "sweep"])
    ap.add_argument("--Ms", type=int, nargs="+", default=[1, 2, 3, 4, 5, 8, 16])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--obt", type=int, default=0, help="tile width for kind 3 (0 = auto)")
    ap.add_argument("--kinds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gtile", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    res = {"env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
           "affinity": sorted(os.sched_getaffinity(0)), "loadavg_start": os.getloadavg()}
    lib = build() if set(a.part) & {"stream", "check", "sweep"} else None
    if "stream" in a.part:
        res["stream"] = part_stream(lib, a.reps)
        print("[stream]", res["stream"], flush=True)
    if "check" in a.part:
        res["check"] = part_check(lib, rng)
        print("[check]", res["check"], flush=True)
    if "sweep" in a.part:
        res["sweep"] = part_sweep(lib, rng, a.Ms, a.reps, a.gtile, a.kinds, a.obt)
    if "torch" in a.part:
        res["torch"] = part_torch(rng, a.Ms, a.reps, int(os.environ.get("OMP_NUM_THREADS", "16")))
    res["loadavg_end"] = os.getloadavg()
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
