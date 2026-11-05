"""Analytic DRAM bytes when a nested-VQ draft reads only the first k of L codebooks.

Layouts
  interleaved : AQLM storage, codes[N, K/d, L] in a container dtype (int8 for b<=8, int16 for b<=16); the L codes
                of one group are contiguous, so one 32 B DRAM sector holds codes of every codebook.
  planar      : codebook-major, codes[L][N, K/d] with b-bit packing; prefix k touches only planes 0..k-1.
Fetched bytes are counted at 32 B sector granularity (a sector is fetched if any needed byte lies in it).
Codebook bytes (L * 2^b * d * fp16) and scales are added per linear layer.

Usage: python -m specrq.e0.layout_bytes [--save RUN]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

SECTOR = 32
D = 8
LLAMA8B = {"q": (4096, 4096), "k": (1024, 4096), "v": (1024, 4096), "o": (4096, 4096),
           "gate": (14336, 4096), "up": (14336, 4096), "down": (4096, 14336)}
CFGS = {"C1": (5, 8), "C2": (2, 12)}


def container_bytes(b):
    return 1 if b <= 8 else 2 if b <= 16 else 4


def interleaved_fetch_per_row(groups, L, b, k):
    """Sectors touched in one output row of codes when reading codebooks 0..k-1."""
    cb = container_bytes(b)
    rec = L * cb
    sectors = set()
    for g in range(groups):
        base = g * rec
        for j in range(k):
            for byte in range(base + j * cb, base + (j + 1) * cb):
                sectors.add(byte // SECTOR)
    return len(sectors) * SECTOR


def layer_bytes(N, K, L, b, k):
    groups = K // D
    il = N * interleaved_fetch_per_row(groups, L, b, k)
    il_full = N * groups * L * container_bytes(b)
    planar = k * math.ceil(N * groups * b / 8)          # tightly packed b-bit planes
    planar_container = k * N * groups * container_bytes(b)
    cbk = k * (2 ** b) * D * 2
    return {"interleaved": il + cbk, "interleaved_full_all_L": il_full + L * (2 ** b) * D * 2,
            "planar_packed": planar + cbk, "planar_container": planar_container + cbk,
            "useful_bits": k * N * groups * b / 8, "codebook_bytes": cbk}


def table():
    out = []
    for cfg, (L, b) in CFGS.items():
        for k in range(1, L + 1):
            tot = {}
            for name, (N, K) in LLAMA8B.items():
                for key, v in layer_bytes(N, K, L, b, k).items():
                    tot[key] = tot.get(key, 0) + v
            params = sum(N * K for N, K in LLAMA8B.values())
            out.append({"cfg": cfg, "L": L, "b": b, "k": k, "bpw_prefix": k * b / D,
                        "MB_per_layer_interleaved": tot["interleaved"] / 1e6,
                        "MB_per_layer_planar_packed": tot["planar_packed"] / 1e6,
                        "MB_per_layer_planar_container": tot["planar_container"] / 1e6,
                        "waste_ratio_interleaved_vs_planar_packed": tot["interleaved"] / tot["planar_packed"],
                        "wasted_fraction_interleaved": 1 - tot["useful_bits"] / tot["interleaved"],
                        "fp16_MB_per_layer": 2 * params / 1e6, "int4g128_MB_per_layer": (params / 2 + params / 64) / 1e6})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    rows = table()
    print(f"{'cfg':3} {'k':>2} {'bpw':>5} {'IL MB':>8} {'planar MB':>9} {'IL/planar':>9} {'IL waste':>8}")
    for r in rows:
        print(f"{r['cfg']:3} {r['k']:2d} {r['bpw_prefix']:5.2f} {r['MB_per_layer_interleaved']:8.2f} "
              f"{r['MB_per_layer_planar_packed']:9.2f} {r['waste_ratio_interleaved_vs_planar_packed']:9.2f} "
              f"{r['wasted_fraction_interleaved']:8.1%}")
    print(f"fp16 {rows[0]['fp16_MB_per_layer']:.1f} MB/layer, int4 g128 {rows[0]['int4g128_MB_per_layer']:.1f} MB/layer")
    if a.save:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"setting": "analytic prefix-k DRAM bytes, Llama-3.1-8B layer",
                                                          "sector_bytes": SECTOR, "d": D, "cfgs": CFGS},
                                   metrics={"rows": rows}))


if __name__ == "__main__":
    main()
