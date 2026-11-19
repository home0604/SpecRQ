"""E10 phase 2, axis 4 (LUT smem layout / swizzle): Monte-Carlo model of shared-memory bank conflicts of the random
LUT / V-codebook gathers, compared with the phase-1 ncu counts (results/E10/lut_tiling_p1_ncu.md, KST1 n131072 bh256).

Model (CUDA guide, cc >= 5): 8-byte accesses are served per half-warp, 16-byte per quarter-warp; within a group the
number of wavefronts is the max over banks (bank pairs / quads) of the number of DISTINCT words hitting it (same word
= broadcast). Codes are uniform random (as in the benchmark). A layout is a bijection code -> slot; with uniform codes
every bijection that spreads codes evenly over banks gives the same balls-in-bins distribution, so no swizzle can
reduce the expected conflicts; the script checks that numerically and checks two alternative layouts.
Usage: python bank_sim_p2.py --out results/E10/lut_tiling_p2_banksim.json"""
from __future__ import annotations

import argparse
import json

import numpy as np


def wavefronts(rng, n, groups, lanes, bins, perm=None, skew=None):
    if skew is None:
        codes = rng.integers(0, 256, (n, groups, lanes))
    else:
        codes = rng.choice(256, size=(n, groups, lanes), p=skew)
    if perm is not None:
        codes = perm[codes]
    tot = np.zeros(n)
    for g in range(groups):
        c = codes[:, g, :]
        b = c % bins
        mx = np.zeros(n)
        for k in range(bins):
            s = np.sort(np.where(b == k, c, -1), axis=1)
            d = ((s[:, 1:] != s[:, :-1]) & (s[:, 1:] >= 0)).sum(1) + (s[:, 0] >= 0)
            mx = np.maximum(mx, d)
        tot += mx
    return float(tot.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    N = a.samples
    r = {}
    r["lut_8B_current"] = wavefronts(rng, N, 2, 16, 16)
    r["lut_8B_random_bijective_swizzle"] = wavefronts(rng, N, 2, 16, 16, perm=rng.permutation(256))
    r["lut_8B_xor_swizzle_j5"] = wavefronts(rng, N, 2, 16, 16, perm=np.arange(256) ^ 5)
    r["lut_split_2x4B_tables"] = 2 * wavefronts(rng, N, 1, 32, 32)
    r["vcb_16B_current"] = wavefronts(rng, N, 4, 8, 8)
    zipf = 1.0 / np.arange(1, 257) ** 0.8
    zipf /= zipf.sum()
    r["lut_8B_zipf0.8_codes_current"] = wavefronts(rng, N, 2, 16, 16, skew=zipf)
    r["lut_8B_zipf0.8_codes_random_swizzle"] = wavefronts(rng, N, 2, 16, 16, perm=rng.permutation(256), skew=zipf)
    T = 256 * 131072  # tokens of KST1 n131072 bh256
    req = T / 2       # warp requests per stage: LUT (16 loads x 32 tokens / 32 lanes per tile... = T/2), VCB (T/2)
    other = 2 * req   # sPw float4 + V-code byte reads, ~1 wavefront each
    pred = req * r["lut_8B_current"] + req * r["vcb_16B_current"] + other
    conf = req * (r["lut_8B_current"] - 2) + req * (r["vcb_16B_current"] - 4)
    r["pred_k1_n131072_bh256"] = dict(smem_ld_wavefronts_M=pred / 1e6, bank_conflicts_M=conf / 1e6,
                                      ncu_wavefronts_M=319.5, ncu_conflicts_M=167.9)
    r["note"] = ("ideal: LUT 2 wavefronts / request (8B x 32 lanes), VCB 4 (16B x 32 lanes). Conflicts per token are "
                 "independent of n and B*Hkv, so the layout cannot create a condition-dependent optimum.")
    print(json.dumps(r, indent=1))
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
