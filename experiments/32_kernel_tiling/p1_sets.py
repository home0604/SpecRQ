"""Named variant sets for E10 phase 1. `python p1_sets.py <set>` prints the variant names (space-separated).
Variant = (kst, nw, minb, red, pf, lutm, sb); see csrc/lut_rag_p1.cuh."""
import sys

import p1_ext as PX


def screen():
    v = []
    for k in (1, 2):
        # (a) warps / register target
        v += [(k, w, b, 0, 0, 0, 0) for w in (4, 8, 16) for b in (1, 2, 3)]
        # (b) in-block partial reduction
        v += [(k, w, b, 1, 0, 0, 0) for w in (4, 8, 16) for b in (1, 2)]
        # SB: static bound (unsafe on real data, speed reference) and lazy max (safe)
        v += [(k, w, 1, r, 0, 0, s) for w in (8, 16) for r in (0, 1) for s in (1, 2)]
        # (c) code prefetch
        v += [(k, w, 1, r, 1, 0, 0) for w in (8, 16) for r in (0, 1)]
        # (d) int8 LUT (per-(stage, j, head) scale / per-head packed)
        v += [(k, w, b, 1, 0, l, 0) for w in (4, 8, 16) for b in (1, 2) for l in (1, 2)]
        # small combos
        v += [(k, w, b, 1, p, l, 2) for w in (8, 16) for b in (1, 2) for p in (0, 1) for l in (0, 2)]
    return sorted(set(v))


def final():
    """finalists after the screen (results/E10/lut_tiling_p1_screen_tables.md): ablation chain + top structs + refs"""
    st = [(16, 1, 0, 0, 0, 0),  # copy of the original structure (baseline of the copy)
          (8, 1, 0, 0, 0, 0),   # NW only
          (16, 1, 0, 0, 0, 2),  # lazy max only
          (16, 1, 1, 0, 0, 0),  # in-block reduction only
          (16, 1, 1, 0, 0, 2), (16, 1, 1, 1, 0, 2), (8, 1, 1, 0, 0, 2), (8, 1, 1, 1, 0, 2), (8, 2, 1, 0, 0, 2),
          (8, 2, 1, 1, 0, 2),
          (16, 1, 1, 0, 0, 1),  # static bound (unsafe on real data) reference
          (8, 1, 1, 1, 2, 2)]   # best int8-LUT struct
    return sorted((k,) + x for k in (1, 2) for x in st)


SETS = {"screen": screen, "final": final}

if __name__ == "__main__":
    print(" ".join(PX.vname(v) for v in SETS[sys.argv[1]]()))
