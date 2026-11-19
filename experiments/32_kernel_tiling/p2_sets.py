"""Named variant sets for E10 phase 2. `python p2_sets.py <set>` prints the variant names (space-separated).
Variant = (kst, nw, ls, dec, pipe, fuse); see csrc/lut_rag_p2.cuh. (k, 8, 0, 0, 0, 0) is the phase-1 global struct
w8_b1_r1_p1_l0_s2 re-implemented in the phase-2 file."""
import sys

import p2_ext as PX


def screen():
    """full factorial over the new axes for NW 8 / 16, plus NW 4 (the only NW where KST2 fits a 2/3-stage ring)."""
    v = []
    for k in (1, 2):
        v += [(k, w, s, d, p, f) for w in (8, 16) for s in (0, 1) for d in (0, 1) for p in (0, 2, 3) for f in (0, 1)]
        v += [(k, 4, s, d, p, f) for s in (0, 1) for d in (0, 1) for p in (0, 2, 3) for f in (0, 1)]
    return sorted(set(v))


SETS = {"screen": screen}

if __name__ == "__main__":
    print(" ".join(PX.vname(v) for v in SETS[sys.argv[1]]()))
