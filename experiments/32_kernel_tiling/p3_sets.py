"""Named variant sets for E10 phase 3. `python p3_sets.py <set>` prints variant names (space-separated).
Variant = (kst, nw, vm, vrep, hs, ab); see csrc/lut_rag_p3.cuh. Variants whose smem does not fit the device (e.g.
KST2 VREP 4/8 on the 4090, 99 KB/block) are skipped at run time (blocks/SM = 0)."""
import sys

import p3_ext as PX


def exact():
    st = [(8, 0, 1, 0), (8, 0, 1, 1), (16, 0, 1, 0),                      # FFMA (= phase-1 struct), + half2 scores
          (8, 1, 1, 0), (8, 1, 2, 0), (8, 1, 4, 0), (8, 1, 8, 0),          # tensor-core P.V, V codebook replicas
          (8, 1, 1, 1), (8, 1, 2, 1), (8, 1, 8, 1), (4, 1, 1, 0), (16, 1, 1, 0), (16, 1, 2, 1),
          (8, 2, 1, 0), (8, 2, 8, 0), (8, 2, 1, 1)]                        # packed HFMA2
    return sorted((k,) + x + (0,) for k in (1, 2) for x in st)


def ablate():
    """timing-only ablations (outputs not meaningful): ab 1 = no V accumulation, 2 = no LUT score gather, 3 = both"""
    st = [(8, 0, 1, 0, a) for a in (1, 2, 3)] + [(16, 0, 1, 0, a) for a in (1, 2, 3)] + \
         [(8, 1, 1, 0, a) for a in (1, 2)] + [(8, 1, 2, 1, 2)]
    return sorted((k,) + x for k in (1, 2) for x in st)


def a100():
    return sorted(set(exact() + ablate()))


SETS = {"exact": exact, "ablate": ablate, "a100": a100, "all": a100}

if __name__ == "__main__":
    print(" ".join(PX.vname(v) for v in SETS[sys.argv[1]]()))
