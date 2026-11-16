"""E5 scale72: per-epoch AQ time for 72B linear shapes on 1 vs N GPUs (DbD AQEngine, multi-GPU = output-group sharding).
Synthetic weight/XTX (random), fixed epoch count (no early stop), so only the cost per epoch is meaningful.
Usage: python experiments/22_scale72/aqbench72.py --shape 8192x8192 --devices cuda:0 [cuda:1] --epochs 2
"""
import argparse
import os
import sys
import time
from argparse import Namespace

DBD = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import aq_engine as AE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="8192x8192", help="out x in")
    ap.add_argument("--devices", nargs="+", default=["cuda:0"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ntok", type=int, default=16384)
    a = ap.parse_args()
    out_f, in_f = map(int, a.shape.split("x"))
    devs = [torch.device(d) for d in a.devices]
    torch.manual_seed(0)
    AE.DEFAULT_ACCUMULATOR_DTYPE = torch.float32
    lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16, device=devs[0])
    nn.init.normal_(lin.weight, std=0.02)
    eng = AE.AQEngine(lin)
    for _ in range(4):
        eng.add_batch(torch.randn(a.ntok // 4, in_f, device=devs[0], dtype=torch.bfloat16))
    eng.hadamard_seed = 1234
    args = Namespace(codebook_weights=[0.5, 0.5], devices=devs, out_group_size=1, in_group_size=8, num_codebooks=2,
                     nbits_per_codebook=8, codebook_value_nbits=16, codebook_value_num_groups=1, scale_nbits=0,
                     init_max_iter=100, init_max_points_per_centroid=None, lr=1e-4, max_epochs=a.epochs,
                     steps_per_epoch=100, relative_mse_tolerance=None, beam_size=1, print_frequency=50, codebook_bits=None)
    torch.cuda.synchronize()
    t0 = time.time()
    eng.quantize(args=args, verbose=True)
    for d in devs:
        torch.cuda.synchronize(d)
    dt = time.time() - t0
    mem = [torch.cuda.max_memory_allocated(d) / 2 ** 30 for d in devs]
    print(f"RESULT shape={a.shape} ndev={len(devs)} epochs={a.epochs} total_s={dt:.1f} peak_gib={[round(m, 2) for m in mem]}")


if __name__ == "__main__":
    main()
