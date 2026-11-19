"""Launch the LUT kernel a few times at one (kst, n, bh, chunk) point for Nsight Compute.
Usage: ncu -k regex:lut_rag_kernel -s 2 -c 1 --section ... python ncu_point.py kst n bh chunk [hkv=2]"""
import sys

import torch

import bench_p0 as BP

kst, n, bh, chunk = (int(x) for x in sys.argv[1:5])
hkv = int(sys.argv[5]) if len(sys.argv) > 5 else 2
B, G, D = bh // hkv, 4, 128
dev = "cuda"
torch.manual_seed(0)
ext = BP.build_ext()
inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
qkv = (torch.randn(B, (hkv * G + 2 * hkv) * D, device=dev) * 0.5).to(torch.bfloat16)
qpos = torch.full((B,), 200000, device=dev, dtype=torch.int32)
mu = torch.randn(2, hkv, D, device=dev) * 0.3
sd = torch.rand(2, hkv, D, device=dev) + 0.5
lo = torch.zeros(bh, device=dev, dtype=torch.int32)
hi = torch.full((bh,), n, device=dev, dtype=torch.int32)
p = torch.randint(0, 256, (2, kst, B, hkv, n, 16), dtype=torch.uint8, device=dev)
cb = (torch.randn(2, hkv, kst, 256, 8, device=dev) * 0.5).to(torch.bfloat16)
splits = -(-n // chunk)
spt = 16 * splits
ws = BP.EK.workspace(B, hkv, G, 1, spt)
for _ in range(4):
    ext.lut_rag(qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)], [cb[0, :, s] for s in range(kst)],
                [cb[1, :, s] for s in range(kst)], inv, ws[0], ws[1], ws[2], lo, hi, qpos, sd[0], mu[0], sd[1], mu[1],
                chunk, splits, B, hkv, ws[3], spt, 0, kst, BP.D_SCALE)
torch.cuda.synchronize()
print("done", kst, n, bh, chunk, splits)
