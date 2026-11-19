"""Launch one phase-1 variant (or 'orig') a few times at one (n, bh, chunk) point for Nsight Compute.
Usage: ncu -k regex:'lut_' -s 2 -c 1 --set full -o out python ncu_point_p1.py VARIANT|orig_k{1,2} n bh chunk [hkv=2]"""
import sys

import torch

import bench_p0 as BP
import p1_ext as PX

name = sys.argv[1]
n, bh, chunk = (int(x) for x in sys.argv[2:5])
hkv = int(sys.argv[5]) if len(sys.argv) > 5 else 2
orig = name.startswith("orig")
kst = int(name[-1]) if orig else PX.parse(name)[0]
B, G, D = bh // hkv, 4, 128
dev = "cuda"
torch.manual_seed(0)
if orig:
    ext = BP.build_ext()
    slots = 16
else:
    import p1_sets
    fin = p1_sets.final()
    E = PX.build(fin if PX.parse(name) in fin else [PX.parse(name)])  # reuse the cached finalist build
    slots = E.occ[name]["slots_per_split"]
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
spt = slots * splits
ws = BP.EK.workspace(B, hkv, G, 1, spt)
args = (qkv, [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)], [cb[0, :, s] for s in range(kst)],
        [cb[1, :, s] for s in range(kst)], inv, ws[0], ws[1], ws[2], lo, hi, qpos, sd[0], mu[0], sd[1], mu[1],
        chunk, splits, B, hkv, ws[3], spt, 0)
for _ in range(4):
    if orig:
        ext.lut_rag(*args, kst, BP.D_SCALE)
    else:
        E.run(name, *args, BP.D_SCALE)
torch.cuda.synchronize()
print("done", name, n, bh, chunk, splits)
