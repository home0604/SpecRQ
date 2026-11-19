"""Launch one implementation a few times at one (n, bh, chunk) point for Nsight Compute (E10 phase 3).
NAME = orig_k{1,2} | origlazy_k{1,2} | p1:<p1 variant> | <p3 variant>.
Usage: ncu -k regex:'lut_' -s 2 -c 1 --metrics ... -o out python ncu_point_p3.py NAME n bh chunk [hkv=2]"""
import sys

import torch

import bench_p0 as BP

name = sys.argv[1]
n, bh, chunk = (int(x) for x in sys.argv[2:5])
hkv = int(sys.argv[5]) if len(sys.argv) > 5 else 2
B, G, D = bh // hkv, 4, 128
dev = "cuda"
torch.manual_seed(0)
if name.startswith("orig"):
    kst = int(name[-1])
    ext = BP.build_ext()
    slots = 16
    lazy = int(name.startswith("origlazy"))
    run = lambda *a: ext.lut_rag(*a, kst, BP.D_SCALE, lazy)  # noqa: E731
elif name.startswith("p1:"):
    import p1_ext as P1
    nm = name[3:]
    E = P1.build([P1.parse(nm)])
    kst, slots = P1.parse(nm)[0], E.occ[nm]["slots_per_split"]
    run = lambda *a: E.run(nm, *a, BP.D_SCALE)  # noqa: E731
else:
    import p3_ext as P3
    import p3_sets
    E = P3.build(p3_sets.a100())  # reuse the cached build of the full set
    kst, slots = P3.parse(name)[0], 1
    run = lambda *a: E.run(name, *a, BP.D_SCALE)  # noqa: E731
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
    run(*args)
torch.cuda.synchronize()
print("done", name, n, bh, chunk, splits)
