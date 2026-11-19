"""Launch one phase-2 variant a few times at one (n, bh, knob) point for Nsight Compute (knob = chunk for DEC 0,
block count P for DEC 1). Builds only that variant.
Usage: ncu -k regex:'lut_p2' -s 2 -c 1 --set full -o out python ncu_point_p2.py VARIANT n bh knob [hkv=2]"""
import sys

import torch

import bench_p0 as BP
import p2_common as PC
import p2_ext as PX

name = sys.argv[1]
n, bh, knob = (int(x) for x in sys.argv[2:5])
hkv = int(sys.argv[5]) if len(sys.argv) > 5 else 2
v = PX.parse(name)
kst, dec = v[0], v[3]
B, G, D = bh // hkv, 4, 128
dev = "cuda"
torch.manual_seed(0)
E = PX.build([v])
inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
qkv = (torch.randn(B, (hkv * G + 2 * hkv) * D, device=dev) * 0.5).to(torch.bfloat16)
qpos = torch.full((B,), 200000, device=dev, dtype=torch.int32)
mu = torch.randn(2, hkv, D, device=dev) * 0.3
sd = torch.rand(2, hkv, D, device=dev) + 0.5
lo = torch.zeros(bh, device=dev, dtype=torch.int32)
hi = torch.full((bh,), n, device=dev, dtype=torch.int32)
p = torch.randint(0, 256, (2, kst, B, hkv, n, 16), dtype=torch.uint8, device=dev)
cb = (torch.randn(2, hkv, kst, 256, 8, device=dev) * 0.5).to(torch.bfloat16)
pl = PC.plan(dec, [n] * bh, knob)
ws = PC.workspace(bh, pl["spt"])
cu_t = torch.tensor(pl["cu"], dtype=torch.int32, device=dev)
bufs = PC.Bufs(B, hkv, kst)
for _ in range(4):
    PC.call(E, name, qkv, p, cb, inv, ws, lo, hi, qpos, sd, mu, pl, cu_t, bufs, B, hkv, kst, BP.D_SCALE)
torch.cuda.synchronize()
print("done", name, n, bh, knob, pl["chunk"], pl["nblk"])
