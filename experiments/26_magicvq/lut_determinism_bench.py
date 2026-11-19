"""lut_rag determinism + timing at B16 x 32k, per-rank TP4 shapes (Hkv=2, G=4).\nUsage: python lut_determinism_bench.py [splits] [kst (2; v8: 1)] [T]"""
import sys, statistics, torch
sys.path.insert(0, "/hai/home/lsh/SpecRQ/experiments/26_magicvq")
import e2e_kern as EK
torch.manual_seed(0)
dev = "cuda"
B, Hkv, G, D = 16, 2, 4, 128
S = int(sys.argv[2]) if len(sys.argv) > 2 else 2
T = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
Hq = Hkv * G
inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
codes = torch.randint(0, 256, (2, S, B, Hkv, T, 16), dtype=torch.uint8, device=dev)
cb = (torch.randn(2, Hkv, S, 256, 8, device=dev) * torch.tensor([1.0, 0.4], device=dev)[None, None, :S, None, None]).to(torch.bfloat16)
mu = torch.randn(2, Hkv, D, device=dev) * 0.3
sd = torch.rand(2, Hkv, D, device=dev) + 0.5
qkv = torch.randn(B, (Hq + 2 * Hkv) * D, device=dev).to(torch.bfloat16)
qpos = torch.full((B,), T - 1, device=dev, dtype=torch.int32)
vlo = torch.full((B * Hkv,), 16, device=dev, dtype=torch.int32)
vhi = torch.full((B * Hkv,), T - 128, device=dev, dtype=torch.int32)
sp = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ch = -(-(T - 144) // sp)
ch = -(-ch // 64) * 64
spt = 16 * sp
ws = EK.workspace(B, Hkv, G, 1, spt)
args = ([codes[0, s] for s in range(S)], [codes[1, s] for s in range(S)], [cb[0, :, s] for s in range(S)],
        [cb[1, :, s] for s in range(S)], inv, ws, spt, 0, sp, ch, vlo, vhi, qpos, sd[0].contiguous(), mu[0].contiguous(),
        sd[1].contiguous(), mu[1].contiguous(), B, Hkv, S)
def run():
    EK.lut_partial(qkv, *args)
run(); torch.cuda.synchronize()
ref = [t.clone() for t in ws[:3]]
ndiff = 0
for i in range(200):
    run()
    if not all(torch.equal(a, b) for a, b in zip(ref, ws[:3])):
        ndiff += 1
print(f"bitwise-different repeats: {ndiff}/200 ; MP max|diff| vs first = "
      f"{(ws[1][torch.isfinite(ref[1])] - ref[1][torch.isfinite(ref[1])]).abs().max().item():.3e}")
times = []
for rep in range(7):
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(100):
        run()
    e1.record(); torch.cuda.synchronize()
    times.append(e0.elapsed_time(e1) / 100 * 1e3)
print(f"kst {S} T {T} splits {sp} chunk {ch}: lut_rag us/call median {statistics.median(times):.2f} min {min(times):.2f} max {max(times):.2f} all {[round(t,2) for t in times]}")
