"""Kernel-time breakdown (torch.profiler, CUDA activity) of one NestModel forward on the SYNTHETIC Llama-3.1-8B
checkpoint for (k, M) cases; groups kernels into linear / lm_head / glue / fwht / attention / kv_write.
Eager forward (not graphed) so absolute sums include no launch gaps. Usage: python -m specrq.infer.profile_forward --run NAME
"""
import argparse
import sys, torch
REPO = "/hai/home/lsh/SpecRQ"
sys.path.insert(0, "/hai/home/lsh/SpecRQ")
from specrq.infer.model import NestModel
from collections import defaultdict
from torch.profiler import profile, ProfilerActivity
ap = argparse.ArgumentParser(); ap.add_argument("--run", default=None); a = ap.parse_args()
m = NestModel.synthetic("/data/models/llama3.1_8b_instruct", verbose=False)
out = []
def group(n):
    if "nest_" in n: return "linear_nest"
    if "gemvx" in n: return "lm_head"
    if "triton" in n: return "glue_compiled"
    if "fwht" in n: return "fwht"
    if "gemm" in n or "sgemm" in n: return "attention_fp32_bmm"
    if "index" in n: return "kv_write_index"
    return "other"
c = m.new_cache(256); p = torch.tensor(128, device="cuda")
for k, M in ((5, 1), (3, 1), (2, 1), (1, 1), (5, 2), (5, 3), (5, 5), (5, 8)):
    t = torch.randint(0, 1000, (M,), device="cuda")
    with torch.no_grad():
        for _ in range(3): m.forward(t, p, c, k, last_only=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            m.forward(t, p, c, k, last_only=True); torch.cuda.synchronize()
    agg = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            agg[e.name[:70]][0] += 1; agg[e.name[:70]][1] += e.device_time_total
    tot = sum(v[1] for v in agg.values())
    print(f"=== k={k} M={M}: kernels {sum(v[0] for v in agg.values())}, GPU sum {tot:.0f} us")
    grp = defaultdict(float)
    for kk, v in agg.items(): grp[group(kk)] += v[1]
    print("  ", {g: round(v, 1) for g, v in grp.items()})
    out.append({"k": k, "M": M, "gpu_sum_us": tot, "groups_us": dict(grp),
                "kernels": {kk: {"n": v[0], "us": v[1]} for kk, v in agg.items()}})
if a.run:
    sys.path.insert(0, REPO)
    from tools.results import save_result
    print("saved", save_result("E4", a.run, config={"setting": "SYNTHETIC Llama-3.1-8B C1 ckpt (timing only), torch.profiler "
                                                     "kernel sums of one eager forward, cache T=256, pos=128, det split-K"},
                               metrics={"cases": out}, model_dir="/data/models/llama3.1_8b_instruct"))
