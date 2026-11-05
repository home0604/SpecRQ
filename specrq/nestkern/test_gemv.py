"""M2 correctness: nest_gemv vs fp32 prefix-k reference, random shapes (fp16/bf16) and the real smoke ckpt layer.

Usage: python -m specrq.nestkern.test_gemv [--ckpt DIR --layer 0] [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.nestkern.ops import nest_gemv
from specrq.nestkern.pack import PlanarVQ, kernel_codes, pack_planar, ref_matmul

REPO = Path(__file__).resolve().parents[2]
SHAPES = [(896, 896), (128, 896), (4864, 896), (896, 4864), (1152, 896), (9728, 896),
          (4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336), (6144, 4096), (28672, 4096)]


def rand_planar(N, K, L=5, dtype=torch.float16, dev="cuda"):
    codes = torch.randint(0, 256, (L, N, K // 8), device=dev, dtype=torch.uint8)
    cb = (torch.randn(L, 256, 8, device=dev) * 3 / (K * L) ** 0.5).to(dtype)
    sc = (torch.rand(N, device=dev) + 0.5).to(dtype)
    return PlanarVQ(codes, cb, sc, N, K)


def relerr(y, yr):
    return ((y.float() - yr).norm() / yr.norm()).item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for N, K in SHAPES:
            p = rand_planar(N, K, dtype=dtype)
            x = torch.randn(1, K, device="cuda").to(dtype)
            kc = kernel_codes(p.codes)
            for k in range(1, 6):
                yr = ref_matmul(x, p, k)
                for variant in (0, 1):
                    e = relerr(nest_gemv(x, kc, p.codebooks, p.scales, k, 0, variant), yr)
                    rows.append({"src": "random", "dtype": str(dtype), "N": N, "K": K, "k": k, "variant": variant,
                                 "rel_err": e, "floor": relerr(yr.to(dtype), yr)})
        print(str(dtype), "random max rel", max(r["rel_err"] for r in rows if r["dtype"] == str(dtype)))
    # real checkpoint layer (read-only)
    sys.path.insert(0, str(REPO / "specrq" / "dbd"))
    layer = torch.load(f"{a.ckpt}/{a.layer}.pth", map_location="cpu", weights_only=False)
    for name, mod in layer.named_modules():
        if not hasattr(mod, "quantized_weight"):
            continue
        q = mod.quantized_weight
        for dtype in (torch.bfloat16, torch.float16):
            p = pack_planar(q.get_codes(), q.get_codebooks().detach(), q.get_scales().detach()).to("cuda", dtype)
            x = torch.randn(1, p.K, device="cuda").to(dtype)
            kc = kernel_codes(p.codes)
            for k in range(1, p.L + 1):
                yr = ref_matmul(x, p, k)
                for variant in (0, 1):
                    e = relerr(nest_gemv(x, kc, p.codebooks, p.scales, k, 0, variant), yr)
                    rows.append({"src": f"ckpt:{a.layer}:{name}", "dtype": str(dtype), "N": p.N, "K": p.K, "k": k,
                                 "variant": variant, "rel_err": e, "floor": relerr(yr.to(dtype), yr)})
    ck = [r for r in rows if r["src"].startswith("ckpt")]
    print("ckpt max rel", max(r["rel_err"] for r in ck), "over", len(ck), "cases")
    worst = max(rows, key=lambda r: r["rel_err"])
    print("worst", worst)
    # criterion: rel_err < 1e-3, or (bf16) within 1.5x of the unavoidable output-rounding floor rel(ref.to(dtype), ref)
    for r in rows:
        r["pass"] = r["rel_err"] < 1e-3 or r["rel_err"] < 1.5 * r["floor"]
    ok = all(r["pass"] for r in rows)
    for dt in ("torch.float16", "torch.bfloat16"):
        sel = [r for r in rows if r["dtype"] == dt]
        print(dt, "max rel", max(r["rel_err"] for r in sel), "max rel/floor", max(r["rel_err"] / r["floor"] for r in sel))
    print("PASS" if ok else "FAIL", "(rel<1e-3 or rel<1.5*output-rounding floor)")
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"setting": "M2 nest_gemv correctness vs fp32 prefix-k ref",
                                                          "ckpt": a.ckpt, "layer": a.layer, "threshold": 1e-3},
                                   metrics={"rows": rows, "max_rel_err": worst["rel_err"], "pass": ok}))


if __name__ == "__main__":
    main()
