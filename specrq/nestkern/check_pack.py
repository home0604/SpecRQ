"""M1 check: planar packer + prefix-k reference vs DbD `_dequantize_weight(num_codebooks=k)` on a real ckpt layer.

Read-only on the checkpoint. Usage:
  python -m specrq.nestkern.check_pack --ckpt /data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny --layer 0 [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DBD = REPO / "specrq" / "dbd"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(DBD))  # DbD pickles reference `src.*`
    from src.utils import _dequantize_weight  # DbD path (prefix via num_codebooks)
    from specrq.nestkern.pack import pack_planar, ref_dequant, unpack_planar

    layer = torch.load(f"{a.ckpt}/{a.layer}.pth", map_location=a.device, weights_only=False)
    rows = []
    for name, mod in layer.named_modules():
        if not hasattr(mod, "quantized_weight"):
            continue
        q = mod.quantized_weight
        codes, cbs, sc = q.get_codes(), q.get_codebooks().detach(), q.get_scales().detach()
        p = pack_planar(codes, cbs, sc)
        c2, cb2, sc2 = unpack_planar(p)
        roundtrip = bool(torch.equal(c2, codes.to(torch.int32)) and torch.equal(cb2, cbs) and torch.equal(sc2, sc))
        for k in range(1, p.L + 1):
            with torch.no_grad():
                w_dbd32 = _dequantize_weight(codes, cbs.float(), sc.float(), k)
                w_ref32 = ref_dequant(p, k, torch.float32)
                w_dbd_nat = _dequantize_weight(codes, cbs, sc, k)          # native dtype (bf16), DbD rounding
                w_ref_nat = ref_dequant(p, k, torch.float32).to(cbs.dtype)  # fp32 sum, one rounding
                w_mod = q(num_codebooks=k)                                  # QuantizedWeight.forward path
            d32 = (w_dbd32 - w_ref32).abs().max().item()
            rel32 = ((w_dbd32 - w_ref32).norm() / w_dbd32.norm()).item()
            dn = (w_dbd_nat.float() - w_ref_nat.float())
            ulp = torch.finfo(cbs.dtype).eps * w_dbd_nat.float().abs().clamp_min(torch.finfo(cbs.dtype).tiny)
            rows.append({"module": name, "N": p.N, "K": p.K, "k": k, "roundtrip_exact": roundtrip,
                         "fp32_maxabs": d32, "fp32_rel": rel32,
                         "native_dtype": str(cbs.dtype),
                         "native_bitexact_frac": (dn == 0).float().mean().item(),
                         "native_max_ulp": (dn.abs() / ulp).max().item(),
                         "native_rel": (dn.norm() / w_dbd_nat.float().norm()).item(),
                         "module_forward_equal_dbd": bool(torch.equal(w_mod, w_dbd_nat))})
            r = rows[-1]
            print(f"{name:18} {p.N}x{p.K} k={k} roundtrip={roundtrip} fp32 maxabs={d32:.2e} rel={rel32:.2e} | "
                  f"{r['native_dtype']} bitexact={r['native_bitexact_frac']:.4f} max_ulp={r['native_max_ulp']:.2f} "
                  f"rel={r['native_rel']:.2e} fwd==dbd {r['module_forward_equal_dbd']}")
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E0", a.save, config={"setting": "M1 planar packer vs DbD dequant", "ckpt": a.ckpt,
                                                          "layer": a.layer, "format": "planar_rowmajor_v1"},
                                   metrics={"rows": rows}, model_dir="/data/models/qwen2.5_0.5b_instruct"))


if __name__ == "__main__":
    main()
