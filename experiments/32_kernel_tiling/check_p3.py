"""E10 phase 3 correctness: p3 variants (csrc/lut_rag_p3.cuh, AB = 0 only) vs the original kernel
(experiments/26_magicvq/csrc/vqattn_lut_rag.cu via the phase-0 extension) and vs an fp64 reference of the dequantized
attention (check_p1.ref_attn), all followed by the same Triton combine. Cases = check_p1 cases (q scale 0.5) plus
q scale 4 (many lazy-max rescales, wider score range).
Thresholds (as check_p1 for fp16-LUT variants): max |o - o_orig| / max |o_orig| < 2e-2, and max rel vs fp64 must not
exceed 2x the original's own max rel vs fp64 (+1e-3).
Usage: python check_p3.py --out results/E10/lut_tiling_p3_check.json --variants k1_w8_v1_r1_h0_a0 ..."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

import bench_p0 as BP
import check_p1 as CP
import p3_ext as PX

G = 4


def run_orig(ext0, c, chunk, lazy=0):
    """original kernel (26_magicvq/csrc/vqattn_lut_rag.cu, 16 partial slots per split); lazy = its lazy-max flag"""
    splits = -(-c["n"] // chunk)
    spt = 16 * splits
    ws = BP.EK.workspace(c["B"], c["Hkv"], G, 1, spt)
    out = torch.empty(c["B"], c["Hkv"] * G * 128, dtype=torch.bfloat16, device="cuda")
    ext0.lut_rag(*CP.args_of(c, ws, chunk, splits, spt), c["kst"], BP.D_SCALE, lazy)
    BP.EK.combine(ws, spt, out, c["B"], c["Hkv"], G, 1)
    return out


def run_var(E, name, c, chunk):
    splits = -(-c["n"] // chunk)
    spt = splits
    ws = BP.EK.workspace(c["B"], c["Hkv"], G, 1, spt)
    out = torch.empty(c["B"], c["Hkv"] * G * 128, dtype=torch.bfloat16, device="cuda")
    E.run(name, *CP.args_of(c, ws, chunk, splits, spt), BP.D_SCALE)
    BP.EK.combine(ws, spt, out, c["B"], c["Hkv"], G, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()
    E = PX.build([PX.parse(n) for n in a.variants])
    ext0 = BP.build_ext()
    lim = E.device_limits()
    names = [n for n in E.names if E.occ[n]["blocks_per_sm"] >= 1]
    skipped = [n for n in E.names if n not in names]
    print("device", torch.cuda.get_device_name(), lim, "not launchable here (smem):", skipped, flush=True)
    res = dict(device=torch.cuda.get_device_name(), limits=lim, arch=os.environ.get("TORCH_CUDA_ARCH_LIST"),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), apps=BP.nvsmi_apps(),
               variants={n: E.occ[n] for n in E.names}, skipped=skipped, cases=[])
    cases = [(2, 2, 4096, False, 1024, 0.5), (4, 2, 6000, True, 512, 0.5), (2, 2, 3000, True, 4096, 0.5),
             (1, 2, 20000, True, 2048, 0.5), (2, 2, 5000, True, 1024, 4.0), (3, 2, 777, True, 256, 4.0)]
    for kst in (1, 2):
        for ci, (B, Hkv, n, rag, chunk, qs) in enumerate(cases):
            c = CP.make_case(B, Hkv, n, kst, rag, seed=100 + ci)
            c["qkv"] = (c["qkv"].float() * (qs / 0.5)).to(torch.bfloat16)
            ref = CP.ref_attn(c).float()
            o0 = run_orig(ext0, c, chunk).float()
            ol = run_orig(ext0, c, chunk, 1).float()
            rs = ref.abs().max().item()
            row = dict(kst=kst, B=B, Hkv=Hkv, n=n, ragged=rag, chunk=chunk, qscale=qs,
                       orig_vs_ref=((o0 - ref).abs().max() / rs).item(),
                       orig_vs_ref_relfro=((o0 - ref).norm() / ref.norm()).item(),
                       origlazy_vs_ref=((ol - ref).abs().max() / rs).item(),
                       origlazy_vs_orig=((ol - o0).abs().max() / o0.abs().max()).item(), var={})
            for nm in names:
                if E.fields[nm]["kst"] != kst or E.fields[nm]["ab"]:
                    continue
                ov = run_var(E, nm, c, chunk).float()
                d = dict(vs_orig_maxrel=((ov - o0).abs().max() / o0.abs().max()).item(),
                         vs_orig_maxabs=(ov - o0).abs().max().item(),
                         vs_orig_bitexact=bool(torch.equal(ov, o0)),
                         vs_ref_maxrel=((ov - ref).abs().max() / rs).item(),
                         vs_ref_relfro=((ov - ref).norm() / ref.norm()).item(),
                         nan=bool(torch.isnan(ov).any()))
                row["var"][nm] = d
                print(f"kst{kst} case{ci} qs{qs} orig_vs_ref {row['orig_vs_ref']:.2e} {nm}: "
                      f"vs_orig {d['vs_orig_maxrel']:.2e} vs_ref {d['vs_ref_maxrel']:.2e} relfro {d['vs_ref_relfro']:.2e}",
                      flush=True)
            res["cases"].append(row)
    summ = {}
    for nm in names:
        if E.fields[nm]["ab"]:
            continue
        rows = [r for r in res["cases"] if nm in r["var"]]
        ds = [r["var"][nm] for r in rows]
        worst = max(d["vs_orig_maxrel"] for d in ds)
        refw = max(d["vs_ref_maxrel"] for d in ds)
        origw = max(r["orig_vs_ref"] for r in rows)
        ratio = max(d["vs_ref_maxrel"] / max(r["orig_vs_ref"], 1e-6) for d, r in zip(ds, rows))
        ok = (not any(d["nan"] for d in ds)) and worst < 2e-2 and all(
            d["vs_ref_maxrel"] <= 2 * r["orig_vs_ref"] + 1e-3 for d, r in zip(ds, rows))
        summ[nm] = dict(max_rel_vs_orig=worst, max_rel_vs_ref=refw, orig_max_rel_vs_ref=origw,
                        max_ratio_vs_ref_over_orig=ratio, max_relfro_vs_ref=max(d["vs_ref_relfro"] for d in ds),
                        orig_max_relfro_vs_ref=max(r["orig_vs_ref_relfro"] for r in rows), passed=ok,
                        bitexact_all=all(d["vs_orig_bitexact"] for d in ds))
        print(nm, summ[nm], flush=True)
    res["summary"] = summ
    res["all_passed"] = all(s["passed"] for s in summ.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("ALL PASSED" if res["all_passed"] else "SOME FAILED", "wrote", a.out)


if __name__ == "__main__":
    main()
