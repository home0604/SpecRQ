"""E10 phase 2 correctness: every phase-2 variant vs the original kernel (experiments/26_magicvq/csrc/vqattn_lut_rag.cu
via the phase-0 extension, + Triton combine) and vs the fp64 reference of check_p1, on random codes incl. ragged
ranges; several knobs per case (DEC 1: block counts whose chunks straddle bh boundaries). FUSE 1 variants are called
twice (the second call checks the self-resetting counters) and must give identical outputs. Also reports bit-exactness
vs the phase-2 re-implementation of the phase-1 struct (k*_w8_s0_d0_p0_f0) at the same knob.
Usage: python check_p2.py --out results/E10/lut_tiling_p2_check.json [--set screen | --variants name ...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import bench_p0 as BP
import check_p1 as CP
import p2_common as PC
import p1_ext as P1X
import p1_sets
import p2_ext as PX
import p2_sets


def run_p2(E, name, c, knob):
    f = E.fields[name]
    B, Hkv, kst = c["B"], c["Hkv"], c["kst"]
    lens = (c["hi"] - c["lo"]).tolist()
    pl = PC.plan(f["dec"], lens, knob)
    ws = PC.workspace(B * Hkv, pl["spt"])
    cu_t = torch.tensor(pl["cu"], dtype=torch.int32, device="cuda")
    bufs = PC.Bufs(B, Hkv, kst)
    outs = []
    for _ in range(2 if f["fuse"] else 1):
        PC.call(E, name, c["qkv"], c["planes"], c["cb"], c["inv"], ws, c["lo"], c["hi"], c["qpos"], c["sd"], c["mu"],
                pl, cu_t, bufs, B, Hkv, kst, BP.D_SCALE)
        if not f["fuse"]:
            BP.EK.combine(ws, pl["spt"], bufs.out, B, Hkv, CP.G, 1)
        torch.cuda.synchronize()
        outs.append(bufs.out.clone())
    rep_ok = all(torch.equal(o, outs[0]) for o in outs)
    return outs[0], pl, rep_ok, bool((bufs.cnt == 0).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default=None)
    ap.add_argument("--variants", nargs="*", default=[])
    a = ap.parse_args()
    vs = [PX.parse(n) for n in a.variants] + (p2_sets.SETS[a.set]() if a.set else [])
    vs += [(k, 8, 0, 0, 0, 0) for k in (1, 2)]
    E = PX.build(vs)
    ext0 = PC.build_orig()
    E1 = P1X.build(p1_sets.final())  # phase-1 finalists (cached build); reference struct w8_b1_r1_p1_l0_s2
    res = dict(cases=[], variants={n: E.occ[n] for n in E.names}, fields=E.fields)
    # (B, Hkv, n, ragged, chunk for orig/DEC0, P list for DEC1)
    cases = [(2, 2, 4096, False, 1024, [4, 7, 64]), (4, 2, 6000, True, 512, [5, 24, 300]),
             (2, 2, 3000, True, 4096, [1, 3, 9]), (1, 2, 20000, True, 2048, [2, 13, 128])]
    for kst in (1, 2):
        for ci, (B, Hkv, n, rag, chunk, Ps) in enumerate(cases):
            c = CP.make_case(B, Hkv, n, kst, rag, seed=100 + ci)
            ref = CP.ref_attn(c).float()
            o0 = CP.run_orig(ext0, c, chunk).float()
            rs = ref.abs().max().item()
            row = dict(kst=kst, B=B, Hkv=Hkv, n=n, ragged=rag, chunk=chunk, Ps=Ps,
                       orig_vs_ref=((o0 - ref).abs().max() / rs).item(), var={})
            basename = f"k{kst}_w8_s0_d0_p0_f0"
            base = {}
            for kb in (chunk, max(256, chunk // 4)):
                base[kb] = run_p2(E, basename, c, kb)[0].float()
                p1o = CP.run_var(E1, f"k{kst}_w8_b1_r1_p1_l0_s2", c, kb).float()
                row.setdefault("p2base_bitexact_vs_p1", []).append(bool(torch.equal(p1o, base[kb])))
            for nm in E.names:
                f = E.fields[nm]
                if f["kst"] != kst or E.occ[nm]["blocks_per_sm"] < 1:
                    continue
                knobs = Ps if f["dec"] else [chunk, max(256, chunk // 4)]
                for kb in knobs:
                    ov, pl, rep_ok, cnt0 = run_p2(E, nm, c, kb)
                    ov = ov.float()
                    d = dict(knob=kb, chunk=pl["chunk"], nblk=pl["nblk"], spt=pl["spt"],
                             vs_orig_maxrel=((ov - o0).abs().max() / o0.abs().max()).item(),
                             vs_ref_maxrel=((ov - ref).abs().max() / rs).item(),
                             vs_ref_relfro=((ov - ref).norm() / ref.norm()).item(),
                             nan=bool(torch.isnan(ov).any()), repeat_identical=rep_ok, counters_reset=cnt0)
                    if not f["dec"]:
                        d["bitexact_vs_p1struct"] = bool(torch.equal(ov, base[kb]))
                    row["var"].setdefault(nm, []).append(d)
                    print(f"kst{kst} case{ci} {nm} {d}", flush=True)
            res["cases"].append(row)
    summ = {}
    for nm in E.names:
        ds = [d for r in res["cases"] for d in r["var"].get(nm, [])]
        if not ds:
            summ[nm] = dict(passed=False, reason="not launchable")
            continue
        k = E.fields[nm]["kst"]
        origw = max(r["orig_vs_ref"] for r in res["cases"] if r["kst"] == k)
        worst = max(d["vs_orig_maxrel"] for d in ds)
        refw = max(d["vs_ref_maxrel"] for d in ds)
        ok = (not any(d["nan"] for d in ds) and worst < 2e-2 and refw <= 1.5 * origw + 1e-3
              and all(d["repeat_identical"] and d["counters_reset"] for d in ds))
        be = [d["bitexact_vs_p1struct"] for d in ds if "bitexact_vs_p1struct" in d]
        summ[nm] = dict(passed=ok, max_rel_vs_orig=worst, max_rel_vs_ref=refw, orig_max_rel_vs_ref=origw,
                        n_tests=len(ds), bitexact_vs_p1struct_all=(all(be) if be else None))
        print(nm, summ[nm], flush=True)
    res["summary"] = summ
    res["p2base_bitexact_vs_p1_all"] = all(x for r in res["cases"] for x in r["p2base_bitexact_vs_p1"])
    res["all_passed"] = all(s["passed"] for s in summ.values() if s.get("reason") != "not launchable")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, "all_passed", res["all_passed"])


if __name__ == "__main__":
    main()
