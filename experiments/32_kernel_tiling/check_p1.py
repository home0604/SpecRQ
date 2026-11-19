"""E10 phase 1 correctness: every variant vs the original kernel (experiments/26_magicvq/csrc/vqattn_lut_rag.cu via the
phase-0 extension) on random codes, both followed by the same Triton combine; plus both vs an fp32 torch reference of
the dequantized attention, and (int8 LUT variants) the LUT score error vs the fp16 LUT, emulated in torch with the
kernel's quantization rule.
Usage: python check_p1.py --out results/E10/lut_tiling_p1_check.json [--variants name ...] [--grid stageA ...]"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import bench_p0 as BP
import p1_ext as PX

D, G, C = 128, 4, 256


def make_case(B, Hkv, n, kst, ragged, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    Hq = Hkv * G
    qkv = (torch.randn(B, (Hq + 2 * Hkv) * D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    qpos = torch.full((B,), 200000, device=dev, dtype=torch.int32)
    mu = torch.randn(2, Hkv, D, device=dev, generator=g) * 0.3
    sd = torch.rand(2, Hkv, D, device=dev, generator=g) + 0.5
    bh = B * Hkv
    if ragged:
        lo = torch.randint(0, n // 4, (bh,), device=dev, generator=g, dtype=torch.int32)
        hi = torch.randint(n // 2, n + 1, (bh,), device=dev, generator=g, dtype=torch.int32)
    else:
        lo = torch.zeros(bh, device=dev, dtype=torch.int32)
        hi = torch.full((bh,), n, device=dev, dtype=torch.int32)
    planes = torch.randint(0, 256, (2, kst, B, Hkv, n, 16), dtype=torch.uint8, device=dev, generator=g)
    cb = (torch.randn(2, Hkv, kst, C, 8, device=dev, generator=g) *
          torch.tensor([1.0, 0.4], device=dev)[None, None, :kst, None, None]).to(torch.bfloat16)
    inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))
    return dict(B=B, Hkv=Hkv, n=n, kst=kst, qkv=qkv, qpos=qpos, mu=mu, sd=sd, lo=lo, hi=hi, planes=planes, cb=cb,
                inv=inv)


def args_of(c, ws, chunk, splits, spt, hi=None):
    kst, p, cb = c["kst"], c["planes"], c["cb"]
    return (c["qkv"], [p[0, s] for s in range(kst)], [p[1, s] for s in range(kst)], [cb[0, :, s] for s in range(kst)],
            [cb[1, :, s] for s in range(kst)], c["inv"], ws[0], ws[1], ws[2], c["lo"], c["hi"] if hi is None else hi,
            c["qpos"], c["sd"][0], c["mu"][0], c["sd"][1], c["mu"][1], chunk, splits, c["B"], c["Hkv"], ws[3], spt, 0)


def run_orig(ext0, c, chunk):
    splits = -(-c["n"] // chunk)
    spt = 16 * splits
    ws = BP.EK.workspace(c["B"], c["Hkv"], G, 1, spt)
    out = torch.empty(c["B"], c["Hkv"] * G * D, dtype=torch.bfloat16, device="cuda")
    ext0.lut_rag(*args_of(c, ws, chunk, splits, spt), c["kst"], BP.D_SCALE)
    BP.EK.combine(ws, spt, out, c["B"], c["Hkv"], G, 1)
    return out


def run_var(E, name, c, chunk):
    splits = -(-c["n"] // chunk)
    spt = E.occ[name]["slots_per_split"] * splits
    ws = BP.EK.workspace(c["B"], c["Hkv"], G, 1, spt)
    out = torch.empty(c["B"], c["Hkv"] * G * D, dtype=torch.bfloat16, device="cuda")
    E.run(name, *args_of(c, ws, chunk, splits, spt), BP.D_SCALE)
    BP.EK.combine(ws, spt, out, c["B"], c["Hkv"], G, 1)
    return out


def rot_q(c):
    """q after RoPE (bf16-rounded like the kernel), float32 [B, Hkv, G, D]."""
    B, Hkv = c["B"], c["Hkv"]
    q = c["qkv"][:, :Hkv * G * D].float().view(B, Hkv, G, D)
    cc = torch.arange(D, device=q.device)
    ang = c["qpos"].float()[:, None] * c["inv"][cc % (D // 2)][None, :]  # [B, D]
    xp = q[..., (cc + D // 2) % D]
    sgn = torch.where(cc < D // 2, -1.0, 1.0).to(q.device)
    rot = q * torch.cos(ang)[:, None, None, :] + sgn * xp * torch.sin(ang)[:, None, None, :]
    return rot.to(torch.bfloat16).float()


def ref_attn(c):
    """fp64 reference of the VQ attention output [B, Hq*D] (dequantized K/V codes, exact softmax)."""
    B, Hkv, kst, n = c["B"], c["Hkv"], c["kst"], c["n"]
    rot = rot_q(c).double()
    sdk, muk, sdv, muv = (c["sd"][0].double(), c["mu"][0].double(), c["sd"][1].double(), c["mu"][1].double())
    out = torch.zeros(B, Hkv, G, D, dtype=torch.float64, device="cuda")
    cbf = c["cb"].float()  # [2, Hkv, kst, C, 8]
    for b in range(B):
        for h in range(Hkv):
            bh = b * Hkv + h
            lo, hi = int(c["lo"][bh]), int(c["hi"][bh])
            K = torch.zeros(hi - lo, 16, 8, device="cuda")
            V = torch.zeros(hi - lo, 16, 8, device="cuda")
            for s in range(kst):
                kc = c["planes"][0, s, b, h, lo:hi].long()
                vc = c["planes"][1, s, b, h, lo:hi].long()
                K += cbf[0, h, s][kc]
                # kernel: per-stage bf16 codeword (pack8 of one stage), summed in fp32
                V += cbf[1, h, s][vc]
            K = K.reshape(-1, D).double() * sdk[h] + muk[h]
            V = V.reshape(-1, D).double() * sdv[h] + muv[h]
            s_ = (rot[b, h] @ K.T) * BP.D_SCALE  # log2 domain
            p = torch.softmax(s_ * math.log(2.0), dim=-1)
            out[b, h] = p @ V
    return out.view(B, -1)


def lut_score_err(c, lutm):
    """Emulated LUT scores (no qmu offset) for every token: fp16 LUT (kernel LUTM 0) vs int8 LUT (LUTM 1/2).
    Returns max abs score error, rms score std, and per-row stats (log2-domain scores)."""
    B, Hkv, kst = c["B"], c["Hkv"], c["kst"]
    rot = rot_q(c)
    q = rot * c["sd"][0][None, :, None, :] * BP.D_SCALE  # [B, Hkv, G, D]
    q = q.view(B, Hkv, G, 16, 8)
    cbk = c["cb"][0].float()  # [Hkv, kst, C, 8]
    lut = torch.einsum("bhgjd,hscd->bhsjcg", q, cbk)  # fp32 [B, Hkv, kst, 16, C, G]
    l16 = lut.half().float()
    if lutm == 1:
        mx = lut.abs().amax(dim=4, keepdim=True)
    else:
        mx = lut.abs().amax(dim=(2, 3, 4), keepdim=True)
    inv = torch.where(mx > 0, 127.0 / mx, torch.zeros_like(mx))
    qi = torch.clamp(torch.round(lut * inv), -127, 127)
    l8 = qi * (mx / 127.0)
    errs, sstd = [], []
    for b in range(B):
        for h in range(Hkv):
            bh = b * Hkv + h
            lo, hi = int(c["lo"][bh]), int(c["hi"][bh])
            s16 = torch.zeros(hi - lo, G, device="cuda")
            s8 = torch.zeros(hi - lo, G, device="cuda")
            for s in range(kst):
                kc = c["planes"][0, s, b, h, lo:hi].long()  # [T, 16]
                for j in range(16):
                    s16 += l16[b, h, s, j][kc[:, j]]
                    s8 += l8[b, h, s, j][kc[:, j]]
            errs.append((s8 - s16).abs().max().item())
            sstd.append(s16.std(0).mean().item())
    return dict(max_abs_score_err=max(errs), mean_score_std=sum(sstd) / len(sstd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="*", default=[])
    a = ap.parse_args()
    names = a.variants
    vs = [PX.parse(n) for n in names]
    E = PX.build(vs)
    ext0 = BP.build_ext()
    res = dict(cases=[], variants={n: E.occ[n] for n in E.names})
    cases = [(2, 2, 4096, False, 1024), (4, 2, 6000, True, 512), (2, 2, 3000, True, 4096), (1, 2, 20000, True, 2048)]
    for kst in (1, 2):
        for ci, (B, Hkv, n, rag, chunk) in enumerate(cases):
            c = make_case(B, Hkv, n, kst, rag, seed=100 + ci)
            ref = ref_attn(c).float()
            o0 = run_orig(ext0, c, chunk).float()
            rs = ref.abs().max().item()
            e_orig_ref = ((o0 - ref).abs().max() / rs).item()
            row = dict(kst=kst, B=B, Hkv=Hkv, n=n, ragged=rag, chunk=chunk, orig_vs_ref=e_orig_ref, var={})
            for nm in E.names:
                if E.fields[nm]["kst"] != kst:
                    continue
                ov = run_var(E, nm, c, chunk).float()
                d = dict(vs_orig_maxrel=((ov - o0).abs().max() / o0.abs().max()).item(),
                         vs_orig_bitexact=bool(torch.equal(ov, o0)),
                         vs_ref_maxrel=((ov - ref).abs().max() / rs).item(),
                         vs_ref_relfro=((ov - ref).norm() / ref.norm()).item(),
                         nan=bool(torch.isnan(ov).any()))
                if E.fields[nm]["lutm"]:
                    d.update(lut_score_err(c, E.fields[nm]["lutm"]))
                row["var"][nm] = d
                print(f"kst{kst} case{ci} {nm}: {d}", flush=True)
            res["cases"].append(row)
    # pass/fail: fp16-LUT variants must match the original up to float reassociation (bf16 output: ~1 ulp)
    summ = {}
    for nm in E.names:
        ds = [r["var"][nm] for r in res["cases"] if nm in r["var"]]
        worst = max(d["vs_orig_maxrel"] for d in ds)
        refw = max(d["vs_ref_maxrel"] for d in ds)
        origw = max(r["orig_vs_ref"] for r in res["cases"] if r["kst"] == E.fields[nm]["kst"])
        lutm = E.fields[nm]["lutm"]
        ok = (not any(d["nan"] for d in ds)) and (worst < 2e-2 if lutm == 0 else refw < 10 * max(origw, 1e-3))
        summ[nm] = dict(max_rel_vs_orig=worst, max_rel_vs_ref=refw, orig_max_rel_vs_ref=origw, passed=ok,
                        bitexact_all=all(d["vs_orig_bitexact"] for d in ds),
                        max_abs_score_err=max((d.get("max_abs_score_err", 0) for d in ds), default=0))
        print(nm, summ[nm], flush=True)
    res["summary"] = summ
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
