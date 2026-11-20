"""E10 phase 3 -> e2e: checks of the opt-in MV_LUT_VARIANT path of e2e_kern.lut_partial (p3 kernels, one partial
slot per split + combine over the used slots) against the original lut_rag path on the same inputs and seeds, both
behind the same bf16 sink / window partials (e2e_real layout: bf16 slots [0, bs), VQ slots after them).

Cases: KST (stages) 1 / 2, q scale 1 / 4, ragged per-(b, kv-head) VQ ranges, small shapes and the TP4 rank shape
(B16, Hkv 2, 32k), splits / chunks = e2e_real tune_plan VQ candidates (chunk 1024 / 2048 / 4096 / 8192), RQ-range
(lo > 0) and hybrid-style (lo = 0, compacted) ranges. Every compiled p3 variant that is launchable on the device, and
the presets (a100 / 4090 / auto, with the smem fallback), are compared.
Pass: max |o_p3 - o_orig| / max |o_orig| <= TOL (2 bf16 ulp = 2^-7; phase 3 measured 2.4e-3 .. 4.4e-3), no NaN, and
error vs an fp32 dequantized reference no worse than 1.25x the original's (+1e-3) where the reference is computed.
Also checks that MV_LUT_VARIANT unset leaves workspace() a plain tuple (the default path is untouched).
Usage: CUDA_VISIBLE_DEVICES=<free gpu> python experiments/26_magicvq/e2e_kern_test_p3.py [--out results/E10/x.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_kern as EK  # noqa: E402

dev = "cuda"
G, D = 4, 128
TOL = 2.0 ** -7
inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))


def rope(x, pos):
    ang = pos[..., None].float() * inv.repeat(2)
    rot = torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1)
    return x * ang.cos() + rot * ang.sin()


def make_case(B, Hkv, T, S, qscale, mode, seed):
    """random codes / codebooks / chnorm stats; bf16 cache for sinks + window; ragged VQ ranges [vlo, vhi)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    Hq = Hkv * G
    c = dict(B=B, Hkv=Hkv, T=T, S=S, qscale=qscale, mode=mode, seed=seed)
    c["codes"] = torch.randint(0, 256, (2, S, B, Hkv, T, 16), generator=g, device=dev, dtype=torch.int32).to(torch.uint8)
    sc = torch.tensor([1.0, 0.4], device=dev)[:S]
    c["cb"] = (torch.randn(2, Hkv, S, 256, 8, generator=g, device=dev) * sc[None, None, :, None, None]).to(torch.bfloat16)
    c["mu"] = torch.randn(2, Hkv, D, generator=g, device=dev) * 0.3
    c["sd"] = torch.rand(2, Hkv, D, generator=g, device=dev) + 0.5
    c["qkv"] = (torch.randn(B, (Hq + 2 * Hkv) * D, generator=g, device=dev) * qscale).to(torch.bfloat16)
    W = 128
    # target position of the query row; bf16 window [qpos + 1 - W, qpos] (+ 16 sinks), VQ part = everything between
    qpos = T + W - 1 - torch.randint(0, max(1, T // 3), (B,), generator=g, device=dev)
    if mode == "rq":  # vqpL2+r128: VQ over [16, qpos + 1 - W) (range mode planes indexed by position)
        vlo = torch.full((B * Hkv,), 16, device=dev)
        vhi = (qpos + 1 - W).repeat_interleave(Hkv)
    else:  # hyb: compacted planes [0, ncomp) (ncomp differs per (b, kv-head))
        vlo = torch.zeros(B * Hkv, device=dev)
        vhi = (qpos + 1 - W).repeat_interleave(Hkv) - torch.randint(0, max(1, T // 4), (B * Hkv,), generator=g,
                                                                    device=dev)
    vhi = vhi.clamp(min=0, max=T)
    if B * Hkv > 2:
        vhi[2] = vlo[2]  # one empty VQ range
    c["qpos"], c["vlo"], c["vhi"] = qpos.int(), vlo.int(), vhi.int()
    c["blo"] = (qpos + 1 - W).int()
    c["kvc"] = torch.randn(2, B, Hkv, T + W, D, generator=g, device=dev).to(torch.bfloat16)
    c["sink"] = torch.arange(16, device=dev, dtype=torch.int32)[None, None].expand(B, Hkv, 16).contiguous()
    c["nsink"] = torch.full((B * Hkv,), 16, device=dev, dtype=torch.int32)
    return c


def run(c, vs, vch, variant):
    """bf16 sinks/window partials + VQ partials + combine; variant None = original lut_rag path."""
    B, Hkv, S = c["B"], c["Hkv"], c["S"]
    EK.LUT_VARIANT = variant
    EK._P3_VID.clear()
    bs, bch = 2, 128  # 16 sinks + 128-token window <= 2 * 128
    spt = bs + 16 * vs  # e2e_real Attn layout
    ws = EK.workspace(B, Hkv, G, 1, spt)
    for t in ws[:3]:
        t.fill_(float("nan"))  # unused slots must never be read
    out = torch.empty(B, Hkv * G * D, device=dev, dtype=torch.bfloat16)
    kvc, cb, codes = c["kvc"], c["cb"], c["codes"]
    EK.bf_partial(c["qkv"], kvc[0], kvc[1], inv, ws, spt, 0, bs, bch, c["blo"], c["qpos"] + 1, c["qpos"], 1, G,
                  c["sink"], c["nsink"])
    EK.lut_partial(c["qkv"], [codes[0, s] for s in range(S)], [codes[1, s] for s in range(S)],
                   [cb[0, :, s] for s in range(S)], [cb[1, :, s] for s in range(S)], inv, ws, spt, bs, vs, vch,
                   c["vlo"], c["vhi"], c["qpos"], c["sd"][0].contiguous(), c["mu"][0].contiguous(),
                   c["sd"][1].contiguous(), c["mu"][1].contiguous(), B, Hkv, S, lazy=0)
    EK.combine(ws, spt, out, B, Hkv, G, 1)
    EK.LUT_VARIANT = None
    return out.float()


def ref(c, bhs):
    """fp32 reference for the listed (b, h) pairs."""
    S = c["S"]
    q = c["qkv"][:, :c["Hkv"] * G * D].float().view(c["B"], c["Hkv"], G, D)
    res = {}
    for b, h in bhs:
        bh = b * c["Hkv"] + h
        a, z = c["vlo"][bh].item(), c["vhi"][bh].item()
        p = c["qpos"][b].item()
        deq = []
        for kv in range(2):
            cc = c["codes"][kv, :, b, h, a:z].long()
            r = sum(c["cb"][kv, h, s].float()[cc[s]] for s in range(S)).reshape(z - a, D)
            deq.append(r * c["sd"][kv, h] + c["mu"][kv, h])
        bpos = torch.cat([torch.arange(16, device=dev), torch.arange(c["blo"][b].item(), p + 1, device=dev)])
        K = torch.cat([c["kvc"][0, b, h, bpos].float(), deq[0]])
        V = torch.cat([c["kvc"][1, b, h, bpos].float(), deq[1]])
        qr = rope(q[b, h], torch.tensor(p, device=dev)).bfloat16().float()
        s_ = qr @ K.T / math.sqrt(D)
        res[(b, h)] = torch.softmax(s_, -1) @ V
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    saved = EK.LUT_VARIANT
    EK.LUT_VARIANT = None
    assert type(EK.workspace(1, 1, G, 1, 4)) is tuple, "default workspace must stay a plain tuple"
    e = EK.p3_ext()
    names = [n for n, _ in e.variants()]
    occ = {n: list(e.occupancy(i)[:4]) for i, n in enumerate(names)}
    launch = [n for n in names if occ[n][0] >= 1]
    print("device", torch.cuda.get_device_name(), "cc", torch.cuda.get_device_capability(), flush=True)
    print("not launchable here:", [n for n in names if n not in launch], flush=True)
    specs = {n: "p3:{}={}".format(n[:2], n[3:]) for n in launch}
    for p in ("a100", "4090", "auto"):
        specs[p] = p
    cases = [  # (B, Hkv, T, mode, chunks) ; chunks = tune_plan VQ candidates (+ small ones for the small shapes)
        (3, 2, 700, "rq", (256, 1024)),
        (4, 2, 6000, "hyb", (1024, 2048, 4096)),
        (16, 2, 32768, "rq", (1024, 2048, 4096, 8192)),
        (16, 2, 32768, "hyb", (2048, 8192)),
    ]
    rows, fails = [], []
    for S in (1, 2):
        for qs in (1.0, 4.0):
            for ci, (B, Hkv, T, mode, chunks) in enumerate(cases):
                c = make_case(B, Hkv, T, S, qs, mode, seed=1000 * S + 10 * ci + int(qs))
                bhs = [(0, 0), (B - 1, Hkv - 1), (1, 0)]
                r = ref(c, bhs)
                for vch in chunks:
                    nvq = int((c["vhi"] - c["vlo"]).max())
                    vs = max(1, -(-nvq // vch))
                    o0 = run(c, vs, vch, None)
                    o1 = run(c, vs, vch, None)
                    om = o0.abs().max().item()
                    o0v = o0.view(B, Hkv, G, D)
                    e0 = max((o0v[b, h] - r[(b, h)]).abs().max().item() for b, h in bhs)
                    row = dict(S=S, qscale=qs, B=B, Hkv=Hkv, T=T, mode=mode, chunk=vch, splits=vs,
                               orig_rerun_bitexact=bool(torch.equal(o0, o1)), orig_vs_ref_maxabs=e0, var={})
                    for key, spec in specs.items():
                        if spec.startswith("p3:") and int(spec[4]) != S:
                            continue
                        ov = run(c, vs, vch, spec)
                        used = EK.LUT_VARIANT_INFO[S]["used"]
                        d = (ov - o0).abs().max().item() / om
                        ovv = ov.view(B, Hkv, G, D)
                        ev = max((ovv[b, h] - r[(b, h)]).abs().max().item() for b, h in bhs)
                        nan = bool(torch.isnan(ov).any())
                        ok = (not nan) and d <= TOL and ev <= 1.25 * e0 + 1e-3
                        row["var"][key] = dict(used=used, fallback=EK.LUT_VARIANT_INFO[S]["fallback"],
                                               vs_orig_maxrel=d, vs_ref_maxabs=ev, nan=nan, ok=ok)
                        if not ok:
                            fails.append((S, qs, ci, vch, key, used, d, ev, e0))
                        print(f"S{S} qs{qs} case{ci} {mode} B{B} T{T} chunk{vch} splits{vs} {key:>14} -> {used}: "
                              f"vs_orig {d:.2e} (max|o| {om:.3f}) vs_ref {ev:.2e} (orig {e0:.2e}) "
                              f"{'ok' if ok else 'FAIL'}", flush=True)
                    rows.append(row)
    EK.LUT_VARIANT = saved
    res = dict(device=torch.cuda.get_device_name(), cc=list(torch.cuda.get_device_capability()), tol=TOL,
               occupancy=occ, launchable=launch, rows=rows, fails=fails, all_passed=not fails,
               variant_info={str(k): v for k, v in EK.LUT_VARIANT_INFO.items()})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("ALL OK" if not fails else f"FAILED {len(fails)}: {fails[:10]}", flush=True)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
