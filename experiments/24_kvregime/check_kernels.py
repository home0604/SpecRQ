"""E6 kvregime: correctness of specrq/kvq kernels against torch references.
  attention: VQ (d8 S2, d16 S4) for every prefix kst, bf16, INT4, INT2; M in {1,2,3,5}; several (B, ctx, splits)
             -> max abs err vs dequantize-then-fp32-SDPA (outputs are bf16, values O(1)).
  encode   : codes vs exact fp32 greedy residual reference (match rate; mismatches must be near-ties: reconstruction
             error of kernel codes <= reference + eps).
  weight dequant: vs gather reference.
Usage: python experiments/24_kvregime/check_kernels.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from specrq.kvq.kernels import (FORMATS, KVCache, attn_workspace, dense_append, inv_freq_llama,  # noqa: E402
                                lut_ok, ref_attention, ref_dequant_kv, ref_rope, split_plan, vq_attention,
                                vq_dequant_weight, vq_encode)

Hkv, G, D = 8, 4, 128
Hq = Hkv * G


def make_cache(fmt, B, T, gen):
    if fmt in FORMATS:
        d, bits = FORMATS[fmt]
        return KVCache(1, B, Hkv, T, D, "vq", d, gen=gen, bits=bits)
    return KVCache(1, B, Hkv, T, D, fmt, gen=gen)


def check_attn(fmt, B, ctx, M, kst, target_programs, gen, inv, backend="cuda"):
    T = ctx + 8
    cache = make_cache(fmt, B, T, gen)
    qkv = torch.randn(B * M, (Hq + 2 * Hkv) * D, device="cuda", generator=gen).to(torch.bfloat16)
    if cache.mode == "vq":
        vq_encode(qkv, cache, 0, ctx, M, Hq, inv)  # new rows encoded (codes depend on qkv) before attention
    elif cache.mode == "bf16":
        dense_append(qkv, cache, 0, ctx, M, Hq, inv)
    n = ctx + M
    splits, chunk = split_plan(B, Hkv, n, 128, target_programs)
    ws = attn_workspace(B, Hkv, G, M, splits * 16)
    out = torch.empty(B * M, Hq * D, device="cuda", dtype=torch.bfloat16)
    vq_attention(qkv, cache, 0, ctx, M, G, inv, out, ws, splits, chunk, kst, backend=backend)
    K, V = ref_dequant_kv(cache, 0, kst)
    ref = ref_attention(qkv, K, V, ctx, M, Hkv, G, inv)
    err = (out.float() - ref).abs().max().item()
    return {"backend": backend if cache.mode == "vq" else "triton", "format": fmt, "kst": kst, "B": B, "ctx": ctx,
            "M": M, "splits": splits, "chunk": chunk, "max_abs_err": err, "ref_max_abs": ref.abs().max().item(),
            "ok": err < 2e-2}


def stage_codes(cache, kv, s, sl):
    """codes of stage s for token slice sl -> long [B, H, n, NSV] (4-bit planes unpacked)."""
    c = cache.planes(0)[kv][s][:, :, sl].long()
    if cache.bits[s] == 4:
        c = torch.stack([c & 15, c >> 4], -1).flatten(-2)
    return c


def check_encode(fmt, B, M, gen, inv, ctx=100):
    cache = make_cache(fmt, B, ctx + 8, gen)
    d, S = cache.d, cache.S
    qkv = (torch.randn(B * M, (Hq + 2 * Hkv) * D, device="cuda", generator=gen) * 0.7).to(torch.bfloat16)
    vq_encode(qkv, cache, 0, ctx, M, Hq, inv)
    res = {"format": fmt, "B": B, "M": M}
    pos = ctx + torch.arange(M, device="cuda").repeat(B)  # rows ordered (b, m)
    for kv in range(2):
        x = qkv[:, (Hq + kv * Hkv) * D:(Hq + (kv + 1) * Hkv) * D].float().view(B * M, Hkv, D)
        if kv == 0:
            x = ref_rope(x, pos[:, None].expand(B * M, Hkv), inv)
        got = torch.stack([stage_codes(cache, kv, s, slice(ctx, ctx + M)) for s in range(S)])  # [S,B,H,M,NSV]
        got = got.permute(0, 1, 3, 2, 4).reshape(S, B * M, Hkv, -1)
        cbs = cache.planes(0)[2 + kv]
        match, rel_k, rel_r = [], [], []
        for h in range(Hkv):
            xs = x[:, h].reshape(B * M, D // d, d)
            cb = [cbs[s][h].float() for s in range(S)]
            r_, rc = xs.clone(), []
            for s in range(S):  # exact fp32 greedy residual reference
                dist = (cb[s] * cb[s]).sum(-1)[None, None] - 2 * torch.einsum("nkd,cd->nkc", r_, cb[s])
                code = dist.argmin(-1)
                rc.append(code)
                r_ = r_ - cb[s][code]
            rc = torch.stack(rc)
            match.append((rc == got[:, :, h]).float().mean().item())

            def recon(codes):
                return sum(cb[s][codes[s]] for s in range(S))
            rel_k.append(((xs - recon(got[:, :, h])).norm() / xs.norm()).item())
            rel_r.append(((xs - recon(rc)).norm() / xs.norm()).item())
        res["KV"[kv]] = {"code_match": sum(match) / Hkv, "rel_err_kernel": sum(rel_k) / Hkv,
                         "rel_err_ref": sum(rel_r) / Hkv}
    res["ok"] = all(abs(res[c]["rel_err_kernel"] - res[c]["rel_err_ref"]) < 1e-3 for c in "KV")
    return res


def check_wdeq(gen):
    N, Kd, d, L = 1024, 4096, 8, 2
    codes = torch.randint(0, 256, (L, N, Kd // d), dtype=torch.uint8, device="cuda", generator=gen)
    cb = torch.randn(L, 256, d, device="cuda", generator=gen).to(torch.bfloat16)
    sc = (torch.rand(N, device="cuda", generator=gen) + 0.5).to(torch.bfloat16)
    out = []
    for k in (1, 2):
        W = torch.empty(N, Kd, device="cuda", dtype=torch.bfloat16)
        vq_dequant_weight(codes, cb, sc, k, W, d)
        ref = sum(cb[l].float()[codes[l].long()] for l in range(k)).reshape(N, Kd) * sc.float()[:, None]
        out.append({"k": k, "max_abs_err": (W.float() - ref).abs().max().item(),
                    "ref_max_abs": ref.abs().max().item()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--save", default=None, help="results/E6 run name (tools.results.save_result)")
    ap.add_argument("--formats", nargs="+", default=["d8S2", "d8K16x4", "d8K16x2K256", "d16S4", "bf16", "int4",
                                                       "int2"])
    ap.add_argument("--old-backends", nargs="*", default=["mmaw", "mma", "cuda", "triton"])
    a = ap.parse_args()
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(0)
    inv = inv_freq_llama(D)
    res = {"attn": [], "encode": [], "wdeq": None}
    cases = []
    fmts = a.formats
    for (B, ctx) in ((1, 1000), (3, 300), (2, 4100)):
        for M in (1, 2, 3, 5):
            for tp in (8, 1024):
                for f in fmts:
                    if f in FORMATS:
                        ks = [1, 2, 4] if f == "d16S4" else range(1, len(FORMATS[f][1]) + 1)  # v1 kernels: d16 kst 1/2/4
                        cases += [(f, B, ctx, M, k, tp) for k in ks]
                    else:
                        cases.append((f, B, ctx, M, None, tp))
    for c in cases:
        f = c[0]
        if f in FORMATS:
            bes = ["mmawp"] if FORMATS[f][0] == 8 else []
            if FORMATS[f][1] == [8] * len(FORMATS[f][1]):
                bes += list(a.old_backends)
            probe = make_cache(f, 1, 16, gen)
            if lut_ok(probe, c[4], c[3]):
                bes.append("lut")
        else:
            bes = ["triton"]
        for be in bes:
            res["attn"].append(check_attn(*c, gen=gen, inv=inv, backend=be))
    for f in fmts:
        if f in FORMATS:
            for M in (1, 5):
                res["encode"].append(check_encode(f, 4, M, gen, inv))
    res["wdeq"] = check_wdeq(gen)
    worst = {}
    for r in res["attn"]:
        key = f"{r['backend']}_{r['format']}_k{r['kst']}"
        worst[key] = max(worst.get(key, 0), r["max_abs_err"])
    res["attn_worst_by_format"] = worst
    res["attn_all_ok"] = all(r["ok"] for r in res["attn"])
    res["encode_all_ok"] = all(r["ok"] for r in res["encode"])
    print(json.dumps({"attn_worst_by_format": worst, "attn_all_ok": res["attn_all_ok"], "n_attn_cases": len(cases),
                      "encode": res["encode"], "wdeq": res["wdeq"]}, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.save:
        from tools.results import save_result
        print("saved", save_result("E6", a.save, config={
            "setting": "kernel correctness (random codes/codebooks/q; Llama-8B head layout Hkv=8, G=4, D=128)",
            "attn_reference": "dequantize (fp32 stage sum -> bf16) then fp32 SDPA with causal mask over new rows",
            "encode_reference": "exact fp32 greedy residual nearest-codeword", "backends": "lut, mmaw, mma, cuda, triton", "formats": a.formats},
            metrics=res))


if __name__ == "__main__":
    main()
