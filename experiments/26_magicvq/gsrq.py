"""E7 magicvq: GSRQ (gain-shape residual VQ, the lab's VQLLM quantizer) for Llama-3.1-8B-Instruct KV, refit on pile-val.

Port of VQLLM/vqllm/quantizer/quantize.py (VQVAEQuantize.init_codebook, fast init + EM path, no STE / backprop),
vectorized over the sub-vector positions (read-only reference; VQLLM/ is not modified or imported):
  * KV of one layer flattened per token to Hkv * 128 dims (K pre-RoPE, mapped HF -> Meta RoPE-pair layout as in
    CLAUDE.md; V unchanged), split into P = Hkv * 128 / D sub-vectors; SubVec1: one codebook per sub-vector position
    (num_subvectors_per_codebook = 1) and per residual stage, C unit-norm shapes (fp16) + per-(position, entry) gains
    (fp16); codeword = gain * shape.
  * init: k-means++ on the raw sub-vectors (<= 50000 samples) -> normalize -> assign by argmax x.s -> gain_k =
    max(1e-4, mean x.s_k); EM (<= 100 iterations, rel. objective tol 1e-4): assign k = argmax 2 g x.s - g^2, shape_k =
    normalize(sum of x / |x| over assigned), gain_k = mean x.s_k (current shapes); stage s + 1 is fit on the residual
    of stages 1..s (independent stages, num_shared_stages = 0).
  * no channel normalization (VQLLM has none); attention sinks / recent window are handled by the caller.
Usage: python experiments/26_magicvq/gsrq.py --D 8 --C 256 --stages 2      (writes OUTROOT/cb/gsrq_D8C256S2.pt)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "25_kvtau"))
OUT = "/data/lsh/results_accept/magicvq/cb"


def to_meta(x):
    """[..., 128] HF k_proj order -> Meta / torchtune RoPE-pair order (x.view(..., 2, 64).transpose(-1, -2))."""
    return x.unflatten(-1, (2, 64)).transpose(-1, -2).flatten(-2)


def from_meta(x):
    return x.unflatten(-1, (64, 2)).transpose(-1, -2).flatten(-2)


@torch.no_grad()
def kmpp(X, K, gen, max_samples=50000):
    """batched k-means++ (squared L2) on raw data. X [P, N, D] -> centers [P, K, D]."""
    P, N, D = X.shape
    if N > max_samples:
        X = X[:, torch.randperm(N, generator=gen)[:max_samples].to(X.device)]
        N = max_samples
    ar = torch.arange(P, device=X.device)
    idx = torch.randint(0, N, (P,), generator=gen).to(X.device)
    C = torch.empty(P, K, D, device=X.device)
    C[:, 0] = X[ar, idx]
    mind = ((X - C[:, :1]) ** 2).sum(-1)  # [P, N]
    for j in range(1, K):
        s = mind.clamp(min=0)
        tot = s.sum(-1, keepdim=True)
        probs = torch.where(tot > 1e-8, s / tot.clamp(min=1e-30), torch.full_like(s, 1.0 / N))
        idx = torch.multinomial(probs, 1).squeeze(1)
        C[:, j] = X[ar, idx]
        mind = torch.minimum(mind, ((X - C[:, j:j + 1]) ** 2).sum(-1))
    return C


@torch.no_grad()
def em_fit(X, C, gen, max_iter=100, tol=1e-4):
    """X [P, N, D] fp32 -> shapes [P, C, D] (unit), gains [P, C]; GSRQ fast init + EM."""
    P, N, D = X.shape
    S = torch.nn.functional.normalize(kmpp(X, C, gen), dim=-1)
    proj = torch.bmm(X, S.transpose(1, 2))  # [P, N, C]
    lab = proj.argmax(-1)
    g = gain_from(proj, lab, C, torch.full((P, C), 1e-4, device=X.device))
    del proj
    xn = torch.nn.functional.normalize(X, dim=-1)
    prev = None
    for it in range(max_iter):
        proj = torch.bmm(X, S.transpose(1, 2))
        gq = g.clamp(min=1e-4)
        sc = 2 * proj * gq[:, None] - gq[:, None] ** 2
        mx, lab = sc.max(-1)
        obj = mx.sum(-1)  # [P]
        acc = torch.zeros_like(S).scatter_add_(1, lab[..., None].expand(-1, -1, D), xn)
        g = gain_from(proj, lab, C, g)
        nrm = acc.norm(dim=-1, keepdim=True)
        S = torch.nn.functional.normalize(torch.where(nrm < 1e-7, S, acc), dim=-1)
        del proj, sc
        if prev is not None and bool(((obj - prev).abs() < tol * prev.abs().clamp(min=1.0)).all()):
            break
        prev = obj
    return S.half(), g.clamp(min=1e-4).half(), it + 1


def gain_from(proj, lab, C, old):
    pa = proj.gather(-1, lab[..., None]).squeeze(-1)
    s = torch.zeros_like(old).scatter_add_(1, lab, pa)
    n = torch.zeros_like(old).scatter_add_(1, lab, torch.ones_like(pa))
    return torch.where(n > 0, (s / n.clamp(min=1)).clamp(min=1e-4), old)


@torch.no_grad()
def gs_encode(X, S, g, chunk=4096):
    """X [P, N, D] fp32, S [P, C, D], g [P, C] -> codes [P, N], recon [P, N, D] (argmax 2 g x.s - g^2)."""
    Sf, gf = S.float(), g.float()
    codes = torch.empty(X.shape[:2], dtype=torch.long, device=X.device)
    rec = torch.empty_like(X)
    for i in range(0, X.shape[1], chunk):
        x = X[:, i:i + chunk]
        sc = 2 * torch.bmm(x, Sf.transpose(1, 2)) * gf[:, None] - gf[:, None] ** 2
        c = sc.argmax(-1)
        codes[:, i:i + chunk] = c
        rec[:, i:i + chunk] = torch.gather(Sf, 1, c[..., None].expand(-1, -1, Sf.shape[-1])) * torch.gather(gf, 1, c)[..., None]
    return codes, rec


class GSQuant:
    """Fake-quant interface compatible with kvtau KVQuant (quant / quant_pos / keep_first / sizes / d)."""

    def __init__(self, path, dev="cuda"):
        st = torch.load(path)
        self.d, self.sizes = st["D"], [st["C"]] * st["stages"]
        self.S = st["stages"]
        self.cb = [[[(s.to(dev), g.to(dev)) for s, g in st["cb"][l][kv]] for kv in range(2)] for l in range(len(st["cb"]))]
        self.keep_first = 0
        self.meta = {k: v for k, v in st.items() if k != "cb"}

    def quant_pos(self, l, kv, x, levels, p0):
        out = self.quant(l, kv, x, levels)
        n = min(self.keep_first - p0, x.shape[1])
        if n > 0:
            for lv in out:
                out[lv][:, :n] = x[:, :n]
        return out

    @torch.no_grad()
    def quant(self, l, kv, x, levels):
        B, T, H, Dh = x.shape
        xf = x.float()
        if kv == 0:
            xf = to_meta(xf)
        X = xf.reshape(B * T, H * Dh // self.d, self.d).transpose(0, 1).contiguous()  # [P, N, D]
        out, acc = {}, torch.zeros_like(X)
        r = X
        for s in range(max(levels)):
            S, g = self.cb[l][kv][s]
            _, rec = gs_encode(r, S, g)
            acc = acc + rec
            r = X - acc
            if s + 1 in levels:
                y = acc.transpose(0, 1).reshape(B, T, H, Dh)
                if kv == 0:
                    y = from_meta(y)
                out[s + 1] = y.to(x.dtype)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--train_tokens", type=int, default=24576)
    ap.add_argument("--heldout_tokens", type=int, default=4096)
    a = ap.parse_args()
    t0 = time.time()
    from calib_kv import collect  # kvtau (read-only): pile_val nsamples x seqlen, bf16 forward, pre-RoPE K + V
    from kvtau_common import load_model
    model = load_model("bf16", draft_b=False)
    Ks, Vs = collect(model, a.nsamples, a.seqlen)
    del model
    torch.cuda.empty_cache()
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(Ks[0].shape[0], generator=gen)
    tr, ho = perm[:a.train_tokens], perm[a.train_tokens:a.train_tokens + a.heldout_tokens]
    D = a.D
    cbs, rep = [], []
    for l in range(len(Ks)):
        row = []
        for kv, T in enumerate((Ks[l], Vs[l])):
            def prep(sel):
                x = T[sel].cuda().float()
                if kv == 0:
                    x = to_meta(x)
                return x.reshape(x.shape[0], -1, D).transpose(0, 1).contiguous()  # [P, N, D]
            Xtr, Xho = prep(tr), prep(ho)
            st, r, rh, acc_h, its = [], Xtr, Xho, torch.zeros_like(Xho), []
            mse = []
            for s in range(a.stages):
                S, g, it = em_fit(r, a.C, gen)
                st.append((S.cpu(), g.cpu()))
                its.append(it)
                _, rec = gs_encode(r, S, g)
                r = r - rec
                _, rech = gs_encode(rh, S, g)
                rh = rh - rech
                mse.append((rh ** 2).sum().item() / (Xho ** 2).sum().item())
            row.append(st)
            rep.append({"layer": l, "kv": "KV"[kv], "rel_mse_heldout": mse, "em_iters": its})
        cbs.append(row)
        if l % 4 == 0:
            print(l, rep[-2:], f"{time.time() - t0:.0f}s", flush=True)
    tag = f"gsrq_D{D}C{a.C}S{a.stages}"
    torch.save({"D": D, "C": a.C, "stages": a.stages, "layout": "meta(K)", "method": "GSRQ (VQLLM port, SubVec1)",
                "cb": cbs}, os.path.join(OUT, f"{tag}.pt"))
    summ = {kvn: [sum(x["rel_mse_heldout"][s] for x in rep if x["kv"] == kvn) / len(Ks) for s in range(a.stages)]
            for kvn in "KV"}
    json.dump({"summary_mean_rel_mse_heldout": summ, "bits_per_stage": math.log2(a.C) / D, "per_layer": rep,
               "wall_s": time.time() - t0, **vars(a)}, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print("summary", summ, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
