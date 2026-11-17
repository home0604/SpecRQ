"""E6 kvtau step 1: fit nested VQ KV codebooks on pile-val KV activations of Llama-3.1-8B-Instruct.

Collects pre-RoPE K and V (bf16 KV forward, weights = bf16 or the 2-bit k2 target) on pile_val nsamples x seqlen
windows (calibration split, disjoint from wikitext2 / c4 / pg19 eval), then per (layer, K/V, kv-head) fits
  greedy : stage-wise k-means on residuals (greedy RQ), and
  refine : `--refine_rounds` rounds of [greedy re-encode -> per-stage closed-form codebook update minimizing the
           prefix-weighted error sum_p w_p ||x - recon_p||^2 (weights per format in FORMATS)], init = greedy.
Writes OUTROOT/<wmode>/cb_<fmt>_<variant>.pt and a JSON of held-out relative MSE per prefix.
Usage: python experiments/25_kvtau/calib_kv.py --wmode w2 [--formats a_d8K256x2 ...] [--layout meta]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtau_common import FORMATS, MODEL, OUTROOT, Fwd, encode_stages, gather_cb, load_model, rope, set_weights, \
    to_layout  # noqa: E402


@torch.no_grad()
def collect(model, nsamples, seqlen):
    from src.datautils import get_loaders
    data = get_loaders("pile_val", nsamples=nsamples, seed=0, seqlen=seqlen, model_path=MODEL)
    f = Fwd(model)
    K = [[] for _ in range(f.nl)]
    V = [[] for _ in range(f.nl)]
    for ids in data:
        ids = ids.cuda()
        h = f.mm.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        cos, sin = f.cos_sin(h, pos)
        for l, layer in enumerate(f.mm.layers):
            q, k, v = f.qkv(layer, h)
            K[l].append(k[0].half().cpu()); V[l].append(v[0].half().cpu())
            h = f.post(layer, h, f.sdpa(rope(q, cos, sin), rope(k, cos, sin), v, causal=True))
    return [torch.cat(x) for x in K], [torch.cat(x) for x in V]


def kmeans(X, C, iters, gen):
    """X [H, N, d] fp32 -> centers [H, C, d] (Lloyd, random-sample init, empty clusters re-seeded)."""
    H, N, d = X.shape
    idx = torch.stack([torch.randperm(N, generator=gen, device="cpu")[:C] for _ in range(H)]).to(X.device)
    cb = torch.gather(X, 1, idx[..., None].expand(-1, -1, d)).clone()
    for _ in range(iters):
        c = encode_stages(X, [cb])[0]
        s = torch.zeros_like(cb).scatter_add_(1, c[..., None].expand(-1, -1, d), X)
        n = torch.zeros(H, C, device=X.device).scatter_add_(1, c, torch.ones_like(c, dtype=torch.float))
        new = s / n.clamp(min=1)[..., None]
        empty = n == 0
        if empty.any():
            r = torch.randint(0, N, (H, C), generator=gen).to(X.device)
            new = torch.where(empty[..., None], torch.gather(X, 1, r[..., None].expand(-1, -1, d)), new)
        cb = new
    return cb


def rel_mse(X, cbs):
    codes = encode_stages(X, cbs)
    acc, out, nx = torch.zeros_like(X), [], (X * X).sum().item()
    for s, cb in enumerate(cbs):
        acc = acc + gather_cb(cb, codes[s])
        out.append(((X - acc) ** 2).sum().item() / nx)
    return out


def refine(X, cbs, wts, rounds):
    cbs = [c.clone() for c in cbs]
    S, H, d = len(cbs), X.shape[0], X.shape[-1]
    for _ in range(rounds):
        codes = encode_stages(X, cbs)
        sel = [gather_cb(cbs[s], codes[s]) for s in range(S)]
        for s in range(S):
            num = torch.zeros_like(cbs[s]); wsum = 0.0
            for p, w in wts.items():  # prefix p uses stages 0..p-1
                if p <= s or w == 0:
                    continue
                r = X - sum(sel[t] for t in range(p) if t != s)
                num += w * torch.zeros_like(cbs[s]).scatter_add_(1, codes[s][..., None].expand(-1, -1, d), r)
                wsum += w
            n = torch.zeros(H, cbs[s].shape[1], device=X.device).scatter_add_(
                1, codes[s], torch.ones_like(codes[s], dtype=torch.float))
            new = num / (wsum * n.clamp(min=1))[..., None]
            cbs[s] = torch.where((n > 0)[..., None], new, cbs[s])
            sel[s] = gather_cb(cbs[s], codes[s])
    return cbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmode", choices=["bf16", "w2"], required=True)
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--layouts", nargs="+", default=["meta", "hf"], help="first = used for saved codebooks; "
                    "others only get a held-out MSE comparison for format a")
    ap.add_argument("--nsamples", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--train_tokens", type=int, default=24576)
    ap.add_argument("--heldout_tokens", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--refine_rounds", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="debug: only the first n layers")
    ap.add_argument("--chnorm", action="store_true", help="static per-(layer, K/V, kv-head, channel) standardization "
                    "(mean / std over calibration tokens, excluding each window's position 0) before VQ; foldable "
                    "into q (K) and the attention output (V), so the VQ kernel cost is unchanged")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--per_pos", action="store_true", help="quality probe: separate codebook per sub-vector position "
                    "(NOT the specrq/kvq granularity; 16x codebook storage)")
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.join(OUTROOT, a.wmode); os.makedirs(out, exist_ok=True)
    model = load_model(a.wmode, draft_b=False)
    set_weights(model, "k2" if a.wmode == "w2" else "bf16")
    Ks, Vs = collect(model, a.nsamples, a.seqlen)
    del model; torch.cuda.empty_cache()
    print(f"collected {Ks[0].shape} per layer, {time.time() - t0:.0f}s", flush=True)
    nl = a.layers or len(Ks)
    mus, sds = None, None
    if a.chnorm:
        keep = torch.ones(Ks[0].shape[0], dtype=torch.bool); keep[::a.seqlen] = False
        mus, sds = [], []
        for l in range(len(Ks)):
            mu, sd = [], []
            for T in (Ks[l], Vs[l]):
                x = T[keep].float()
                mu.append(x.mean(0)); sd.append(x.std(0).clamp(min=1e-4))  # [H, 128]
                T.sub_(mu[-1].half()).div_(sd[-1].half())
            mus.append(mu); sds.append(sd)
    gen = torch.Generator().manual_seed(0)
    Ntok = Ks[0].shape[0]
    perm = torch.randperm(Ntok, generator=gen)
    tr, ho = perm[:a.train_tokens], perm[a.train_tokens:a.train_tokens + a.heldout_tokens]
    report = {"wmode": a.wmode, "nsamples": a.nsamples, "seqlen": a.seqlen, "train_tokens": a.train_tokens,
              "heldout_tokens": a.heldout_tokens, "iters": a.iters, "refine_rounds": a.refine_rounds, "formats": {}}
    for fmt in a.formats:
        d, sizes, dlev, wts = FORMATS[fmt]
        for layout in (a.layouts if fmt == a.formats[0] else a.layouts[:1]):
            res = {"greedy": [], "refine": []}
            cbs_all = {"greedy": [], "refine": []}
            for l in range(nl):
                row = {"greedy": [None, None], "refine": [None, None]}
                for kv, T in enumerate((Ks[l], Vs[l])):
                    def prep(sel):
                        x = to_layout(T[sel].cuda().float(), layout)  # [n, H, 128]
                        x = x.reshape(x.shape[0], x.shape[1], -1, d)  # [n, H, NSV, d]
                        if a.per_pos:  # one codebook per (head, sub-vector position): groups [H * NSV, n, d]
                            return x.permute(1, 2, 0, 3).reshape(-1, x.shape[0], d)
                        return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, d)
                    Xtr, Xho = prep(tr), prep(ho)
                    cbs, r = [], Xtr.clone()
                    for C in sizes:
                        cb = kmeans(r, C, a.iters, gen)
                        cbs.append(cb)
                        r = r - gather_cb(cb, encode_stages(r, [cb])[0])
                    rcbs = refine(Xtr, cbs, wts, a.refine_rounds)
                    for var, cc in (("greedy", cbs), ("refine", rcbs)):
                        row[var][kv] = cc
                        res[var].append({"layer": l, "kv": "KV"[kv], "rel_mse": rel_mse(Xho, cc)})
                for var in ("greedy", "refine"):
                    cbs_all[var].append([[c.cpu() for c in row[var][kv]] for kv in range(2)])
                if l % 8 == 0:
                    print(fmt, layout, l, [x for x in res["greedy"][-2:]], [x["rel_mse"] for x in res["refine"][-2:]],
                          f"{time.time() - t0:.0f}s", flush=True)
            summ = {}
            for var in ("greedy", "refine"):
                for kvn in "KV":
                    xs = [x["rel_mse"] for x in res[var] if x["kv"] == kvn]
                    summ[f"{var}_{kvn}_mean_rel_mse_per_prefix"] = [sum(x[s] for x in xs) / len(xs) for s in range(len(sizes))]
                if layout == a.layouts[0] and nl == len(Ks):
                    torch.save({"d": d, "sizes": sizes, "layout": layout, "variant": var, "wmode": a.wmode,
                                "draft_levels": dlev, "refine_weights": wts, "cb": cbs_all[var],
                                "mu": mus, "sd": sds, "per_pos": a.per_pos},
                               os.path.join(out, f"cb_{fmt}{a.suffix}_{var}.pt"))
            report["formats"][f"{fmt}|{layout}"] = {"summary": summ, "per_layer": res}
            print(fmt, layout, json.dumps(summ), flush=True)
    report["wall_s"] = time.time() - t0
    json.dump(report, open(os.path.join(out, f"calib_report{a.suffix}{'_dbg' if a.layers else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
