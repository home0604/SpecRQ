"""E7 magicvq: fit d16 K256 x 2 (0.5 bit per stage; --d / --sizes for other formats, e.g. d4 K16 x 2) nested VQ KV codebooks for bf16 Llama-3.1-8B-Instruct on pile-val
calibration KV (disjoint from PG-19 eval), same recipe as experiments/25_kvtau/calib_kv.py (--chnorm, meta layout,
greedy stage-wise k-means; functions imported read-only from there). Output: /data/lsh/results_accept/magicvq/cb/.
Usage: python experiments/26_magicvq/calib_d16.py
"""
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "25_kvtau"))
from calib_kv import collect, kmeans, rel_mse  # noqa: E402
from kvtau_common import encode_stages, gather_cb, load_model, to_layout  # noqa: E402

OUT = "/data/lsh/results_accept/magicvq/cb"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--post", action="store_true", help="POST-RoPE K (HF layout, as specrq/kvq kernels store it): "
                    "calibration keys rotated at their window position + a random offset in [0, 32768 - seqlen]")
    a = ap.parse_args()
    t0 = time.time()
    d, sizes, layout, seqlen, nsamples, ntr, nho, iters = a.d, a.sizes, "hf" if a.post else "meta", 4096, 16, 24576, 4096, 30
    tag = f"d{d}K{sizes[0]}x{len(sizes)}" if len(set(sizes)) == 1 else f"d{d}K" + "K".join(map(str, sizes))
    if a.post:
        tag += "_post"
    model = load_model("bf16", draft_b=False)
    Ks, Vs = collect(model, nsamples, seqlen)
    if a.post:
        g = torch.Generator().manual_seed(1)
        off = torch.randint(0, 32768 - seqlen, (nsamples,), generator=g)
        pos = (torch.arange(seqlen)[None] + off[:, None]).flatten().cuda()
        cos, sin = model.model.rotary_emb(torch.zeros(1, 1, 1, device="cuda"), pos[None])
        cos, sin = cos[0][:, None].cpu(), sin[0][:, None].cpu()  # [N, 1, 128]
        for l in range(len(Ks)):
            x = Ks[l].float()
            Ks[l] = (x * cos + torch.cat((-x[..., 64:], x[..., :64]), -1) * sin).half()
    del model; torch.cuda.empty_cache()
    keep = torch.ones(Ks[0].shape[0], dtype=torch.bool); keep[::seqlen] = False
    mus, sds = [], []
    for l in range(len(Ks)):
        mu, sd = [], []
        for T in (Ks[l], Vs[l]):
            x = T[keep].float()
            mu.append(x.mean(0)); sd.append(x.std(0).clamp(min=1e-4))
            T.sub_(mu[-1].half()).div_(sd[-1].half())
        mus.append(mu); sds.append(sd)
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(Ks[0].shape[0], generator=gen)
    tr, ho = perm[:ntr], perm[ntr:ntr + nho]
    cbs_all, res = [], []
    for l in range(len(Ks)):
        row = []
        for kv, T in enumerate((Ks[l], Vs[l])):
            def prep(sel):
                x = to_layout(T[sel].cuda().float(), layout)
                x = x.reshape(x.shape[0], x.shape[1], -1, d)
                return x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, d)
            Xtr, Xho = prep(tr), prep(ho)
            cbs, r = [], Xtr.clone()
            for C in sizes:
                cb = kmeans(r, C, iters, gen)
                cbs.append(cb)
                r = r - gather_cb(cb, encode_stages(r, [cb])[0])
            row.append([c.cpu() for c in cbs])
            res.append({"layer": l, "kv": "KV"[kv], "rel_mse": rel_mse(Xho, cbs)})
        cbs_all.append(row)
        if l % 8 == 0:
            print(l, res[-2:], f"{time.time() - t0:.0f}s", flush=True)
    torch.save({"d": d, "sizes": sizes, "layout": layout, "variant": "greedy", "wmode": "bf16", "post_rope": a.post, "draft_levels": [1],
                "cb": cbs_all, "mu": mus, "sd": sds, "per_pos": False}, os.path.join(OUT, f"cb_{tag}_cn_greedy.pt"))
    summ = {kvn: [sum(x["rel_mse"][s] for x in res if x["kv"] == kvn) / len(Ks) for s in range(2)] for kvn in "KV"}
    json.dump({"summary_mean_rel_mse_per_prefix": summ, "per_layer": res, "wall_s": time.time() - t0,
               "nsamples": nsamples, "seqlen": seqlen, "train_tokens": ntr, "heldout_tokens": nho, "iters": iters},
              open(os.path.join(OUT, f"calib_{tag}_cn.json"), "w"), indent=1)
    print("summary", summ, flush=True)


if __name__ == "__main__":
    main()
