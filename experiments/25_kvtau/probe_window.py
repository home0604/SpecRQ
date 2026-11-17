"""E6 kvtau probe: target PPL of a 2-bit VQ KV cache with the sink token and a bf16 recent window of R tokens
(query i reads bf16 K/V for j > i - R, VQ K/V otherwise; KIVI-style residual window). Teacher-forced, fp32 attention.
Usage: python experiments/25_kvtau/probe_window.py --wmode w2 --cb a_d8K256x2_cn_greedy --windows 0 32 128 --nwin 10
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtau_common import MODEL, OUTROOT, Fwd, KVQuant, densify, load_model, rope, set_weights  # noqa: E402


@torch.no_grad()
def attn_window(f, q, kq, vq, kr, vr, R, qchunk=512):
    _, T, H, D = q.shape
    Hkv, Gq = f.Hkv, f.G
    Kq, Vq = kq[0].float().permute(1, 0, 2), vq[0].float().permute(1, 0, 2)
    Kr, Vr = kr[0].float().permute(1, 0, 2), vr[0].float().permute(1, 0, 2)
    out = torch.empty_like(q)
    for i0 in range(0, T, qchunk):
        i1 = min(T, i0 + qchunk); Tq = i1 - i0
        qi = torch.arange(i0, i1, device=q.device)
        qq = (q[0, i0:i1].float() / math.sqrt(D)).view(Tq, Hkv, Gq, D).permute(1, 2, 0, 3)
        jj = torch.arange(i1, device=q.device)
        recent = (jj[None, :] > qi[:, None] - R)[None, None]
        sq = torch.matmul(qq, Kq[:, None, :i1].transpose(-1, -2))
        sr = torch.matmul(qq, Kr[:, None, :i1].transpose(-1, -2))
        s = torch.where(recent, sr, sq).masked_fill((jj[None, :] > qi[:, None])[None, None], float("-inf"))
        p = s.softmax(-1)
        o = torch.matmul(p * (~recent), Vq[:, None, :i1]) + torch.matmul(p * recent, Vr[:, None, :i1])
        out[0, i0:i1] = o.permute(2, 0, 1, 3).reshape(Tq, H, D).to(q.dtype)
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmode", default="w2")
    ap.add_argument("--cb", nargs="+", default=["a_d8K256x2_cn_greedy"])
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 32, 128])
    ap.add_argument("--keep_first", type=int, default=1)
    ap.add_argument("--nwin", type=int, default=10)
    ap.add_argument("--seqlen", type=int, default=4096)
    a = ap.parse_args()
    m = load_model(a.wmode, draft_b=False)
    if a.wmode == "w2":
        densify(m)
    set_weights(m, "k2" if a.wmode == "w2" else "bf16")
    f = Fwd(m)
    from src.datautils import get_loaders
    enc = get_loaders("wikitext2", seed=0, model_path=MODEL, seqlen=a.seqlen, eval_mode=True)
    res = {}
    for cb in a.cb:
        kvq = KVQuant(os.path.join(OUTROOT, a.wmode, f"cb_{cb}.pt"))
        kvq.keep_first = a.keep_first
        for R in a.windows:
            nll, n = 0.0, 0
            for i in range(a.nwin):
                ids = enc[:, i * a.seqlen:(i + 1) * a.seqlen].cuda()
                pos = torch.arange(ids.shape[1], device=ids.device)
                h = f.mm.embed_tokens(ids)
                cos, sin = f.cos_sin(h, pos)
                for l, layer in enumerate(f.mm.layers):
                    q, k, v = f.qkv(layer, h)
                    kq = kvq.quant_pos(l, 0, k, [kvq.S], 0)[kvq.S]
                    vq = kvq.quant_pos(l, 1, v, [kvq.S], 0)[kvq.S]
                    o = attn_window(f, rope(q, cos, sin), rope(kq, cos, sin), vq, rope(k, cos, sin), v, R)
                    h = f.post(layer, h, o)
                for j in range(0, ids.shape[1] - 1, 1024):
                    lt = f.logits(h[0, j:min(j + 1024, ids.shape[1] - 1)])
                    nll += F.cross_entropy(lt, ids[0, j + 1:j + 1 + lt.shape[0]], reduction="sum").item(); n += lt.shape[0]
            res[f"{cb}|keep{a.keep_first}|R{R}"] = math.exp(nll / n)
            print(cb, "R", R, "wt2 ppl", res[f"{cb}|keep{a.keep_first}|R{R}"], flush=True)
    json.dump({"wmode": a.wmode, "nwin": a.nwin, "seqlen": a.seqlen, "wt2_ppl": res},
              open(os.path.join(OUTROOT, a.wmode, "probe_window.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
