"""E10 phase 1, static-bound (SB) knob: gap = bound - max score per (query, q-head, split), log2 units (the kernel's
exp2 domain), where bound = sum_{stage<KST} sum_j max_c LUT[stage][j][c] (fp16 LUT entries like the kernel).
p = exp2(s - bound) underflows in fp32 for gap > ~126 (normal range; denormals to ~149).

Data:
  real    : Llama-3.1-8B-Instruct activations (/data/lsh/ckpts/magicvq_v6/acts: pre-RoPE K, Q at sampled qpos),
            K RoPE'd and encoded (greedy RQ, chnorm) with the E7 draft-trained t2w codebook (post-RoPE, HF layout),
            q RoPE'd at qpos; VQ token range [16, qpos - 127] as in e2e_real (sinks + window are bf16 elsewhere)
  unif    : same q / codebooks, uniform random codes (what the bench uses)
  synth   : bench_p0 synthetic q / codebooks (randn), uniform random codes
Usage: python sb_gap.py --out results/E10/lut_tiling_p1_sbgap.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25_kvtau"))
from kvtau_common import encode_stages  # noqa: E402

ACTS = "/data/lsh/ckpts/magicvq_v6/acts"
CBF = "/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt"
D, G, HKV = 128, 4, 8
DSC = 128 ** -0.5 * 1.4426950408889634
CHUNKS = (512, 2048, 8192, 1 << 30)


def rope(x, cos, sin):
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    return x * cos + torch.cat((-x2, x1), -1) * sin


def gaps_for(qp, cb, codes, lo, hi, kst):
    """qp [Hq, D] scaled q; cb [Hkv, S, 256, 8]; codes [S, Hkv, T, 16] long. Returns dict chunk -> gap tensor
    [Hq, nchunks] (log2 units), plus true max - bound for sanity."""
    Hq = qp.shape[0]
    q = qp.view(HKV, G, 16, 8)
    lut = torch.einsum("hgjd,hscd->hsjcg", q, cb[:, :kst]).half().float()  # [Hkv, kst, 16, 256, G]
    bound = lut.amax(dim=3).sum(dim=(1, 2))  # [Hkv, G]
    s = torch.zeros(HKV, hi - lo, G, device=qp.device)
    for st in range(kst):
        for j in range(16):
            idx = codes[st, :, lo:hi, j]  # [Hkv, n]
            s += torch.gather(lut[:, st, j], 1, idx[..., None].expand(-1, -1, G))
    out = {}
    n = hi - lo
    for ch in CHUNKS:
        ns = max(1, -(-n // ch))
        pad = ns * min(ch, n) - n
        sp = torch.nn.functional.pad(s.transpose(1, 2), (0, pad), value=-float("inf"))  # [Hkv, G, ns*ch]
        mx = sp.view(HKV, G, ns, -1).amax(-1)
        out[ch] = (bound[..., None] - mx).reshape(Hq, ns)
    return out


def summarize(xs):
    x = torch.cat([t.flatten() for t in xs]).double()
    x = x[torch.isfinite(x)]
    qs = torch.quantile(x[torch.randperm(len(x))[:1 << 24]], torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0], dtype=x.dtype))
    return dict(n=len(x), min=qs[0].item(), p01=qs[1].item(), median=qs[2].item(), p99=qs[3].item(),
                max=x.max().item(), frac_gt_64=(x > 64).double().mean().item(),
                frac_gt_100=(x > 100).double().mean().item(), frac_gt_126=(x > 126).double().mean().item(),
                neg=(x < -1e-3).double().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", type=int, nargs="+", default=[0, 5, 11, 13, 15])
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2, 8, 16, 24, 31])
    ap.add_argument("--nq", type=int, default=24)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    st = torch.load(CBF, map_location="cpu", weights_only=False)
    rp = torch.load(f"{ACTS}/rope.pt")
    cos, sin = rp["cos"].to(dev), rp["sin"].to(dev)
    acc = {(d, k, ch): [] for d in ("real", "unif") for k in (1, 2) for ch in CHUNKS}
    per_layer = {}
    for l in a.layers:
        cb = torch.stack([c.float() for c in st["cb"][l][0]], 1).to(dev)  # [Hkv, S, 256, 8]
        mu, sd = st["mu"][l][0].to(dev).float(), st["sd"][l][0].to(dev).float()  # [Hkv, D]
        for s_ in a.seqs:
            x = torch.load(f"{ACTS}/s{s_}_L{l}.pt")
            K = x["K"].to(dev).float()  # [T, Hkv, D] pre-RoPE
            T = K.shape[0]
            pos = torch.arange(T, device=dev)
            Kp = rope(K, cos[pos][:, None], sin[pos][:, None])
            xl = ((Kp - mu) / sd).permute(1, 0, 2).reshape(HKV, T * 16, 8)
            codes = torch.stack(encode_stages(xl, [cb[:, i] for i in range(2)]), 0).view(2, HKV, T, 16)
            ucodes = torch.randint(0, 256, codes.shape, device=dev)
            sel = torch.randperm(x["Q"].shape[0])[:a.nq]
            for qi in sel.tolist():
                qpos = int(x["qpos"][qi])
                hi = qpos - 127
                if hi - 16 < 1024:
                    continue
                q = x["Q"][qi].to(dev).float()  # [Hq, D]
                qr = rope(q, cos[qpos], sin[qpos]).to(torch.bfloat16).float()
                qp = qr * sd.repeat_interleave(G, 0) * DSC
                for k in (1, 2):
                    for nm, cd in (("real", codes), ("unif", ucodes)):
                        gp = gaps_for(qp, cb, cd, 16, hi, k)
                        for ch, t in gp.items():
                            acc[(nm, k, ch)].append(t.cpu())
                            per_layer.setdefault((nm, k, ch, l), []).append(t.cpu())
        print("layer", l, "done", flush=True)
    res = dict(cbfile=CBF, acts=ACTS, seqs=a.seqs, layers=a.layers, nq=a.nq, chunks=list(CHUNKS),
               note="gap = bound - max score in split, log2 units; fp32 exp2 underflow at gap > ~126",
               summary={f"{d}_k{k}_c{ch if ch < 1 << 30 else 'all'}": summarize(v) for (d, k, ch), v in acc.items()},
               per_layer={f"{d}_k{k}_c{ch if ch < 1 << 30 else 'all'}_L{l}": summarize(v)
                          for (d, k, ch, l), v in per_layer.items()})
    # synthetic bench data
    sys.path.insert(0, str(HERE))
    import check_p1 as CK
    syn = {}
    for k in (1, 2):
        c = CK.make_case(4, 2, 16384, k, False, seed=7)
        rot = CK.rot_q(c)
        qp = (rot * c["sd"][0][None, :, None, :] * DSC).view(4, 2 * G, D)
        cbk = c["cb"][0].float()  # [Hkv, kst, C, 8]
        for ch in CHUNKS:
            vals = []
            for b in range(4):
                q = qp[b].view(2, G, 16, 8)
                lut = torch.einsum("hgjd,hscd->hsjcg", q, cbk).half().float()
                bound = lut.amax(dim=3).sum(dim=(1, 2))
                sc = torch.zeros(2, 16384, G, device=dev)
                for st_ in range(k):
                    for j in range(16):
                        idx = c["planes"][0, st_, b, :, :, j].long()
                        sc += torch.gather(lut[:, st_, j], 1, idx[..., None].expand(-1, -1, G))
                n = 16384
                cc = min(ch, n)
                mx = sc.transpose(1, 2).reshape(2, G, n // cc, cc).amax(-1)
                vals.append((bound[..., None] - mx).cpu())
            syn[f"synth_k{k}_c{ch if ch < 1 << 30 else 'all'}"] = summarize(vals)
    res["summary"].update(syn)
    for kk, v in res["summary"].items():
        print(kk, {x: round(y, 3) if isinstance(y, float) else y for x, y in v.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
