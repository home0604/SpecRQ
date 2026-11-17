"""E7 magicvq: attention-aware training of GSRQ KV codebooks (port of VQLLM's attention-KL objective).

VQLLM/vqllm/quantizer/quantize.py defines scaled_dot_attn_scores / attn_softmax_and_kl (KL between the attention
distributions with quantized vs original keys); in the current VQLLM code the KL loss types are commented out of
VQVAEQuantize.forward and the post-init training loop of recipes/full_finetune_single_device.py is commented out
('fast' = init + EM only). This script ports the objective (read-only reference) and trains the GSRQ codebooks
(unit shapes + gains, all stages) layer-locally on pile-val activations of the bf16 model:
  K : KL( softmax(q k^T / sqrt(d)) || softmax(q khat^T / sqrt(d)) ), causal, RoPE applied to k and khat (pre-RoPE VQ)
  V : || P (vhat - v) ||^2 / || P v ||^2 with P the exact causal attention (attention-weighted output error)
  both summed over the nested prefixes (stage 1 and stages 1..S), first `sinks` positions kept exact, as in the draft.
Codes are re-assigned every step with the current codebooks (GSRQ score argmax 2 g x.s - g^2); gradients reach the
selected shapes / gains (assignment is not differentiated). Adam, `steps` steps per layer, one window per step.
Init: an EM-fitted GSRQ file from gsrq.py. Output: same format (+ "trained": loss log).
Usage: python experiments/26_magicvq/gsrq_train.py --init gsrq_D8C256S2 [--steps 150 --lr 2e-3]
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "25_kvtau"))
from gsrq import OUT, from_meta, to_meta  # noqa: E402


@torch.no_grad()
def collect_qkv(model, nsamples, seqlen):
    from src.datautils import get_loaders
    from kvtau_common import MODEL, Fwd, rope
    data = get_loaders("pile_val", nsamples=nsamples, seed=1, seqlen=seqlen, model_path=MODEL)
    f = Fwd(model)
    Q = [[] for _ in range(f.nl)]
    K = [[] for _ in range(f.nl)]
    V = [[] for _ in range(f.nl)]
    for ids in data:
        ids = ids.cuda()
        h = f.mm.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        cos, sin = f.cos_sin(h, pos)
        for l, layer in enumerate(f.mm.layers):
            q, k, v = f.qkv(layer, h)
            Q[l].append(q[0].half().cpu()); K[l].append(k[0].half().cpu()); V[l].append(v[0].half().cpu())
            h = f.post(layer, h, f.sdpa(rope(q, cos, sin), rope(k, cos, sin), v, causal=True))
        cs = (cos[0].float().cpu(), sin[0].float().cpu())
    cat = lambda L: [torch.cat(x) for x in L]  # noqa: E731
    return cat(Q), cat(K), cat(V), cs


def rope_t(x, cos, sin):
    """x [T, H, D] fp32, cos/sin [T, D]."""
    c, s = cos[:, None], sin[:, None]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * c + torch.cat((-x2, x1), -1) * s


def gs_quant(X, shapes, gains, nst):
    """X [P, N, D] fp32 -> list of prefix recons (differentiable wrt shapes / gains); shapes normalized here."""
    out, acc, r = [], torch.zeros_like(X), X
    for s in range(nst):
        S = F.normalize(shapes[s], dim=-1)
        g = gains[s].clamp(min=1e-4)
        with torch.no_grad():
            sc = 2 * torch.bmm(r.detach(), S.transpose(1, 2)) * g[:, None] - g[:, None] ** 2
            c = sc.argmax(-1)  # [P, N]
        sel = torch.gather(S, 1, c[..., None].expand(-1, -1, S.shape[-1])) * torch.gather(g, 1, c)[..., None]
        acc = acc + sel
        r = X - acc
        out.append(acc)
    return out


def to_flat(x, kv, D):
    """[N, H, 128] -> [P, N, D] (K: Meta layout)."""
    if kv == 0:
        x = to_meta(x)
    return x.reshape(x.shape[0], -1, D).transpose(0, 1)


def from_flat(y, kv, H):
    y = y.transpose(0, 1).reshape(y.shape[1], H, 128)
    return from_meta(y) if kv == 0 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="gsrq_D8C256S2")
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--sinks", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    st = torch.load(os.path.join(OUT, f"{a.init}.pt"))
    D, C, S, vnorm = st["D"], st["C"], st["stages"], bool(st.get("vnorm", False))
    from kvtau_common import load_model
    model = load_model("bf16", draft_b=False)
    Q, K, V, (cos, sin) = collect_qkv(model, a.nsamples, a.seqlen)
    del model
    torch.cuda.empty_cache()
    print(f"collected {len(Q)} layers, {Q[0].shape}, {time.time() - t0:.0f}s", flush=True)
    cos, sin = cos.cuda(), sin.cuda()
    T = a.seqlen
    nl = a.layers or len(Q)
    gen = torch.Generator().manual_seed(0)
    causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
    qmask = torch.arange(T, device="cuda") >= a.sinks  # loss over queries that see quantized keys
    log, newcb = [], []
    for l in range(nl):
        row = []
        for kv in range(2):
            shapes = [torch.nn.Parameter(st["cb"][l][kv][s][0].float().cuda()) for s in range(S)]
            gains = [torch.nn.Parameter(st["cb"][l][kv][s][1].float().cuda()) for s in range(S)]
            opt = torch.optim.Adam(shapes + gains, lr=a.lr)
            hist = []
            init_state = [t.detach().clone() for t in shapes + gains]
            evals = {}
            for it in range(-1, a.steps + 2):
                # it = -1: held-out eval before training; it = steps + 1: held-out eval after; window nsamples-1 is held out
                ev = it in (-1, a.steps + 1)
                w = a.nsamples - 1 if ev else int(torch.randint(0, a.nsamples - 1, (1,), generator=gen))
                sl = slice(w * T, (w + 1) * T)
                q = Q[l][sl].cuda().float()  # [T, 32, 128]
                k = K[l][sl].cuda().float()  # [T, 8, 128]
                v = V[l][sl].cuda().float()
                H = k.shape[1]
                G = q.shape[1] // H
                qr = rope_t(q, cos, sin).view(T, H, G, 128).permute(1, 2, 0, 3) / math.sqrt(128)  # [H,G,T,128]
                kr = rope_t(k, cos, sin).permute(1, 0, 2)  # [H,T,128]
                logits_ref = torch.matmul(qr, kr[:, None].transpose(-1, -2)).masked_fill(~causal, -1e9)
                lp_ref = logits_ref.log_softmax(-1)  # [H,G,T,T]
                x = v if kv == 1 else k
                nrm = None
                if kv == 1 and vnorm:
                    nrm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    x = x / nrm
                X = to_flat(x, kv, D)
                with torch.set_grad_enabled(not ev):
                    recs = gs_quant(X, shapes, gains, S)
                loss = 0.0
                parts = []
                for rc in recs:  # nested prefixes 1..S
                    xh = from_flat(rc, kv, H)
                    if nrm is not None:
                        xh = xh * nrm
                    xh = torch.cat([(v if kv == 1 else k)[:a.sinks], xh[a.sinks:]])
                    if kv == 0:
                        kh = rope_t(xh, cos, sin).permute(1, 0, 2)
                        lg = torch.matmul(qr, kh[:, None].transpose(-1, -2)).masked_fill(~causal, -1e9)
                        kl = (lp_ref.exp() * (lp_ref - lg.log_softmax(-1))).sum(-1)  # [H,G,T]
                        lpart = kl[..., qmask].mean()
                    else:
                        P = lp_ref.exp()
                        o_ref = torch.matmul(P, v.permute(1, 0, 2)[:, None])
                        o_hat = torch.matmul(P, xh.permute(1, 0, 2)[:, None])
                        lpart = ((o_hat - o_ref) ** 2)[..., qmask, :].sum() / (o_ref ** 2)[..., qmask, :].sum()
                    loss = loss + lpart
                    parts.append(lpart.item())
                if ev:
                    evals["before" if it == -1 else "after"] = parts
                    continue
                if it % 25 == 0:
                    hist.append({"it": it, "loss_prefix": parts})
                if it == a.steps:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            keep = sum(evals["after"]) < sum(evals["before"])
            if not keep:  # held-out loss did not improve: keep the EM init for this (layer, K/V)
                with torch.no_grad():
                    for t, t0_ in zip(shapes + gains, init_state):
                        t.copy_(t0_)
            row.append([(F.normalize(sh.detach(), dim=-1).half().cpu(), g.detach().clamp(min=1e-4).half().cpu())
                        for sh, g in zip(shapes, gains)])
            log.append({"layer": l, "kv": "KV"[kv], "hist": hist, "heldout": evals, "kept_trained": keep})
            del shapes, gains, opt
        newcb.append(row)
        print(l, [(x["kv"], [round(v, 4) for v in x["heldout"]["before"]], [round(v, 4) for v in x["heldout"]["after"]],
                   x["kept_trained"]) for x in log[-2:]],
              f"{time.time() - t0:.0f}s", flush=True)
    for l in range(nl, len(st["cb"])):
        newcb.append(st["cb"][l])
    tag = f"{a.init}_kl" if nl == len(st["cb"]) else f"{a.init}_kl_smoke"
    torch.save({**{k: v for k, v in st.items() if k != "cb"}, "cb": newcb,
                "method": st["method"] + " + attention-aware training (K: attn KL, V: attn-weighted MSE)",
                "train_args": vars(a)}, os.path.join(OUT, f"{tag}.pt"))
    json.dump({"args": vars(a), "log": log, "wall_s": time.time() - t0},
              open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    print("saved", tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
