"""E7 magicvq v6 step 2: draft-specific (attention-output) training of the 'p' RQ KV codebooks, layer-local.

Format kept EXACTLY as 'p' (cb_d8K256x2_post_cn_greedy.pt, the format of the v4b/v5 kernels): post-RoPE K (HF layout)
and V, per-(layer, K/V, kv-head) channel standardisation (mu, sd [Hkv, 128]), d = 8 sub-vectors, per-(layer, K/V,
kv-head) codebooks of 256 x 8 per stage, 2 stages, greedy nearest-codeword residual encode in the standardised space.
Trainable: the codebooks of the stages read by the target level (L1: stage 1; L2: stages 1 and 2) and mu / sd of K and
V. Assignments are recomputed every step with the current parameters (nearest codeword, not differentiated); the
gradient reaches the selected codewords and mu / sd (straight-through w.r.t. the assignment).

Objective (per layer, data from rqd_collect.py): queries q at sampled positions i (prompt positions in [T/4, T) and,
for the instruction data, the target's generated positions) attend over all keys j <= i, with
  exact bf16 KV for j < 16 (sinks), j > i - R (recent window; R = 0: only j = i) and VQ KV (level L) otherwise,
  loss = || (o_hat - o) W_o^T ||^2 / || o W_o^T ||^2   (attention output error after o_proj, summed over heads)
with o the exact attention. RoPE positions are shifted by a random offset in [0, 32768 - T] per step (the post-RoPE
codebooks must serve keys at positions up to 32k; the exact attention is offset-invariant). Held-out sequences
(one per source) give the before / after loss; a layer keeps its trained parameters only if the held-out loss drops.
Output: /data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_<tag>.pt (+ .json log)
Usage: python experiments/26_magicvq/rqd_train.py --level 1 --R 128 --tag t1w --layers 0 8   (then --merge)
"""
import argparse
import json
import math
import os
import time

import torch

ACTS = "/data/lsh/ckpts/magicvq_v6/acts"
TRAIN = "/data/lsh/ckpts/magicvq_v6/train"
CBDIR = "/data/lsh/results_accept/magicvq/cb"
BASE = f"{CBDIR}/cb_d8K256x2_post_cn_greedy.pt"
MODEL = "/data/models/llama3.1_8b_instruct"
SINKS, D, HKV, H, d = 16, 128, 8, 32, 8


def rope(x, cos, sin):
    """x [T, h, D] fp32, cos/sin [T, D] (HF rotate-half)."""
    c, s = cos[:, None], sin[:, None]
    return x * c + torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1) * s


def load_wo(l):
    from safetensors import safe_open
    import glob
    key = f"model.layers.{l}.self_attn.o_proj.weight"
    for fn in sorted(glob.glob(f"{MODEL}/*.safetensors")):
        with safe_open(fn, "pt") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(key)


class Q:
    """trainable 'p'-format quantizer of one (layer, K/V)."""

    def __init__(self, cbs, mu, sd, level, train_stats=True):
        self.level = level
        self.cb = [torch.nn.Parameter(c.cuda().float().clone()) if s < level else c.cuda().float()
                   for s, c in enumerate(cbs)]
        self.mu0, self.sd0 = mu.cuda().float(), sd.cuda().float()
        self.dmu = torch.nn.Parameter(torch.zeros_like(self.mu0))
        self.lsd = torch.nn.Parameter(torch.zeros_like(self.sd0))
        self.train_stats = train_stats

    def params(self):
        return [c for c in self.cb[:self.level]] + ([self.dmu, self.lsd] if self.train_stats else [])

    def stats(self):
        return self.mu0 + self.sd0 * self.dmu, self.sd0 * self.lsd.exp()

    def __call__(self, x):
        """x [T, Hkv, D] fp32 (post-RoPE K or V) -> recon at `level` (differentiable wrt codewords, mu, sd)."""
        T = x.shape[0]
        mu, sd = self.stats()
        xs = ((x - mu) / sd).permute(1, 0, 2).reshape(HKV, T * (D // d), d)  # [H, N, d]
        r = xs.detach().clone()
        acc = 0
        for s in range(self.level):
            cb = self.cb[s]
            with torch.no_grad():
                cn = (cb * cb).sum(-1)
                c = torch.empty(r.shape[:2], dtype=torch.long, device=r.device)
                for i in range(0, r.shape[1], 1 << 17):
                    c[:, i:i + (1 << 17)] = (cn[:, None] - 2 * torch.bmm(r[:, i:i + (1 << 17)], cb.detach().transpose(1, 2))).argmin(-1)
            sel = torch.gather(cb, 1, c[..., None].expand(-1, -1, d))
            acc = acc + sel
            r = r - sel.detach()
        y = acc.reshape(HKV, T, D).permute(1, 0, 2)
        return y * sd + mu

    def export(self):
        mu, sd = self.stats()
        return [c.detach().float().cpu() for c in self.cb], mu.detach().cpu(), sd.detach().cpu()


def attn_err(q, qpos, K, V, Kh, Vh, R, Wo, oref=None, chunk=128):
    """q [n, H, D] post-RoPE scaled, qpos [n] (absolute, offset-free), K/V exact post-RoPE [T, Hkv, D], Kh/Vh VQ.
    Returns (sum ||err Wo^T||^2, sum ||o Wo^T||^2) over the n queries."""
    T = K.shape[0]
    G = H // HKV
    jj = torch.arange(T, device=q.device)
    num = den = 0.0
    Kt, Kht = K.permute(1, 2, 0), Kh.permute(1, 2, 0)  # [Hkv, D, T]
    Vt, Vht = V.permute(1, 0, 2), Vh.permute(1, 0, 2)  # [Hkv, T, D]
    for i0 in range(0, q.shape[0], chunk):
        qq = q[i0:i0 + chunk]
        n = qq.shape[0]
        ip = qpos[i0:i0 + chunk]
        qg = qq.view(n, HKV, G, D).permute(1, 2, 0, 3).reshape(HKV, G * n, D)
        causal = (jj[None, :] <= ip[:, None])  # [n, T]
        exact = (jj[None, :] < SINKS) | (jj[None, :] > ip[:, None] - max(R, 1))
        with torch.no_grad():
            se = torch.bmm(qg, Kt).view(HKV, G, n, T).masked_fill(~causal, float("-inf"))
            o = torch.bmm(se.softmax(-1).view(HKV, G * n, T), Vt).view(HKV, G, n, D)
            y = (o.permute(2, 0, 1, 3).reshape(n, H * D) @ Wo.t())
        sq = torch.bmm(qg, Kht).view(HKV, G, n, T)
        s = torch.where(exact, se, sq).masked_fill(~causal, float("-inf"))
        p = s.softmax(-1)
        pe = (p * exact).view(HKV, G * n, T)
        pq = (p * (~exact)).view(HKV, G * n, T)
        oh = (torch.bmm(pe, Vt) + torch.bmm(pq, Vht)).view(HKV, G, n, D)
        e = (oh.permute(2, 0, 1, 3).reshape(n, H * D) - o.permute(2, 0, 1, 3).reshape(n, H * D)) @ Wo.t()
        num = num + (e * e).sum()
        den = den + (y * y).sum()
    return num, den


def run_layer(l, a, meta, base, cos, sin, log):
    Wo = load_wo(l).cuda().float()
    data = []
    for s in meta["seqs"]:
        x = torch.load(f"{ACTS}/s{s['idx']}_L{l}.pt")
        data.append({k: v.cuda() for k, v in x.items()})
    ho = set(a.heldout)
    tr = [i for i in range(len(data)) if i not in ho]
    qs = [Q(base["cb"][l][kv], base["mu"][l][kv], base["sd"][l][kv], a.level, not a.no_stats) for kv in range(2)]
    params = qs[0].params() + qs[1].params()
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.5 * (1 + math.cos(math.pi * min(it, a.steps) / a.steps)))
    gen = torch.Generator().manual_seed(1000 + l)
    scale = 1 / math.sqrt(D)

    def loss_on(i, off, qsel=None, grad=True):
        x = data[i]
        T = x["K"].shape[0]
        pos = torch.arange(T, device="cuda") + off
        K = rope(x["K"].float(), cos[pos], sin[pos])
        V = x["V"].float()
        qp = x["qpos"] if qsel is None else x["qpos"][qsel]
        qq = x["Q"].float() if qsel is None else x["Q"][qsel].float()
        q = rope(qq, cos[qp + off], sin[qp + off]) * scale
        with torch.set_grad_enabled(grad):
            Kh, Vh = qs[0](K), qs[1](V)
            Kh = torch.cat([K[:SINKS], Kh[SINKS:]])
            Vh = torch.cat([V[:SINKS], Vh[SINKS:]])
            return attn_err(q, qp, K, V, Kh, Vh, a.R, Wo)

    def heldout():
        torch.backends.cuda.matmul.allow_tf32 = False
        num = den = 0.0
        with torch.no_grad():
            for i in sorted(ho):
                n_, d_ = loss_on(i, a.eval_off, grad=False)
                num += n_.item(); den += d_.item()
        torch.backends.cuda.matmul.allow_tf32 = a.tf32
        return num / den

    t0 = time.time()
    init = [p.detach().clone() for p in params]
    before = heldout()
    hist = []
    for it in range(a.steps):
        i = tr[int(torch.randint(0, len(tr), (1,), generator=gen))]
        T = data[i]["K"].shape[0]
        off = int(torch.randint(0, max(1, 32768 - T), (1,), generator=gen))
        nq = data[i]["qpos"].numel()
        qsel = torch.randperm(nq, generator=gen)[:a.nq_step].cuda()
        num, den = loss_on(i, off, qsel)
        loss = num / den
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 50 == 0 or it == a.steps - 1:
            hist.append((it, round(loss.item(), 5)))
    after = heldout()
    keep = after < before
    if not keep:
        with torch.no_grad():
            for p, p0 in zip(params, init):
                p.copy_(p0)
    out = [qq.export() for qq in qs]
    rec = {"layer": l, "heldout_before": before, "heldout_after": after, "kept": keep, "hist": hist,
           "s": time.time() - t0}
    log(rec)
    torch.save({"layer": l, "kv": out, "rec": rec}, f"{TRAIN}/{a.tag}/L{l}.pt")
    del data
    torch.cuda.empty_cache()
    return rec


def merge(a):
    base = torch.load(BASE, weights_only=False)
    st = {k: v for k, v in base.items()}
    cb, mu, sd, recs = [], [], [], []
    for l in range(32):
        x = torch.load(f"{TRAIN}/{a.tag}/L{l}.pt")
        cb.append([x["kv"][kv][0] for kv in range(2)])
        mu.append([x["kv"][kv][1] for kv in range(2)])
        sd.append([x["kv"][kv][2] for kv in range(2)])
        recs.append(x["rec"])
    st.update({"cb": cb, "mu": mu, "sd": sd, "trained": {"tag": a.tag, "level": a.level, "R": a.R, "args": vars(a),
                                                          "per_layer": recs}})
    torch.save(st, f"{CBDIR}/cb_d8K256x2_post_cn_{a.tag}.pt")
    json.dump({"args": vars(a), "per_layer": recs,
               "mean_before": sum(r["heldout_before"] for r in recs) / 32,
               "mean_after": sum(r["heldout_after"] if r["kept"] else r["heldout_before"] for r in recs) / 32,
               "train_s": sum(r["s"] for r in recs)},
              open(f"{CBDIR}/cb_d8K256x2_post_cn_{a.tag}.json", "w"), indent=1)
    print("merged", a.tag, sum(r["kept"] for r in recs), "layers kept",
          round(sum(r["heldout_before"] for r in recs) / 32, 4), "->",
          round(sum(r["heldout_after"] if r["kept"] else r["heldout_before"] for r in recs) / 32, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--R", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layers", type=int, nargs=2, default=[0, 32])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nq_step", type=int, default=256)
    ap.add_argument("--eval_off", type=int, default=8192)
    ap.add_argument("--heldout", type=int, nargs="*", default=None, help="sequence idx held out (default: last of each src)")
    ap.add_argument("--no_stats", action="store_true", help="freeze mu / sd")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(f"{ACTS}/meta.json"))
    if a.heldout is None:
        last = {}
        for s in meta["seqs"]:
            last[s["src"]] = s["idx"]
        a.heldout = sorted(last.values())
    if a.merge:
        return merge(a)
    os.makedirs(f"{TRAIN}/{a.tag}", exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    base = torch.load(BASE, weights_only=False)
    rp = torch.load(f"{ACTS}/rope.pt")
    cos, sin = rp["cos"].cuda(), rp["sin"].cuda()
    t0 = time.time()
    logf = open(f"{TRAIN}/{a.tag}/log_L{a.layers[0]}_{a.layers[1]}.jsonl", "a")

    def log(r):
        print(json.dumps({k: v for k, v in r.items() if k != "hist"}), r["hist"][::2], f"{time.time() - t0:.0f}s",
              flush=True)
        logf.write(json.dumps(r) + "\n"); logf.flush()
    print("heldout seqs", a.heldout, [meta["seqs"][i]["name"] for i in a.heldout], flush=True)
    for l in range(*a.layers):
        if os.path.exists(f"{TRAIN}/{a.tag}/L{l}.pt"):
            continue
        run_layer(l, a, meta, base, cos, sin, log)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
