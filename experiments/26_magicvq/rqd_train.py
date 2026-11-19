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
--loss mse (v7 control, tag t2mw): same data / parameters / steps / lr / STE / RoPE offset / held-out rule, but the loss is
the relative reconstruction error of the VQ-coded K and V in the original (de-standardised, post-RoPE) space,
  ||K_hat - K||^2 / ||K||^2 + ||V_hat - V||^2 / ||V||^2  over positions j in [16, max(qpos) - R]
(= the positions some sampled query reads in VQ form; sinks and the R-window excluded as in the attention loss).
--eval_cb TAG=PATH ...: diagnostic only, held-out attention-output loss and K/V MSE (both objectives) per layer for each
codebook file (level / R as given); writes <CBDIR>/rqd_eval_<eval_tag>.json.
--nest W (v8, level 2, tag t12w): nested-prefix objective W * attn_loss(stage 1 only) + attn_loss(stages 1+2), so one
codebook serves the 1-bit prefix draft and the 2-bit draft (held-out rule on the same combined ratio).
v9: --init PATH (warm start from a trained codebook, e.g. t2w, incl. its mu / sd), --freeze_s1 (stage 1 and mu / sd
frozen at the init, e.g. t1w; only stage 2 trained, = dedicated prefix + residual), --layer_list (pilot subsets),
--nest_norm (nested terms divided by their init held-out values, so W is a relative priority); nested runs log the
held-out L1 and L2 losses separately (heldout_L1L2_before / _after; with --nest_lag a 3rd entry = reference L1).
--nest_lag EPS: minimise L2 s.t. L1 <= (1 + EPS) L1_ref, L1_ref = stage-1 loss of --lag_ref (t1w) on the same batch;
W (start --nest) is a Lagrange multiplier, W <- max(0, W + lag_eta (EMA_0.9(L1 / L1_ref) - 1 - EPS)) per step.
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

    def __init__(self, cbs, mu, sd, level, train_stats=True, freeze_s1=False):
        self.level = level
        self.cb = [torch.nn.Parameter(c.cuda().float().clone()) if s < level and not (freeze_s1 and s == 0)
                   else c.cuda().float() for s, c in enumerate(cbs)]
        self.mu0, self.sd0 = mu.cuda().float(), sd.cuda().float()
        self.dmu = torch.nn.Parameter(torch.zeros_like(self.mu0))
        self.lsd = torch.nn.Parameter(torch.zeros_like(self.sd0))
        self.train_stats = train_stats

    def params(self):
        return [c for c in self.cb[:self.level] if isinstance(c, torch.nn.Parameter)] + ([self.dmu, self.lsd] if self.train_stats else [])

    def stats(self):
        return self.mu0 + self.sd0 * self.dmu, self.sd0 * self.lsd.exp()

    def __call__(self, x, all_levels=False):
        """x [T, Hkv, D] fp32 (post-RoPE K or V) -> recon at `level` (differentiable wrt codewords, mu, sd);
        all_levels: list of the recons after stage 1 .. level (nested prefixes)."""
        T = x.shape[0]
        mu, sd = self.stats()
        xs = ((x - mu) / sd).permute(1, 0, 2).reshape(HKV, T * (D // d), d)  # [H, N, d]
        r = xs.detach().clone()
        acc = 0
        outs = []
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
            if all_levels:
                outs.append(acc.reshape(HKV, T, D).permute(1, 0, 2) * sd + mu)
        if all_levels:
            return outs
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


def mse_err(qpos, K, V, Kh, Vh, R):
    """relative K / V reconstruction error over the VQ-read positions [SINKS, max(qpos) - R] (R = 0: max(qpos) - 1)."""
    hi = int(qpos.max()) - max(R, 1) + 1
    sl = slice(SINKS, max(SINKS + 1, hi))
    eK = ((Kh[sl] - K[sl]) ** 2).sum() / (K[sl] ** 2).sum()
    eV = ((Vh[sl] - V[sl]) ** 2).sum() / (V[sl] ** 2).sum()
    return eK + eV, eK.detach(), eV.detach()


def run_layer(l, a, meta, base, cos, sin, log, lag_ref=None):
    Wo = load_wo(l).cuda().float()
    data = []
    for s in meta["seqs"]:
        x = torch.load(f"{ACTS}/s{s['idx']}_L{l}.pt")
        data.append({k: v.cuda() for k, v in x.items()})
    ho = set(a.heldout)
    tr = [i for i in range(len(data)) if i not in ho]
    qs = [Q(base["cb"][l][kv], base["mu"][l][kv], base["sd"][l][kv], a.level, not (a.no_stats or a.freeze_s1),
            a.freeze_s1) for kv in range(2)]
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
            if a.nest is not None:  # v8: nested prefix loss  w * attn(stage 1) + attn(stages 1+2)  (same den)
                Ks, Vs = qs[0](K, True), qs[1](V, True)
                tot = 0.0
                terms.clear()
                for s_, (Kh, Vh) in enumerate(zip(Ks, Vs)):
                    Kh = torch.cat([K[:SINKS], Kh[SINKS:]])
                    Vh = torch.cat([V[:SINKS], Vh[SINKS:]])
                    n_, d_ = attn_err(q, qp, K, V, Kh, Vh, a.R, Wo)
                    terms.append(n_.detach())
                    tot = tot + (wl[0] if s_ == 0 else 1.0) * n_ / ref[s_]  # v9 --nest_norm: ref = init held-out
                if qref is not None:  # v9 --nest_lag: stage-1 loss of the reference (t1w) codebook on the same batch
                    with torch.no_grad():
                        Kr, Vr = qref[0](K), qref[1](V)
                        Kr = torch.cat([K[:SINKS], Kr[SINKS:]])
                        Vr = torch.cat([V[:SINKS], Vr[SINKS:]])
                        terms.append(attn_err(q, qp, K, V, Kr, Vr, a.R, Wo)[0])
                return tot, d_
            Kh, Vh = qs[0](K), qs[1](V)
            Kh = torch.cat([K[:SINKS], Kh[SINKS:]])
            Vh = torch.cat([V[:SINKS], Vh[SINKS:]])
            if a.loss == "mse":  # (loss, 1): held-out = mean over held-out sequences of the per-sequence relative error
                return mse_err(x["qpos"], K, V, Kh, Vh, a.R)[0], torch.ones((), device="cuda")
            return attn_err(q, qp, K, V, Kh, Vh, a.R, Wo)

    terms = []  # nested: per-level numerators of the last loss_on call (stage 1, stages 1+2[, reference stage 1])
    wl = [a.nest]  # nested: current stage-1 weight (fixed, or the Lagrange multiplier under --nest_lag)
    qref = None
    if a.nest_lag is not None:  # v9: min L2 s.t. L1 <= L1_ref (1 + eps); W <- max(0, W + eta (EMA(L1 / L1_ref) - 1 - eps))
        qref = [Q(lag_ref["cb"][l][kv], lag_ref["mu"][l][kv], lag_ref["sd"][l][kv], 1, False) for kv in range(2)]
        rbar = None
    ref = [1.0, 1.0]  # nested: per-term normalisers (raw objective: 1)
    split = {}

    def heldout(key=None):
        torch.backends.cuda.matmul.allow_tf32 = False
        num = den = 0.0
        tn = [0.0, 0.0, 0.0]
        with torch.no_grad():
            for i in sorted(ho):
                n_, d_ = loss_on(i, a.eval_off, grad=False)
                num += n_.item(); den += d_.item()
                for k_, t_ in enumerate(terms):
                    tn[k_] += t_.item()
        torch.backends.cuda.matmul.allow_tf32 = a.tf32
        if key and a.nest is not None:  # v9: separate held-out L1 / L2 attention loss
            split[key] = [tn[0] / den, tn[1] / den] + ([tn[2] / den] if qref is not None else [])
        return num / den

    t0 = time.time()
    init = [p.detach().clone() for p in params]
    before = heldout("before")
    if a.nest is not None and a.nest_norm:  # v9: W * L1 / L1_init + L2 / L2_init (W = relative priority)
        ref[0], ref[1] = split["before"]
        before = a.nest + 1.0
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
        if qref is not None:
            r_ = (terms[0] / terms[2]).item()
            rbar = r_ if rbar is None else 0.9 * rbar + 0.1 * r_
            wl[0] = max(0.0, wl[0] + a.lag_eta * (rbar - 1 - a.nest_lag))
        if it % 50 == 0 or it == a.steps - 1:
            hist.append((it, round(loss.item(), 5)) + ((round(wl[0], 4), round(rbar, 4)) if qref is not None else ()))
    after = heldout("after")
    keep = after < before
    if qref is not None:  # lag: keep unless held-out L2 got > 1 % worse, or L1 got worse without an L2 gain
        (b1, b2), (a1, a2) = split["before"][:2], split["after"][:2]
        keep = (a2 <= 1.01 * b2 and a1 < b1) or (a2 < b2 and a1 <= b1)
    if not keep:
        with torch.no_grad():
            for p, p0 in zip(params, init):
                p.copy_(p0)
    out = [qq.export() for qq in qs]
    rec = {"layer": l, "heldout_before": before, "heldout_after": after, "kept": keep, "hist": hist, "W_final": wl[0],
           **({"heldout_L1L2_before": split["before"], "heldout_L1L2_after": split["after"]} if split else {}),
           "s": time.time() - t0}
    log(rec)
    torch.save({"layer": l, "kv": out, "rec": rec}, f"{TRAIN}/{a.tag}/L{l}.pt")
    del data
    torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def eval_layer(l, a, meta, cbs, cos, sin):
    """held-out attention-output loss (pooled ratio, as run_layer's held-out) and K / V relative MSE for each codebook."""
    torch.backends.cuda.matmul.allow_tf32 = False
    Wo = load_wo(l).cuda().float()
    out = {}
    data = {i: {k: v.cuda() for k, v in torch.load(f"{ACTS}/s{meta['seqs'][i]['idx']}_L{l}.pt").items()} for i in a.heldout}
    for tag, st in cbs.items():
        qs = [Q(st["cb"][l][kv], st["mu"][l][kv], st["sd"][l][kv], a.level, False) for kv in range(2)]
        num = den = 0.0
        eks, evs = [], []
        for i, x in data.items():
            T = x["K"].shape[0]
            pos = torch.arange(T, device="cuda") + a.eval_off
            K = rope(x["K"].float(), cos[pos], sin[pos])
            V = x["V"].float()
            q = rope(x["Q"].float(), cos[x["qpos"] + a.eval_off], sin[x["qpos"] + a.eval_off]) / math.sqrt(D)
            Kh, Vh = qs[0](K), qs[1](V)
            Kh = torch.cat([K[:SINKS], Kh[SINKS:]])
            Vh = torch.cat([V[:SINKS], Vh[SINKS:]])
            n_, d_ = attn_err(q, x["qpos"], K, V, Kh, Vh, a.R, Wo)
            num += n_.item(); den += d_.item()
            _, ek, ev = mse_err(x["qpos"], K, V, Kh, Vh, a.R)
            eks.append(ek.item()); evs.append(ev.item())
        out[tag] = {"attn": num / den, "mseK": sum(eks) / len(eks), "mseV": sum(evs) / len(evs)}
    print(l, json.dumps(out), flush=True)
    return out


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
    st.update({"cb": cb, "mu": mu, "sd": sd, "trained": {"tag": a.tag, "level": a.level, "R": a.R, "loss": a.loss,
                                                          "args": vars(a),
                                                          "per_layer": recs}})
    if a.nest is not None:
        st["trained"]["nest"] = a.nest
    if a.init or a.freeze_s1:
        st["trained"].update(init=a.init, freeze_s1=a.freeze_s1)
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
    ap.add_argument("--loss", default="attn", choices=["attn", "mse"], help="v7: mse = K/V reconstruction control")
    ap.add_argument("--nest", type=float, default=None, metavar="W",
                    help="v8: nested prefix training (level 2): loss = W * attn(stage 1) + attn(stages 1+2)")
    ap.add_argument("--init", default=None, help="v9: start codebook file (default: greedy 'p' BASE), e.g. t2w warm start")
    ap.add_argument("--freeze_s1", action="store_true", help="v9: stage 1 and mu / sd frozen (train stage 2 only)")
    ap.add_argument("--layer_list", type=int, nargs="*", default=None, help="v9: explicit layers (overrides --layers)")
    ap.add_argument("--nest_norm", action="store_true",
                    help="v9: normalise the nested terms by their init held-out values: W * L1/L1_0 + L2/L2_0")
    ap.add_argument("--nest_lag", type=float, default=None, metavar="EPS",
                    help="v9: constrained nested training, --nest = initial W (Lagrange multiplier)")
    ap.add_argument("--lag_eta", type=float, default=0.1)
    ap.add_argument("--lag_ref", default=f"{CBDIR}/cb_d8K256x2_post_cn_t1w.pt", help="v9: L1 reference codebook")
    ap.add_argument("--eval_cb", nargs="*", default=None, metavar="TAG=PATH", help="diagnostic held-out eval only")
    a = ap.parse_args()
    meta = json.load(open(f"{ACTS}/meta.json"))
    if a.heldout is None:
        last = {}
        for s in meta["seqs"]:
            last[s["src"]] = s["idx"]
        a.heldout = sorted(last.values())
    if a.merge:
        return merge(a)
    if a.eval_cb:
        rp = torch.load(f"{ACTS}/rope.pt")
        cos, sin = rp["cos"].cuda(), rp["sin"].cuda()
        cbs = {t: torch.load(p, weights_only=False) for t, p in (x.split("=", 1) for x in a.eval_cb)}
        res = {l: eval_layer(l, a, meta, cbs, cos, sin) for l in (a.layer_list or range(*a.layers))}
        mean = {t: {k: sum(res[l][t][k] for l in res) / len(res) for k in ("attn", "mseK", "mseV")} for t in cbs}
        print("mean", json.dumps(mean), flush=True)
        json.dump({"args": vars(a), "per_layer": res, "mean": mean},
                  open(f"{CBDIR}/rqd_eval_{a.tag}.json", "w"), indent=1)
        return
    os.makedirs(f"{TRAIN}/{a.tag}", exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    base = torch.load(a.init or BASE, weights_only=False)
    print("init", a.init or BASE, flush=True)
    lag_ref = torch.load(a.lag_ref, weights_only=False) if a.nest_lag is not None else None
    rp = torch.load(f"{ACTS}/rope.pt")
    cos, sin = rp["cos"].cuda(), rp["sin"].cuda()
    t0 = time.time()
    logf = open(f"{TRAIN}/{a.tag}/log_L{a.layers[0]}_{a.layers[1]}.jsonl", "a")

    def log(r):
        print(json.dumps({k: v for k, v in r.items() if k != "hist"}), r["hist"][::2], f"{time.time() - t0:.0f}s",
              flush=True)
        logf.write(json.dumps(r) + "\n"); logf.flush()
    print("heldout seqs", a.heldout, [meta["seqs"][i]["name"] for i in a.heldout], flush=True)
    for l in (a.layer_list or range(*a.layers)):
        if os.path.exists(f"{TRAIN}/{a.tag}/L{l}.pt"):
            continue
        run_layer(l, a, meta, base, cos, sin, log, lag_ref)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
