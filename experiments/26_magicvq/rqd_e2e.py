"""E7 magicvq v10: end-to-end (output-distribution) fine-tuning of the 'p' VQ KV draft codebooks (AQLM-style global
fine-tune on top of the layer-local rqd_train.py codebooks).

Draft = the target's weights reading a VQ-coded KV (format 'p', see rqd_train.py: post-RoPE K / V, per-(layer, K/V,
kv-head) mu / sd, d 8, K 256 per stage, level L = first L stages), bf16 sinks (j < 16) and a bf16 recent window.
Exactly the step-1 draft of tau_eval's "vq..L<k>+r<R>" round semantics: a draft query at position i reads, at every
layer, the TARGET's context KV j <= i - 1 (exact for j < 16 and j > i - R, VQ-coded otherwise) plus its own K / V at i
(computed from the draft's hidden state, bf16-exact). So the draft forward is a query-only forward: only the n sampled
query positions carry draft hidden states (no block-causal approximation; the per-query R window is exact).

Per step (sequence ids [T], cached exact pre-RoPE K / V of all layers from a bf16 prefill, reused for --inner steps):
  RoPE offset off ~ U[0, 32768 - T) (as rqd_train); n query positions (uniform over [T/4, T); instruction sequences:
  half from the target-generated region);
  teacher = the same query-only forward with all context KV exact (== the target's next-token distribution);
  draft   = query-only forward with VQ context (STE through the nearest-codeword assignment, as rqd_train.Q), per-layer
            gradient checkpointing; trainable: codebooks of the read stages + mu / sd (weights frozen);
  loss    = kl : KL(p_target || p_draft)            (forward KL, mean over positions)
            ce : CE(target top-1) + beta * relu(delta - margin), margin = log p_d(y) - max_{k != y} log p_d(k)
            tv : total variation 0.5 |p_t - p_d|_1    (sampling-SD acceptance = 1 - TV)
  nested (--nest W, level 2): W * loss(stage 1) + loss(stages 1+2) (two draft forwards; --nest_norm divides each term
  by its initial held-out value); --nest_lag EPS: min loss(1+2) s.t. loss(1) <= (1 + EPS) loss_ref(1), loss_ref =
  stage-1 loss of --lag_ref (t1w) on the same batch, W = Lagrange multiplier (as rqd_train v9).
Logged: train KL / top-1 / TV / CE per step, held-out (disjoint sequences, fixed offset and positions) every
--eval_every steps at every level read by the codebook (L1 also for level-2 codebooks: the nested prefix).

Modes:  prep  : build train / held-out token sequences (PG-19 books < 40k tokens never used by any eval, pile-val
                held-out end, LongBench 2wikimqa / musique (quote instruction) and narrativeqa (summary instruction) with
                the target's greedy output appended) -> /data/lsh/ckpts/magicvq_v10/seqs_T<T>.pt
        train : fine-tune -> /data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_<tag>.{pt,json}
        eval  : held-out metrics of codebook files (--eval_cb TAG=PATH ...)
Usage: python experiments/26_magicvq/rqd_e2e.py prep --T 8192
       python experiments/26_magicvq/rqd_e2e.py train --tag e2e_kl_l2 --level 2 --init <t2w.pt> --steps 300
"""
import argparse
import functools
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tau_eval as TE  # noqa: E402
from mv_common import MODEL, PG19  # noqa: E402
from rqd_train import Q, CBDIR, SINKS, rope  # noqa: E402  (same quantizer / STE / RoPE as the layer-local training)

DATA = "/data/lsh/ckpts/magicvq_v10"
QUOTE_T = ("Answer the question based on the given passages. First state the answer in one sentence. Then quote, word "
           "for word, every sentence from the passages that supports the answer, one quote per line, each prefixed with "
           "its passage number.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
           "given passages. First state the answer in one sentence, then give the verbatim supporting quotes with "
           "their passage numbers.\n\nQuestion: {input}\nAnswer:")
STORY_T = ("You are given a story. Summarize the story in detail: the main characters, the setting and the most "
           "important events in order.\n\nStory:\n{context}\n\nNow, summarize the story in detail.\n\nSummary:")


# ------------------------------------------------------------------------------------------------ data (prep)
def pg19_books(tok, T, n_win):
    """PG-19 test books with T - 1 <= tokens(text[:600k]) and tokens < 40k (the rqd_collect rule: never selected by
    tau_eval / e2e (>= 41984) or the synthetic haystacks (>= 40000)); windows BOS + ids[w (T-1) : (w+1) (T-1)]."""
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19}/*.parquet")))
    out = []
    for i, text in enumerate(df["text"]):
        ids = tok(text[:600000], add_special_tokens=False).input_ids
        n4 = len(tok(text[:400000], add_special_tokens=False).input_ids)
        if T - 1 <= len(ids) < 40000 and n4 < 40000:
            ws = [(f"pg19_row{i}_w{w}", [tok.bos_token_id] + ids[w * (T - 1):(w + 1) * (T - 1)])
                  for w in range(min(n_win, len(ids) // (T - 1)))]
            out.append(ws)
    return out  # per book: list of windows


def instr_prompts(tok, T, gen, n_per):
    from lb_tasks import _finish
    out = {}
    for name, files, templ in (("2wikimqa", ("2wikimqa", "2wikimqa_e"), QUOTE_T), ("musique", ("musique",), QUOTE_T),
                               ("narrativeqa", ("narrativeqa",), STORY_T)):
        rows = []
        for f in files:
            rows += [json.loads(l) for l in open(f"/data/lsh/datasets/longbench/data/{f}.jsonl")]
        rng = random.Random(10)
        rng.shuffle(rows)
        seen, lst = set(), []
        for r in rows:
            if r["_id"] in seen:
                continue
            seen.add(r["_id"])
            L = len(tok(r["context"], add_special_tokens=False).input_ids)
            if L < 0.6 * (T - gen):
                continue
            ids = _finish(tok, "hotpotqa_q", templ.format(context=r["context"], input=r.get("input", "")), T - gen)
            lst.append((f"{name}_{r['_id']}", ids))
            if len(lst) >= n_per:
                break
        out[name] = lst
    return out


def prep(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    books = pg19_books(tok, a.T, 2)
    print("pg19 books", len(books), [len(b) for b in books], f"{time.time() - t0:.0f}s", flush=True)
    hb = books[-a.ho_pg19:]  # held-out = different books
    train = [("pg19",) + w for b in books[:-a.ho_pg19] for w in b]
    held = [("pg19",) + b[0] for b in hb]
    sys.path.insert(0, os.path.join(HERE, "..", "..", "specrq", "dbd"))
    from src.datautils import get_pile_val
    ws = get_pile_val(a.n_pile + a.ho_pile, a.T, tok, heldout=True)
    pile = [("pile", f"pile_heldout{i}", w[0].tolist()) for i, w in enumerate(ws)]
    train += pile[:a.n_pile]
    held += pile[a.n_pile:]
    ins = instr_prompts(tok, a.T, a.gen, a.n_instr + a.ho_instr)
    print("instr", {k: len(v) for k, v in ins.items()}, f"{time.time() - t0:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    f = TE.M(model)
    maxT = a.T + 8
    cs = f.cos_sin(torch.arange(maxT, device="cuda"))
    Kst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    Vst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    for src, lst in ins.items():
        for k, (name, ids) in enumerate(lst):
            path = TE.prefill_and_generate(f, torch.tensor(ids, device="cuda"), a.gen, Kst, Vst, cs)
            rec = ("instr_" + src, name, ids + path, len(ids))
            (train if k < a.n_instr else held).append(rec)
            print(src, k, len(ids), len(path), repr(tok.decode(path[:40])), f"{time.time() - t0:.0f}s", flush=True)
    seqs = lambda L: [{"src": r[0], "name": r[1], "ids": r[2], "P": r[3] if len(r) > 3 else len(r[2])} for r in L]
    out = {"T": a.T, "gen": a.gen, "train": seqs(train), "heldout": seqs(held), "args": vars(a)}
    os.makedirs(DATA, exist_ok=True)
    torch.save(out, f"{DATA}/seqs_T{a.T}.pt")
    print("saved", len(out["train"]), "train", len(out["heldout"]), "held-out", f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ model pieces
@torch.no_grad()
def prefill_kv(f, ids, Kst, Vst):
    """exact bf16 prefill (SDPA causal, as rqd_collect / tau_eval); fills Kst / Vst[l][:T] with pre-RoPE K, V."""
    TE.prefill_and_generate(f, ids, 0, Kst, Vst, PF_CS)


PF_CS = None


def qattn(qr, qpos, K, V, Kh, Vh, ks, vs, R, chunk=128):
    """query-only attention. qr [n, H, D] fp32 (post-RoPE, scaled), qpos [n] (offset-free), K / V [Tc, Hkv, D] exact
    context (post-RoPE K), Kh / Vh VQ-coded context or None (teacher: all exact), ks / vs [n, Hkv, D] the query's own
    K (post-RoPE) / V. Context j <= i - 1: exact for j < SINKS or j > i - R, VQ otherwise; own key at i exact."""
    n, H, D = qr.shape
    Hkv = K.shape[1]
    G = H // Hkv
    Tc = K.shape[0]
    jj = torch.arange(Tc, device=qr.device)
    Kt, Vt = K.permute(1, 2, 0), V.permute(1, 0, 2)
    if Kh is not None:
        Kht, Vht = Kh.permute(1, 2, 0), Vh.permute(1, 0, 2)
    outs = []
    for i0 in range(0, n, chunk):
        qq = qr[i0:i0 + chunk]
        m = qq.shape[0]
        ip = qpos[i0:i0 + chunk]
        qg = qq.view(m, Hkv, G, D).permute(1, 2, 0, 3).reshape(Hkv, G * m, D)
        causal = jj[None, :] <= ip[:, None] - 1  # [m, Tc]
        s = torch.bmm(qg, Kt).view(Hkv, G, m, Tc)
        if Kh is not None:
            exact = (jj[None, :] < SINKS) | (jj[None, :] > ip[:, None] - max(R, 1))
            s = torch.where(exact, s, torch.bmm(qg, Kht).view(Hkv, G, m, Tc))
        s = s.masked_fill(~causal, float("-inf"))
        kself = ks[i0:i0 + chunk].permute(1, 0, 2)  # [Hkv, m, D]
        sself = (qg.view(Hkv, G, m, D) * kself[:, None]).sum(-1, keepdim=True)  # [Hkv, G, m, 1]
        p = torch.cat([s, sself], -1).softmax(-1)
        pc, ps = p[..., :Tc], p[..., Tc:]
        if Kh is not None:
            o = (torch.bmm((pc * exact).view(Hkv, G * m, Tc), Vt) + torch.bmm((pc * ~exact).view(Hkv, G * m, Tc), Vht))
        else:
            o = torch.bmm(pc.reshape(Hkv, G * m, Tc), Vt)
        o = o.view(Hkv, G, m, D) + ps * vs[i0:i0 + chunk].permute(1, 0, 2)[:, None]
        outs.append(o.permute(2, 0, 1, 3).reshape(m, H, D))
    return torch.cat(outs)


class Ctx:
    """per-step context: exact pre-RoPE K / V of all layers (bf16, [T, Hkv, D]), query positions, RoPE offset."""

    def __init__(self, Kst, Vst, qpos, off, cos, sin):
        self.Kst, self.Vst, self.qpos, self.off, self.cos, self.sin = Kst, Vst, qpos, off, cos, sin
        self.Tc = int(qpos.max())  # context keys j <= max(qpos) - 1

    def kv(self, l):
        pos = torch.arange(self.Tc, device="cuda") + self.off
        return rope(self.Kst[l][:self.Tc].float(), self.cos[pos], self.sin[pos]), self.Vst[l][:self.Tc].float()


def layer_fn(f, l, h, ctx, qs, lv, R):
    """one decoder layer of the query-only forward. qs: [Q_K, Q_V] of layer l (None: teacher), lv: stages read."""
    layer = f.mm.layers[l]
    n = h.shape[0]
    q, k, v = f.qkv(layer, h)
    pq = ctx.qpos + ctx.off
    qr = rope(q.float(), ctx.cos[pq], ctx.sin[pq]) / math.sqrt(f.D)
    kr = rope(k.float(), ctx.cos[pq], ctx.sin[pq])
    K, V = ctx.kv(l)
    Kh = Vh = None
    if qs is not None:
        Kh = qs[0](K, all_levels=True)[lv - 1]
        Vh = qs[1](V, all_levels=True)[lv - 1]
        Kh = torch.cat([K[:SINKS], Kh[SINKS:]])
        Vh = torch.cat([V[:SINKS], Vh[SINKS:]])
    o = qattn(qr, ctx.qpos, K, V, Kh, Vh, kr, v.float(), R)
    return f.post(layer, h, o.to(torch.bfloat16).reshape(n, -1))


def qforward(f, ids_q, ctx, qsets, lv, R, ckpt):
    """ids_q [n] (token at each query position) -> logits [n, V] fp32. qsets: per layer [Q_K, Q_V] or None."""
    h = f.mm.embed_tokens(ids_q)
    for l in range(f.nl):
        fn = functools.partial(layer_fn, f, l, ctx=ctx, qs=None if qsets is None else qsets[l], lv=lv, R=R)
        h = checkpoint(fn, h, use_reentrant=False) if ckpt else fn(h)
    return f.logits(h)


def dist_metrics(zt, zd, a, grad=True):
    """per-position losses. zt teacher logits (no grad), zd draft logits. Returns (loss, dict of means)."""
    lt = zt.log_softmax(-1)
    ld = zd.log_softmax(-1)
    pt = lt.exp()
    y = zt.argmax(-1)
    kl = (pt * (lt - ld)).sum(-1)
    tv = 0.5 * (pt - ld.exp()).abs().sum(-1)
    ce = -ld.gather(1, y[:, None])[:, 0]
    ldo = ld.scatter(1, y[:, None], float("-inf"))
    margin = -ce - ldo.amax(-1)
    top1 = (zd.argmax(-1) == y).float()
    if a.loss == "kl":
        loss = kl.mean()
    elif a.loss == "ce":
        loss = (ce + a.beta * F.relu(a.delta - margin)).mean()
    elif a.loss == "tv":
        loss = tv.mean()
    elif a.loss == "klce":
        loss = kl.mean() + a.beta * (ce + F.relu(a.delta - margin)).mean()
    m = {"kl": kl.mean().item(), "tv": tv.mean().item(), "top1": top1.mean().item(), "ce": ce.mean().item(),
         "margin_neg": (margin < 0).float().mean().item()}
    return loss, m


# ------------------------------------------------------------------------------------------------ train / eval
def sample_qpos(seq, n, gen):
    T = len(seq["ids"])
    lo = max(T // 4, SINKS + 256)
    if seq["P"] < T:  # instruction sequence with the target's greedy output: half of the queries in the generated region
        g = torch.arange(seq["P"] - 1, T, dtype=torch.long)
        g = g[torch.randperm(g.numel(), generator=gen)[: n // 2]]
        r = torch.randint(lo, T, (n - g.numel(),), generator=gen)
        return torch.cat([g, r]).unique()
    return torch.randint(lo, T, (n,), generator=gen).unique()


def load_cb_qsets(path, level, train=False, freeze_s1=False, no_stats=False):
    st = torch.load(path, weights_only=False)
    qsets = [[Q(st["cb"][l][kv], st["mu"][l][kv], st["sd"][l][kv], level, train and not (no_stats or freeze_s1),
                freeze_s1 and train) for kv in range(2)] for l in range(32)]
    if not train:
        for qq in qsets:
            for x in qq:
                for p in x.params():
                    p.requires_grad_(False)
    return st, qsets


class Runner:
    def __init__(self, a):
        from transformers import AutoModelForCausalLM
        global PF_CS
        self.a = a
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.f = TE.M(model)
        self.data = torch.load(f"{DATA}/seqs_T{a.T}.pt")
        cs = self.f.cos_sin(torch.arange(40000, device="cuda"))
        PF_CS = cs
        self.cos, self.sin = cs[0][0], cs[1][0]
        maxT = max(len(s["ids"]) for s in self.data["train"] + self.data["heldout"]) + 8
        self.Kst = [torch.empty(maxT, self.f.Hkv, self.f.D, device="cuda", dtype=torch.bfloat16) for _ in range(self.f.nl)]
        self.Vst = [torch.empty(maxT, self.f.Hkv, self.f.D, device="cuda", dtype=torch.bfloat16) for _ in range(self.f.nl)]
        self.cur = None
        # fixed held-out batch: positions per sequence (seeded), offset a.eval_off
        g = torch.Generator().manual_seed(123)
        self.ho = [(i, sample_qpos(s, a.nq_eval, g)) for i, s in enumerate(self.data["heldout"])]

    def load_seq(self, seq):
        if self.cur != seq["name"]:
            prefill_kv(self.f, torch.tensor(seq["ids"], device="cuda"), self.Kst, self.Vst)
            self.cur = seq["name"]

    def batch(self, seq, qpos, off):
        self.load_seq(seq)
        qpos = qpos.cuda()
        ids = torch.tensor(seq["ids"], device="cuda")[qpos]
        return ids, Ctx(self.Kst, self.Vst, qpos, off, self.cos, self.sin)

    @torch.no_grad()
    def teacher(self, ids, ctx):
        return qforward(self.f, ids, ctx, None, 0, self.a.R, False)

    @torch.no_grad()
    def heldout(self, qsets, levels):
        """held-out metrics (means over all held-out positions) of qsets at each level in `levels`."""
        a = self.a
        acc = {lv: {} for lv in levels}
        ntot = 0
        for i, qpos in self.ho:
            ids, ctx = self.batch(self.data["heldout"][i], qpos, a.eval_off)
            zt = self.teacher(ids, ctx)
            n = ids.shape[0]
            ntot += n
            for lv in levels:
                zd = qforward(self.f, ids, ctx, qsets, lv, a.R, False)
                _, m = dist_metrics(zt, zd, a)
                for k_, v_ in m.items():
                    acc[lv][k_] = acc[lv].get(k_, 0.0) + v_ * n
        return {f"L{lv}": {k_: v_ / ntot for k_, v_ in acc[lv].items()} for lv in levels}


def train(a):
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    r = Runner(a)
    f = r.f
    t0 = time.time()
    st, qsets = load_cb_qsets(a.init, a.level, True, a.freeze_s1, a.no_stats)
    params = [p for qq in qsets for x in qq for p in x.params()]
    print("init", a.init, "trainable tensors", len(params), "elems", sum(p.numel() for p in params), flush=True)
    ref_q = load_cb_qsets(a.lag_ref, 1)[1] if a.nest_lag is not None else None
    levels = [1, 2] if a.level == 2 else [1]
    log_path = f"{CBDIR}/cb_d8K256x2_post_cn_{a.tag}.jsonl"
    logf = open(log_path, "w")

    def log(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    if a.check:  # sanity: draft with exact context == teacher; teacher top-1 vs the full SDPA prefill
        seq = r.data["heldout"][0]
        g = torch.Generator().manual_seed(5)
        ids, ctx = r.batch(seq, sample_qpos(seq, 128, g), a.eval_off)
        zt = r.teacher(ids, ctx)
        ident = [[lambda x, all_levels=True: [x, x], lambda x, all_levels=True: [x, x]] for _ in range(32)]
        with torch.no_grad():
            zd = qforward(f, ids, ctx, ident, 2, a.R, False)
            ctx0 = Ctx(r.Kst, r.Vst, ctx.qpos, 0, r.cos, r.sin)
            zt0 = r.teacher(ids, ctx0)
        _, m = dist_metrics(zt, zd, a)
        _, m0 = dist_metrics(zt, zt0, a)
        # full-sequence bf16 forward argmax (independent path) at the query positions
        full = torch.tensor(seq["ids"], device="cuda")
        with torch.no_grad():
            hs = f.mm(full[None]).last_hidden_state[0]
            out = f.m.lm_head(hs[ctx.qpos]).float()
        top_full = (out.argmax(-1) == zt.argmax(-1)).float().mean().item()
        log({"check": {"exact_vq_vs_teacher": m, "offset0_vs_offset": m0, "teacher_top1_vs_hf_forward": top_full}})
        del out, hs

    hist = []
    h0 = r.heldout(qsets, levels)
    log({"step": 0, "heldout": h0, "s": round(time.time() - t0, 1)})
    hist.append({"step": 0, "heldout": h0})
    ref = {1: 1.0, 2: 1.0}
    if a.nest is not None and a.nest_norm:
        k_ = a.loss if a.loss in ("kl", "tv", "ce") else "kl"
        ref = {1: h0["L1"][k_], 2: h0["L2"][k_]}
    opt = torch.optim.Adam(params, lr=a.lr)
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(it, a.steps) / a.steps)))
    gen = torch.Generator().manual_seed(a.seed)
    W = a.nest if a.nest is not None else 0.0
    rbar = None
    best = (float("inf"), None)
    key = "kl" if a.loss not in ("ce",) else "ce"
    tr = r.data["train"]
    seq = None
    for it in range(a.steps):
        if it % a.inner == 0:
            seq = tr[int(torch.randint(0, len(tr), (1,), generator=gen))]
        T = len(seq["ids"])
        off = int(torch.randint(0, max(1, 32768 - T), (1,), generator=gen))
        qpos = sample_qpos(seq, a.nq, gen)
        ids, ctx = r.batch(seq, qpos, off)
        zt = r.teacher(ids, ctx)
        rec = {"step": it + 1, "seq": seq["name"]}
        if a.nest is None:
            zd = qforward(f, ids, ctx, qsets, a.level, a.R, True)
            loss, m = dist_metrics(zt, zd, a)
            rec.update({f"L{a.level}": m})
        else:
            zd1 = qforward(f, ids, ctx, qsets, 1, a.R, True)
            l1, m1 = dist_metrics(zt, zd1, a)
            if W > 0:
                (W * l1 / ref[1]).backward()
            del zd1
            zd2 = qforward(f, ids, ctx, qsets, 2, a.R, True)
            l2, m2 = dist_metrics(zt, zd2, a)
            loss = l2 / ref[2]
            rec.update({"L1": m1, "L2": m2, "W": W})
            if ref_q is not None:
                with torch.no_grad():
                    zr = qforward(f, ids, ctx, ref_q, 1, a.R, False)
                    lr_, mr = dist_metrics(zt, zr, a)
                rr = l1.item() / max(lr_.item(), 1e-8)
                rbar = rr if rbar is None else 0.9 * rbar + 0.1 * rr
                W = max(0.0, W + a.lag_eta * (rbar - 1 - a.nest_lag))
                rec.update({"Lref": mr, "rbar": rbar})
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip) if a.clip > 0 else torch.zeros(())
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        rec.update({"loss": loss.item(), "gnorm": gn.item(), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        if it % a.log_every == 0:
            log(rec)
        hist.append(rec)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.steps:
            ho = r.heldout(qsets, levels)
            log({"step": it + 1, "heldout": ho, "s": round(time.time() - t0, 1)})
            hist.append({"step": it + 1, "heldout": ho})
            sel = ho[f"L{a.level}"][key] if a.nest is None else (W * ho["L1"][key] / ref[1] + ho["L2"][key] / ref[2])
            if a.nest is not None and a.nest_lag is not None:
                sel = ho["L2"][key]
            if sel < best[0]:
                best = (sel, it + 1)
                save_cb(a, st, qsets, hist, best, "")
    save_cb(a, st, qsets, hist, best, "_last")
    print("done best", best, f"{time.time() - t0:.0f}s", flush=True)


def save_cb(a, st, qsets, hist, best, suffix):
    out = {k: v for k, v in st.items()}
    cb, mu, sd = [], [], []
    for l in range(32):
        ex = [qsets[l][kv].export() for kv in range(2)]
        cb.append([ex[kv][0] for kv in range(2)])
        mu.append([ex[kv][1] for kv in range(2)])
        sd.append([ex[kv][2] for kv in range(2)])
    out.update({"cb": cb, "mu": mu, "sd": sd,
                "trained": {"tag": a.tag, "kind": "e2e", "level": a.level, "R": a.R, "loss": a.loss, "args": vars(a),
                            "best_step": best[1], "init": a.init}})
    torch.save(out, f"{CBDIR}/cb_d8K256x2_post_cn_{a.tag}{suffix}.pt")
    json.dump({"args": vars(a), "best": best, "hist": hist},
              open(f"{CBDIR}/cb_d8K256x2_post_cn_{a.tag}{suffix}.json", "w"), indent=1)


def evaluate(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    r = Runner(a)
    res = {}
    for t, p in (x.split("=", 1) for x in a.eval_cb):
        _, qsets = load_cb_qsets(p, 2)
        res[t] = r.heldout(qsets, [1, 2])
        print(t, json.dumps(res[t]), flush=True)
    json.dump({"args": vars(a), "res": res}, open(f"{CBDIR}/rqd_e2e_eval_{a.tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "train", "eval"])
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--ho_pg19", type=int, default=4)
    ap.add_argument("--n_pile", type=int, default=24)
    ap.add_argument("--ho_pile", type=int, default=4)
    ap.add_argument("--n_instr", type=int, default=16, help="per instruction source (2wikimqa, musique, narrativeqa)")
    ap.add_argument("--ho_instr", type=int, default=3)
    ap.add_argument("--tag", default="e2e_dbg")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--init", default=f"{CBDIR}/cb_d8K256x2_post_cn_t2w.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--nq", type=int, default=256)
    ap.add_argument("--nq_eval", type=int, default=256)
    ap.add_argument("--inner", type=int, default=4, help="optimizer steps per prefilled sequence")
    ap.add_argument("--eval_off", type=int, default=16384)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--loss", default="kl", choices=["kl", "ce", "tv", "klce"])
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--nest", type=float, default=None)
    ap.add_argument("--nest_norm", action="store_true")
    ap.add_argument("--nest_lag", type=float, default=None)
    ap.add_argument("--lag_eta", type=float, default=0.1)
    ap.add_argument("--lag_ref", default=f"{CBDIR}/cb_d8K256x2_post_cn_t1w.pt")
    ap.add_argument("--freeze_s1", action="store_true")
    ap.add_argument("--no_stats", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--eval_cb", nargs="*", default=None)
    a = ap.parse_args()
    if a.mode == "prep":
        return prep(a)
    if a.mode == "eval":
        return evaluate(a)
    train(a)


if __name__ == "__main__":
    main()
