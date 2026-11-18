"""E8a phase A: prefill-time (causal) and generation-time (oracle) attention features per sequence, for the E7 tau
records (lb_tau.py runs v4/v4b/v6, merged by tau_records.py). No drafting, no timing.

Per sequence (prompt rebuilt with lb_tasks.py, same id; target greedy output = the stored text re-tokenized):
  phase 1 (prefill, identical to tau_eval.prefill_and_generate's prefill: bf16 SDPA, same q/k RoPE): for every layer,
    the SnapKV selection (tau_eval.snap_select, window 32, pool 5, maxb 4096 -> prefixes 512 / 2048 / 4096 = the
    drafts' sets) and fp32 attention of the last W = 32 prompt queries ("obs" = SnapKV observation window).
  phase 2 (oracle, not causal): teacher-forced target pass over the generated tokens; the same statistics for the
    generated-position queries (target attention over 0..i), plus the target argmax vs the re-tokenized text
    (reproducibility check of prompt + trajectory).
Statistics per (layer, kv-head), averaged over the 4 query heads of the group and the queries of the block:
  cov{b}      attention mass on the SnapKV-{b} draft set (selected b - 32 tokens + window [P-32, i])
  far{b}      captured fraction of the mass outside the window (prompt keys < P - 32)
  covho{b}    (obs only) split-half: selection from obs queries 0..15, mass of obs queries 16..31 (held out)
  ent, n90    entropy (nats) and #keys for 90 % mass;  mfar128: mass on keys j <= i - 128 (outside an r128 window)
  e_snap{b}   relative attention-output error of the SnapKV-{b} draft (softmax restricted to the draft set)
  e_vq_{cb}   relative attention-output error of RQ2+r128 (keys/values j <= i - 128 and j >= 16 VQ, 2 stages)
  e_hyb_{cb}  same for the hybrid hyb512pL2 (SnapKV-512 set + window + sinks bf16, rest VQ)
  o_*         the same errors after o_proj (group columns of W_o, relative to |W_o o| of the full layer)
  cb in {def: cb_d8K256x2_post_cn_greedy ('p'), t2w: cb_d8K256x2_post_cn_t2w ('pt2w')}.
Output: /data/lsh/results_accept/accept_pred/feat_<run>_<task>_ctx<ctx>.jsonl (one line per id; arrays [32][8]).
Usage: python experiments/29_accept_pred/features.py --ctx 16384 --tasks syn_quote gov_report [--limit 2]
       [--cpu --trunc 600 for a CPU smoke test of the code path on a truncated prompt]
"""
from __future__ import annotations

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
sys.path.insert(0, os.path.join(HERE, "..", "26_magicvq"))
import tau_eval as TE  # noqa: E402
from mv_common import CB, MODEL, SINKS, KVQuant, make_quant  # noqa: E402
from lb_tasks import SYN_TASKS, lb_prompts, load_haystack_books, syn_prompts  # noqa: E402
import tau_records  # noqa: E402

OUT = "/data/lsh/results_accept/accept_pred"
W = 32
BUDGETS = (512, 2048, 4096)
R = 128
CBS = {"def": "p", "t2w": "pt2w"}
STATS = (["cov512", "cov2048", "cov4096", "far512", "far2048", "far4096", "ent", "n90", "mfar128",
          "e_snap512", "e_snap4096", "o_snap512", "o_snap4096"]
         + [f"{p}_{k}_{c}" for p in ("e", "o") for k in ("vq", "hyb") for c in CBS])


def _rel(d, o):
    return d.norm(dim=-1) / o.norm(dim=-1).clamp(min=1e-12)


@torch.no_grad()
def block_stats(f, Wo, qs, ipos, Kx, Vx, kvq, sel, P, acc):
    """qs [n, H, D] fp32 RoPE'd and scaled; ipos [n] query positions; keys 0..T-1 (T = ipos.max() + 1 used).
    Kx/Vx [T, Hkv, D] fp32 exact (post-RoPE K); kvq {cb: (Kq, Vq)} VQ (sinks exact); sel [Hkv, maxb - W] SnapKV order.
    Accumulates per (Hkv) sums over (G heads x n queries) into acc[stat] ([Hkv] tensors) and acc['_n']."""
    n, H, D = qs.shape
    Hkv, G = f.Hkv, f.G
    T = int(ipos.max().item()) + 1
    dev = qs.device
    jj = torch.arange(T, device=dev)
    K, V = Kx[:T], Vx[:T]
    qg = qs.view(n, Hkv, G, D).permute(1, 2, 0, 3)  # [Hkv, G, n, D]
    s = torch.einsum("hgnd,thd->hgnt", qg, K)
    causal = jj[None, :] <= ipos[:, None]  # [n, T]
    s.masked_fill_(~causal[None, None], float("-inf"))
    p = s.softmax(-1)
    o = torch.einsum("hgnt,thd->hgnd", p, V)
    win = (jj[None, :] >= P - W) & causal  # SnapKV window (+ generated tokens)
    setm = {}
    for b in BUDGETS:
        m = torch.zeros(Hkv, T, dtype=torch.bool, device=dev)
        m.scatter_(1, sel[:, : b - W], True)
        setm[b] = m[:, None, None, :] | win[None, None]  # [Hkv, 1, n, T]
    farm = (jj[None, :] < P - W)[None, None]
    mfar = farm.expand_as(p)
    pfar = (p * mfar).sum(-1)
    for b in BUDGETS:
        cov = (p * setm[b]).sum(-1)
        acc[f"cov{b}"] += cov.sum((1, 2))
        acc[f"far{b}"] += ((cov - (p * win[None, None]).sum(-1)) / pfar.clamp(min=1e-12)).sum((1, 2))
    acc["ent"] += (-(p * torch.log(p.clamp(min=1e-30))).sum(-1)).sum((1, 2))
    ps = p.sort(-1, descending=True).values.cumsum(-1)
    acc["n90"] += ((ps < 0.9).sum(-1) + 1).float().sum((1, 2))
    del ps
    rec = jj[None, :] > ipos[:, None] - R  # within the r128 window (relative)
    acc["mfar128"] += (p * (~rec & causal)[None, None]).sum(-1).sum((1, 2))
    # reference after o_proj: |W_o o| of the full layer (all heads)
    of = o.permute(2, 0, 1, 3).reshape(n, H * D)
    on = (of @ Wo.t()).norm(dim=-1)  # [n]

    def errs(name, d):  # d [Hkv, G, n, D] output delta
        acc[f"e_{name}"] += _rel(d, o).sum((1, 2))
        dd = d.permute(0, 2, 1, 3).reshape(Hkv, n, G * D)  # group columns
        og = torch.stack([dd[h] @ Wo[:, h * G * D:(h + 1) * G * D].t() for h in range(Hkv)])  # [Hkv, n, hid]
        acc[f"o_{name}"] += (og.norm(dim=-1) / on.clamp(min=1e-12)[None]).sum(1) * G  # per query, x G (n*G norm.)
    for b in (512, 4096):
        pm = p * setm[b]
        od = torch.einsum("hgnt,thd->hgnd", pm, V) / pm.sum(-1, keepdim=True).clamp(min=1e-30)
        errs(f"snap{b}", od - o)
    sink = jj < SINKS
    exact_vq = (rec | sink[None, :])[None, None]  # [1,1,n,T]
    for c, (Kq, Vq) in kvq.items():
        sq = torch.einsum("hgnd,thd->hgnt", qg, Kq[:T])
        for kind, em in (("vq", exact_vq), ("hyb", setm[512] | sink[None, None, None, :])):
            sm = torch.where(em, s, sq)
            sm.masked_fill_(~causal[None, None], float("-inf"))
            pm = sm.softmax(-1)
            om = torch.einsum("hgnt,thd->hgnd", pm * em, V) + torch.einsum("hgnt,thd->hgnd", pm * ~em, Vq[:T])
            errs(f"{kind}_{c}", om - o)
            del sm, pm, om
        del sq
    acc["_n"] += n * G


def new_acc(Hkv, dev):
    a = {k: torch.zeros(Hkv, device=dev) for k in STATS}
    a["_n"] = 0
    return a


def fin(acc):
    return {k: (v / acc["_n"]).tolist() for k, v in acc.items() if k != "_n"}


@torch.no_grad()
def run_seq(f, ids, path, Kst, Vst, cs, kvqs, qchunk):
    """ids [P] prompt, path [N+1] target greedy tokens. Returns dict of per-layer stats (lists [nl][Hkv])."""
    P = ids.shape[0]
    cos, sin = cs
    dev = ids.device
    scale = 1.0 / math.sqrt(f.D)
    TE.SNAP["maxb"] = max(BUDGETS)
    obs, obs_ho, orc, sels = [], [], [], []
    # ---- phase 1: prefill (same ops as tau_eval.prefill_and_generate)
    h = f.mm.embed_tokens(ids)
    for l, layer in enumerate(f.mm.layers):
        q, k, v = f.qkv(layer, h)
        Kst[l][:P], Vst[l][:P] = k, v
        qr = TE.rp(q.float(), cos[0, :P], sin[0, :P]).to(torch.bfloat16)
        kr = TE.rp(k.float(), cos[0, :P], sin[0, :P]).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(qr.transpose(0, 1)[None], kr.repeat_interleave(f.G, 1).transpose(0, 1)[None],
                                           v.repeat_interleave(f.G, 1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
        sel = TE.snap_select(f, qr, kr, P)  # [Hkv, 4096 - 32]
        sels.append(sel)
        # obs-window stats: fp32 attention like tau_eval's draft attention (fp32 post-RoPE keys)
        Kx = TE.rp(k.float(), cos[0, :P], sin[0, :P])
        Vx = v.float()
        kvq = {}
        for c, fmt in CBS.items():
            Kq = kvqs[fmt].quant_pos(l, 0, Kx[None], [2], 0)[2][0]
            Vq = kvqs[fmt].quant_pos(l, 1, Vx[None], [2], 0)[2][0]
            kvq[c] = (Kq, Vq)
        Wo = layer.self_attn.o_proj.weight.float()
        qs = TE.rp(q[P - W:].float(), cos[0, P - W:P], sin[0, P - W:P]) * scale
        acc = new_acc(f.Hkv, dev)
        block_stats(f, Wo, qs, torch.arange(P - W, P, device=dev), Kx, Vx, kvq, sel, P, acc)
        obs.append(fin(acc))
        # split-half held-out coverage: SnapKV selection from obs queries 0..15, mass of queries 16..31
        obs_ho.append(holdout_cov(f, qr, kr, P))
        del Kx, Vx, kvq, Wo
        h = f.post(layer, h, o.view(P, -1))
    nxt0 = f.logits(h[-1:]).argmax(-1).item()
    del h
    # ---- phase 2: oracle stats over the generated positions (teacher-forced target)
    N = len(path) - 1
    pathT = torch.tensor(path, device=dev)
    Tt = P + N
    ht = f.mm.embed_tokens(pathT[:N])
    ipos = torch.arange(P, P + N, device=dev)
    for l, layer in enumerate(f.mm.layers):
        q, k, v = f.qkv(layer, ht)
        Kst[l][P:Tt], Vst[l][P:Tt] = k, v
        Kx = TE.rp(Kst[l][:Tt].float(), cos[0, :Tt], sin[0, :Tt])
        Vx = Vst[l][:Tt].float()
        qt = TE.rp(q.float(), cos[0, P:Tt], sin[0, P:Tt]) * scale
        o = TE.attend(qt, [(Kx, Vx, lambda i0, i1: torch.arange(Tt, device=dev)[None, :] <= ipos[i0:i1, None])])
        kvq = {}
        for c, fmt in CBS.items():
            kvq[c] = (kvqs[fmt].quant_pos(l, 0, Kx[None], [2], 0)[2][0], kvqs[fmt].quant_pos(l, 1, Vx[None], [2], 0)[2][0])
        Wo = layer.self_attn.o_proj.weight.float()
        acc = new_acc(f.Hkv, dev)
        for i0 in range(0, N, qchunk):
            i1 = min(N, i0 + qchunk)
            block_stats(f, Wo, qt[i0:i1], ipos[i0:i1], Kx, Vx, kvq, sels[l], P, acc)
        orc.append(fin(acc))
        ht = f.post(layer, ht, o.to(torch.bfloat16).view(N, -1))
        del Kx, Vx, kvq, Wo
    tgt = torch.cat([f.logits(ht[i:i + 512]).argmax(-1) for i in range(0, N, 512)])
    agree = (tgt == pathT[1:N + 1]).float().mean().item()
    return {"obs": obs, "obs_ho": obs_ho, "orc": orc, "tf_agree": agree, "first_tok_ok": nxt0 == path[0]}


@torch.no_grad()
def holdout_cov(f, qr, kr, P):
    """SnapKV selection from obs queries [P-32, P-16) (pool 5, summed over G), coverage of queries [P-16, P)."""
    w, h2 = W, W // 2
    q = qr[P - w:].float() / math.sqrt(f.D)
    k = kr.float()
    sc = torch.einsum("wkgd,tkd->kgwt", q.view(w, f.Hkv, f.G, f.D), k)  # [Hkv, G, w, P]
    jj = torch.arange(P, device=q.device)
    sc.masked_fill_(jj[None, None, None, :] > (P - w + torch.arange(w, device=q.device))[None, None, :, None], float("-inf"))
    p = sc.softmax(-1)
    a = p[:, :, :h2, : P - w].sum(2).reshape(f.H, -1)
    a = F.avg_pool1d(a[:, None], kernel_size=5, padding=2, stride=1)[:, 0].view(f.Hkv, f.G, -1).sum(1)
    out = {}
    pe = p[:, :, h2:]  # evaluation queries
    win = pe[..., P - w:].sum(-1)
    far = pe[..., : P - w].sum(-1)
    for b in BUDGETS:
        idx = a.topk(min(b - w, P - w), -1).indices
        m = torch.zeros(f.Hkv, P - w, dtype=torch.bool, device=q.device)
        m.scatter_(1, idx, True)
        cap = (pe[..., : P - w] * m[:, None, None]).sum(-1)
        out[f"covho{b}"] = (cap + win).mean((1, 2)).tolist()
        out[f"farho{b}"] = (cap / far.clamp(min=1e-12)).mean((1, 2)).tolist()
    return out


def timing_running():
    """True if an E7 timing run (e2e_real.py) is alive: this job must not share GPUs with it."""
    import subprocess
    r = subprocess.run(["pgrep", "-f", r"python[0-9.]* .*[e]2e_real\.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def prompts_for(tok, task, ctx, ids_needed, books_cache):
    if task in SYN_TASKS:
        if "b" not in books_cache:
            books_cache["b"] = load_haystack_books(tok)
        nmax = max(int(i.rsplit("_", 1)[1]) for i in ids_needed) + 1
        cands = syn_prompts(tok, task, ctx, nmax, books_cache["b"])
    else:
        cands = lb_prompts(tok, task, ctx, 100000, 0.5)
    return {c["id"]: c["ids"] for c in cands if c["id"] in ids_needed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--run", default="v1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qchunk", type=int, default=16)
    ap.add_argument("--cpu", action="store_true", help="smoke test on CPU (use with --trunc)")
    ap.add_argument("--trunc", type=int, default=None, help="smoke: keep only the last TRUNC prompt tokens (+BOS)")
    ap.add_argument("--gen_trunc", type=int, default=None, help="smoke: use only the first GEN_TRUNC generated tokens")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cpu" if a.cpu else "cuda"
    t0 = time.time()
    recs = tau_records.load()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
    f = TE.M(model)
    kvqs = {}
    for fmt in CBS.values():
        q = KVQuant(CB[fmt], dev="cpu") if a.cpu else make_quant(fmt)
        q.keep_first = SINKS
        kvqs[fmt] = q
    maxT = a.ctx + 300
    Kst = [torch.empty(maxT, f.Hkv, f.D, device=dev, dtype=torch.bfloat16) for _ in range(f.nl)]
    Vst = [torch.empty(maxT, f.Hkv, f.D, device=dev, dtype=torch.bfloat16) for _ in range(f.nl)]
    cs = f.cos_sin(torch.arange(maxT, device=dev))
    os.makedirs(OUT, exist_ok=True)
    books_cache = {}
    for task in a.tasks:
        if (task, a.ctx) in tau_records.UNREPRODUCIBLE:
            print("skip unreproducible", task, a.ctx, flush=True)
            continue
        keys = [k for k in recs if k[0] == task and k[1] == a.ctx]
        out_path = os.path.join(OUT, f"feat_{a.run}_{task}_ctx{a.ctx}.jsonl")
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(x)["id"] for x in open(out_path)}
        todo = [k for k in keys if k[2] not in done][: a.limit]
        if not todo:
            continue
        pr = prompts_for(tok, task, a.ctx, {k[2] for k in todo}, books_cache)
        print(task, a.ctx, "todo", len(todo), "prompts found", len(pr), f"{time.time() - t0:.0f}s", flush=True)
        for k in todo:
            if not a.cpu and timing_running():
                print("e2e_real.py timing process detected -> stop (resumable)", flush=True)
                sys.exit(3)
            e = recs[k]
            if k[2] not in pr:
                print("MISSING prompt", k, flush=True)
                continue
            ids = pr[k[2]]
            path = tok(e["text"], add_special_tokens=False).input_ids
            okP = len(ids) == e["P"]
            if a.trunc:
                ids = ids[:1] + ids[-a.trunc:]
            if a.gen_trunc:
                path = path[: a.gen_trunc + 1]
            t1 = time.time()
            r = run_seq(f, torch.tensor(ids, device=dev), path, Kst, Vst, cs, kvqs, a.qchunk)
            line = {"task": task, "ctx": a.ctx, "id": k[2], "P": len(ids), "P_rec": e["P"], "P_match": okP,
                    "N_retok": len(path) - 1, "N_rec": e["N"], "gen_rec": e["gen"], "smoke": bool(a.trunc or a.gen_trunc),
                    "tf_agree": r["tf_agree"], "first_tok_ok": r["first_tok_ok"],
                    "obs": r["obs"], "obs_ho": r["obs_ho"], "orc": r["orc"], "sec": time.time() - t1}
            with open(out_path, "a") as fo:
                fo.write(json.dumps(line) + "\n")
            if dev == "cuda":
                torch.cuda.empty_cache()
            om = lambda d, s: sum(sum(x) for x in (L[s] for L in d)) / (len(d) * f.Hkv)  # noqa: E731
            print(task, k[2], "P", len(ids), okP, "N", len(path) - 1, "tf_agree", round(r["tf_agree"], 4),
                  "first", r["first_tok_ok"], "cov4096", round(om(r["obs"], "cov4096"), 4),
                  "e_vq_def", round(om(r["obs"], "e_vq_def"), 4), "orc cov4096", round(om(r["orc"], "cov4096"), 4),
                  f"{time.time() - t1:.1f}s",
                  (f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB" if dev == "cuda" else ""), flush=True)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
