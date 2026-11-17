"""E7 magicvq step 1: greedy acceptance tau of compressed-KV self-speculative drafts on long PG-19 inputs.

Per prompt: bf16 prefill (SDPA), greedy AR generation of `gen` tokens with the bf16 target (bf16 KV), then one
teacher-forced pass over the generated region (layer by layer) that computes the target's verify-mode argmax and,
for every draft, G parallel "streams" (stream m = draft step m of a round, see mv_common round semantics).
tau(gamma) = tokens per round of the greedy round walk (accepted + 1). See mv_common.py for the draft definitions.
Usage: python experiments/26_magicvq/tau_eval.py --ctx 8192 --n 8 --gen 256 --run v1
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_common import CB, EOS, POST, make_quant, SINKS, kv_tokens_equiv, MODEL, OUTROOT, KVQuant, attend, bits_of, build_prompts, draft_specs, kivi_fake, \
    load_books, rope  # noqa: E402

DEFAULT_DRAFTS = (["exact"] + [f"evict{b}" for b in (128, 256, 512, 1024, 2048, 4096)]
                  + [f"{v}{'' if r == 0 else f'+r{r}'}" for v in ("vqaL2", "vqaL1", "vqeL1") for r in (0, 64, 128, 256, 512)]
                  + ["vqeL2", "vqcL1", "kivi4", "kivi2", "kivi2+r128"])


class M:
    def __init__(self, model):
        self.m, self.mm = model, model.model
        c = model.config
        self.H, self.Hkv, self.D = c.num_attention_heads, c.num_key_value_heads, c.hidden_size // c.num_attention_heads
        self.G = self.H // self.Hkv
        self.nl = c.num_hidden_layers

    def cos_sin(self, pos):
        x = torch.zeros(1, 1, 1, device=pos.device, dtype=torch.float32)
        c, s = self.mm.rotary_emb(x, pos[None])
        return c, s  # [1, T, D] fp32

    def qkv(self, layer, h):
        T = h.shape[0]
        x = layer.input_layernorm(h)
        a = layer.self_attn
        return (a.q_proj(x).view(T, self.H, self.D), a.k_proj(x).view(T, self.Hkv, self.D),
                a.v_proj(x).view(T, self.Hkv, self.D))

    def post(self, layer, h, o, chunk=2048):
        out = torch.empty_like(h)
        for i in range(0, h.shape[0], chunk):
            x = h[i:i + chunk] + layer.self_attn.o_proj(o[i:i + chunk])
            out[i:i + chunk] = x + layer.mlp(layer.post_attention_layernorm(x))
        return out

    def logits(self, h):
        return self.m.lm_head(self.mm.norm(h)).float()


def rp(x, c, s):
    """x [T, H, D] fp32, c/s [T, D] -> RoPE'd (HF rotate-half)."""
    return rope(x[None], c[None], s[None])[0]


SNAP = {"maxb": 0, "win": 32, "ks": 5, "idx": []}  # SnapKV selection (MagicDec Engine/SnapKV defaults)


@torch.no_grad()
def snap_select(f, qr, kr, P):
    """MagicDec / SnapKV prefill selection for one layer: softmax rows of the last `win` prompt queries over all prompt
    keys (causal inside the window), summed over the window rows, avg-pooled (kernel 5) over keys < P - win, summed
    over the G query heads of each kv-group; returns sorted top indices [Hkv, maxb - win] (prefix = smaller budget)."""
    w, ks = SNAP["win"], SNAP["ks"]
    q = qr[P - w:].float() / math.sqrt(f.D)  # [w, H, D]
    k = kr.float()  # [P, Hkv, D]
    sc = torch.einsum("wkgd,tkd->kgwt", q.view(w, f.Hkv, f.G, f.D), k).reshape(f.H, w, P)  # [H, w, P]
    jj = torch.arange(P, device=q.device)
    sc.masked_fill_(jj[None, None, :] > (P - w + torch.arange(w, device=q.device))[None, :, None], float("-inf"))
    a = sc.softmax(-1)[..., : P - w].sum(1)  # [H, P - w]
    a = F.avg_pool1d(a[:, None], kernel_size=ks, padding=ks // 2, stride=1)[:, 0]
    a = a.view(f.Hkv, f.G, -1).sum(1)
    return a.topk(min(SNAP["maxb"] - w, P - w), -1).indices  # [Hkv, n]


@torch.no_grad()
def prefill_and_generate(f, ids, gen, Kst, Vst, cs):
    """ids [P] -> path (list of gen token ids). Kst/Vst [nl] of [P+gen, Hkv, D] bf16 (pre-RoPE K, V) filled for 0..P-1
    by prefill and P.. by AR decode."""
    P = ids.shape[0]
    cos, sin = cs
    h = f.mm.embed_tokens(ids)
    for l, layer in enumerate(f.mm.layers):
        q, k, v = f.qkv(layer, h)
        Kst[l][:P], Vst[l][:P] = k, v
        qr = rp(q.float(), cos[0, :P], sin[0, :P]).to(torch.bfloat16)
        kr = rp(k.float(), cos[0, :P], sin[0, :P]).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(qr.transpose(0, 1)[None], kr.repeat_interleave(f.G, 1).transpose(0, 1)[None],
                                           v.repeat_interleave(f.G, 1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
        if SNAP["maxb"]:
            if l == 0:
                SNAP["idx"] = []
            SNAP["idx"].append(snap_select(f, qr, kr, P))
        h = f.post(layer, h, o.view(P, -1))
    if gen == 0:
        return []
    nxt = f.logits(h[-1:]).argmax(-1)
    path = [nxt.item()]
    del h
    for t in range(gen - 1):
        if path[-1] in EOS:
            break
        p = P + t
        h = f.mm.embed_tokens(nxt)
        for l, layer in enumerate(f.mm.layers):
            q, k, v = f.qkv(layer, h)
            Kst[l][p], Vst[l][p] = k[0], v[0]
            qr = rp(q.float(), cos[0, p:p + 1], sin[0, p:p + 1]).to(torch.bfloat16)  # [1,H,D]
            kr = rp(Kst[l][:p + 1].float(), cos[0, :p + 1], sin[0, :p + 1]).to(torch.bfloat16)
            qq = qr.view(f.Hkv, f.G, f.D)[None]  # [1,Hkv,G,D]
            o = F.scaled_dot_product_attention(qq, kr.transpose(0, 1)[None], Vst[l][:p + 1].transpose(0, 1)[None])
            h = f.post(layer, h, o.reshape(1, -1))
        nxt = f.logits(h).argmax(-1)
        path.append(nxt.item())
    return path


def walk(matches, gamma, T):
    i, iters, acc = 0, 0, 0
    while i < T:
        a = 0
        while a < gamma and i + a < T and matches[a][i + a]:
            a += 1
        acc += a; iters += 1; i += a + 1
    return T, iters, acc


@torch.no_grad()
def tf_pass(f, ids_full, P, Kst, Vst, cs, specs, kvqs, G, metrics=False):
    """Teacher-forced pass over the generated region (query positions P .. P+N-1, N = len(ids_full) - P - 1).
    Returns target argmax [N] and per draft predictions [G, N] (argmax), step-1 TV is not computed."""
    N = ids_full.shape[0] - P - 1
    Tt = P + N  # total positions with KV (0 .. P+N-1)
    cos, sin = cs[0][0], cs[1][0]
    dev = ids_full.device
    scale = 1.0 / math.sqrt(f.D)
    xg = ids_full[P:P + N]
    ht = f.mm.embed_tokens(xg)
    hd = {n: ht[None].expand(G, -1, -1).contiguous() for n, s in specs.items()}
    ipos = torch.arange(P, P + N, device=dev)  # query positions
    jall = torch.arange(Tt, device=dev)
    for l, layer in enumerate(f.mm.layers):
        # ---- target (exact bf16 KV, causal)
        q, k, v = f.qkv(layer, ht)
        Kst[l][P:P + N], Vst[l][P:P + N] = k, v
        Kpre, Vx = Kst[l][:Tt].float(), Vst[l][:Tt].float()
        Kx = rp(Kpre, cos[:Tt], sin[:Tt])
        qt = rp(q.float(), cos[P:P + N], sin[P:P + N]) * scale

        def m_le(off, lo=0, hi=Tt):
            jj = jall[lo:hi]
            return lambda i0, i1: jj[None, :] <= (ipos[i0:i1, None] - off)
        o = attend(qt, [(Kx, Vx, m_le(0))])
        ht_new = f.post(layer, ht, o.to(torch.bfloat16).view(N, -1))
        # ---- drafts: G streams each
        def run_drafts(names, ctxs):
          for n in names:
              s = specs[n]
              h = hd[n]  # [G, N, hid]
              qs, ks, vs = f.qkv(layer, h.view(G * N, -1))
              qs = qs.view(G, N, f.H, f.D)
              ks, vs = ks.view(G, N, f.Hkv, f.D).float(), vs.view(G, N, f.Hkv, f.D).float()
              ksr = rope(ks, cos[None, P:P + N].expand(G, -1, -1), sin[None, P:P + N].expand(G, -1, -1))
              if s["kind"] == "vq" and s["R"] == 0:  # own in-round KV stored VQ-encoded
                  q_ = kvqs[s["fmt"]]
                  lk, lv = s["lk"], s["lv"]
                  if lk and s["fmt"] in POST:
                      ksr = q_.quant(l, 0, ksr, [lk])[lk]
                  elif lk:
                      ksr = rope(q_.quant(l, 0, ks, [lk])[lk], cos[None, P:P + N].expand(G, -1, -1),
                                 sin[None, P:P + N].expand(G, -1, -1))
                  if lv:
                      vs = q_.quant(l, 1, vs, [lv])[lv]
              out = torch.empty(G, N, f.H, f.D, device=dev, dtype=torch.bfloat16)
              for m in range(1, G + 1):
                  qm = rp(qs[m - 1].float(), cos[P:P + N], sin[P:P + N]) * scale
                  # in-round keys: stream m' (1..m) at query index t - m + m'
                  tix = torch.arange(N, device=dev)
                  Ke, Ve, val = [], [], []
                  for mp in range(1, m + 1):
                      tt = tix - m + mp
                      ok = tix - m + 1 >= 0  # round start >= P
                      tc = tt.clamp(min=0)
                      Ke.append(ksr[mp - 1, tc]); Ve.append(vs[mp - 1, tc]); val.append(ok)
                  extra = (torch.stack(Ke), torch.stack(Ve), torch.stack(val))
                  if s["kind"] == "exact":
                      parts = [(Kx, Vx, m_le(m))]
                  elif s["kind"] == "topk":  # oracle: exact attention restricted to the top-k scores per query head
                      parts = [(Kx, Vx, m_le(m), s["budget"])]
                  elif s["kind"] == "snapkv":
                      sel = SNAP["idx"][l][:, : s["budget"] - SNAP["win"]]  # [Hkv, n] per kv-head token indices
                      ns = sel.shape[1]
                      Ks = torch.gather(Kx.transpose(0, 1), 1, sel[..., None].expand(-1, -1, f.D)).transpose(0, 1)
                      Vs = torch.gather(Vx.transpose(0, 1), 1, sel[..., None].expand(-1, -1, f.D)).transpose(0, 1)
                      lo = P - SNAP["win"]
                      parts = [(Ks, Vs, lambda i0, i1, ns=ns: torch.ones(i1 - i0, ns, dtype=torch.bool, device=dev)),
                               (Kx[lo:Tt], Vx[lo:Tt], m_le(m, lo, Tt))]
                  elif s["kind"] == "evict":
                      W, S0 = s["budget"] - s["sinks"], s["sinks"]
                      lo = max(S0, P - W - G)
                      jw = jall[lo:Tt]
                      parts = [(Kx[:S0], Vx[:S0], m_le(m, 0, S0)),
                               (Kx[lo:Tt], Vx[lo:Tt],
                                lambda i0, i1, jw=jw, m=m, W=W: (jw[None, :] <= ipos[i0:i1, None] - m)
                                & (jw[None, :] > ipos[i0:i1, None] - W))]
                  elif s["kind"] in ("vq", "kivi"):
                      Kc, Vc = ((ctxs[(s["fmt"], "K", s["lk"])], ctxs[(s["fmt"], "V", s["lv"])]) if s["kind"] == "vq"
                                else ctxs[("kivi", s["bits"], s["group"])])
                      R = s.get("R", 0)
                      if R == 0:
                          parts = [(Kc, Vc, m_le(m))]
                      else:
                          lo = max(0, P - R - G)
                          jw = jall[lo:Tt]
                          parts = [(Kc, Vc, lambda i0, i1, m=m, R=R: (jall[None, :] <= ipos[i0:i1, None] - m)
                                    & (jall[None, :] <= ipos[i0:i1, None] - R)),
                                   (Kx[lo:Tt], Vx[lo:Tt], lambda i0, i1, jw=jw, m=m, R=R:
                                    (jw[None, :] <= ipos[i0:i1, None] - m) & (jw[None, :] > ipos[i0:i1, None] - R))]
                  out[m - 1] = attend(qm, parts, extra).to(torch.bfloat16)
              hd[n] = f.post(layer, h.reshape(G * N, -1), out.view(G * N, -1)).view(G, N, -1)
              del qs, ks, vs, ksr, out
        # ---- drafts, grouped by compressed context (computed once per group, freed after the group)
        def ctx_key(s):
            return ("vq", s["fmt"]) if s["kind"] == "vq" else ("kivi", s["bits"], s["group"]) if s["kind"] == "kivi" else None
        groups = {}
        for n, s in specs.items():
            groups.setdefault(ctx_key(s), []).append(n)
        for key, names in groups.items():
            ctxs = {}
            if key is not None and key[0] == "vq":
                q_ = kvqs[key[1]]
                lks = sorted({specs[n]["lk"] for n in names} - {0})
                lvs = sorted({specs[n]["lv"] for n in names} - {0})
                post = key[1] in POST
                ctxs[(key[1], "K", 0)], ctxs[(key[1], "V", 0)] = Kx, Vx
                if lks:
                    rk = q_.quant_pos(l, 0, (Kx if post else Kpre)[None], lks, 0)
                    for lk in lks:
                        ctxs[(key[1], "K", lk)] = rk[lk][0] if post else rp(rk[lk][0], cos[:Tt], sin[:Tt])
                    del rk
                if lvs:
                    rv = q_.quant_pos(l, 1, Vx[None], lvs, 0)
                    for lv in lvs:
                        ctxs[(key[1], "V", lv)] = rv[lv][0]
                    del rv
            elif key is not None:
                b, gsz = key[1], key[2]
                kq = kivi_fake(Kpre, b, 0, gsz); kq[:SINKS] = Kpre[:SINKS]
                vq = kivi_fake(Vx, b, 1, gsz); vq[:SINKS] = Vx[:SINKS]
                ctxs[("kivi", b, gsz)] = (rp(kq, cos[:Tt], sin[:Tt]), vq)
                del kq, vq
            run_drafts(names, ctxs)
            del ctxs
        ht = ht_new
        del Kx, Vx, Kpre
    if metrics:  # step-1 draft (stream 0) vs target, teacher-forced on ids_full; labels = next tokens
        y = ids_full[P + 1:P + 1 + N]
        mets = {}
        lts = [f.logits(ht[i:i + 256]).log_softmax(-1) for i in range(0, N, 256)]
        mets["target"] = {"nll": sum(F.nll_loss(lt, y[i * 256:i * 256 + lt.shape[0]], reduction="sum").item()
                                     for i, lt in enumerate(lts)), "n": N}
        for n, h in hd.items():
            a = {"nll": 0.0, "kl": 0.0, "tv": 0.0, "top1": 0.0, "n": N}
            for i, lt in enumerate(lts):
                ld = f.logits(h[0, i * 256:i * 256 + lt.shape[0]]).log_softmax(-1)
                yy = y[i * 256:i * 256 + lt.shape[0]]
                a["nll"] += F.nll_loss(ld, yy, reduction="sum").item()
                pt = lt.exp()
                a["kl"] += (pt * (lt - ld)).sum().item()
                a["tv"] += 0.5 * (pt - ld.exp()).abs().sum().item()
                a["top1"] += (lt.argmax(-1) == ld.argmax(-1)).sum().item()
            mets[n] = a
        del lts
        return mets
    tgt = torch.cat([f.logits(ht[i:i + 512]).argmax(-1) for i in range(0, N, 512)])
    preds = {}
    for n, h in hd.items():
        hf = h.reshape(G * N, -1)
        preds[n] = torch.cat([f.logits(hf[i:i + 512]).argmax(-1) for i in range(0, G * N, 512)]).view(G, N)
    return tgt, preds


@torch.no_grad()
def run_qual(a, f, tok, books, specs, kvqs, Kst, Vst, cs, out_path, t0):
    per = []
    for bi, b in enumerate(books[:a.n][: a.limit]):
        ids = torch.tensor([tok.bos_token_id] + b[: a.ctx - 1 + a.gen + 1], device="cuda")  # prompt ctx + gen + 1
        P = a.ctx
        prefill_and_generate(f, ids[:P], 0, Kst, Vst, cs)
        mets = tf_pass(f, ids, P, Kst, Vst, cs, specs, kvqs, 1, metrics=True)
        per.append({"idx": bi, "P": P, "metrics": mets})
        torch.cuda.empty_cache()
        print("qual", bi, P, f"{time.time() - t0:.0f}s", {n: (round(v["kl"] / v["n"], 3), round(v["top1"] / v["n"], 3))
                                                          for n, v in mets.items() if n != "target"}, flush=True)
    agg = {}
    for n in ["target"] + list(specs):
        tot = {k: sum(r["metrics"][n][k] for r in per) for k in per[0]["metrics"][n]}
        agg[n] = {"ppl": math.exp(tot["nll"] / tot["n"]), "n_tokens": tot["n"]}
        if n != "target":
            agg[n].update({"kl_target_draft": tot["kl"] / tot["n"], "tv": tot["tv"] / tot["n"],
                           "top1_agree": tot["top1"] / tot["n"]})
    res = {"config": {**vars(a), "specs": specs, "bits": {n: bits_of(s, kvqs) for n, s in specs.items()},
                      "kv_tokens_equiv": {n: kv_tokens_equiv(s, kvqs, a.ctx) for n, s in specs.items()}},
           "agg": agg, "per_prompt": per, "wall_s": time.time() - t0}
    json.dump(res, open(out_path, "w"), indent=1)
    for n, v in agg.items():
        print(n, json.dumps({k: round(x, 4) for k, x in v.items()}), flush=True)
    print("done", out_path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=["pg19", "chat"])
    ap.add_argument("--drafts", nargs="+", default=DEFAULT_DRAFTS)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--run", default="v1")
    ap.add_argument("--limit", type=int, default=None, help="debug: prompts per set")
    ap.add_argument("--dump_matches", action="store_true")
    ap.add_argument("--mode", choices=["tau", "qual"], default="tau",
                    help="qual: teacher-forced step-1 draft vs target on the TRUE book continuation (gen tokens after "
                         "the ctx-token PG-19 prompt): TV, KL(target||draft), top-1 agreement, PPL")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    books = load_books(tok, 2 * a.n, 8192 + 32768 + 1024)
    if a.mode == "qual":
        a.sets = ["pg19"]
        a.G = 1
    sets = build_prompts(tok, books, a.ctx, a.n, a.sets)
    print("prompts", {k: [len(x) for x in v] for k, v in sets.items()}, f"{time.time() - t0:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    f = M(model)
    specs = draft_specs(a.drafts)
    specs = {n: s for n, s in specs.items() if not (s["kind"] in ("evict", "snapkv", "topk") and s["budget"] >= a.ctx)}
    SNAP["maxb"] = max([s["budget"] for s in specs.values() if s["kind"] == "snapkv"], default=0)
    kvqs = {}
    for s in specs.values():
        if s["kind"] == "vq" and s["fmt"] not in kvqs:
            kvqs[s["fmt"]] = make_quant(s["fmt"])
            kvqs[s["fmt"]].keep_first = 0 if (s["fmt"].startswith("cv") and s["fmt"].endswith("n")) else SINKS
    maxT = max(len(x) for v in sets.values() for x in v) + a.gen + 8
    Kst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    Vst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    pos = torch.arange(maxT, device="cuda")
    cs = f.cos_sin(pos)
    out_dir = os.path.join(OUTROOT, "tau")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{a.mode}_{a.run}_ctx{a.ctx}.json")
    if a.mode == "qual":
        return run_qual(a, f, tok, books, specs, kvqs, Kst, Vst, cs, out_path, t0)
    res = {"config": {**vars(a), "specs": specs, "bits": {n: bits_of(s, kvqs) for n, s in specs.items()},
                      "kv_tokens_equiv": {n: kv_tokens_equiv(s, kvqs, a.ctx) for n, s in specs.items()},
                      "model": MODEL}, "per_prompt": []}
    for sname, prompts in sets.items():
        for pi, p in enumerate(prompts[: a.limit]):
            t1 = time.time()
            ids = p.cuda()
            P = ids.shape[0]
            path = prefill_and_generate(f, ids, a.gen, Kst, Vst, cs)
            t2 = time.time()
            full = torch.cat([ids, torch.tensor(path, device="cuda")])
            tgt, preds = tf_pass(f, full, P, Kst, Vst, cs, specs, kvqs, a.G)
            N = tgt.shape[0]
            pathn = torch.tensor(path[1:], device="cuda")
            rec = {"set": sname, "idx": pi, "P": P, "gen": len(path), "N": N,
                   "target_tf_vs_gen": (tgt == pathn).float().mean().item(), "drafts": {}}
            for n, pr in preds.items():
                mt = (pr == tgt[None])  # [G, N]
                matches = mt.tolist()
                d = {"match_rate_step": mt.float().mean(1).tolist()}
                if a.dump_matches:  # per-position step-m matches (for replaying acceptance in the e2e timing loop)
                    d["matches"] = [[int(x) for x in row] for row in matches]
                for g in a.gammas:
                    if g <= a.G:
                        d[f"g{g}"] = walk(matches, g, N)
                rec["drafts"][n] = d
            res["per_prompt"].append(rec)
            torch.cuda.empty_cache()
            print(sname, pi, P, len(path), f"gen {t2 - t1:.0f}s tf {time.time() - t2:.0f}s",
                  {n: (round(d["match_rate_step"][0], 3), round(d["g4"][0] / d["g4"][1], 2)) for n, d in
                   rec["drafts"].items()}, f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            json.dump(res, open(out_path + ".partial", "w"))
    # aggregate
    agg = {}
    for sname in sets:
        per = [r for r in res["per_prompt"] if r["set"] == sname]
        if not per:
            continue
        agg[sname] = {"n_prompts": len(per), "mean_P": sum(r["P"] for r in per) / len(per),
                      "mean_N": sum(r["N"] for r in per) / len(per),
                      "target_tf_vs_gen": sum(r["target_tf_vs_gen"] for r in per) / len(per), "drafts": {}}
        for n in specs:
            A = {"alpha_step": [sum(r["drafts"][n]["match_rate_step"][m] * r["N"] for r in per) / sum(r["N"] for r in per)
                                for m in range(a.G)]}
            for g in a.gammas:
                if g <= a.G:
                    toks = sum(r["drafts"][n][f"g{g}"][0] for r in per)
                    its = sum(r["drafts"][n][f"g{g}"][1] for r in per)
                    per_tau = [r["drafts"][n][f"g{g}"][0] / r["drafts"][n][f"g{g}"][1] for r in per]
                    A[f"tau_g{g}"] = toks / its
                    A[f"tau_g{g}_per_prompt_min_max"] = [min(per_tau), max(per_tau)]
            agg[sname]["drafts"][n] = A
    res["agg"] = agg
    res["wall_s"] = time.time() - t0
    json.dump(res, open(out_path, "w"), indent=1)
    for sname, A in agg.items():
        print(sname, json.dumps({n: [round(v["alpha_step"][0], 3)] + [round(v[f"tau_g{g}"], 2) for g in a.gammas]
                                 for n, v in A["drafts"].items()}), flush=True)
    print("done", out_path, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
