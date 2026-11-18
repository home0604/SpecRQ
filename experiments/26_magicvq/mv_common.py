"""E7 magicvq: shared pieces for the tau evaluation of compressed-KV self-speculative drafts (MagicDec-style).

Target = Llama-3.1-8B-Instruct, bf16 weights, bf16 full KV. Draft = the same weights, reading a compressed KV:
  evict<B>   : StreamingLLM / MagicDec eviction, 16 attention sinks + the most recent B - 16 tokens (MagicDec's
               KVCache.prefill_draft keeps cache[:, :16] + the last kv_len - 16 tokens). True absolute RoPE positions
               are used (MagicDec re-indexes positions inside its draft buffer; differs only in the sink-query distance).
  vq<fmt>L<k>: every token of the context VQ-encoded (pre-RoPE K, plain V, per-(layer, K/V, kv-head) codebooks,
               channel-standardized, Meta RoPE-pair layout; codebooks fitted on pile-val KV), first k stages read;
               positions 0..15 (attention sinks, same 16 as the eviction draft) kept bf16.
  vq..+r<R>  : hybrid, bf16 for the R most recent positions (relative to the query) and VQ before.
  kivi<b>    : KIVI-like scalar baseline, K per-channel asymmetric over groups of 128 tokens, V per-token asymmetric
               over the 128 channels of a head (b-bit codes + fp16 scale/zero: b + 0.25 bit/elem); in-round tokens bf16.
Round semantics (exact for greedy acceptance, teacher-forced along the target's greedy path): draft step m (1..G) of a
round that starts at position p queries position i = p + m - 1 and reads (a) the context, positions j <= i - m = p - 1:
the TARGET's KV (computed by the target; for compressed drafts in the draft's compressed form, i.e. accepted tokens
are re-encoded from the verified target KV), restricted by the eviction window for evict drafts, and (b) its own KV
for positions p .. i, computed by draft steps 1..m (stored in the draft's format: VQ drafts encode their own new
tokens, eviction / KIVI / hybrid keep them bf16). Attention for target and drafts: fp32 scores/softmax/PV.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "25_kvtau"))
from kvtau_common import KVQuant, rope  # noqa: E402  (read-only reuse)

MODEL = "/data/models/llama3.1_8b_instruct"
PG19 = "/data/lsh/datasets/pg19-test/data"
OUTROOT = "/data/lsh/results_accept/magicvq"
CB = {  # format -> codebook file (bf16-weight KV, pile-val, chnorm, greedy RQ)
    "a": "/data/lsh/results_accept/kvtau/bf16/cb_a_d8K256x2_cn_greedy.pt",   # d8 K256 x 2 : 1 bit / stage
    "c": "/data/lsh/results_accept/kvtau/bf16/cb_c_d8K16x4_cn_greedy.pt",    # d8 K16 x 4  : 0.5 bit / stage
    "e": os.path.join(OUTROOT, "cb", "cb_d16K256x2_cn_greedy.pt"),           # d16 K256 x 2: 0.5 bit / stage
    "p": os.path.join(OUTROOT, "cb", "cb_d8K256x2_post_cn_greedy.pt"),       # POST-RoPE K (HF layout) d8 K256 x 2
    "q": os.path.join(OUTROOT, "cb", "cb_d16K256x2_post_cn_greedy.pt"),      # POST-RoPE K (HF layout) d16 K256 x 2
}
POST = {"p", "q"}
for _t in ("t1", "t1w", "t2", "t2w", "t1x", "t1wx", "t2x", "t2wx", "t2mw"):  # v6 (v7: t2mw = MSE-loss control): 'p' format, draft-trained (rqd_train.py)
    CB["p" + _t] = os.path.join(OUTROOT, "cb", f"cb_d8K256x2_post_cn_{_t}.pt")
    POST.add("p" + _t)
CB["g8"] = os.path.join(OUTROOT, "cb", "gsrq_D8C256S2.pt")    # GSRQ (VQLLM port) D8 C256 x 2: 1 bit / stage
CB["g16"] = os.path.join(OUTROOT, "cb", "gsrq_D16C256S2.pt")  # GSRQ D16 C256 x 2: 0.5 bit / stage
for _t, _f in (("gv8", "gsrq_D8C256S2_vn"), ("gv16", "gsrq_D16C256S2_vn"), ("gk8", "gsrq_D8C256S2_kl"),
               ("gk16", "gsrq_D16C256S2_kl"), ("gkv8", "gsrq_D8C256S2_vn_kl"), ("gkv16", "gsrq_D16C256S2_vn_kl")):
    CB[_t] = os.path.join(OUTROOT, "cb", _f + ".pt")  # vn: V per-token gain; kl: attention-aware trained


def make_quant(fmt):
    if fmt.startswith("cv"):  # CommVQ: cv1 / cv2 (official gumbel encode), cv1a / cv2a (argmax encode); suffix n: no sinks
        from commvq_quant import CVQuant
        return CVQuant(int(fmt[2]), "argmax" if "a" in fmt[3:] else "gumbel")
    if fmt.startswith("g"):
        from gsrq import GSQuant
        return GSQuant(CB[fmt])
    return KVQuant(CB[fmt])  # formats whose K codes are post-RoPE (the specrq/kvq kernel convention)
EOS = {128001, 128008, 128009}
QUESTIONS = [
    "Summarize what happens in this passage in five sentences.",
    "Who are the main characters in this text, and how would you describe each of them?",
    "Describe the setting of this passage and the mood it creates.",
    "What is the central conflict in this passage? Explain briefly.",
    "List the most important events in this passage in order.",
    "Write a short review of this passage as if for a literary magazine.",
    "Explain what the narrator thinks about the other characters.",
    "What questions does this passage leave unanswered? List them.",
]


def draft_specs(names):
    """name -> dict(kind, ...). Bits/elem and bytes-equivalent window are filled in by bits_of()."""
    out = {}
    for n in names:
        if n == "exact":
            out[n] = {"kind": "exact"}
        elif n.startswith("snapkv"):
            out[n] = {"kind": "snapkv", "budget": int(n[6:]), "window": 32, "kernel": 5}
        elif n.startswith("topk"):
            out[n] = {"kind": "topk", "budget": int(n[4:])}
        elif n.startswith("evict"):
            out[n] = {"kind": "evict", "budget": int(n[5:]), "sinks": 16}
        elif n.startswith("kivi"):
            base, _, r = n.partition("+r")
            b, _, g = base[4:].partition("g")
            out[n] = {"kind": "kivi", "bits": int(b), "group": int(g) if g else 128, "R": int(r) if r else 0}
        elif n.startswith("hyb"):  # v4 hybrid: SnapKV-selected W tokens bf16 + every other token VQ (disjoint sets)
            import re
            m = re.fullmatch(r"hyb(\d+)([a-z0-9_]+?)L(\d)", n)
            assert m, n
            k = int(m.group(3))
            out[n] = {"kind": "hyb", "budget": int(m.group(1)), "fmt": m.group(2), "level": k, "lk": k, "lv": k, "R": 0}
        elif n.startswith("vq"):
            import re
            m = re.fullmatch(r"vq([a-z0-9_]+?)(?:L(\d)|K(\d)V(\d))(?:\+r(\d+))?", n)
            assert m, n
            fmt = m.group(1)
            lk, lv = (int(m.group(2)),) * 2 if m.group(2) else (int(m.group(3)), int(m.group(4)))
            out[n] = {"kind": "vq", "fmt": fmt, "level": max(lk, lv), "lk": lk, "lv": lv,
                      "R": int(m.group(5)) if m.group(5) else 0}  # level 0 on a side = that side kept bf16
        else:
            raise ValueError(n)
    return out


SINKS = 16


def kv_tokens_equiv(spec, kvqs, ctx):
    """Draft KV bytes read per step (per layer-head, in bf16-token units) at context ctx: eviction budget, or
    ctx * bits / 16 for the compressed part + bf16 sinks + bf16 recent window (codebooks / scales not counted)."""
    if spec["kind"] in ("evict", "snapkv", "topk"):  # snapkv: budget at decode start (+ generated tokens appended)
        return min(spec["budget"], ctx)
    if spec["kind"] == "exact":
        return ctx
    R = spec.get("R", 0)
    cbt = 0.0
    if spec["kind"] == "hyb":  # W bf16 (SnapKV set incl. its 32-token window) + the other ctx - W tokens VQ
        R = spec["budget"] - SINKS
    if spec["kind"] in ("vq", "hyb"):
        q = kvqs[spec["fmt"]]
        cbt = (q.cb_bytes() if hasattr(q, "cb_bytes") else
               (cb_bytes(q, spec["lk"]) + cb_bytes(q, spec["lv"])) / 2) / BF16_TOKEN_BYTES
    return (ctx - SINKS - R) * bits_of(spec, kvqs) / 16 + SINKS + R + cbt


BF16_TOKEN_BYTES = 32 * 2 * 8 * 128 * 2  # Llama-3.1-8B: layers x (K, V) x kv-heads x head-dim x 2 B


def cb_bytes(q, level):
    """bytes of the codebooks (fp16 / bf16) read by a draft step using `level` stages (all layers, K and V); amortized
    over the batch in a real system (counted here per sequence = B = 1 view)."""
    n = 0
    for l in range(len(q.cb)):
        for kv in range(2):
            for st in q.cb[l][kv][:level]:
                n += sum(t.numel() for t in (st if isinstance(st, tuple) else (st,)))
    return 2 * n


def bits_of(spec, kvqs):
    """bits / element averaged over K and V (a side at level 0 counts 16)."""
    if spec["kind"] in ("vq", "hyb"):
        q = kvqs[spec["fmt"]]
        if hasattr(q, "bits_k"):  # CommVQ: fixed rate per side
            bk, bv = q.bits_k(), q.bits_v()
        else:
            bk = sum(math.log2(c) for c in q.sizes[:spec["lk"]]) / q.d
            bv = sum(math.log2(c) for c in q.sizes[:spec["lv"]]) / q.d + (16 / 128 if getattr(q, "vnorm", False) else 0)
        bk = bk if spec["lk"] else 16.0
        bv = bv if spec["lv"] else 16.0
        return (bk + bv) / 2
    if spec["kind"] == "kivi":
        return spec["bits"] + 32 / spec["group"]  # fp16 scale + zero per group
    return 16.0


# ---------------------------------------------------------------- prompts
def load_books(tok, n, min_tokens, max_chars=600000):
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19}/*.parquet")))
    books = []
    for text in df["text"]:
        ids = tok(text[:max_chars], add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            books.append(ids)
        if len(books) >= n:
            break
    return books


def build_prompts(tok, books, ctx, n, sets=("pg19", "chat")):
    """pg19: BOS + first ctx-1 tokens of book i (raw continuation, MagicDec-style); chat: Llama-3.1 chat template with
    an excerpt from token 8192 of book n+i plus a question (instruction-style long context; built locally, no
    standard long-context instruction benchmark is cached)."""
    bos = [tok.bos_token_id]
    out = {}
    if "pg19" in sets:
        out["pg19"] = [torch.tensor(bos + b[: ctx - 1]) for b in books[:n]]
    if "chat" in sets:
        out["chat"] = []
        for i, b in enumerate(books[n: 2 * n]):
            q = QUESTIONS[i % len(QUESTIONS)]
            lo, hi, best = 50, ctx, None
            while lo <= hi:
                mid = (lo + hi) // 2
                doc = tok.decode(b[8192: 8192 + mid])
                msg = [{"role": "user", "content": f"Read the following text.\n\n<text>\n{doc}\n</text>\n\n{q}"}]
                ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
                if len(ids) <= ctx:
                    best, lo = ids, mid + 1
                else:
                    hi = mid - 1
            out["chat"].append(torch.tensor(best))
    return out


# ---------------------------------------------------------------- compressors
def kivi_fake(x, bits, kv, group=128):
    """x [T, H, D] fp32 pre-RoPE K (kv=0) or V (kv=1) -> recon. K: per-channel over groups of `group` consecutive
    tokens; V: per-token over groups of `group` channels (KIVI: asymmetric, fp16 scale / zero per group)."""
    T, H, D = x.shape
    L = 2 ** bits - 1
    if kv == 0:
        pad = (-T) % group
        xp = torch.cat([x, x[-1:].expand(pad, H, D)]) if pad else x
        g = xp.view(-1, group, H, D)
        lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
        s = ((hi - lo) / L).clamp(min=1e-6)
        r = (torch.round((g - lo) / s).clamp(0, L) * s + lo).view(-1, H, D)[:T]
        return r
    xg = x.unflatten(-1, (D // group, group))
    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    s = ((hi - lo) / L).clamp(min=1e-6)
    return (torch.round((xg - lo) / s).clamp(0, L) * s + lo).flatten(-2)


# ---------------------------------------------------------------- attention
def attend(q, parts, extra=None, qchunk=64):
    """q [Nq, H, D] fp32 (RoPE'd, scaled). parts: list of (K [Tk, Hkv, D] fp32 post-RoPE, V [Tk, Hkv, D] fp32,
    mask(i0, i1) -> bool [i1 - i0, Tk] allowed[, topk: keep only the k largest allowed scores per query head]). extra: (Ke [m, Nq, Hkv, D], Ve, valid [m, Nq] bool) per-query keys
    (in-round). Returns [Nq, H, D] fp32."""
    Nq, H, D = q.shape
    Hkv = parts[0][0].shape[1]
    Gq = H // Hkv
    out = torch.empty_like(q)
    if any(len(p) > 3 for p in parts):
        qchunk = min(qchunk, 16)  # topk values / indices of the full score rows
    Kt = [p[0].permute(1, 2, 0) for p in parts]  # [Hkv, D, Tk]
    Vt = [p[1].permute(1, 0, 2) for p in parts]  # [Hkv, Tk, D]
    for i0 in range(0, Nq, qchunk):
        i1 = min(Nq, i0 + qchunk)
        n = i1 - i0
        qq = q[i0:i1].view(n, Hkv, Gq, D).permute(1, 2, 0, 3).reshape(Hkv, Gq * n, D)
        ss = []
        for part, kt in zip(parts, Kt):
            mf = part[2]
            s = torch.bmm(qq, kt).view(Hkv, Gq, n, -1)
            s.masked_fill_(~mf(i0, i1)[None, None], float("-inf"))
            if len(part) > 3 and part[3] < s.shape[-1]:
                thr = s.topk(part[3], -1).values[..., -1:]
                s.masked_fill_(s < thr, float("-inf"))
            ss.append(s)
        if extra is not None:
            Ke, Ve, valid = extra
            ke = Ke[:, i0:i1].permute(2, 1, 0, 3)  # [Hkv, n, m, D]
            se = torch.einsum("hgnd,hnmd->hgnm", qq.view(Hkv, Gq, n, D), ke)
            se.masked_fill_(~valid[:, i0:i1].t()[None, None], float("-inf"))
            ss.append(se)
        mx = torch.stack([s.amax(-1) for s in ss], -1).amax(-1, keepdim=True)  # [Hkv,Gq,n,1]
        ps = [torch.exp(s - mx) for s in ss]
        den = sum(p.sum(-1, keepdim=True) for p in ps)
        o = 0
        for p, vt in zip(ps[:len(parts)], Vt):
            o = o + torch.bmm(p.view(Hkv, Gq * n, -1), vt).view(Hkv, Gq, n, D)
        if extra is not None:
            ve = Ve[:, i0:i1].permute(2, 1, 0, 3)  # [Hkv, n, m, D]
            o = o + torch.einsum("hgnm,hnmd->hgnd", ps[-1], ve)
        o = o / den
        out[i0:i1] = o.permute(2, 0, 1, 3).reshape(n, H, D)
    return out
