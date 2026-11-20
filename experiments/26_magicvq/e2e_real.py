"""E7 magicvq v5: FULLY REAL end-to-end self-speculative decoding on TP=4 (GPUs 4-7), bf16 Llama-3.1-8B-Instruct.

Nothing is replayed or modeled:
  * real weights (/data/models/llama3.1_8b_instruct), Megatron-style TP=4 shards per rank: 8 q-heads / 2 kv-heads,
    MLP 3584, lm_head vocab/4; vocab-parallel embedding (one extra small all_reduce per forward; frees 0.74 GiB / GPU
    that B16 x 32k needs); global greedy argmax by an int64 (value, -index) MAX all_reduce.
  * real prefill of the real prompts (per sequence, layer-wise: row-chunked GEMMs + one causal flash SDPA over the
    whole prompt), bf16 KV cache [L, 2, B, Hkv, T, D] (post-RoPE K).
  * real draft-cache construction from that cache (timed separately = "compression"):
      snapkv<W>   : SnapKV selection (last 32 prompt queries, softmax, pool 5, sum over the group; = tau_eval
                    snap_select) -> index list of W-32 positions per (layer, seq, kv-head); the draft reads those
                    positions + the window [P-32, qpos] straight from the target cache (no copy).
      vqpL2+r128  : 2-stage RQ (d8 K256, post-RoPE K, chnorm, calibrated pile-val codebooks 'p') encode of positions
                    [16, P); draft attention = VQ over [16, qpos-127] + bf16 sinks [0,16) + bf16 window
                    [qpos-127, qpos] from the target cache; per round the gamma+1 verified tokens are encoded.
      hyb512pL2   : SnapKV-512 set (sinks + top-480 per head) and window [P-32, qpos] bf16 from the target cache,
                    every other prompt token RQ2 (compacted per (layer, seq, kv-head) code planes); no per-round
                    encode (generated tokens stay in the bf16 window), = tau_eval 'hyb'.
  * real decode loop, CUDA graphs with device-side lengths: per round [gamma draft steps] -> [verify gamma+1] ->
    [greedy acceptance on device: accepted prefix, bonus token, per-sequence lengths advance by a+1 (ragged)] ->
    [RQ encode of the gamma+1 verified positions, RQ draft only] -> host reads a done flag (one sync per round).
    Draft rows' own K/V are written into the target cache at their positions (overwritten by the verify), so the
    window / in-round tokens are bf16 exactly as in tau_eval. Sequences that reached `gen` tokens are frozen (static
    batch: they still occupy their rows until the slowest sequence finishes).
    --accounting all_active (or both): MagicDec official accounting, measured: no row is frozen, the batch stops
    when the longest row has >= gen tokens, throughput = all tokens of all rows / decode wall (record key
    'all_active'); 'both' times static and all-active reps interleaved with shared plans / graphs / AR baseline.
  * AR baseline: same model code, same kernels (ragged Triton split-K flash-decoding, splits tuned the same way),
    CUDA graphs of 8 steps, no per-step host sync.
Timing: warm-up run excluded; `reps` timed runs (median, min, max); CUDA events per graph segment for the breakdown;
host = wall - GPU segments. Co-tenant check (nvidia-smi compute apps) recorded before and after every point.
Usage (4 ranks): torchrun --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --task syn_quote --ctx 32768 --B 16 \
      --configs ar snapkv4096:3 vqpL2+r128:6 hyb512pL2:6 --tag v5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2e_kern as EK  # noqa: E402
from specrq.kvq.kernels import split_plan  # noqa: E402  (pure python helper)
from specrq.nestkern.gpucheck import cotenants  # noqa: E402

MODEL = "/data/models/llama3.1_8b_instruct"
OUTROOT = "/data/lsh/results_accept/magicvq"
OUT = f"{OUTROOT}/e2e_real"
CBFILE = f"{OUTROOT}/cb/cb_d8K256x2_post_cn_greedy.pt"
HID, D, NL, HQA, HKVA, INTER, VOCAB, EPS = 4096, 128, 32, 32, 8, 14336, 128256, 1e-5
EOS = {128001, 128008, 128009}
SINKS, WIN = 16, 32
RANK, WORLD = 0, 1
LEGACY_T = False  # --legacy_T (reproduce the pre-fix cache length)
SOFT_REP = False  # v7: --soft_rep_check
DRAFT_FWD = None  # v14: --draft_w4 -> Marlin INT4 draft forward (e2e_w4.make_forward); None = bf16 `forward`
HEADN = 32  # rep-mismatch fix: first HEADN cache positions of every (layer, k/v, seq, kv-head) are snapshotted after
#             prefill and compared after every config (an out-of-range append wraps into the next head's sinks)


def log(*a):
    if RANK == 0:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def nvsmi_apps():
    import subprocess
    idx = dict(r.split(", ")[::-1] for r in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                                                             "--format=csv,noheader"], capture_output=True,
                                                            text=True).stdout.strip().splitlines())
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    gq = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout.strip().splitlines()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "apps": [{"pid": int(p), "mem": m, "gpu": idx.get(u.strip())} for p, m, u in
                     (x.split(", ") for x in rows if x.strip())],
            "gpus": [dict(zip(("index", "mem_used", "util"), r.split(", "))) for r in gq]}


def cocheck(own):
    """co-tenant record of GPUs 4-7 (this job's ranks excluded): apps + per-GPU memory / utilization."""
    n = nvsmi_apps()
    others = [x for x in n["apps"] if x["gpu"] in ("4", "5", "6", "7") and x["pid"] not in own]
    return {"cotenant": bool(others), "others": others, "nvsmi": n}


# ------------------------------------------------------------------------------------------------------------ model
class Shard:
    def __init__(self):
        from safetensors import safe_open
        hq, hkv, inter, vs = HQA // WORLD, HKVA // WORLD, INTER // WORLD, VOCAB // WORLD
        self.hq, self.hkv, self.inter, self.vs, self.G = hq, hkv, inter, vs, hq // hkv
        wm = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
        fh = {}

        def sl(name):
            f = wm[name]
            if f not in fh:
                fh[f] = safe_open(f"{MODEL}/{f}", "pt", device="cpu")
            return fh[f].get_slice(name)
        r = RANK
        cu = lambda t: t.to(torch.bfloat16).cuda().contiguous()  # noqa: E731
        self.layers = []
        for l in range(NL):
            p = f"model.layers.{l}."
            q = sl(p + "self_attn.q_proj.weight")[r * hq * D:(r + 1) * hq * D]
            k = sl(p + "self_attn.k_proj.weight")[r * hkv * D:(r + 1) * hkv * D]
            v = sl(p + "self_attn.v_proj.weight")[r * hkv * D:(r + 1) * hkv * D]
            g = sl(p + "mlp.gate_proj.weight")[r * inter:(r + 1) * inter]
            u = sl(p + "mlp.up_proj.weight")[r * inter:(r + 1) * inter]
            self.layers.append({
                "qkv": cu(torch.cat([q, k, v])), "o": cu(sl(p + "self_attn.o_proj.weight")[:, r * hq * D:(r + 1) * hq * D]),
                "gu": cu(torch.cat([g, u])), "down": cu(sl(p + "mlp.down_proj.weight")[:, r * inter:(r + 1) * inter]),
                "n1": cu(sl(p + "input_layernorm.weight")[:]), "n2": cu(sl(p + "post_attention_layernorm.weight")[:])})
        self.embed = cu(sl("model.embed_tokens.weight")[r * vs:(r + 1) * vs])
        self.lm_head = cu(sl("lm_head.weight")[r * vs:(r + 1) * vs])
        self.norm = cu(sl("model.norm.weight")[:])
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        cfg = AutoConfig.from_pretrained(MODEL)
        inv, sc = ROPE_INIT_FUNCTIONS["llama3"](cfg, "cuda")
        assert sc == 1.0
        self.inv = inv.float().contiguous()


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


def allreduce(x):
    dist.all_reduce(x)
    return x


def embed(m, tokens):
    loc = tokens - RANK * m.vs
    ok = (loc >= 0) & (loc < m.vs)
    e = F.embedding(loc.clamp(0, m.vs - 1), m.embed) * ok[:, None].to(torch.bfloat16)
    return allreduce(e)


IMAX = 2 ** 31 - 1


def gargmax(m, h):
    """global greedy argmax over the vocab shards; ties -> smallest index."""
    lg = rms(h, m.norm) @ m.lm_head.t()
    mx, ix = lg.float().max(-1)
    b = mx.view(torch.int32)
    key32 = torch.where(b >= 0, b, b ^ 0x7FFFFFFF).to(torch.int64)
    key = (key32 << 32) | (IMAX - (ix + RANK * m.vs)).to(torch.int64)
    dist.all_reduce(key, op=dist.ReduceOp.MAX)
    return IMAX - (key & 0xFFFFFFFF)


def layer_post(m, L, h, o):
    h = h + allreduce(o @ L["o"].t())
    gu = rms(h, L["n2"]) @ L["gu"].t()
    return h + allreduce((F.silu(gu[:, :m.inter]) * gu[:, m.inter:]) @ L["down"].t())


def forward(m, tokens, attn):
    h = embed(m, tokens)
    for l, L in enumerate(m.layers):
        qkv = rms(h, L["n1"]) @ L["qkv"].t()
        h = layer_post(m, L, h, attn(qkv, l))
    return gargmax(m, h)


# ---------------------------------------------------------------------------------------------------------- prefill
@torch.no_grad()
def prefill(m, tc, b, ids, qlast, CH=4096):
    """one sequence: KV into tc[:, :, b, :, :P], last 32 RoPE'd queries per layer into qlast[l][b]; -> first token."""
    P = ids.shape[0]
    h = torch.cat([embed(m, ids[c:c + CH]) for c in range(0, P, CH)])
    ang = torch.arange(P, device="cuda").float()[:, None] * m.inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None], torch.cat([ang.sin()] * 2, -1)[:, None]
    del ang

    def rope(x, c0, c1):
        return x * cos[c0:c1] + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1) * sin[c0:c1]
    hq, hkv = m.hq, m.hkv
    qb = torch.empty(P, hq, D, device="cuda", dtype=torch.bfloat16)
    for l, L in enumerate(m.layers):
        for c0 in range(0, P, CH):
            c1 = min(P, c0 + CH)
            qkv = rms(h[c0:c1], L["n1"]) @ L["qkv"].t()
            qb[c0:c1] = rope(qkv[:, :hq * D].view(-1, hq, D).float(), c0, c1).to(torch.bfloat16)
            tc[l, 0, b, :, c0:c1] = rope(qkv[:, hq * D:(hq + hkv) * D].view(-1, hkv, D).float(), c0, c1).to(
                torch.bfloat16).transpose(0, 1)
            tc[l, 1, b, :, c0:c1] = qkv[:, (hq + hkv) * D:].view(-1, hkv, D).transpose(0, 1)
            del qkv
        qlast[l][b] = qb[P - WIN:].clone()
        K = tc[l, 0, b, :, :P].repeat_interleave(m.G, 0)[None]
        V = tc[l, 1, b, :, :P].repeat_interleave(m.G, 0)[None]
        o = F.scaled_dot_product_attention(qb.transpose(0, 1)[None], K, V, is_causal=True)[0].transpose(0, 1)
        del K, V
        o = o.reshape(P, hq * D)
        for c0 in range(0, P, CH):
            h[c0:c0 + CH] = layer_post(m, L, h[c0:c0 + CH], o[c0:c0 + CH])
        del o
    return gargmax(m, h[P - 1:P])[0]


# ------------------------------------------------------------------------------------------------------ compression
@torch.no_grad()
def snap_scores(m, tc, qlast, l, b, P):
    """tau_eval.snap_select on the cached (post-RoPE) keys: [Hkv, P - WIN] pooled window attention."""
    q = qlast[l][b].float() / math.sqrt(D)  # [WIN, hq, D]
    k = tc[l, 0, b, :, :P].float()  # [hkv, P, D]
    sc = torch.einsum("wkgd,ktd->kgwt", q.view(WIN, m.hkv, m.G, D), k).reshape(m.hq, WIN, P)
    jj = torch.arange(P, device="cuda")
    sc.masked_fill_(jj[None, None, :] > (P - WIN + torch.arange(WIN, device="cuda"))[None, :, None], float("-inf"))
    a = sc.softmax(-1)[..., :P - WIN].sum(1)
    a = F.avg_pool1d(a[:, None], kernel_size=5, padding=2, stride=1)[:, 0]
    return a.view(m.hkv, m.G, -1).sum(1)


class CB:
    def __init__(self, m, path=CBFILE):
        st = torch.load(path, map_location="cpu", weights_only=False)
        assert st["d"] == 8 and st["layout"] == "hf" and st["sizes"] == [256, 256] and st["post_rope"]
        h0, h1 = RANK * m.hkv, (RANK + 1) * m.hkv
        self.f32 = torch.stack([torch.stack([torch.stack([c[h0:h1] for c in st["cb"][l][kv]], 1)
                                             for kv in range(2)]) for l in range(NL)]).cuda().contiguous()
        self.bf = self.f32.to(torch.bfloat16).contiguous()  # [L, 2, Hkv, S, 256, 8]
        self.mu = torch.stack([torch.stack([st["mu"][l][kv][h0:h1] for kv in range(2)]) for l in range(NL)]).cuda().float().contiguous()
        self.sd = torch.stack([torch.stack([st["sd"][l][kv][h0:h1] for kv in range(2)]) for l in range(NL)]).cuda().float().contiguous()


class Draft:
    """draft cache structures + per-layer attention for one draft spec."""

    def __init__(self, name, m, st, cb):
        self.name, self.m, self.st, self.cb = name, m, st, cb
        self.stream = name.startswith("stream")
        if name.startswith("snapkv"):
            self.kind, self.W = "snapkv", int(name[6:])
        elif self.stream:  # v7: StreamingLLM-W (MagicDec): 16 sinks + last W-16 prompt tokens, then the growing window
            self.kind, self.W = "snapkv", int(name[6:])
        elif name.startswith("hyb"):
            self.kind, self.W = "hyb", int(name[3:].split("p")[0])
            self.kst = int(name.split("L")[-1]) if "L" in name else 2  # v6: hyb512pL1 (1-bit VQ part)
        elif name.startswith("vqpL"):
            self.kind, self.kst, self.R = "rq", int(name[4]), int(name.split("+r")[1])
        else:
            raise ValueError(name)

    @torch.no_grad()
    def build(self, tc, qlast):
        """compression from the target cache (all layers); allocates on first call, rewrites in place after."""
        m, st, B, hkv = self.m, self.st, self.st.B, self.m.hkv
        Pl = st.P_list
        dev = "cuda"
        if self.kind in ("snapkv", "hyb"):
            nsel = self.W - WIN
            if not hasattr(self, "idx"):
                nmax = nsel + (SINKS if self.kind == "hyb" else 0)
                self.idx = torch.zeros(NL, B, hkv, nmax, dtype=torch.int32, device=dev)
                self.nidx = torch.zeros(NL, B * hkv, dtype=torch.int32, device=dev)
                if self.kind == "hyb":
                    Tc = max(Pl) - WIN
                    self.codes = torch.zeros(NL, 2, 2, B, hkv, Tc, 16, dtype=torch.uint8, device=dev)
                    self.ncomp = torch.zeros(NL, B * hkv, dtype=torch.int32, device=dev)
                    self.src = torch.zeros(B, hkv, Tc, dtype=torch.int32, device=dev)
                    self.Tc = Tc
            for l in range(NL):
                for b in range(B):
                    P = Pl[b]
                    if self.stream:  # fixed positions [0, 16) + [P - W + 16, P - 32) (+ window [P - 32, qpos] in attn)
                        pos = torch.cat([torch.arange(SINKS, device=dev), torch.arange(P - self.W + SINKS, P - WIN, device=dev)])
                        self.idx[l, b, :, :pos.numel()] = pos.int()[None]
                        self.nidx[l, b * hkv:(b + 1) * hkv] = pos.numel()
                        continue
                    top = snap_scores(m, tc, qlast, l, b, P).topk(min(nsel, P - WIN), -1).indices
                    if self.kind == "snapkv":
                        self.idx[l, b, :, :top.shape[1]] = top.sort(-1).values.int()
                        self.nidx[l, b * hkv:(b + 1) * hkv] = top.shape[1]
                    else:  # kept = sinks + top-480 (ascending) first, complement (ascending) after: stable sorts
                        keep = torch.zeros(hkv, P - WIN, dtype=torch.int32, device=dev)
                        keep.scatter_(1, top, 1)
                        keep[:, :SINKS] = 1
                        n_keep = keep.sum(1)
                        kfirst = torch.sort(1 - keep, dim=-1, stable=True).indices
                        cfirst = torch.sort(keep, dim=-1, stable=True).indices
                        nm = self.idx.shape[-1]
                        self.idx[l, b] = kfirst[:, :nm].int()
                        self.src[b, :, :P - WIN] = cfirst.int()
                        self.nidx[l, b * hkv:(b + 1) * hkv] = n_keep.int()
                        self.ncomp[l, b * hkv:(b + 1) * hkv] = ((P - WIN) - n_keep).int()
                if self.kind == "hyb":
                    EK.rq_encode(tc[l], self.codes[l], self.cb.f32[l], self.cb.mu[l], self.cb.sd[l], self.ncomp[l],
                                 self.Tc, src=self.src)
            self.zero = torch.zeros(B * hkv, dtype=torch.int32, device=dev)
        else:  # rq: positions [16, P) in place (planes indexed by position)
            if not hasattr(self, "codes"):
                self.codes = torch.zeros(NL, 2, self.kst, B, hkv, st.T, 16, dtype=torch.uint8, device=dev)
                self.sinks = torch.arange(SINKS, device=dev, dtype=torch.int32)[None, None].expand(B, hkv, SINKS).contiguous()
                self.nsinks = torch.full((B * hkv,), SINKS, device=dev, dtype=torch.int32)
                self.vlo = torch.full((B * hkv,), SINKS, device=dev, dtype=torch.int32)
            lo = torch.full((B,), SINKS, device=dev, dtype=torch.int32)
            cnt = (torch.tensor(Pl, device=dev, dtype=torch.int32) - SINKS).repeat_interleave(hkv).contiguous()
            for l in range(NL):
                EK.rq_encode(tc[l], self.codes[l], self.cb.f32[l], self.cb.mu[l], self.cb.sd[l], cnt,
                             max(Pl) - SINKS, lo=lo)

    def free(self):
        for k in ("idx", "nidx", "codes", "ncomp", "src", "sinks", "nsinks", "vlo", "zero"):
            if hasattr(self, k):
                delattr(self, k)

    def nv_max(self, st):
        """max bf16 virtual length / VQ length (for split plans)."""
        T = st.T
        if self.kind == "snapkv":
            return self.W - WIN + WIN + st.gen + 16, 0
        if self.kind == "hyb":
            return self.idx.shape[-1] + WIN + st.gen + 16, self.Tc
        return SINKS + self.R + 8, T


# ------------------------------------------------------------------------------------------------------- attention
class Attn:
    """attention of one step (M rows per sequence) for role 'target' or a draft; device-side positions.
    prep(base_len) (in graph) sets qpos = len + m0 and the ranges; __call__(qkv, l) appends the rows' K / V to the
    target cache and runs the partial kernels + combine."""

    def __init__(self, m, st, M, m0=0, draft=None, plan=None):
        self.m, self.st, self.M, self.m0, self.d = m, st, M, m0, draft
        B, hkv = st.B, m.hkv
        dev = "cuda"
        self.qpos = torch.zeros(B, dtype=torch.int32, device=dev)
        self.blo = torch.zeros(B, dtype=torch.int32, device=dev)
        self.bhi = torch.zeros(B, dtype=torch.int32, device=dev)
        self.vhi = torch.zeros(B * hkv, dtype=torch.int32, device=dev)
        self.out = torch.empty(B * M, m.hq * D, dtype=torch.bfloat16, device=dev)
        self.plan = plan  # ((bf_splits, bf_chunk), (vq_splits, vq_chunk) or None)
        (bs, _), vq = plan
        self.spt = bs + (16 * vq[0] if vq else 0)
        self.ws = EK.workspace(B, hkv, m.G, M, self.spt)

    def prep(self, length):
        st = self.st
        torch.add(length, self.m0, out=self.qpos)
        if self.d is None:
            self.bhi.copy_(self.qpos + self.M)
            return
        k = self.d.kind
        torch.add(self.qpos, 1, out=self.bhi)
        if k in ("snapkv", "hyb"):
            self.blo.copy_(st.Pm32)
        else:
            torch.add(self.qpos, 1 - self.d.R, out=self.blo)
            self.vhi.copy_(self.blo.repeat_interleave(self.m.hkv))

    def __call__(self, qkv, l):
        m, st, d, M = self.m, self.st, self.d, self.M
        tcl = st.tc[l]
        EK.append_kv(qkv, tcl, m.inv, self.qpos, M, m.hq)
        (bs, bch), vq = self.plan
        if d is None:
            EK.bf_partial(qkv, tcl[0], tcl[1], m.inv, self.ws, self.spt, 0, bs, bch, self.blo, self.bhi, self.qpos,
                          M, m.G)
        elif d.kind == "snapkv":
            EK.bf_partial(qkv, tcl[0], tcl[1], m.inv, self.ws, self.spt, 0, bs, bch, self.blo, self.bhi, self.qpos,
                          M, m.G, d.idx[l], d.nidx[l])
        elif d.kind == "hyb":
            EK.bf_partial(qkv, tcl[0], tcl[1], m.inv, self.ws, self.spt, 0, bs, bch, self.blo, self.bhi, self.qpos,
                          M, m.G, d.idx[l], d.nidx[l])
            self._vq(qkv, l, d.zero, d.ncomp[l])
        else:
            EK.bf_partial(qkv, tcl[0], tcl[1], m.inv, self.ws, self.spt, 0, bs, bch, self.blo, self.bhi, self.qpos,
                          M, m.G, d.sinks, d.nsinks)
            self._vq(qkv, l, d.vlo, self.vhi)
        return EK.combine(self.ws, self.spt, self.out, st.B, m.hkv, m.G, M)

    def _vq(self, qkv, l, lo, hi):
        d, cb = self.d, self.d.cb
        (bs, _), (vs, vch) = self.plan
        c = d.codes[l]
        EK.lut_partial(qkv, [c[0, s] for s in range(d.kst)], [c[1, s] for s in range(d.kst)],
                       [cb.bf[l, 0, :, s] for s in range(d.kst)], [cb.bf[l, 1, :, s] for s in range(d.kst)],
                       self.m.inv, self.ws, self.spt, bs, vs, vch, lo, hi, self.qpos, cb.sd[l, 0], cb.mu[l, 0],
                       cb.sd[l, 1], cb.mu[l, 1], self.st.B, self.m.hkv, d.kst)


def tune_plan(m, st, M, draft, reps=5):
    """splits / chunk per partial kernel: fastest of a few program-count targets (attention only, all layers, at the
    current lengths = start of decode, which is within 1 % of the longest length of the run)."""
    B, hkv = st.B, m.hkv
    nb, nvq = (st.T, 0) if draft is None else draft.nv_max(st)
    qkv = torch.randn(B * M, (m.hq + 2 * hkv) * D, device="cuda").to(torch.bfloat16) * 0.1
    st.reset()  # lengths = prompt lengths; the appends below write rows >= P (scratch, rewritten by the decode)
    res = {}

    def timeit(plan):  # CUDA-graph timing of the attention of all layers (eager timing is launch-bound / noisy)
        a = Attn(m, st, M, 0, draft, plan)
        a.prep(st.length)
        f = lambda: [a(qkv, l) for l in range(NL)]  # noqa: E731
        f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            s.record()
            g.replay()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 2)
        del g
        t = torch.tensor([statistics.median(ts)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()
    bf_c = sorted({split_plan(B, hkv, nb, 64, tp) for tp in (128, 256, 512, 1024, 2048)})
    vq_c = sorted({(-(-nvq // ch), ch) for ch in (1024, 2048, 4096, 8192)}) if nvq else [None]
    best = None
    v0 = vq_c[len(vq_c) // 2] if nvq else None
    for bp in bf_c:  # bf16 part first with a middle VQ candidate, then the VQ part with the best bf16 plan
        t = timeit((bp, v0))
        res[f"bf{bp}"] = t
        if best is None or t < best[0]:
            best = (t, (bp, v0))
    if nvq:
        for vp in vq_c:
            if vp == v0:
                continue
            t = timeit((best[1][0], vp))
            res[f"vq{vp}"] = t
            if t < best[0]:
                best = (t, (best[1][0], vp))
    return best[1], {"attn_ms": best[0], "candidates": res}


# ---------------------------------------------------------------------------------------------------------- state
class State:
    def __init__(self, m, B, T, gen, gmax):
        self.B, self.T, self.gen = B, T, gen
        dev = "cuda"
        self.tc = torch.empty(NL, 2, B, m.hkv, T, D, dtype=torch.bfloat16, device=dev)
        self.length = torch.zeros(B, dtype=torch.int32, device=dev)
        self.old_len = torch.zeros(B, dtype=torch.int32, device=dev)
        self.ngen = torch.zeros(B, dtype=torch.int64, device=dev)
        self.cap = gen + gmax + 2
        self.out = torch.zeros(B, self.cap, dtype=torch.int64, device=dev)
        self.cur = torch.zeros(B, dtype=torch.int64, device=dev)
        self.done = torch.zeros(1, dtype=torch.int32, device=dev)
        self.rc = torch.zeros(1, dtype=torch.int64, device=dev)
        self.acc_log = torch.full((gen + 2, B), -2, dtype=torch.int64, device=dev)

    def set_prompts(self, P_list, g0):
        self.P_list = P_list
        self.Pmax = max(P_list)
        self.P = torch.tensor(P_list, dtype=torch.int32, device="cuda")
        self.Pm32 = self.P - WIN
        self.g0 = g0

    def reset(self):
        self.length.copy_(self.P)
        self.ngen.zero_()
        self.out.zero_()
        self.cur.copy_(self.g0)
        self.done.zero_()
        self.rc.zero_()
        self.acc_log.fill_(-2)


def capture(fn, pool):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return g


def ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def gmax_time(x):
    t = torch.tensor([x], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# ------------------------------------------------------------------------------------------------------ AR runner
def run_ar(m, st, reps, K=8):
    plan, tune = tune_plan(m, st, 1, None)
    a = Attn(m, st, 1, 0, None, plan)
    B = st.B

    def step():
        a.prep(st.length)
        nxt = forward(m, st.cur, a)
        st.out.scatter_(1, st.ngen.clamp(max=st.cap - 1)[:, None], nxt[:, None])
        st.cur.copy_(nxt)
        st.length.add_(1)
        st.ngen.add_(1)

    def steps():
        for _ in range(K):
            step()
    st.reset()
    pool = torch.cuda.graph_pool_handle()
    g = capture(steps, pool)
    nrep = st.gen // K
    runs = []
    outs = None
    for rep in range(reps + 1):
        st.reset()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        e0 = ev()
        for _ in range(nrep):
            g.replay()
        e1 = ev()
        torch.cuda.synchronize()
        wall = gmax_time(time.perf_counter() - t0)
        gpu = e0.elapsed_time(e1) / 1e3
        o = st.out[:, :st.gen].clone()
        if outs is None:
            outs = o
        assert torch.equal(o, outs), "AR outputs differ between reps"
        if rep > 0:
            runs.append({"wall_s": wall, "gpu_s": gpu, "tok_per_s": B * st.gen / wall})
    del g
    return {"plan": plan, "tune": tune, "runs": runs, "step_ms": statistics.median(r["wall_s"] for r in runs) / st.gen * 1e3}, outs


# ---------------------------------------------------------------------------------------------------- spec runner
def run_spec(m, st, draft, gamma, reps, modes=("static",)):
    """modes: 'static' (each row stops at gen, finished rows idle until the slowest row is done) and/or
    'all_active' (MagicDec official accounting: no row is ever frozen; the batch stops when the LONGEST row has
    >= gen new tokens; throughput counts every token of every row). Both modes share plans, draft / verify / encode
    graphs; only the accept graph differs. Timed reps are interleaved (rep 0 of each mode = warm-up)."""
    B = st.B
    G1 = gamma + 1
    vplan, vtune = tune_plan(m, st, G1, None)
    dplan, dtune = tune_plan(m, st, 1, draft)
    dsteps = [Attn(m, st, 1, j, draft, dplan) for j in range(gamma)]
    va = Attn(m, st, G1, 0, None, vplan)
    dtok = torch.zeros(B, G1, dtype=torch.int64, device="cuda")
    targ = torch.zeros(B, G1, dtype=torch.int64, device="cuda")
    jj = torch.arange(G1, device="cuda")
    enc_cnt = torch.full((B * m.hkv,), G1, dtype=torch.int32, device="cuda")

    def draft_fn():
        dtok[:, 0].copy_(st.cur)
        for j in range(gamma):
            a = dsteps[j]
            a.prep(st.length)
            dtok[:, j + 1].copy_((DRAFT_FWD or forward)(m, dtok[:, j], a))

    def verify_fn():
        va.prep(st.length)
        targ.copy_(forward(m, dtok.reshape(-1), va).view(B, G1))

    def accept_fn(all_active=False):
        match = (dtok[:, 1:] == targ[:, :gamma]).to(torch.int64)
        acc = match.cumprod(1).sum(1)
        # all_active: every row is active every round (at round start all rows have ngen <= gen - 1, so writes stay
        # below gen + gamma < cap and cache positions below P + gen + gamma <= T)
        active = torch.ones_like(acc, dtype=torch.bool) if all_active else st.ngen < st.gen
        inc = torch.where(active, acc + 1, torch.zeros_like(acc))
        idx = (st.ngen[:, None] + jj[None]).clamp(max=st.cap - 1)
        mask = active[:, None] & (jj[None] <= acc[:, None])
        cur = st.out.gather(1, idx)
        st.out.scatter_(1, idx, torch.where(mask, targ, cur))
        st.acc_log.index_copy_(0, st.rc.clamp(max=st.acc_log.shape[0] - 1),
                               torch.where(active, acc, torch.full_like(acc, -1))[None])
        st.rc.add_(1)
        st.old_len.copy_(st.length)
        st.length.add_(inc.to(torch.int32))
        st.ngen.add_(inc)
        st.cur.copy_(torch.where(active, targ.gather(1, acc[:, None])[:, 0], st.cur))
        if all_active:
            st.done.copy_((st.ngen >= st.gen).any().to(torch.int32)[None])
        else:
            st.done.copy_((st.ngen >= st.gen).all().to(torch.int32)[None])

    def enc_fn():
        for l in range(NL):
            EK.rq_encode(st.tc[l], draft.codes[l], draft.cb.f32[l], draft.cb.mu[l], draft.cb.sd[l], enc_cnt, G1,
                         lo=st.old_len)
    st.reset()
    pool = torch.cuda.graph_pool_handle()
    g_d = capture(draft_fn, pool)
    st.reset()
    g_v = capture(verify_fn, pool)
    g_as = {mode: capture(lambda f=(mode == "all_active"): accept_fn(f), pool) for mode in modes}
    g_e = capture(enc_fn, pool) if draft.kind == "rq" else None
    res = {mode: {"runs": [], "out": None, "log": None, "ngen": None} for mode in modes}
    for rep, mode in [(r_, md) for r_ in range(reps + 1) for md in modes]:
        g_a = g_as[mode]
        st.reset()
        torch.cuda.synchronize()
        dist.barrier()
        evs = []
        t0 = time.perf_counter()
        rounds = 0
        while True:
            e0 = ev()
            g_d.replay()
            e1 = ev()
            g_v.replay()
            e2 = ev()
            g_a.replay()
            e3 = ev()
            if g_e is not None:
                g_e.replay()
            e4 = ev()
            evs.append((e0, e1, e2, e3, e4))
            rounds += 1
            if st.done.item() or rounds >= st.gen + 1:
                break
        torch.cuda.synchronize()
        wall = gmax_time(time.perf_counter() - t0)
        # rep-mismatch fix guard: the last draft / verify of a row wrote positions up to length + gamma (static mode:
        # frozen rows keep running draft + verify at their final length until the slowest row is done)
        mx = int(st.length.max().item()) + gamma
        if mx >= st.T:
            R_ = res[mode]
            R_.setdefault("cache_overflow", []).append({"rep": rep, "max_pos": mx, "T": st.T})
            log(f"  WARNING cache overflow: wrote position {mx} >= T={st.T} (rep {rep}, {mode})")
            assert LEGACY_T, "cache overflow with the fixed T"
        seg = [sum(a.elapsed_time(b) for a, b in ((x[i], x[i + 1]) for x in evs)) / 1e3 for i in range(4)]
        R = res[mode]
        ngen = st.ngen.clone()
        o = st.out[:, :st.gen].clone() if mode == "static" else st.out.clone()
        lg = st.acc_log[:rounds].clone()
        if R["out"] is None:
            R["out"], R["log"], R["ngen"] = o, lg, ngen
        if mode == "static":  # as before: the first gen tokens of every row are identical across reps
            if SOFT_REP and not torch.equal(o, R["out"]):  # v7 --soft_rep_check: record instead of abort
                dv = (o != R["out"])
                fd = torch.where(dv.any(1), dv.int().argmax(1), torch.full_like(ngen, -1))
                R.setdefault("rep_out_differs", []).append({"rep": rep, "first_div_per_row": fd.tolist()})
                log(f"  WARNING static spec output of rep {rep} differs from rep 0 (rows {int(dv.any(1).sum())})")
            else:
                assert torch.equal(o, R["out"]), "spec outputs differ between reps"
        elif SOFT_REP:
            n = torch.minimum(ngen, R["ngen"])
            ok = (torch.arange(o.shape[1], device=o.device)[None] >= n[:, None]) | (o == R["out"])
            if not bool(ok.all()):
                R.setdefault("rep_out_differs", []).append({"rep": rep, "rows": int((~ok).any(1).sum())})
                log(f"  WARNING all_active spec output of rep {rep} differs from rep 0 on the overlap")
        else:
            # all_active: tokens must agree on each row's overlap; per-row LENGTHS could differ between reps when a
            # draft kernel is not bit-deterministic (the VQ LUT kernel was, via a shared-memory float atomicAdd of the
            # q.mu term, until v6e2e; now a fixed-order reduction): 'rep_ngen_differs' records it
            n = torch.minimum(ngen, R["ngen"])
            ok = (torch.arange(o.shape[1], device=o.device)[None] >= n[:, None]) | (o == R["out"])
            assert bool(ok.all()), "spec outputs differ between reps on the overlap (all_active)"
        if rep > 0:
            ntok = B * st.gen if mode == "static" else int(ngen.sum().item())
            R["runs"].append({"wall_s": wall, "rounds": rounds, "tokens": ntok, "tok_per_s": ntok / wall,
                              "ngen": ngen.tolist(),
                              "draft_s": seg[0], "verify_s": seg[1], "accept_s": seg[2], "encode_s": seg[3],
                              "host_s": wall - sum(seg)})
    del g_d, g_v, g_as, g_a, g_e
    base = {"draft_plan": dplan, "draft_tune": dtune, "verify_plan": vplan, "verify_tune": vtune}
    for mode in modes:
        if "cache_overflow" in res[mode]:
            base.setdefault("cache_overflow", {})[mode] = res[mode]["cache_overflow"]
    if "static" in res:
        base["runs"] = res["static"]["runs"]
        if "rep_out_differs" in res["static"]:
            base["rep_out_differs"] = res["static"]["rep_out_differs"]
    if "all_active" in res:
        R = res["all_active"]
        ng = R["ngen"].tolist()
        base["all_active"] = {"runs": R["runs"], "ngen": ng, "total_tokens": sum(ng),
                              "tau": statistics.median(x["tokens"] / (B * x["rounds"]) for x in R["runs"]),
                              "rep_ngen_differs": any(x["ngen"] != ng for x in R["runs"]),
                              "out": [row[:n] for row, n in zip(R["out"].tolist(), ng)],
                              "acc_log": R["log"].tolist()}
        if "rep_out_differs" in R:
            base["all_active"]["rep_out_differs"] = R["rep_out_differs"]
    S = res.get("static") or res["all_active"]
    return base, S["out"][:, :st.gen], S["log"]


# ------------------------------------------------------------------------------------------------------------- main
def main():
    global RANK, WORLD, SOFT_REP, LEGACY_T, DRAFT_FWD
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="ar | <draft>:<gamma>")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip", type=int, default=0, help="first prompt index")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cbfile", default=CBFILE, help="v6: 'p'-format codebook file (e.g. a draft-trained one)")
    ap.add_argument("--cbx", nargs="*", default=[], metavar="TAG=PATH",
                    help="extra codebooks for per-draft use: config '<draft>@TAG:<gamma>' (e.g. vqpL2+r128@t2w:4) "
                         "uses PATH instead of --cbfile; same AR baseline / process")
    ap.add_argument("--accounting", default="static", choices=["static", "all_active", "both"],
                    help="batch accounting of the spec loop (both = interleaved reps in one run, shared graphs / AR)")
    ap.add_argument("--soft_rep_check", action="store_true",
                    help="v7: record (rep_out_differs) instead of aborting when a rep's spec output differs from rep 0")
    ap.add_argument("--lazy_max", action="store_true",
                    help="E10: lazy online max in the VQ LUT kernel (exact; same as MV_LUT_LAZY=1); default off")
    ap.add_argument("--legacy_T", action="store_true",
                    help="reproduce the pre-fix cache length T = ceil64(Pmax + gen + gmax + 2) (static-mode frozen rows "
                         "then write up to Pmax + gen + 2 gamma >= T at gamma 6 -> wrap into the next head's sinks)")
    ap.add_argument("--draft_w4", default=None, metavar="MARLIN_DIR",
                    help="v14: draft linears as Marlin INT4 shards from MARLIN_DIR (e2e_w4.py); verify stays bf16")
    a = ap.parse_args()
    if a.lazy_max:
        EK.LAZY_MAX = True
    SOFT_REP = a.soft_rep_check
    LEGACY_T = a.legacy_T
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(RANK)
    lst = [None] * WORLD
    dist.all_gather_object(lst, os.getpid())
    own = set(lst)
    cot = lambda: [c for c in cotenants() if c[0] not in own]  # noqa: E731
    rec = {"args": vars(a), "world": WORLD, "nvsmi_apps_start": nvsmi_apps(), "cotenants_start": cot(),
           "co_start": cocheck(own)}
    rec["lut_lazy_max"] = EK.LAZY_MAX
    log("co-tenants at start:", rec["cotenants_start"] or "none")
    prompts = json.load(open(f"{OUT}/prompts_{a.task}_ctx{a.ctx}.json"))[a.skip:a.skip + a.B]
    assert len(prompts) == a.B, len(prompts)
    t0 = time.time()
    m = Shard()
    torch.cuda.synchronize()
    log(f"weights loaded {time.time() - t0:.0f}s, alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    rec["draft_w4"] = None
    if a.draft_w4:  # v14
        import e2e_w4
        w4 = e2e_w4.W4(a.draft_w4, RANK, WORLD, NL, D, HQA, HKVA, INTER)
        DRAFT_FWD = e2e_w4.make_forward(sys.modules[__name__], w4)
        rec["draft_w4"] = w4.info()
        torch.cuda.synchronize()
        log(f"draft INT4 (Marlin) loaded: {w4.gib:.2f} GiB/rank, alloc {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    gammas = [int(c.split(":")[1]) for c in a.configs if c != "ar"]
    gmax = max(gammas + [1])
    P_list = [p["P"] for p in prompts]
    # rep-mismatch fix: in static accounting a row frozen at ngen <= gen + gamma keeps running draft + verify (writes
    # at length .. length + gamma) until the slowest row finishes -> max position P + gen + 2 gamma. The old bound
    # (gen + gmax + 2) let gamma >= 3 rows write past T into the next (seq, kv-head)'s first positions (attention
    # sinks / BOS) of the target cache, persisting for the rest of the process.
    T = -(-(max(P_list) + a.gen + (gmax if a.legacy_T else 2 * gmax) + 2) // 64) * 64
    st = State(m, a.B, T, a.gen, gmax)
    log(f"target cache T={T}: {st.tc.numel() * 2 / 2**30:.2f} GiB, free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    # ---- prefill
    qlast = [[None] * a.B for _ in range(NL)]
    g0 = torch.zeros(a.B, dtype=torch.int64, device="cuda")
    pf = []
    for b, p in enumerate(prompts):
        ids = torch.tensor(p["ids"], device="cuda")
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()
        g0[b] = prefill(m, st.tc, b, ids, qlast)
        torch.cuda.synchronize()
        pf.append(gmax_time(time.perf_counter() - t1))
        torch.cuda.empty_cache()
    st.set_prompts(P_list, g0)
    head0 = st.tc[:, :, :, :, :HEADN].clone()  # rep-mismatch fix: prompt-head integrity reference

    def head_changed():
        ch = (st.tc[:, :, :, :, :HEADN] != head0).any(-1)  # [L, 2, B, Hkv, HEADN]
        n = torch.tensor([int(ch.any(-1).sum())], device="cuda")
        dist.all_reduce(n)
        rows = ch.any(-1).any(-1).any(0).any(0).int()  # [B]
        dist.all_reduce(rows)
        pos = ch.any(0).any(0).any(0).any(0).int()
        dist.all_reduce(pos)
        return {"n_heads_changed": int(n.item()), "rows": [b for b, x in enumerate(rows.tolist()) if x],
                "positions": [p for p, x in enumerate(pos.tolist()) if x]}
    rec["prefill"] = {"total_s": sum(pf), "per_seq_s": pf, "P": P_list,
                      "peak_GiB": torch.cuda.max_memory_allocated() / 2**30}
    log(f"prefill {sum(pf):.1f}s (per seq {statistics.median(pf):.2f}s), peak {rec['prefill']['peak_GiB']:.2f} GiB")
    cb = CB(m, a.cbfile)
    rec["cbfile"] = a.cbfile
    cbs = {"": cb}
    rec["cbx"] = {}
    for x in a.cbx:
        tag, path = x.split("=", 1)
        cbs[tag] = CB(m, path)
        rec["cbx"][tag] = path
    rec["configs"] = {}
    ar_out = None
    for c in a.configs:
        torch.cuda.reset_peak_memory_stats()
        co_b = cocheck(own) if RANK == 0 else None
        tc0 = time.time()
        if c == "ar":
            r, out = run_ar(m, st, a.reps)
            ar_out = out
            r["out"] = out.tolist()
            rec["configs"]["ar"] = r
            r["co_before"], r["co_after"] = co_b, cocheck(own) if RANK == 0 else None
            r["prompt_head_after"] = head_changed()
            log(f"[ar] {statistics.median(x['tok_per_s'] for x in r['runs']):.1f} tok/s, step {r['step_ms']:.2f} ms")
        else:
            name, g = c.split(":")
            g = int(g)
            name, _, cbtag = name.partition("@")  # '<draft>@TAG': codebook from --cbx TAG=PATH
            d = Draft(name, m, st, cbs[cbtag])
            ct = []
            for rep in range(4):  # compression: warm-up + 3 timed
                torch.cuda.synchronize()
                dist.barrier()
                t1 = time.perf_counter()
                d.build(st.tc, qlast)
                torch.cuda.synchronize()
                if rep:
                    ct.append(gmax_time(time.perf_counter() - t1))
            modes = ("static", "all_active") if a.accounting == "both" else (a.accounting,)
            r, out, lg = run_spec(m, st, d, g, a.reps, modes)
            r["compress_s"] = {"median": statistics.median(ct), "all": ct}
            r["out"] = out.tolist()  # static: gen tokens / row; all_active only: first gen slots (rows may be shorter)
            r["acc_log"] = lg.tolist()
            if "all_active" in r and ar_out is not None:  # greedy match vs AR on each row's overlapping prefix
                aa = r["all_active"]
                fd, mt = [], []
                for row, ao in zip(aa["out"], ar_out.tolist()):
                    n = min(len(row), len(ao))
                    dv = [i for i in range(n) if row[i] != ao[i]]
                    fd.append(dv[0] if dv else None)
                    mt.append(sum(int(row[i] == ao[i]) for i in range(n)) / max(1, n))
                aa["first_divergence_vs_ar"], aa["match_vs_ar_overlap"] = fd, mt
            r["peak_GiB"] = torch.cuda.max_memory_allocated() / 2**30
            rec["configs"][c] = r
            r["co_before"], r["co_after"] = co_b, cocheck(own) if RANK == 0 else None
            r["prompt_head_after"] = hc = head_changed()
            if hc["n_heads_changed"]:
                log(f"  WARNING prompt-head cache changed after {c}: {hc}")
            d.free()
            del d
            rr = r.get("runs") or r["all_active"]["runs"]
            if "all_active" in r:
                aa = r["all_active"]
                log(f"[{c}] all_active {statistics.median(x['tok_per_s'] for x in aa['runs']):.1f} tok/s, rounds "
                    f"{aa['runs'][0]['rounds']}, tokens {aa['total_tokens']}, tau {aa['tau']:.2f}")
            log(f"[{c}] {statistics.median(x['tok_per_s'] for x in rr):.1f} tok/s, rounds "
                f"{rr[0]['rounds']}, compress {statistics.median(ct):.2f}s, peak {r['peak_GiB']:.2f} GiB "
                f"({time.time() - tc0:.0f}s)")
        torch.cuda.empty_cache()
        if RANK == 0:
            os.makedirs(OUT, exist_ok=True)
            json.dump(rec, open(f"{OUT}/run_{a.tag}_{a.task}_ctx{a.ctx}_B{a.B}.json.partial", "w"))
    rec["g0"] = g0.tolist()
    rec["nvsmi_apps_end"] = nvsmi_apps()
    rec["cotenants_end"] = cot()
    rec["co_end"] = cocheck(own)
    rec["cotenant_any"] = bool(rec["co_start"]["cotenant"] or rec["co_end"]["cotenant"] or any(
        (x.get("co_before") or {}).get("cotenant") or (x.get("co_after") or {}).get("cotenant")
        for x in rec["configs"].values()))
    rec["wall_s"] = time.time() - t0
    if RANK == 0:
        json.dump(rec, open(f"{OUT}/run_{a.tag}_{a.task}_ctx{a.ctx}_B{a.B}.json", "w"))
    log("done", f"{time.time() - t0:.0f}s, co-tenants at end:", rec["cotenants_end"] or "none")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
