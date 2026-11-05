"""
KV store for self-speculative decoding with one shared KV representation.

Two caches per layer (post-RoPE, model dtype, on that layer's device):
  target cache : what the target attends to
  draft cache  : what the draft attends to (aliases the target cache for exact / sparse drafts)

Representations (`rep`):
  per-token, pre-RoPE (IntRep, RQRep): the target's K/V for new tokens are quantized once at write;
      the target cache gets the target reconstruction, the draft cache the draft one (e.g. RQ prefix).
  block, post-RoPE (QuantSpecRep): K/V are written exact; full blocks are quantized in place when
      tokens are committed (keeps G..2G recent tokens in full precision, as in QuantSpec).
  sparse (StreamingRep, SnapKVRep): exact shared cache; the draft attends to a subset of positions.

Draft passes write provisional K/V for proposed tokens; the next verify overwrites those positions.
Rolling back needs no copy: every pass passes `start`, and attention only reads [:start+T].
"""

import math
import re

import torch
import torch.nn.functional as F


# ============================================================================ per-token reps

class FP16Rep:
    """Draft and target both see exact KV. Sanity check: acceptance ≈ 1 (greedy)."""
    name, per_token, shared = "fp16", True, True
    bits_target = bits_draft = 16.0

    def quantize(self, x, layer, kind):
        return x, x

    def draft_new(self, x, layer, kind):
        return x


class IntRep:
    """Draft: per-token asymmetric INT-b with groups of `group` channels. Target: exact KV."""
    per_token, shared = True, False

    def __init__(self, bits=4, group=32):
        self.b, self.group = bits, group
        self.name = f"int{bits}g{group}"
        self.bits_target = 16.0
        self.bits_draft = bits + 32.0 / group  # fp16 scale + zero per group

    def _q(self, x):
        sh = x.shape
        t = x.float().reshape(*sh[:-1], -1, self.group)
        lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
        s = ((hi - lo) / (2 ** self.b - 1)).clamp(min=1e-8)
        return (((t - lo) / s).round().clamp(0, 2 ** self.b - 1) * s + lo).reshape(sh).to(x.dtype)

    def quantize(self, x, layer, kind):
        return x, self._q(x)

    def draft_new(self, x, layer, kind):
        return self._q(x)


_PAT = re.compile(r"layers\.(\d+)\.attention\.quantizer\.(key|value)\.(\d+)\.(codebooks\.(\d+)\.weight|gains)")


def load_vqllm_codebooks(path, layer_dev, fmt=None, fmt_stages=None, fmt_rest=None):
    """VQLLM checkpoint -> {"key"/"value": per-layer effective codewords (S, NSV, C, D) fp32}.

    fmt / fmt_rest: storage format of the shape codewords (specrq.formats) for stages < fmt_stages / the rest
    (None = as stored). Gains are kept as stored.

    Effective codeword = gain * shape (GSRQ). SubVec n: n consecutive sub-vectors share one shape codebook
    (VQLLM `num_subvectors_per_codebook`); gains are always per sub-vector.
    HF-model checkpoints (hf_model00*.pt) use the "model.layers." prefix; both are accepted.
    """
    import glob
    import os
    files = sorted(glob.glob(os.path.join(glob.escape(path), "*.pt"))) if os.path.isdir(path) else [path]  # dir names hold "[16]"
    shapes, gains = {}, {}
    for f in files:
        ck = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        for k, t in ck.items():
            m = _PAT.search(k)
            if not m:
                continue
            l, kv, s = int(m.group(1)), m.group(2), int(m.group(3))
            if m.group(4) == "gains":
                gains[(kv, l, s)] = t
            else:
                shapes[(kv, l, s, int(m.group(5)))] = t
    if fmt or fmt_rest:
        from specrq.formats import quantize
        ns = fmt_stages if fmt_stages is not None else 1 << 30
        shapes = {k: quantize(t.float(), fmt if k[2] < ns else fmt_rest) for k, t in shapes.items()}
    out = {}
    for kv in ("key", "value"):
        L = 1 + max(l for (k_, l, s) in gains if k_ == kv)
        S = 1 + max(s for (k_, l, s) in gains if k_ == kv)
        NSV = gains[(kv, 0, 0)].shape[0]
        per_layer = []
        for l in range(L):
            # shared codebooks (SubVec n > 1): consecutive sub-vectors i use shape codebook i // n; gains stay per sub-vector
            n_cb = 1 + max(c for (k_, l_, s_, c) in shapes if k_ == kv and l_ == l and s_ == 0)
            share = NSV // n_cb
            sh = torch.stack([torch.stack([shapes[(kv, l, s, i // share)] for i in range(NSV)]) for s in range(S)])
            g = torch.stack([gains[(kv, l, s)] for s in range(S)])
            per_layer.append((sh.float() * g.float().unsqueeze(-1)).to(layer_dev[l]))
        out[kv] = per_layer
    return out


class RQRep:
    """Shared RQ indices (pre-RoPE). Draft reads the first m stages, target `target_stages` (or exact KV).

    window=G (default 128): like QuantSpec, the most recent G..2G tokens stay exact in both caches and are encoded
    block-wise in commit() once they age out (stored post-RoPE K is un-rotated, encoded, re-rotated).
    window=None: every token is encoded on write.
    """
    per_token, shared = True, False

    def __init__(self, model, codebook, m=2, target="fp16", draft_new="quant", target_stages=None, window=128,
                 cw_format=None, cw_stages=None, cw_format_rest=None):
        """cw_format / cw_format_rest: shape-codeword storage format of stages < cw_stages / the remaining stages."""
        self.cw = load_vqllm_codebooks(codebook, model.layer_dev, cw_format, cw_stages, cw_format_rest)
        S, NSV, C, D = self.cw["key"][0].shape
        assert NSV * D == model.n_kv * model.head_dim, "codebook does not match model KV width"
        self.S, self.NSV, self.C, self.D, self.m = S, NSV, C, D, m
        self.ts = target_stages or S
        assert m <= self.ts <= S
        self.target, self.draft_new_mode = target, draft_new
        self.G, self.model, self.qlen = window, model, 0
        self.sq = {kv: [c.pow(2).sum(-1) for c in self.cw[kv]] for kv in self.cw}
        rate = math.log2(C) / D
        self.bits_draft = m * rate
        self.bits_target = 16.0 if target == "fp16" else self.ts * rate
        self.name = (f"rq_S{S}C{C}D{D}_m{m}_" + ("fp16" if target == "fp16" else f"t{self.ts}")
                     + (f"_w{window}" if window else "")
                     + (f"_cw{cw_format}" + (f"x{cw_stages}" if cw_stages else "") if cw_format else "")
                     + (f"_rest{cw_format_rest}" if cw_format_rest else ""))

    def _encode_decode(self, x, layer, kind, stages_out):
        sh = x.shape
        r = x.float().reshape(-1, self.NSV, self.D).clone()
        cw, sq = self.cw[kind][layer], self.sq[kind][layer]
        ar = torch.arange(self.NSV, device=x.device)
        acc = torch.zeros_like(r)
        outs = {}
        for s in range(max(stages_out)):
            dist = sq[s].unsqueeze(0) - 2 * torch.einsum("nvd,vcd->nvc", r, cw[s])
            c = cw[s][ar, dist.argmin(-1)]
            r -= c
            acc += c
            if s + 1 in stages_out:
                outs[s + 1] = acc.reshape(sh).to(x.dtype, copy=True)  # acc keeps changing
        return outs

    def _codes(self, x, layer, kind):
        """x pre-RoPE -> (target, draft) reconstructions."""
        if self.target == "fp16":
            return x, self._encode_decode(x, layer, kind, {self.m})[self.m]
        o = self._encode_decode(x, layer, kind, {self.m, self.ts})
        return o[self.ts], o[self.m]

    def quantize(self, x, layer, kind):
        if self.G:   # windowed: exact on write, encoded in commit()
            return x, x
        return self._codes(x, layer, kind)

    def draft_new(self, x, layer, kind):
        if self.G or self.draft_new_mode == "fp16":
            return x
        return self._encode_decode(x, layer, kind, {self.m})[self.m]

    def _rot(self, K, a, b, inverse):
        """K: (B, H, T, hd) <-> RoPE at positions [a, b) (Meta interleaved layout)."""
        cos_t, sin_t = self.model.rot[K.device]
        cos, sin = cos_t[a:b][None, None], sin_t[a:b][None, None]
        if inverse:
            sin = -sin
        xf = K.float().unflatten(-1, (-1, 2))
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)

    def commit(self, store, n):
        if not self.G:
            return
        while n - self.qlen >= 2 * self.G:
            a, b = self.qlen, self.qlen + self.G
            for l in range(len(store.tk)):
                K = self._rot(store.tk[l][:, :, a:b], a, b, inverse=True).transpose(1, 2)   # (B, T, H, hd) pre-RoPE
                V = store.tv[l][:, :, a:b].float().transpose(1, 2)
                kt, kd = self._codes(K, l, "key")
                vt, vd = self._codes(V, l, "value")
                dt = store.tk[l].dtype
                if self.target != "fp16":
                    store.tk[l][:, :, a:b] = self._rot(kt.transpose(1, 2), a, b, inverse=False).to(dt)
                    store.tv[l][:, :, a:b] = vt.transpose(1, 2).to(dt)
                store.dk[l][:, :, a:b] = self._rot(kd.transpose(1, 2), a, b, inverse=False).to(dt)
                store.dv[l][:, :, a:b] = vd.transpose(1, 2).to(dt)
            self.qlen = b


# ============================================================================ QuantSpec (block)

class QuantSpecRep:
    """QuantSpec hierarchical INT8 = upper INT4 + lower INT4 (Tiwari et al., ICML'25).

    Keys: asymmetric, channel-wise, groups of G tokens.  Values: asymmetric, token-wise, groups of G channels.
    Upper:  C_U = clamp(round((x - Z) / S4), 0, 15),  S4 = 16 * S8, Z = min   (S8 = (max - min) / 255)
    Lower:  C_L = clamp(round((x - x̂_U) / S8), -8, 7)
    Draft reads x̂_U, target reads x̂_U + C_L * S8.  The most recent G..2G tokens stay full precision.
    Applied to post-RoPE K (the cache contents).
    """
    per_token, shared = False, False

    def __init__(self, group=128, bu=4, bl=4, target="fp16", scale="rtn"):
        """bu/bl: upper/lower bits (default INT8 = INT4 + INT4).
        scale: "rtn" (upper = RTN over the full range) or "paper" (QuantSpec's formula; only sound at 4 + 4).
        target: "fp16" (target keeps exact KV; isolates the draft's low-bit error), "full" (upper + lower), or
        "sep{b}" (target = independent plain b-bit RTN, draft = separate bu-bit RTN copy; not nested, stores bu + b)."""
        self.G, self.bu, self.bl, self.target, self.scale = group, bu, bl, target, scale
        self.name = (f"quantspec_g{group}" + ("" if (bu, bl) == (4, 4) else f"_u{bu}l{bl}") + f"_{target}"
                     + ("_paper" if scale == "paper" else ""))
        self.bits_draft = bu + 32.0 / group
        self.sep = int(target[3:]) if target.startswith("sep") else None
        self.bits_target = (16.0 if target == "fp16" else (self.sep if self.sep else bu + bl) + 32.0 / group)
        self.qlen = 0  # tokens [0, qlen) are quantized

    # new tokens enter the full-precision buffer; quantization happens per block in commit()
    quantize = FP16Rep.quantize
    draft_new = FP16Rep.draft_new

    def _hier(self, t, dim):
        if self.sep:   # draft = bu-bit RTN, target = independent sep-bit RTN (bl = 0 gives plain RTN)
            return self._hier_bits(t, dim, self.bu, 0)[0], self._hier_bits(t, dim, self.sep, 0)[0]
        return self._hier_bits(t, dim, self.bu, self.bl)

    def _hier_bits(self, t, dim, bu, bl):
        lo, hi = t.amin(dim, keepdim=True), t.amax(dim, keepdim=True)
        if self.scale == "paper":   # QuantSpec: S_U = 2^bl * S, S = range / (2^(bu+bl) - 1); top of range unreachable
            s = ((hi - lo) / (2 ** (bu + bl) - 1)).clamp(min=1e-8)
            su = 2 ** bl * s
            xu = ((t - lo) / su).round().clamp(0, 2 ** bu - 1) * su + lo
            cl = ((t - xu) / s).round().clamp(-2 ** (bl - 1), 2 ** (bl - 1) - 1)
            return xu, xu + cl * s
        # upper = plain RTN over the full range; lower = 2^bl mid-rise levels inside one upper cell
        su = ((hi - lo) / (2 ** bu - 1)).clamp(min=1e-8)
        s, o = su / 2 ** bl, (2 ** bl - 1) / 2
        xu = ((t - lo) / su).round().clamp(0, 2 ** bu - 1) * su + lo
        cl = ((t - xu) / s + o).round().clamp(0, 2 ** bl - 1)
        return xu, xu + (cl - o) * s

    def quantize_block(self, K, V):
        """K, V: (B, H, G, hd) exact block -> (K_draft, K_target, V_draft, V_target)."""
        kd, kt = self._hier(K.float(), dim=2)                       # per channel over the G tokens
        B, H, T, D = V.shape
        vd, vt = self._hier(V.float().view(B, H, T, D // self.G if D >= self.G else 1, -1), dim=-1)
        to = lambda x, ref: x.reshape(ref.shape).to(ref.dtype)
        return to(kd, K), to(kt, K), to(vd, V), to(vt, V)

    def commit(self, store, n):
        while n - self.qlen >= 2 * self.G:
            a, b = self.qlen, self.qlen + self.G
            for l in range(len(store.tk)):
                kd, kt, vd, vt = self.quantize_block(store.tk[l][:, :, a:b], store.tv[l][:, :, a:b])
                if self.target != "fp16":
                    store.tk[l][:, :, a:b], store.tv[l][:, :, a:b] = kt, vt
                store.dk[l][:, :, a:b], store.dv[l][:, :, a:b] = kd, vd
            self.qlen = b


# ============================================================================ sparse drafts

class StreamingRep:
    """StreamingLLM draft: attention sinks + most recent window, total `budget` tokens. Target exact."""
    per_token, shared = True, True
    bits_target = bits_draft = 16.0

    def __init__(self, budget, sink=4):
        self.budget, self.sink = budget, sink
        self.name = f"streaming_b{budget}_s{sink}"

    quantize = FP16Rep.quantize
    draft_new = FP16Rep.draft_new

    def draft_view(self, l, K, V, end):
        if end <= self.budget:
            return K[:, :, :end], V[:, :, :end]
        idx = torch.cat([torch.arange(self.sink), torch.arange(end - (self.budget - self.sink), end)]).to(K.device)
        return K.index_select(2, idx), V.index_select(2, idx)


class SnapKVRep:
    """SnapKV draft: per-head top prompt positions by pooled attention of the last `window` prompt
    queries, plus the window, plus every generated token; `budget` covers the prompt part. Target exact."""
    per_token, shared = True, True
    bits_target = bits_draft = 16.0

    def __init__(self, budget, window=32, kernel=5):
        self.budget, self.window, self.kernel = budget, window, kernel
        self.name = f"snapkv_b{budget}_w{window}_k{kernel}"
        self.sel, self.prompt_len = {}, None

    quantize = FP16Rep.quantize
    draft_new = FP16Rep.draft_new

    def observe(self, l, q, K):
        """q: (B, w, Hq, hd) last-window queries (post-RoPE); K: (B, Hkv, P, hd) full prompt keys."""
        B, w, Hq, hd = q.shape
        P, Hkv = K.shape[2], K.shape[1]
        if P <= self.budget:
            self.sel[l] = None
            return
        G = Hq // Hkv
        qf = q.float().permute(0, 2, 1, 3)                                       # (B, Hq, w, hd)
        Kf = K.float().repeat_interleave(G, 1)                                   # (B, Hq, P, hd)
        s = qf @ Kf.transpose(-1, -2) / math.sqrt(hd)                            # (B, Hq, w, P)
        qpos = torch.arange(P - w, P, device=K.device)
        s = s.masked_fill(torch.arange(P, device=K.device)[None, :] > qpos[:, None], float("-inf"))
        a = s.softmax(-1)[..., : P - w].sum(2)                                   # (B, Hq, P-w)
        a = a.view(B, Hkv, G, -1).mean(2)                                        # GQA: average the group
        a = F.avg_pool1d(a, self.kernel, stride=1, padding=self.kernel // 2)
        top = a.topk(self.budget - w, -1).indices.sort(-1).values                # (B, Hkv, budget-w)
        win = torch.arange(P - w, P, device=K.device).expand(B, Hkv, w)
        self.sel[l] = torch.cat([top, win], -1)                                  # (B, Hkv, budget)

    def draft_view(self, l, K, V, end):
        sel = self.sel.get(l)
        if sel is None:
            return K[:, :, :end], V[:, :, :end]
        gen = torch.arange(self.prompt_len, end, device=K.device).expand(*sel.shape[:2], -1)
        idx = torch.cat([sel, gen], -1)
        g = lambda X: X.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, X.shape[-1]))
        return g(K), g(V)


# ============================================================================ store

class KVStore:
    def __init__(self, model, rep, batch, max_len):
        self.rep = rep
        mk = lambda: [torch.empty((batch, model.n_kv, max_len, model.head_dim), device=model.layer_dev[l],
                                  dtype=model.dtype) for l in range(model.n_layers)]
        self.tk, self.tv = mk(), mk()
        self.dk, self.dv = (self.tk, self.tv) if rep.shared else (mk(), mk())
        self.max_len = max_len
        if hasattr(rep, "qlen"):
            rep.qlen = 0
        if isinstance(rep, SnapKVRep):
            rep.sel, rep.prompt_len = {}, None

    def write(self, l, k, v, start, mode, rope):
        """k, v: (B, T, n_kv, hd) pre-RoPE. Returns (K, V, kpos) for attention."""
        T = k.shape[1]
        end = start + T
        assert end <= self.max_len
        pos = torch.arange(start, end, device=k.device)
        if mode == "target":
            kt, kd = self.rep.quantize(k, l, "key")
            vt, vd = self.rep.quantize(v, l, "value")
            self.tk[l][:, :, start:end] = rope(kt, pos).transpose(1, 2)
            self.tv[l][:, :, start:end] = vt.transpose(1, 2)
            if not self.rep.shared:
                self.dk[l][:, :, start:end] = rope(kd, pos).transpose(1, 2)
                self.dv[l][:, :, start:end] = vd.transpose(1, 2)
            return self.tk[l][:, :, :end], self.tv[l][:, :, :end], torch.arange(end, device=k.device)
        kd = self.rep.draft_new(k, l, "key")
        vd = self.rep.draft_new(v, l, "value")
        self.dk[l][:, :, start:end] = rope(kd, pos).transpose(1, 2)
        self.dv[l][:, :, start:end] = vd.transpose(1, 2)
        if hasattr(self.rep, "draft_view"):
            K, V = self.rep.draft_view(l, self.dk[l], self.dv[l], end)
            return K, V, None  # draft passes are single-token: no mask needed
        return self.dk[l][:, :, :end], self.dv[l][:, :, :end], torch.arange(end, device=k.device)

    def after_prefill(self, n):
        if isinstance(self.rep, SnapKVRep):
            self.rep.prompt_len = n
        self.commit(n)

    def commit(self, n):
        if hasattr(self.rep, "commit"):
            self.rep.commit(self, n)
