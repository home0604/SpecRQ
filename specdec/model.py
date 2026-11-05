"""
Minimal Llama forward over Meta-format (consolidated.00.pth + params.json) or HF-format checkpoints.

Why our own forward instead of HF:
  - q/k are kept in Meta layout (interleaved RoPE pairs) — the layout VQLLM codebooks are trained on
    (torchtune converts HF checkpoints back to Meta layout), so pre-RoPE keys go to the codec as is.
  - We need a hook between k/v projection and RoPE, two KV caches (draft / target) that we write and
    roll back ourselves, sparse draft attention, and an optional INT4-weight draft.
  - Layers can be split across GPUs (`devices`), since MHA KV + two caches overflow one 24 GB card.
"""

import glob
import json
import math
import os

import torch
import torch.nn.functional as F


def _ffn_dim(dim, multiple_of, ffn_dim_multiplier):
    hidden = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


def _llama3_scale_freqs(freqs, factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, old_ctx=8192):
    low_wavelen, high_wavelen = old_ctx / low_freq_factor, old_ctx / high_freq_factor
    out = []
    for f in freqs.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < high_wavelen:
            out.append(f)
        elif wavelen > low_wavelen:
            out.append(f / factor)
        else:
            smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            out.append((1 - smooth) * f / factor + smooth * f)
    return torch.tensor(out, dtype=freqs.dtype)


def _hf_to_meta_qk(w, n_heads):
    """HF rotate_half row order -> Meta interleaved row order (inverse of the HF conversion permute)."""
    out_dim, in_dim = w.shape
    hd = out_dim // n_heads
    return w.view(n_heads, 2, hd // 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def _load_meta(ckpt_dir):
    params_path = os.path.join(ckpt_dir, "params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(ckpt_dir, "original", "params.json")
    p = json.load(open(params_path))
    sd = torch.load(os.path.join(os.path.dirname(params_path), "consolidated.00.pth"),
                    map_location="cpu", mmap=True, weights_only=True)
    cfg = dict(dim=p["dim"], n_layers=p["n_layers"], n_heads=p["n_heads"], n_kv=p.get("n_kv_heads", p["n_heads"]),
               vocab=p["vocab_size"], eps=p["norm_eps"], rope_theta=p.get("rope_theta", 10000.0),
               rope_scaling={"type": "llama3"} if p.get("use_scaled_rope") else None)
    return cfg, sd


def _load_hf(ckpt_dir):
    c = json.load(open(os.path.join(ckpt_dir, "config.json")))
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    raw = {}
    if files:
        from safetensors.torch import load_file
        for f in files:
            raw.update(load_file(f))
    else:
        for f in sorted(glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin"))):
            raw.update(torch.load(f, map_location="cpu", mmap=True, weights_only=True))
    nh, nkv = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    sd = {"tok_embeddings.weight": raw["model.embed_tokens.weight"], "norm.weight": raw["model.norm.weight"]}
    if "lm_head.weight" in raw:
        sd["output.weight"] = raw["lm_head.weight"]
    for l in range(c["num_hidden_layers"]):
        h, m = f"model.layers.{l}.", f"layers.{l}."
        sd[m + "attention.wq.weight"] = _hf_to_meta_qk(raw[h + "self_attn.q_proj.weight"], nh)
        sd[m + "attention.wk.weight"] = _hf_to_meta_qk(raw[h + "self_attn.k_proj.weight"], nkv)
        sd[m + "attention.wv.weight"] = raw[h + "self_attn.v_proj.weight"]
        sd[m + "attention.wo.weight"] = raw[h + "self_attn.o_proj.weight"]
        sd[m + "feed_forward.w1.weight"] = raw[h + "mlp.gate_proj.weight"]
        sd[m + "feed_forward.w3.weight"] = raw[h + "mlp.up_proj.weight"]
        sd[m + "feed_forward.w2.weight"] = raw[h + "mlp.down_proj.weight"]
        sd[m + "attention_norm.weight"] = raw[h + "input_layernorm.weight"]
        sd[m + "ffn_norm.weight"] = raw[h + "post_attention_layernorm.weight"]
    rs = c.get("rope_scaling")
    if rs is not None:
        rs = {"type": rs.get("rope_type", rs.get("type")), **rs}
    cfg = dict(dim=c["hidden_size"], n_layers=c["num_hidden_layers"], n_heads=nh, n_kv=nkv,
               vocab=c["vocab_size"], eps=c["rms_norm_eps"], rope_theta=c.get("rope_theta", 10000.0),
               rope_scaling=rs)
    return cfg, sd


class Int4Weight:
    """Packed INT4 weight (asymmetric RTN, groups of `group` input channels), dequantized on use.
    bits < 4 uses the same 4-bit container with codes in [0, 2^bits - 1]."""

    def __init__(self, w, group=128, bits=4):
        out_dim, in_dim = w.shape
        t = w.float().view(out_dim, in_dim // group, group)
        lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
        qmax = 2 ** bits - 1
        s = ((hi - lo) / qmax).clamp(min=1e-8)
        q = ((t - lo) / s).round().clamp(0, qmax).to(torch.uint8).view(out_dim, in_dim)
        self.packed = q[:, 0::2] | (q[:, 1::2] << 4)
        self.scale, self.zero = s.to(w.dtype), lo.to(w.dtype)
        self.shape, self.group, self.dtype = w.shape, group, w.dtype

    @classmethod
    def from_codes(cls, codes, scale, zero, dtype, group):
        """codes: (out, in) uint8 in [0, 15]; scale/zero: (out, in/group, 1)."""
        self = cls.__new__(cls)
        self.packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        self.scale, self.zero = scale.to(dtype), zero.to(dtype)
        self.shape, self.group, self.dtype = codes.shape, group, dtype
        return self

    def dequant(self):
        out_dim, in_dim = self.shape
        q = torch.stack([self.packed & 0xF, self.packed >> 4], -1).view(out_dim, in_dim // self.group, self.group)
        return (q.to(self.dtype) * self.scale + self.zero).view(out_dim, in_dim)


def nested_int(w, bu=4, bl=4, group=128):
    """Hierarchical (bu + bl)-bit weight (groups of `group` input channels): draft = upper bu bits, target = both.

    Returns (target weight = upper + lower, dequantized; draft = upper bu-bit as Int4Weight).
    Upper: plain RTN, C_U = clamp(round((w - lo) / S_U), 0, 2^bu - 1), S_U = (hi - lo) / (2^bu - 1)
    Lower: mid-rise inside one upper cell, C_L = clamp(round((w - ŵ_U) / S_L + o), 0, 2^bl - 1), o = (2^bl - 1) / 2,
           S_L = S_U / 2^bl, ŵ = ŵ_U + (C_L - o) S_L  (2^bl evenly spaced levels across [-S_U/2, S_U/2])
    QuantSpec's S_U = 2^bl (hi - lo) / (2^(bu+bl) - 1) with C_L in [-2^(bl-1), 2^(bl-1) - 1] is fine at 4 + 4, but at
    2 + 2 it leaves the top of the range unreachable and clips the lower code to [-S_U/2, S_U/4].
    """
    out_dim, in_dim = w.shape
    t = w.float().view(out_dim, in_dim // group, group)
    lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
    su = ((hi - lo) / (2 ** bu - 1)).clamp(min=1e-8)
    s = su / 2 ** bl
    cu = ((t - lo) / su).round().clamp(0, 2 ** bu - 1)
    wu = cu * su + lo
    o = (2 ** bl - 1) / 2
    cl = ((t - wu) / s + o).round().clamp(0, 2 ** bl - 1)
    target = (wu + (cl - o) * s).view(out_dim, in_dim).to(w.dtype)
    draft = Int4Weight.from_codes(cu.to(torch.uint8).view(out_dim, in_dim), su, lo, w.dtype, group)
    return target, draft


class Llama:
    """Weights + forward. Attention reads/writes K/V through a KVStore (see kvstore.py)."""

    def __init__(self, ckpt_dir, devices=("cuda:0",), dtype=torch.bfloat16, max_seq_len=8192, draft_wbits=16):
        """draft_wbits: 16 (draft = target weights), 2/3/4 (separate RTN draft copy, target bf16),
        "n{2b}" e.g. "n8"/"n6"/"n4" (nested: target dequantized 2b-bit, draft its upper b bits)."""
        if os.path.exists(os.path.join(ckpt_dir, "config.json")) and (
                glob.glob(os.path.join(ckpt_dir, "*.safetensors")) or glob.glob(os.path.join(ckpt_dir, "*.bin"))):
            cfg, sd = _load_hf(ckpt_dir)
        else:
            cfg, sd = _load_meta(ckpt_dir)
        self.dim, self.n_layers = cfg["dim"], cfg["n_layers"]
        self.n_heads, self.n_kv = cfg["n_heads"], cfg["n_kv"]
        self.head_dim = self.dim // self.n_heads
        self.vocab, self.eps = cfg["vocab"], cfg["eps"]
        self.dtype = dtype
        self.devices = [torch.device(d) for d in devices]
        self.layer_dev = [self.devices[l * len(self.devices) // self.n_layers] for l in range(self.n_layers)]
        self.device = self.devices[0]
        self.out_dev = self.layer_dev[-1]
        self.draft_wbits = draft_wbits

        g = lambda k, d: sd[k].to(device=d, dtype=dtype)
        self.tok_emb = g("tok_embeddings.weight", self.device)
        self.norm = g("norm.weight", self.out_dev)
        self.output = g("output.weight", self.out_dev) if "output.weight" in sd else self.tok_emb.to(self.out_dev)
        self.layers = []
        for l in range(self.n_layers):
            pre, d = f"layers.{l}.", self.layer_dev[l]
            L = dict(
                wqkv=torch.cat([g(pre + "attention.wq.weight", d), g(pre + "attention.wk.weight", d),
                                g(pre + "attention.wv.weight", d)], 0),
                wo=g(pre + "attention.wo.weight", d),
                w13=torch.cat([g(pre + "feed_forward.w1.weight", d), g(pre + "feed_forward.w3.weight", d)], 0),
                w2=g(pre + "feed_forward.w2.weight", d),
                attn_norm=g(pre + "attention_norm.weight", d), ffn_norm=g(pre + "ffn_norm.weight", d))
            if isinstance(draft_wbits, int) and draft_wbits < 16:   # separate RTN copy (QuantSpec draft); target bf16
                L["draft"] = {k: Int4Weight(L[k], bits=draft_wbits) for k in ("wqkv", "wo", "w13", "w2")}
            elif isinstance(draft_wbits, str):  # nested: target upper+lower, draft upper half of the same codes
                b = int(draft_wbits[1:]) // 2
                L["draft"] = {}
                for k in ("wqkv", "wo", "w13", "w2"):
                    L[k], L["draft"][k] = nested_int(L[k], b, b)
            self.layers.append(L)
        del sd
        self.ffn_dim = self.layers[0]["w2"].shape[1]

        freqs = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        rs = cfg["rope_scaling"]
        pos_div = 1.0
        if rs is not None and rs["type"] == "llama3":
            freqs = _llama3_scale_freqs(freqs, **{k: rs[k] for k in ("factor", "low_freq_factor", "high_freq_factor")
                                                  if k in rs})
        elif rs is not None and rs["type"] == "linear":
            pos_div = float(rs["factor"])
        t = torch.arange(max_seq_len, dtype=torch.float32) / pos_div
        ang = torch.outer(t, freqs)
        self.rot = {d: (ang.cos().to(d), ang.sin().to(d)) for d in self.devices}
        self.max_seq_len = max_seq_len

    # -------------------------------------------------------------- helpers
    def weight_bytes(self):
        """Bytes read per decode step (embedding lookup reads one row, so it is excluded)."""
        n = self.output.numel() + self.norm.numel()
        for L in self.layers:
            n += sum(L[k].numel() for k in ("wqkv", "wo", "w13", "w2", "attn_norm", "ffn_norm"))
        return n * self.tok_emb.element_size()

    def _rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def rope(self, x, pos):
        """x: (B, T, H, hd) Meta/interleaved layout; pos: (T,) positions on x's device."""
        cos_t, sin_t = self.rot[x.device]
        cos, sin = cos_t[pos][None, :, None, :], sin_t[pos][None, :, None, :]
        xf = x.float().unflatten(-1, (-1, 2))
        x0, x1 = xf[..., 0], xf[..., 1]
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2).to(x.dtype)

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, tokens, start, kv, mode, logits_last=None, observer=None):
        """tokens: (B, T) at positions [start, start+T).

        kv: KVStore. mode: "target" (prefill / verify) or "draft" (uses INT4 weights if enabled).
        observer(l, q, K): called with post-RoPE queries and the layer's full K (SnapKV selection).
        Returns float32 logits for the last `logits_last` positions (all if None), on devices[0].
        """
        B, T = tokens.shape
        h = F.embedding(tokens.to(self.device), self.tok_emb)
        qd, kvd = self.n_heads * self.head_dim, self.n_kv * self.head_dim
        G = self.n_heads // self.n_kv
        use_q4 = mode == "draft" and self.draft_wbits != 16
        for l, L in enumerate(self.layers):
            dev = self.layer_dev[l]
            h = h.to(dev)
            W = {k: L["draft"][k].dequant() for k in L["draft"]} if use_q4 else L
            pos = torch.arange(start, start + T, device=dev)
            x = self._rms(h, L["attn_norm"])
            q, k, v = (x @ W["wqkv"].T).split([qd, kvd, kvd], -1)
            q = self.rope(q.view(B, T, self.n_heads, self.head_dim), pos)
            # K/V (pre-RoPE, Meta layout) go through the KV representation; RoPE is applied inside.
            K, V, kpos = kv.write(l, k.view(B, T, self.n_kv, self.head_dim), v.view(B, T, self.n_kv, self.head_dim),
                                  start, mode, self.rope)
            if observer is not None:
                observer(l, q, K)
            # K, V: (B, n_kv, Tk, hd); kpos: (Tk,) absolute positions (sparse drafts select a subset)
            qg = q.view(B, T, self.n_kv, G, self.head_dim).permute(0, 2, 3, 1, 4).reshape(B, self.n_kv, G * T, self.head_dim)
            if T == 1:
                o = F.scaled_dot_product_attention(qg, K, V)
            else:
                mask = (kpos[None, :] <= pos[:, None]).repeat(G, 1)       # (G*T, Tk)
                o = F.scaled_dot_product_attention(qg, K, V, attn_mask=mask)
            o = o.view(B, self.n_kv, G, T, self.head_dim).permute(0, 3, 1, 2, 4).reshape(B, T, qd)
            h = h + o @ W["wo"].T
            x = self._rms(h, L["ffn_norm"])
            gate, up = (x @ W["w13"].T).split(self.ffn_dim, -1)
            h = h + (F.silu(gate) * up) @ W["w2"].T
            del W
        if logits_last is not None:
            h = h[:, -logits_last:]
        h = h.to(self.out_dev)
        return (self._rms(h, self.norm) @ self.output.T).float().to(self.device)

    @torch.no_grad()
    def prefill(self, tokens, kv, chunk=2048, observer_last=None):
        """Chunked prefill (bounds the attention mask to chunk x ctx). Returns last-position logits.

        observer_last: (window, fn) -> fn(l, q, K) is called on the final chunk with its last `window` queries.
        """
        T = tokens.shape[1]
        logits = None
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            obs = None
            if observer_last is not None and e == T:
                window, fn = observer_last
                obs = lambda l, q, K, w=window: fn(l, q[:, -w:], K)
            logits = self.forward(tokens[:, s:e], s, kv, "target", logits_last=1, observer=obs)
        kv.after_prefill(T)
        return logits
