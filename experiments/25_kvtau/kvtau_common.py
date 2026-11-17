"""E6 kvtau: shared pieces (model loading, custom Llama forward with a fake-quant nested VQ KV cache, KV codebooks).

KV format: pre-RoPE K and plain V of every kv-head (D = 128) are split into NSV = 128 / d sub-vectors (d = 8) in the
chosen layout ('hf' = HF k_proj row order, 'meta' = RoPE pairs (i, i + 64) adjacent, i.e. the Meta / torchtune order);
stage s has one codebook of C_s codewords per (layer, K/V, kv-head, stage) shared over the NSV positions (same
granularity as specrq/kvq). Encoding = greedy residual nearest codeword (exact fp32), per token, so a teacher-forced
full-sequence forward with fake-quant KV equals incremental decoding with a quantized cache (up to bf16 numerics).
A k-stage prefix reads the first k codes: recon_k = sum_{s<k} cb_s[code_s]. RoPE is applied after dequantization.
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
sys.path.insert(0, REPO)

MODEL = "/data/models/llama3.1_8b_instruct"
CKPT = "/data/lsh/ckpts/weightvq/sub1b8/d8K256L2_N_noft_n192"
DRAFT_B = "/data/lsh/results_accept/sub1b8/baseN_d8K256L2/kd1/train_B_bf16/draft_codebooks.pt"
OUTROOT = "/data/lsh/results_accept/kvtau"

FORMATS = {  # name -> (d, per-stage codebook sizes, draft prefix levels, refine prefix weights)
    "a_d8K256x2": (8, [256, 256], [1], {1: 1.0, 2: 1.0}),
    "b_d8K16K16K256": (8, [16, 16, 256], [1, 2], {1: 1.0, 2: 1.0, 3: 1.0}),
    "c_d8K16x4": (8, [16, 16, 16, 16], [1, 2], {1: 1.0, 2: 1.0, 4: 1.0}),
}


# ---------------------------------------------------------------- model
def load_model(wmode, dev="cuda:0", draft_b=True):
    """wmode 'bf16' (HF weights) or 'w2' (nested 2-bit DbD ckpt, k=2 target; draft-B k1 copies attached)."""
    from transformers import AutoModelForCausalLM
    if wmode == "bf16":
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    else:
        from src.modelutils import get_model
        sys.path.insert(0, os.path.join(REPO, "experiments", "06_accept"))
        from draft_target_eval import shrink_codes_
        m = get_model(MODEL, CKPT, "auto")
        print("codes -> uint8 for", shrink_codes_(m), "linears", flush=True)
    m = m.to(dev).eval()
    if wmode == "w2" and draft_b:
        from src.draft import load_draft_state_dict
        print("draft-B copies for", load_draft_state_dict(m, torch.load(DRAFT_B)), "linears", flush=True)
    return m


def set_weights(model, w):
    """w: 'bf16' (no-op), 'k2' (2-bit target, all stages, target codebooks), 'k1B' (stage 1 + trained draft copies)."""
    if w == "bf16":
        return
    from src.aq import QuantizedLinear
    import types
    k, use_draft = {"k2": (2, False), "k1B": (1, True)}[w]
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            if hasattr(m, "_kvtau_w2"):
                if w == "k2":
                    m.forward = types.MethodType(_dense_forward, m)
                elif "forward" in m.__dict__:
                    del m.forward
            m.active_num_codebooks = k
            if hasattr(m, "draft_k") or hasattr(m, "_draft_k_saved"):
                if not hasattr(m, "_draft_k_saved"):
                    m._draft_k_saved = m.draft_k
                m.draft_k = m._draft_k_saved if use_draft else None


def densify(model):
    """[w2] cache the k=2 target weights as dense bf16 (same values as QuantizedWeight(num_codebooks=2)); set_weights
    then routes 'k2' through them (fast decode) and 'k1B' through the on-the-fly dequant path."""
    from src.aq import QuantizedLinear
    n = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m._kvtau_w2 = m.quantized_weight(num_codebooks=2).detach().to(torch.bfloat16)
            n += 1
    return n


def _dense_forward(self, x):
    from src.hadamard import rotate_input
    signs = getattr(self, "rotation_signs", None)
    if signs is not None:
        x = rotate_input(x, signs, getattr(self, "hadamard_block_size", 1))
    return F.linear(x, self._kvtau_w2, self.bias)


def rope(x, cos, sin):
    """x [B, T, H, D]; cos/sin [B, T, D] (HF rotate-half convention)."""
    c, s = cos[:, :, None, :], sin[:, :, None, :]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return x * c + torch.cat((-x2, x1), -1) * s


# ---------------------------------------------------------------- KV quantizer
def to_layout(x, layout):
    """[..., 128] HF order -> layout order."""
    if layout == "hf":
        return x
    return x.unflatten(-1, (2, 64)).transpose(-1, -2).flatten(-2)


def from_layout(x, layout):
    if layout == "hf":
        return x
    return x.unflatten(-1, (64, 2)).transpose(-1, -2).flatten(-2)


@torch.no_grad()
def encode_stages(r, cbs, chunk=1 << 16):
    """r [H, N, d] fp32 residual (modified copy), cbs list of [H, C_s, d] fp32 -> codes list of [H, N] long."""
    r = r.clone()
    codes = []
    for cb in cbs:
        cn = (cb * cb).sum(-1)  # [H, C]
        cs = torch.empty(r.shape[:2], dtype=torch.long, device=r.device)
        for i in range(0, r.shape[1], chunk):
            rr = r[:, i:i + chunk]
            dist = cn[:, None, :] - 2 * torch.bmm(rr, cb.transpose(1, 2))
            c = dist.argmin(-1)
            cs[:, i:i + chunk] = c
            r[:, i:i + chunk] = rr - torch.gather(cb, 1, c[..., None].expand(-1, -1, cb.shape[-1]))
        codes.append(cs)
    return codes


def gather_cb(cb, c):
    return torch.gather(cb, 1, c[..., None].expand(-1, -1, cb.shape[-1]))


class KVQuant:
    """cb[l][kv] = list over stages of fp32 [H, C_s, d] on device."""

    def __init__(self, path, dev="cuda:0"):
        st = torch.load(path)
        self.d, self.layout, self.sizes = st["d"], st["layout"], st["sizes"]
        self.per_pos = bool(st.get("per_pos", False))
        self.cb = [[[c.to(dev).float() for c in st["cb"][l][kv]] for kv in range(2)] for l in range(len(st["cb"]))]
        self.S = len(self.sizes)
        self.meta = {k: v for k, v in st.items() if k != "cb"}
        self.mu = self.sd = None
        if st.get("mu") is not None:  # static per-channel standardization (calib_kv.py --chnorm)
            self.mu = [[t.to(dev).float() for t in x] for x in st["mu"]]
            self.sd = [[t.to(dev).float() for t in x] for x in st["sd"]]
        self.keep_first = 0  # positions < keep_first stay bf16 (attention-sink option; 0 = quantize every token)

    def quant_pos(self, l, kv, x, levels, p0):
        """quant() with the keep_first exception; x [B, T, H, D] covers absolute positions p0 .. p0 + T - 1."""
        out = self.quant(l, kv, x, levels)
        n = min(self.keep_first - p0, x.shape[1])
        if n > 0:
            for lv in out:
                out[lv][:, :n] = x[:, :n]
        return out

    @torch.no_grad()
    def quant(self, l, kv, x, levels):
        """x [B, T, H, 128] (pre-RoPE K or V, any float dtype) -> {level: recon same shape/dtype}; level <= S stages."""
        B, T, H, D = x.shape
        d = self.d
        xf = x.float()
        if self.mu is not None:
            xf = (xf - self.mu[l][kv]) / self.sd[l][kv]
        xl = to_layout(xf, self.layout).reshape(B * T, H, D // d, d)
        if self.per_pos:
            xl = xl.permute(1, 2, 0, 3).reshape(H * (D // d), B * T, d)
        else:
            xl = xl.permute(1, 0, 2, 3).reshape(H, -1, d)
        nmax = max(levels)
        codes = encode_stages(xl, self.cb[l][kv][:nmax])
        out, acc = {}, torch.zeros_like(xl)
        for s in range(nmax):
            acc = acc + gather_cb(self.cb[l][kv][s], codes[s])
            if s + 1 in levels:
                if self.per_pos:
                    y = acc.reshape(H, D // d, B * T, d).permute(2, 0, 1, 3).reshape(B, T, H, D)
                else:
                    y = acc.reshape(H, B * T, D // d, d).permute(1, 0, 2, 3).reshape(B, T, H, D)
                y = from_layout(y, self.layout)
                if self.mu is not None:
                    y = y * self.sd[l][kv] + self.mu[l][kv]
                out[s + 1] = y.to(x.dtype)
        return out


# ---------------------------------------------------------------- forward
class Fwd:
    """Custom Llama forward over the HF modules (QuantizedLinear or nn.Linear projections)."""

    def __init__(self, model):
        self.m = model
        self.mm = model.model
        cfg = model.config
        self.H, self.Hkv, self.D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        self.G = self.H // self.Hkv
        self.nl = cfg.num_hidden_layers

    def cos_sin(self, h, pos):
        return self.mm.rotary_emb(h, pos[None])

    def qkv(self, layer, h):
        B, T, _ = h.shape
        x = layer.input_layernorm(h)
        a = layer.self_attn
        q = a.q_proj(x).view(B, T, self.H, self.D)
        k = a.k_proj(x).view(B, T, self.Hkv, self.D)
        v = a.v_proj(x).view(B, T, self.Hkv, self.D)
        return q, k, v

    def post(self, layer, h, o, chunk=2048):
        """o_proj + residual + MLP, in row chunks of `chunk` tokens (memory at long context x G draft steps)."""
        B, T = o.shape[:2]
        hf, of = h.reshape(B * T, -1), o.reshape(B * T, -1)
        out = torch.empty_like(hf)
        for i in range(0, B * T, chunk):
            x = hf[i:i + chunk] + layer.self_attn.o_proj(of[i:i + chunk])
            out[i:i + chunk] = x + layer.mlp(layer.post_attention_layernorm(x))
        return out.view(B, T, -1)

    def sdpa(self, q, k, v, causal):
        """q [B,T,H,D], k/v [B,Tk,Hkv,D] -> [B,T,H,D]"""
        k = k.repeat_interleave(self.G, dim=2)
        v = v.repeat_interleave(self.G, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
        return o.transpose(1, 2)

    @torch.no_grad()
    def forward(self, ids, kvq=None, level=None, cache=None, collect=None, return_hidden=False, fp32attn=False):
        """Target-style forward. kvq/level: fake-quant KV at `level` stages (None = bf16 KV).
        cache: dict with 'k','v' lists (post-RoPE, dequantized) and 'pos'; appended in place (decode).
        collect: dict level -> lists of post-RoPE (K, V) recon per layer for draft prefix levels (teacher-forced)."""
        B, T = ids.shape
        p0 = cache["pos"] if cache is not None else 0
        pos = torch.arange(p0, p0 + T, device=ids.device)
        h = self.mm.embed_tokens(ids)
        cos, sin = self.cos_sin(h, pos)
        for l, layer in enumerate(self.mm.layers):
            q, k, v = self.qkv(layer, h)
            if kvq is not None:
                lv = sorted(set([level] + (list(collect.keys()) if collect is not None else [])))
                rk, rv = kvq.quant_pos(l, 0, k, lv, p0), kvq.quant_pos(l, 1, v, lv, p0)
                if collect is not None:
                    for c in collect:
                        collect[c].append((rope(rk[c], cos, sin), rv[c]))
                k, v = rk[level], rv[level]
            elif collect is not None:
                for c in collect:
                    collect[c].append((rope(k, cos, sin), v))
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            if cache is not None:
                if len(cache["k"]) <= l:
                    cache["k"].append(k); cache["v"].append(v)
                else:
                    cache["k"][l] = torch.cat([cache["k"][l], k], 1); cache["v"][l] = torch.cat([cache["v"][l], v], 1)
                k, v = cache["k"][l], cache["v"][l]
            assert T == 1 or k.shape[1] == T, "multi-token forward only without prior cache"
            if fp32attn:
                h = self.post(layer, h, self.attn_fp32_causal(q, k, v))
            else:
                h = self.post(layer, h, self.sdpa(q, k, v, causal=T > 1))
        if cache is not None:
            cache["pos"] = p0 + T
        if return_hidden:
            return h
        return self.m.lm_head(self.mm.norm(h)).float()

    def attn_fp32_causal(self, q, k, v, qchunk=512):
        """Same arithmetic as draft_steps (fp32 scores/softmax/PV, bf16 out), plain causal; q [1,T,H,D], k/v [1,T,Hkv,D].
        Used for the teacher-forced target pass so that target and draft logits share one attention implementation."""
        _, T, H, D = q.shape
        Hkv, Gq = self.Hkv, self.G
        Kf, Vf = k[0].float().permute(1, 0, 2), v[0].float().permute(1, 0, 2)
        out = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D)
        for i0 in range(0, T, qchunk):
            i1 = min(T, i0 + qchunk)
            Tq = i1 - i0
            qi = torch.arange(i0, i1, device=q.device)
            qq = (q[0, i0:i1].float() * scale).view(Tq, Hkv, Gq, D).permute(1, 2, 0, 3)
            s = torch.matmul(qq, Kf[:, None, :i1].transpose(-1, -2))
            jj = torch.arange(i1, device=q.device)
            s = s.masked_fill((jj[None, :] > qi[:, None])[None, None], float("-inf"))
            o = torch.matmul(s.softmax(-1), Vf[:, None, :i1])
            out[0, i0:i1] = o.permute(2, 0, 1, 3).reshape(Tq, H, D).to(q.dtype)
        return out

    def logits(self, h):
        return self.m.lm_head(self.mm.norm(h)).float()

    @torch.no_grad()
    def draft_steps(self, ids, tkv, kvq, level, G, qchunk=512):
        """Teacher-forced multi-step draft along a fixed token path (shared-KV semantics).
        F_m (m = 1..G) = the m-th draft step of a round: query position i attends to the TARGET's KV (read at `level`
        stages; tkv[l] = (K post-RoPE, V)) for positions j <= i - m and to the DRAFT's own KV (computed by F_{m'} and
        encoded at `level` stages; level None = bf16) for positions i - m + m' (m' = 1..m), the last being i itself.
        Returns final hidden [G, T, hidden]."""
        _, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        h = self.mm.embed_tokens(ids).expand(G, -1, -1).contiguous()
        cos, sin = self.cos_sin(h[:1], pos)
        scale = 1.0 / math.sqrt(self.D)
        Hkv, Gq, D = self.Hkv, self.G, self.D
        for l, layer in enumerate(self.mm.layers):
            q, k, v = self.qkv(layer, h)  # [G, T, ...]
            if kvq is not None and level is not None:
                k, v = kvq.quant_pos(l, 0, k, [level], 0)[level], kvq.quant_pos(l, 1, v, [level], 0)[level]
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            Kt, Vt = tkv[l][0][0], tkv[l][1][0]  # [T, Hkv, D]
            Ktf = Kt.float().permute(1, 0, 2)  # [Hkv, T, D]
            Vtf = Vt.float().permute(1, 0, 2)
            kf, vf = k.float(), v.float()  # [G, T, Hkv, D]
            out = torch.empty(G, T, self.H, D, dtype=h.dtype, device=h.device)
            for m in range(1, G + 1):
                for i0 in range(0, T, qchunk):
                    i1 = min(T, i0 + qchunk)
                    Tq = i1 - i0
                    qi = torch.arange(i0, i1, device=h.device)
                    qq = (q[m - 1, i0:i1].float() * scale).view(Tq, Hkv, Gq, D).permute(1, 2, 0, 3)  # [Hkv,Gq,Tq,D]
                    kend = max(0, i1 - m)
                    if kend > 0:
                        st = torch.matmul(qq, Ktf[:, None, :kend].transpose(-1, -2))  # [Hkv,Gq,Tq,kend]
                        jj = torch.arange(kend, device=h.device)
                        st = st.masked_fill((jj[None, :] > (qi[:, None] - m))[None, None], float("-inf"))
                    else:
                        st = qq.new_empty(Hkv, Gq, Tq, 0)
                    so, vo = [], []
                    for mp in range(1, m + 1):
                        j = qi - m + mp
                        valid = j >= 0
                        jc = j.clamp(min=0)
                        ko = kf[mp - 1, jc].permute(1, 0, 2)  # [Hkv, Tq, D]
                        s = (qq * ko[:, None]).sum(-1)  # [Hkv,Gq,Tq]
                        so.append(s.masked_fill(~valid[None, None], float("-inf")))
                        vo.append(vf[mp - 1, jc].permute(1, 0, 2))  # [Hkv,Tq,D]
                    S = torch.cat([st, torch.stack(so, -1)], -1)
                    P = S.softmax(-1)
                    o = torch.matmul(P[..., :kend], Vtf[:, None, :kend]) if kend > 0 else 0
                    for idx in range(m):
                        o = o + P[..., kend + idx, None] * vo[idx][:, None]
                    out[m - 1, i0:i1] = o.permute(2, 0, 1, 3).reshape(Tq, self.H, D).to(h.dtype)
            h = self.post(layer, h, out)
        return h
