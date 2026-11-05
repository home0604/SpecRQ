"""Inference runner for DbD nested-VQ checkpoints (Qwen2 / Llama) on the nestkern kernels.

Load path: DbD layer pickles (`{i}.pth`, QuantizedLinear modules) + `not_quantized_weights.pt` -> planar
(`specrq.nestkern.pack`) -> per-linear kernel layouts:
  mma  (`mma_codes`)    always (all M <= 16; M > 16 is processed in chunks of 16 rows)
  gemv (`kernel_codes`) only when K <= 4096 (M=1 dispatch rule from E0/M3: gemv if M == 1 and K <= 4096)
Each linear keeps its own codebooks (so q/k/v and gate/up are not fused). Optional draft codebook sets
{k_d: (codebooks [k_d, 256, 8], scales [N])} can be installed per linear (DbD src/draft.py state-dict format).
Hadamard input rotation (DbD rotation_signs / hadamard_block_size) uses the nestkern block FWHT kernel.
embed / norms / lm_head stay bf16 (lm_head = embed when tied).

Forward is written for static shapes so decode steps can be captured in CUDA graphs: the KV cache is a static
[n_layers, H_kv, T_max, D] tensor pair, the write positions and attention mask come from a device `pos` tensor.
Math follows HF Qwen2/Llama (RMSNorm in fp32, rotate_half RoPE from transformers' ROPE_INIT_FUNCTIONS, fp32
softmax inside SDPA as in the DbD-pickled Qwen2SdpaAttention), so logits should match the DbD/HF path to bf16
rounding. T_max should be a multiple of 8 (SDPA mask alignment).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from specrq.nestkern.ops import block_fwht_signed, nest_gemv, nest_mma
from specrq.nestkern.pack import kernel_codes, mma_codes, pack_planar

REPO = Path(__file__).resolve().parents[2]
DBD = REPO / "specrq" / "dbd"
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class NestLinear:
    GEMV_MAX_K = 4096
    DETERMINISTIC = True  # ordered split-K reduction in nest_mma (bitwise reproducible); False = fp32 atomics

    def __init__(self, qlin, device, dtype):
        qw = qlin.quantized_weight
        p = pack_planar(qw.get_codes(), qw.get_codebooks().detach(), qw.get_scales().detach()).to(device, dtype)
        self.N, self.K, self.L = p.N, p.K, p.L
        self.mcodes = mma_codes(p.codes)
        self.kcodes = kernel_codes(p.codes) if self.K <= self.GEMV_MAX_K else None
        self.cb, self.sc = p.codebooks, p.scales
        b = getattr(qlin, "bias", None)
        self.bias = None if b is None else b.detach().to(device, dtype)
        s = getattr(qlin, "rotation_signs", None)
        self.hb = int(getattr(qlin, "hadamard_block_size", 1) or 1)
        self.signs = None if (s is None or self.hb <= 1) else s.detach().float().to(device).contiguous()
        self.draft = {}  # k_d -> (codebooks [k_d, 256, 8], scales [N])

    @classmethod
    def synthetic(cls, N, K, device, dtype, L=5, hadamard_block=256, bias=False, gen=None):
        """Random C1-format linear (codes/codebooks/scales/signs) built directly on the GPU (timing only)."""
        self = cls.__new__(cls)
        self.N, self.K, self.L = N, K, L
        codes = torch.randint(0, 256, (L, N, K // 8), device=device, dtype=torch.uint8, generator=gen)
        self.mcodes = mma_codes(codes)
        self.kcodes = kernel_codes(codes) if K <= cls.GEMV_MAX_K else None
        del codes
        self.cb = (torch.randn(L, 256, 8, device=device, generator=gen) * (3.0 / (K * L) ** 0.5)).to(dtype)
        self.sc = (torch.rand(N, device=device, generator=gen) + 0.5).to(dtype)
        self.bias = (torch.randn(N, device=device, generator=gen) * 0.01).to(dtype) if bias else None
        self.hb = hadamard_block if (hadamard_block > 1 and K % hadamard_block == 0) else 1
        self.signs = (torch.randint(0, 2, (K,), device=device, generator=gen).float() * 2 - 1) if self.hb > 1 else None
        self.draft = {}
        return self

    def set_draft(self, codebooks, scales):
        """codebooks [k_d, 256, 1, 8] or [k_d, 256, 8]; scales [N, 1, 1, 1] or [N] (DbD draft_state_dict format)."""
        cb = codebooks.reshape(codebooks.shape[0], 256, 8).to(self.cb.device, self.cb.dtype).contiguous()
        self.draft[cb.shape[0]] = (cb, scales.reshape(-1).to(self.sc.device, self.sc.dtype).contiguous())

    def __call__(self, x, k, use_draft=False):
        if self.signs is not None:
            x = block_fwht_signed(x, self.signs, self.hb)
        cb, sc = self.draft[k] if (use_draft and k in self.draft) else (self.cb, self.sc)
        M = x.shape[0]
        if M == 1 and self.kcodes is not None:
            y = nest_gemv(x, self.kcodes, cb, sc, k)
        elif M <= 16:
            y = nest_mma(x, self.mcodes, cb, sc, k, deterministic=self.DETERMINISTIC)
        else:
            y = torch.cat([nest_mma(x[i:i + 16], self.mcodes, cb, sc, k, deterministic=self.DETERMINISTIC)
                           for i in range(0, M, 16)])
        return y if self.bias is None else y + self.bias

    def nbytes(self, k):
        return k * self.N * self.K // 8 + k * 256 * 8 * 2 + 2 * self.N


# ---------------------------------------------------------------------------------------------- fused glue (decode)
def _rms_f(x, w, eps: float):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def _rope_f(q, k, cos, sin):
    return q * cos + _rot_half(q) * sin, k * cos + _rot_half(k) * sin


def _act_f(g, u):
    return F.silu(g) * u


def _add_rms_f(h, d, w, eps: float):
    h = h + d
    return h, _rms_f(h, w, eps)


def _attn_f(qh, kc, vc, mask, scale: float):
    """qh [Hkv, G*M, D], kc/vc [Hkv, T, D] (bf16), mask [G*M, T] bool -> [Hkv, G*M, D]; fp32 scores/softmax/PV."""
    s = torch.bmm(qh.float(), kc.float().transpose(1, 2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    return torch.bmm(torch.softmax(s, -1), vc.float()).to(qh.dtype)


_FUSED = {"rms": _rms_f, "rope": _rope_f, "act": _act_f, "add_rms": _add_rms_f, "attn": _attn_f}
_COMPILED = {}


def fused(name, use_compile):
    if not use_compile:
        return _FUSED[name]
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(_FUSED[name], dynamic=False, fullgraph=True)
    return _COMPILED[name]


@dataclass
class Cache:
    k: torch.Tensor  # [n_layers, H_kv, T, D]
    v: torch.Tensor


def _rope_inv_freq(cfg, device):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    rs = getattr(cfg, "rope_scaling", None) or {}
    rtype = rs.get("rope_type", rs.get("type", "default"))
    inv_freq, scaling = ROPE_INIT_FUNCTIONS[rtype](cfg, device)
    return inv_freq.float(), float(scaling)


class NestModel:
    def __init__(self, model_path, ckpt_dir, device="cuda", dtype=torch.bfloat16, verbose=True):
        from transformers import AutoConfig
        self.cfg = cfg = AutoConfig.from_pretrained(model_path)
        self.device, self.dtype = torch.device(device), dtype
        self.n_layers, self.H, self.Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        if str(DBD) not in sys.path:
            sys.path.insert(0, str(DBD))  # DbD pickles reference `src.*`
        nq = torch.load(os.path.join(ckpt_dir, "not_quantized_weights.pt"), map_location="cpu", weights_only=False)
        g = lambda n: nq[n].detach().to(self.device, dtype)
        self.embed = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in nq else self.embed
        self.layers = []
        for i in range(self.n_layers):
            lay = torch.load(os.path.join(ckpt_dir, f"{i}.pth"), map_location="cpu", weights_only=False)
            mods = dict(lay.named_modules())
            d = {"in_norm": lay.input_layernorm.weight.detach().to(self.device, dtype),
                 "post_norm": lay.post_attention_layernorm.weight.detach().to(self.device, dtype)}
            for name in LINEARS:
                d[name.split(".")[1]] = NestLinear(mods[name], self.device, dtype)
            self.layers.append(d)
            del lay
            if verbose and i % 8 == 0:
                print(f"[infer] loaded layer {i}/{self.n_layers}", flush=True)
        self.L = self.layers[0]["q_proj"].L
        self.inv_freq, self.rope_scaling = _rope_inv_freq(cfg, self.device)
        self.use_compile = True  # torch.compile the elementwise glue on decode-sized calls (M <= 16)
        self.decode_attn = "fp32"  # "fp32" (compiled fp32 bmm/softmax) or "sdpa" for M <= 16
        self._rope_len = 0
        self.ckpt_dir, self.model_path = ckpt_dir, model_path

    @classmethod
    def synthetic(cls, model_path, device="cuda", dtype=torch.bfloat16, L=5, hadamard=True, seed=0, verbose=True):
        """Real embed / norms / lm_head from the HF safetensors of `model_path` (only those tensors are read) and
        random C1 linears (codes, codebooks, scales, Hadamard signs) generated on the GPU. For timing only."""
        import json as _json
        from safetensors import safe_open
        from transformers import AutoConfig
        self = cls.__new__(cls)
        self.cfg = cfg = AutoConfig.from_pretrained(model_path)
        self.device, self.dtype = torch.device(device), dtype
        self.n_layers, self.H, self.Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        idx = _json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]

        def get(name):
            with safe_open(os.path.join(model_path, idx[name]), framework="pt", device=str(self.device)) as f:
                return f.get_tensor(name).to(dtype)
        self.embed = get("model.embed_tokens.weight")
        self.norm = get("model.norm.weight")
        self.lm_head = get("lm_head.weight") if "lm_head.weight" in idx else self.embed
        hid, inter = cfg.hidden_size, cfg.intermediate_size
        shapes = {"q_proj": (self.H * self.D, hid), "k_proj": (self.Hkv * self.D, hid), "v_proj": (self.Hkv * self.D, hid),
                  "o_proj": (hid, self.H * self.D), "gate_proj": (inter, hid), "up_proj": (inter, hid),
                  "down_proj": (hid, inter)}
        bias = getattr(cfg, "attention_bias", False) or cfg.model_type == "qwen2"
        gen = torch.Generator(device=self.device).manual_seed(seed)
        self.layers = []
        for i in range(self.n_layers):
            d = {"in_norm": get(f"model.layers.{i}.input_layernorm.weight"),
                 "post_norm": get(f"model.layers.{i}.post_attention_layernorm.weight")}
            for n, (N, K) in shapes.items():
                d[n] = NestLinear.synthetic(N, K, self.device, dtype, L, 256 if hadamard else 1,
                                            bias=bias and n in ("q_proj", "k_proj", "v_proj"), gen=gen)
            self.layers.append(d)
            if verbose and i % 8 == 0:
                print(f"[infer] synthetic layer {i}/{self.n_layers}", flush=True)
        self.L = L
        self.inv_freq, self.rope_scaling = _rope_inv_freq(cfg, self.device)
        self.use_compile, self.decode_attn, self._rope_len = True, "fp32", 0
        self.ckpt_dir, self.model_path = f"synthetic:C1 L={L} hadamard={hadamard} seed={seed}", model_path
        return self

    # ------------------------------------------------------------------ draft codebooks
    def load_draft(self, sd: dict) -> int:
        """Install draft codebook sets from a DbD draft_state_dict ({'model.layers.i.<lin>.draft_codebooks', ...})."""
        n = 0
        for i, lay in enumerate(self.layers):
            for name in LINEARS:
                key = f"model.layers.{i}.{name}"
                if f"{key}.draft_codebooks" in sd:
                    lay[name.split(".")[1]].set_draft(sd[f"{key}.draft_codebooks"], sd[f"{key}.draft_scales"])
                    n += 1
        return n

    # ------------------------------------------------------------------ cache
    def new_cache(self, T) -> Cache:
        T = (T + 15) // 16 * 16
        shp = (self.n_layers, self.Hkv, T, self.D)
        return Cache(torch.zeros(shp, device=self.device, dtype=self.dtype),
                     torch.zeros(shp, device=self.device, dtype=self.dtype))

    def linear_bytes(self, k):
        return sum(lay[n.split(".")[1]].nbytes(k) for lay in self.layers for n in LINEARS)

    # ------------------------------------------------------------------ forward
    def _rms(self, x, w):
        return _rms_f(x, w, self.eps)

    _rot_half = staticmethod(_rot_half)

    def _rope_tables(self, T):
        if T > self._rope_len:
            n = max(T, 4096)
            freqs = torch.arange(n, device=self.device).float()[:, None] * self.inv_freq[None]
            emb = torch.cat((freqs, freqs), -1)
            self._cos = (emb.cos() * self.rope_scaling).to(self.dtype)
            self._sin = (emb.sin() * self.rope_scaling).to(self.dtype)
            self._rope_len = n
        return self._cos, self._sin

    def forward(self, tokens, pos, cache: Cache, k=None, use_draft=False, last_only=False, capture=None):
        """tokens [M] long (device); pos: 0-d long device tensor (position of tokens[0]); returns logits [M or 1, V]."""
        k = self.L if k is None else k
        M = tokens.shape[0]
        T = cache.k.shape[2]
        uc = self.use_compile and M <= 16
        rms, rope, act, add_rms = (fused(n, uc) for n in ("rms", "rope", "act", "add_rms"))
        attn_c = fused("attn", True) if (uc and self.decode_attn == "fp32") else None
        positions = pos + torch.arange(M, device=self.device)
        ct, st = self._rope_tables(T)
        cos, sin = ct[positions][:, None, :], st[positions][:, None, :]
        allowed = torch.arange(T, device=self.device)[None, :] <= positions[:, None]  # [M, T]
        G = self.H // self.Hkv
        mask = allowed.repeat(G, 1)[None, None]  # [1, 1, G*M, T], rows ordered (g, m)
        scale = self.D ** -0.5
        h = F.embedding(tokens, self.embed)
        x = rms(h, self.layers[0]["in_norm"], self.eps)
        for li, lay in enumerate(self.layers):
            q = lay["q_proj"](x, k, use_draft).view(M, self.H, self.D)
            kk = lay["k_proj"](x, k, use_draft).view(M, self.Hkv, self.D)
            v = lay["v_proj"](x, k, use_draft).view(M, self.Hkv, self.D)
            q, kk = rope(q, kk, cos, sin)
            cache.k[li].index_copy_(1, positions, kk.transpose(0, 1))
            cache.v[li].index_copy_(1, positions, v.transpose(0, 1))
            # GQA without repeating K/V: the G query heads of a KV head are stacked as G*M query rows of one SDPA
            # head (fused kernel, fp32 accumulation; bf16-rounded scores in an eager matmul are too coarse when
            # |q.k| is large, e.g. Qwen2.5 layer 0 has scores up to ~1.7e3)
            qh = q.view(M, self.Hkv, G, self.D).permute(1, 2, 0, 3).reshape(1, self.Hkv, G * M, self.D)
            if attn_c is not None:  # decode: fp32 attention, compiled (cheaper than mem-efficient SDPA for small M)
                o = attn_c(qh[0], cache.k[li], cache.v[li], mask[0, 0], scale)
            else:
                o = F.scaled_dot_product_attention(qh, cache.k[li][None], cache.v[li][None], attn_mask=mask,
                                                   scale=scale)
            o = o.view(self.Hkv, G, M, self.D).permute(2, 0, 1, 3).reshape(M, -1)
            h, x = add_rms(h, lay["o_proj"](o, k, use_draft), lay["post_norm"], self.eps)
            a = act(lay["gate_proj"](x, k, use_draft), lay["up_proj"](x, k, use_draft))
            nxt = self.layers[li + 1]["in_norm"] if li + 1 < self.n_layers else self.norm
            h, x = add_rms(h, lay["down_proj"](a, k, use_draft), nxt, self.eps)
            if capture is not None:
                capture.append(h)
        if last_only:
            x = x[-1:]
        return F.linear(x, self.lm_head)

    def prefill(self, tokens, cache, k=None, use_draft=False):
        """Process the prompt (any length, chunks of 16 inside the linears); returns last-position logits [1, V]."""
        pos = torch.zeros((), dtype=torch.long, device=self.device)
        return self.forward(tokens, pos, cache, k, use_draft, last_only=True)


def load_draft_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)
