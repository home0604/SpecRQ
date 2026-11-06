"""Inference runner for DbD nested-VQ checkpoints (Qwen2 / Llama) on the nestkern kernels.

Load path: DbD layer pickles (`{i}.pth`, QuantizedLinear modules) + `not_quantized_weights.pt` -> planar
(`specrq.nestkern.pack`) -> per-linear kernel layouts:
  mma  (`mma_codes`)    always (all M <= 16; M > 16 is processed in chunks of 16 rows)
  gemv (`kernel_codes`) only when K <= 4096. Dispatch: M == 1 and K <= 4096 -> nest_gemv (E0/M3);
                        M in {2, 3} and N*K <= 1.2e6 (and K <= 4096) -> nest_gemv_m (P4); else nest_mma
Each linear keeps its own codebooks (so q/k/v and gate/up are not fused). Optional draft codebook sets
{k_d: (codebooks [k_d, 256, 8], scales [N])} can be installed per linear (DbD src/draft.py state-dict format).
Hadamard input rotation (DbD rotation_signs / hadamard_block_size) uses the nestkern block FWHT kernel.
embed / norms / lm_head stay bf16 (lm_head = embed when tied).
FR-Spec draft vocab (set_draft_vocab): a contiguous [V_d, H] bf16 copy of the lm_head rows of the V_d most frequent
tokens; forward(..., vd=V_d) returns logits over that subset (column j = token dvocab[V_d][0][j]); target / verify
use vd=None (full lm_head). The copy is extra memory (V_d * H * 2 bytes; Llama-3.1-8B V_d=32k: 0.27 GB).

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

from specrq.nestkern.ops import act_rot, add_rms_rot, block_fwht_signed, nest_gemv, nest_gemv_m, nest_mma
from specrq.nestkern.pack import kernel_codes, mma_codes, pack_planar

REPO = Path(__file__).resolve().parents[2]
DBD = REPO / "specrq" / "dbd"
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class NestLinear:
    GEMV_MAX_K = 4096
    GEMV_M_MAX_NK = 1_200_000  # P4: small-M gemv beats mma only for tiny shapes (Qwen q/o, k/v) at M = 2..3
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

    def __call__(self, x, k, use_draft=False, rotated=False):
        if self.signs is not None and not rotated:
            x = block_fwht_signed(x, self.signs, self.hb)
        cb, sc = self.draft[k] if (use_draft and k in self.draft) else (self.cb, self.sc)
        M = x.shape[0]
        if M == 1 and self.kcodes is not None:
            y = nest_gemv(x, self.kcodes, cb, sc, k)
        elif M <= 3 and self.kcodes is not None and self.N * self.K <= self.GEMV_M_MAX_NK:
            y = nest_gemv_m(x, self.kcodes, cb, sc, k)
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


def _attn_bmm_f(qh, kc, vc, mask, scale: float):
    """qh [Hkv, G*M, D], kc/vc [Hkv, T, D] (bf16), mask [G*M, T] bool -> [Hkv, G*M, D]; fp32 scores/softmax/PV."""
    s = torch.bmm(qh.float(), kc.float().transpose(1, 2)) * scale
    s = s.masked_fill(~mask, float("-inf"))
    return torch.bmm(torch.softmax(s, -1), vc.float()).to(qh.dtype)


def _attn_f(qh, kc, vc, mask, scale: float):
    """Same math as _attn_bmm_f written as broadcast-reductions so inductor emits fused reduction kernels instead of
    cuBLAS fp32 bmm (which switches to a 128x128 sgemm tile at G*M >= 20 rows: +1 ms per Llama verify, E4 L1)."""
    s = (qh.float()[:, :, None, :] * kc.float()[:, None, :, :]).sum(-1) * scale
    s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, -1)[..., None] * vc.float()[:, None]).sum(-2).to(qh.dtype)


_FUSED = {"rms": _rms_f, "rope": _rope_f, "act": _act_f, "add_rms": _add_rms_f, "attn": _attn_f}
_COMPILED = {}


def fused(name, use_compile):
    if not use_compile:
        return _FUSED[name]
    if name not in _COMPILED:
        _COMPILED[name] = torch.compile(_FUSED[name], dynamic=False, fullgraph=True)
    return _COMPILED[name]


class GenLinear:
    """General nested-VQ linear on the nestg kernels: group dim d in {8,16}, per-plane code bits (4/8/12/16, 8/12 may
    be mixed), L <= 10, codebook values q in {16 (bf16), 8 (int8 + scale), 4 (int4 + scale)}. Same call interface as
    NestLinear. Dispatch: M == 1 -> nestg GEMV (M1_PATH = "gemv") or MMA ("mma"); M <= 16 -> nestg MMA; larger M in
    chunks of 16. Built synthetic (random codes/codebooks/scales/signs) for timing."""
    M1_PATH = "auto"  # "gemv" | "mma" | "auto": time both once per prefix k (outside graph capture) and keep the faster
    SMEM_MAX = 96 * 1024

    def __init__(self, gv, signs=None, hb=1, bias=None):
        self.g, self.N, self.K, self.L = gv, gv.N, gv.K, gv.L
        self.signs, self.hb, self.bias, self.draft = signs, hb, bias, {}
        self.m1 = {}  # k -> chosen M=1 path

    def _m1_path(self, x, k):
        if self.M1_PATH != "auto":
            return self.M1_PATH
        if k not in self.m1:
            if torch.cuda.is_current_stream_capturing():
                return "gemv"
            ts = {}
            for path in ("gemv", "mma"):
                f = (lambda: self.g.gemv(x, k)) if path == "gemv" else (lambda: self.g.mma(x, k))
                for _ in range(3):
                    f()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(20):
                    f()
                e.record()
                torch.cuda.synchronize()
                ts[path] = s.elapsed_time(e)
            self.m1[k] = min(ts, key=ts.get)
        return self.m1[k]

    @classmethod
    def synthetic(cls, N, K, device, dtype, d, bits, q, hadamard_block=256, bias=False, gen=None):
        from specrq.nestkern.ops import GenVQ
        L = len(bits)
        codes = torch.stack([torch.randint(0, 2 ** b, (N, K // d), device=device, generator=gen, dtype=torch.int32)
                             for b in bits])
        cbs = [torch.randn(2 ** b, d, device=device, generator=gen) * (3.0 / (K * L) ** 0.5) for b in bits]
        gv = GenVQ(codes, cbs, torch.rand(N, device=device, generator=gen) + 0.5, list(bits), d, q, smem_max=cls.SMEM_MAX)
        gv.cb_deq = None  # reference values not needed for timing
        del codes
        hb = hadamard_block if (hadamard_block > 1 and K % hadamard_block == 0) else 1
        signs = (torch.randint(0, 2, (K,), device=device, generator=gen).float() * 2 - 1) if hb > 1 else None
        b = (torch.randn(N, device=device, generator=gen) * 0.01).to(dtype) if bias else None
        return cls(gv, signs, hb, b)

    def __call__(self, x, k, use_draft=False, rotated=False):
        if self.signs is not None and not rotated:
            x = block_fwht_signed(x, self.signs, self.hb)
        M = x.shape[0]
        if M == 1 and self._m1_path(x, k) == "gemv":
            y = self.g.gemv(x, k)[0]
        elif M <= 16:
            y = self.g.mma(x, k)[0]
        else:
            y = torch.cat([self.g.mma(x[i:i + 16], k)[0] for i in range(0, M, 16)])
        return y if self.bias is None else y + self.bias

    def nbytes(self, k):
        return self.g.nbytes(k)

    def plan(self, path, k, M=1):
        x = torch.zeros(M, self.K, device=self.g.dev, dtype=torch.bfloat16)
        return (self.g.gemv(x, k, dry_run=True) if path == "gemv" else self.g.mma(x, k, dry_run=True))[1]


class MarlinLinear:
    """Random Marlin FP16 x INT4 (g128) linear for timing baselines (fp16 only, no Hadamard). Random int32 words are
    valid packed int4 weights in Marlin layout, so no packing is needed."""

    def __init__(self, N, K, device, gen=None):
        import marlin  # noqa: F401
        assert N % 128 == 0 and K % 128 == 0
        self.N, self.K, self.L, self.hb, self.signs, self.draft, self.bias = N, K, 1, 1, None, {}, None
        self.B = torch.randint(-2 ** 31, 2 ** 31 - 1, (K // 16, N * 16 // 8), device=device, dtype=torch.int32,
                               generator=gen)
        self.s = (torch.rand(K // 128, N, device=device, generator=gen) * 2e-3 + 1e-3).half()
        self.ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device=device)

    def __call__(self, x, k=None, use_draft=False, rotated=False):
        import marlin
        y = torch.empty(x.shape[0], self.N, dtype=x.dtype, device=x.device)
        marlin.mul(x, self.B, y, self.s, self.ws)
        return y

    def nbytes(self, k=None):
        return self.N * self.K // 2 + self.s.numel() * 2


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
        self.fused_glue = True  # nestkern glue kernels (add+RMSNorm+FWHT, SiLU*mul+FWHT) on decode-sized calls
        self._rope_len = 0
        self.ckpt_dir, self.model_path = ckpt_dir, model_path

    @classmethod
    def synthetic(cls, model_path, device="cuda", dtype=torch.bfloat16, L=5, hadamard=True, seed=0, verbose=True,
                  linear_impl="nest", vq=None):
        """Real embed / norms / lm_head from the HF safetensors of `model_path` (only those tensors are read) and
        random C1 linears (codes, codebooks, scales, Hadamard signs) generated on the GPU. For timing only.
        vq = {"d": 8|16, "bits": int or per-plane list, "q": 16|8|4} selects the general nestg linears (GenLinear)
        instead of the specialized C1 kernels (linear_impl="nest") or Marlin INT4 (linear_impl="marlin")."""
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
                if linear_impl == "marlin":
                    d[n] = MarlinLinear(N, K, self.device, gen=gen)
                elif vq is not None:
                    bits = vq["bits"] if isinstance(vq["bits"], (list, tuple)) else [vq["bits"]] * vq["L"]
                    d[n] = GenLinear.synthetic(N, K, self.device, dtype, vq["d"], bits, vq["q"],
                                               256 if hadamard else 1, bias=bias and n in ("q_proj", "k_proj", "v_proj"),
                                               gen=gen)
                else:
                    d[n] = NestLinear.synthetic(N, K, self.device, dtype, L, 256 if hadamard else 1,
                                                bias=bias and n in ("q_proj", "k_proj", "v_proj"), gen=gen)
            self.layers.append(d)
            if verbose and i % 8 == 0:
                print(f"[infer] synthetic layer {i}/{self.n_layers}", flush=True)
        self.L = (self.layers[0]["q_proj"].L if vq is not None else L) if linear_impl == "nest" else 1
        self.inv_freq, self.rope_scaling = _rope_inv_freq(cfg, self.device)
        self.use_compile, self.decode_attn, self._rope_len, self.fused_glue = True, "fp32", 0, True
        self.vq = vq
        self.ckpt_dir = (f"synthetic:nestg {vq} hadamard={hadamard} seed={seed}" if vq is not None else
                         f"synthetic:C1 L={L} hadamard={hadamard} seed={seed}" if linear_impl == "nest"
                         else f"synthetic:Marlin INT4 g128 seed={seed}")
        self.model_path = model_path
        return self

    def _setup_fused(self):
        """Per-layer packed sign groups for the fused glue: (C, signs [S, K]) for q/k/v and gate/up, C for down.
        A group is fused only if all its linears share the block size (or none is rotated)."""
        def group(lins):
            hbs = {l.hb if l.signs is not None else 1 for l in lins}
            if len(hbs) != 1:
                return None
            hb = hbs.pop()
            if hb == 1:
                return (128, None)
            return (hb, torch.stack([l.signs for l in lins]).contiguous())
        for lay in self.layers:
            lay["g_qkv"] = group([lay["q_proj"], lay["k_proj"], lay["v_proj"]])
            lay["g_gu"] = group([lay["gate_proj"], lay["up_proj"]])
            dn = lay["down_proj"]
            lay["g_down"] = (dn.hb, dn.signs) if dn.signs is not None else (128, None)
        hid = self.layers[0]["q_proj"].K
        self.fused_ok = all(l["g_qkv"] and l["g_gu"] for l in self.layers) and hid % 128 == 0 and \
            all(l["down_proj"].K % l["g_down"][0] == 0 for l in self.layers)

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

    # ------------------------------------------------------------------ FR-Spec draft vocab
    def set_draft_vocab(self, ids):
        """ids: int64 [V_d] token ids (specrq.infer.frspec.load_subset). Gathers the lm_head rows into a contiguous
        buffer. Returns V_d (the key for forward(vd=...))."""
        if not hasattr(self, "dvocab"):
            self.dvocab = {}
        ids = ids.to(self.device, torch.long).contiguous()
        self.dvocab[ids.numel()] = (ids, self.lm_head.index_select(0, ids).contiguous())
        return ids.numel()

    def draft_vocab_bytes(self):
        return {vd: w.numel() * w.element_size() + ids.numel() * 8 for vd, (ids, w) in getattr(self, "dvocab", {}).items()}

    def to_token_ids(self, idx, vd=None):
        """Map argmax indices of forward(vd=...) logits back to token ids (device gather; graph-capturable)."""
        return idx if vd is None else self.dvocab[vd][0][idx]

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

    def forward(self, tokens, pos, cache: Cache, k=None, use_draft=False, last_only=False, capture=None, vd=None):
        """tokens [M] long (device); pos: 0-d long device tensor (position of tokens[0]); returns logits [M or 1, V]
        (vd=None) or [M or 1, V_d] over the FR-Spec draft vocab (map argmax with to_token_ids)."""
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
        fg = self.fused_glue and uc
        if fg:
            if not hasattr(self, "fused_ok"):
                self._setup_fused()
            fg = self.fused_ok
        if fg:
            C, sg = self.layers[0]["g_qkv"]
            _, xs = add_rms_rot(h, None, self.layers[0]["in_norm"], sg, C, self.eps)
        else:
            x = rms(h, self.layers[0]["in_norm"], self.eps)
        for li, lay in enumerate(self.layers):
            if fg:
                rq = lay["g_qkv"][1] is not None
                xq, xk, xv = (xs[0], xs[1], xs[2]) if rq else (xs, xs, xs)
                q = lay["q_proj"](xq, k, use_draft, rotated=True).view(M, self.H, self.D)
                kk = lay["k_proj"](xk, k, use_draft, rotated=True).view(M, self.Hkv, self.D)
                v = lay["v_proj"](xv, k, use_draft, rotated=True).view(M, self.Hkv, self.D)
            else:
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
            nxt = self.layers[li + 1]["in_norm"] if li + 1 < self.n_layers else self.norm
            if fg:
                C, sg = lay["g_gu"]
                h, xs = add_rms_rot(h, lay["o_proj"](o, k, use_draft), lay["post_norm"], sg, C, self.eps)
                xg, xu = (xs[0], xs[1]) if sg is not None else (xs, xs)
                Cd, sd = lay["g_down"]
                a = act_rot(lay["gate_proj"](xg, k, use_draft, rotated=True),
                            lay["up_proj"](xu, k, use_draft, rotated=True), sd, Cd)
                dn = lay["down_proj"](a, k, use_draft, rotated=True)
                if li + 1 < self.n_layers:
                    C, sg = self.layers[li + 1]["g_qkv"]
                    h, xs = add_rms_rot(h, dn, nxt, sg, C, self.eps)
                else:
                    h, x = add_rms_rot(h, dn, nxt, None, 128, self.eps)
            else:
                h, x = add_rms(h, lay["o_proj"](o, k, use_draft), lay["post_norm"], self.eps)
                a = act(lay["gate_proj"](x, k, use_draft), lay["up_proj"](x, k, use_draft))
                h, x = add_rms(h, lay["down_proj"](a, k, use_draft), nxt, self.eps)
            if capture is not None:
                capture.append(h)
        if last_only:
            x = x[-1:]
        return F.linear(x, self.lm_head if vd is None else self.dvocab[vd][1])

    def prefill(self, tokens, cache, k=None, use_draft=False, vd=None):
        """Process the prompt (any length, chunks of 16 inside the linears); returns last-position logits [1, V]
        ([1, V_d] with vd)."""
        pos = torch.zeros((), dtype=torch.long, device=self.device)
        return self.forward(tokens, pos, cache, k, use_draft, last_only=True, vd=vd)


def load_draft_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)
