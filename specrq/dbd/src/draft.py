"""[specrq] Draft-only codebooks for nested additive VQ (Phase 5B).

The target uses all M codebooks with the shared codes, target codebooks and target scales (frozen).
The draft uses the first k_d codes (shared index, frozen) but its OWN copy of the first k_d codebooks and of the
per-row scales; only these copies are trained, to make p_D^(k_d) close to p_T^(M) (TV / KL / JSD).
Extra storage per linear: k_d * K * g * 16 bit (codebooks) + d_out * 16 bit (scales) -- no extra index bits.

QuantizedLinear._forward uses the draft copies when active_num_codebooks == draft_k (see src/aq.py).
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn

from src.aq import QuantizedLinear


def attach_draft_codebooks(model: nn.Module, draft_k: int, train_scales: bool = True) -> int:
    """Add draft copies (init = target values) to every QuantizedLinear. Returns number of trainable draft params."""
    n = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            qw = m.quantized_weight
            cb = qw.get_codebooks()[:draft_k].detach().clone()
            sc = qw.get_scales().detach().clone()
            m.draft_codebooks = nn.Parameter(cb.float(), requires_grad=True)
            m.draft_scales = nn.Parameter(sc.float(), requires_grad=train_scales)
            m.draft_k = int(draft_k)
            n += cb.numel() + (sc.numel() if train_scales else 0)
    return n


def draft_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for m in model.modules():
        if isinstance(m, QuantizedLinear) and getattr(m, "draft_codebooks", None) is not None:
            yield m.draft_codebooks
            if m.draft_scales.requires_grad:
                yield m.draft_scales


def draft_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, QuantizedLinear) and getattr(m, "draft_codebooks", None) is not None:
            out[f"{name}.draft_codebooks"] = m.draft_codebooks.detach().cpu()
            out[f"{name}.draft_scales"] = m.draft_scales.detach().cpu()
            out[f"{name}.draft_k"] = torch.tensor(m.draft_k)
    return out


def load_draft_state_dict(model: nn.Module, sd: Dict[str, torch.Tensor], trainable: bool = False,
                          train_scales: bool = True) -> int:
    """Install draft copies from a state dict (e.g. trained, or rounded to q=8/4 by quantize_draft.py)."""
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, QuantizedLinear) and f"{name}.draft_codebooks" in sd:
            dev = m.quantized_weight.codebooks.device
            m.draft_codebooks = nn.Parameter(sd[f"{name}.draft_codebooks"].to(dev).float(), requires_grad=trainable)
            m.draft_scales = nn.Parameter(sd[f"{name}.draft_scales"].to(dev).float(), requires_grad=trainable and train_scales)
            m.draft_k = int(sd[f"{name}.draft_k"])
            n += 1
    return n


def divergence(lt: torch.Tensor, ld: torch.Tensor, kind: str = "tv") -> torch.Tensor:
    """Per-token divergence between teacher logits lt (target) and student logits ld (draft); returns [N].
    tv: 0.5 * |p_T - p_D|_1 (fixed training objective; 1 - TV = speculative-sampling acceptance).
    fkl: KL(p_T || p_D), fallback only."""
    lpt, lpd = lt.float().log_softmax(-1), ld.float().log_softmax(-1)
    pt, pd = lpt.exp(), lpd.exp()
    if kind == "tv":
        return 0.5 * (pt - pd).abs().sum(-1)
    if kind == "fkl":
        return (pt * (lpt - lpd)).sum(-1)
    raise ValueError(kind)


# ---------------- q-bit draft codebooks (Phase 5D) ----------------
QBITS = {"bf16": 16, "int8": 8, "int4": 4, "fp8_e4m3": 8,
         # 16-level (or fewer) LUT formats, value = scale * level (rounding-only comparison, experiments/11_qcmp/run_fmt4.sh)
         "int3": 3, "nf4": 4, "nf3": 3, "fp4_e2m1": 4, "posit4_es0": 4, "posit4_es1": 4}


def _normal_float_levels(bits: int, offset: float = 0.9677083) -> torch.Tensor:
    """QLoRA NormalFloat levels (bitsandbytes create_normal_map, use_extra_value=True): 2^(b-1) positive quantiles,
    2^(b-1)-1 negative quantiles, plus an exact 0; normalized to max |level| = 1. bits=4 reproduces the NF4 table."""
    from scipy.stats import norm
    import numpy as np
    pos = norm.ppf(np.linspace(offset, 0.5, 2 ** (bits - 1) + 1)[:-1]).tolist()
    neg = (-norm.ppf(np.linspace(offset, 0.5, 2 ** (bits - 1))[:-1])).tolist()
    v = torch.tensor(sorted(pos + [0.0] + neg), dtype=torch.float64)
    return (v / v.abs().max()).float()


def _posit_levels(n: int, es: int) -> torch.Tensor:
    """All finite values of posit<n, es> (NaR excluded): 2^n - 1 levels incl. 0, symmetric."""
    vals = [0.0]
    for b in range(1, 2 ** (n - 1)):  # positive patterns 0 b_{n-2} .. b_0
        bits = [(b >> i) & 1 for i in range(n - 2, -1, -1)]
        r0, m = bits[0], 0
        while m < len(bits) and bits[m] == r0:
            m += 1
        k = m - 1 if r0 == 1 else -m
        rest = bits[m + 1:]
        e_bits = rest[:es] + [0] * (es - len(rest[:es]))
        e = int("".join(map(str, e_bits)), 2) if es else 0
        f_bits = rest[es:]
        f = sum(bb * 2.0 ** -(i + 1) for i, bb in enumerate(f_bits))
        vals.append((2.0 ** (2 ** es)) ** k * 2.0 ** e * (1 + f))
    vals = sorted(set(vals))
    return torch.tensor(sorted([-v for v in vals if v > 0] + vals), dtype=torch.float32)


_LUT = {}


def lut_levels(fmt: str) -> torch.Tensor:
    if fmt not in _LUT:
        if fmt == "nf4":
            _LUT[fmt] = _normal_float_levels(4)
        elif fmt == "nf3":
            _LUT[fmt] = _normal_float_levels(3)
        elif fmt == "fp4_e2m1":
            p = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            _LUT[fmt] = torch.cat([-p.flip(0)[:-1], p])
        elif fmt.startswith("posit4_es"):
            _LUT[fmt] = _posit_levels(4, int(fmt[-1]))
        else:
            raise ValueError(fmt)
    return _LUT[fmt]


_LUT_FMTS = ("nf4", "nf3", "fp4_e2m1", "posit4_es0", "posit4_es1")


def round_fmt(x: torch.Tensor, scale: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt in _LUT_FMTS:  # nearest level of the (sorted) level set; |x/scale| beyond the extremes clips
        lv = lut_levels(fmt).to(x.device, x.dtype)
        return lv[torch.bucketize(x / scale, (lv[1:] + lv[:-1]) / 2)] * scale
    if fmt in ("int8", "int4", "int3"):
        qmax = 2 ** (QBITS[fmt] - 1) - 1
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    if fmt == "fp8_e4m3":
        y = torch.clamp(x / scale, -448.0, 448.0)
        return y.to(torch.float8_e4m3fn).to(x.dtype) * scale
    if fmt == "bf16":
        return x.to(torch.bfloat16).to(x.dtype)
    raise ValueError(fmt)


def best_scale(x: torch.Tensor, fmt: str, dims) -> torch.Tensor:
    """Per-group scale (reduced over dims) minimizing MSE on a grid; scales are stored fp16.
    INT: clip ratio in [0.25, 1] x max|v|/qmax. FP8 e4m3: 2^-12 .. 2^4 x max|v|/448 (spans the format's dynamic range,
    CLAUDE.md gotcha: its dense region sits far below max, so the search must not be limited to max|v| -> 448)."""
    amax = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-12)
    if fmt in ("int8", "int4", "int3"):
        qmax = 2 ** (QBITS[fmt] - 1) - 1
        cands = [amax * r / qmax for r in torch.linspace(0.25, 1.0, 61, device=x.device)]
        best = amax / qmax
    elif fmt == "fp8_e4m3":
        cands = [amax / 448.0 * r for r in 2.0 ** torch.arange(-12, 4.25, 0.25, device=x.device)]
        best = amax / 448.0
    elif fmt in _LUT_FMTS:
        # LUT formats: max|v| may map anywhere from 2^-6 to 2^0.5 x the largest level (log grid, 32 steps/octave).
        # Spans the whole dynamic range of wide formats (posit4_es1: 1/16..16, fp4: 0.5..6) instead of a fixed
        # [0.25, 1] clip window (CLAUDE.md gotcha); r > 1 leaves headroom (only useful for asymmetric sets, e.g. NF).
        lmax = lut_levels(fmt).abs().max().item()
        cands = [amax / lmax * r for r in 2.0 ** torch.arange(-6, 0.5 + 1e-9, 1 / 32, device=x.device)]
        best = amax / lmax
    else:
        raise ValueError(fmt)
    best = best.half().float()  # max-abs fallback
    best_err = torch.full_like(amax, float("inf"))
    for sc in cands:
        sc = sc.half().float()  # scales are stored fp16; tiny candidates underflow to 0 -> invalid
        err = ((round_fmt(x, sc.clamp_min(1e-30), fmt) - x) ** 2).sum(dim=dims, keepdim=True)
        err = torch.where((sc > 0) & torch.isfinite(err), err, torch.full_like(err, float("inf")))
        m = err < best_err
        best, best_err = torch.where(m, sc, best), torch.where(m, err, best_err)
    return best


GRAN_DIMS = {"codebook": (1, 2, 3), "dim": (1, 2)}  # draft_codebooks: [k, K, out_group, in_group]


class STEFakeQuant:
    """Straight-through fake quantization of the draft codebooks (forward: rounded, backward: identity).
    The scales are refreshed with best_scale() by refresh(); they stay fixed in between."""

    def __init__(self, fmt: str, gran: str):
        self.fmt, self.dims, self.scale = fmt, GRAN_DIMS[gran], None

    def refresh(self, cb: torch.Tensor):
        with torch.no_grad():
            self.scale = None if self.fmt == "bf16" else best_scale(cb.detach().float(), self.fmt, self.dims)

    def __call__(self, cb: torch.Tensor) -> torch.Tensor:
        if self.fmt == "bf16":
            q = round_fmt(cb.detach().float(), None, "bf16")
        else:
            if self.scale is None:
                self.refresh(cb)
            q = round_fmt(cb.detach().float(), self.scale, self.fmt)
        return cb + (q.to(cb.dtype) - cb).detach()


def attach_fakequant(model: nn.Module, fmt: str, gran: str) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear) and getattr(m, "draft_codebooks", None) is not None:
            m.draft_codebook_transform = STEFakeQuant(fmt, gran)
            n += 1
    return n


def refresh_fakequant(model: nn.Module):
    for m in model.modules():
        t = getattr(m, "draft_codebook_transform", None)
        if t is not None:
            t.refresh(m.draft_codebooks)


@torch.no_grad()
def bake_fakequant(model: nn.Module):
    """Replace draft codebooks by their rounded values (for saving / evaluation) and drop the transform."""
    for m in model.modules():
        t = getattr(m, "draft_codebook_transform", None)
        if t is not None:
            m.draft_codebooks.data.copy_(t(m.draft_codebooks).float())
            m.draft_codebook_transform = None
