"""
Codeword storage formats (fake-quant: quantize, then dequantize back to fp32).

Applied to unit-norm GSRQ shape codewords, one scale per codeword (absmax mapped to the format's max).
Storage cost per codeword element = format bits; the per-codeword fp16 scale adds 16/D bits per element.

  fp16, bf16        : cast
  int8, int4, int3  : symmetric uniform, levels ±(2^(b-1)-1)
  fp8               : e4m3fn
  fp4               : e2m1  {0, .5, 1, 1.5, 2, 3, 4, 6}
  nf4               : NormalFloat4 levels (QLoRA)
"""

import torch

_FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_FP4 = torch.cat([-_FP4.flip(0)[:-1], _FP4])
_NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                     -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                     0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                     0.7229568362236023, 1.0])

BITS = {"fp16": 16, "bf16": 16, "int8": 8, "fp8": 8, "int4": 4, "fp4": 4, "nf4": 4, "int3": 3}


def _grid(x, levels):
    """Round x (already scaled into the grid's range) to the nearest level (levels sorted ascending)."""
    lv = levels.to(x.device, x.dtype)
    return lv[torch.bucketize(x, (lv[1:] + lv[:-1]) / 2)]


def quantize(x, fmt):
    """x: (..., D) codewords -> same shape, values representable in `fmt` (per-codeword scale)."""
    if fmt in (None, "fp32"):
        return x
    if fmt == "fp16":
        return x.half().float()
    if fmt == "bf16":
        return x.bfloat16().float()
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    if fmt.startswith("int"):
        q = 2 ** (int(fmt[3:]) - 1) - 1
        s = amax / q
        return (x / s).round().clamp(-q, q) * s
    if fmt == "fp8":
        s = amax / 448.0
        return (x / s).to(torch.float8_e4m3fn).float() * s
    if fmt == "fp4":
        s = amax / 6.0
        return _grid(x / s, _FP4) * s
    if fmt == "nf4":
        return _grid(x / amax, _NF4) * amax
    raise ValueError(f"unknown format {fmt}")
