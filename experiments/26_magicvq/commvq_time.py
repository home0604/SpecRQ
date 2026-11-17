"""E7 magicvq: timing harness for the OFFICIAL CommVQ 1-bit decode kernels (commvq/triton_kernels.py @ d1f0377:
calculate_QC_T_triton, gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum, attn_weights_mul_value_triton) in the
call sequence of commvq/modeling_llama_triton.py (decode branch: q x commutative centers -> score gather over residual
steps with per-token RoPE tables -> concat with the bf16 residual-window scores -> softmax -> value bit-plane kernel ->
codebook matmul -> + residual-window values), plus the official encoders for the new token (KeyCompressor.encode:
cdist residual search over 4096 codewords x R steps; ValueCompressor.encode: LayerNorm + 2-layer MLP + quantize).
Random codes / synthetic centers of the real shapes (timing only; the kernels assume one prescale per token over
the batch, so numerics for B > 1 are not meaningful). Single GPU, all 8 kv heads (the V code mixes all heads, so a
head-sharded TP split is not possible without reading the full V code on every rank).
"""
from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, "/data/lsh/envs/commvq_pylib")
sys.path.insert(0, "/data/lsh/src/CommVQ")
from commvq.triton_kernels import (attn_weights_mul_value_triton, calculate_QC_T_triton,  # noqa: E402
                                   gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum)
from commvq.compress_evaluation import ValueCompressor  # noqa: E402


class CommVQLayerState:
    def __init__(self, B, N, R=11, Wres=128, H=32, Hkv=8, D=128, gen=None, dev="cuda"):
        bf = torch.bfloat16
        self.B, self.N, self.R, self.H, self.Hkv, self.D, self.Wres = B, N, R, H, Hkv, D, Wres
        N8 = (N + 7) // 8 * 8
        self.N8 = N8
        self.centers = torch.randn(H, 64, R, 64, 2, 2, device=dev, generator=gen).to(bf)
        self.key_code = torch.randint(0, 4096, (B, Hkv, R, N8), device=dev, generator=gen).to(torch.int16)
        self.kprescale = (torch.rand(B, 1, 1, N8, device=dev, generator=gen) + 0.5).to(bf)
        self.qsin = torch.rand(N8, 64, device=dev, generator=gen).to(bf)
        self.qcos = torch.rand(N8, 64, device=dev, generator=gen).to(bf)
        self.value_code = torch.randint(0, 256, (B, N8 // 8, 1024), device=dev, generator=gen).to(torch.uint8)
        self.vprescale = (torch.rand(B, N8, 1, device=dev, generator=gen) + 0.5).to(bf)
        # = embedding.unflatten(-1, (Hkv, D)).repeat_interleave(G, 1).transpose(0, 1) as in modeling_llama_triton.py
        self.codebook = (torch.randn(1024, Hkv, D, device=dev, generator=gen) * 0.03).repeat_interleave(H // Hkv, 1).transpose(0, 1).contiguous().to(bf)
        self.learnable_scale = torch.ones(1024, device=dev, dtype=bf)
        self.res_k = torch.randn(B, Hkv, Wres, D, device=dev, generator=gen).to(bf)
        self.res_v = torch.randn(B, Wres, Hkv * D, device=dev, generator=gen).to(bf)
        self.q = torch.randn(B, H, 1, 64, 2, device=dev, generator=gen).to(bf)
        # encoders (official modules, random weights)
        self.kcb = torch.randn(R, Hkv, 4096, D, device=dev, generator=gen).to(bf)
        self.vc = ValueCompressor(quant_bits=1, feat_dim=Hkv * D).to(dev).to(bf)
        self.vc.CONST1, self.vc.CONST2 = self.vc.CONST1.to(dev), self.vc.CONST2.to(dev)
        self.newk = torch.randn(B, 1, Hkv * D, device=dev, generator=gen).to(bf)

    def attention(self):
        B, H, Hkv, D = self.B, self.H, self.Hkv, self.D
        QC_T = calculate_QC_T_triton(self.q, self.centers)
        w1 = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum(self.key_code, QC_T, self.qsin, self.qcos,
                                                                     self.kprescale, factor=math.sqrt(D))
        q2 = self.q.transpose(-2, -1).flatten(-2)  # [B, H, 1, 128]
        rk = self.res_k.repeat_interleave(H // Hkv, 1)
        w2 = torch.matmul(q2, rk.transpose(2, 3)) / math.sqrt(D)
        w = torch.cat([w1, w2], dim=3).float().softmax(-1).to(torch.bfloat16)
        o1 = attn_weights_mul_value_triton(attn_weights=w, value_code=self.value_code, prescale=self.vprescale,
                                           learnable_scale=self.learnable_scale, eps=1e-6, codebook=self.codebook)
        rv = self.res_v.view(B, -1, Hkv, D).transpose(1, 2).repeat_interleave(H // Hkv, 1)
        o2 = torch.matmul(w[..., self.N8:], rv)
        return o1 + o2

    def encode(self):
        """official encoders on the new token of every sequence (1 token x B)."""
        x = self.newk.view(self.B, 1, self.Hkv, 2, 64).transpose(3, 4).flatten(2)
        pre = torch.norm(x, dim=-1, keepdim=True)
        t = (x / pre).view(self.B, self.Hkv, self.D).permute(1, 0, 2).contiguous()  # [Hkv, B, D]
        for r in range(self.R):
            cbr = self.kcb[r]
            d = torch.cdist(t.float(), cbr.float())
            a = d.argmin(-1)
            t = t - torch.gather(cbr, 1, a[..., None].expand(-1, -1, self.D))
        self.vc.encode(self.newk)
