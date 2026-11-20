"""E7 v14: INT4 (Marlin) draft weights for the TP=4 real e2e (e2e_real.py --draft_w4 DIR).

QuantSpec-style: the draft's linears (qkv / o / gate_up / down) run as Marlin FP16 x INT4 (GPTQ g128, E4 checkpoint),
sharded exactly like the target TP (column-parallel qkv / gate_up over output columns, row-parallel o / down over
input rows). Norms, embedding, lm_head, attention kernels and all-reduces are the target's (bf16). The verify pass is
the unchanged bf16 forward. The draft's in-round K/V come from the INT4 qkv (written into the target cache at the
draft positions and overwritten by the verify, as for the bf16 draft) = v12 semantics.

Sharding is done directly on the packed Marlin tensors (no repacking): in Marlin's layout B [K/16, 2N] a column
slice [n0, n1) with n0, n1 multiples of 64 is the packed column slice [2 n0, 2 n1) (the weight permutation acts within
1024-element chunks = 4 n-tiles of 16, the scale permutation within 64 consecutive columns of a group row), and a row
slice [k0, k1) with k0, k1 multiples of 128 (group size) is B rows [k0/16, k1/16) and s rows [k0/128, k1/128).
Checked bit-exactly against the dense dequant in e2e_w4_test.py.

Marlin takes fp16 activations: the draft casts the bf16 normed hidden state to fp16 before each linear and the fp16
output back to bf16 (residual stream / all-reduce / attention stay bf16 as in the target code).
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

MARLIN_SRC = "/data/lsh/src/marlin"


def _marlin():
    if MARLIN_SRC not in sys.path:
        sys.path.insert(0, MARLIN_SRC)
    import marlin  # noqa: F401  (env nestvq: /data/lsh/src/marlin, built for sm_89)
    return marlin


class MLin:
    """one Marlin linear shard: B int32 [K/16, 2N], s fp16 [K/128, N], private zeroed workspace."""

    def __init__(self, B, s):
        self.B = B.cuda().contiguous()
        self.s = s.cuda().contiguous()
        self.K, self.N = self.B.shape[0] * 16, self.s.shape[1]
        assert self.B.shape[1] == 2 * self.N and self.s.shape[0] * 128 == self.K
        self.ws = torch.zeros(self.N // 128 * 16, dtype=torch.int32, device="cuda")

    def __call__(self, x):
        mar = sys.modules["marlin"]
        a = x.to(torch.float16).contiguous()
        c = torch.empty(a.shape[0], self.N, dtype=torch.float16, device=a.device)
        mar.mul(a, self.B, c, self.s, self.ws, -1, -1, -1, 16)
        return c.to(torch.bfloat16)


def _cols(pk, name, n0, n1):
    assert n0 % 64 == 0 and n1 % 64 == 0
    return pk[name + ".B"][:, 2 * n0:2 * n1], pk[name + ".s"][:, n0:n1]


def _rows(pk, name, k0, k1):
    assert k0 % 128 == 0 and k1 % 128 == 0
    return pk[name + ".B"][k0 // 16:k1 // 16], pk[name + ".s"][k0 // 128:k1 // 128]


class W4:
    """per-rank Marlin INT4 shards of all 32 layers (same split as e2e_real.Shard)."""

    def __init__(self, ckpt_dir, rank, world, NL=32, D=128, HQA=32, HKVA=8, INTER=14336):
        _marlin()
        self.dir = ckpt_dir
        self.meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
        pk = torch.load(os.path.join(ckpt_dir, "marlin_linears.pt"), map_location="cpu", mmap=True, weights_only=True)
        hq, hkv, inter = HQA // world, HKVA // world, INTER // world
        r = rank
        self.layers = []
        nbytes = 0
        for l in range(NL):
            p = f"model.layers.{l}."
            qkv = [_cols(pk, p + "self_attn.q_proj", r * hq * D, (r + 1) * hq * D),
                   _cols(pk, p + "self_attn.k_proj", r * hkv * D, (r + 1) * hkv * D),
                   _cols(pk, p + "self_attn.v_proj", r * hkv * D, (r + 1) * hkv * D)]
            gu = [_cols(pk, p + "mlp.gate_proj", r * inter, (r + 1) * inter),
                  _cols(pk, p + "mlp.up_proj", r * inter, (r + 1) * inter)]
            L = {"qkv": MLin(torch.cat([x[0] for x in qkv], 1), torch.cat([x[1] for x in qkv], 1)),
                 "o": MLin(*_rows(pk, p + "self_attn.o_proj", r * hq * D, (r + 1) * hq * D)),
                 "gu": MLin(torch.cat([x[0] for x in gu], 1), torch.cat([x[1] for x in gu], 1)),
                 "down": MLin(*_rows(pk, p + "mlp.down_proj", r * inter, (r + 1) * inter))}
            nbytes += sum(x.B.numel() * 4 + x.s.numel() * 2 + x.ws.numel() * 4 for x in L.values())
            self.layers.append(L)
        self.gib = nbytes / 2 ** 30
        del pk

    def info(self):
        return {"dir": self.dir, "format": self.meta.get("format"), "groupsize": self.meta.get("groupsize"),
                "gib_per_rank": self.gib, "act_dtype": "fp16 (bf16 hidden cast per linear)",
                "linears": "qkv, o, gate_up, down (Marlin mul, thread 128x128 auto for M<=16, max_par 16)"}


def make_forward(E, w4):
    """draft forward with the same signature / structure as e2e_real.forward; E = the e2e_real module."""

    def layer_post(m, L, W, h, o):
        h = h + E.allreduce(W["o"](o))
        gu = W["gu"](E.rms(h, L["n2"]))
        return h + E.allreduce(W["down"](F.silu(gu[:, :m.inter]) * gu[:, m.inter:]))

    def forward(m, tokens, attn):
        h = E.embed(m, tokens)
        for l, (L, W) in enumerate(zip(m.layers, w4.layers)):
            qkv = W["qkv"](E.rms(h, L["n1"]))
            h = layer_post(m, L, W, h, attn(qkv, l))
        return E.gargmax(m, h)
    return forward
