"""E7 magicvq v13: v12_w4.py (draft-weight sources '@<src>') with a bounded pool of GPU shadow layers.

v12 keeps one shadow decoder layer (~0.44 GB bf16) per weight source for the whole run; with 5-6 sources that does not
fit next to the 32k teacher-forced pass on a 24 GB 4090 (v12 peak 22.6 GiB with one source). Here the shadow layers
form an LRU pool of V13_NSHADOW layers (default 1) keyed (source, layer); a miss reloads the layer's 7 linears from the
mmapped source file exactly as v12 does (same tensors, same bf16 cast, target norms). Only the cache policy changes,
so a draft's outputs are bit-identical to v12 (anchor: snapkv4096@w4 must reproduce the v12 record per sample).
Order drafts by source in --drafts: tau_eval.tf_pass runs drafts in that order inside each layer, so reloads per layer
= number of distinct sources.
Usage: V12_WSRC="w4=...,k3=..." V12_TAGS=... python experiments/26_magicvq/v13_tau.py {lb_tau.py|tau_eval.py} <args>
"""
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v12_w4 as W  # noqa: E402

NMAX = int(os.environ.get("V13_NSHADOW", "1"))
POOL = []  # [[layer module, (src, l)]], most recently used last
STATS = {"hit": 0, "miss": 0}


def _draft_layer(l, layer, spec):
    src = spec.get("wsrc")
    if src is None:
        return layer
    for i, e in enumerate(POOL):
        if e[1] == (src, l):
            POOL.append(POOL.pop(i))
            STATS["hit"] += 1
            return e[0]
    STATS["miss"] += 1
    if len(POOL) < NMAX:
        POOL.append([copy.deepcopy(layer), None])
    else:
        POOL.append(POOL.pop(0))
    e = POOL[-1]
    lw = e[0]
    with torch.no_grad():  # = v12_w4._draft_layer's load
        lw.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        lw.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight)
        for lin, sub in W.LIN.items():
            dst = getattr(getattr(lw, sub), lin).weight
            if W.SOURCES[src] == "target":
                w = getattr(getattr(layer, sub), lin).weight
            else:
                w = W.SOURCES[src].get(f"model.layers.{l}.{sub}.{lin}")
            assert w.shape == dst.shape, (l, lin, w.shape, dst.shape)
            dst.copy_(w.to(dst.device).to(dst.dtype))
            assert getattr(getattr(layer, sub), lin).bias is None, "bias not handled"
    e[1] = (src, l)
    return lw


if __name__ == "__main__":
    W._draft_layer = _draft_layer  # v12_w4.patch() injects the module-global _draft_layer into tau_eval
    W.main()
    print(f"[v13] shadow pool {NMAX}: {STATS}", flush=True)
