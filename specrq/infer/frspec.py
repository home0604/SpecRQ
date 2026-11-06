"""FR-Spec draft vocabulary (arXiv 2502.14856): the draft computes lm_head rows only for the V_d most frequent tokens
and takes its argmax inside that subset; target / verify keep the full bf16 lm_head, so greedy output is unchanged.

Frequency tables come from experiments/08_frspec/token_freq.py (pile-val, per tokenizer):
  /data/lsh/frspec/llama31_8b_instruct_pileval, /data/lsh/frspec/qwen25_05b_instruct_pileval
"""
from __future__ import annotations

import json
import os

import torch

FREQ_DIRS = {"llama": "/data/lsh/frspec/llama31_8b_instruct_pileval",
             "qwen2": "/data/lsh/frspec/qwen25_05b_instruct_pileval"}


def default_freq_dir(cfg):
    return FREQ_DIRS.get(cfg.model_type)


def load_subset(freq_dir, V_d, V_model=None):
    """Token ids of the draft vocab, V_d in total: the forced stop tokens (meta.json `forced`, always included) followed
    by the most frequent other ids in frequency order. Returns int64 [V_d] (CPU)."""
    order = torch.load(os.path.join(freq_dir, "order.pt"), weights_only=True)
    meta = json.load(open(os.path.join(freq_dir, "meta.json")))
    if V_model is not None:
        assert order.numel() == V_model, f"frequency table has {order.numel()} ids, model vocab {V_model}"
    assert 0 < V_d <= order.numel()
    forced = [int(i) for i in meta.get("forced", [])]
    fs = set(forced)
    rest = order[~torch.isin(order, torch.tensor(forced, dtype=torch.long))] if forced else order
    ids = torch.cat([torch.tensor(forced, dtype=torch.long), rest[:V_d - len(forced)]])
    assert ids.numel() == V_d and ids.unique().numel() == V_d and fs <= set(ids.tolist())
    return ids


def coverage(freq_dir, ids):
    """Fraction of pile-val token occurrences covered by `ids`."""
    counts = torch.load(os.path.join(freq_dir, "counts.pt"), weights_only=True)
    return float(counts[ids].sum() / counts.sum())
