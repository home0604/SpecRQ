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
DEFAULT_VD = 32768  # supervisor decision 2026-11-06 (= FR-Spec paper choice for Llama-3)
# chat-template special tokens always kept in the draft vocab (supervisor decision 2026-11-06), on top of the
# generation_config stop tokens in meta.json `forced`. Header-name tokens (e.g. `assistant`) are NOT forced.
CHAT_FORCED = {"llama31_8b_instruct_pileval": {"<|start_header_id|>": 128006, "<|end_header_id|>": 128007,
                                               "<|eot_id|>": 128009},
               "qwen25_05b_instruct_pileval": {"<|im_start|>": 151644, "<|im_end|>": 151645}}


def default_freq_dir(cfg):
    return FREQ_DIRS.get(cfg.model_type)


def forced_ids(freq_dir, chat=True):
    """Ids always in the draft vocab: generation_config stop tokens (meta.json `forced`) + chat specials (CHAT_FORCED)."""
    meta = json.load(open(os.path.join(freq_dir, "meta.json")))
    f = [int(i) for i in meta.get("forced", [])]
    if chat:
        f += [i for i in CHAT_FORCED.get(os.path.basename(os.path.normpath(freq_dir)), {}).values() if i not in f]
    return f


def load_subset(freq_dir, V_d, V_model=None, chat=True):
    """Token ids of the draft vocab, exactly V_d in total: forced_ids(freq_dir, chat) followed by the most frequent
    other ids in frequency order. chat=False reproduces the stop-tokens-only subset of results/E4/frspec_qwen05_* and
    l1_..._v5_frspec. Returns int64 [V_d] (CPU)."""
    order = torch.load(os.path.join(freq_dir, "order.pt"), weights_only=True)
    if V_model is not None:
        assert order.numel() == V_model, f"frequency table has {order.numel()} ids, model vocab {V_model}"
    assert 0 < V_d <= order.numel()
    forced = forced_ids(freq_dir, chat)
    fs = set(forced)
    rest = order[~torch.isin(order, torch.tensor(forced, dtype=torch.long))] if forced else order
    ids = torch.cat([torch.tensor(forced, dtype=torch.long), rest[:V_d - len(forced)]])
    assert ids.numel() == V_d and ids.unique().numel() == V_d and fs <= set(ids.tolist())
    return ids


def coverage(freq_dir, ids):
    """Fraction of pile-val token occurrences covered by `ids`."""
    counts = torch.load(os.path.join(freq_dir, "counts.pt"), weights_only=True)
    return float(counts[ids].sum() / counts.sum())
