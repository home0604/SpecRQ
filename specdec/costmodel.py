"""
Memory-traffic cost model for self-speculative decoding.

Decode is memory-bound, so a step costs ≈ bytes read / bandwidth:
  weights (same for draft and target here)  +  KV(ctx) at the representation's bits/element
  (+ the codebooks the step touches, when they are read from DRAM).

The PyTorch reference stores both caches as dequantized bf16, so its wall-clock cannot show KV
savings. This model gives the speed-up an index-reading attention kernel would reach, using the
*measured* tokens/iteration. Assumes acceptance does not change with ctx/batch.
"""


def kv_bytes_per_token(model, bits):
    return 2 * model.n_layers * model.n_kv * model.head_dim * bits / 8


def codebook_bytes(rep, model, stages, bytes_per_elem=2):
    if not hasattr(rep, "NSV"):
        return 0.0
    return 2 * model.n_layers * stages * rep.NSV * rep.C * rep.D * bytes_per_elem


def step_bytes(model, ctx, batch, bits, cb_bytes=0.0):
    return model.weight_bytes() + batch * ctx * kv_bytes_per_token(model, bits) + cb_bytes


def modeled_speedup(model, rep, tokens_per_iter, gamma, ctx, batch, include_codebook=True):
    """Speed-up of spec decoding (draft m stages + verify) over AR with the exact bf16/fp16 target."""
    cb_d = codebook_bytes(rep, model, getattr(rep, "m", 0)) if include_codebook else 0.0
    cb_t = codebook_bytes(rep, model, getattr(rep, "S", 0)) if include_codebook and rep.bits_target < 16 else 0.0
    t_ar = step_bytes(model, ctx, batch, 16.0)
    t_draft = step_bytes(model, ctx, batch, rep.bits_draft, cb_d)
    t_verify = step_bytes(model, ctx, batch, rep.bits_target, cb_t)   # γ+1 tokens, still memory-bound
    return tokens_per_iter * t_ar / (gamma * t_draft + t_verify)
