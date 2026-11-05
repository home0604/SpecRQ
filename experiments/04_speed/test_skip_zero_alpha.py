"""Unit test for [specrq] skip_zero_weight_prefixes: loss and gradients must be bit-identical to the original
(all-prefix) _compute_mse, and a short quantize() run must give identical codes/codebooks/scales.
Uses Qwen2.5-0.5B layer-0 mlp.down_proj / self_attn.q_proj weights and a random-data XX^T (fp64)."""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
from argparse import Namespace
import torch, torch.nn as nn
from safetensors import safe_open
from aq_engine import AQEngine
from src.aq import QuantizedWeight

torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda:0")
f = safe_open("/data/models/qwen2.5_0.5b_instruct/model.safetensors", "pt", device="cuda")
args = Namespace(out_group_size=1, in_group_size=8, num_codebooks=5, nbits_per_codebook=8, codebook_value_nbits=16,
                 codebook_value_num_groups=1, scale_nbits=0, init_max_iter=20, init_max_points_per_centroid=None,
                 devices=[dev], lr=1e-4, max_epochs=3, steps_per_epoch=10, relative_mse_tolerance=None, print_frequency=1000,
                 beam_size=1, codebook_weights=[0, 0, 0.5, 0, 0.5])

def engine(name, seed=0):
    W = f.get_tensor(f"model.layers.0.{name}.weight")
    lin = nn.Linear(W.shape[1], W.shape[0], bias=False, device=dev, dtype=W.dtype); lin.weight.data.copy_(W)
    e = AQEngine(lin)
    g = torch.Generator(device="cuda").manual_seed(seed)
    X = torch.randn(4096, W.shape[1], device=dev, dtype=torch.float32, generator=g)
    e.add_batch(X)
    return e

ok = True
for name in ["self_attn.q_proj", "mlp.down_proj"]:
    # (1) single-call loss/grad identity on the same quantized weight
    e = engine(name)
    torch.manual_seed(0); random.seed(0)
    e.quantized_weight = QuantizedWeight(reference_weight=e._reference_weight().float(), out_group_size=1, in_group_size=8,
        num_codebooks=5, nbits_per_codebook=8, codebook_value_nbits=16, codebook_value_num_groups=1, scale_nbits=0,
        max_iter=20, devices=[dev], verbose=False)
    res = {}
    for skip in (False, True):
        e.skip_zero_weight_prefixes = skip
        for p in e.quantized_weight.parameters(): p.grad = None
        loss = e._compute_mse(); loss.backward()
        res[skip] = (loss.detach().clone(), [p.grad.clone() for p in e.quantized_weight.parameters() if p.grad is not None])
    same_loss = torch.equal(res[False][0], res[True][0])
    same_grad = all(torch.equal(a, b) for a, b in zip(res[False][1], res[True][1])) and len(res[False][1]) == len(res[True][1])
    print(f"{name}: loss old={res[False][0].item()!r} new={res[True][0].item()!r} bit_equal={same_loss} grads_bit_equal={same_grad} (n_grads={len(res[True][1])})")
    ok &= same_loss and same_grad
    # (2) short end-to-end quantize(): identical seeds -> identical codes/codebooks/scales
    # determinism control: the original code run twice (old, old2) vs the new code (new)
    outs = {}
    for tag, skip in (("old", False), ("old2", False), ("new", True)):
        e = engine(name); e.skip_zero_weight_prefixes = skip
        torch.manual_seed(1); random.seed(1)
        qw = e.quantize(args=args, verbose=False)
        outs[tag] = (qw.get_codes().clone(), qw.codebooks.detach().clone(), qw.scales.detach().clone())
    for tag in ("old2", "new"):
        eq = [torch.equal(a, b) for a, b in zip(outs["old"], outs[tag])]
        codes_agree = (outs["old"][0] == outs[tag][0]).float().mean().item()
        print(f"{name}: quantize() old vs {tag}: codes/codebooks/scales bit_equal={eq} code_agreement={codes_agree:.4f}")
print("ALL_BIT_EQUAL" if ok else "MISMATCH")
