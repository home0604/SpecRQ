"""Rotation-only check for specrq/dbd --hadamard: replace every decoder-layer nn.Linear by
y = (W R^T) (R x) with R = blockdiag(H_B) diag(S) (same seeds/block sizes as main.py --hadamard, no quantization)
and compare with the original model: logits max|diff| and wikitext2/c4 PPL (same protocol as main.py perplexity_eval:
non-overlapping seqlen windows of the joined test text).
Usage: python check_rotation_equivalence.py <model_dir> <dtype: float32|bfloat16> <seqlen> <out_json>
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from src.datautils import get_loaders
from src.hadamard import layer_seed, make_signs, pick_block_size, rotate_input, rotate_weight

model_dir, dtype, seqlen, out = sys.argv[1], getattr(torch, sys.argv[2]), int(sys.argv[3]), sys.argv[4]
SEED = 0  # main.py --seed default

class RotLinear(nn.Module):
    def __init__(self, lin, signs, b):
        super().__init__()
        self.weight = nn.Parameter(rotate_weight(lin.weight.detach().float(), signs, b).to(lin.weight.dtype), requires_grad=False)
        self.bias = lin.bias
        self.register_buffer("signs", signs)
        self.b = b
    def forward(self, x):
        return F.linear(rotate_input(x, self.signs, self.b), self.weight, self.bias)

def build(rotate):
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype).cuda().eval()
    info = {}
    if rotate:
        for li, layer in enumerate(m.model.layers):
            for name, mod in list(layer.named_modules()):
                if isinstance(mod, nn.Linear):
                    b = pick_block_size(mod.in_features)
                    signs = make_signs(mod.in_features, layer_seed(SEED, li, name), device="cuda")
                    parent = layer.get_submodule(name.rsplit(".", 1)[0])
                    setattr(parent, name.rsplit(".", 1)[1], RotLinear(mod, signs, b))
                    info[name] = b
    return m, info

@torch.no_grad()
def ppl_and_logits(m, enc, keep=4):
    n = enc.numel() // seqlen
    nll, kept = 0.0, []
    for i in range(n):
        ids = enc[:, i * seqlen:(i + 1) * seqlen].cuda()
        lg = m(ids).logits.float()
        nll += F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum").item()
        if i < keep: kept.append(lg.cpu())
    return float(torch.exp(torch.tensor(nll / (n * (seqlen - 1))))), kept, n

res = {"model_dir": model_dir, "dtype": sys.argv[2], "seqlen": seqlen, "seed": SEED}
encs = {d: get_loaders(d, seed=0, model_path=model_dir, seqlen=seqlen, eval_mode=True) for d in ["wikitext2", "c4"]}
base, _ = build(False)
ref = {d: ppl_and_logits(base, e) for d, e in encs.items()}
del base; torch.cuda.empty_cache()
rot, info = build(True)
res["block_sizes"] = sorted(set(info.values())); res["n_rotated_linears_per_layer"] = len(info)
for d, e in encs.items():
    p_rot, lg_rot, n = ppl_and_logits(rot, e)
    p_ref, lg_ref, _ = ref[d]
    diffs = [(a - b).abs().max().item() for a, b in zip(lg_ref, lg_rot)]
    scale = max(a.abs().max().item() for a in lg_ref)
    agree = sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(lg_ref, lg_rot)) / len(lg_ref)
    res[d] = {"n_windows": n, "ppl_ref": p_ref, "ppl_rot": p_rot, "logit_maxabs_diff_first4": max(diffs),
              "logit_maxabs_ref": scale, "top1_agree_first4": agree}
    print(d, json.dumps(res[d]), flush=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
