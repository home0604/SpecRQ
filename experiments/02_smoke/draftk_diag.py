"""Diagnostic: wikitext2 PPL (first N windows, seqlen 512) of a saved DbD model with per-module prefix k.
Usage: python draftk_diag.py <model_dir> <ckpt_dir> <n_windows>"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch, torch.nn.functional as F
from src.modelutils import get_model
from src.datautils import get_loaders
from src.aq import QuantizedLinear
model_dir, ckpt, nwin = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = get_model(model_dir, ckpt, "auto").cuda().eval()
enc = get_loaders("wikitext2", seed=0, model_path=model_dir, seqlen=512, eval_mode=True)
mods = [(n, x) for n, x in m.named_modules() if isinstance(x, QuantizedLinear)]
@torch.no_grad()
def ppl(rule):
    for n, x in mods: x.active_num_codebooks = rule(n)
    nll = 0.0
    for i in range(nwin):
        ids = enc[:, i * 512:(i + 1) * 512].cuda()
        lg = m(ids).logits.float()
        nll += F.cross_entropy(lg[0, :-1], ids[0, 1:], reduction="sum").item()
    return float(torch.exp(torch.tensor(nll / (nwin * 511))))
L = lambda n: int(n.split(".")[2])
cases = {
 "all_k5": lambda n: 5, "all_k4": lambda n: 4, "all_k3": lambda n: 3, "all_k2": lambda n: 2,
 "k3_but_o_proj_k5": lambda n: 5 if "o_proj" in n else 3,
 "k3_but_down_proj_k5": lambda n: 5 if "down_proj" in n else 3,
 "k3_but_L11_o_proj_k5": lambda n: 5 if (L(n) == 11 and "o_proj" in n) else 3,
 "k3_but_outliers_k5": lambda n: 5 if (L(n), n.split(".")[-1]) in {(11, "o_proj"), (9, "o_proj"), (2, "down_proj"), (21, "down_proj"), (3, "down_proj")} else 3,
 "k3_attn_only": lambda n: 3 if "self_attn" in n else 5,
 "k3_mlp_only": lambda n: 3 if "mlp" in n else 5,
}
res = {}
for name, rule in cases.items():
    res[name] = ppl(rule); print(name, round(res[name], 3), flush=True)
print(json.dumps({"n_windows": nwin, "ppl": res}))
