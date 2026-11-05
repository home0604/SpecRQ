"""Per-module code agreement (fraction of identical code entries, per prefix codebook) between two DbD ckpt dirs.
Usage: python compare_codes.py <dirA> <dirB> <n_layers>"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch
from src.aq import QuantizedLinear
a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = {}
for li in range(n):
    la = torch.load(f"{a}/{li}.pth", map_location="cpu", weights_only=False)
    lb = torch.load(f"{b}/{li}.pth", map_location="cpu", weights_only=False)
    mb = dict(lb.named_modules())
    for name, m in la.named_modules():
        if isinstance(m, QuantizedLinear):
            ca, cb = m.quantized_weight.get_codes(), mb[name].quantized_weight.get_codes()
            eq = (ca == cb).float()
            out[f"L{li}.{name}"] = {"all": eq.mean().item(), "per_codebook": [round(x, 4) for x in eq.mean(dim=(0, 1)).tolist()]}
            print(f"L{li}.{name:18s} agree={out[f'L{li}.{name}']['all']:.4f} per_cb={out[f'L{li}.{name}']['per_codebook']}")
print(json.dumps({"mean_all": sum(v["all"] for v in out.values()) / len(out)}))
