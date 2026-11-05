"""Weight-space relative MSE of prefix-k reconstructions for layers saved by specrq/dbd/main.py.

rel_mse(k) = ||W - W_hat^(k)||_F^2 / ||W||_F^2, with W_hat^(k) = AQ.forward(num_codebooks=k) (first k codebooks).
Usage (from specrq/dbd so that pickled `src.aq` classes resolve):
  python ../../experiments/02_smoke/prefix_mse.py --save DIR --model /data/models/qwen2.5_0.5b_instruct --ks 1 2 3 4 5
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch
from safetensors import safe_open
from src.aq import QuantizedLinear

ap = argparse.ArgumentParser()
ap.add_argument("--save", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--ks", type=int, nargs="+", default=[3, 5])
ap.add_argument("--out", default=None)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.model, "*.safetensors")))
key2file = {}
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys():
            key2file[k] = f

rows = []
layer_files = sorted(glob.glob(os.path.join(a.save, "[0-9]*.pth")), key=lambda p: int(re.findall(r"(\d+)\.pth", p)[0]))
for lf in layer_files:
    li = int(re.findall(r"(\d+)\.pth", lf)[0])
    layer = torch.load(lf, map_location="cuda", weights_only=False)
    for name, mod in layer.named_modules():
        if not isinstance(mod, QuantizedLinear):
            continue
        key = f"model.layers.{li}.{name}.weight"
        with safe_open(key2file[key], "pt", device="cuda") as h:
            W = h.get_tensor(key).float()
        qw = mod.quantized_weight
        r = {"layer": li, "module": name, "shape": list(W.shape), "M": int(qw.num_codebooks),
             "bpw_est": float(qw.estimate_nbits_per_parameter())}
        with torch.no_grad():
            for k in a.ks:
                Wk = qw(num_codebooks=k).float()
                r[f"rel_mse_k{k}"] = float(((W - Wk) ** 2).sum() / (W ** 2).sum())
        rows.append(r)
        print(json.dumps(r), flush=True)
    del layer

if a.out:
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
