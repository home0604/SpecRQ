"""Collect XX^T (AQEngine.add_batch, fp64) for selected linears of Qwen2.5-0.5B, for the e-means init comparison.

Calibration = main.py recipe: get_loaders(wikitext2, nsamples 1024, seqlen 4096, seed 0), then the same
torch.randperm val split (val 32), XTX over the remaining 992 train sequences.
Difference from a real main.py run: inputs are propagated through the FP (unquantized) model, so for layers > 0 the
XTX differs from the run's XTX (which sees quantized earlier layers). Layer 0 is identical up to kernel numerics.
Weights are saved in their checkpoint dtype (bf16), as main.py's AQEngine sees them.

Usage (cwd anywhere): python experiments/17_emeans/collect_xtx.py --layers 0 12 23 --out /data/lsh/temp/emeans_init/xtx
"""
import argparse
import os
import sys
import time

DBD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)

import torch  # noqa: E402

from aq_engine import AQEngine  # noqa: E402
from src.datautils import get_loaders  # noqa: E402
from src.modelutils import get_model  # noqa: E402

# linears that share one input: XTX computed once per group
GROUPS = {
    "attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "attn_out": ["self_attn.o_proj"],
    "mlp_in": ["mlp.gate_proj", "mlp.up_proj"],
    "mlp_out": ["mlp.down_proj"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    p.add_argument("--nsamples", type=int, default=1024)
    p.add_argument("--val_size", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = get_loaders("wikitext2", nsamples=a.nsamples, seed=a.seed, model_path=a.model, seqlen=a.seqlen)
    all_ids = torch.randperm(len(data))  # same RNG call order as main.py quantize_model
    train_ids = all_ids[a.val_size:]
    train = [data[i] for i in train_ids]
    print(f"train seqs {len(train)} first ids {train_ids[:5].tolist()}", flush=True)

    model = get_model(a.model, None, "auto").train(False).cuda()
    layers = model.model.layers
    engines = {}
    hooks = []
    for li in a.layers:
        for g, names in GROUPS.items():
            mod = layers[li].get_submodule(names[0])
            eng = AQEngine(mod)
            engines[(li, g)] = eng

            def hook(m, inp, out, eng=eng):
                eng.add_batch(inp[0].data)

            hooks.append(mod.register_forward_hook(hook))
    t0 = time.time()
    with torch.no_grad():
        for j, x in enumerate(train):
            model.model(x.cuda())  # base model only; logits not needed
            if j % 64 == 0:
                print(f"seq {j}/{len(train)} {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    for (li, g), eng in engines.items():
        for name in GROUPS[g]:
            w = layers[li].get_submodule(name).weight.detach().cpu()
            torch.save({"layer": li, "name": name, "weight": w, "XTX": eng.XTX.cpu(), "nsamples": eng.nsamples,
                        "calib": f"wikitext2 n{a.nsamples} seqlen{a.seqlen} seed{a.seed} val{a.val_size} FP-propagated"},
                       os.path.join(a.out, f"l{li}_{name}.pt"))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
