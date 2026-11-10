"""Real INT4 baseline weights: GPTQ (Marlin repo variant, /data/lsh/src/marlin @1f25790, gptq/{gptq,quant}.py) on a
Llama HF checkpoint -> Marlin FP16 x INT4 format (symmetric, group size 128).

Follows marlin/gptq/llama2.py (llama_sequential + llama_pack) with these changes:
  * calibration = pile_val (specrq/dbd/src/datautils.get_pile_val, local pile-val-backup) nsamples x seqlen
    (default 128 x 2048) instead of RedPajama;
  * the model is loaded in fp16 (Marlin packs only torch.half weights; Llama-3.1 ships bf16);
  * decoder-layer kwargs (attention_mask / position_ids / position_embeddings / cache_position) are captured from
    the first layer call and replayed (transformers 4.46 Llama layers take position_embeddings);
  * each layer is packed right after it is quantized and only the Marlin tensors (B, s) are saved, plus a per-linear
    kernel check: marlin.mul(x) vs x @ W_fakequant^T on random x (M = 16).
GPTQ settings = llama2.py defaults: wbits 4, groupsize 128, percdamp 0.1, Hessian-based grid clipping on,
true_sequential off (--true-sequential to enable), sym.
Output: <save>/marlin_linears.pt {"model.layers.i.<lin>.B": int32, ".s": fp16}, <save>/meta.json.
Usage: python experiments/13_int4_marlin/gptq_marlin.py --model DIR --save DIR [--nsamples 128 --seqlen 2048]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARLIN_SRC = "/data/lsh/src/marlin"
sys.path.insert(0, os.path.join(MARLIN_SRC, "gptq"))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
from gptq import GPTQ  # noqa: E402  (marlin/gptq/gptq.py; sets allow_tf32=False)
from quant import Quantizer  # noqa: E402
import marlin  # noqa: E402

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
SEQUENTIAL = [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"],
              ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--save", required=True)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--percdamp", type=float, default=0.1)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--true-sequential", action="store_true")
    ap.add_argument("--n_layers", type=int, default=None, help="smoke test: quantize only the first n layers, no save")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.save, "marlin_linears.pt")):
        sys.exit(f"refusing to overwrite {a.save}")
    os.makedirs(a.save, exist_ok=True)
    dev = torch.device("cuda:0")
    t0 = time.time()

    from transformers import AutoTokenizer, LlamaForCausalLM
    from src.datautils import get_pile_val
    tok = AutoTokenizer.from_pretrained(a.model)
    calib = get_pile_val(a.nsamples, a.seqlen, tok, seed=a.seed)
    model = LlamaForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                             attn_implementation="sdpa")
    model.config.use_cache = False
    model.eval()
    layers = model.model.layers

    # ---- capture layer-0 inputs
    model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    layers[0].to(dev)
    inps = torch.zeros(a.nsamples, a.seqlen, model.config.hidden_size, dtype=torch.float16, device=dev)
    cache = {"i": 0, "kw": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kw):
            inps[cache["i"]] = inp[0]
            cache["i"] += 1
            cache["kw"] = kw
            raise ValueError

    layers[0] = Catcher(layers[0])
    for b in calib:
        try:
            model(b.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0].cpu()
    model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    kw = {k: v for k, v in cache["kw"].items() if k in ("attention_mask", "position_ids", "position_embeddings",
                                                         "cache_position")}
    print("captured layer kwargs:", {k: (type(v).__name__ if not isinstance(v, torch.Tensor) else tuple(v.shape))
                                     for k, v in kw.items()}, flush=True)
    outs = torch.zeros_like(inps)

    packed, checks, errs = {}, {}, {}
    seq = SEQUENTIAL if a.true_sequential else [LINEARS]
    for i in range(a.n_layers or len(layers)):
        layer = layers[i].to(dev)
        mods = dict(layer.named_modules())
        for names in seq:
            gq = {}
            for n in names:
                gq[n] = GPTQ(mods[n])
                gq[n].quantizer = Quantizer()
                gq[n].quantizer.configure(4, sym=True)

            def hook(n):
                def f(_, inp, out):
                    gq[n].add_batch(inp[0].data, out.data)
                return f
            hs = [mods[n].register_forward_hook(hook(n)) for n in names]
            for j in range(a.nsamples):
                layer(inps[j:j + 1], **kw)
            for h in hs:
                h.remove()
            for n in names:
                print(i, n, flush=True)
                scale, _zero = gq[n].fasterquant(percdamp=a.percdamp, groupsize=a.groupsize, clip=not a.no_clip)
                lin = mods[n]  # weight now holds the fake-quantized values (fp16)
                ml = marlin.Layer(lin.in_features, lin.out_features, groupsize=a.groupsize).to(dev)
                ml.pack(lin, scale.to(dev))
                x = torch.randn(16, lin.in_features, device=dev, dtype=torch.float16)
                ref = (x.float() @ lin.weight.float().t())
                y = ml(x).float()
                rel = ((y - ref).norm() / ref.norm()).item()
                key = f"model.layers.{i}.{n}"
                checks[key] = rel
                assert rel < 1e-2, f"Marlin pack check failed for {key}: rel err {rel}"
                packed[key + ".B"] = ml.B.cpu()
                packed[key + ".s"] = ml.s.cpu()
                del ml
                gq[n].layer = None
            del gq
        for j in range(a.nsamples):
            outs[j] = layer(inps[j:j + 1], **kw)[0][0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f"[gptq] layer {i} done, max pack-check rel err so far {max(checks.values()):.2e}, "
              f"{time.time() - t0:.0f}s", flush=True)

    if a.n_layers:
        print("smoke test done", checks, f"{time.time() - t0:.0f}s")
        return
    torch.save(packed, os.path.join(a.save, "marlin_linears.pt"))
    meta = {"model": a.model, "format": "Marlin FP16xINT4, sym, group 128 (B int32 [K/16, N*2], s fp16 [K/128, N])",
            "gptq_src": f"{MARLIN_SRC} @1f25790 gptq/gptq.py + quant.py", "calibration": "pile_val (pile-val-backup, "
            "specrq/dbd/src/datautils.get_pile_val)", "nsamples": a.nsamples, "seqlen": a.seqlen, "seed": a.seed,
            "percdamp": a.percdamp, "groupsize": a.groupsize, "clip": not a.no_clip,
            "true_sequential": a.true_sequential, "load_dtype": "float16", "wall_s": time.time() - t0,
            "pack_check_rel_err_max": max(checks.values()), "pack_check_rel_err": checks,
            "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    json.dump(meta, open(os.path.join(a.save, "meta.json"), "w"), indent=1)
    print("saved", a.save, {k: v for k, v in meta.items() if k != "pack_check_rel_err"}, flush=True)


if __name__ == "__main__":
    main()
