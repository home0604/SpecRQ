"""Scalar baseline for the claim-1 d-sweep: GPTQ INTb (sym, group 128) fake-quant on Qwen2.5-0.5B-Instruct, quality only.

GPTQ = Marlin repo variant (/data/lsh/src/marlin @1f25790, gptq/{gptq,quant}.py), the same code path as
experiments/13_int4_marlin/gptq_marlin.py, with these changes:
  * Qwen2 (AutoModelForCausalLM, bf16 = checkpoint dtype) instead of Llama/fp16; no Marlin packing (fake-quant only:
    each linear's weight is replaced by the GPTQ dequantized values, computed in fp32 and stored in bf16);
  * --wbits 2/3 (sym: levels (q - 2^(b-1)) * s, q in [0, 2^b - 1]);
  * calibration = the VQ sweep's: wikitext2 train 1024 x 4096 (specrq/dbd get_loaders, seed 0), so scalar and VQ see
    the same data. Layer inputs live on the GPU (2 x 1024 x 4096 x 896 bf16 = 15 GB).
GPTQ settings = llama2.py defaults: percdamp 0.1, Hessian-based grid clipping on, blocksize 128, no act-order,
true_sequential off. bpw (with scales) = b + 16/128.
Eval (same protocol as the VQ runs): PPL@4096 wikitext2 test / c4 val (= specrq/dbd main.py perplexity_eval), and
KL(p_bf16 || p_q), top-1 agreement on 40 x 512 windows (= draft_target_eval.py 'kl_fp16_target'/'top1_fp16_target';
reference = original model in bf16 on the same GPU).
Usage: python experiments/15_dsweep/gptq_fake.py --wbits 2 [--nsamples 1024 --seqlen 4096] [--n_layers N] [--save_result]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join("/data/lsh/src/marlin", "gptq"))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
sys.path.insert(0, REPO)
from gptq import GPTQ  # noqa: E402  (sets allow_tf32=False)
from quant import Quantizer  # noqa: E402
from src.datautils import get_loaders  # noqa: E402

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


@torch.no_grad()
def gptq_quantize(model, calib, a, dev):
    layers = model.model.layers
    model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    layers[0].to(dev)
    inps = torch.zeros(len(calib), a.seqlen, model.config.hidden_size, dtype=model.dtype, device=dev)
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
    kw = {k: v for k, v in cache["kw"].items() if k in ("attention_mask", "position_ids", "position_embeddings",
                                                         "cache_position")}
    print("captured layer kwargs:", {k: (type(v).__name__ if not isinstance(v, torch.Tensor) else tuple(v.shape))
                                     for k, v in kw.items()}, flush=True)
    outs = torch.zeros_like(inps)
    errs, t_layers = {}, []
    for i in range(a.n_layers or len(layers)):
        t0 = time.time()
        layer = layers[i].to(dev)
        mods = dict(layer.named_modules())
        gq = {}
        for n in LINEARS:
            gq[n] = GPTQ(mods[n])
            gq[n].quantizer = Quantizer()
            gq[n].quantizer.configure(a.wbits, sym=True)

        def hook(n):
            def f(_, inp, out):
                gq[n].add_batch(inp[0].data, out.data)
            return f
        hs = [mods[n].register_forward_hook(hook(n)) for n in LINEARS]
        for j in range(len(calib)):
            layer(inps[j:j + 1], **kw)
        for h in hs:
            h.remove()
        for n in LINEARS:
            w0 = mods[n].weight.data.float().clone()
            gq[n].fasterquant(percdamp=a.percdamp, groupsize=a.groupsize, clip=not a.no_clip)
            wq = mods[n].weight.data.float()
            errs[f"{i}.{n}"] = ((wq - w0).norm() / w0.norm()).item()
            # sanity: per 128-group, at most 2^b distinct levels
            g = wq.reshape(wq.shape[0], -1, a.groupsize)[:4, :4]
            assert max(len(torch.unique(x)) for x in g.reshape(-1, a.groupsize)) <= 2 ** a.wbits
            gq[n].layer = None
        del gq
        for j in range(len(calib)):
            outs[j] = layer(inps[j:j + 1], **kw)[0][0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        t_layers.append(time.time() - t0)
        print(f"[gptq] layer {i} done {t_layers[-1]:.1f}s, rel weight err "
              f"{[round(errs[f'{i}.{n}'], 4) for n in LINEARS]}", flush=True)
    del inps, outs
    model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()
    return errs, t_layers


@torch.no_grad()
def ppl_eval(m, model_path, dev, seqlen=4096):
    out = {}
    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=model_path, seqlen=seqlen, eval_mode=True)
        n = enc.numel() // seqlen
        nll = []
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            lg = m.lm_head(m.model(ids)[0])[0, :-1]  # as main.py perplexity_eval: logits in model dtype
            nll.append(F.cross_entropy(lg, ids[0, 1:]).float() * seqlen)
        out[f"{name}"] = torch.exp(torch.stack(nll).sum() / (n * seqlen)).item()
        print(f"[eval] ppl@{seqlen} {name} = {out[name]:.4f} ({n} windows)", flush=True)
    return out


@torch.no_grad()
def kl_eval(q, ref, model_path, dev, n_windows=40, seqlen=512, rdev=None):
    out = {}
    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=model_path, seqlen=seqlen, eval_mode=True)
        n = min(n_windows, enc.numel() // seqlen)
        acc = {k: [] for k in ("kl", "top1", "nll_q", "nll_ref")}
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            lq = q(ids).logits.float()[0][:-1]
            lr = (ref[name][i].to(dev).float() if isinstance(ref, dict)
                  else ref(ids.to(rdev or dev)).logits.float()[0][:-1].to(dev))
            y = ids[0, 1:]
            pr, lpr, lpq = lr.softmax(-1), lr.log_softmax(-1), lq.log_softmax(-1)
            acc["kl"].append((pr * (lpr - lpq)).sum(-1).cpu())
            acc["top1"].append((lr.argmax(-1) == lq.argmax(-1)).float().cpu())
            acc["nll_q"].append(F.cross_entropy(lq, y, reduction="none").cpu())
            acc["nll_ref"].append(F.cross_entropy(lr, y, reduction="none").cpu())
        c = {k: torch.cat(v) for k, v in acc.items()}
        out[name] = {"n_windows": n, "n_tokens": int(c["kl"].numel()), "kl_fp16_target": c["kl"].mean().item(),
                     "top1_fp16_target": c["top1"].mean().item(), "ppl_target": torch.exp(c["nll_q"].mean()).item(),
                     "ppl_fp16": torch.exp(c["nll_ref"].mean()).item()}
        print(f"[eval] kl {name}: {out[name]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--wbits", type=int, required=True)
    ap.add_argument("--nsamples", type=int, default=1024)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--percdamp", type=float, default=0.1)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--no_clip", action="store_true")
    ap.add_argument("--n_layers", type=int, default=None, help="smoke test: quantize only the first n layers, no eval")
    ap.add_argument("--save_result", action="store_true")
    ap.add_argument("--calib", default="wikitext2", help="wikitext2 (Qwen sweep) or pile_val (Llama, CLAUDE.md)")
    ap.add_argument("--ref_device", default=None, help="device of the bf16 KL reference (default: same GPU)")
    ap.add_argument("--ref_logits_cpu", action="store_true",
                    help="single GPU: after PPL, move the quantized model to the CPU, compute the bf16 reference logits "
                         "on the GPU and keep them on the CPU (bf16, exact lm_head output), then evaluate KL")
    ap.add_argument("--run_name", default=None, help="result name (default qwen05_dsweep_gptq_int<b>)")
    ap.add_argument("--model_tag", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--no_quant", action="store_true", help="reference: evaluate the original bf16 model (PPL@4096 only)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    if a.no_quant:
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        print(json.dumps({"bf16_ppl_4096": ppl_eval(m, a.model, dev)}), flush=True)
        return
    calib = get_loaders(a.calib, nsamples=a.nsamples, seed=a.seed, seqlen=a.seqlen, model_path=a.model)
    calib = [b[0] if isinstance(b, (tuple, list)) else b for b in calib]
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model.eval()
    errs, t_layers = gptq_quantize(model, calib, a, dev)
    t_quant = time.time() - t0
    print(f"[gptq] quantization done {t_quant:.0f}s", flush=True)
    if a.n_layers:
        return
    model.to(dev)
    ppl = ppl_eval(model, a.model, dev)
    rdev = torch.device(a.ref_device) if a.ref_device else dev
    if a.ref_logits_cpu:
        model.cpu(); torch.cuda.empty_cache()
        r = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        ref = {}
        with torch.no_grad():
            for name in ("wikitext2", "c4"):
                enc = get_loaders(name, seed=0, model_path=a.model, seqlen=512, eval_mode=True)
                n = min(40, enc.numel() // 512)
                ref[name] = [r(enc[:, i * 512:(i + 1) * 512].to(dev)).logits[0][:-1].to(torch.bfloat16).cpu() for i in range(n)]
        del r; torch.cuda.empty_cache(); model.to(dev)
    else:
        ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(rdev).eval()
    kl = kl_eval(model, ref, a.model, dev, rdev=rdev)
    lin_shapes = [(m.out_features, m.in_features) for n, m in model.model.layers[0].named_modules()
                  if n in LINEARS]
    nw = sum(o * i for o, i in lin_shapes)
    nsc = sum(o * (i // a.groupsize) for o, i in lin_shapes)
    metrics = {"ppl_4096": {f"{k}_k1": v for k, v in ppl.items()}, "target_quality": kl,
               "target_bpw": {"index_bpw": float(a.wbits), "bpw_with_codebooks_scales": a.wbits + 16 * nsc / nw},
               "layer_time_s": t_layers, "quant_wall_s": t_quant, "rel_weight_err": errs,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30, "wall_s": time.time() - t0}
    print(json.dumps({k: metrics[k] for k in ("ppl_4096", "target_quality", "target_bpw")}), flush=True)
    if a.save_result:
        from tools.results import save_result
        cfg = {"model": a.model_tag, "cfg": f"gptq_int{a.wbits}_g{a.groupsize}", "wbits": a.wbits,
               "groupsize": a.groupsize, "sym": True,
               "setting": f"GPTQ (marlin repo gptq.py) INT{a.wbits} sym g{a.groupsize}, percdamp {a.percdamp}, "
                          f"clip {not a.no_clip}, no act-order, {a.calib} {a.nsamples}x{a.seqlen} calib (= VQ sweep), "
                          "fake-quant bf16 weights, no Hadamard",
               "eval": "PPL@4096 wt2/c4 (main.py protocol); KL/top-1 vs bf16 original on 40x512 wt2 + c4 windows",
               "scripts": ["experiments/15_dsweep/gptq_fake.py"]}
        print(save_result("E5", a.run_name or f"qwen05_dsweep_gptq_int{a.wbits}", cfg, metrics, model_dir=a.model))


if __name__ == "__main__":
    main()
