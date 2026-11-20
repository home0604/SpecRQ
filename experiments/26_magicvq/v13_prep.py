"""E7 magicvq v13: dense draft-weight sources below INT4 for the v12 '@<src>' mechanism (v12_w4.py), and their PPL.

Output format = the v12 INT4 source: one safetensors file keyed model.layers.<l>.{self_attn.{q,k,v,o}_proj,
mlp.{gate,up,down}_proj} -> dense [N, K] weight in the ORIGINAL (un-rotated) input basis. Only the 7 linears per layer;
norms / embeddings / lm_head stay the target's in the v12 harness.

Sub-commands
  c1    C1 weight-VQ checkpoint (DbD/AQLM QuantizedLinear pickles <ckpt>/<l>.pth, d=8 K=256 L=5, block-Hadamard input
        rotation) -> nested prefix k (first k codes / codebooks, target scales), or with --draft_cb the draft-only
        codebooks + scales of src/draft.py (shared first-k codes, E3 'B' drafts). The dense weight is read off the
        module's own forward (QuantizedLinear._forward incl. rotate_input and the draft path) on identity rows in fp32:
        W[:, j] = qlin(e_j), so it is exactly what DbD main.py / draft_target_eval evaluate. Stored in bf16 (= the dtype
        the v12 shadow layer casts to; no double rounding). Per linear: rel. error of F.linear(x, W) vs qlin(x) on
        random x (fp32), and the max |diff| of the layer's norms vs the original HF checkpoint (block fine-tuning in DbD
        may update them; the v12 harness uses the target's norms).
  gptq  GPTQ INT<b> sym g128 fake-quant (= experiments/15_dsweep/gptq_fake.gptq_quantize, the same Marlin-repo GPTQ
        code as experiments/13_int4_marlin/gptq_marlin.py) with the INT4 source's recipe: fp16 load, pile_val 128 x 2048
        (get_pile_val, seed 0), percdamp 0.1, clip on, true_sequential off. Weights stored as fp16 (as the INT4 file).
  ppl   PPL@4096 wikitext2 test / c4 val (= specrq/dbd main.py perplexity_eval protocol, gptq_fake.ppl_eval) of the HF
        model in bf16 with the 7 linears per layer taken from a source file (cast to bf16, exactly as the v12 draft
        forward), target norms; --src bf16 = original model. --norms_from_c1 DIR: also take the C1 checkpoint's layer
        norms + final norm (diagnostic only; the v12 harness never does this).
  cmp   per-tensor rel. difference of two source files (e.g. GPTQ INT4 re-run vs the v12 INT4 file, smoke check).
Usage (env nestvq):
  python experiments/26_magicvq/v13_prep.py c1 --k 3 --out DIR/w_bf16.safetensors
  python experiments/26_magicvq/v13_prep.py c1 --k 2 --draft_cb /data/lsh/results_accept/llama31_c1_uniform_kd2/train_B_bf16/draft_codebooks.pt --out ...
  python experiments/26_magicvq/v13_prep.py gptq --wbits 3 --out DIR/w_fp16.safetensors [--n_layers 1]
  python experiments/26_magicvq/v13_prep.py ppl --src FILE|bf16 --tag NAME [--out JSON]
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = "/data/models/llama3.1_8b_instruct"
C1 = "/data/lsh/ckpts/weightvq/llama31_8b/c1_had_uniform_pile_fp32_ft5"
LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _hf_get(model_path):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        with safe_open(os.path.join(model_path, idx[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name)
    return get


@torch.no_grad()
def cmd_c1(a):
    sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))  # DbD pickles reference `src.*`
    from src.aq import QuantizedLinear  # noqa: F401
    from safetensors.torch import save_file
    if os.path.exists(a.out):
        sys.exit(f"refusing to overwrite {a.out}")
    dev = torch.device("cuda:0")
    dsd = torch.load(a.draft_cb, map_location="cpu", weights_only=False) if a.draft_cb else None
    hf = _hf_get(MODEL)
    sd, checks, t0 = {}, {}, time.time()
    gen = torch.Generator(device=dev).manual_seed(0)
    n_layers = a.n_layers or 32
    for l in range(n_layers):
        lay = torch.load(os.path.join(a.ckpt, f"{l}.pth"), map_location="cpu", weights_only=False)
        mods = dict(lay.named_modules())
        nd = {}
        for nn_ in ("input_layernorm", "post_attention_layernorm"):
            w0 = hf(f"model.layers.{l}.{nn_}.weight").float()
            nd[nn_] = (getattr(lay, nn_).weight.detach().float().cpu() - w0).abs().max().item()
        for name in LINEARS:
            q = mods[name].to(dev).float()
            q.active_num_codebooks = a.k
            if dsd is not None:
                key = f"model.layers.{l}.{name}"
                assert int(dsd[key + ".draft_k"]) == a.k, (key, int(dsd[key + ".draft_k"]), a.k)
                q.draft_codebooks = torch.nn.Parameter(dsd[key + ".draft_codebooks"].to(dev).float(), requires_grad=False)
                q.draft_scales = torch.nn.Parameter(dsd[key + ".draft_scales"].to(dev).float(), requires_grad=False)
                q.draft_k = a.k
            K, N = q.in_features, q.out_features
            W = torch.empty(N, K, dtype=torch.float32, device=dev)
            for i in range(0, K, 4096):
                e = torch.zeros(min(4096, K - i), K, device=dev)
                e[:, i:i + e.shape[0]] = torch.eye(e.shape[0], device=dev)
                W[:, i:i + e.shape[0]] = q(e).t()
            assert q.bias is None
            x = torch.randn(16, K, device=dev, generator=gen)
            ref = q(x)
            rel = ((F.linear(x, W) - ref).norm() / ref.norm()).item()
            relbf = ((F.linear(x, W.bfloat16().float()) - ref).norm() / ref.norm()).item()
            w0 = hf(f"model.layers.{l}.{name}.weight").to(dev).float()
            werr = ((W - w0).norm() / w0.norm()).item()
            checks[f"{l}.{name}"] = {"rel_fwd_fp32": rel, "rel_fwd_bf16": relbf, "rel_weight_err_vs_orig": werr}
            assert rel < 1e-4, (l, name, rel)
            sd[f"model.layers.{l}.{name}"] = W.bfloat16().cpu()
            mods[name].cpu()
            del W, w0
        checks[f"{l}.norm_maxabs_diff_vs_orig"] = nd
        del lay
        torch.cuda.empty_cache()
        print(f"[c1] layer {l} k={a.k} draft_cb={'yes' if dsd else 'no'} norms diff {nd} "
              f"werr {[round(checks[f'{l}.{n}']['rel_weight_err_vs_orig'], 4) for n in LINEARS]} "
              f"fwd rel {max(checks[f'{l}.{n}']['rel_fwd_fp32'] for n in LINEARS):.1e} {time.time() - t0:.0f}s",
              flush=True)
    meta = {"src": a.ckpt, "k": a.k, "draft_cb": a.draft_cb or "", "format": "dense bf16 [N, K], original input basis "
            "(W[:, j] = QuantizedLinear(e_j) in fp32 incl. Hadamard rotation)", "n_layers": n_layers}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file(sd, a.out, metadata={k: str(v) for k, v in meta.items()})
    json.dump({"meta": meta, "checks": checks}, open(a.out.rsplit(".", 1)[0] + ".checks.json", "w"), indent=1)
    print("wrote", a.out, len(sd), f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def cmd_gptq(a):
    sys.path.insert(0, os.path.join(REPO, "experiments", "15_dsweep"))
    from gptq_fake import gptq_quantize, LINEARS as GL  # noqa: E402 (imports marlin gptq + dbd src)
    from src.datautils import get_pile_val
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, LlamaForCausalLM
    if os.path.exists(a.out):
        sys.exit(f"refusing to overwrite {a.out}")
    dev = torch.device("cuda:0")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    calib = get_pile_val(a.nsamples, a.seqlen, tok, seed=a.seed)
    model = LlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                             attn_implementation="sdpa")
    model.config.use_cache = False
    model.eval()
    a.percdamp, a.groupsize, a.no_clip = 0.1, 128, False
    errs, t_layers = gptq_quantize(model, calib, a, dev)
    sd = {}
    for l in range(a.n_layers or len(model.model.layers)):
        mods = dict(model.model.layers[l].named_modules())
        for n in GL:
            sd[f"model.layers.{l}.{n}"] = mods[n].weight.data.to(torch.float16).contiguous()
    meta = {"model": MODEL, "format": f"dense fp16 [N, K] GPTQ INT{a.wbits} sym g128 fake-quant", "wbits": a.wbits,
            "calibration": "pile_val get_pile_val", "nsamples": a.nsamples, "seqlen": a.seqlen, "seed": a.seed,
            "percdamp": 0.1, "groupsize": 128, "clip": True, "true_sequential": False, "load_dtype": "float16",
            "gptq_code": "experiments/15_dsweep/gptq_fake.gptq_quantize (marlin repo gptq.py @1f25790)",
            "wall_s": time.time() - t0, "n_layers": a.n_layers or 32}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_file(sd, a.out, metadata={k: str(v) for k, v in meta.items()})
    json.dump({"meta": meta, "rel_weight_err": errs, "layer_time_s": t_layers},
              open(a.out.rsplit(".", 1)[0] + ".meta.json", "w"), indent=1)
    print("wrote", a.out, len(sd), f"{time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def cmd_ppl(a):
    # c4 val: with HF_DATASETS_OFFLINE=1 the data_files config hash differs from the cached one and load_dataset fails;
    # the repo's earlier PPL jobs (e.g. e5_lds_gptq_int3) ran without it (hub metadata lookup only, cached data).
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    sys.path.insert(0, os.path.join(REPO, "experiments", "15_dsweep"))
    from gptq_fake import ppl_eval
    from transformers import LlamaForCausalLM
    dev = torch.device("cuda:0")
    t0 = time.time()
    m = LlamaForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                         attn_implementation="sdpa")
    if a.src != "bf16":
        from safetensors import safe_open
        with safe_open(a.src, framework="pt", device="cpu") as f:
            for l, layer in enumerate(m.model.layers):
                mods = dict(layer.named_modules())
                for n in LINEARS:
                    w = f.get_tensor(f"model.layers.{l}.{n}")
                    assert w.shape == mods[n].weight.shape
                    mods[n].weight.data.copy_(w.to(torch.bfloat16))
    if a.norms_from_c1:
        sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
        for l, layer in enumerate(m.model.layers):
            lay = torch.load(os.path.join(a.norms_from_c1, f"{l}.pth"), map_location="cpu", weights_only=False)
            for nn_ in ("input_layernorm", "post_attention_layernorm"):
                getattr(layer, nn_).weight.data.copy_(getattr(lay, nn_).weight.detach().to(torch.bfloat16))
        nq = torch.load(os.path.join(a.norms_from_c1, "not_quantized_weights.pt"), map_location="cpu", weights_only=False)
        m.model.norm.weight.data.copy_(nq["model.norm.weight"].to(torch.bfloat16))
    m.config.use_cache = False
    m = m.to(dev).eval()
    ppl = ppl_eval(m, MODEL, dev)
    rec = {"tag": a.tag, "src": a.src, "norms_from_c1": a.norms_from_c1 or "", "ppl_4096": ppl,
           "dtype": "bf16 nn.Linear", "wall_s": time.time() - t0}
    print("RESULT", json.dumps(rec), flush=True)
    if a.out:
        d = json.load(open(a.out)) if os.path.exists(a.out) else {}
        d[a.tag] = rec
        json.dump(d, open(a.out, "w"), indent=1)


def cmd_cmp(a):
    from safetensors import safe_open
    fa, fb = safe_open(a.a, "pt", device="cpu"), safe_open(a.b, "pt", device="cpu")
    ks = sorted(set(fa.keys()) & set(fb.keys()))
    for k in ks:
        x, y = fa.get_tensor(k).float(), fb.get_tensor(k).float()
        print(k, f"rel {((x - y).norm() / y.norm()).item():.3e} frac_neq {(x != y).float().mean().item():.3e}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("c1")
    p.add_argument("--ckpt", default=C1)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--draft_cb", default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--out", required=True)
    p = sp.add_parser("gptq")
    p.add_argument("--wbits", type=int, required=True)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--out", required=True)
    p = sp.add_parser("ppl")
    p.add_argument("--src", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--norms_from_c1", default=None)
    p.add_argument("--out", default=None)
    p = sp.add_parser("cmp")
    p.add_argument("a")
    p.add_argument("b")
    a = ap.parse_args()
    {"c1": cmd_c1, "gptq": cmd_gptq, "ppl": cmd_ppl, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
