"""Quality of the real INT4 Marlin baseline vs the original model, on the same protocol as the C1 VQ numbers:
  ppl : PPL @4096 on wikitext2 test / c4 val (specrq/dbd get_loaders eval_mode, non-overlapping 4096 windows,
        exp(mean token CE)) = specrq/dbd/main.py perplexity_eval (E3 'ppl_4096').
  kl  : 40 x 512 windows of wikitext2 test and c4 val (draft_target_eval.py protocol): KL(p_ref || p_q), top-1
        agreement and PPL of both, ref = original model in bf16 (as draft_target_eval's "fp16" reference, which
        loads the model in its own dtype = bf16). Ref logits are kept on the CPU in bf16 (= the HF lm_head output
        before .float(), so exact), then the quantized model is evaluated.
Models: bf16 / fp16 = original HF weights in that dtype; marlin = HF Llama in fp16 with the 7 linears per layer
replaced by marlin.Layer (real Marlin kernel) loaded from marlin_linears.pt; int4dense = the same INT4 weights
dequantized (through the Marlin kernel itself, M=16 identity rows) into dense fp16 nn.Linear weights. Marlin with
M > 16 is not bitwise reproducible (fp16 global reduction), so int4dense gives the deterministic PPL of the same weights.
Usage: python experiments/13_int4_marlin/eval_quality.py --marlin-ckpt FILE --steps ppl:bf16 ppl:fp16 ppl:marlin kl:marlin --out JSON
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
from src.datautils import get_loaders  # noqa: E402

LIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load(model_path, kind, marlin_ckpt, dev):
    from transformers import LlamaForCausalLM
    dtype = torch.bfloat16 if kind == "bf16" else torch.float16  # fp16 for fp16 / marlin / int4dense
    m = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
                                         attn_implementation="sdpa")
    if kind == "marlin":
        import marlin
        marlin.replace_linear(m, lambda n: n.split(".")[-1] in LIN, groupsize=128)
        r = m.load_state_dict(torch.load(marlin_ckpt, map_location="cpu"), strict=False)
        assert not r.unexpected_keys, r.unexpected_keys[:5]
        bad = [k for k in r.missing_keys if k.endswith((".B", ".s"))]
        assert not bad, bad[:5]
        n = sum(isinstance(x, marlin.Layer) for x in m.modules())
        assert n == 7 * m.config.num_hidden_layers, n
        print(f"[eval] {n} marlin.Layer modules loaded", flush=True)
    if kind == "int4dense":  # same INT4 weights, dequantized once to dense fp16 (deterministic cuBLAS GEMMs)
        import marlin
        pk = torch.load(marlin_ckpt, map_location="cpu", mmap=True)
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in LIN:
                mod.weight.data = dequant_marlin(pk[name + ".B"], pk[name + ".s"], dev).cpu()
        print("[eval] dequantized Marlin weights into dense fp16 linears", flush=True)
    m.config.use_cache = False
    return m.to(dev).eval()


@torch.no_grad()
def dequant_marlin(B, s, dev):
    """W [N, K] fp16 = Marlin layer applied to identity rows (M = 16 per call: bitwise deterministic path); each
    output is the single product (q - 8) * s of the kernel's own fp16 dequantization."""
    import marlin
    K, N = B.shape[0] * 16, s.shape[1]
    B, s = B.to(dev), s.to(dev)
    ws = torch.zeros(N // 128 * 16, dtype=torch.int32, device=dev)
    W = torch.empty(K, N, dtype=torch.float16, device=dev)
    eye = torch.eye(16, dtype=torch.float16, device=dev)
    x = torch.zeros(16, K, dtype=torch.float16, device=dev)
    for i in range(0, K, 16):
        x.zero_()
        x[:, i:i + 16] = eye
        marlin.mul(x, B, W[i:i + 16], s, ws)
    return W.t().contiguous()


@torch.no_grad()
def ppl_eval(m, model_path, dev, seqlen=4096):
    out = {}
    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=model_path, seqlen=seqlen, eval_mode=True)
        n = enc.numel() // seqlen
        nll = []
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            # as specrq/dbd/main.py perplexity_eval: final norm + lm_head in the model dtype, CE on those logits
            lg = m.lm_head(m.model(ids)[0])[0, :-1]
            nll.append(F.cross_entropy(lg, ids[0, 1:]).float() * seqlen)
        out[name] = torch.exp(torch.stack(nll).sum() / (n * seqlen)).item()
        out[f"{name}_n_windows"] = n
        print(f"[eval] ppl@{seqlen} {name} = {out[name]:.4f} ({n} windows)", flush=True)
    return out


@torch.no_grad()
def ref_logits(m, model_path, dev, n_windows=40, seqlen=512, cache_dir=None):
    res = {}
    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=model_path, seqlen=seqlen, eval_mode=True)
        n = min(n_windows, enc.numel() // seqlen)
        if cache_dir:  # keep host RAM low (user cgroup limit): one bf16 logits file per window on local disk
            os.makedirs(cache_dir, exist_ok=True)
            paths = []
            for i in range(n):
                fp = os.path.join(cache_dir, f"{name}_{i}.pt")
                if not os.path.exists(fp):  # reused across runs (same model / ref dtype / windows)
                    torch.save(m(enc[:, i * seqlen:(i + 1) * seqlen].to(dev)).logits[0].to(torch.bfloat16).cpu(), fp)
                paths.append(fp)
            res[name] = (enc, n, paths)
        else:
            res[name] = (enc, n, [m(enc[:, i * seqlen:(i + 1) * seqlen].to(dev)).logits[0].to(torch.bfloat16).cpu()
                                  for i in range(n)])
    return res


@torch.no_grad()
def kl_eval(q, refs, dev, seqlen=512):
    out = {}
    for name, (enc, n, rls) in refs.items():
        acc = {k: [] for k in ("kl", "top1", "nll_q", "nll_ref")}
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            lq = q(ids).logits.float()[0][:-1]
            lr = (torch.load(rls[i]) if isinstance(rls[i], str) else rls[i]).to(dev).float()[:-1]
            y = ids[0, 1:]
            lpr, lpq = lr.log_softmax(-1), lq.log_softmax(-1)
            acc["kl"].append((lpr.exp() * (lpr - lpq)).sum(-1).cpu())
            acc["top1"].append((lr.argmax(-1) == lq.argmax(-1)).float().cpu())
            acc["nll_q"].append(F.cross_entropy(lq, y, reduction="none").cpu())
            acc["nll_ref"].append(F.cross_entropy(lr, y, reduction="none").cpu())
        c = {k: torch.cat(v) for k, v in acc.items()}
        out[name] = {"n_windows": n, "n_tokens": int(c["kl"].numel()), "kl_ref_q": c["kl"].mean().item(),
                     "top1_ref_q": c["top1"].mean().item(), "ppl_q": torch.exp(c["nll_q"].mean()).item(),
                     "ppl_ref": torch.exp(c["nll_ref"].mean()).item()}
        print(f"[eval] kl {name}: {out[name]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--marlin-ckpt", default=None)
    ap.add_argument("--steps", nargs="+", default=["ppl:bf16", "ppl:fp16", "ppl:marlin", "kl:marlin"])
    ap.add_argument("--kl-ref", default="bf16")
    ap.add_argument("--n_windows", type=int, default=40)
    ap.add_argument("--ref-cache-dir", default=None, help="store reference logits on disk (bf16, per window)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res.update({"model": a.model, "marlin_ckpt": a.marlin_ckpt, "kl_ref": a.kl_ref})
    t0 = time.time()
    refs = None
    for st in a.steps:
        what, kind = st.split(":")
        if what == "kl" and refs is None:
            m = load(a.model, a.kl_ref, a.marlin_ckpt, dev)
            if a.ref_cache_dir:
                a.ref_cache_dir = os.path.join(a.ref_cache_dir, f"{os.path.basename(a.model)}_{a.kl_ref}_w{a.n_windows}")
            refs = ref_logits(m, a.model, dev, a.n_windows, cache_dir=a.ref_cache_dir)
            del m
            torch.cuda.empty_cache()
        m = load(a.model, kind, a.marlin_ckpt, dev)
        if what == "ppl":
            res[f"ppl_4096_{kind}"] = ppl_eval(m, a.model, dev)
        elif what == "kl":
            res[f"kl_{a.kl_ref}_vs_{kind}"] = kl_eval(m, refs, dev)
        del m
        torch.cuda.empty_cache()
        res["wall_s"] = time.time() - t0
        json.dump(res, open(a.out, "w"), indent=1)  # after every step
        print(f"[eval] step {st} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
