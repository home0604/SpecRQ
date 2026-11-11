"""Diagnostic (not a method): is the RQ target loss due to the per-row scales frozen after stage 1?
For a finished checkpoint, keep codes and codebooks fixed and refit only the per-row scale of every QuantizedLinear
by activation-weighted least squares: for row i, alpha_i = (w^_i H w_i^T) / (w^_i H w^_i^T), scale_i *= alpha_i, where
w^ is the current dequantized weight (original, un-rotated space), w the original weight and H = sum x x^T of the layer
inputs of the *original* bf16 model on NCAL wikitext2 train windows (an approximation of the sequential pipeline's H,
which uses quantized-model inputs). Same storage format (only scales change). Reports PPL@4096 wt2/c4 at k=L
before/after (same formula as specrq/dbd main.py perplexity_eval) and the weighted relative error per linear.
Usage: python experiments/16_rq/diag_scale_refit.py --ckpt DIR [--ckpt DIR ...] [--ncal 128]
"""
import argparse, json, math, os, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from src.aq import QuantizedLinear
from src.datautils import get_loaders
from src.modelutils import get_model


@torch.no_grad()
def ppl(m, model_path, dev, seqlen=4096):
    out = {}
    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=model_path, seqlen=seqlen, eval_mode=True)
        n = enc.numel() // seqlen
        nll = []
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
            lg = m.lm_head(m.model(ids)[0])[0, :-1]
            nll.append(F.cross_entropy(lg, ids[0, 1:]).float() * seqlen)
        out[name] = torch.exp(torch.stack(nll).sum() / (n * seqlen)).item()
    return out


@torch.no_grad()
def collect_H(ref, model_path, dev, ncal, seqlen=4096, name="wikitext2"):
    data = get_loaders(name, nsamples=ncal, seed=0, seqlen=seqlen, model_path=model_path)
    H, hooks = {}, []
    for name, mod in ref.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            H[name] = torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64, device=dev)

            def f(_, inp, __, n=name):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                H[n] += x.T @ x
            hooks.append(mod.register_forward_hook(f))
    for b in data:
        b = b[0] if isinstance(b, (list, tuple)) else b
        ref(b.to(dev))
    for h in hooks:
        h.remove()
    return H


@torch.no_grad()
def dense_weight(ql, dev):
    eye = torch.eye(ql.quantized_weight.in_features, device=dev, dtype=torch.float32)
    k = ql.active_num_codebooks
    ql.active_num_codebooks = None
    m32 = ql.to(torch.float32)
    w = (m32(eye) - m32(torch.zeros_like(eye[:1]))).T.contiguous()  # [out, in], original input space, bias removed
    ql.active_num_codebooks = k
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--ncal", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hdata", default="wikitext2", help="calibration set for H (wikitext2 train | c4 train)")
    ap.add_argument("--no_eval", action="store_true", help="only the weighted relative errors (no PPL)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    t0 = time.time()
    ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    H = collect_H(ref, a.model, dev, a.ncal, name=a.hdata)
    Wref = {n: m.weight.detach().float() for n, m in ref.named_modules() if n in H}
    print(f"H collected ({len(H)} linears, {a.ncal}x4096) {time.time() - t0:.0f}s", flush=True)
    res = {}
    for ck in a.ckpt:
        q = get_model(a.model, ck, "auto").to(dev).eval()
        before = None if a.no_eval else ppl(q, a.model, dev)
        print(ck, "before", before, flush=True)
        errs, alphas = {}, []
        for name, mod in q.named_modules():
            if not isinstance(mod, QuantizedLinear):
                continue
            Hn, w = H[name], Wref[name].double()
            dt = mod.quantized_weight.codebooks.dtype
            what = dense_weight(mod, dev).double()
            mod.to(dt)
            num = ((what @ Hn) * w).sum(1)
            den = ((what @ Hn) * what).sum(1).clamp_min(1e-30)
            alpha = num / den
            e0 = ((what - w) @ Hn * (what - w)).sum() / (w @ Hn * w).sum()
            what2 = what * alpha[:, None]
            e1 = ((what2 - w) @ Hn * (what2 - w)).sum() / (w @ Hn * w).sum()
            errs[name] = (e0.item(), e1.item())
            alphas.append(alpha.float().cpu())
            sc = mod.quantized_weight.scales
            sc.data.mul_(alpha.view(-1, 1, 1, 1).to(sc.dtype))
        after = None if a.no_eval else ppl(q, a.model, dev)
        al = torch.cat(alphas)
        e = torch.tensor(list(errs.values()))
        r = {"ppl_before": before, "ppl_after_scale_refit": after,
             "alpha": {"mean": al.mean().item(), "std": al.std().item(), "p05": al.quantile(.05).item(), "p95": al.quantile(.95).item()},
             "rel_weighted_err_mean_before": e[:, 0].mean().item(), "rel_weighted_err_mean_after": e[:, 1].mean().item()}
        print(ck, json.dumps(r), flush=True)
        res[ck] = r
        del q
        torch.cuda.empty_cache()
    if a.out:
        json.dump({"ncal": a.ncal, "results": res, "wall_s": time.time() - t0}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
