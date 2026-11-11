"""Stage-by-stage check of the RQ wrapper (experiments/16_rq/rq_main.py) on single linears of layer 0.
For each linear: H = XX^T of its layer-0 inputs (bf16 model, NCAL wikitext2 train windows; at layer 0 these are exactly
the pipeline's inputs), Hadamard rotation as in main.py (layer_seed(0, 0, name)). Then, with the same code paths as
the RQ run, record the weighted relative error  e(prefix) = tr(dW H dW^T) / tr(W H W^T)  of the prefix 1..s:
  (a) RQ after the k-means init of stage s, (b) RQ after the stage-s optimization (Adam + beam),
  (c) RQw (residual k-means init of all stages, the AQ init) at the same prefix.
Expect b <= a for every s, and b <= c at s = 1.
Usage: python experiments/16_rq/diag_stagewise.py [--d 32 --L 8] [--linears q_proj,down_proj] [--ncal 64]
"""
import argparse, json, os, random, sys, time
from argparse import Namespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoModelForCausalLM
from src.datautils import get_loaders
from src.hadamard import layer_seed
import aq_engine
import rq_main


def werr(eng, k):
    """weighted relative error of prefix k, in the engine's (rotated) space"""
    with torch.no_grad():
        ref = eng._reference_weight().to(torch.float64)
        dw = eng.quantized_weight(num_codebooks=k).to(torch.float64) - ref
        H = eng.XTX.to(torch.float64)
        return ((dw @ H) * dw).sum().item() / ((ref @ H) * ref).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--linears", default="self_attn.q_proj,self_attn.k_proj,mlp.down_proj")
    ap.add_argument("--ncal", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda:0")
    torch.manual_seed(0); random.seed(0)
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev).eval()
    layer = m.model.layers[0]
    names = a.linears.split(",")
    engines = {n: aq_engine.AQEngine(layer.get_submodule(n), accumulator_dtype=torch.float64) for n in names}
    hooks = [layer.get_submodule(n).register_forward_hook(lambda mod, inp, out, n=n: engines[n].add_batch(inp[0].data))
             for n in names]
    data = get_loaders("wikitext2", nsamples=a.ncal, seed=0, seqlen=4096, model_path=a.model)
    with torch.no_grad():
        for b in data:
            b = b[0] if isinstance(b, (list, tuple)) else b
            m.model.layers[0](m.model.embed_tokens(b.to(dev)), position_embeddings=m.model.rotary_emb(
                m.model.embed_tokens(b.to(dev)), torch.arange(b.shape[1], device=dev)[None]))
    for h in hooks:
        h.remove()
    args = Namespace(devices=[dev], out_group_size=1, in_group_size=a.d, num_codebooks=a.L, nbits_per_codebook=a.nbits,
                     codebook_value_nbits=16, codebook_value_num_groups=1, scale_nbits=0, init_max_iter=100,
                     init_max_points_per_centroid=None, lr=1e-4, steps_per_epoch=100, max_epochs=a.max_epochs,
                     relative_mse_tolerance=0.01, beam_size=1, print_frequency=10**9, codebook_bits=None,
                     codebook_weights=None)
    res = {}
    for n in names:
        eng = engines[n]
        eng.layer.weight.data = eng.layer.weight.data.float()
        eng.hadamard_seed = layer_seed(0, 0, n)
        H0 = eng.XTX.clone()
        # (c) RQw: AQ init only
        random.seed(0); torch.manual_seed(0)
        rq_main._setup(eng, args)
        c = [werr(eng, k) for k in range(1, a.L + 1)]
        # (a)/(b) RQ act, instrumented copy of rq_main.rq_quantize's stage loop
        eng.XTX.copy_(H0); eng.rotated_weight = None
        random.seed(0); torch.manual_seed(0)
        rq_main._setup(eng, args)
        qw = eng.quantized_weight
        A, B, losses = [], [], []
        for s in range(a.L):
            if s > 0:
                rq_main._reinit_stage(eng, args, s)
            A.append(werr(eng, s + 1))
            eng.codebook_weights = tuple(1.0 if j == s else 0.0 for j in range(a.L))
            params = [qw.codebooks]
            mask = torch.zeros_like(qw.codebooks); mask[s] = 1
            cb_before = qw.codebooks.detach().clone()
            hook = qw.codebooks.register_hook(lambda g, mm=mask: g * mm)
            if s == 0:
                params.append(qw.scales)
            else:
                qw.scales.requires_grad_(False)
            sc_before = qw.scales.detach().clone()
            codes_before = qw.codes.detach().clone()
            opt = torch.optim.Adam(params, lr=args.lr, betas=(0.0, 0.95), amsgrad=True)
            prev = float("inf"); hist = []
            with torch.enable_grad():
                for epoch in range(args.max_epochs):
                    stop = False
                    for step in range(args.steps_per_epoch):
                        loss = eng._compute_mse()
                        if step == 0:
                            hist.append(loss.item())
                            if loss.item() / prev > 0.99:
                                stop = True; break
                            prev = min(prev, loss.item())
                        opt.zero_grad(); loss.backward(); opt.step()
                    if stop:
                        break
                    rq_main._beam_stage(eng, args, s, random.getrandbits(256))
            hook.remove()
            B.append(werr(eng, s + 1))
            other = [j for j in range(a.L) if j != s]
            losses.append({"stage": s, "epoch_start_losses": hist[:3] + hist[-2:], "n_epochs": len(hist),
                           "other_codebooks_changed": bool((qw.codebooks.detach()[other] - cb_before[other]).abs().max() > 0),
                           "other_codes_changed": bool((qw.codes[..., [j for j in other if j < s]] != codes_before[..., [j for j in other if j < s]]).any()) if s > 0 else False,
                           "scales_changed": bool((qw.scales.detach() - sc_before).abs().max() > 0)})
        res[n] = {"a_after_init": A, "b_after_opt": B, "c_RQw": c, "stage_info": losses}
        print(n, json.dumps({k: [round(x, 5) for x in v] for k, v in res[n].items() if k != "stage_info"}), flush=True)
        for li in losses:
            print("   ", json.dumps(li), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
