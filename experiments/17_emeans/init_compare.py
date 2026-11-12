"""Layer-level comparison of AQ codebook initializations: torch k-means (AQLM default) vs faiss k-means vs e-means.

Each run = one linear (XTX from collect_xtx.py) quantized with the unchanged AQEngine.quantize of specrq/dbd
(Hadamard, target-only alpha, rel_tol 0.01, lr 1e-4, beam 1, 100 Adam steps/epoch = run_dsweep.sh recipe);
only the clustering inside src.aq.init_aq_kmeans is swapped by monkeypatching (specrq/dbd is not edited):
  torch   : src.kmeans.fit_kmeans (random-point init, <=100 Lloyd iters, allclose stop every 10) = default
  faiss   : init_aq_kmeans(use_faiss=True) -> fit_faiss_kmeans with gpu=False (faiss-cpu 1.16.0 wheel has no GPU),
            faiss.Kmeans(niter=100) defaults (max_points_per_centroid 256 subsampling, seed 1234)
  emeansE : faiss.contrib.e_means.QuantizedEmeans(d, K, quantizer=None, nepochs=E, seed=s) on the GPU in fp32,
            native torch assignment backend; CUDA-graph replay disabled (torch 2.4 has no torch.cuda.MemPool;
            the eager path runs the same update rule). Codes = exact nearest centroid (find_nearest_cluster),
            as for torch k-means.
Metrics per run: init time (cuda-synced, all L stages), init relative weight MSE of the normalized (rotated,
row-scale-divided) weight that k-means fits, init/final AQ loss = tr(dW XX^T dW^T)/out (prefix L), relative to
tr(W XX^T W^T)/out, per-epoch loss trajectory, #epochs to rel_tol stop, total quantize time.

Usage: python experiments/17_emeans/init_compare.py --xtx DIR --out FILE.jsonl --cfgs d8K256L2 --layers 0 12 23
       --names self_attn.q_proj ... --variants torch:0 torch:1 faiss:0 emeans5:0
"""
import argparse
import json
import os
import random
import sys
import time
from argparse import Namespace

DBD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import aq_engine  # noqa: E402
import src.aq as saq  # noqa: E402
import src.kmeans as skm  # noqa: E402
from src.hadamard import layer_seed  # noqa: E402

CFGS = {"d8K256L2": (8, 8, 2), "d32K256L8": (32, 8, 8), "d8K4096L2": (8, 12, 2)}  # d, nbits, L
ORIG_INIT = saq.init_aq_kmeans
ORIG_FIT = saq.fit_kmeans
ORIG_FAISS = saq.fit_faiss_kmeans


def sync():
    torch.cuda.synchronize()


def make_emeans_fit(nepochs, seed):
    from faiss.contrib.e_means import QuantizedEmeans
    QuantizedEmeans._replay_eligible = lambda self, x: False  # no torch.cuda.MemPool in torch 2.4 -> eager steps

    def fit(data, k, max_iter=None, devices=None, **kw):
        x = data.float().contiguous()
        em = QuantizedEmeans(x.shape[1], k, quantizer=None, nepochs=nepochs, seed=seed, assignment_backend="native",
                             capture_assignments=False)
        cent, _ = em.train(x)
        fit.recipes.append({kk: em.recipe.get(kk) for kk in ("batch_size", "alpha0", "steps_per_pass")})
        return cent.float(), None, None
    fit.recipes = []
    return fit


def run_one(rec, cfg, variant, seed, args):
    d, nbits, L = CFGS[cfg]
    W = rec["weight"].cuda()
    out_f, in_f = W.shape
    lin = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=W.dtype)
    with torch.no_grad():
        lin.weight.copy_(W)
    eng = aq_engine.AQEngine(lin)
    eng.XTX.copy_(rec["XTX"].to(eng.XTX.dtype))
    eng.nsamples = rec["nsamples"]
    eng.hadamard_seed = layer_seed(0, rec["layer"], rec["name"])
    qargs = Namespace(
        codebook_weights=[0.0] * (L - 1) + [1.0], devices=[torch.device("cuda:0")], out_group_size=1,
        in_group_size=d, num_codebooks=L, nbits_per_codebook=nbits, codebook_value_nbits=16,
        codebook_value_num_groups=1, scale_nbits=0, init_max_iter=100, init_max_points_per_centroid=None,
        lr=1e-4, max_epochs=args.max_epochs, steps_per_epoch=100, relative_mse_tolerance=0.01, beam_size=1,
        print_frequency=10**9,
    )
    info = {"init_s": None, "init_wmse_rel": None}
    if variant.startswith("emeans"):
        fit = make_emeans_fit(float(variant[len("emeans"):]), seed)
        saq.fit_kmeans = fit
    else:
        fit = None
        saq.fit_kmeans = ORIG_FIT
    saq.fit_faiss_kmeans = lambda data, k, **kw: ORIG_FAISS(
        data, k, max_iter=kw.get("max_iter", 100), gpu=False, max_points_per_centroid=kw.get("max_points_per_centroid"),
        verbose=False)

    def init_wrap(reference_weight, **kw):
        if variant == "faiss":
            kw["use_faiss"] = True
        sync()
        t = time.perf_counter()
        codes, codebooks = ORIG_INIT(reference_weight, **kw)
        sync()
        info["init_s"] = time.perf_counter() - t
        with torch.no_grad():  # reconstruct the normalized weight from codes (out_group_size 1)
            rec_w = torch.zeros_like(reference_weight).view(out_f, in_f // d, d)
            for j in range(codebooks.shape[0]):
                rec_w += codebooks[j, :, 0, :][codes[..., j].long()]
            rec_w = rec_w.view_as(reference_weight)
            info["init_wmse_rel"] = float((rec_w - reference_weight).square().sum() / reference_weight.square().sum())
        return codes, codebooks
    saq.init_aq_kmeans = init_wrap

    losses, n_beam = [], [0]
    orig_mse = aq_engine.AQEngine._compute_mse
    orig_beam = aq_engine.AQEngine.beam_search_update_codes_

    def mse_wrap(self, *a, **kw):
        v = orig_mse(self, *a, **kw)
        losses.append(float(v.detach()))
        return v

    def beam_wrap(self, *a, **kw):
        n_beam[0] += 1
        return orig_beam(self, *a, **kw)
    aq_engine.AQEngine._compute_mse = mse_wrap
    aq_engine.AQEngine.beam_search_update_codes_ = beam_wrap
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        sync()
        t = time.perf_counter()
        eng.quantize(args=qargs, verbose=False)
        sync()
        total_s = time.perf_counter() - t
    finally:
        aq_engine.AQEngine._compute_mse = orig_mse
        aq_engine.AQEngine.beam_search_update_codes_ = orig_beam
        saq.init_aq_kmeans, saq.fit_kmeans, saq.fit_faiss_kmeans = ORIG_INIT, ORIG_FIT, ORIG_FAISS
    with torch.no_grad():  # reference energy tr(W XX^T W^T)/out in the rotated basis (loss normalizer)
        Wr = eng._reference_weight().to(eng.XTX.dtype)
        ref = float((Wr @ eng.XTX).flatten() @ Wr.flatten() / out_f)
    epoch_losses = losses[0::qargs.steps_per_epoch]  # step-0 loss of every epoch (+ the final early-stop check)
    epochs = n_beam[0]
    return {
        "cfg": cfg, "layer": rec["layer"], "name": rec["name"], "variant": variant, "seed": seed,
        "shape": [out_f, in_f], "init_s": info["init_s"], "init_wmse_rel": info["init_wmse_rel"],
        "init_loss": losses[0], "final_loss": epoch_losses[-1], "ref_energy": ref,
        "init_loss_rel": losses[0] / ref, "final_loss_rel": epoch_losses[-1] / ref,
        "epochs": epochs, "stopped_early": epochs < qargs.max_epochs, "total_s": total_s,
        "aq_s": total_s - info["init_s"], "epoch_losses": epoch_losses,
        "emeans_recipe": fit.recipes if fit is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xtx", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cfgs", nargs="+", default=["d8K256L2"])
    p.add_argument("--layers", type=int, nargs="+", default=[0, 12, 23])
    p.add_argument("--names", nargs="+", default=["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj",
                                                  "mlp.up_proj", "mlp.down_proj"])
    p.add_argument("--variants", nargs="+", default=["torch:0", "torch:1", "faiss:0", "emeans5:0", "emeans5:1"])
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--faiss_threads", type=int, default=8)
    a = p.parse_args()
    import faiss
    faiss.omp_set_num_threads(a.faiss_threads)
    print(f"faiss {faiss.__version__} {faiss.__file__} torch {torch.__version__}", flush=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["cfg"], r["layer"], r["name"], r["variant"], r["seed"]))
    for cfg in a.cfgs:
        for li in a.layers:
            for name in a.names:
                rec = torch.load(os.path.join(a.xtx, f"l{li}_{name}.pt"))
                for v in a.variants:
                    variant, seed = v.split(":")
                    seed = int(seed)
                    if (cfg, li, name, variant, seed) in done:
                        continue
                    r = run_one(rec, cfg, variant, seed, a)
                    print(f"[cmp] {cfg} l{li} {name} {variant}:{seed} init {r['init_s']:.2f}s wmse {r['init_wmse_rel']:.4f}"
                          f" loss_rel init {r['init_loss_rel']:.5f} final {r['final_loss_rel']:.5f} epochs {r['epochs']}"
                          f" total {r['total_s']:.1f}s", flush=True)
                    with open(a.out, "a") as f:
                        f.write(json.dumps(r) + "\n")
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
