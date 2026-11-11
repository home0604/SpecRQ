"""Collect one claim-1 d-sweep run (run_dsweep.sh) into results/E5/qwen05_dsweep_<cfg>_<obj>.json.
Metrics: PPL@4096 wt2/c4 for every prefix k (main.py log), target bpw (index; + codebooks + scales, same accounting as
experiments/10_screen/save_screen.py), KL(FP16||target) and top-1 agreement (draft_target_eval kt=L, 40x512 windows),
per-layer time and losses, peak RSS / wall time of the training process (/usr/bin/time -v).
Usage: python save_dsweep.py --cfg C --d D --nbits B --L L --obj T|N --cbw "a1 .." --ckpt DIR --eval JSON --log LOG
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "07_kd12"))
sys.path.insert(0, os.path.join(HERE, "..", "09_c2_timing"))
sys.path.insert(0, os.path.join(HERE, "..", "10_screen"))
from tools.results import save_result
from save_kd12 import parse_ppl, suite_rows
from parse_timing import parse as parse_timing
from save_screen import bpw


LLAMA31_8B = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]


def bpw_shapes(k, d, nbits, shapes):
    """save_screen.bpw for other linear shapes: index k*nbits/d + (k*2^nbits*d*16 codebook + d_out*16 scale bits)/#w"""
    n = sum(o * i for o, i in shapes)
    cb = len(shapes) * k * 2 ** nbits * d * 16
    sc = sum(o for o, _ in shapes) * 16
    return {"index_bpw": k * nbits / d, "bpw_with_codebooks_scales": k * nbits / d + (cb + sc) / n}


def time_v(log):
    out = {}
    for line in open(log, errors="replace"):
        if "Maximum resident set size" in line:
            out["train_peak_rss_gib"] = int(line.split(":")[-1]) / 2**20
        elif "Elapsed (wall clock) time" in line:
            t = line.rsplit(" ", 1)[-1].strip().split(":")
            out["train_wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
    return out


def main():
    ap = argparse.ArgumentParser()
    for a in ("--cfg", "--ckpt", "--log", "--eval", "--obj", "--cbw"):
        ap.add_argument(a, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--nbits", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--kds", default="", help="draft prefixes with a copy-draft (A) eval under <kd_root>/kd<kd>/eval")
    ap.add_argument("--kd_root", default=None)
    ap.add_argument("--run_suffix", default="")
    ap.add_argument("--arch", default="qwen05", choices=["qwen05", "llama31"])
    ap.add_argument("--ft_epochs", type=int, default=0)
    ap.add_argument("--calib_note", default="pile_val 192x4096 calib (val 32), accum fp32")
    a = ap.parse_args()
    global bpw
    if a.arch == "llama31":
        bpw = lambda k, d, nb: bpw_shapes(k, d, nb, LLAMA31_8B)
    m = parse_ppl([a.log])
    m.pop("ppl_512", None)
    m["target_bpw"] = bpw(a.L, a.d, a.nbits)
    m["prefix_bpw"] = {str(k): bpw(k, a.d, a.nbits) for k in range(1, a.L + 1)}
    lay = parse_timing(a.log)
    m["layer_time_s"] = [x["layer_time_s"] for x in lay]
    m["out_loss"] = [x["out_loss"] for x in lay]
    m["out_val_loss"] = [x["out_val_loss"] for x in lay]
    m.update(time_v(a.log))
    r = json.load(open(a.eval))
    m["target_quality"] = {ds: {k: r[ds][k] for k in ("ppl_target", "ppl_fp16", "kl_fp16_target", "top1_fp16_target")}
                           for ds in ("wikitext2", "c4")}
    m["eval_json"] = a.eval
    if a.kds.split():  # copy draft A per prefix kd (TV, top-1, greedy tau), same fields as the E5 screening records
        m["kd"], m["table"] = {}, []
        for kd in a.kds.split():
            rows = suite_rows(f"{a.kd_root}/kd{kd}")
            m["kd"][kd] = {"bpw": bpw(int(kd), a.d, a.nbits), **rows}
            for v, rr in rows.items():
                m["table"].append({"cfg": a.cfg, "obj": a.obj, "kd": int(kd), "draft": v[0], **m["kd"][kd]["bpw"],
                                   "wt2_tv": rr["wikitext2"]["tv_mean"], "wt2_top1": rr["wikitext2"]["top1_agree"],
                                   "c4_tv": rr["c4"]["tv_mean"], "c4_top1": rr["c4"]["top1_agree"],
                                   **{f"tau_{kind}_g{g}": rr["greedy"][kind][f"g{g}_tok_per_iter"]
                                      for kind in ("instruct", "wikitext2") for g in (1, 2, 4)},
                                   "ppl4096_wt2_k": m["ppl_4096"].get(f"wikitext2_k{kd}")})
    cfg = {"model": "Qwen2.5-0.5B-Instruct" if a.arch == "qwen05" else "Llama-3.1-8B-Instruct", "cfg": a.cfg, "d": a.d, "K": 2 ** a.nbits, "L": a.L, "objective": a.obj,
           "codebook_weights": [float(x) for x in a.cbw.split()], "ckpt": a.ckpt,
           "setting": f"Hadamard, {'target-only alpha (last stage 1)' if a.obj == 'T' else (f'nested uniform alpha 1/{a.L}' if a.obj == 'N' else ('stage-wise calibration-aware RQ (experiments/16_rq/rq_main.py SPECRQ_RQ=act; scales trained in stage 1 only)' if a.obj == 'RQ' else ('weight-MSE residual k-means RQ (AQ init only, SPECRQ_RQ=wmse)' if a.obj == 'RQw' else 'nested alpha ' + a.cbw)))}, "
                      "wikitext2 1024x4096 calib (val 32), accum fp64, AQ max_epochs 1000 / rel_tol 0.01, beam 1, "
                      "NO block fine-tune (--finetune_max_epochs 0); activations on GPU (no --offload_activations)",
           "eval": "PPL@4096 wt2/c4 (main.py, every prefix k); KL/top-1 = draft_target_eval.py kt=L kd=1: 40x512 wt2 + 40x512 c4 windows"
                   + (f"; copy draft A kd={a.kds}: draft_target_eval kt=L --no_fp16 (TV/top-1 40x512 windows, greedy 8 chat + 8 wt2 prompts x 128 tok)" if a.kds.split() else ""),
           "log": a.log, "scripts": ["experiments/15_dsweep/run_dsweep.sh", "experiments/15_dsweep/save_dsweep.py"]}
    if a.arch == "llama31":
        cfg["setting"] = cfg["setting"].replace("wikitext2 1024x4096 calib (val 32), accum fp64", a.calib_note)
    if a.ft_epochs:
        cfg["setting"] = cfg["setting"].replace("NO block fine-tune (--finetune_max_epochs 0)",
                                                f"block fine-tune {a.ft_epochs} epochs (C1 recipe: batch 32, lr 1e-5, early stop 3, keep best)")
        cfg["ft_epochs"] = a.ft_epochs
    p = save_result("E5", f"{a.arch}_dsweep_{a.cfg}_{a.obj}{a.run_suffix}", cfg, m, model_dir="/data/models/qwen2.5_0.5b_instruct" if a.arch == "qwen05" else "/data/models/llama3.1_8b_instruct")
    print(p)
    print(json.dumps({k: m[k] for k in ("ppl_4096", "target_bpw", "target_quality")}))


if __name__ == "__main__":
    main()
