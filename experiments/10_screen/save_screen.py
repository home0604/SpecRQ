"""Collect one E5 screening config (run_screen.sh) into results/E5/qwen05_screen_<cfg>_noft.json and print the
draft-bpw table row(s). bpw accounting (Qwen2.5-0.5B linears, all 24 layers identical):
  index bpw(k) = k * nbits / d ; with codebooks: + (k * K * d * 16 bit per linear + d_out * 16 bit scales) / #weights
(matches DbD's 'curent_avg_bits' at k = L).
Usage: python save_screen.py --cfg C --d D --nbits B --L L --ckpt DIR --out DIR --log LOG --kds "1 2"
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "07_kd12"))
sys.path.insert(0, os.path.join(HERE, "..", "09_c2_timing"))
from tools.results import save_result
from save_kd12 import parse_ppl, suite_rows
from parse_timing import parse as parse_timing

H, I, KV = 896, 4864, 128
LINEARS = [(H, H), (KV, H), (KV, H), (H, H), (I, H), (I, H), (H, I)]  # (d_out, d_in): q k v o gate up down


def bpw(k, d, nbits):
    """nbits: int (all stages) or list of per-stage bits"""
    bits = [nbits] * k if isinstance(nbits, int) else list(nbits)[:k]
    n = sum(o * i for o, i in LINEARS)
    cb = len(LINEARS) * sum(2 ** b for b in bits) * d * 16
    sc = sum(o for o, _ in LINEARS) * 16
    return {"index_bpw": sum(bits) / d, "bpw_with_codebooks_scales": sum(bits) / d + (cb + sc) / n}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--cfg", "--ckpt", "--out", "--log", "--kds"):
        ap.add_argument(a, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--nbits", required=True, help="int or per-stage list b1,b2,..")
    ap.add_argument("--L", type=int, required=True)
    a = ap.parse_args()
    nb = [int(x) for x in a.nbits.split(",")]
    a.nbits = nb[0] if len(nb) == 1 else nb
    m = parse_ppl([a.log])
    m["target_bpw"] = bpw(a.L, a.d, a.nbits)
    lay = parse_timing(a.log)
    m["layer_time_s"] = [x["layer_time_s"] for x in lay]
    m["out_loss"] = [x["out_loss"] for x in lay]
    m["out_val_loss"] = [x["out_val_loss"] for x in lay]
    m["kd"] = {}
    table = []
    for kd in a.kds.split():
        rows = suite_rows(f"{a.out}/kd{kd}")
        m["kd"][kd] = {"bpw": bpw(int(kd), a.d, a.nbits), **rows}
        for v, r in rows.items():
            table.append({"cfg": a.cfg, "kd": int(kd), "draft": v[0], **m["kd"][kd]["bpw"],
                          "wt2_tv": r["wikitext2"]["tv_mean"], "wt2_top1": r["wikitext2"]["top1_agree"],
                          "c4_tv": r["c4"]["tv_mean"], "c4_top1": r["c4"]["top1_agree"],
                          **{f"tau_{kind}_g{g}": r["greedy"][kind][f"g{g}_tok_per_iter"]
                             for kind in ("instruct", "wikitext2") for g in (1, 2, 4)},
                          "ppl4096_wt2_k": m["ppl_4096"].get(f"wikitext2_k{kd}")})
            if "kl_fp16_target" in r["wikitext2"]:
                m["target_quality"] = {ds: {k: r[ds][k] for k in ("ppl_target", "ppl_fp16", "kl_fp16_target", "top1_fp16_target")}
                                       for ds in ("wikitext2", "c4")}
    m["table"] = sorted(table, key=lambda x: (x["index_bpw"], x["draft"]))
    cfg = {"model": "Qwen2.5-0.5B-Instruct", "cfg": a.cfg, "d": a.d, "K": 2 ** a.nbits if isinstance(a.nbits, int) else [2 ** b for b in a.nbits], "L": a.L, "ckpt": a.ckpt,
           "setting": f"Hadamard, uniform alpha 1/{a.L}, wikitext2 1024x4096 calib, accum fp64, AQ max_epochs 1000 / rel_tol 0.01, "
                      "NO block fine-tune (--finetune_max_epochs 0)",
           "draft": "A = target-copy codebooks; B = draft-only codebooks+scales TV-trained (pile_val_heldout 1024x512, 300 steps)",
           "eval": "draft_target_eval.py kt=L: 40x512 wt2 + 40x512 c4 windows; greedy 8 chat + 8 wt2 prompts x 128 tokens",
           "log": a.log, "scripts": ["experiments/10_screen/run_screen.sh", "experiments/10_screen/save_screen.py"]}
    p = save_result("E5", f"qwen05_screen_{a.cfg}_noft", cfg, m, model_dir="/data/models/qwen2.5_0.5b_instruct")
    print(p)
    for t in m["table"]:
        print(json.dumps(t))


if __name__ == "__main__":
    main()
