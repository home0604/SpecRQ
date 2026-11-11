"""Collect the E5 scale check (run_scale.sh) for Qwen2.5-32B-Instruct vs Qwen2.5-0.5B-Instruct into
results/E5/qwen32b_vs_05_L2_nested.json and print the comparison table.
Usage: python save_scale.py --run qwen32b_vs_05_L2_nested --m TAG:MODEL_DIR:JOBLOG [--m ...] [--ns 512 --seq 4096]
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "07_kd12"))
sys.path.insert(0, os.path.join(HERE, "..", "09_c2_timing"))
from tools.results import save_result
from save_kd12 import parse_ppl, suite_rows
from parse_timing import parse as parse_timing


def linears(cfg):
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    KV = cfg["num_key_value_heads"] * (H // cfg["num_attention_heads"])
    return [(H, H), (KV, H), (KV, H), (H, H), (I, H), (I, H), (H, I)], cfg["num_hidden_layers"]


def bpw(k, cfg, d=8, nbits=8):
    lin, nl = linears(cfg)
    n = sum(o * i for o, i in lin)
    cb = len(lin) * k * 2 ** nbits * d * 16
    sc = sum(o for o, _ in lin) * 16
    return {"index_bpw": k * nbits / d, "bpw_with_codebooks_scales": k * nbits / d + (cb + sc) / n,
            "quantized_linear_params": n * nl}


def rss_gb(log):
    v = [int(m.group(1)) for m in re.finditer(r"Maximum resident set size \(kbytes\): (\d+)", open(log, errors="replace").read())]
    return max(v) / 1e6 if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="qwen32b_vs_05_L2_nested")
    ap.add_argument("--m", action="append", required=True, help="TAG:MODEL_DIR:JOBLOG[,JOBLOG2..]")
    ap.add_argument("--ns", type=int, default=512)
    ap.add_argument("--seq", type=int, default=4096)
    a = ap.parse_args()
    M, table = {}, []
    for spec in a.m:
        tag, mdir, logs = spec.split(":")
        logs = logs.split(",")
        cfg = json.load(open(os.path.join(mdir, "config.json")))
        ckpt = f"/data/lsh/ckpts/weightvq/scale/{tag}_d8K256L2_noft"
        out = f"/data/lsh/results_accept/scale/{tag}_d8K256L2_noft"
        ppl = parse_ppl(logs + [x for x in (f"{out}/q_ppl.log", f"{out}/fp16_ppl.log") if os.path.exists(x)])["ppl_4096"]
        lay = [x for lg in logs for x in parse_timing(lg)]
        rows = suite_rows(f"{out}/kd1")
        m = {"model_dir": mdir, "ckpt": ckpt, "eval_dir": out, "job_logs": logs,
             "ppl_4096": {"fp16": {"wikitext2": ppl.get("wikitext2_kall"), "c4": ppl.get("c4_kall")},
                          "target_k2": {"wikitext2": ppl.get("wikitext2_k2"), "c4": ppl.get("c4_k2")},
                          "draft_k1": {"wikitext2": ppl.get("wikitext2_k1"), "c4": ppl.get("c4_k1")}},
             "bpw": {"k2": bpw(2, cfg), "k1": bpw(1, cfg)},
             "draft_linear_bytes_ratio_kd1": 0.5,
             "layer_time_s": [x["layer_time_s"] for x in lay], "out_loss": [x["out_loss"] for x in lay],
             "train_peak_rss_gb": max([r for r in map(rss_gb, logs) if r] or [None]) if logs else None,
             "kd1": rows}
        if "A_bf16" in rows:
            m["target_quality_512"] = {ds: {k: rows["A_bf16"][ds].get(k) for k in
                                            ("ppl_fp16", "ppl_target", "kl_fp16_target", "top1_fp16_target")}
                                       for ds in ("wikitext2", "c4")}
        M[tag] = m
        for v, r in rows.items():
            table.append({"model": tag, "draft": v[0], "ppl4096_wt2_fp16": m["ppl_4096"]["fp16"]["wikitext2"],
                          "ppl4096_wt2_k2": m["ppl_4096"]["target_k2"]["wikitext2"],
                          "ppl4096_wt2_k1": m["ppl_4096"]["draft_k1"]["wikitext2"],
                          "kl_fp16_target_wt2": m.get("target_quality_512", {}).get("wikitext2", {}).get("kl_fp16_target"),
                          "wt2_tv": r["wikitext2"]["tv_mean"], "wt2_top1": r["wikitext2"]["top1_agree"],
                          "c4_tv": r["c4"]["tv_mean"], "c4_top1": r["c4"]["top1_agree"],
                          **{f"tau_{kind}_g{g}": r["greedy"][kind][f"g{g}_tok_per_iter"]
                             for kind in ("instruct", "wikitext2") for g in (1, 2, 4)}})
    config = {"recipe": f"d8 K=256 L=2 (target 2 bit/weight, draft kd1 = first codebook, 1 bit/weight index), uniform alpha "
                        f"[0.5,0.5], Hadamard, accum fp32, wikitext2 {a.ns}x{a.seq} calibration (val 32), AQ max_epochs 1000 / "
                        "rel_tol 0.01, NO block fine-tune (--finetune_max_epochs 0)",
              "draft": "A = target-copy codebooks; B = draft-only codebooks+scales TV-trained (pile_val_heldout 1024x512, "
                       "300 steps, batch 4, lr 1e-4)",
              "eval": "PPL @4096 wt2/c4 (DbD main.py); draft_target_eval.py kt=2 kd=1: 40x512 wt2 + 40x512 c4 windows; greedy "
                      "8 chat + 8 wt2 prompts x 128 tokens (greedy simulation, not measured decoding)",
              "cost_note": "kd1 reads 1 of 2 code planes: linear-bytes ratio draft/target = 0.5 (codes only)",
              "scripts": ["experiments/12_scale/run_scale.sh", "experiments/12_scale/save_scale.py"]}
    p = save_result("E5", a.run, config, {"models": M, "table": table}, model_dir=M[list(M)[0]]["model_dir"])
    print(p)
    for t in table:
        print(json.dumps(t))


if __name__ == "__main__":
    main()
