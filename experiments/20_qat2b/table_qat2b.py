"""E5 qat2b table: rounding-only (fmt2b, A_<label>) vs target-side QAT (qat2b, Q_<label>) per (ckpt, format label).
kd == kt: the "draft" is the target with ALL codebooks stored in the given format; TV/top1/KL(p_T||p_D) are vs the
unmodified bf16-codebook target (same ckpt, codes); ppl_fmt = PPL of the format-stored target, ppl_bf16 = bf16 target.
Reads /data/lsh/results_accept/{fmt2b,qat2b}/<ckpt>_kd<kd>/eval/{A,Q}_<label>.json and qat2b train_<label>/train_log.json.
Usage: python experiments/20_qat2b/table_qat2b.py [--md] [--save RUN]
"""
import glob, json, os, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = "/data/lsh/results_accept"


def metrics(r):
    x = {}
    for ds in ("wikitext2", "c4"):
        for k, s in (("tv", "tv_mean"), ("top1", "top1_agree"), ("kl", "kl_t_d"), ("ppl_fmt", "ppl_draft"),
                     ("ppl_bf16", "ppl_target")):
            x[f"{ds}_{k}"] = r[ds][s]
    for kind in ("instruct", "wikitext2"):
        x[f"greedy_match_{kind}"] = r["greedy_sim"][kind]["draft_match_rate_on_target_path"]
    return x


def collect():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/qat2b/*_kd*") + glob.glob(f"{ROOT}/qat2b/fixgrid/*_kd*")):
        variant = "fixgrid" if "/fixgrid/" in d else "refresh10"
        cfg, kd = os.path.basename(d).rsplit("_kd", 1); kd = int(kd)
        for p in sorted(glob.glob(f"{d}/eval/Q_*.json")):
            label = os.path.basename(p)[2:-5]
            q = json.load(open(p)); assert q["kd"] == q["kt"] == kd
            a = json.load(open(f"{ROOT}/fmt2b/{cfg}_kd{kd}/eval/A_{label}.json"))
            meta = json.load(open(f"{ROOT}/fmt2b/{cfg}_kd{kd}/draft_q/A_{label}.meta.json"))
            tl = json.load(open(f"{d}/train_{label}/train_log.json")); ta = tl["args"]
            spec = ta["qat_fmt_stages"] or ",".join([ta["qat_fmt"]] * kd)
            assert spec == (meta["args"].get("fmt_stages") or ",".join([meta["args"]["fmt"]] * kd)), (spec, meta["args"])
            log = tl["log"]
            x = {"variant": variant, "ckpt": cfg, "kd": kd, "kt": kd, "label": label, "stage_fmts": spec, "q_mean_bits": meta["q"],
                 "cb_bytes_per_linear": (meta["bits_codebook"] + meta["bits_scales"]) / 8 / meta["n_linears"],
                 "rounding": metrics(a), "qat": metrics(q), "rounding_eval_json": f"{ROOT}/fmt2b/{cfg}_kd{kd}/eval/A_{label}.json",
                 "qat_eval_json": p, "qat_ckpt": f"{d}/train_{label}/draft_codebooks.pt",
                 "train": {"steps": ta["steps"], "batch": ta["batch"], "seqlen": ta["seqlen"], "lr": ta["lr"],
                           "div": ta["div"], "nsamples": ta["nsamples"], "data": ta["data_name"],
                           "qat_refresh": ta["qat_refresh"], "init": ta["init_draft"],
                           "train_tv_first10": sum(l["tv"] for l in log[:10]) / len(log[:10]),
                           "train_tv_last10": sum(l["tv"] for l in log[-10:]) / len(log[-10:]),
                           "wall_s": log[-1]["t"], "peak_mem_gib": tl["peak_mem_gib"],
                           "teacher_check": tl.get("teacher_check")},
                 "eval_wall_s": q.get("wall_s")}
            rows.append(x)
    return rows


if __name__ == "__main__":
    rows = collect()
    f = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    keys = ["wikitext2_tv", "c4_tv", "wikitext2_top1", "c4_top1", "wikitext2_kl", "c4_kl", "wikitext2_ppl_fmt", "c4_ppl_fmt"]
    hdr = ["variant", "ckpt", "label"] + [f"{k}_{m}" for k in keys for m in ("rnd", "qat")] + ["wikitext2_ppl_bf16", "c4_ppl_bf16", "train_wall_s"]
    sep, pre, post = (" | ", "| ", " |") if "--md" in sys.argv else ("\t", "", "")
    print(pre + sep.join(hdr) + post)
    if "--md" in sys.argv:
        print("|" + "---|" * len(hdr))
    for x in rows:
        v = [x["variant"], x["ckpt"], x["label"]] + [x[m][k] for k in keys for m in ("rounding", "qat")] + \
            [x["qat"]["wikitext2_ppl_bf16"], x["qat"]["c4_ppl_bf16"], x["train"]["wall_s"]]
        print(pre + sep.join(f(t) for t in v) + post)
    if "--save" in sys.argv:
        sys.path.insert(0, REPO)
        from tools.results import save_result
        run = sys.argv[sys.argv.index("--save") + 1]
        cfg = {"model": "Llama-3.1-8B-Instruct", "exp": "E5 qat2b",
               "setting": "target-side (kd == kt) codebook VALUE-format QAT on 2-bit nested d-sweep ckpts; weight index "
               "bits unchanged. Student = copy of all kt codebooks with STE fake-quant at stage_fmts (one fp16 scale per "
               "codebook), init = fmt2b rounding-only copy. variant refresh10: fake-quant scales re-searched (MSE) every 10 "
               "steps and once more before baking (--qat_refresh 10, script default); variant fixgrid: searched once at "
               "init (= the rounding-only grid) and kept through training and bake (--qat_refresh 0). refresh10 bakes can "
               "land on a re-searched grid the latent was not trained on (d8 mix_int4-int8: train-window TV 0.126 after "
               "bake vs ~0.074 during training, /data/lsh/results_accept/qat2b/diag); per-row scales "
               "trainable (stored fp32 in the copy, cast to bf16 in the forward). Teacher = unmodified bf16-codebook target "
               "(set_k(use_draft=False); bit-exact check vs plain target logits recorded in train.teacher_check). Codes frozen.",
               "train": "experiments/06_accept/train_draft_codebooks.py --div tv --steps 300 --batch 4 --micro_batch 1 "
               "--lr 1e-4 --nsamples 1024 --seqlen 512 (pile_val_heldout, disjoint from calibration and eval) "
               "--codes_uint8 --checkpoint_linears --check_teacher --qat_fmt/--qat_fmt_stages --qat_refresh {10,0} --init_draft "
               "A_<label>.pt. Budget (steps, batch, seqlen, data, loss) = Llama B-draft runs (results_accept/llama31_c1_"
               "uniform_kd*/train_B_bf16); lr 1e-5 instead of their 1e-4 (40-step probe on train-batch TV, qat2b/lrprobe: "
               "1e-4 raises TV from the rounding init over the first ~20 steps)",
               "eval": "draft_target_eval.py --kd L --kt L --n_windows 40 --codes_uint8 --no_fp16 (fmt2b flags); tv/top1/kl "
               "= vs the bf16 target; ppl_fmt = PPL of the format-stored target",
               "baseline": "rounding = results/E5/llama31_2bit_fmt_sweep.json rows (same A_<label> eval jsons)",
               "scripts": ["experiments/20_qat2b/run_qat2b_q.sh", "experiments/20_qat2b/queue.txt", "experiments/20_qat2b/queue_fixgrid.txt",
                           "experiments/20_qat2b/table_qat2b.py", "experiments/06_accept/train_draft_codebooks.py",
                           "specrq/dbd/src/draft.py", "experiments/06_accept/draft_target_eval.py"]}
        print(save_result("E5", run, cfg, {"rows": rows}, model_dir="/data/models/llama3.1_8b_instruct"))
