"""4-bit (and 3-bit) draft codebook formats (rounding only), Qwen no-FT screening ckpts; companion of table_qcmp.py.
Rows: q16 (screening bf16 evals), int8/int4 (qcmp evals, /data/lsh/results_accept/qcmp) and the new LUT formats
(run_fmt4.sh evals, /data/lsh/results_accept/qcmp_fmt4/<cfg>_kd<k>/eval/{A,B}_<fmt>.json). Also reports the codebook
rel-MSE and per-plane kurtosis from the quantize_draft meta json, and Delta vs int4 of the same (cfg, kd, draft).
Usage: python experiments/11_qcmp/table_fmt4.py [--md] [--save RUN]
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_qcmp import REPO, SCREEN, QCMP, cfg_info, row

FMT4 = "/data/lsh/results_accept/qcmp_fmt4"
FMTS = ["bf16", "int8", "int4", "nf4", "fp4_e2m1", "posit4_es0", "posit4_es1", "int3", "nf3"]
QB = {"bf16": 16, "int8": 8, "int4": 4, "nf4": 4, "fp4_e2m1": 4, "posit4_es0": 4, "posit4_es1": 4, "int3": 3, "nf3": 3}


def collect():
    info = cfg_info(); rows = []
    for d in sorted(glob.glob(f"{FMT4}/*_kd*")):
        cfg, kd = os.path.basename(d).rsplit("_kd", 1); kd = int(kd)
        for var in ("A", "B"):
            for fmt in FMTS:
                src = "qcmp_fmt4"
                p = f"{d}/eval/{var}_{fmt}.json"
                if not os.path.exists(p):
                    src = {"bf16": "screen"}.get(fmt, "qcmp")
                    p = (f"{SCREEN}/{cfg}_noft/kd{kd}/eval/{var}_bf16.json" if fmt == "bf16"
                         else f"{QCMP}/{cfg}_kd{kd}/eval/{var}_{fmt}.json")
                if not os.path.exists(p):
                    continue
                x = row(p, cfg, kd, QB[fmt], var, info[cfg]); x["fmt"] = fmt; x["source"] = src
                mp = p.replace("/eval/", "/draft_q/").replace(".json", ".meta.json")
                if os.path.exists(mp):
                    m = json.load(open(mp))
                    x["cb_rel_mse"] = m["codebook_rel_mse_mean"]
                    if "plane_kurtosis" in m:
                        x["plane_kurt_median"] = m["plane_kurtosis"]["median"]
                rows.append(x)
    base = {(x["cfg"], x["kd"], x["draft"]): x for x in rows if x["fmt"] == "int4"}
    for x in rows:
        b = base.get((x["cfg"], x["kd"], x["draft"]))
        if b:
            x["d_wt2_tv_vs_int4"] = x["wikitext2_tv"] - b["wikitext2_tv"]
            x["d_tau_instruct_g4_vs_int4"] = x["tau_instruct_g4"] - b["tau_instruct_g4"]
            x["d_tau_wt2_g4_vs_int4"] = x["tau_wikitext2_g4"] - b["tau_wikitext2_g4"]
    return rows


if __name__ == "__main__":
    rows = collect()
    hdr = ["cfg", "kd", "draft", "fmt", "q", "source", "cb_rel_mse", "plane_kurt_median", "wikitext2_tv", "c4_tv",
           "wikitext2_top1", "wikitext2_ppl_draft", "tau_instruct_g1", "tau_instruct_g2", "tau_instruct_g4",
           "tau_wikitext2_g1", "tau_wikitext2_g2", "tau_wikitext2_g4", "d_wt2_tv_vs_int4", "d_tau_instruct_g4_vs_int4",
           "d_tau_wt2_g4_vs_int4"]
    f = lambda v: "" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    sep, pre, post = (" | ", "| ", " |") if "--md" in sys.argv else ("\t", "", "")
    print(pre + sep.join(hdr) + post)
    if "--md" in sys.argv:
        print("|" + "---|" * len(hdr))
    for x in rows:
        print(pre + sep.join(f(x.get(h)) for h in hdr) + post)
    if "--save" in sys.argv:
        sys.path.insert(0, REPO)
        from tools.results import save_result
        run = sys.argv[sys.argv.index("--save") + 1]
        cfg = {"model": "Qwen2.5-0.5B-Instruct", "setting": "no-FT screening ckpts (results/E5/qwen05_screen_*_noft.json); "
               "draft-only codebook copies rounded (no retraining) to 4-bit (and 3-bit) LUT formats; target q16",
               "rounding": "experiments/06_accept/quantize_draft.py --gran codebook --fmt F: one fp16 scale per (linear, "
               "draft plane) x level set. int4 +-7 / int3 +-3: clip ratio [0.25,1] x max|v|/qmax (qcmp, unchanged). "
               "nf4/nf3 (QLoRA NormalFloat), fp4_e2m1 +-{0,.5,1,1.5,2,3,4,6}, posit4_es0 +-{0,1/4,1/2,3/4,1,3/2,2,4}, "
               "posit4_es1 +-{0,1/16,1/4,1/2,1,2,4,16}: max|v| -> 2^-6..2^0.5 x largest level, 32 steps/octave, MSE",
               "draft": "A = target-codebook copy; B = TV-trained draft copy from screening (train_B_bf16), rounded the same way",
               "eval": "draft_target_eval.py kt=L, 40x512 wt2 + 40x512 c4; greedy 8 chat + 8 wt2 prompts x 128 tokens",
               "sources": "fmt rows from /data/lsh/results_accept/qcmp_fmt4 (source=qcmp_fmt4); bf16 from screening, "
               "int8/int4 from qcmp unless re-run in qcmp_fmt4 (C1 kd3 int4 re-run = reproduction check)",
               "scripts": ["experiments/11_qcmp/run_fmt4.sh", "experiments/11_qcmp/table_fmt4.py"]}
        print(save_result("E5", run, cfg, {"rows": rows}, model_dir="/data/models/qwen2.5_0.5b_instruct"))
