"""Draft codebook precision comparison (q = 16/8/4) on the Qwen no-FT screening ckpts.
q=16 rows = screening evals (/data/lsh/results_accept/screen/<cfg>_noft/kd<k>/eval/{A,B}_bf16.json);
q=8/4 rows = run_qcmp.sh evals (/data/lsh/results_accept/qcmp/<cfg>_kd<k>/eval/{A,B}_int{8,4}.json).
Axes: draft index bpw = sum_{j<kd} log2 K_j / d; draft codebook bytes per linear = sum_{j<kd} K_j * d * q/8
(+ 2 bytes fp16 scale per plane for q<16) -- the on-chip (smem) footprint that a prefix-kd draft kernel must hold.
Usage: python experiments/11_qcmp/table_qcmp.py [--md] [--save RUN]
"""
import glob, json, math, os, sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
SCREEN = "/data/lsh/results_accept/screen"
QCMP = "/data/lsh/results_accept/qcmp"


def cfg_info():
    out = {}
    for p in glob.glob(os.path.join(REPO, "results/E5/qwen05_screen_*_noft.json")):
        c = json.load(open(p))["config"]
        K = c["K"] if isinstance(c["K"], list) else [c["K"]] * c["L"]
        out[c["cfg"]] = {"d": c["d"], "K": K, "L": c["L"]}
    return out


def row(path, cfg, kd, q, var, info):
    r = json.load(open(path)); g = r["greedy_sim"]
    K = info["K"][:kd]; d = info["d"]
    x = {"cfg": cfg, "kd": kd, "q": q, "draft": var, "K_stages": K, "d": d,
         "index_bpw": sum(math.log2(k) for k in K) / d,
         "cb_bytes_per_linear": sum(k * d * q / 8 for k in K) + (2 * len(K) if q < 16 else 0),
         "eval_json": path}
    for ds in ("wikitext2", "c4"):
        x[f"{ds}_tv"] = r[ds]["tv_mean"]; x[f"{ds}_top1"] = r[ds]["top1_agree"]; x[f"{ds}_ppl_draft"] = r[ds]["ppl_draft"]
    for kind in ("instruct", "wikitext2"):
        for gm in (1, 2, 4):
            x[f"tau_{kind}_g{gm}"] = g[kind][f"gamma{gm}"]["tokens_per_iter"]
    return x


def collect():
    info = cfg_info(); rows = []
    for d in sorted(glob.glob(f"{QCMP}/*_kd*")):
        name = os.path.basename(d); cfg, kd = name.rsplit("_kd", 1); kd = int(kd)
        for var in ("A", "B"):
            p16 = f"{SCREEN}/{cfg}_noft/kd{kd}/eval/{var}_bf16.json"
            if os.path.exists(p16):
                rows.append(row(p16, cfg, kd, 16, var, info[cfg]))
            for q in (8, 4):
                p = f"{d}/eval/{var}_int{q}.json"
                if os.path.exists(p):
                    rows.append(row(p, cfg, kd, q, var, info[cfg]))
    rows.sort(key=lambda x: (x["index_bpw"], x["cb_bytes_per_linear"], x["cfg"], x["draft"]))
    return rows


if __name__ == "__main__":
    rows = collect()
    hdr = ["cfg", "kd", "q", "draft", "index_bpw", "cb_bytes_per_linear", "wikitext2_tv", "wikitext2_top1", "c4_tv",
           "wikitext2_ppl_draft", "tau_instruct_g1", "tau_instruct_g2", "tau_instruct_g4", "tau_wikitext2_g1",
           "tau_wikitext2_g2", "tau_wikitext2_g4"]
    f = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    if "--md" in sys.argv:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for x in rows:
            print("| " + " | ".join(f(x[h]) for h in hdr) + " |")
    else:
        print("\t".join(hdr))
        for x in rows:
            print("\t".join(f(x[h]) for h in hdr))
    if "--save" in sys.argv:
        sys.path.insert(0, REPO)
        from tools.results import save_result
        run = sys.argv[sys.argv.index("--save") + 1]
        cfg = {"model": "Qwen2.5-0.5B-Instruct", "setting": "no-FT screening ckpts (results/E5/qwen05_screen_*_noft.json); "
               "draft-only codebook copies rounded to q in {16,8,4}; target q16",
               "rounding": "experiments/06_accept/quantize_draft.py --gran codebook: one fp16 scale per (linear, draft "
               "codebook plane), symmetric INT, clip ratio MSE-searched in [0.25,1] x max|v|/qmax; int8 +-127, int4 +-7 "
               "(nestkern pack uses per-plane max-abs and int4 [-8,7]); q16 = bf16 copy (screening evals)",
               "draft": "A = target-codebook copy; B = TV-trained draft copy from screening (train_B_bf16), rounded the same way",
               "eval": "draft_target_eval.py kt=L, 40x512 wt2 + 40x512 c4; greedy 8 chat + 8 wt2 prompts x 128 tokens",
               "scripts": ["experiments/11_qcmp/run_qcmp.sh", "experiments/11_qcmp/table_qcmp.py"]}
        print(save_result("E5", run, cfg, {"rows": rows}, model_dir="/data/models/qwen2.5_0.5b_instruct"))
