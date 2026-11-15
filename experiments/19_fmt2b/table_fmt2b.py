"""E5 fmt2b table: codebook-format sweep (rounding only) on the Llama-3.1-8B-Instruct 2-bit nested d-sweep ckpts.
Reads /data/lsh/results_accept/fmt2b/<ckpt>_kd<kd>/{eval,draft_q}/A_<fmt>.{json,meta.json} written by run_fmt2b.sh.
mode = "draft" (kd < kt: prefix draft A rounded to fmt) or "target_side" (kd == kt: all target codebooks rounded; the
draft-vs-target TV/top1 is the target's own degradation if its codebooks were stored in fmt); "_mixed" = per-stage
formats (quantize_draft.py --fmt_stages, label mix_<front>-<back>, run_fmt2b_q.sh).
cb_bytes_per_linear = measured (bits_codebook + bits_scales) / 8 / n_linears of the rounded copy (scales: 1 fp16 per
codebook; bf16 = 2 B/value, no scale). d_* = delta vs the bf16 row of the same (ckpt, kd).
Usage: python experiments/19_fmt2b/table_fmt2b.py [--md] [--save RUN]
"""
import glob, json, os, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = "/data/lsh/results_accept/fmt2b"
FMTS = ["bf16", "int8", "fp8_e4m3", "nf4", "fp4_e2m1", "int4", "posit4_es0", "nf3", "int3"]
CK = {"d8K256L2_T_FT_n192": (8, 256, 2), "d32K256L8_T_FT_n192": (32, 256, 8), "d32K256L8_N_k4_FT_n192": (32, 256, 8)}


def collect():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/*_kd*")):
        cfg, kd = os.path.basename(d).rsplit("_kd", 1); kd = int(kd)
        dd, K, L = CK[cfg]
        labs = [os.path.basename(p)[2:-5] for p in glob.glob(f"{d}/eval/A_*.json")]
        for fmt in [f for f in FMTS if f in labs] + sorted(f for f in labs if f not in FMTS):
            p = f"{d}/eval/A_{fmt}.json"
            r = json.load(open(p)); g = r["greedy_sim"]; assert r["kd"] == kd and r["kt"] == L
            m = json.load(open(p.replace("/eval/", "/draft_q/").replace(".json", ".meta.json")))
            x = {"ckpt": cfg, "d": dd, "K": K, "L": L, "kd": kd, "kt": L,
                 "mode": ("target_side" if kd == L else "draft") + ("_mixed" if m["args"].get("fmt_stages") else ""),
                 "fmt": fmt, "stage_fmts": m["args"].get("fmt_stages") or ",".join([m["args"]["fmt"]] * kd), "q": m["q"],
                 "index_bpw_draft": kd * 8 / dd, "cb_rel_mse": m["codebook_rel_mse_mean"],
                 "cb_rel_mse_max": m["codebook_rel_mse_max"],
                 "cb_bytes_per_linear": (m["bits_codebook"] + m["bits_scales"]) / 8 / m["n_linears"],
                 "plane_kurt_median": m["plane_kurtosis"]["median"], "eval_json": p, "wall_s": r.get("wall_s")}
            for ds in ("wikitext2", "c4"):
                for k, s in (("tv", "tv_mean"), ("top1", "top1_agree"), ("ppl_draft", "ppl_draft"), ("ppl_target", "ppl_target"),
                             ("kl_t_d", "kl_t_d")):
                    x[f"{ds}_{k}"] = r[ds][s]
            for kind in ("instruct", "wikitext2"):
                for gm in (1, 2, 4):
                    x[f"tau_{kind}_g{gm}"] = g[kind][f"gamma{gm}"]["tokens_per_iter"]
            rows.append(x)
    base = {(x["ckpt"], x["kd"]): x for x in rows if x["fmt"] == "bf16"}
    for x in rows:
        b = base.get((x["ckpt"], x["kd"]))
        if b:
            for k in ("wikitext2_tv", "c4_tv", "wikitext2_top1", "tau_instruct_g2", "tau_wikitext2_g2"):
                x[f"d_{k}_vs_bf16"] = x[k] - b[k]
    return rows


if __name__ == "__main__":
    rows = collect()
    hdr = ["ckpt", "kd", "kt", "mode", "fmt", "q", "cb_rel_mse", "cb_bytes_per_linear", "wikitext2_tv", "c4_tv",
           "wikitext2_top1", "c4_top1", "wikitext2_ppl_draft", "c4_ppl_draft", "wikitext2_ppl_target",
           "tau_instruct_g1", "tau_instruct_g2", "tau_instruct_g4", "tau_wikitext2_g1", "tau_wikitext2_g2",
           "tau_wikitext2_g4", "d_wikitext2_tv_vs_bf16", "d_tau_instruct_g2_vs_bf16"]
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
        cfg = {"model": "Llama-3.1-8B-Instruct", "exp": "E5 fmt2b",
               "ckpts": {k: f"/data/lsh/ckpts/weightvq/dsweep_llama/{k}" for k in CK},
               "setting": "2-bit/weight nested d-sweep ckpts (results/E5/llama31_dsweep_*.json); draft A = copy of the "
               "first kd target codebooks rounded (no retraining) to a codebook VALUE format; weight index bits unchanged. "
               "kd<kt: prefix draft (d8L2 kd1, d32L8 kd4). kd==kt: target-side probe (all codebooks rounded; target "
               "calls never take the draft path) = target degradation if stored in that format",
               "rounding": "experiments/06_accept/quantize_draft.py --gran codebook --fmt F: one fp16 scale per (linear, "
               "codebook). int8/int4/int3: clip ratio [0.25,1] x max|v|/qmax; fp8_e4m3: scale grid 2^-12..2^4 x max|v|/448; "
               "nf4/nf3/fp4_e2m1/posit4_es0 LUT: max|v| -> 2^-6..2^0.5 x largest level, MSE search",
               "eval": "draft_target_eval.py --codes_uint8 --no_fp16 kt=L, 40x512 wt2 + 40x512 c4 teacher-forced; greedy "
               "8 chat + 8 wt2 prompts x 128 tokens (tau = simulated tokens/iteration, not measured speed)",
               "mixed": "target_side_mixed rows: per-stage formats (stage_fmts), d8L2 (stage1,stage2), d32L8 front 4 / back 4 stages",
               "scripts": ["experiments/19_fmt2b/run_fmt2b.sh", "experiments/19_fmt2b/run_fmt2b_q.sh", "experiments/19_fmt2b/table_fmt2b.py",
                           "experiments/06_accept/quantize_draft.py", "experiments/06_accept/draft_target_eval.py"]}
        print(save_result("E5", run, cfg, {"rows": rows}, model_dir="/data/models/llama3.1_8b_instruct"))
