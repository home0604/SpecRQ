"""Table of a q-suite output dir: per variant TV, 1-TV, top-1, KL(T||D), draft PPL, greedy tokens/iter (gamma 1/2/4),
plus draft codebook q and overhead bpw. Usage: python summarize_q_suite.py <outdir> [--save_exp E3 --run_prefix NAME]"""
import glob, json, os, sys
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"{d}/eval/*.json")):
    r = json.load(open(p)); name = os.path.basename(p)[:-5]
    v = name.split("_", 1)[1] if "_" in name else name
    meta_p = f"{d}/draft_q/{v}.meta.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    q = meta.get("q", 16 if v == "bf16" else None)
    row = {"variant": name, "q": q, "overhead_bpw": meta.get("draft_codebook_overhead_bits_per_weight"),
           "cb_relmse": meta.get("codebook_rel_mse_mean")}
    for ds in ("wikitext2", "c4"):
        x = r[ds]
        row.update({f"{ds}_tv": x["tv_mean"], f"{ds}_top1": x["top1_agree"], f"{ds}_kl_t_d": x["kl_t_d"], f"{ds}_ppl_draft": x["ppl_draft"],
                    f"{ds}_ppl_target": x["ppl_target"]})
    for kind in ("instruct", "wikitext2"):
        g = r["greedy_sim"][kind]
        for gm in (1, 2, 4):
            row[f"greedy_{kind}_g{gm}_tok_per_iter"] = g[f"gamma{gm}"]["tokens_per_iter"]
    rows.append(row)
hdr = ["variant", "q", "wikitext2_tv", "c4_tv", "wikitext2_top1", "c4_top1", "wikitext2_ppl_draft", "greedy_instruct_g4_tok_per_iter", "greedy_wikitext2_g4_tok_per_iter", "cb_relmse"]
print(" | ".join(hdr))
for row in rows:
    print(" | ".join(f"{row[h]:.4g}" if isinstance(row[h], float) else str(row[h]) for h in hdr))
json.dump(rows, open(f"{d}/summary.json", "w"), indent=1)
