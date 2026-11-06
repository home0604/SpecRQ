"""One table over all E5 screening records (results/E5/qwen05_screen_*_noft.json), sorted by draft index bpw.
Columns: cfg, kd, draft (A = target-copy, B = TV-trained), index bpw, bpw incl. codebooks+scales, wt2/c4 TV, top-1,
greedy tokens/iter gamma 1/2/4 (chat / wt2), draft-prefix wt2 PPL@4096, and the config's target (k=L) quality.
Usage: python experiments/10_screen/table_screen.py [--md]
"""
import glob, json, os, sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
rows = []
for p in sorted(glob.glob(os.path.join(REPO, "results/E5/qwen05_screen_*_noft.json"))):
    r = json.load(open(p)); m = r["metrics"]; c = r["config"]
    L = c["L"]; tq = m.get("target_quality", {}).get("wikitext2", {})
    for t in m["table"]:
        rows.append({**t, "L": L, "target_bpw": m["target_bpw"]["bpw_with_codebooks_scales"],
                     "target_ppl4096_wt2": m["ppl_4096"].get(f"wikitext2_k{L}"),
                     "target_kl_fp16": tq.get("kl_fp16_target"), "target_top1_fp16": tq.get("top1_fp16_target")})
rows.sort(key=lambda x: (x["index_bpw"], x["cfg"], x["draft"]))
hdr = ["cfg", "kd", "draft", "index_bpw", "bpw_with_codebooks_scales", "wt2_tv", "wt2_top1", "c4_tv",
       "tau_instruct_g1", "tau_instruct_g2", "tau_instruct_g4", "tau_wikitext2_g1", "tau_wikitext2_g2", "tau_wikitext2_g4",
       "ppl4096_wt2_k", "target_bpw", "target_ppl4096_wt2", "target_kl_fp16", "target_top1_fp16"]
f = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
if "--md" in sys.argv:
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for x in rows:
        print("| " + " | ".join(f(x.get(h)) for h in hdr) + " |")
else:
    print("\t".join(hdr))
    for x in rows:
        print("\t".join(f(x.get(h)) for h in hdr))
