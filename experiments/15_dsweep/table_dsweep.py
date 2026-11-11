"""Claim-1 d-sweep tables from results/E5/qwen05_dsweep_*.json (VQ T/N runs and GPTQ INTb baselines).
Main table per bit width: cfg, objective, L, bpw (index / incl. codebooks+scales), PPL@4096 wt2/c4 (target),
KL(fp16||target) and top-1 wt2/c4 (40x512 windows), training time (sum of per-layer times, h).
Appendix (--prefix): PPL@4096 wt2/c4 of every prefix k (draft candidates) with its bpw.
Usage: python experiments/15_dsweep/table_dsweep.py [--md] [--prefix] [--arch qwen05|llama31]
"""
import glob, json, os, sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ARCH = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "qwen05"  # qwen05 | llama31
f = lambda v: f"{v:.4g}" if isinstance(v, float) else ("-" if v is None else str(v))


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(REPO, f"results/E5/{ARCH}_dsweep_*.json"))):
        r = json.load(open(p)); m, c = r["metrics"], r["config"]
        L = c.get("L", 1)
        tq = m.get("target_quality", {})
        bits = round(m["target_bpw"]["index_bpw"])
        rows.append({"bits": bits, "cfg": c["cfg"], "obj": c.get("objective", "-"), "L": c.get("L", "-"),
                     "index_bpw": m["target_bpw"]["index_bpw"], "bpw_incl": m["target_bpw"]["bpw_with_codebooks_scales"],
                     "ppl_wt2": m["ppl_4096"].get(f"wikitext2_k{L}"), "ppl_c4": m["ppl_4096"].get(f"c4_k{L}"),
                     "kl_wt2": tq.get("wikitext2", {}).get("kl_fp16_target"), "kl_c4": tq.get("c4", {}).get("kl_fp16_target"),
                     "top1_wt2": tq.get("wikitext2", {}).get("top1_fp16_target"),
                     "top1_c4": tq.get("c4", {}).get("top1_fp16_target"),
                     "train_h": sum(m.get("layer_time_s", [])) / 3600 or None,
                     "_m": m, "_c": c, "_r": r["run"], "run": r["run"].split("_dsweep_", 1)[-1]})
    return rows


def emit(hdr, rows, md):
    if md:
        print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
        for x in rows:
            print("| " + " | ".join(f(x.get(h)) for h in hdr) + " |")
    else:
        print("\t".join(hdr))
        for x in rows:
            print("\t".join(f(x.get(h)) for h in hdr))


def main():
    md = "--md" in sys.argv
    rows = load()
    hdr = ["run", "obj", "L", "index_bpw", "bpw_incl", "ppl_wt2", "ppl_c4", "kl_wt2", "kl_c4", "top1_wt2", "top1_c4", "train_h"]
    for b in sorted({x["bits"] for x in rows}):
        print(f"\n### {b}-bit (index)\n")
        emit(hdr, sorted([x for x in rows if x["bits"] == b and not x["_r"].endswith("_kd24")], key=lambda x: (x["cfg"].startswith("gptq"), x["_c"].get("K", 0), x["L"] if isinstance(x["L"], int) else 0, x["obj"], x["_r"])), md)
    dr = [dict(t, run=x["_r"].replace(f"{ARCH}_dsweep_", "")) for x in rows for t in x["_m"].get("table", [])]
    if dr:
        print("\n### copy-draft (A) per draft prefix kd (target k=L)\n")
        emit(["run", "kd", "index_bpw", "wt2_tv", "wt2_top1", "c4_tv", "c4_top1", "tau_instruct_g1", "tau_instruct_g2",
              "tau_instruct_g4", "tau_wikitext2_g1", "tau_wikitext2_g2", "tau_wikitext2_g4", "ppl4096_wt2_k"],
             sorted(dr, key=lambda t: (t["kd"], t["run"])), md)
    if "--prefix" in sys.argv:
        print("\n### prefix k (draft candidates)\n")
        pr = []
        for x in rows:
            if x["cfg"].startswith("gptq"):
                continue
            for k in range(1, x["L"] + 1):
                pb = x["_m"].get("prefix_bpw", {}).get(str(k), {})
                pr.append({"cfg": x["cfg"], "obj": x["obj"], "k": k, "index_bpw": pb.get("index_bpw"),
                           "bpw_incl": pb.get("bpw_with_codebooks_scales"),
                           "ppl_wt2": x["_m"]["ppl_4096"].get(f"wikitext2_k{k}"), "ppl_c4": x["_m"]["ppl_4096"].get(f"c4_k{k}")})
        emit(["cfg", "obj", "k", "index_bpw", "bpw_incl", "ppl_wt2", "ppl_c4"], pr, md)


if __name__ == "__main__":
    main()
