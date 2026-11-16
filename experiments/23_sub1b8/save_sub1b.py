"""E5 sub1b8: collect one config (target ckpt + draft A/B evals + synthetic cachefit timing) into
results/E5/llama31_sub1b_<cfg>.json; --table prints/saves the combined table over all collected configs.
tau = greedy-simulation tokens/iteration (draft_target_eval greedy_sim, 8 chat + 8 wt2 prompts x 128 tok).
Cost: c = t_draft(k)/T_AR(target) and break-even tau = (g t_draft + t_fwd(g+1) + ovh)/T_AR from the SYNTHETIC
timing (specrq.infer.config_sweep, results/E4/llama31_sub1b_cachefit_<cfg>_q16.json); modeled, not measured decoding.
Usage: python save_sub1b.py --cfg C --bits 4,4,4,4 --d 8 --ckpt DIR --log TRAINLOG --eval_root DIR --kds "1 2" [--ppl_from REC]
       python save_sub1b.py --table
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)
from tools.results import save_result

SHAPES = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]
T_AR_REF_CFG = "d8K256L2"


def bpw(bits, d):
    n = sum(o * i for o, i in SHAPES)
    cb = len(SHAPES) * sum(2 ** b for b in bits) * d * 16
    sc = sum(o for o, _ in SHAPES) * 16
    return {"index_bpw": sum(bits) / d, "bpw_with_codebooks_scales": sum(bits) / d + (cb + sc) / n}


def parse_log(log):
    out = {"ppl_4096": {}}
    for line in open(log, errors="replace"):
        m = re.search(r"RESULT dataset=(\w+) k=(\d+) ppl=([0-9.]+)", line)
        if m and f"{m[1]}_k{m[2]}" not in out["ppl_4096"]:
            out["ppl_4096"][f"{m[1]}_k{m[2]}"] = float(m[3])
        if "Maximum resident set size" in line:
            out["train_peak_rss_gib"] = int(line.split(":")[-1]) / 2**20
        elif "Elapsed (wall clock) time" in line:
            t = line.rsplit(" ", 1)[-1].strip().split(":")
            out["train_wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
    return out


def draft_row(js):
    r = json.load(open(js))
    g = r["greedy_sim"]
    return {"wt2_tv": r["wikitext2"]["tv_mean"], "wt2_top1": r["wikitext2"]["top1_agree"],
            "c4_tv": r["c4"]["tv_mean"], "c4_top1": r["c4"]["top1_agree"],
            "wt2_ppl_draft_512": r["wikitext2"]["ppl_draft"], "c4_ppl_draft_512": r["c4"]["ppl_draft"],
            **{f"tau_{kind}_g{k}": g[kind][f"gamma{k}"]["tokens_per_iter"] for kind in ("instruct", "wikitext2") for k in (1, 2, 4)},
            "eval_json": js}


def timing(cfg):
    p = os.path.join(REPO, f"results/E4/llama31_sub1b_cachefit_{cfg}_q16.json")
    m = json.load(open(p))["metrics"]
    T = m["ar"][str(m["L"])]["step_us"] / 1e3
    out = {"json": p, "T_ar_ms": T, "kd": {}}
    for r in m["model"]:
        e = out["kd"].setdefault(str(r["kd"]), {"t_draft_ms": r["t_draft_ms"], "c": r["t_draft_ms"] / T})
        e[f"tau_req_g{r['gamma']}"] = r["tau_required"]
        e[f"round_ms_g{r['gamma']}"] = r["round_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg"); ap.add_argument("--bits"); ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--ckpt"); ap.add_argument("--log"); ap.add_argument("--eval_root"); ap.add_argument("--kds", default="1")
    ap.add_argument("--ppl_from", default=None, help="take PPL/target quality/train time from an existing E5 record")
    ap.add_argument("--note", default="")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        return table()
    bits = [int(b) for b in a.bits.split(",")]
    L = len(bits)
    m = {}
    if a.ppl_from:
        old = json.load(open(a.ppl_from))["metrics"]
        m.update({k: old[k] for k in ("ppl_4096", "train_wall_s", "train_peak_rss_gib") if k in old})
        m["source_record"] = a.ppl_from
    else:
        m.update(parse_log(a.log))
    first = None
    m["kd"] = {}
    tm = timing(a.cfg)
    m["timing_synthetic"] = tm
    for kd in a.kds.split():
        e = {"bits": bits[:int(kd)], "bpw": bpw(bits[:int(kd)], a.d)}
        for v in ("A", "B"):
            js = f"{a.eval_root}/kd{kd}/eval/{v}_bf16.json"
            if os.path.exists(js):
                e[v] = draft_row(js)
                r = json.load(open(js))
                if first is None and "ppl_fp16" in r["wikitext2"]:
                    first = r
            tl = f"{a.eval_root}/kd{kd}/train_B_bf16/train_log.json"
            if v == "B" and os.path.exists(tl):
                t = json.load(open(tl))
                e["B_train"] = {"wall_s": t["log"][-1]["t"], "final_tv": t["log"][-1]["tv"], "peak_mem_gib": t["peak_mem_gib"],
                                "teacher_check": t.get("teacher_check"), "lr": t["args"]["lr"]}
        e["cost"] = tm["kd"].get(kd)
        m["kd"][kd] = e
    if first is not None:
        m["target_quality"] = {ds: {k: first[ds][k] for k in ("ppl_target", "ppl_fp16", "kl_fp16_target", "top1_fp16_target")}
                               for ds in ("wikitext2", "c4")}
    m["target_bpw"] = bpw(bits, a.d)
    cfg = {"model": "Llama-3.1-8B-Instruct", "cfg": a.cfg, "d": a.d, "code_bits": bits, "K": [2 ** b for b in bits], "L": L,
           "objective": "T (target-only alpha 0..0,1)", "ckpt": a.ckpt,
           "setting": "Hadamard, pile_val 192x4096 calib (val 32), accum fp32, AQ rel_tol 0.01, beam 1, block FT 5 epochs "
                      "(batch 32, lr 1e-5, early stop 3, keep best), seed 0, activations on GPU (= dsweep_llama *_T_FT_n192 recipe)",
           "draft_A": "target-copy prefix codebooks/scales (kd stages)",
           "draft_B": "draft-only codebooks+scales of the kd prefix, TV-trained vs full target (use_draft=False teacher, "
                      "--check_teacher), pile_val_heldout 1024x512, 300 steps, batch 4, lr 1e-4, micro_batch 1",
           "eval": "draft_target_eval kt=L: TV/top-1 40x512 wt2 + 40x512 c4 windows; tau = greedy simulation tokens/iter, "
                   "8 chat + 8 wt2 prompts x 128 tok (shared KV, not measured decoding); target PPL@4096 wt2/c4 from main.py",
           "cost": "SYNTHETIC cachefit timing (random weights, same kernel config, q16, vd 32768), c = t_draft/T_AR, "
                   "break-even tau = round_ms/T_AR (own target AR); modeled [I], not measured spec decoding",
           "scripts": ["experiments/23_sub1b8/train_sub1b.sh", "experiments/23_sub1b8/eval_sub1b.sh",
                       "experiments/23_sub1b8/save_sub1b.py"], "note": a.note}
    p = save_result("E5", f"llama31_sub1b_{a.cfg}", cfg, m, model_dir="/data/models/llama3.1_8b_instruct")
    print(p)


def table():
    import glob
    ref_T = None
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(REPO, "results/E5/llama31_sub1b_*.json"))) if "table" not in p]
    for r in recs:
        if r["config"]["cfg"] == T_AR_REF_CFG:
            ref_T = r["metrics"]["timing_synthetic"]["T_ar_ms"]
    rows = []
    for r in recs:
        m, c = r["metrics"], r["config"]
        tq = m.get("target_quality", {})
        for kd, e in m["kd"].items():
            for v in ("A", "B"):
                if v not in e:
                    continue
                d = e[v]
                cost = e["cost"] or {}
                row = {"cfg": c["cfg"], "draft": f"k{kd}{v}", "draft_bits": e["bits"], "draft_index_bpw": e["bpw"]["index_bpw"],
                       "ppl4096_wt2": m["ppl_4096"].get(f"wikitext2_k{c['L']}"), "ppl4096_c4": m["ppl_4096"].get(f"c4_k{c['L']}"),
                       "kl_fp16_target_wt2": tq.get("wikitext2", {}).get("kl_fp16_target"),
                       "tv_wt2": d["wt2_tv"], "top1_wt2": d["wt2_top1"], "tv_c4": d["c4_tv"], "top1_c4": d["c4_top1"],
                       "c": cost.get("c"), "T_ar_ms": m["timing_synthetic"]["T_ar_ms"]}
                for g in (1, 2, 4):
                    ti, tw = d[f"tau_instruct_g{g}"], d[f"tau_wikitext2_g{g}"]
                    be = cost.get(f"tau_req_g{g}")
                    row[f"tau_g{g}"] = [ti, tw]
                    row[f"breakeven_g{g}"] = be
                    row[f"above_g{g}"] = [ti >= be, tw >= be] if be else None
                    if ref_T and be:  # modeled speed-up vs AR of the d8K256L2 target (cross-config reference)
                        R = cost[f"round_ms_g{g}"]
                        row[f"model_speedup_vs_ownAR_g{g}"] = [t * m["timing_synthetic"]["T_ar_ms"] / R for t in (ti, tw)]
                        row[f"model_speedup_vs_d8K256L2AR_g{g}"] = [t * ref_T / R for t in (ti, tw)]
                rows.append(row)
    for row in rows:
        print(json.dumps(row))
    return rows


if __name__ == "__main__":
    main()
