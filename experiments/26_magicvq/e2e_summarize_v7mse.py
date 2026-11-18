"""E7 magicvq v7mse: default ('p' greedy, pile calib k-means RQ) vs t2mw (MSE-objective control, rqd_train.py --loss mse)
vs t2w (attention-output-loss trained, v6) codebooks, fully real e2e (e2e_real.py --accounting both).

Per point the runs v7mse / v7mseh / v7msev / v7mseg512 (hybrid / RQ2 processes, each with its own AR + best SnapKV).
Columns: tau static / aa and all-active / static speedups vs the AR of the same process, per draft and codebook;
ratios t2w / t2mw (attention-loss gain over MSE training) and t2mw / default (training/data gain) for tau and
all-active speedup at the same draft and gamma; best VQ / best SnapKV per codebook (all-active).
Diagnostic: held-out attention-output loss and K/V MSE (rqd_train.py --eval_cb, files rqd_eval_v7mse_L*.json).
Usage: python experiments/26_magicvq/e2e_summarize_v7mse.py [--save]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e2e_summarize_aa import LABEL, OUT, RES, REPO, analyze, v5_derived  # noqa: E402

sys.path.insert(0, REPO)
CBD = "/data/lsh/results_accept/magicvq/cb"
CBS = ("def", "t2mw", "t2w")


def diag():
    per = {}
    for f in sorted(glob.glob(f"{CBD}/rqd_eval_v7mse_L*.json")):
        per.update({int(k): v for k, v in json.load(open(f))["per_layer"].items()})
    if not per:
        return None, []
    mean = {t: {k: sum(per[l][t][k] for l in per) / len(per) for k in ("attn", "mseK", "mseV")} for t in CBS}
    tr = json.load(open(f"{CBD}/cb_d8K256x2_post_cn_t2mw.json")) if os.path.exists(f"{CBD}/cb_d8K256x2_post_cn_t2mw.json") else {}
    L = ["## Held-out diagnostic (rqd_train.py --eval_cb, 3 held-out sequences, RoPE offset 8192, level 2, R 128; mean "
         f"over {len(per)} layers)", "",
         "attn = ||(o_hat - o) W_o^T||^2 / ||o W_o^T||^2 (t2w's training objective); mseK / mseV = relative K / V "
         "reconstruction error on the VQ-read positions (t2mw's objective, de-standardised post-RoPE space).", "",
         "| codebook | attn loss | mse K | mse V |", "|---|---|---|---|"]
    for t in CBS:
        L.append(f"| {t} | {mean[t]['attn']:.5f} | {mean[t]['mseK']:.4f} | {mean[t]['mseV']:.4f} |")
    if tr:
        L += ["", f"t2mw training: {sum(r['kept'] for r in tr['per_layer'])}/32 layers kept (held-out MSE objective "
              f"{tr['mean_before']:.4f} -> {tr['mean_after']:.4f}), train time {tr['train_s']:.0f} s summed over layers."]
    L += ["", "per layer (attn loss def / t2mw / t2w):", "",
          "| layer | def | t2mw | t2w | mseK def / t2mw / t2w | mseV def / t2mw / t2w |", "|---|---|---|---|---|---|"]
    for l in sorted(per):
        x = per[l]
        L.append(f"| {l} | {x['def']['attn']:.5f} | {x['t2mw']['attn']:.5f} | {x['t2w']['attn']:.5f} | "
                 + " / ".join(f"{x[t]['mseK']:.4f}" for t in CBS) + " | " + " / ".join(f"{x[t]['mseV']:.4f}" for t in CBS)
                 + " |")
    return {"mean": mean, "per_layer": per}, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    v5d = v5_derived()
    pts = {}
    for tag in ("v7mse", "v7mseh", "v7msev", "v7mseg512"):
        for f in sorted(glob.glob(f"{OUT}/run_{tag}_*.json")):
            q = analyze(f, v5d)
            k = (q["task"], q["ctx"], q["B"], q["gen"])
            for r in q["rows"]:
                r["proc"] = tag
                base, _, cb = r["draft"].partition("@")
                r["base"], r["cb"] = base, (cb or "def")
            p = pts.setdefault(k, {"name": q["name"], "task": q["task"], "ctx": q["ctx"], "B": q["B"], "gen": q["gen"],
                                   "procs": {}, "rows": [], "cotenant_any": False})
            p["procs"][tag] = {"ar_tok_s": q["ar_tok_s"], "ar_step_ms": q["ar_step_ms"], "raw": f,
                               "cotenant_any": q["cotenant_any"]}
            p["rows"] += q["rows"]
            p["cotenant_any"] = p["cotenant_any"] or q["cotenant_any"]
    dg, DL = diag()
    L = ["# E7 magicvq v7mse: MSE-trained control codebook (t2mw) vs attention-loss trained (t2w) vs default, measured "
         "e2e (TP=4 RTX 4090, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, EOS ignored)", "",
         "- t2mw: rqd_train.py --loss mse --level 2 --R 128 --steps 600 --lr 3e-3 --tf32 (same data, parameters, steps, "
         "lr, STE, RoPE offset, held-out keep rule as t2w; loss = ||K_hat-K||^2/||K||^2 + ||V_hat-V||^2/||V||^2 on the "
         "VQ-read positions). default = cb_d8K256x2_post_cn_greedy.pt (k-means RQ on pile-val).",
         "- e2e_real.py --accounting both --cbx t2w=... t2mw=...; speedup vs the AR of the same process; 3 timed reps "
         "after warm-up, median. Ratios at the same draft / gamma / process.", ""]
    S = ["## Summary: tau and all-active speedup ratios at the same draft and gamma", "",
         "| point | draft | gamma | tau static def / t2mw / t2w | aa speedup def / t2mw / t2w | static speedup def / t2mw / "
         "t2w | **tau t2w/t2mw** | tau t2mw/def | **aa t2w/t2mw** | aa t2mw/def |", "|---|---|---|---|---|---|---|---|---|---|"]
    B_ = ["## Best VQ/hybrid / best SnapKV per codebook (all-active / static)", "",
          "| point | AR tok/s | best SnapKV aa / static | def aa / static | t2mw aa / static | t2w aa / static | co-tenant |",
          "|---|---|---|---|---|---|---|"]
    body = []
    for k in sorted(pts, key=lambda k: ({"syn_quote": 0, "hotpotqa_q": 1, "pg19": 2}.get(k[0], 9), -k[2], k[3])):
        p = pts[k]
        by = {(r["base"], r["gamma"], r["cb"]): r for r in p["rows"]}
        p["ratios"] = []
        for (base, g, cb), r in sorted(by.items()):
            if cb != "t2w" or base.startswith("snapkv"):
                continue
            d, m = by.get((base, g, "def")), by.get((base, g, "t2mw"))
            if not (d and m):
                continue
            x = {"draft": base, "gamma": g, "proc": r["proc"],
                 "tau": [d["tau_static"], m["tau_static"], r["tau_static"]],
                 "aa": [d["aa_speedup"], m["aa_speedup"], r["aa_speedup"]],
                 "static": [d["static_speedup"], m["static_speedup"], r["static_speedup"]]}
            p["ratios"].append(x)
            S.append(f"| {p['name']} | {LABEL.get(base, base)} | {g} | " + " / ".join(f"{v:.2f}" for v in x["tau"]) + " | "
                     + " / ".join(f"{v:.3f}" for v in x["aa"]) + " | " + " / ".join(f"{v:.3f}" for v in x["static"])
                     + f" | **{x['tau'][2] / x['tau'][1]:.3f}** | {x['tau'][1] / x['tau'][0]:.3f} | "
                     f"**{x['aa'][2] / x['aa'][1]:.3f}** | {x['aa'][1] / x['aa'][0]:.3f} |")
        snap = [r for r in p["rows"] if r["base"].startswith("snapkv")]
        bs = {acc: max(r[acc] for r in snap) for acc in ("aa_speedup", "static_speedup")}
        cells, p["best"] = [], {"snapkv": bs}
        for cb in CBS:
            rs = [r for r in p["rows"] if r["cb"] == cb and not r["base"].startswith("snapkv")]
            if not rs:
                cells.append("-")
                continue
            ba, bst = max(rs, key=lambda r: r["aa_speedup"]), max(rs, key=lambda r: r["static_speedup"])
            p["best"][cb] = {"aa": [ba["config"], ba["aa_speedup"], ba["aa_speedup"] / bs["aa_speedup"]],
                             "static": [bst["config"], bst["static_speedup"], bst["static_speedup"] / bs["static_speedup"]]}
            cells.append(f"{ba['aa_speedup'] / bs['aa_speedup']:.3f} / {bst['static_speedup'] / bs['static_speedup']:.3f}")
        ars = " / ".join(f"{v['ar_tok_s']:.1f}" for v in p["procs"].values())
        B_.append(f"| {p['name']} | {ars} | {bs['aa_speedup']:.3f} / {bs['static_speedup']:.3f} | " + " | ".join(cells)
                  + f" | {'YES' if p['cotenant_any'] else 'no'} |")
        body += [f"## {p['name']}", "", "processes: " + "; ".join(
            f"{t}: AR {v['ar_tok_s']:.1f} tok/s, co-tenant {'YES' if v['cotenant_any'] else 'no'}, {os.path.basename(v['raw'])}"
            for t, v in p["procs"].items()), "",
            "| draft | cb | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | "
            "aa rows identical to AR | co-tenant |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in p["rows"]:
            ar = p["procs"][r["proc"]]["ar_tok_s"]
            body.append(f"| {LABEL.get(r['base'], r['base'])} | {r['cb']} | {r['proc']} | {r['gamma']} | "
                        f"{r['static_speedup']:.3f} | **{r['aa_speedup']:.3f}** ({r['aa_tok_s_min_max'][0] / ar:.3f}-"
                        f"{r['aa_tok_s_min_max'][1] / ar:.3f}) | {r['tau_static']:.2f} / {r['aa_tau']:.2f} | "
                        f"{r['static_round_ms']:.1f} / {r['aa_round_ms']:.1f} | {r['aa_n_rows_identical_overlap']}/{p['B']} | "
                        f"{'YES' if r['cotenant'] else 'no'} |")
        body.append("")
    md = "\n".join(L + S + [""] + B_ + [""] + DL + [""] + body)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_v7mse_tables.md", "w").write(md + "\n")
        from tools.results import save_result
        if dg:
            save_result("E7", "llama8b_magicvq_v7mse_diag", config={"what": "held-out attn loss / K,V MSE per layer",
                        "cbs": {t: f"{CBD}/cb_d8K256x2_post_cn_{t if t != 'def' else 'greedy'}.pt" for t in CBS}},
                        metrics=dg, model_dir="/data/models/llama3.1_8b_instruct")
        for k, p in pts.items():
            save_result("E7", f"llama8b_magicvq_v7mse_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), e2e_real.py "
                                "--accounting both --cbx t2w=... t2mw=...", "B": p["B"], "T": p["ctx"], "gen": p["gen"],
                                "task": p["task"], "configs": [r["config"] for r in p["rows"]]},
                        metrics={"procs": p["procs"], "rows": p["rows"], "ratios": p["ratios"], "best": p["best"],
                                 "cotenant_any": p["cotenant_any"]},
                        model_dir="/data/models/llama3.1_8b_instruct",
                        notes="raw: " + "; ".join(v["raw"] for v in p["procs"].values()))


if __name__ == "__main__":
    main()
