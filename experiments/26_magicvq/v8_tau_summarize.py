"""E7 magicvq v8 (1-bit KV drafts), phase 1: tau of 1-bit drafts vs the 2-bit t2w drafts and SnapKV, same samples.

v8 runs (v8_tau.sh) evaluate new drafts on exactly the v6 samples (--only_run); the v6 records hold the reference drafts
(snapkv*, vqpt2wL2+r128, hyb512pt2wL2, vqpt1wL1+r128, hyb512pt1wL1, untrained vqpL*). Records are joined by sample id
(LongBench / synthetic) or prompt index (PG-19); anchors present in both runs must agree exactly (checked, reported).
Also folds in the held-out diagnostics of v8_train.sh (rqd_train.py --eval_cb at level 1 and 2).
tau(g) = pooled accepted+1 tokens / rounds over the samples (greedy round walk, as lb_summarize / rqd_summarize);
ratio CIs: paired bootstrap over samples (90 %, 2000 resamples).
Usage: python experiments/26_magicvq/v8_tau_summarize.py [--save]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = "/data/lsh/results_accept/magicvq"
CBD = f"{ROOT}/cb"
OUTMD = f"{REPO}/results/E7/llama8b_magicvq_v8_1bit_tau_tables.md"
GAMMAS = (3, 4, 6)
POINTS = [  # task, ctx, [(v8 file, v6 file)]
    ("syn_quote", 32768, [("lb/tau_v8a_syn_quote_ctx32768.json", "lb/tau_v6a_syn_quote_ctx32768.json"),
                          ("lb/tau_v8b_syn_quote_ctx32768.json", "lb/tau_v6b_syn_quote_ctx32768.json")]),
    ("hotpotqa_q", 16384, [(f"lb/tau_v8{r}_hotpotqa_q_ctx16384.json", f"lb/tau_v6{r}_hotpotqa_q_ctx16384.json")
                           for r in ("a", "b", "a2")]),
    ("gov_report", 16384, [(f"lb/tau_v8{r}_gov_report_ctx16384.json", f"lb/tau_v6{r}_gov_report_ctx16384.json")
                           for r in ("a", "b")]),
    ("gov_report", 32768, [(f"lb/tau_v8{r}_gov_report_ctx32768.json", f"lb/tau_v6{r}_gov_report_ctx32768.json")
                           for r in ("a", "b")]),
    ("pg19", 32768, [("tau/tau_v8_ctx32768.json", "tau/tau_v6_ctx32768.json")]),
]
REF2 = ["vqpt2wL2+r128", "hyb512pt2wL2"]
SNAP = ["snapkv2048", "snapkv4096"]
ONE = ["vqpt1wL1+r128", "vqpt1wL1+r512", "hyb512pt1wL1", "vqpt2wL1+r128", "hyb512pt2wL1", "vqpt12wL1+r128",
       "vqpt12wL1+r512", "hyb512pt12wL1", "vqpL1+r128"]
TWO_NEST = ["vqpt12wL2+r128", "hyb512pt12wL2"]
LABEL = {
    "vqpt2wL2+r128": "RQ2+r128 t2w (2-bit)", "hyb512pt2wL2": "SnapKV512+RQ2 t2w (2-bit hybrid)",
    "vqpt1wL1+r128": "RQ1+r128 t1w (dedicated 1-stage)", "vqpt1wL1+r512": "RQ1+r512 t1w",
    "hyb512pt1wL1": "SnapKV512+RQ1 t1w", "vqpt2wL1+r128": "RQ1+r128 t2w stage 1 (nested prefix)",
    "hyb512pt2wL1": "SnapKV512+RQ1 t2w stage 1 (nested)", "vqpt12wL1+r128": "RQ1+r128 t12w (nested-trained)",
    "vqpt12wL1+r512": "RQ1+r512 t12w", "hyb512pt12wL1": "SnapKV512+RQ1 t12w",
    "vqpt12wL2+r128": "RQ2+r128 t12w", "hyb512pt12wL2": "SnapKV512+RQ2 t12w",
    "vqpL1+r128": "RQ1+r128 untrained 'p'", "snapkv2048": "SnapKV-2048", "snapkv4096": "SnapKV-4096",
}


def load_point(pairs):
    """-> list of per-sample {draft: {g: (toks, its)}}, anchor check dict."""
    rows, chk = [], {"compared": 0, "mismatch": 0, "drafts": set()}
    for f8, f6 in pairs:
        p8, p6 = f"{ROOT}/{f8}", f"{ROOT}/{f6}"
        if not os.path.exists(p8):
            continue
        r8, r6 = json.load(open(p8)), json.load(open(p6))
        key = (lambda r: r["id"]) if "id" in r8["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        m6 = {key(r): r for r in r6["per_prompt"]}
        for r in r8["per_prompt"]:
            o = m6[key(r)]
            assert o["P"] == r["P"] and o["N"] == r["N"], (f8, key(r))
            d = {}
            for src in (o, r):
                for n, v in src["drafts"].items():
                    g = {x: tuple(v[f"g{x}"][:2]) for x in GAMMAS}
                    if n in d:
                        chk["compared"] += 1
                        chk["drafts"].add(n)
                        chk["mismatch"] += int(d[n] != g)
                    d[n] = g
            rows.append(d)
    chk["drafts"] = sorted(chk["drafts"])
    return rows, chk


def tau(rows, n, g, idx=None):
    idx = range(len(rows)) if idx is None else idx
    t = sum(rows[i][n][g][0] for i in idx)
    s = sum(rows[i][n][g][1] for i in idx)
    return t / s


def ratio_ci(rows, a, b, g, nb=2000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    r0 = tau(rows, a, g) / tau(rows, b, g)
    bs = []
    for _ in range(nb):
        idx = [rnd.randrange(n) for _ in range(n)]
        bs.append(tau(rows, a, g, idx) / tau(rows, b, g, idx))
    bs.sort()
    return r0, bs[int(0.05 * nb)], bs[int(0.95 * nb) - 1]


def diag():
    out = {}
    for lv in (1, 2):
        per = {}
        for lo in (0, 8, 16, 24):
            p = f"{CBD}/rqd_eval_v8_lv{lv}_L{lo}.json"
            if not os.path.exists(p):
                return None
            for l, r in json.load(open(p))["per_layer"].items():
                for t, v in r.items():
                    per.setdefault(t, {})[int(l)] = v
        out[f"level{lv}"] = {t: {k: sum(v[k] for v in d.values()) / len(d) for k in ("attn", "mseK", "mseV")}
                             for t, d in per.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    L = ["# E7 magicvq v8 phase 1: tau of 1 bit/element VQ KV drafts (same samples as v6)", "",
         "- Drafts are read from v6 records (references) and v8 records (new 1-bit drafts + anchors) joined per sample; "
         "anchors present in both runs must give identical round walks.",
         "- Codebooks (all d8 K256, post-RoPE K, per-head channel norm, 16 bf16 sinks): t2w = v6 2-stage attention-loss "
         "codebook; 'stage 1' = its first stage alone (nested prefix, 1 bit); t1w = v6 dedicated 1-stage attention-loss "
         "codebook (t2w recipe at level 1); t12w = v8 nested-prefix training (loss stage 1 + stages 1+2, t2w recipe).",
         "- tau = pooled tokens / round (greedy, teacher-forced rounds, = lb_tau / tau_eval). Ratios: paired bootstrap "
         "90 % CI over samples.", ""]
    dg = diag()
    if dg:
        L += ["## Held-out diagnostics (rqd_train.py --eval_cb, R = 128, mean over 32 layers)", "",
              "| codebook | L1 attn loss | L1 mseK | L1 mseV | L2 attn loss | L2 mseK | L2 mseV |", "|---|---|---|---|---|---|---|"]
        for t in dg["level1"]:
            x, y = dg["level1"][t], dg["level2"][t]
            L.append(f"| {t} | {x['attn']:.4f} | {x['mseK']:.4f} | {x['mseV']:.4f} | {y['attn']:.4f} | {y['mseK']:.4f} | "
                     f"{y['mseV']:.4f} |")
        L += ["", "(t1w at L2 and t2w at L1 are prefix / extension reads of codebooks trained for the other level.)", ""]
    rec = {"diag": dg, "points": {}}
    for task, ctx, pairs in POINTS:
        rows, chk = load_point(pairs)
        if not rows:
            continue
        drafts = [n for n in SNAP + REF2 + ONE + TWO_NEST if all(n in r for r in rows)]
        pt = {"n": len(rows), "anchor_check": chk, "tau": {}, "ratio_vs_rq2": {}, "ratio_vs_snap4096": {}}
        L += [f"## {task} ctx {ctx} (n = {len(rows)}; anchors compared {chk['compared']}, mismatches {chk['mismatch']})", "",
              "| draft | tau g3 | tau g4 | tau g6 | vs RQ2+r128 t2w g3 / g4 / g6 [CI g4] | vs hyb RQ2 t2w g4 [CI] | "
              "vs SnapKV-4096 g4 |", "|---|---|---|---|---|---|---|"]
        for n in drafts:
            ts = {g: tau(rows, n, g) for g in GAMMAS}
            pt["tau"][n] = ts
            rv = {g: ratio_ci(rows, n, "vqpt2wL2+r128", g) for g in GAMMAS}
            rh = ratio_ci(rows, n, "hyb512pt2wL2", 4)
            rs = ratio_ci(rows, n, "snapkv4096", 4)
            pt["ratio_vs_rq2"][n] = {"vq_r128": rv, "hyb512_g4": rh}
            pt["ratio_vs_snap4096"][n] = rs
            L.append(f"| {LABEL.get(n, n)} | {ts[3]:.2f} | {ts[4]:.2f} | {ts[6]:.2f} | {rv[3][0]:.3f} / {rv[4][0]:.3f} / "
                     f"{rv[6][0]:.3f} [{rv[4][1]:.3f}, {rv[4][2]:.3f}] | {rh[0]:.3f} [{rh[1]:.3f}, {rh[2]:.3f}] | "
                     f"{rs[0]:.3f} |")
        L.append("")
        rec["points"][f"{task}_ctx{ctx}"] = pt
    txt = "\n".join(L)
    print(txt)
    if a.save:
        open(OUTMD, "w").write(txt + "\n")
        sys.path.insert(0, REPO)
        from tools.results import save_result
        save_result("E7", "llama8b_magicvq_v8_1bit_tau",
                    config={"model": "Llama-3.1-8B-Instruct", "setting": "tau of 1-bit VQ KV drafts (nested prefix of "
                            "t2w, dedicated t1w, nested-trained t12w) vs 2-bit t2w and SnapKV, v6 samples",
                            "points": [(t, c) for t, c, _ in POINTS], "gammas": GAMMAS, "drafts_1bit": ONE,
                            "codebooks": {"t2w": f"{CBD}/cb_d8K256x2_post_cn_t2w.pt", "t1w": f"{CBD}/cb_d8K256x2_post_cn_t1w.pt",
                                          "t12w": f"{CBD}/cb_d8K256x2_post_cn_t12w.pt"}},
                    metrics=json.loads(json.dumps(rec, default=list)), model_dir="/data/models/llama3.1_8b_instruct")
        print("saved", OUTMD)


if __name__ == "__main__":
    main()
