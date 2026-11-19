"""E7 magicvq v10: end-to-end (output-distribution) fine-tuned KV draft codebooks (rqd_e2e.py) - training curves and tau.

Training: <CBDIR>/cb_d8K256x2_post_cn_<tag>.jsonl (held-out KL / top-1 / TV / CE at L1 and L2 every eval_every steps).
tau: lb_tau / tau_eval records of the v10 runs (v10_tau.sh) joined per sample with the v6 records (anchors
vqpt2wL2+r128 / hyb512pt2wL2 must reproduce the v6 values sample by sample; t1w L1 references from v6).
Usage: python experiments/26_magicvq/v10_summarize.py --train TAG ... [--tau TAG ...] [--run v10] [--save]
"""
import argparse
import json
import os
import random

ROOT = "/data/lsh/results_accept/magicvq"
CBD = f"{ROOT}/cb"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = f"{REPO}/results/E7/llama8b_magicvq_v10_e2e"


def tau(rows, n, g, idx=None):
    idx = range(len(rows)) if idx is None else idx
    return sum(rows[i][n][g][0] for i in idx) / sum(rows[i][n][g][1] for i in idx)


def ratio_ci(rows, a, b, g, nb=2000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    bs = sorted(tau(rows, a, g, ix) / tau(rows, b, g, ix)
                for ix in ([rnd.randrange(n) for _ in range(n)] for _ in range(nb)))
    return tau(rows, a, g) / tau(rows, b, g), bs[int(0.05 * nb)], bs[int(0.95 * nb) - 1]


def points(run):
    return [
        ("syn_quote", 32768, [(f"lb/tau_{run}{r}_syn_quote_ctx32768.json", f"lb/tau_v6{r}_syn_quote_ctx32768.json") for r in "ab"]),
        ("hotpotqa_q", 16384, [(f"lb/tau_{run}{r}_hotpotqa_q_ctx16384.json", f"lb/tau_v6{r}_hotpotqa_q_ctx16384.json")
                               for r in ("a", "b", "a2")]),
        ("gov_report", 16384, [(f"lb/tau_{run}{r}_gov_report_ctx16384.json", f"lb/tau_v6{r}_gov_report_ctx16384.json") for r in "ab"]),
        ("gov_report", 32768, [(f"lb/tau_{run}{r}_gov_report_ctx32768.json", f"lb/tau_v6{r}_gov_report_ctx32768.json") for r in "ab"]),
        ("pg19", 32768, [(f"tau/tau_{run}_ctx32768.json", "tau/tau_v6_ctx32768.json")]),
    ]


def load_point(pairs, G=(3, 4, 6)):
    rows, chk = [], {"compared": 0, "mismatch": 0}
    for fn, f6 in pairs:
        pn, p6 = f"{ROOT}/{fn}", f"{ROOT}/{f6}"
        if not os.path.exists(pn):
            continue
        rn, r6 = json.load(open(pn)), json.load(open(p6))
        key = (lambda r: r["id"]) if "id" in rn["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        m6 = {key(r): r for r in r6["per_prompt"]}
        for r in rn["per_prompt"]:
            o = m6[key(r)]
            assert o["P"] == r["P"] and o["N"] == r["N"], (fn, key(r))
            d = {}
            for src in (o, r):
                for n, v in src["drafts"].items():
                    g = {x: tuple(v[f"g{x}"][:2]) for x in G}
                    if n in d:
                        chk["compared"] += 1
                        chk["mismatch"] += int(d[n] != g)
                    d[n] = g
            rows.append(d)
    return rows, chk


def train_tables(tags):
    L, rec = [], {}
    for t in tags:
        p = f"{CBD}/cb_d8K256x2_post_cn_{t}.jsonl"
        if not os.path.exists(p):
            continue
        ho, tr = [], []
        for line in open(p):
            r = json.loads(line)
            if "heldout" in r:
                ho.append(r)
            elif "step" in r:
                tr.append(r)
        args = json.load(open(f"{CBD}/cb_d8K256x2_post_cn_{t}_last.json"))["args"] if os.path.exists(
            f"{CBD}/cb_d8K256x2_post_cn_{t}_last.json") else {}
        rec[t] = {"args": args, "heldout": ho, "train": tr}
        L += [f"## {t}: level {args.get('level')}, loss {args.get('loss')}, lr {args.get('lr')}, init "
              f"{os.path.basename(str(args.get('init')))}, nest {args.get('nest')}", "",
              "| step | " + " | ".join(f"{lv} {k}" for lv in ho[0]["heldout"] for k in ("KL", "top-1", "TV")) + " |",
              "|---|" + "---|" * (3 * len(ho[0]["heldout"]))]
        for r in ho:
            L.append(f"| {r['step']} | " + " | ".join(f"{m['kl']:.4f} | {m['top1']:.4f} | {m['tv']:.4f}"
                                                     for m in r["heldout"].values()) + " |")
        L.append("")
    return L, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--tau", nargs="*", default=None, metavar="TAG")
    ap.add_argument("--run", nargs="+", default=["v10"])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    L, rec = train_tables(a.train)
    L = ["# E7 magicvq v10: end-to-end output-distribution fine-tuning of the VQ KV draft codebooks", "",
         "Held-out (rqd_e2e.py): disjoint sequences, fixed RoPE offset and positions; metrics of the step-1 draft vs the "
         "target at L1 (stage 1) and L2 (stages 1+2), R 128.", ""] + L
    out = {"train": rec}
    if a.tau:
        L += ["## tau (same samples as v6; ratio vs t2w at the same draft form, 90% bootstrap CI)", "",
              "| task | ctx | n | anchor check | draft | gamma | tau | ref | ref tau | ratio [90% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        out["tau"] = {}
        for task, ctx, pairs in [x for run in a.run for x in points(run)]:
            rows, chk = load_point(pairs)
            if not rows:
                continue
            for t in a.tau:
                for form, lv, ref in (("vqp{}L2+r128", 2, "vqpt2wL2+r128"), ("hyb512p{}L2", 2, "hyb512pt2wL2"),
                                      ("vqp{}L1+r128", 1, "vqpt1wL1+r128"), ("hyb512p{}L1", 1, "hyb512pt1wL1")):
                    n = form.format(t)
                    if n not in rows[0]:
                        continue
                    for g in ((4,) if n.startswith("vq") else (4, 6)):
                        r_, lo, hi = ratio_ci(rows, n, ref, g)
                        L.append(f"| {task} | {ctx} | {len(rows)} | {chk['mismatch']}/{chk['compared']} | {n} | {g} | "
                                 f"{tau(rows, n, g):.3f} | {ref} | {tau(rows, ref, g):.3f} | {r_:.3f} [{lo:.3f}, {hi:.3f}] |")
                        out["tau"][f"{task}_{ctx}_{n}_g{g}"] = {"tau": tau(rows, n, g), "ref": ref,
                                                                "ref_tau": tau(rows, ref, g), "ratio": r_,
                                                                "ci90": [lo, hi], "n": len(rows), "anchor_check": chk}
        L.append("")
    print("\n".join(L))
    if a.save:
        open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
        import sys
        sys.path.insert(0, REPO)
        from tools.results import save_result
        save_result("E7", "llama8b_magicvq_v10_e2e" + ("_tau" if a.tau else "_train"),
                    config={"setting": "E7 v10 e2e output-distribution fine-tune of VQ KV draft codebooks",
                            "train_tags": a.train, "tau_tags": a.tau, "runs": a.run},
                    metrics=out, model_dir="/data/models/llama3.1_8b_instruct")


if __name__ == "__main__":
    main()
