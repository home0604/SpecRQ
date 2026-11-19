"""E7 magicvq v11: nested end-to-end output-KL fine-tune of the VQ KV draft codebook (rqd_e2e.py --nest W, loss
W KL(L1) + KL(L2)) - held-out curves vs the dedicated e2e codebooks (e2ekl1b at L1, e2ekl2b at L2) and tau.

tau: v11 records (v11_tau.sh, --only_run on the v6 samples) joined per sample with the v6 records (t2w / t1w references,
anchors vqpt2wL2+r128 / hyb512pt2wL2 must match sample by sample), the v10 records (e2ekl2b) and the v10l1 records
(e2ekl1b). Every draft name present in more than one record file is compared (anchor check = mismatches / compared).
L1 gain retention = (tau_new / tau_t1w - 1) / (tau_e2ekl1b / tau_t1w - 1), bootstrap over samples (same resample for
all three drafts); n/a where the e2ekl1b gain over t1w is < 1 % (pg19 hybrids).
Usage: python experiments/26_magicvq/v11_summarize.py --train TAG ... [--tau TAG ...] [--run v11] [--save]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v10_summarize import ROOT, CBD, REPO, tau, train_tables  # noqa: E402

OUT = f"{REPO}/results/E7/llama8b_magicvq_v11_nestkl"
HARD = {("syn_quote", 32768), ("hotpotqa_q", 16384), ("gov_report", 16384), ("gov_report", 32768)}
REF_L1_KL = ("e2ekl1b", "L1")  # dedicated held-out references (best checkpoint)
REF_L2_KL = ("e2ekl2b", "L2")


def points(run):
    lb = lambda r, t, c: [f"lb/tau_{x}{r}_{t}_ctx{c}.json" for x in (run, "v6", "v10", "v10l1")]
    return [
        ("syn_quote", 32768, [lb(r, "syn_quote", 32768) for r in "ab"]),
        ("hotpotqa_q", 16384, [lb(r, "hotpotqa_q", 16384) for r in ("a", "b", "a2")]),
        ("gov_report", 16384, [lb(r, "gov_report", 16384) for r in "ab"]),
        ("gov_report", 32768, [lb(r, "gov_report", 32768) for r in "ab"]),
        ("pg19", 32768, [[f"tau/tau_{x}_ctx32768.json" for x in (run, "v6", "v10", "v10l1")]]),
    ]


def load_point(groups, G=(3, 4, 6)):
    """groups: per sample group, [new file, v6, v10, v10l1]; rows keyed on the new file's samples."""
    rows, chk = [], {"compared": 0, "mismatch": 0, "mismatch_names": []}
    for files in groups:
        paths = [f"{ROOT}/{fn}" for fn in files]
        if not os.path.exists(paths[0]):
            continue
        recs = [json.load(open(p)) for p in paths]
        key = (lambda r: r["id"]) if "id" in recs[0]["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        maps = [{key(r): r for r in x["per_prompt"]} for x in recs[1:]]
        for r in recs[0]["per_prompt"]:
            srcs = [r] + [m[key(r)] for m in maps]
            for o in srcs[1:]:
                assert o["P"] == r["P"] and o["N"] == r["N"], (files[0], key(r))
            d = {}
            for src in srcs:
                for n, v in src["drafts"].items():
                    g = {x: tuple(v[f"g{x}"][:2]) for x in G}
                    if n in d:
                        chk["compared"] += 1
                        if d[n] != g:
                            chk["mismatch"] += 1
                            chk["mismatch_names"].append(n)
                    d[n] = g
            rows.append(d)
    chk["mismatch_names"] = sorted(set(chk["mismatch_names"]))
    return rows, chk


def boot(rows, fn, nb=2000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    bs = sorted(fn([rnd.randrange(n) for _ in range(n)]) for _ in range(nb))
    return fn(range(n)), bs[int(0.05 * nb)], bs[int(0.95 * nb) - 1]


def best_heldout(tag, lv):
    """held-out metrics at the checkpoint saved as <tag>.pt (best_step of its json)."""
    j = json.load(open(f"{CBD}/cb_d8K256x2_post_cn_{tag}_last.json"))
    bs = j["best"][1]
    for h in j["hist"]:
        if "heldout" in h and h["step"] == bs:
            return h["heldout"][lv], bs
    raise KeyError(tag)


def heldout_cmp(tags):
    r1, _ = best_heldout(*REF_L1_KL)
    r2, _ = best_heldout(*REF_L2_KL)
    t1w_l1 = json.loads(open(f"{CBD}/cb_d8K256x2_post_cn_e2ekl1b.jsonl").readline())["heldout"]["L1"]
    t2w_l2 = json.loads(open(f"{CBD}/cb_d8K256x2_post_cn_e2ekl2b.jsonl").readline())["heldout"]["L2"]
    L = ["## held-out vs dedicated e2e codebooks (KL; L1 gain retention = (KL_t1w - KL) / (KL_t1w - KL_e2ekl1b))", "",
         f"References: t1w L1 KL {t1w_l1['kl']:.4f}, e2ekl1b L1 KL {r1['kl']:.4f} (top-1 {r1['top1']:.4f}); "
         f"t2w L2 KL {t2w_l2['kl']:.4f}, e2ekl2b L2 KL {r2['kl']:.4f} (top-1 {r2['top1']:.4f}).", "",
         "| codebook | step | L1 KL | L1 top-1 | L1 KL / e2ekl1b | L1 gain retention | L2 KL | L2 top-1 | L2 KL / e2ekl2b |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {"ref": {"t1w_L1": t1w_l1, "e2ekl1b_L1": r1, "t2w_L2": t2w_l2, "e2ekl2b_L2": r2}, "tags": {}}
    for t in tags:
        p = f"{CBD}/cb_d8K256x2_post_cn_{t}_last.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        out["tags"][t] = {"best_step": j["best"][1], "steps": {}}
        for h in j["hist"]:
            if "heldout" not in h:
                continue
            a, b = h["heldout"]["L1"], h["heldout"]["L2"]
            ret = (t1w_l1["kl"] - a["kl"]) / (t1w_l1["kl"] - r1["kl"])
            mark = " (best)" if h["step"] == j["best"][1] else ""
            L.append(f"| {t} | {h['step']}{mark} | {a['kl']:.4f} | {a['top1']:.4f} | {a['kl'] / r1['kl']:.3f} | {ret:.2f} | "
                     f"{b['kl']:.4f} | {b['top1']:.4f} | {b['kl'] / r2['kl']:.3f} |")
            out["tags"][t]["steps"][h["step"]] = {"L1": a, "L2": b, "L1_retention": ret,
                                                   "L1_kl_vs_e2ekl1b": a["kl"] / r1["kl"],
                                                   "L2_kl_vs_e2ekl2b": b["kl"] / r2["kl"]}
    return L + [""], out


def tau_tables(tags, runs):
    L = ["## tau (same samples as v6; 90% bootstrap CI over samples)", "",
         "L2 rows: ratio vs t2w and vs e2ekl2b. L1 rows: ratio vs t1w and vs e2ekl1b, and retention of the e2ekl1b gain "
         "over t1w. go: L2 vs e2ekl2b >= 0.99 at every row; L1 retention >= 2/3 at the hard points (syn_quote, hotpotqa, "
         "gov16, gov32).", "",
         "| task | ctx | n | anchor check | draft | gamma | tau | vs t2w / t1w [90% CI] | vs e2ekl2b / e2ekl1b [90% CI] | "
         "L1 retention [90% CI] | go row |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for task, ctx, groups in [x for run in runs for x in points(run)]:
        rows, chk = load_point(groups)
        if not rows:
            continue
        hard = (task, ctx) in HARD
        for t in tags:
            for form, lv in (("vqp{}L2+r128", 2), ("hyb512p{}L2", 2), ("vqp{}L1+r128", 1), ("hyb512p{}L1", 1)):
                n = form.format(t)
                if n not in rows[0]:
                    continue
                base = form.format("t2w" if lv == 2 else "t1w")
                ded = form.format("e2ekl2b" if lv == 2 else "e2ekl1b")
                for g in ((4,) if n.startswith("vq") else (4, 6)):
                    rb = boot(rows, lambda ix: tau(rows, n, g, ix) / tau(rows, base, g, ix))
                    rd = boot(rows, lambda ix: tau(rows, n, g, ix) / tau(rows, ded, g, ix)) if ded in rows[0] else None
                    ret = None
                    if lv == 1 and ded in rows[0]:
                        def rfn(ix):
                            den = tau(rows, ded, g, ix) / tau(rows, base, g, ix) - 1
                            return (tau(rows, n, g, ix) / tau(rows, base, g, ix) - 1) / den if den else float("nan")
                        ret = boot(rows, rfn)
                        if abs(tau(rows, ded, g) / tau(rows, base, g) - 1) < 0.01:  # e2ekl1b gain ~0: undefined
                            ret = None
                    if lv == 2:
                        go = None if rd is None else rd[0] >= 0.99
                    else:
                        go = (ret[0] >= 2 / 3) if (hard and ret is not None) else None
                    f3 = lambda x: "-" if x is None else f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
                    f2 = lambda x: "-" if x is None else f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]"
                    L.append(f"| {task} | {ctx} | {len(rows)} | {chk['mismatch']}/{chk['compared']} | {n} | {g} | "
                             f"{tau(rows, n, g):.3f} | {f3(rb)} | {f3(rd)} | {f2(ret)} | "
                             f"{'-' if go is None else ('yes' if go else 'NO')} |")
                    out[f"{task}_{ctx}_{n}_g{g}"] = {
                        "tau": tau(rows, n, g), "base": base, "base_tau": tau(rows, base, g), "ratio_base": rb,
                        "ded": ded, "ded_tau": tau(rows, ded, g) if ded in rows[0] else None, "ratio_ded": rd,
                        "l1_retention": ret, "hard": hard, "go_row": go, "n": len(rows), "anchor_check": chk}
    return L + [""], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--tau", nargs="*", default=None, metavar="TAG")
    ap.add_argument("--run", nargs="+", default=["v11"])
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    Lc, cmp_ = heldout_cmp(a.train)
    Lt, rec = train_tables(a.train)
    L = ["# E7 magicvq v11: nested end-to-end output-KL fine-tune (W KL(L1) + KL(L2)) of the VQ KV draft codebook", "",
         "Held-out (rqd_e2e.py): disjoint sequences, fixed RoPE offset and positions; step-1 draft vs target at L1 "
         "(stage 1) and L2 (stages 1+2), R 128.", ""] + Lc + Lt
    out = {"heldout_cmp": cmp_, "train": rec}
    if a.tau:
        Lt2, out["tau"] = tau_tables(a.tau, a.run)
        L += Lt2
    print("\n".join(L))
    if a.save:
        open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
        sys.path.insert(0, REPO)
        from tools.results import save_result
        save_result("E7", "llama8b_magicvq_v11_nestkl" + ("_tau" if a.tau else "_train"),
                    config={"setting": "E7 v11 nested e2e output-KL fine-tune of VQ KV draft codebooks",
                            "train_tags": a.train, "tau_tags": a.tau, "runs": a.run},
                    metrics=out, model_dir="/data/models/llama3.1_8b_instruct")


if __name__ == "__main__":
    main()
