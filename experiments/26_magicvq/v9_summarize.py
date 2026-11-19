"""E7 magicvq v9: nested-prefix KV draft codebook retry (t12w_v2*), held-out attention loss per layer at L1 / L2 and tau.

Held-out: rqd_train.py --eval_cb jsons (R 128, held-out seqs 10 / 14 / 18, eval_off 8192). References t1w / t2w / t12w
(v8) from rqd_eval_v8_lv{1,2}_L*.json; v9 variants from rqd_eval_<tag>*.json given on the command line (--evals).
Train logs: per-layer kept flag and separate held-out L1 / L2 before / after (nested runs, v9 code).
tau: lb_tau / tau_eval records of the v9 runs joined with the v6 records per sample (anchor vqpt2wL2+r128 must match).
Usage: python experiments/26_magicvq/v9_summarize.py --evals 'rqd_eval_v9p_*' [--layers 1 4 ...] [--tau] [--save]
"""
import argparse
import glob
import json
import os
import random

ROOT = "/data/lsh/results_accept/magicvq"
CBD = f"{ROOT}/cb"
TRAIN = "/data/lsh/ckpts/magicvq_v6/train"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = f"{REPO}/results/E7/llama8b_magicvq_v9_nest"


def load_evals(pats):
    """-> {level: {tag: {layer: attn}}}"""
    per = {1: {}, 2: {}}
    files = sorted(glob.glob(f"{CBD}/rqd_eval_v8_lv*_L*.json"))
    for p in pats:
        files += sorted(glob.glob(f"{CBD}/{p}.json"))
    for f in files:  # later files override earlier ones for the same (tag, layer)
        d = json.load(open(f))
        lv = d["args"]["level"]
        for l, r in d["per_layer"].items():
            for t, v in r.items():
                per[lv].setdefault(t, {})[int(l)] = v["attn"]
    return per


def train_log(tag):
    out = {}
    for f in glob.glob(f"{TRAIN}/{tag}/log_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["layer"]] = r
    return out


def tau(rows, n, g, idx=None):
    idx = range(len(rows)) if idx is None else idx
    return sum(rows[i][n][g][0] for i in idx) / sum(rows[i][n][g][1] for i in idx)


def ratio_ci(rows, a, b, g, nb=2000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    bs = sorted(tau(rows, a, g, ix) / tau(rows, b, g, ix)
                for ix in ([rnd.randrange(n) for _ in range(n)] for _ in range(nb)))
    return tau(rows, a, g) / tau(rows, b, g), bs[int(0.05 * nb)], bs[int(0.95 * nb) - 1]


PILOT = [1, 4, 8, 12, 16, 20, 24, 28]
POINTS = [  # task, ctx, [(v9 file, v6 file)]
    ("syn_quote", 32768, [(f"lb/tau_v9{r}_syn_quote_ctx32768.json", f"lb/tau_v6{r}_syn_quote_ctx32768.json") for r in "ab"]),
    ("hotpotqa_q", 16384, [(f"lb/tau_v9{r}_hotpotqa_q_ctx16384.json", f"lb/tau_v6{r}_hotpotqa_q_ctx16384.json")
                           for r in ("a", "b", "a2")]),
    ("gov_report", 16384, [(f"lb/tau_v9{r}_gov_report_ctx16384.json", f"lb/tau_v6{r}_gov_report_ctx16384.json") for r in "ab"]),
    ("gov_report", 32768, [(f"lb/tau_v9{r}_gov_report_ctx32768.json", f"lb/tau_v6{r}_gov_report_ctx32768.json") for r in "ab"]),
    ("pg19", 32768, [("tau/tau_v9_ctx32768.json", "tau/tau_v6_ctx32768.json")]),
]


def load_point(pairs, G=(3, 4, 6)):
    rows, chk = [], {"compared": 0, "mismatch": 0}
    for f9, f6 in pairs:
        p9, p6 = f"{ROOT}/{f9}", f"{ROOT}/{f6}"
        if not os.path.exists(p9):
            continue
        r9, r6 = json.load(open(p9)), json.load(open(p6))
        key = (lambda r: r["id"]) if "id" in r9["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        m6 = {key(r): r for r in r6["per_prompt"]}
        for r in r9["per_prompt"]:
            o = m6[key(r)]
            assert o["P"] == r["P"] and o["N"] == r["N"], (f9, key(r))
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


def heldout_section(per, tags_, layers_, name):
    tags = ["t1w", "t2w", "t12w"] + [t for t in per[1] if t.startswith("t12w_")] if tags_ is None else \
        ["t1w", "t2w", "t12w"] + tags_
    layers = layers_ or sorted(set.intersection(*[set(per[1][t]) & set(per[2].get(t, {})) for t in tags]))
    L = [f"## Held-out attention loss (R 128), {name}: layers {layers}", "",
         "| codebook | L1 mean | L1 mean excl. layer 1 | L1 / t1w (per-layer median) | L2 mean | L2 mean excl. layer 1 | "
         "L2 / t2w (per-layer median) | layers L2 > t2w +1 % |", "|---|---|---|---|---|---|---|---|"]
    rec = {"layers": layers, "per_layer": {}, "summary": {}}
    for t in tags:
        x1 = [per[1][t][l] for l in layers]
        x2 = [per[2][t][l] for l in layers]
        r1 = sorted(per[1][t][l] / per[1]["t1w"][l] for l in layers)
        r2 = sorted(per[2][t][l] / per[2]["t2w"][l] for l in layers)
        nl = [l for l in layers if l != 1]
        worse = sum(per[2][t][l] > 1.01 * per[2]["t2w"][l] for l in layers)
        s = {"L1": sum(x1) / len(x1), "L1_ex1": sum(per[1][t][l] for l in nl) / len(nl), "L1_rel_med": r1[len(r1) // 2],
             "L2": sum(x2) / len(x2), "L2_ex1": sum(per[2][t][l] for l in nl) / len(nl), "L2_rel_med": r2[len(r2) // 2],
             "n_L2_worse_1pct": worse}
        rec["summary"][t] = s
        rec["per_layer"][t] = {l: [per[1][t][l], per[2][t][l]] for l in layers}
        L.append(f"| {t} | {s['L1']:.4f} | {s['L1_ex1']:.4f} | {s['L1_rel_med']:.3f} | {s['L2']:.4f} | {s['L2_ex1']:.4f} | "
                 f"{s['L2_rel_med']:.3f} | {worse}/{len(layers)} |")
    L += ["", "Per layer (L1 / L2):", "", "| layer | " + " | ".join(tags) + " |", "|---" * (len(tags) + 1) + "|"]
    for l in layers:
        L.append(f"| {l} | " + " | ".join(f"{per[1][t][l]:.4f} / {per[2][t][l]:.4f}" for t in tags) + " |")
    L += ["", "Train logs (kept; held-out L1 / L2 before -> after, from the training run; lag: final W):", ""]
    for t in tags[3:]:
        tl = train_log(t)
        if not tl:
            continue
        rec.setdefault("train", {})[t] = tl
        cells = []
        for l in sorted(tl):
            r = tl[l]
            s = f"L{l} {'K' if r['kept'] else 'rev'}"
            if "heldout_L1L2_before" in r:
                b, f = r["heldout_L1L2_before"], r["heldout_L1L2_after"]
                s += f" L1 {b[0]:.4f}->{f[0]:.4f} L2 {b[1]:.4f}->{f[1]:.4f}"
                if len(b) > 2:
                    s += f" (t1w L1 {b[2]:.4f}, W {r.get('W_final', 0):.2f})"
            cells.append(s)
        n_l2 = sum(1 for r in tl.values() if "heldout_L1L2_before" in r and r["kept"]
                   and r["heldout_L1L2_after"][1] > r["heldout_L1L2_before"][1])
        L.append(f"- {t}: " + "; ".join(cells) + (f"  -> kept layers with held-out L2 worse than init: {n_l2}"
                                                  if any("heldout_L1L2_before" in r for r in tl.values()) else ""))
    L.append("")
    return L, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="*", default=["rqd_eval_v9*"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--tags", nargs="*", default=None, help="v9 codebook tags to show (default: all found)")
    ap.add_argument("--tau", nargs="*", default=None, metavar="TAG", help="v9 cb tags (mv_common names) for the tau table")
    ap.add_argument("--pilot_tags", nargs="*", default=None, help="tags of the pilot section (layers 1 4 ... 28)")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    per = load_evals(a.evals)
    L, rec = [], {"heldout": {}}
    sections = [("full", a.tags, a.layers)]
    if a.pilot_tags:
        sections.insert(0, ("pilot", a.pilot_tags, PILOT))
    for name, tg, ly in sections:
        Ls, r = heldout_section(per, tg, ly, name)
        L += Ls
        rec["heldout"][name] = r
    l1 = {t: train_log(t).get(1) for t in ("t12w_v2l1w3", "t12w_v2l1w10")}
    if any(l1.values()):  # layer-1 Pareto points (warm t2w, nested raw W, 1000 steps, layer 1 only)
        g = lambda t, k: per[k].get(t, {}).get(1)
        L += ["## Layer 1 (held-out L1 / L2): nested W trade-off", "",
              "| codebook | L1 | L2 |", "|---|---|---|"]
        for t in ("t1w", "t2w", "t12w", "t12w_v2w03", "t12w_v2w10", "t12w_v2lag3"):
            if g(t, 1) is not None and g(t, 2) is not None:
                L.append(f"| {t} | {g(t, 1):.4f} | {g(t, 2):.4f} |")
        for t, r in l1.items():
            if r:
                L.append(f"| {t} (W = {r['W_final']:g}, train log) | {r['heldout_L1L2_after'][0]:.4f} | "
                         f"{r['heldout_L1L2_after'][1]:.4f} |")
        rec["layer1_pareto"] = {t: r and r["heldout_L1L2_after"] for t, r in l1.items()}
        L.append("")
    if a.tau:
        L += ["", "## tau (greedy, teacher-forced rounds; same samples as v6 / v8; ratio = paired bootstrap 90 % CI)", ""]
        rec["tau"] = {}
        refs = {}
        for t in a.tau:
            for lv, ref_vq, ref_h in ((1, "vqpt1wL1+r128", "hyb512pt1wL1"), (2, "vqpt2wL2+r128", "hyb512pt2wL2")):
                refs[f"vqp{t}L{lv}+r128"] = ref_vq
                refs[f"hyb512p{t}L{lv}"] = ref_h
        for task, ctx, pairs in POINTS:
            rows, chk = load_point(pairs)
            if not rows:
                continue
            L += [f"### {task} ctx {ctx} (n = {len(rows)}; anchors compared {chk['compared']}, mismatch {chk['mismatch']})", "",
                  "| draft | tau g3 | tau g4 | tau g6 | vs reference g4 [CI] |", "|---|---|---|---|---|"]
            pt = {"n": len(rows), "anchor_check": chk, "tau": {}, "ratio_g4": {}}
            order = ["vqpt1wL1+r128", "hyb512pt1wL1", "vqpt2wL1+r128", "hyb512pt2wL1", "vqpt2wL2+r128", "hyb512pt2wL2"]
            order += [n for n in refs]
            for n in order:
                if not all(n in r for r in rows):
                    continue
                t3, t4, t6 = (tau(rows, n, g) for g in (3, 4, 6))
                pt["tau"][n] = {3: t3, 4: t4, 6: t6}
                cell = ""
                if n in refs and all(refs[n] in r for r in rows):
                    r0, lo, hi = ratio_ci(rows, n, refs[n], 4)
                    pt["ratio_g4"][n] = [refs[n], r0, lo, hi]
                    cell = f"{r0:.3f} [{lo:.3f}, {hi:.3f}] vs {refs[n]}"
                L.append(f"| {n} | {t3:.3f} | {t4:.3f} | {t6:.3f} | {cell} |")
            L.append("")
            rec["tau"][f"{task}_{ctx}"] = pt
    if a.tau:  # go criterion: L1 tau >= t1w tau - 1 %, L2 tau >= t2w tau - 1 % (g4, every point, r128 and hybrid)
        L += ["## Go criterion (tau g4 ratio vs dedicated reference; go = every ratio >= 0.99)", "",
              "| point | " + " | ".join(n for n in refs) + " |", "|---" * (len(refs) + 1) + "|"]
        ok = True
        for k, pt in rec["tau"].items():
            cells = []
            for n in refs:
                if n in pt["ratio_g4"]:
                    r0 = pt["ratio_g4"][n][1]
                    ok &= r0 >= 0.99
                    cells.append(f"{r0:.3f}")
                else:
                    cells.append("-")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        ok = ok if rec["tau"] else None
        rec["go"] = ok
        L += ["", f"**go = {ok}** (points evaluated: {len(rec['tau'])} / {len(POINTS)})", ""]
    L = ["# E7 magicvq v9: nested-prefix KV draft codebook retry (t12w_v2*)", "",
         "- Held-out: rqd_train.py --eval_cb (R 128, held-out seqs 10 / 14 / 18, RoPE offset 8192); attention-output "
         "loss after o_proj, pooled ratio. t1w / t2w / t12w references from the v8 eval files.",
         "- tau: greedy, teacher-forced rounds, same samples as v6 / v8 (--only_run); references t1w / t2w drafts from "
         "the v6 records; anchor vqpt2wL2+r128 rerun and compared sample by sample.", ""] + L
    print("\n".join(L))
    if a.save:
        import sys
        sys.path.insert(0, REPO)
        from tools.results import save_result
        open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
        p = save_result("E7", os.path.basename(OUT) + "_summary", vars(a), json.loads(json.dumps(rec, default=str)),
                        "/data/models/llama3.1_8b_instruct", notes="v9 nested-prefix retry; see _tables.md")
        print("saved", p, OUT + "_tables.md")


if __name__ == "__main__":
    main()
