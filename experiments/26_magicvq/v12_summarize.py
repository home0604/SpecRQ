"""E7 magicvq v12: tau of drafts with INT4 weights (E4 GPTQ g128 Marlin checkpoint, dequantized dense) + compressed KV vs
the same drafts with bf16 weights, on the v6 samples (v12_tau.sh). Target = bf16 weights + bf16 KV in both cases.

Per point the v12 records are joined per sample with the v6 records (anchors snapkv*, vqpt2wL2+r128, hyb512pt2wL2) and
the v11 records (anchor vqpe2enkl_w10L1+r128): every plain (bf16-weight) draft present in an older record must reproduce
its round-walk (tokens, rounds) at every common gamma (anchor check = mismatches / compared).
Tables: tau bf16 -> INT4 per gamma; retention tau_INT4 / tau_bf16 [90% bootstrap CI over samples]; VQ / SnapKV tau
ratio with bf16 vs INT4 weights and their ratio (does weight error hit VQ and SnapKV drafts equally?); analytic speed
model note (NOT a measurement) with the draft step reduced by the INT4 weight-read saving.
Usage: python experiments/26_magicvq/v12_summarize.py [--run v12] [--save]
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v10_summarize import REPO, ROOT  # noqa: E402

OUT = f"{REPO}/results/E7/llama8b_magicvq_v12_w4"
GAMMAS = (1, 2, 3, 4, 5, 6)
VQ = ("vqpt2wL2+r128", "hyb512pt2wL2", "vqpe2enkl_w10L1+r128")
BEST_SNAP = {("syn_quote", 32768): "snapkv4096", ("gov_report", 32768): "snapkv4096", ("pg19", 32768): "snapkv4096",
             ("hotpotqa_q", 16384): "snapkv2048", ("gov_report", 16384): "snapkv4096"}  # best SnapKV in v6e2e / v8 e2e


def points(run):
    lb = lambda r, t, c: [f"lb/tau_{x}{r}_{t}_ctx{c}.json" for x in (run, "v6", "v11")]
    return [
        ("syn_quote", 32768, [lb(r, "syn_quote", 32768) for r in "ab"]),
        ("gov_report", 32768, [lb(r, "gov_report", 32768) for r in "ab"]),
        ("pg19", 32768, [[f"tau/tau_{x}_ctx32768.json" for x in (run, "v6", "v11")]]),
        ("hotpotqa_q", 16384, [lb(r, "hotpotqa_q", 16384) for r in ("a", "b", "a2")]),
        ("gov_report", 16384, [lb(r, "gov_report", 16384) for r in "ab"]),
    ]


def load_point(groups):
    rows, chk = [], {"compared": 0, "mismatch": 0, "mismatch_names": [], "anchored": []}
    for files in groups:
        paths = [f"{ROOT}/{fn}" for fn in files]
        if not os.path.exists(paths[0]):
            continue
        recs = [json.load(open(p)) for p in paths if os.path.exists(p)]
        key = (lambda r: r["id"]) if "id" in recs[0]["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        maps = [{key(r): r for r in x["per_prompt"]} for x in recs[1:]]
        for r in recs[0]["per_prompt"]:
            d = {n: {g: tuple(v[f"g{g}"][:2]) for g in GAMMAS} for n, v in r["drafts"].items()}
            for m in maps:
                o = m.get(key(r))
                if o is None:
                    continue
                assert o["P"] == r["P"] and o["N"] == r["N"], (files[0], key(r))
                for n, v in o["drafts"].items():
                    if n not in d:
                        continue
                    for g in GAMMAS:
                        if f"g{g}" in v:
                            chk["compared"] += 1
                            if tuple(v[f"g{g}"][:2]) != d[n][g]:
                                chk["mismatch"] += 1
                                chk["mismatch_names"].append(n)
                    chk["anchored"].append(n)
            rows.append(d)
    chk["mismatch_names"] = sorted(set(chk["mismatch_names"]))
    chk["anchored"] = sorted(set(chk["anchored"]))
    return rows, chk


def tau(rows, n, g, idx=None):
    idx = range(len(rows)) if idx is None else idx
    return sum(rows[i][n][g][0] for i in idx) / sum(rows[i][n][g][1] for i in idx)


def boot(rows, fn, nb=2000, seed=0):
    rnd = random.Random(seed)
    n = len(rows)
    bs = sorted(fn([rnd.randrange(n) for _ in range(n)]) for _ in range(nb))
    return fn(range(n)), bs[int(0.05 * nb)], bs[int(0.95 * nb) - 1]


f3 = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"

# ---- analytic speed model inputs (measured e2e, TP=4 RTX 4090, B16, all-active; results/E7/llama8b_magicvq_v8_1bit_e2e_tables.md)
E2E = {  # point: AR step ms, verify ms (incl. other) per gamma, draft step ms per draft (bf16 weights)
    ("syn_quote", 32768): {"ar": 26.41, "verify": {4: 31.82, 6: 33.23},
                           "draft": {"snapkv4096": 11.38, "hyb512pt2wL2": 12.18, "vqpt2wL2+r128": 11.90,
                                     "vqpe2enkl_w10L1+r128": 10.84, "exact": 26.41}},
    ("pg19", 32768): {"ar": 27.03, "verify": {4: 32.46, 6: 33.98},
                      "draft": {"snapkv4096": 11.27, "hyb512pt2wL2": 12.16, "vqpt2wL2+r128": 12.04,
                                "vqpe2enkl_w10L1+r128": 10.95, "exact": 27.03}},
}
# INT4 weight-read saving per draft step per rank: E4 batch-1 single 4090 full-model decode step dense bf16 16.75 ms vs
# Marlin INT4 6.86 ms (results/E4/l1_llama31_real_int4_marlin_baseline.json) -> (16.75 - 6.86) / 4 ranks = 2.47 ms.
DW = 2.47
DW_RANGE = (1.5, 3.5)


def analytic(res_rows):
    L = ["## Analytic note (NOT a measurement): speed model S = tau / (gamma c + v) with INT4 draft weights", "",
         "c = T_draft_step / T_AR, v = T_verify(+other) / T_AR from the measured v8 e2e records (TP=4 4090, B16, bf16 "
         "weights; results/E7/llama8b_magicvq_v8_1bit_e2e_tables.md: syn_quote 32k / pg19 32k; L1 row uses the nested "
         "RQ1+r128 t2w-stage1 kernel time, exact = AR step). INT4 draft: T_draft - dW with dW = "
         f"{DW} ms per rank per step (E4: batch-1 single-4090 decode step dense 16.75 ms vs Marlin 6.86 ms, / 4 ranks; "
         f"sensitivity dW {DW_RANGE[0]}-{DW_RANGE[1]} ms). Verify unchanged (target bf16). tau = this run's teacher-forced "
         "tau (not the e2e all-active tau). Break-even = VQ/SnapKV tau ratio at which S_VQ = S_SnapKV.", "",
         "| point | gamma | draft | c bf16 -> INT4 | v | S bf16 (model) | S INT4 (model) | S INT4 dW lo / hi | "
         "break-even VQ/Snap tau ratio bf16 -> INT4 | measured tau ratio bf16 -> INT4 |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for pt, e in E2E.items():
        rows = res_rows.get(pt)
        if not rows:
            continue
        sn = BEST_SNAP[pt]
        for g in (4, 6):
            v = e["verify"][g] / e["ar"]
            for n in ("exact", sn) + VQ:
                if n not in rows[0] or n + "@w4" not in rows[0]:
                    continue
                td = e["draft"][n]
                cb, cw = td / e["ar"], (td - DW) / e["ar"]
                clo, chi = (td - DW_RANGE[0]) / e["ar"], (td - DW_RANGE[1]) / e["ar"]
                tb, tw = tau(rows, n, g), tau(rows, n + "@w4", g)
                sb, sw = tb / (g * cb + v), tw / (g * cw + v)
                slo, shi = tw / (g * clo + v), tw / (g * chi + v)
                be = mr = "-"
                rec = {"c_bf16": cb, "c_w4": cw, "v": v, "tau_bf16": tb, "tau_w4": tw, "S_bf16": sb, "S_w4": sw,
                       "S_w4_dw_lo_hi": [slo, shi]}
                if n in VQ and sn + "@w4" in rows[0]:
                    cs_b, cs_w = e["draft"][sn] / e["ar"], (e["draft"][sn] - DW) / e["ar"]
                    beb, bew = (g * cb + v) / (g * cs_b + v), (g * cw + v) / (g * cs_w + v)
                    mrb, mrw = tb / tau(rows, sn, g), tw / tau(rows, sn + "@w4", g)
                    be, mr = f"{beb:.3f} -> {bew:.3f}", f"{mrb:.3f} -> {mrw:.3f}"
                    rec.update({"breakeven_bf16": beb, "breakeven_w4": bew, "tau_ratio_bf16": mrb, "tau_ratio_w4": mrw})
                L.append(f"| {pt[0]} {pt[1] // 1024}k | {g} | {n} | {cb:.3f} -> {cw:.3f} | {v:.3f} | {sb:.3f} | {sw:.3f} | "
                         f"{slo:.3f} / {shi:.3f} | {be} | {mr} |")
                out[f"{pt[0]}_{pt[1]}_{n}_g{g}"] = rec
    return L + [""], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v12")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    L = ["# E7 magicvq v12: INT4-weight drafts (QuantSpec-style) + compressed KV - greedy acceptance tau", "",
         "- Target: Llama-3.1-8B-Instruct bf16 weights + bf16 KV (unchanged). Draft weights '@w4' = E4 INT4 GPTQ g128 "
         "(pile 128x2048, Marlin checkpoint) dequantized once to dense fp16 (same routine as int4dense in "
         "experiments/13_int4_marlin/eval_quality.py: wt2 PPL 7.14 vs bf16 6.75) and cast to bf16 for the draft forward "
         "(cast rel. error ~0.17% vs INT4 rel. error 12-17%); norms, embeddings and lm_head bf16. All '@w4' drafts share "
         "these weights. The context KV the draft reads is the target's KV (bf16 weights) in the draft's KV format; the "
         "draft's in-round K/V come from the INT4 weights.",
         "- Samples = v6 samples (--only_run v6*, pg19 = the 36 v6 prompts); plain drafts must reproduce v6 / v11 per sample.",
         "- exact@w4 = weights-only draft (uncompressed bf16 KV); exact = bf16-weight G-stream reference.",
         f"- SnapKV reference for the VQ/SnapKV ratio = per-point best e2e budget: {', '.join(f'{k[0]} {k[1]//1024}k {v}' for k, v in BEST_SNAP.items())}.",
         ""]
    out = {"points": {}}
    res_rows = {}
    T1 = ["## tau: bf16 weights -> INT4 weights (pooled tokens / rounds)", "",
          "| task | ctx | n | anchor check | draft | " + " | ".join(f"g{g}" for g in GAMMAS) + " |",
          "|---|---|---|---|---|" + "---|" * len(GAMMAS)]
    T2 = ["## retention tau_INT4 / tau_bf16 [90% bootstrap CI over samples]", "",
          "| task | ctx | draft | g2 | g3 | g4 | g6 |", "|---|---|---|---|---|---|---|"]
    T3 = ["## VQ / SnapKV tau ratio with bf16 vs INT4 draft weights (double ratio = (VQ/Snap)_INT4 / (VQ/Snap)_bf16 = "
          "retention_VQ / retention_Snap; < 1: INT4 weights hurt the VQ draft more)", "",
          "| task | ctx | VQ draft | SnapKV | gamma | VQ/Snap bf16 [CI] | VQ/Snap INT4 [CI] | double ratio [CI] |",
          "|---|---|---|---|---|---|---|---|"]
    for task, ctx, groups in points(a.run):
        rows, chk = load_point(groups)
        if not rows:
            continue
        res_rows[(task, ctx)] = rows
        P = out["points"][f"{task}_{ctx}"] = {"n": len(rows), "anchor_check": chk, "tau": {}, "retention": {},
                                              "vq_snap": {}}
        bases = [n for n in rows[0] if "@" not in n and n + "@w4" in rows[0]]
        for n in bases:
            for nn in (n, n + "@w4"):
                P["tau"][nn] = {g: tau(rows, nn, g) for g in GAMMAS}
            T1.append(f"| {task} | {ctx} | {len(rows)} | {chk['mismatch']}/{chk['compared']} | {n} | " +
                      " | ".join(f"{tau(rows, n, g):.2f} -> {tau(rows, n + '@w4', g):.2f}" for g in GAMMAS) + " |")
            rets = {}
            for g in (2, 3, 4, 6):
                rets[g] = boot(rows, lambda ix: tau(rows, n + "@w4", g, ix) / tau(rows, n, g, ix))
            P["retention"][n] = rets
            T2.append(f"| {task} | {ctx} | {n} | " + " | ".join(f3(rets[g]) for g in (2, 3, 4, 6)) + " |")
        sn = BEST_SNAP[(task, ctx)]
        snaps = [s for s in [sn] + sorted(bases) if s in bases and s.startswith("snapkv")]
        snaps = list(dict.fromkeys(snaps))
        for v in VQ:
            if v not in bases:
                continue
            for s in snaps:
                for g in (3, 4, 6):
                    rb = boot(rows, lambda ix: tau(rows, v, g, ix) / tau(rows, s, g, ix))
                    rw = boot(rows, lambda ix: tau(rows, v + "@w4", g, ix) / tau(rows, s + "@w4", g, ix))
                    dr = boot(rows, lambda ix: (tau(rows, v + "@w4", g, ix) / tau(rows, s + "@w4", g, ix)) /
                              (tau(rows, v, g, ix) / tau(rows, s, g, ix)))
                    P["vq_snap"][f"{v}|{s}|g{g}"] = {"bf16": rb, "w4": rw, "double": dr}
                    if s == sn:
                        T3.append(f"| {task} | {ctx} | {v} | {s} | {g} | {f3(rb)} | {f3(rw)} | {f3(dr)} |")
    TA, out["analytic"] = analytic(res_rows)
    L += T1 + [""] + T2 + [""] + T3 + [""] + TA
    print("\n".join(L))
    if a.save:
        open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
        sys.path.insert(0, REPO)
        from tools.results import save_result
        save_result("E7", "llama8b_magicvq_v12_w4_tau",
                    config={"setting": "E7 v12: INT4-weight (GPTQ g128 dense dequant) drafts + compressed KV, tau on v6 samples",
                            "run": a.run, "w4_source": "/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors",
                            "marlin_ckpt": "/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048/marlin_linears.pt",
                            "best_snap": {f"{k[0]}_{k[1]}": v for k, v in BEST_SNAP.items()}, "analytic_dW_ms": DW,
                            "analytic_e2e_inputs": {f"{k[0]}_{k[1]}": v for k, v in E2E.items()}},
                    metrics=out, model_dir="/data/models/llama3.1_8b_instruct")


if __name__ == "__main__":
    main()
