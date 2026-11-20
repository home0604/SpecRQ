"""E7 magicvq v13: tau of drafts with weights below INT4 + compressed KV, next to the v12 bf16 / INT4 rows.

Weight sources ('@<src>' suffix; target = bf16 weights + bf16 KV throughout):
  (none) bf16 target weights, w4 = GPTQ INT4 g128 (v12), k3 = C1 weight-VQ nested prefix k=3 (~3 bit),
  k2B = C1 prefix k=2 codes with draft-only (TV-trained, E3 'B') codebooks + scales (~2 bit), k2 = C1 plain prefix
  k=2 (~2 bit, weights-only draft only), g3 = GPTQ INT3 g128, g2 = GPTQ INT2 g128 (weights-only draft only).
Per point the v13 records are joined per sample with the v12 records (bf16 and @w4 rows reused, not rerun); every draft
present in both (anchor snapkv4096@w4) must reproduce its round-walk at every gamma. v12's own anchor check vs v6 / v11
is in results/E7/llama8b_magicvq_v12_w4_tables.md.
Tables: PPL per source; tau per gamma; retention tau_src / tau_bf16 [90% bootstrap CI]; double ratio
(VQ/SnapKV)_src / (VQ/SnapKV)_bf16; analytic speed model (NOT a measurement) with the draft weight-read saving scaled
by weight bytes, anchored to v12's INT4 saving.
Usage: python experiments/26_magicvq/v13_summarize.py [--run v13] [--save]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v10_summarize import REPO, ROOT  # noqa: E402
from v12_summarize import BEST_SNAP, DW, DW_RANGE, E2E, GAMMAS, VQ, boot, f3, tau  # noqa: E402

OUT = f"{REPO}/results/E7/llama8b_magicvq_v13_lowbit"
SRCS = ("w4", "k3", "k2B", "g3", "k2", "g2")
DESC = {"w4": "GPTQ INT4 g128 (v12)", "k3": "C1 nested prefix k=3", "k2B": "C1 k=2 codes + draft-only codebooks (E3 B)",
        "k2": "C1 nested prefix k=2 (plain)", "g3": "GPTQ INT3 g128", "g2": "GPTQ INT2 g128"}
# weight bits per weight for the analytic model (task-given; GPTQ sym g128 with one fp16 scale is b + 0.125)
BITS = {"": 16.0, "w4": 4.25, "g3": 3.25, "g2": 2.25}
LIN_SHAPES = [(4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (14336, 4096), (14336, 4096), (4096, 14336)]


def c1_bits(k):
    """C1 draft bits / weight incl. k codebooks (256 x 8 fp16) and one fp16 scale per output row (NestLinear.nbytes)."""
    nb = sum(k * n * kk // 8 + k * 256 * 8 * 2 + 2 * n for n, kk in LIN_SHAPES)
    return nb * 8 / sum(n * kk for n, kk in LIN_SHAPES)


BITS.update({"k3": c1_bits(3), "k2B": c1_bits(2), "k2": c1_bits(2)})
dw = lambda s: DW * (16.0 - BITS[s]) / (16.0 - BITS["w4"])  # draft weight-read saving per rank per step (ms)


def points(run):
    lb = lambda r, t, c: [f"lb/tau_{x}{r}_{t}_ctx{c}.json" for x in (run, "v12")]
    return [
        ("syn_quote", 32768, [lb(r, "syn_quote", 32768) for r in "ab"]),
        ("gov_report", 32768, [lb(r, "gov_report", 32768) for r in "ab"]),
        ("pg19", 32768, [[f"tau/tau_{x}_ctx32768.json" for x in (run, "v12")]]),
        ("hotpotqa_q", 16384, [lb(r, "hotpotqa_q", 16384) for r in ("a", "b", "a2")]),
        ("gov_report", 16384, [lb(r, "gov_report", 16384) for r in "ab"]),
    ]


def load_point(groups):
    """rows: per sample {draft: {gamma: (tokens, rounds)}} = v13 drafts + v12 drafts (union); check = overlap names."""
    rows, chk = [], {"compared": 0, "mismatch": 0, "mismatch_names": [], "anchored": [], "missing_v12": 0}
    for files in groups:
        p13, p12 = (f"{ROOT}/{fn}" for fn in files)
        if not os.path.exists(p13):
            continue
        r13, r12 = json.load(open(p13)), json.load(open(p12))
        key = (lambda r: r["id"]) if "id" in r13["per_prompt"][0] else (lambda r: (r["set"], r["idx"]))
        m12 = {key(r): r for r in r12["per_prompt"]}
        for r in r13["per_prompt"]:
            d = {n: {g: tuple(v[f"g{g}"][:2]) for g in GAMMAS} for n, v in r["drafts"].items()}
            o = m12.get(key(r))
            if o is None:
                chk["missing_v12"] += 1
                continue
            assert o["P"] == r["P"] and o["N"] == r["N"], (files[0], key(r))
            for n, v in o["drafts"].items():
                t = {g: tuple(v[f"g{g}"][:2]) for g in GAMMAS}
                if n in d:
                    for g in GAMMAS:
                        chk["compared"] += 1
                        if t[g] != d[n][g]:
                            chk["mismatch"] += 1
                            chk["mismatch_names"].append(n)
                    chk["anchored"].append(n)
                else:
                    d[n] = t
            rows.append(d)
    chk["mismatch_names"] = sorted(set(chk["mismatch_names"]))
    chk["anchored"] = sorted(set(chk["anchored"]))
    return rows, chk


def nm(n, s):
    return n if not s else f"{n}@{s}"


def analytic(res_rows):
    L = ["## Analytic note (NOT a measurement): S = tau / (gamma c + v), draft weight-read saving scaled by weight bytes",
         "",
         "c = T_draft_step / T_AR, v = T_verify(+other) / T_AR from the measured v8 e2e records (TP=4 4090, B16, bf16 "
         "weights; as v12). Draft with weight source X: T_draft - dW(X), dW(X) = dW_INT4 * (16 - b_X) / (16 - 4.25), "
         f"dW_INT4 = {DW} ms per rank per step (v12 anchor: E4 batch-1 single-4090 dense 16.75 ms vs Marlin 6.86 ms, / 4 "
         f"ranks). b_X: " + ", ".join(f"{s or 'bf16'} {BITS[s]:.3f}" for s in ("",) + SRCS) +
         " (C1: k bit + codebooks + row scales). [I] Assumes the low-bit kernel is purely weight-read bound at the INT4 "
         "efficiency; the C1 VQ decode kernel is not (E0/E4), so the C1 rows are an upper bound. Verify unchanged "
         "(target bf16). tau = teacher-forced tau of this run.", "",
         "| point | gamma | draft | S bf16 | " + " | ".join(f"S {s}" for s in SRCS) + " |",
         "|---|---|---|---|" + "---|" * len(SRCS)]
    out = {}
    for pt, e in E2E.items():
        rows = res_rows.get(pt)
        if not rows:
            continue
        sn = BEST_SNAP[pt]
        for g in (4, 6):
            v = e["verify"][g] / e["ar"]
            for n in ("exact", sn) + VQ:
                if n not in rows[0]:
                    continue
                td = e["draft"][n]
                cells, rec = [], {}
                for s in ("",) + SRCS:
                    if nm(n, s) not in rows[0]:
                        cells.append("-")
                        continue
                    c = (td - (dw(s) if s else 0.0)) / e["ar"]
                    t = tau(rows, nm(n, s), g)
                    S = t / (g * c + v)
                    rec[s or "bf16"] = {"c": c, "tau": t, "S": S, "dW_ms": dw(s) if s else 0.0}
                    if s:
                        lo = (td - dw(s) * DW_RANGE[0] / DW) / e["ar"]
                        hi = (td - dw(s) * DW_RANGE[1] / DW) / e["ar"]
                        rec[s]["S_dw_lo_hi"] = [t / (g * lo + v), t / (g * hi + v)]
                    cells.append(f"{S:.3f}")
                L.append(f"| {pt[0]} {pt[1] // 1024}k | {g} | {n} | " + " | ".join(cells) + " |")
                out[f"{pt[0]}_{pt[1]}_{n}_g{g}"] = rec
    return L + [""], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v13")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    ppl = {}
    for f in (f"{OUT}_ppl_ref.json", f"{OUT}_ppl.json"):
        if os.path.exists(f):
            ppl.update(json.load(open(f)))
    L = ["# E7 magicvq v13: draft weights below INT4 (C1 weight-VQ nested prefix, GPTQ INT3/INT2) + compressed KV - "
         "greedy acceptance tau", "",
         "- Target: Llama-3.1-8B-Instruct bf16 weights + bf16 KV (unchanged). Draft weight source '@X' = dense "
         "dequantized weights of the 7 linears per layer (cast to bf16), norms / embeddings / lm_head = target's. "
         "Context KV the draft reads = target KV in the draft's KV format; in-round K/V from the draft weights (v12 "
         "mechanism, v13_tau.py = v12_w4.py with a 1-layer GPU shadow pool). Dense dequant: tau only; no low-bit kernel.",
         "- Sources: " + "; ".join(f"{s} = {DESC[s]}" for s in SRCS) + ".",
         "- Samples = v6 samples (same as v12); bf16 and @w4 rows are the v12 records (not rerun); snapkv4096@w4 rerun "
         "as anchor. Weights-only only for k2 and g2.",
         f"- SnapKV reference for the VQ/SnapKV ratio = per-point best e2e budget (as v12).", ""]
    T0 = ["## Weight-source quality: PPL@4096 (wikitext2 test / c4 val; bf16 nn.Linear, target norms; "
          "experiments/26_magicvq/v13_prep.py ppl)", "", "| source | wikitext2 | c4 |", "|---|---|---|"]
    for t in ("bf16", "w4", "c1k5", "k3", "k2B", "g3", "k2", "g2", "c1k5_c1norms"):
        k = {"k3": "c1k3", "k2B": "c1k2B", "k2": "c1k2"}.get(t, t)
        if k in ppl:
            T0.append(f"| {t} | {ppl[k]['ppl_4096']['wikitext2']:.3f} | {ppl[k]['ppl_4096']['c4']:.3f} |")
    out = {"points": {}, "ppl_4096": ppl, "bits": BITS}
    res_rows = {}
    T1 = ["## tau (pooled tokens / rounds) per weight source", "",
          "| task | ctx | n | anchor check | draft | src | " + " | ".join(f"g{g}" for g in GAMMAS) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(GAMMAS)]
    T2 = ["## retention tau_src / tau_bf16 [90% bootstrap CI over samples]", "",
          "| task | ctx | draft | src | g2 | g4 | g6 |", "|---|---|---|---|---|---|---|"]
    T3 = ["## double ratio (VQ/SnapKV)_src / (VQ/SnapKV)_bf16 [90% CI] (= retention_VQ / retention_Snap; < 1: the "
          "low-bit weights hurt the VQ draft more)", "",
          "| task | ctx | VQ draft | SnapKV | gamma | " + " | ".join(s for s in SRCS if s not in ("k2", "g2")) + " |",
          "|---|---|---|---|---|" + "---|" * 4]
    for task, ctx, groups in points(a.run):
        rows, chk = load_point(groups)
        if not rows:
            continue
        res_rows[(task, ctx)] = rows
        P = out["points"][f"{task}_{ctx}"] = {"n": len(rows), "anchor_check": chk, "tau": {}, "retention": {},
                                              "double": {}}
        bases = [n for n in rows[0] if "@" not in n]
        for n in bases:
            for s in ("",) + SRCS:
                if nm(n, s) not in rows[0]:
                    continue
                P["tau"][nm(n, s)] = {g: tau(rows, nm(n, s), g) for g in GAMMAS}
                T1.append(f"| {task} | {ctx} | {len(rows)} | {chk['mismatch']}/{chk['compared']} | {n} | {s or 'bf16'} | "
                          + " | ".join(f"{tau(rows, nm(n, s), g):.2f}" for g in GAMMAS) + " |")
                if s:
                    rets = {g: boot(rows, lambda ix: tau(rows, nm(n, s), g, ix) / tau(rows, n, g, ix)) for g in (2, 4, 6)}
                    P["retention"][nm(n, s)] = rets
                    T2.append(f"| {task} | {ctx} | {n} | {s} | " + " | ".join(f3(rets[g]) for g in (2, 4, 6)) + " |")
        sn = BEST_SNAP[(task, ctx)]
        for v in VQ:
            if v not in bases:
                continue
            for g in (4, 6):
                cells = []
                for s in SRCS:
                    if s in ("k2", "g2"):
                        continue
                    if nm(v, s) not in rows[0] or nm(sn, s) not in rows[0]:
                        cells.append("-")
                        continue
                    dr = boot(rows, lambda ix: (tau(rows, nm(v, s), g, ix) / tau(rows, nm(sn, s), g, ix)) /
                              (tau(rows, v, g, ix) / tau(rows, sn, g, ix)))
                    P["double"][f"{v}|{sn}|{s}|g{g}"] = dr
                    cells.append(f3(dr))
                T3.append(f"| {task} | {ctx} | {v} | {sn} | {g} | " + " | ".join(cells) + " |")
    TA, out["analytic"] = analytic(res_rows)
    L += T0 + [""] + T1 + [""] + T2 + [""] + T3 + [""] + TA
    print("\n".join(L))
    if a.save:
        open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
        sys.path.insert(0, REPO)
        from tools.results import save_result
        D = "/data/lsh/ckpts/weightvq/llama31_8b"
        save_result("E7", "llama8b_magicvq_v13_lowbit_tau",
                    config={"setting": "E7 v13: drafts with weights below INT4 (C1 weight-VQ prefix k3 / k2B / k2, GPTQ "
                                       "INT3 / INT2 g128, dense dequant) + compressed KV, tau on v6 samples; bf16 / INT4 "
                                       "rows from v12", "run": a.run,
                            "sources": {"w4": f"{D}/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors",
                                        "k3": f"{D}/c1ft5_k3_dense/w_bf16.safetensors",
                                        "k2B": f"{D}/c1ft5_k2B_dense/w_bf16.safetensors",
                                        "k2": f"{D}/c1ft5_k2_dense/w_bf16.safetensors",
                                        "g3": f"{D}/gptq3_g128_pile128x2048_dense/w_fp16.safetensors",
                                        "g2": f"{D}/gptq2_g128_pile128x2048_dense/w_fp16.safetensors"},
                            "c1_ckpt": f"{D}/c1_had_uniform_pile_fp32_ft5",
                            "k2B_draft_cb": "/data/lsh/results_accept/llama31_c1_uniform_kd2/train_B_bf16/draft_codebooks.pt",
                            "best_snap": {f"{k[0]}_{k[1]}": v for k, v in BEST_SNAP.items()}, "bits": BITS,
                            "analytic_dW_int4_ms": DW, "scripts": ["experiments/26_magicvq/v13_prep.py",
                                                                   "experiments/26_magicvq/v13_prep.sh",
                                                                   "experiments/26_magicvq/v13_tau.py",
                                                                   "experiments/26_magicvq/v13_tau.sh",
                                                                   "experiments/26_magicvq/v13_summarize.py"]},
                    metrics=out, model_dir="/data/models/llama3.1_8b_instruct")


if __name__ == "__main__":
    main()
