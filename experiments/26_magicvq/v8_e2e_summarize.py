"""E7 magicvq v8 (1-bit KV drafts), phase 2: measured e2e (e2e_real.py --accounting both, TP=4 B16) of 1-bit VQ
drafts vs the 2-bit t2w drafts, SnapKV and AR of the SAME process, plus a PROJECTION of the all-active speedup to
64k / 128k context from the measured per-round time components.

Measured (per point, per config): all-active speedup = all tokens / decode wall vs AR tok/s of the same process
(headline), static speedup, aa tau, per-round draft / verify / other ms (CUDA events of the graph segments).
Projection (labelled PROJECTED, not measured): for every config measured at both a 16k point and a 32k point (same
config string incl. gamma, B16), the per-round components are linear in context (bandwidth-bound attention; weights /
all_reduce constant):  T_draft_step(ctx), T_verify(ctx), T_other(ctx), and AR T_step(ctx) are fitted as a + b * ctx
through the two measured context means (mean prompt length + gen / 2, averaged over the points at that ctx). Then
  S_proj(ctx) = tau_aa(task, 32k) * T_AR(ctx) / (gamma * T_draft_step(ctx) + T_verify(ctx) + T_other(ctx))
with tau held at its measured 32k value (assumption: tau does not change with ctx; at 64k / 128k B16 does not fit on
4x 24 GB, so this is a model of a larger-memory system with the same bandwidth / kernels).
Usage: python experiments/26_magicvq/v8_e2e_summarize.py [--tag v8e2e] [--save]
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e2e_summarize_aa import OUT, RES, REPO, analyze  # noqa: E402

sys.path.insert(0, REPO)
LAB = {"snapkv2048": "SnapKV-2048", "snapkv4096": "SnapKV-4096", "vqpL2+r128@t2w": "RQ2+r128 t2w",
       "hyb512pL2@t2w": "hyb512+RQ2 t2w", "vqpL1+r128@t2w": "RQ1+r128 t2w-stage1 (nested)",
       "vqpL1+r512@t2w": "RQ1+r512 t2w-stage1 (nested)", "hyb512pL1@t2w": "hyb512+RQ1 t2w-stage1 (nested)",
       "vqpL1+r128@t1w": "RQ1+r128 t1w", "vqpL1+r512@t1w": "RQ1+r512 t1w", "hyb512pL1@t1w": "hyb512+RQ1 t1w"}
PROJ_CTX = (32768, 65536, 131072)
med = statistics.median


def kind(d):
    if d.startswith("snapkv") or d.startswith("stream"):
        return "snapkv"
    if "L2" in d and d.endswith("@t2w"):
        return "rq2"
    if "L1" in d:
        return "1bit"
    return "other"


def comps(r):
    pr = r["aa_per_round_ms"]
    return {"draft_step": pr["draft_s"] / r["gamma"], "verify": pr["verify_s"],
            "other": pr["accept_s"] + pr["encode_s"] + pr["host_s"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v8e2e")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{OUT}/run_{a.tag}*_ctx*_B*.json"))
    pts = {}
    for f in files:
        p = analyze(f, {})
        p["proc"] = os.path.basename(f).split("_")[1]
        key = (p["task"], p["ctx"], p["B"], p["gen"])
        pts.setdefault(key, []).append(p)
    L = [f"# E7 magicvq v8 phase 2: measured e2e of 1-bit VQ KV drafts (TP=4 RTX 4090 GPUs 4-7, bf16 Llama-3.1-8B-"
         "Instruct, B16, gen 256, all-active headline)", "",
         "- e2e_real.py --accounting both --soft_rep_check; codebooks via --cbx t2w / t1w; '@t2w' at L1 = stage 1 of the "
         "2-stage t2w codebook (nested prefix, LUT kernel KST=1). Speedups vs the AR of the same process; ratios across "
         "processes use each process's own AR.", ""]
    summ = ["## Summary per point (all-active measured, best gamma per draft kind; rows whose outputs differ across "
            "reps (--soft_rep_check) are excluded from 'best', their aa tau is not reproducible)", "",
            "| point | AR tok/s | best SnapKV | best RQ2 t2w | best 1-bit | 1-bit / RQ2 | 1-bit / SnapKV | RQ2 / SnapKV | "
            "higher but excluded (outputs differ across reps) | co-tenant |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    body, rec, allrows = [], {}, []
    for key in sorted(pts, key=lambda k: (k[0], -k[1])):
        ps = pts[key]
        rows = [dict(r, proc=p["proc"], ar_tok_s=p["ar_tok_s"], ar_step_ms=p["ar_step_ms"], meanP=p_meanP(p))
                for p in ps for r in p["rows"]]
        allrows += [dict(r, task=key[0], ctx=key[1]) for r in rows]
        best, flagged = {}, {}
        for r in rows:
            k = kind(r["draft"])
            tgt = flagged if r["aa_rep_ngen_differs"] else best  # rep-divergent rows: aa not reproducible, excluded
            if k not in tgt or r["aa_speedup"] > tgt[k]["aa_speedup"]:
                tgt[k] = r
        co = any(p["cotenant_any"] or p["ar_cotenant"] for p in ps)
        name = f"{key[0]} ctx {key[1]} B{key[2]}"

        def cell(k):
            r = best.get(k)
            return "-" if r is None else f"{r['aa_speedup']:.3f} ({LAB.get(r['draft'], r['draft'])} g{r['gamma']}, {r['proc']})"

        def rat(x, y):
            return "-" if x not in best or y not in best else f"**{best[x]['aa_speedup'] / best[y]['aa_speedup']:.3f}**"
        ars = " / ".join(f"{p['ar_tok_s']:.1f}" for p in ps)
        fl = "; ".join(f"{LAB.get(r['draft'], r['draft'])} g{r['gamma']} {r['aa_speedup']:.3f}" for r in flagged.values()
                       if r["aa_speedup"] > best.get(kind(r["draft"]), {"aa_speedup": 0})["aa_speedup"])
        summ.append(f"| {name} | {ars} | {cell('snapkv')} | {cell('rq2')} | "
                    f"{cell('1bit')} | {rat('1bit', 'rq2')} | {rat('1bit', 'snapkv')} | {rat('rq2', 'snapkv')} | "
                    f"{fl or '-'} | {'YES' if co else 'no'} |")
        body += [f"## {name}", "", "processes: " + "; ".join(
            f"{p['proc']}: AR {p['ar_tok_s']:.1f} tok/s (step {p['ar_step_ms']:.2f} ms), co-tenant "
            f"{'YES' if p['cotenant_any'] or p['ar_cotenant'] else 'no'} (start {p['cotenants_start'] or 'none'}, end "
            f"{p['cotenants_end'] or 'none'}), {os.path.basename(p['raw'])}" for p in ps), "",
            "| draft | proc | gamma | **all-active** (min-max) | static | tau aa / static | ms/round aa | draft step ms | "
            "verify ms | other ms | aa rows identical to AR | outputs differ across reps |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            c = comps(r)
            a0, a1 = r["aa_tok_s_min_max"]
            body.append(f"| {LAB.get(r['draft'], r['draft'])}{' *' if r['cotenant'] else ''} | {r['proc']} | {r['gamma']} | "
                        f"**{r['aa_speedup']:.3f}** ({a0 / r['ar_tok_s']:.3f}-{a1 / r['ar_tok_s']:.3f}) | "
                        f"{r['static_speedup']:.3f} | {r['aa_tau']:.2f} / {r['tau_static']:.2f} | {r['aa_round_ms']:.1f} | "
                        f"{c['draft_step']:.2f} | {c['verify']:.2f} | {c['other']:.2f} | "
                        f"{r['aa_n_rows_identical_overlap']}/{key[2]} | {'yes' if r['aa_rep_ngen_differs'] else 'no'} |")
        body.append("")
        rec[name] = {"procs": [{k: p[k] for k in ("proc", "ar_tok_s", "ar_step_ms", "ar_tok_s_runs", "ar_cotenant",
                                                  "cotenant_any", "cotenants_start", "cotenants_end", "raw")} for p in ps],
                     "rows": rows, "best": {k: {"draft": r["draft"], "gamma": r["gamma"], "aa_speedup": r["aa_speedup"]}
                                            for k, r in best.items()}}
    # ---------------------------------------------------------------- projection
    proj = projection(allrows)
    L += summ + [""] + proj["md"] + [""] + body
    md = "\n".join(L)
    print(md)
    if a.save:
        out = f"{RES}/llama8b_magicvq_v8_1bit_e2e_tables.md"
        open(out, "w").write(md + "\n")
        from tools.results import save_result
        for name, v in rec.items():
            t, c, b = name.split()[0], int(name.split()[2]), int(name.split()[3][1:])
            save_result("E7", f"llama8b_magicvq_v8_1bit_e2e_{t}_ctx{c}_B{b}_gen256",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), fully real "
                                "e2e self-spec (e2e_real.py --accounting both), 1-bit VQ KV drafts vs RQ2 t2w / SnapKV",
                                "B": b, "T": c, "gen": 256, "task": t, "configs": [r["config"] for r in v["rows"]]},
                        metrics=json.loads(json.dumps(v, default=str)), model_dir="/data/models/llama3.1_8b_instruct",
                        notes="all-active = headline; static also measured in the same run")
        save_result("E7", "llama8b_magicvq_v8_1bit_projection",
                    config={"what": "PROJECTED all-active speedup at 64k / 128k B16 from the measured 16k / 32k per-round "
                            "components (linear in ctx), tau held at the measured 32k value", "ctx": PROJ_CTX},
                    metrics=json.loads(json.dumps({k: v for k, v in proj.items() if k != "md"}, default=str)),
                    model_dir="/data/models/llama3.1_8b_instruct", notes="projection, not a measurement")
        print("saved", out)


def p_meanP(p):
    """mean context of the decode of one process: mean prompt length + gen / 2."""
    R = json.load(open(p["raw"]))
    return statistics.mean(R["prefill"]["P"]) + p["gen"] / 2


def fit(xs, ys):
    """two (or more) points -> least-squares a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def projection(rows):
    """draft step: linear fit in ctx per DRAFT (pooled over gamma / processes: the draft step does not depend on gamma);
    verify: linear fit per GAMMA (pooled over drafts: the target verify does not depend on the draft); other (accept /
    encode / host) per config at 32k (small). S_proj for every 32k row with its measured aa tau."""
    md = ["## PROJECTED all-active speedup at 64k / 128k (B16; per-round components linear in ctx, tau = measured 32k)",
          "", "Not a measurement (64k / 128k at B16 does not fit on 4x24 GB): draft step fitted per draft over its 16k and "
          "32k points (pooled over gamma), verify fitted per gamma over all drafts, AR step over all AR baselines; contexts "
          "= mean prompt + gen/2. tau held at its measured 32k value: optimistic for SnapKV (fixed budget covers a "
          "shrinking fraction of the context), neutral for VQ. c = T_draft_step / T_AR. Rows whose outputs differ across "
          "reps are skipped.", ""]

    def lin(pts):
        by = {}
        for x, y in pts:
            by.setdefault(round(x, -3), []).append((x, y))
        if len(by) < 2:
            return None
        xs = [statistics.mean(x for x, _ in v) for v in by.values()]
        ys = [statistics.mean(y for _, y in v) for v in by.values()]
        return fit(xs, ys)
    ok = [r for r in rows if not r["aa_rep_ngen_differs"]]
    far = lin([(r["meanP"], r["ar_step_ms"]) for r in rows])
    fd = {d: lin([(r["meanP"], comps(r)["draft_step"]) for r in ok if r["draft"] == d]) for d in {r["draft"] for r in ok}}
    fv = {g: lin([(r["meanP"], comps(r)["verify"]) for r in ok if r["gamma"] == g]) for g in {r["gamma"] for r in ok}}
    md += [f"AR step ms = {far[0]:.2f} + {far[1] * 1e3:.4f} * ctx/1000; draft step ms (a + b * ctx/1000): "
           + "; ".join(f"{LAB.get(d, d)} {v[0]:.2f} + {v[1] * 1e3:.4f}" for d, v in sorted(fd.items()) if v)
           + "; verify ms: " + "; ".join(f"g{g} {v[0]:.2f} + {v[1] * 1e3:.4f}" for g, v in sorted(fv.items()) if v), "",
           "| task | draft | gamma | tau aa @32k | c @32k / 64k / 128k | S meas 32k | S proj 32k / 64k / 128k | "
           "S proj / best SnapKV proj @32k / 64k / 128k | S proj / best RQ2 t2w proj |", "|---|---|---|---|---|---|---|---|---|"]
    T = lambda f_, x: f_[0] + f_[1] * x  # noqa: E731
    out = {}
    for r in ok:
        if r["ctx"] < 32768 or not fd.get(r["draft"]) or not fv.get(r["gamma"]):
            continue
        g, o = r["gamma"], comps(r)["other"]
        S = [r["aa_tau"] * T(far, x) / (g * T(fd[r["draft"]], x) + T(fv[g], x) + o) for x in PROJ_CTX]
        c = [T(fd[r["draft"]], x) / T(far, x) for x in PROJ_CTX]
        out[(r["task"], r["config"])] = {"draft": r["draft"], "gamma": g, "tau_aa_32k": r["aa_tau"], "c": dict(zip(PROJ_CTX, c)),
                                         "S_proj": dict(zip(PROJ_CTX, S)), "S_meas_32k": r["aa_speedup"]}
    for task in sorted({k[0] for k in out}):
        snap = [v for k, v in out.items() if k[0] == task and kind(v["draft"]) == "snapkv"]
        bs = {x: max(v["S_proj"][x] for v in snap) for x in PROJ_CTX} if snap else None
        rq2 = [v for k, v in out.items() if k[0] == task and kind(v["draft"]) == "rq2"]
        br = {x: max(v["S_proj"][x] for v in rq2) for x in PROJ_CTX} if rq2 else None
        for (t, cfg), v in sorted(out.items(), key=lambda kv: (kind(kv[1]["draft"]), kv[0][1])):
            if t != task:
                continue
            rel = " / ".join(f"{v['S_proj'][x] / bs[x]:.3f}" for x in PROJ_CTX) if bs else "-"
            v["vs_best_snapkv_proj"] = {x: v["S_proj"][x] / bs[x] for x in PROJ_CTX} if bs else None
            v["vs_best_rq2_proj"] = {x: v["S_proj"][x] / br[x] for x in PROJ_CTX} if br else None
            rel2 = " / ".join(f"{v['S_proj'][x] / br[x]:.3f}" for x in PROJ_CTX) if br else "-"
            md.append(f"| {t} | {LAB.get(v['draft'], v['draft'])} | {v['gamma']} | {v['tau_aa_32k']:.2f} | "
                      + " / ".join(f"{v['c'][x]:.3f}" for x in PROJ_CTX) + f" | {v['S_meas_32k']:.3f} | "
                      + " / ".join(f"{v['S_proj'][x]:.3f}" for x in PROJ_CTX) + f" | {rel} | {rel2} |")
    return {"md": md, "fits": {"ar": far, "draft_step": fd, "verify": fv},
            "proj": {f"{k[0]}|{k[1]}": v for k, v in out.items()}}


if __name__ == "__main__":
    main()
