"""E7 magicvq v6: old (MSE k-means) vs draft-trained (rqd_train.py) 'p' RQ KV codebooks -> tau, computed speedup.

tau: lb_tau.py runs v6a (= the v4 sample ids, --only_run v4) + v6b (= v4b ids, --only_run v4b) merged per (task, ctx),
and tau_eval.py PG-19 runs (tau_v6_ctx<ctx>.json, books 0..35 of the >= 41984-token list). Old and trained drafts
are evaluated in the same process on the same target trajectories (paired).
speedup [computed from measured parts] = max_gamma tau(g) * T_AR / (g * T_draft + T_verify(g + 1) + T_enc(g + 1)),
the lb_summarize.py formula. A trained codebook has the SAME format / kernel as 'p', so it takes the step time of the
corresponding 'p' draft. Step times: TP=4 bench_tp_v4b (SnapKV/evict, rq2 base/+r128, hyb rq2: measured); rq1
(1-bit) entries are only in bench_tp_v1/v1b: their c = T_draft / T_AR of that run, rescaled by the v4b T_AR
('c from v1 run'); hyb512pL1 has no kernel record -> c ESTIMATED [I] (lb_summarize rule, v1 run).
CIs: paired prompt bootstrap (2000 resamples, 90 %), each draft at its own full-data best gamma.
Usage: python experiments/26_magicvq/rqd_summarize.py [--plots] [--save]
"""
import argparse
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lb_summarize import enc_us, timing_table, tp_records  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LB = "/data/lsh/results_accept/magicvq/lb"
TAU = "/data/lsh/results_accept/magicvq/tau"
GAM = [1, 2, 3, 4, 6]
POINTS = [(16, 32768), (8, 32768), (16, 16384)]
TASKS = ["syn_quote", "hotpotqa_q", "gov_report", "pg19"]
TRAINED = {"pt1": "pL1 trained (R0)", "pt1w": "pL1 trained (R128)", "pt2": "pL2 trained (R0)", "pt2w": "pL2 trained (R128)"}


def base_name(n):
    """trained draft name -> the 'p' draft with the same kernel / step time."""
    return re.sub(r"p(t1w|t2w|t1|t2)(L\d)", r"p\2", n)


def load_tau(runs=("v6a", "v6b"), pg_run="v6"):
    W = {}
    for run in runs:
        for fn in sorted(glob.glob(f"{LB}/tau_{run}_*_ctx*.json")):
            d = json.load(open(fn))
            key = (d["config"]["task"], d["config"]["ctx"])
            X = W.setdefault(key, {"per": [], "files": [], "n_by_run": {}})
            have = {r["id"] for r in X["per"]}
            new = [r for r in d["per_prompt"] if r["id"] not in have]
            X["per"] += new
            X["files"].append(fn)
            X["n_by_run"][run] = len(new)
            X["same_text"] = X.get("same_text", 0) + sum(
                1 for r in new if any(v for k, v in r["meta"].items() if k.startswith("same_text_as")))
    for fn in sorted(glob.glob(f"{TAU}/tau_{pg_run}_ctx*.json")):
        d = json.load(open(fn))
        per = [r for r in d["per_prompt"] if r["set"] == "pg19"]
        W[("pg19", d["config"]["ctx"])] = {"per": per, "files": [fn], "n_by_run": {pg_run: len(per)}}
    return W


def tau_of(per, n, g, idx=None):
    idx = range(len(per)) if idx is None else idx
    return sum(per[i]["drafts"][n][f"g{g}"][0] for i in idx) / sum(per[i]["drafts"][n][f"g{g}"][1] for i in idx)


def timings(tp_run="v6"):
    v4b = tp_records((tp_run,))
    v1 = tp_records(("v1", "v1b"))
    out = {}
    for pt in POINTS:
        rec = v4b[pt]
        tt = timing_table(rec)
        T_AR = rec["target"]["1"]["full"]["us"]
        tt1 = timing_table(v1[pt])
        T_AR1 = v1[pt]["target"]["1"]["full"]["us"]
        src = {}
        for n, x in tt1.items():
            if ("pL1" in n) and n not in tt:
                name, fam, Td, enc, est = x
                enc2 = None if enc is None else {k: v * T_AR / T_AR1 for k, v in enc.items()}
                tt[n] = (name + " [v1 run, rescaled]", fam, Td * T_AR / T_AR1, enc2, est)
                src[n] = "v1"
        out[pt] = {"tt": tt, "T_AR": T_AR, "Tv": {g: rec["target"][str(g + 1)]["full"]["us"] for g in GAM}, "src": src}
    return out


def speed(tauf, tm, n):
    x = tm["tt"].get(base_name(n))
    if x is None:
        return None
    _, _, Td, enc, est = x
    best = None
    for g in GAM:
        su = tauf(g) * tm["T_AR"] / (g * Td + tm["Tv"][g] + enc_us(enc, g))
        if best is None or su > best[0]:
            best = (su, g)
    return {"speedup": best[0], "gamma": best[1], "c": Td / tm["T_AR"], "estimated_c": est,
            "c_src": "v1 run" if base_name(n) in tm["src"] else "main run"}


def boot_ratio(per, f_num, f_den, nb=2000, seed=0):
    rng = random.Random(seed)
    N = len(per)
    v = []
    for _ in range(nb):
        idx = [rng.randrange(N) for _ in range(N)]
        v.append(f_num(idx) / f_den(idx))
    v.sort()
    return [v[int(0.05 * nb)], v[int(0.95 * nb) - 1]]


def analyse(tp_run="v6"):
    W = load_tau()
    TM = timings(tp_run)
    out = {"workloads": {}}
    for (task, ctx), X in sorted(W.items(), key=lambda kv: (kv[0][1], TASKS.index(kv[0][0]))):
        per = X["per"]
        if not per:
            continue
        names = sorted(set.intersection(*[set(r["drafts"]) for r in per]))
        A = {"task": task, "ctx": ctx, "n": len(per), "n_by_run": X["n_by_run"], "files": X["files"],
             "mean_P": sum(r["P"] for r in per) / len(per), "tau": {}, "alpha1": {}, "points": {}, "paired": {}}
        for n in names:
            A["tau"][n] = {f"g{g}": tau_of(per, n, g) for g in GAM}
            A["alpha1"][n] = sum(r["drafts"][n]["match_rate_step"][0] * r["N"] for r in per) / sum(r["N"] for r in per)
        # paired trained vs old tau at gamma 4 (and 6), CI over prompts
        for n in names:
            b = base_name(n)
            if b != n and b in names:
                A["paired"][n] = {"old": b}
                for g in (3, 4, 6):
                    A["paired"][n][f"g{g}"] = {
                        "ratio": tau_of(per, n, g) / tau_of(per, b, g),
                        "ci90": boot_ratio(per, lambda idx, n=n, g=g: tau_of(per, n, g, idx),
                                           lambda idx, b=b, g=g: tau_of(per, b, g, idx))}
                # trained 1-bit vs OLD 2-bit (same window variant)
                if "L1" in n:
                    b2 = b.replace("pL1", "pL2")
                    if b2 in names:
                        A["paired"][n]["vs_old_2bit"] = {"draft": b2, **{f"g{g}": tau_of(per, n, g) / tau_of(per, b2, g)
                                                                         for g in (3, 4, 6)}}
        for B, pctx in POINTS:
            if pctx != ctx:
                continue
            tm = TM[(B, pctx)]
            rows = {}
            for n in names:
                s = speed(lambda g, n=n: A["tau"][n][f"g{g}"], tm, n)
                if s:
                    rows[n] = s
            snaps = {n: r for n, r in rows.items() if n.startswith("snapkv")}
            bs = max(snaps, key=lambda n: snaps[n]["speedup"])
            for n, r in rows.items():
                r["vs_best_snapkv"] = r["speedup"] / rows[bs]["speedup"]
                if n.startswith("snapkv"):
                    continue
                x = tm["tt"][base_name(n)]

                def su(idx, n=n, g=r["gamma"], x=x):
                    return tau_of(per, n, g, idx) * tm["T_AR"] / (g * x[2] + tm["Tv"][g] + enc_us(x[3], g))
                sx = tm["tt"][base_name(bs)]

                def sb(idx, g=rows[bs]["gamma"], x=sx):
                    return tau_of(per, bs, g, idx) * tm["T_AR"] / (g * x[2] + tm["Tv"][g])
                r["vs_best_snapkv_ci90"] = boot_ratio(per, su, sb)
                b = base_name(n)
                if b != n and b in rows:
                    rb = rows[b]
                    xb = tm["tt"][b]

                    def sob(idx, g=rb["gamma"], x=xb, b=b):
                        return tau_of(per, b, g, idx) * tm["T_AR"] / (g * x[2] + tm["Tv"][g] + enc_us(x[3], g))
                    r["vs_old"] = r["speedup"] / rb["speedup"]
                    r["vs_old_ci90"] = boot_ratio(per, su, sob)
            A["points"][f"B{B}_ctx{pctx}"] = {"best_snapkv": bs, "best_snapkv_speedup": rows[bs]["speedup"],
                                              "best_snapkv_gamma": rows[bs]["gamma"], "rows": rows}
        out["workloads"][f"{task}_ctx{ctx}"] = A
    return out


def tables(out):
    L = []
    P = L.append
    P("# E7 magicvq v6 - draft-specific RQ KV codebooks (attention-output trained) vs MSE k-means 'p' codebooks")
    P("Same format / kernel as 'p' (d8 K256 x 2, post-RoPE K, per-head codebooks + per-channel mu/sd) -> same step "
      "time c. pt1 / pt1w: trained for 1 stage (L1) with R = 0 / 128 exact recent window in the loss; pt2 / pt2w: 2 "
      "stages (L2). hyb512*: SnapKV-512 bf16 + VQ rest. speedup = [computed from measured parts]: tau (this run) x "
      "TP=4 step times (bench_tp_v6 = the v4b command + rq1 / hyb rq1, synthetic weights, GPUs 4-7; '(v1)' = c taken from the v1 timing run rescaled by T_AR; [I] = estimated c). "
      "Decode-only metric. Old and trained drafts are paired (same process, same target trajectories).\n")
    P("## workloads")
    P("| task | ctx | n | n per run | mean P |")
    P("|---|---|---|---|---|")
    for A in out["workloads"].values():
        P(f"| {A['task']} | {A['ctx']} | {A['n']} | {A['n_by_run']} | {A['mean_P']:.0f} |")
    P("\n## tau: trained vs old codebook (same draft config), ratio at gamma 3 / 4 / 6 with paired-bootstrap 90 % CI")
    P("| task | ctx | trained draft | old draft | tau g4 old -> trained | ratio g3 | ratio g4 [CI] | ratio g6 [CI] | "
      "1-bit trained / old 2-bit (g3 / g4 / g6) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for A in out["workloads"].values():
        for n, p in A["paired"].items():
            v2 = p.get("vs_old_2bit")
            P(f"| {A['task']} | {A['ctx']} | {n} | {p['old']} | {A['tau'][p['old']]['g4']:.2f} -> {A['tau'][n]['g4']:.2f} | "
              f"{p['g3']['ratio']:.3f} | {p['g4']['ratio']:.3f} [{p['g4']['ci90'][0]:.3f}, {p['g4']['ci90'][1]:.3f}] | "
              f"{p['g6']['ratio']:.3f} [{p['g6']['ci90'][0]:.3f}, {p['g6']['ci90'][1]:.3f}] | "
              + (f"{v2['g3']:.3f} / {v2['g4']:.3f} / {v2['g6']:.3f} ({v2['draft']})" if v2 else "-") + " |")
    for pt in ("B16_ctx32768", "B8_ctx32768", "B16_ctx16384"):
        P(f"\n## {pt}: computed speedup (gamma) per draft; vs best SnapKV and vs the old codebook, 90 % CI")
        P("| task | best SnapKV | draft | c | speedup (g) | vs best SnapKV [CI] | vs old codebook [CI] |")
        P("|---|---|---|---|---|---|---|")
        for A in out["workloads"].values():
            X = A["points"].get(pt)
            if not X:
                continue
            for n, r in sorted(X["rows"].items(), key=lambda kv: -kv[1]["speedup"]):
                if n.startswith("snapkv") and n != X["best_snapkv"]:
                    continue
                ci = r.get("vs_best_snapkv_ci90")
                vo = (f"{r['vs_old']:.3f} [{r['vs_old_ci90'][0]:.3f}, {r['vs_old_ci90'][1]:.3f}]" if "vs_old" in r else "-")
                P(f"| {A['task']} | {X['best_snapkv']} g{X['best_snapkv_gamma']} {X['best_snapkv_speedup']:.3f} | {n} | "
                  f"{r['c']:.3f}{' [I]' if r['estimated_c'] else ''}{' (v1)' if r['c_src'] == 'v1 run' else ''} | "
                  f"{r['speedup']:.3f} (g{r['gamma']}) | {r['vs_best_snapkv']:.3f}"
                  + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f" | {vo} |")
    P("\n## tau per draft (g1 / g2 / g3 / g4 / g6) and alpha1")
    for A in out["workloads"].values():
        P(f"\n### {A['task']} ctx {A['ctx']} (n={A['n']})")
        P("| draft | alpha1 | g1 | g2 | g3 | g4 | g6 |")
        P("|---|---|---|---|---|---|---|")
        for n in sorted(A["tau"], key=lambda n: A["tau"][n]["g4"]):
            t = A["tau"][n]
            P(f"| {n} | {A['alpha1'][n]:.3f} | " + " | ".join(f"{t[f'g{g}']:.2f}" for g in GAM) + " |")
    return "\n".join(L)


def plots(out, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    pairs = [("vqpL1", "vqpt1L1"), ("vqpL1+r128", "vqpt1wL1+r128"), ("hyb512pL1", "hyb512pt1wL1"),
             ("vqpL2", "vqpt2L2"), ("vqpL2+r128", "vqpt2wL2+r128"), ("hyb512pL2", "hyb512pt2wL2")]
    labels = ["RQ1", "RQ1+r128", "SnapKV512+RQ1", "RQ2", "RQ2+r128", "SnapKV512+RQ2"]
    for pt in ("B16_ctx32768", "B16_ctx16384"):
        ws = [A for A in out["workloads"].values() if pt in A["points"]]
        if not ws:
            continue
        fig, axes = plt.subplots(1, len(ws), figsize=(4.2 * len(ws), 4.0), squeeze=False)
        for ax, A in zip(axes[0], ws):
            X = A["points"][pt]
            for i, ((o, t), lab) in enumerate(zip(pairs, labels)):
                for n, col, mk in ((o, "#9aa0a6", "o"), (t, "#1f6feb", "s")):
                    r = X["rows"].get(n)
                    if not r:
                        continue
                    ax.scatter(r["c"], r["speedup"], color=col, marker=mk, s=36, zorder=3,
                               facecolors="none" if r["estimated_c"] else col)
                if o in X["rows"] and t in X["rows"]:
                    ax.annotate("", xy=(X["rows"][t]["c"], X["rows"][t]["speedup"]),
                                xytext=(X["rows"][o]["c"], X["rows"][o]["speedup"]),
                                arrowprops=dict(arrowstyle="->", color="#1f6feb", lw=0.8))
                    ax.text(X["rows"][t]["c"], X["rows"][t]["speedup"], " " + lab, fontsize=7, va="bottom")
            sn = [(n, r) for n, r in X["rows"].items() if n.startswith("snapkv")]
            ax.scatter([r["c"] for _, r in sn], [r["speedup"] for _, r in sn], color="#d1242f", marker="^", s=36,
                       label="SnapKV")
            ax.axhline(X["best_snapkv_speedup"], color="#d1242f", lw=0.7, ls="--")
            ax.axhline(1.0, color="k", lw=0.5)
            ax.set_title(f"{A['task']} ctx {A['ctx']}", fontsize=9)
            ax.set_xlabel("draft step cost c = T_draft / T_AR (measured, TP=4)")
            ax.set_ylabel("speedup vs AR (computed: tau x measured step times)")
        axes[0][0].scatter([], [], color="#9aa0a6", label="old (k-means) codebook")
        axes[0][0].scatter([], [], color="#1f6feb", marker="s", label="draft-trained codebook")
        axes[0][0].scatter([], [], color="#1f6feb", marker="s", facecolors="none", label="c estimated [I]")
        axes[0][0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"E7 v6: draft-trained RQ KV codebooks, {pt.replace('_', ' ')} "
                     "(computed speedup, decode only; 1-bit c from the v1 timing run)", fontsize=9)
        fig.tight_layout()
        fn = f"{base}_speedup_vs_c_{pt}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        files.append(fn)
    # tau g4 bars old vs trained
    ws = list(out["workloads"].values())
    fig, axes = plt.subplots(1, len(ws), figsize=(3.6 * len(ws), 3.6), squeeze=False)
    for ax, A in zip(axes[0], ws):
        xs = range(len(pairs))
        old = [A["tau"].get(o, {}).get("g4", float("nan")) for o, _ in pairs]
        new = [A["tau"].get(t, {}).get("g4", float("nan")) for _, t in pairs]
        ax.bar([x - 0.2 for x in xs], old, 0.4, color="#9aa0a6", label="old")
        ax.bar([x + 0.2 for x in xs], new, 0.4, color="#1f6feb", label="trained")
        bs = max((n for n in A["tau"] if n.startswith("snapkv")), key=lambda n: A["tau"][n]["g4"])
        ax.axhline(A["tau"][bs]["g4"], color="#d1242f", ls="--", lw=0.8, label=bs)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{A['task']} {A['ctx'] // 1024}k (n={A['n']})", fontsize=9)
        ax.set_ylabel("tau at gamma 4 (measured)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fn = f"{base}_tau_g4_old_vs_trained.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    files.append(fn)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--tp", default="v6", help="bench_tp_<tp>.json timing run (v6: same command as v4b + rq1 / hyb rq1)")
    a = ap.parse_args()
    out = analyse(a.tp)
    out["tp_run"] = a.tp
    txt = tables(out)
    print(txt)
    base = os.path.join(REPO, "results", "E7", f"llama8b_magicvq_{a.tag}")
    open(base + "_tables.md", "w").write(txt + "\n")
    json.dump(out, open(base + "_table.json", "w"), indent=1)
    if a.plots:
        out["plots"] = plots(out, base)
    if a.save:
        sys.path.insert(0, REPO)
        from tools.results import save_result
        tr = {t: json.load(open(f"/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_{t}.json"))
              for t in ("t1", "t1w", "t2", "t2w")}
        print(save_result("E7", f"llama8b_magicvq_{a.tag}", {
            "model": "Llama-3.1-8B-Instruct bf16 target; drafts = same weights + compressed KV",
            "codebooks": {t: {"args": x["args"], "train_s": x["train_s"], "heldout_rel_err_before": x["mean_before"],
                              "heldout_rel_err_after": x["mean_after"]} for t, x in tr.items()},
            "data": "rqd_collect.py: 11 PG-19 test books with < 40k tokens (never used by eval loaders), 4 pile-val "
                    "heldout-end windows, 4 LongBench 2wikimqa (task not in eval) with target answers; held-out "
                    "1 per source",
            "tau_runs": ["v6a (v4 ids)", "v6b (v4b ids)", "tau_eval v6 PG-19 books 0..35"],
            "timing": "TP=4 bench_tp_v4b (+ v1/v1b for 1-bit, rescaled)"},
            out, "/data/models/llama3.1_8b_instruct",
            notes="speedup computed from measured tau x measured step times (decode only); trained codebooks keep the "
                  "'p' format so c is unchanged"))


if __name__ == "__main__":
    main()
