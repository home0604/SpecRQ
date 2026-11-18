"""E8a phase A analysis: are per-sequence tau (and alpha1) of compressed-KV drafts predictable from prefill features,
and how much of the per-sequence draft-selection headroom does a simple predictor recover?

Inputs: tau records (tau_records.py) + features (features.py, /data/lsh/results_accept/accept_pred/feat_<run>_*.jsonl).
Step-time model (measured, all-active accounting, TP=4 RTX 4090, B=16): per-step draft time t_d(draft) and verify time
t_v(gamma) = v0 + v1 * gamma from results/E7/llama8b_magicvq_v6e2e_* (32k: syn_quote ctx32768 B16 gen256; 16k:
hotpotqa_q ctx16384 B16 gen256). SnapKV-4096 at 16k uses the 32k SnapKV-4096 draft step (budget-bound, ctx-independent).
Per-sequence speedup S_i(d) = tau_i(d, g_d) * T_AR / (g_d * t_d + t_v(g_d)); g_d fixed per (draft, ctx) = argmax of the
pooled mean S. Pool throughput (all-active, batch of mixed rows) = arithmetic mean of S_i.
Usage: /hai/anaconda3/bin/python3 experiments/29_accept_pred/analyze.py [--run v1] [--out_tag v1]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import tau_records  # noqa: E402

FEATDIR = "/data/lsh/results_accept/accept_pred"
E7 = os.path.join(REPO, "results", "E7")
TIMING = {16384: "llama8b_magicvq_v6e2e_hotpotqa_q_ctx16384_B16_gen256.json",
          32768: "llama8b_magicvq_v6e2e_syn_quote_ctx32768_B16_gen256.json"}
SETS = {"def": ["snapkv4096", "hyb512pL2", "vqpL2+r128"], "t2w": ["snapkv4096", "hyb512pL2@t2w", "vqpL2+r128@t2w"]}
GAMMAS = range(1, 7)


# ------------------------------------------------------------------ step-time model
def step_model():
    out = {}
    for ctx, fn in TIMING.items():
        m = json.load(open(os.path.join(E7, fn)))["metrics"]
        td, g, v = defaultdict(list), [], []
        for r in m["rows"]:
            pr = r["aa_per_round_ms"]
            td[r["config"].split(":")[0]].append(pr["draft_s"] / r["gamma"])
            g.append(r["gamma"]); v.append(pr["verify_s"])
        v1, v0 = np.polyfit(g, v, 1)
        oh = float(np.median([r["aa_round_ms"] - r["aa_per_round_ms"]["draft_s"] - r["aa_per_round_ms"]["verify_s"]
                              for r in m["rows"]]))
        out[ctx] = {"T_AR": m["ar_step_ms"], "t_d": {k: float(np.mean(x)) for k, x in td.items()}, "v0": float(v0),
                    "v1": float(v1), "overhead": oh, "source": fn}
    out[16384]["t_d"].setdefault("snapkv4096", out[32768]["t_d"]["snapkv4096"])
    return out


def t_round(sm, ctx, draft, g):
    s = sm[ctx]
    return g * s["t_d"][draft] + s["v0"] + s["v1"] * g + s["overhead"]


# ------------------------------------------------------------------ data
def agg_feats(L, prefix):
    """L: list over layers of {stat: [Hkv]} -> sequence-level aggregates."""
    out = {}
    for s in L[0]:
        a = np.array([l[s] for l in L], dtype=float)  # [nl, Hkv]
        if s == "n90":
            a = np.log(a)
            s = "logn90"
        out[f"{prefix}{s}_mean"] = float(a.mean())
        out[f"{prefix}{s}_q10"] = float(np.quantile(a, 0.1))
        out[f"{prefix}{s}_q90"] = float(np.quantile(a, 0.9))
        out[f"{prefix}{s}_min"] = float(a.min())
        out[f"{prefix}{s}_max"] = float(a.max())
    return out


def load_rows(run):
    recs = tau_records.load()
    rows, heads = [], []
    for fn in sorted(glob.glob(f"{FEATDIR}/feat_{run}_*.jsonl")):
        for line in open(fn):
            x = json.loads(line)
            if x.get("smoke"):
                continue
            key = (x["task"], x["ctx"], x["id"])
            e = recs[key]
            r = {"task": x["task"], "ctx": x["ctx"], "id": x["id"], "P": x["P"], "tf_agree": x["tf_agree"],
                 "P_match": x["P_match"], "first_tok_ok": x["first_tok_ok"], "N": e["N"]}
            r.update(agg_feats(x["obs"], "obs_"))
            r.update(agg_feats(x["orc"], "orc_"))
            for s in x["obs_ho"][0]:
                a = np.array([l[s] for l in x["obs_ho"]])
                r[f"obs_{s}_mean"] = float(a.mean()); r[f"obs_{s}_q10"] = float(np.quantile(a, 0.1))
            for p in ("obs_", "orc_"):  # contrast features: SnapKV-4096 error minus hybrid / RQ2 error (after o_proj)
                for c in ("def", "t2w"):
                    r[f"{p}d_snap4096_hyb_{c}"] = r[f"{p}o_snap4096_mean"] - r[f"{p}o_hyb_{c}_mean"]
                    r[f"{p}d_snap4096_vq_{c}"] = r[f"{p}o_snap4096_mean"] - r[f"{p}o_vq_{c}_mean"]
            r["drafts"] = e["drafts"]
            rows.append(r)
            heads.append({"task": x["task"], "ctx": x["ctx"], "id": x["id"],
                          "cov4096": np.array([l["cov4096"] for l in x["obs"]]),
                          "cov512": np.array([l["cov512"] for l in x["obs"]]),
                          "o_vq_def": np.array([l["o_vq_def"] for l in x["obs"]]),
                          "e_vq_def": np.array([l["e_vq_def"] for l in x["obs"]]),
                          "o_hyb_def": np.array([l["o_hyb_def"] for l in x["obs"]]),
                          "orc_cov4096": np.array([l["cov4096"] for l in x["orc"]]),
                          "orc_o_vq_def": np.array([l["o_vq_def"] for l in x["orc"]])})
    return rows, heads


# ------------------------------------------------------------------ correlations
def within(x, y, g):
    """pooled within-group correlation: Pearson of task-demeaned values; Spearman of task-demeaned within-task ranks;
    mean of per-task Spearman (n-weighted, tasks with n >= 6)."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(g)
    xd, yd, xr, yr = x.copy(), y.copy(), np.empty_like(x), np.empty_like(y)
    per = []
    for t in np.unique(g):
        m = g == t
        xd[m] -= x[m].mean(); yd[m] -= y[m].mean()
        xr[m] = (stats.rankdata(x[m]) - 0.5) / m.sum() - 0.5
        yr[m] = (stats.rankdata(y[m]) - 0.5) / m.sum() - 0.5
        if m.sum() >= 6 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            per.append((stats.spearmanr(x[m], y[m])[0], m.sum(), t))
    wp = float(np.sum([r * n for r, n, _ in per]) / max(1, np.sum([n for _, n, _ in per]))) if per else float("nan")
    rp = stats.pearsonr(xd, yd)[0] if np.std(xd) > 0 and np.std(yd) > 0 else float("nan")
    rs = stats.pearsonr(xr, yr)[0] if np.std(xr) > 0 and np.std(yr) > 0 else float("nan")
    return {"within_pearson": float(rp), "within_spearman_ranks": float(rs), "mean_task_spearman": wp,
            "per_task_spearman": {t: round(float(r), 3) for r, _, t in per}}


def corr_block(rows, feats, targets):
    out = {}
    tasks = [r["task"] for r in rows]
    for tn, tf in targets.items():
        y = np.array([tf(r) for r in rows], float)
        ok = np.isfinite(y)
        for fn in feats:
            x = np.array([r[fn] for r in rows], float)
            m = ok & np.isfinite(x)
            if m.sum() < 10 or np.std(x[m]) == 0:
                continue
            g = np.array(tasks)[m]
            d = {"n": int(m.sum()), "pooled_spearman": float(stats.spearmanr(x[m], y[m])[0]),
                 "pooled_pearson": float(stats.pearsonr(x[m], y[m])[0])}
            d.update(within(x[m], y[m], g))
            out[f"{tn}|{fn}"] = d
    return out


# ------------------------------------------------------------------ selection headroom
def speed_table(rows, sm, ctx, drafts):
    """S [n, nd] per-sequence speedup at the per-draft fixed gamma (pooled-mean optimum), gamma per draft."""
    gam, S = {}, np.zeros((len(rows), len(drafts)))
    for j, d in enumerate(drafts):
        best = None
        for g in GAMMAS:
            s = np.array([r["drafts"][d][f"tau_g{g}"] for r in rows]) * sm[ctx]["T_AR"] / t_round(sm, ctx, d, g)
            if best is None or s.mean() > best[0]:
                best = (s.mean(), g, s)
        gam[d], S[:, j] = best[1], best[2]
    return S, gam


def logistic_fit(X, y, w=None, l2=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0 / l2, max_iter=2000)
    clf.fit(X, y, sample_weight=w)
    return clf


def selection(rows, sm, ctx, drafts, feat_sets):
    """oracle / best-single / task-oracle / LOTO-CV predictors (threshold on one feature; logistic regression on 2-3
    features, cost-sensitive: sample weight = |S_vq - S_snap|) for the binary choice SnapKV-4096 vs the best non-SnapKV
    draft of the set (by pooled mean), and the 3-way oracle."""
    S, gam = speed_table(rows, sm, ctx, drafts)
    tasks = np.array([r["task"] for r in rows])
    mean = S.mean(0)
    jb = int(mean.argmax())
    res = {"n": len(rows), "gamma": gam, "mean_speedup": dict(zip(drafts, mean.round(4).tolist())),
           "best_single": drafts[jb], "best_single_S": float(mean[jb]), "oracle3_S": float(S.max(1).mean())}
    # task oracle (best draft per task, chosen on the same data)
    to = np.zeros(len(rows))
    tb = {}
    for t in np.unique(tasks):
        m = tasks == t
        j = int(S[m].mean(0).argmax()); tb[t] = drafts[j]; to[m] = S[m, j]
    res["task_oracle_S"] = float(to.mean()); res["task_best"] = tb
    # upper bound: per-sequence oracle over (draft, gamma)
    Sg = np.stack([np.array([r["drafts"][d][f"tau_g{g}"] for r in rows]) * sm[ctx]["T_AR"] / t_round(sm, ctx, d, g)
                   for d in drafts for g in GAMMAS], 1)
    res["oracle_draft_gamma_S"] = float(Sg.max(1).mean())
    res["oracle_draft_gamma_gain_vs_best_single"] = res["oracle_draft_gamma_S"] / float(mean[jb]) - 1
    # binary: SnapKV-4096 vs best other
    js = drafts.index("snapkv4096")
    jo = max((j for j in range(len(drafts)) if j != js), key=lambda j: mean[j])
    res["binary_pair"] = [drafts[js], drafts[jo]]
    S2 = S[:, [js, jo]]
    y = (S2[:, 1] > S2[:, 0]).astype(int)
    res["oracle2_S"] = float(S2.max(1).mean()); res["frac_vq_better"] = float(y.mean())
    base = float(max(S2[:, 0].mean(), S2[:, 1].mean()))
    res["best_single2_S"] = base
    preds = {}
    for name, fs in feat_sets.items():
        X = np.array([[r[f] for f in fs] for r in rows], float)
        pick = np.zeros(len(rows), int)
        for t in np.unique(tasks):  # leave-one-task-out
            tr, te = tasks != t, tasks == t
            if len(fs) == 1:  # threshold rule, direction and threshold fitted on the training tasks
                x = X[tr, 0]
                cands = np.unique(np.quantile(x, np.linspace(0, 1, 41)))
                best = (-1, None, None)
                for sgn in (1, -1):
                    for c in cands:
                        pk = (sgn * (X[tr, 0] - c) > 0).astype(int)
                        v = S2[tr][np.arange(tr.sum()), pk].mean()
                        if v > best[0]:
                            best = (v, sgn, c)
                pick[te] = (best[1] * (X[te, 0] - best[2]) > 0).astype(int)
            else:
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
                ytr = y[tr]
                if ytr.min() == ytr.max():
                    pick[te] = ytr[0]
                    continue
                clf = logistic_fit((X[tr] - mu) / sd, ytr, w=np.abs(S2[tr, 1] - S2[tr, 0]) + 1e-3)
                pick[te] = clf.predict((X[te] - mu) / sd)
        sp = S2[np.arange(len(rows)), pick]
        preds[name] = {"features": fs, "S": float(sp.mean()), "gain_vs_best_single": float(sp.mean() / base - 1),
                       "recovered_frac_of_oracle_gain": float((sp.mean() - base) / max(1e-12, res["oracle2_S"] - base)),
                       "acc": float((pick == y).mean())}
    res["predictors"] = preds
    res["oracle2_gain_vs_best_single"] = res["oracle2_S"] / base - 1
    res["oracle3_gain_vs_best_single"] = res["oracle3_S"] / res["best_single_S"] - 1
    res["task_oracle_gain_vs_best_single"] = res["task_oracle_S"] / res["best_single_S"] - 1
    res["per_task"] = {t: {"n": int((tasks == t).sum()), "mean": dict(zip(drafts, S[tasks == t].mean(0).round(3).tolist())),
                           "oracle3": round(float(S[tasks == t].max(1).mean()), 3),
                           "frac_vq_better": round(float(y[tasks == t].mean()), 3)} for t in np.unique(tasks)}
    return res


# ------------------------------------------------------------------ per-head heterogeneity
def head_hetero(heads):
    out = {}
    for ctx in sorted({h["ctx"] for h in heads}):
        H = [h for h in heads if h["ctx"] == ctx]
        d = {}
        for s in ("cov4096", "cov512", "e_vq_def", "o_vq_def", "o_hyb_def", "orc_cov4096", "orc_o_vq_def"):
            A = np.stack([h[s][1:].ravel() for h in H])  # skip layer 0 (dense attention everywhere)
            q = np.quantile(A, [0.1, 0.5, 0.9], axis=1)  # per sequence, across heads
            # head identity stability: Spearman of the head vectors between pairs of sequences (same task / any)
            rs_same, rs_any = [], []
            for i in range(len(H)):
                for j in range(i + 1, min(len(H), i + 25)):
                    r = stats.spearmanr(A[i], A[j])[0]
                    (rs_same if H[i]["task"] == H[j]["task"] else rs_any).append(r)
            d[s] = {"across_heads_q10_median": float(np.median(q[0])), "across_heads_q50_median": float(np.median(q[1])),
                    "across_heads_q90_median": float(np.median(q[2])),
                    "q90_over_q10_median": float(np.median(q[2] / np.maximum(q[0], 1e-9))),
                    "head_rank_stability_same_task": float(np.mean(rs_same)) if rs_same else None,
                    "head_rank_stability_diff_task": float(np.mean(rs_any)) if rs_any else None,
                    "frac_heads_below_0.9": float((A < 0.9).mean()) if "cov" in s else None,
                    "share_of_total_from_worst_10pct_heads": float(np.mean(
                        [np.sort(a)[::-1][: max(1, len(a) // 10)].sum() / a.sum() for a in A])) if "cov" not in s else None}
        out[ctx] = d
    return out


# ------------------------------------------------------------------ main
def tau_of(d, g=4, key="tau"):
    def f(r):
        x = r["drafts"].get(d)
        if x is None:
            return np.nan
        return x[f"tau_g{g}"] if key == "tau" else x["alpha1"]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--out_tag", default="v1")
    a = ap.parse_args()
    rows, heads = load_rows(a.run)
    sm = step_model()
    print("rows", len(rows), "tf_agree min", min(r["tf_agree"] for r in rows), "P_match all", all(r["P_match"] for r in rows))
    bad = [(r["task"], r["ctx"], r["id"], round(r["tf_agree"], 3)) for r in rows if r["tf_agree"] < 0.97 or not r["P_match"]]
    print("tf_agree < 0.97 or P mismatch:", bad)
    keep = [r for r in rows if r["tf_agree"] >= 0.97 and r["P_match"]]
    base_feats = ["obs_cov512_mean", "obs_cov2048_mean", "obs_cov4096_mean", "obs_cov4096_q10", "obs_cov4096_min",
                  "obs_far512_mean", "obs_far4096_mean", "obs_covho4096_mean", "obs_covho512_mean", "obs_farho4096_mean",
                  "obs_ent_mean", "obs_logn90_mean", "obs_mfar128_mean",
                  "obs_o_snap4096_mean", "obs_o_snap4096_q90", "obs_o_snap512_mean", "obs_e_snap4096_mean",
                  "obs_o_vq_def_mean", "obs_o_vq_def_q90", "obs_o_hyb_def_mean", "obs_e_vq_def_mean",
                  "obs_o_vq_t2w_mean", "obs_o_hyb_t2w_mean",
                  "obs_d_snap4096_hyb_def", "obs_d_snap4096_vq_def", "obs_d_snap4096_hyb_t2w", "P"]
    orc_feats = [f.replace("obs_", "orc_") for f in base_feats if f.startswith("obs_") and "ho" not in f]
    lt = lambda d1, d2, g=4: (lambda r: np.log(r["drafts"][d1][f"tau_g{g}"] / r["drafts"][d2][f"tau_g{g}"])  # noqa: E731
                              if d1 in r["drafts"] and d2 in r["drafts"] else np.nan)
    targets = {"tau4_snapkv4096": tau_of("snapkv4096"), "tau4_snapkv512": tau_of("snapkv512"),
               "tau4_hyb512pL2": tau_of("hyb512pL2"), "tau4_vqpL2+r128": tau_of("vqpL2+r128"),
               "alpha1_snapkv4096": tau_of("snapkv4096", key="a"), "alpha1_vqpL2+r128": tau_of("vqpL2+r128", key="a"),
               "alpha1_hyb512pL2": tau_of("hyb512pL2", key="a"),
               "logratio4_hyb_vs_snap4096": lt("hyb512pL2", "snapkv4096"),
               "logratio4_vq_vs_snap4096": lt("vqpL2+r128", "snapkv4096"),
               "tau4_hyb512pL2@t2w": tau_of("hyb512pL2@t2w"),
               "logratio4_hybt2w_vs_snap4096": lt("hyb512pL2@t2w", "snapkv4096")}
    res = {"n_rows": len(rows), "n_kept": len(keep), "excluded": bad, "step_model": sm, "corr": {}, "selection": {},
           "heads": head_hetero([h for h, r in zip(heads, rows) if r in keep])}
    for ctx in sorted({r["ctx"] for r in keep}):
        R = [r for r in keep if r["ctx"] == ctx]
        res["corr"][ctx] = {"prefill": corr_block(R, base_feats, targets), "oracle": corr_block(R, orc_feats, targets)}
        fsets = {"thr:obs_cov4096_mean": ["obs_cov4096_mean"], "thr:obs_far4096_mean": ["obs_far4096_mean"],
                 "thr:obs_covho4096_mean": ["obs_covho4096_mean"], "thr:obs_d_snap4096_hyb_def": ["obs_d_snap4096_hyb_def"],
                 "thr:obs_o_snap4096_mean": ["obs_o_snap4096_mean"], "thr:obs_logn90_mean": ["obs_logn90_mean"],
                 "lr:cov4096+o_hyb+ent": ["obs_cov4096_mean", "obs_o_hyb_def_mean", "obs_ent_mean"],
                 "lr:o_snap4096+o_hyb+far4096": ["obs_o_snap4096_mean", "obs_o_hyb_def_mean", "obs_far4096_mean"],
                 "ORACLE-FEAT thr:orc_d_snap4096_hyb_def": ["orc_d_snap4096_hyb_def"],
                 "ORACLE-FEAT lr:orc cov4096+o_hyb+o_snap": ["orc_cov4096_mean", "orc_o_hyb_def_mean", "orc_o_snap4096_mean"]}
        res.setdefault("obs_vs_orc", {})[ctx] = {
            f"{a}->{a.replace('obs_', 'orc_')}": {"pooled_spearman": float(stats.spearmanr([r[a] for r in R], [r[a.replace('obs_', 'orc_')] for r in R])[0]),
                                                 **within([r[a] for r in R], [r[a.replace('obs_', 'orc_')] for r in R], [r["task"] for r in R])}
            for a in ("obs_cov4096_mean", "obs_cov512_mean", "obs_o_snap4096_mean", "obs_o_vq_def_mean", "obs_o_hyb_def_mean",
                      "obs_d_snap4096_hyb_def")}
        res["selection"][f"{ctx}|def"] = selection(R, sm, ctx, SETS["def"], fsets)
        Rt = [r for r in R if "hyb512pL2@t2w" in r["drafts"]]
        if len({r["task"] for r in Rt}) >= 2:
            ft = {k.replace("hyb_def", "hyb_t2w"): [f.replace("hyb_def", "hyb_t2w") for f in v] for k, v in fsets.items()}
            res["selection"][f"{ctx}|t2w"] = selection(Rt, sm, ctx, SETS["t2w"], ft)
    os.makedirs(os.path.join(REPO, "results", "E8"), exist_ok=True)
    out = os.path.join(REPO, "results", "E8", f"accept_pred_{a.out_tag}_analysis.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    # flat per-sequence table for plotting / re-analysis
    flat = [{k: v for k, v in r.items() if k != "drafts"} | {f"tau4:{d}": r["drafts"][d]["tau_g4"] for d in r["drafts"]}
            | {f"alpha1:{d}": r["drafts"][d]["alpha1"] for d in r["drafts"]} for r in keep]
    json.dump(flat, open(os.path.join(REPO, "results", "E8", f"accept_pred_{a.out_tag}_per_seq.json"), "w"), default=float)
    print("wrote", out)
    return res


if __name__ == "__main__":
    main()
