"""E7 magicvq v4: tau on far-context workloads (lb_tau.py) x v3 measured TP=4 step times -> break-even analysis.

For every (task, ctx, timing point) and draft X with a TP=4 timing (v3 bench_tp_v1 + v1b, synthetic weights, GPUs 4-7):
  speedup_X = max_gamma tau_X(gamma) * T_AR / (gamma * T_draft_X + T_verify(gamma + 1) + T_enc_X(gamma + 1))
              [computed from measured parts, as mgpu_summarize.py]
  ratio_X(gamma)  = tau_X(gamma) / tau_S(gamma), S = the SnapKV budget with c_S <= c_X and the highest tau at gamma
  BE_X(gamma)     = (gamma * T_draft_X + T_verify + T_enc_X) / (gamma * T_draft_S + T_verify)  (ratio needed to tie S)
  X "passes" if speedup_X > the best SnapKV speedup over all budgets and gammas on that workload.
Hybrid hyb<W><fmt>L<k> (SnapKV-selected W tokens bf16 + all other tokens VQ) has no measured kernel. Its step time is
ESTIMATED [I] from v3 measured parts: T(rq+r512) + T(evictW) - T(evict512) (rq05 for qL1, rq1 for pL1, rq2 for pL2; the VQ
part over ctx - W instead of ctx - 512 is not credited; T_enc = 0 because generated tokens stay bf16 in the SnapKV
window, as in the tau simulation). CommVQ has no TP timing (v3 PP only) -> tau / ratio only.
timing point -> tau ctx: B16x16k and B8x16k use ctx 16k tau, B16x32k and B8x32k use ctx 32k tau.
Usage: /hai/anaconda3/bin/python experiments/26_magicvq/lb_summarize.py --run v4 [--plots] [--save]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mgpu_summarize import GAM, ROOT, candidates  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LBROOT = "/data/lsh/results_accept/magicvq/lb"
POINTS = [(16, 32768), (16, 16384), (8, 32768), (8, 16384)]
TASK_ORDER = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p", "hotpotqa_q", "syn_quote", "syn_kv", "syn_cont"]
HYB_RQ = {"qL1": "rq05", "pL1": "rq1", "pL2": "rq2"}


HYB_MEAS = {"rq2": "pL2", "rq1": "pL1", "rq05": "qL1"}  # v4b bench_tp hyb[<rq>][W] -> hyb<W><fmt>


def tp_records(runs=("v1", "v1b")):
    TP = None
    for run in runs:
        X = json.load(open(f"{ROOT}/bench_tp_{run}.json"))
        if TP is None:
            TP = X
        else:
            have = {(r["B"], r["ctx"]) for r in TP["records"]}
            TP["records"] += [r for r in X["records"] if (r["B"], r["ctx"]) not in have]
    return {(r["B"], r["ctx"]): r for r in TP["records"]}


def timing_table(rec):
    """tau draft name -> (timing name, family, T_draft_us, enc dict or None, estimated?)"""
    out = {}
    for name, fam, Td, enc, taus in candidates(rec):
        for t in taus:
            if t.startswith("kivi") and "+r128" not in t:
                continue
            out.setdefault(t, (name, fam, Td, enc, False))
    ev = rec["evict"]
    for fl, rq in HYB_RQ.items():
        if rq not in rec.get("rq", {}) or "r512" not in rec["rq"][rq] or "512" not in ev:
            continue
        base = rec["rq"][rq]["r512"]["full"]["us"]
        for W in (512, 1024, 2048, 4096):
            if str(W) in ev:
                Td = base + ev[str(W)]["full"]["us"] - ev["512"]["full"]["us"]
                out[f"hyb{W}{fl}"] = (f"est:{rq}+r512+bf16[{W}]", "hybrid", Td, None, True)
    for rq, by in rec.get("hyb", {}).items():  # v4b: measured hybrid kernel (no per-round encode)
        for W, x in by.items():
            out[f"hyb{W}{HYB_MEAS[rq]}"] = (f"hyb:{rq}[{x['vq_tokens']}]+bf16[{W}]", "hybrid", x["full"]["us"], None, False)
    return out


def enc_us(enc, g):
    if enc is None:
        return 0.0
    return enc.get(str(g + 1), 0.0)


def family(d):
    if d.startswith("snapkv"):
        return "SnapKV"
    if d.startswith("evict"):
        return "StreamingLLM"
    if d.startswith("hyb"):
        return ("hybrid SnapKV+CommVQ1" if "cv1" in d else "hybrid SnapKV+RQ0.5" if "qL1" in d else
                "hybrid SnapKV+RQ2" if "pL2" in d else "hybrid SnapKV+RQ1")
    if d.startswith("vqcv"):
        return "CommVQ1" + ("+win" if "+r" in d else "")
    if d.startswith("vqp"):
        return ("RQ1" if "L1" in d else "RQ2") + ("+win" if "+r" in d else "")
    if d.startswith("kivi"):
        return "KIVI+win"
    return d


def task_score(task, recs):
    """did the TARGET solve the synthetic task (uses the far context)? mean over samples, in [0, 1]."""
    import difflib
    sc = []
    for r in recs:
        ref, t = r.get("meta", {}).get("ref"), r.get("text", "")
        if ref is None:
            return None
        if task == "syn_quote":
            t = t.strip().strip('"')
            sc.append(difflib.SequenceMatcher(None, ref[:len(t) + 50], t).ratio())
        elif task == "syn_kv":
            sc.append(sum(x.split(": ")[1] in t for x in ref) / len(ref))
        elif task == "syn_cont":  # records reproduced verbatim among those that fit in the generated tokens
            lines = set(l.strip() for l in t.splitlines())
            n = max(1, sum(1 for l in t.splitlines() if " | " in l))
            sc.append(sum(x in lines for x in ref[:n]) / n)
    return sum(sc) / len(sc) if sc else None


def bootstrap(walks, rows, sname, T_AR, Tv, nb=2000, seed=0):
    """paired prompt bootstrap (90 % interval) of speedup_X / speedup_bestSnapKV at each draft's and SnapKV's own
    full-data best gamma, for drafts within 5 % of the best SnapKV (tau = sum tokens / sum rounds over the resample)."""
    import random
    rng = random.Random(seed)
    cand = [n for n, r in rows.items() if not n.startswith("snapkv") and r["speedup"] >= 0.95 * rows[sname]["speedup"]]
    if not cand:
        return {}
    gs = rows[sname]["gamma"]

    def su(n, g, idx):
        t = sum(walks[i][n][g][0] for i in idx)
        it = sum(walks[i][n][g][1] for i in idx)
        r = rows[n]
        return t / it * T_AR / (g * r["Td"] + Tv[g] + enc_us(r["enc"], g))
    out = {n: [] for n in cand}
    N = len(walks)
    for _ in range(nb):
        idx = [rng.randrange(N) for _ in range(N)]
        base = su(sname, gs, idx)
        for n in cand:
            out[n].append(su(n, rows[n]["gamma"], idx) / base)
    res = {}
    for n, v in out.items():
        v.sort()
        res[n] = [v[int(0.05 * nb)], v[int(0.95 * nb) - 1]]
    return res


def merged(runs, patch=(), tasks=None):
    """(task, ctx) -> {per_prompt, skipped, kv_tokens_equiv, files}: per_prompt of all `runs` (disjoint ids, first
    run wins), drafts of `patch` runs added to the records with the same id; only drafts present in every record."""
    M = {}
    for run in list(runs) + list(patch):
        for fn in sorted(glob.glob(f"{LBROOT}/tau_{run}_*_ctx*.json")):
            d = json.load(open(fn))
            key = (d["config"]["task"], d["config"]["ctx"])
            if tasks and key[0] not in tasks:
                continue
            if run in patch:
                if key not in M:
                    continue
                byid = {r["id"]: r for r in M[key]["per_prompt"]}
                for r in d["per_prompt"]:
                    if r["id"] in byid:  # patch only if the target trajectory is identical (same prompt + output);
                        # otherwise the patched draft drops out of this workload (hotpotqa_q 32k prompts are not
                        # reproducible across processes: lb_tasks builds them from a str set -> PYTHONHASHSEED order)
                        same = r["text"] == byid[r["id"]]["text"]
                        byid[r["id"]].setdefault("patched_text_same", {})[run] = same
                        if same:
                            byid[r["id"]]["drafts"].update(r["drafts"])
                M[key]["kv_tokens_equiv"].update(d["config"]["kv_tokens_equiv"])
                M[key]["files"].append(fn)
                continue
            W = M.setdefault(key, {"per_prompt": [], "skipped": [], "kv_tokens_equiv": {}, "files": [], "n_by_run": {}})
            have = {r["id"] for r in W["per_prompt"]}
            new = [r for r in d["per_prompt"] if r["id"] not in have]
            W["per_prompt"] += new
            W["skipped"] += d["skipped"]
            W["n_by_run"][run] = len(new)
            for n, v in d["config"]["kv_tokens_equiv"].items():
                W["kv_tokens_equiv"].setdefault(n, v)
            W["files"].append(fn)
    for W in M.values():
        common = set.intersection(*[set(r["drafts"]) for r in W["per_prompt"]]) if W["per_prompt"] else set()
        for r in W["per_prompt"]:
            r["drafts"] = {n: x for n, x in r["drafts"].items() if n in common}
    return M


def agg_of(per, skipped):
    names = list(per[0]["drafts"])
    G = len(per[0]["drafts"][names[0]]["match_rate_step"])
    agg = {"n_prompts": len(per), "n_skipped": len(skipped), "mean_P": sum(r["P"] for r in per) / len(per),
           "mean_N": sum(r["N"] for r in per) / len(per), "drafts": {}}
    totN = sum(r["N"] for r in per)
    for n in names:
        A = {"alpha_step": [sum(r["drafts"][n]["match_rate_step"][m] * r["N"] for r in per) / totN for m in range(G)]}
        for g in range(1, G + 1):
            A[f"tau_g{g}"] = sum(r["drafts"][n][f"g{g}"][0] for r in per) / sum(r["drafts"][n][f"g{g}"][1] for r in per)
        agg["drafts"][n] = A
    return agg


def analyse(run, runs=None, patch=(), tasks=None, tp_runs=("v1", "v1b")):
    TP = tp_records(tp_runs)
    taus = {}
    for (task, ctx), Mw in merged(runs or [run], patch, tasks).items():
        per = Mw["per_prompt"]
        if not per:
            continue
        A = agg_of(per, Mw["skipped"])
        taus[(task, ctx)] = {"agg": A, "score": task_score(task, per), "n_by_run": Mw["n_by_run"],
                             "walks": [{n: {g: x[f"g{g}"][:2] for g in GAM if f"g{g}" in x}
                                        for n, x in r["drafts"].items()} for r in per],
                             "kvfrac": {n: v / ctx for n, v in Mw["kv_tokens_equiv"].items() if n in A["drafts"]},
                             "files": Mw["files"],
                             "patched_text_same": [all(r.get("patched_text_same", {}).values()) for r in per]}
    out = {"run": run, "runs": runs or [run], "patch": list(patch), "tp_runs": list(tp_runs), "workloads": {}}
    for (task, ctx), T in sorted(taus.items(), key=lambda x: (x[0][1], TASK_ORDER.index(x[0][0]) if x[0][0] in TASK_ORDER else 99)):
        A = T["agg"]
        W = {"task": task, "ctx": ctx, "n": A["n_prompts"], "n_skipped": A["n_skipped"], "mean_P": A["mean_P"],
             "n_by_run": T["n_by_run"], "patched_trajectory_identical": sum(T["patched_text_same"]),
             "mean_N": A["mean_N"], "target_task_score": T["score"], "tau": {n: {f"g{g}": v[f"tau_g{g}"] for g in GAM if f"tau_g{g}" in v}
                                            for n, v in A["drafts"].items()},
             "alpha1": {n: v["alpha_step"][0] for n, v in A["drafts"].items()}, "kv_frac": T["kvfrac"], "points": {}}
        gams = [g for g in GAM if all(f"g{g}" in W["tau"][n] for n in W["tau"])]
        for B, pctx in POINTS:
            if pctx != ctx or (B, pctx) not in TP:
                continue
            rec = TP[(B, pctx)]
            tt = timing_table(rec)
            T_AR = rec["target"]["1"]["full"]["us"]
            Tv = {g: rec["target"][str(g + 1)]["full"]["us"] for g in gams}
            rows = {}
            for n in W["tau"]:
                if n not in tt:
                    continue
                tname, fam, Td, enc, est = tt[n]
                best = None
                for g in gams:
                    su = W["tau"][n][f"g{g}"] * T_AR / (g * Td + Tv[g] + enc_us(enc, g))
                    if best is None or su > best[0]:
                        best = (su, g)
                rows[n] = {"timing": tname, "c": Td / T_AR, "T_draft_ms": Td / 1e3, "estimated_c": est,
                           "speedup": best[0], "gamma": best[1], "enc": enc, "Td": Td}
            snaps = {n: r for n, r in rows.items() if n.startswith("snapkv")}
            best_s = max(snaps.items(), key=lambda kv: kv[1]["speedup"])
            for n, r in rows.items():
                r["ratio"], r["BE"], r["ref_snapkv"] = {}, {}, {}
                for g in gams:
                    cand = [s for s, rs in snaps.items() if rs["c"] <= r["c"] + 1e-9] or list(snaps)
                    s = max(cand, key=lambda s: W["tau"][s][f"g{g}"])
                    rs = snaps[s]
                    r["ratio"][g] = W["tau"][n][f"g{g}"] / W["tau"][s][f"g{g}"]
                    r["BE"][g] = (g * r["Td"] + Tv[g] + enc_us(r["enc"], g)) / (g * rs["Td"] + Tv[g])
                    r["ref_snapkv"][g] = s
                r["margin_best_gamma"] = r["ratio"][r["gamma"]] / r["BE"][r["gamma"]]
                r["passes"] = (not n.startswith("snapkv")) and r["speedup"] > best_s[1]["speedup"]
                r["vs_best_snapkv"] = r["speedup"] / best_s[1]["speedup"]
            boot = bootstrap(T["walks"], rows, best_s[0], T_AR, Tv)
            for n, ci in boot.items():
                rows[n]["vs_best_snapkv_ci90"] = ci
            for r in rows.values():
                del r["Td"], r["enc"]
            W["points"][f"B{B}_ctx{pctx}"] = {"B": B, "ctx": pctx, "T_AR_ms": T_AR / 1e3,
                                              "T_verify_ms": {g: Tv[g] / 1e3 for g in gams},
                                              "best_snapkv": best_s[0], "best_snapkv_speedup": best_s[1]["speedup"],
                                              "best_snapkv_gamma": best_s[1]["gamma"], "rows": rows}
        out["workloads"][f"{task}_ctx{ctx}"] = W
    # PG-19 / chat reference (v3 table)
    v3 = json.load(open(os.path.join(REPO, "results", "E7", "llama8b_magicvq_v3_table.json")))["tau"]
    out["v3_ref"] = {s: {c: {r["draft"]: {k: r[k] for k in r if k.startswith("tau_g")} for r in v3[s][str(c)]}
                         for c in (16384, 32768)} for s in ("pg19", "chat")}
    return out


def tables(out):
    L = []
    P = L.append
    ws = out["workloads"]
    P(f"# E7 magicvq {out.get('tag', 'v4')} - far-context workloads: tau and break-even vs SnapKV")
    P(f"tau runs {out['runs']} (disjoint samples merged) + patch runs {out['patch']} (extra drafts on the same samples); "
      f"TP=4 timing runs {out['tp_runs']} (synthetic weights, GPUs 4-7).")
    P("speedup = [computed from measured parts]: measured tau x measured TP=4 step times. hyb* c: measured where the "
      "timing run has a hybrid kernel record (hyb:<rq>[ctx-W]+bf16[W], no per-round encode: generated tokens stay in "
      "the bf16 part), else ESTIMATED [I]. Drafts with no step time in the timing runs have tau only. CommVQ: no TP "
      "timing.\n")
    P("## workloads")
    P("| task | ctx | n | n per run | skipped (<128 tok) | mean P | mean N | target task score (synthetic) |")
    P("|---|---|---|---|---|---|---|---|")
    for W in ws.values():
        sc = W["target_task_score"]
        P(f"| {W['task']} | {W['ctx']} | {W['n']} | {W.get('n_by_run', {})} | {W['n_skipped']} | {W['mean_P']:.0f} | "
          f"{W['mean_N']:.0f} | {'-' if sc is None else f'{sc:.2f}'} |")
    P("target task score: syn_quote = char similarity of the target output to the true paragraph; syn_kv = fraction of "
      "the 16 codes the target output contains; syn_cont = fraction of the generated ledger records that match the "
      "true records verbatim.")
    # headline: per workload x point, best SnapKV vs best non-SnapKV
    for pt in ("B16_ctx32768", "B8_ctx32768", "B16_ctx16384", "B8_ctx16384"):
        P(f"\n## {pt}: best SnapKV vs best other draft per family (speedup, gamma) [computed from measured parts; hyb c as in the header]")
        fams = ["StreamingLLM", "RQ1+win", "RQ2+win", "RQ1", "RQ2", "KIVI+win", "hybrid SnapKV+RQ0.5", "hybrid SnapKV+RQ1",
                "hybrid SnapKV+RQ2"]
        P("| task | best SnapKV | " + " | ".join(fams) + " | best non-SnapKV / SnapKV |")
        P("|---|---|" + "---|" * (len(fams) + 1))
        for W in ws.values():
            if pt not in W["points"]:
                continue
            X = W["points"][pt]
            fb = {}
            for n, r in sorted(X["rows"].items(), key=lambda kv: -kv[1]["speedup"]):
                fb.setdefault(family(n), (n, r))
            cells = [f"{fb[f][1]['speedup']:.3f} ({fb[f][0]}, g{fb[f][1]['gamma']})" if f in fb else "-" for f in fams]
            ns = [(n, r) for n, r in X["rows"].items() if not n.startswith("snapkv")]
            bn, br = max(ns, key=lambda kv: kv[1]["speedup"])
            P(f"| {W['task']} | {X['best_snapkv_speedup']:.3f} ({X['best_snapkv']}, g{X['best_snapkv_gamma']}) | "
              + " | ".join(cells) + f" | {br['vs_best_snapkv']:.3f} ({bn}) |")
    P("\n## drafts that beat the best SnapKV (computed speedup ratio > 1), with paired-bootstrap 90 % CI over prompts")
    P("c from the timing runs above; [I] = hybrid with c estimated (no kernel record). StreamingLLM rows with the same c "
      "and tau as SnapKV (ties) are omitted. CI: paired prompt bootstrap (2000 resamples), each draft at its own "
      "full-data best gamma.")
    P("| point | task | draft | c | speedup (g) | best SnapKV (g) | ratio | 90 % CI |")
    P("|---|---|---|---|---|---|---|---|")
    for pt in ("B16_ctx32768", "B8_ctx32768", "B16_ctx16384", "B8_ctx16384"):
        for W in ws.values():
            X = W["points"].get(pt)
            if not X:
                continue
            for n, r in sorted(X["rows"].items(), key=lambda kv: -kv[1]["vs_best_snapkv"]):
                if n.startswith("snapkv") or r["vs_best_snapkv"] <= 1.0 + 1e-9:
                    continue
                ci = r.get("vs_best_snapkv_ci90")
                P(f"| {pt} | {W['task']} | {n} | {r['c']:.3f}{' [I]' if r['estimated_c'] else ''} | "
                  f"{r['speedup']:.3f} (g{r['gamma']}) | {X['best_snapkv_speedup']:.3f} ({X['best_snapkv']}, "
                  f"g{X['best_snapkv_gamma']}) | {r['vs_best_snapkv']:.3f} | "
                  + (f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-") + " |")
    # ratio vs break-even at fixed gamma
    for pt in ("B16_ctx32768", "B16_ctx16384", "B8_ctx32768"):
        P(f"\n## {pt}: tau ratio vs SnapKV (highest tau among budgets with c <= c_X) and break-even ratio, gamma 3 / 4 / 6")
        P("cell = ratio / BE (pass if ratio >= BE); * = passes at that gamma")
        drafts = [d for d in ["vqpL1+r512", "vqpL2", "vqpL2+r128", "vqpL2+r512", "kivi4g32+r128", "hyb1024qL1",
                              "hyb2048qL1", "hyb1024pL1", "hyb2048pL1", "hyb512pL2", "hyb1024pL2", "hyb2048pL2"]
                  if any(d in W["points"].get(pt, {}).get("rows", {}) for W in ws.values())]
        P("| task | " + " | ".join(drafts) + " |")
        P("|---|" + "---|" * len(drafts))
        for W in ws.values():
            if pt not in W["points"]:
                continue
            X = W["points"][pt]["rows"]
            cells = []
            for d in drafts:
                if d not in X:
                    cells.append("-"); continue
                r = X[d]
                cells.append(" / ".join(f"{r['ratio'][g]:.2f}:{r['BE'][g]:.2f}{'*' if r['ratio'][g] >= r['BE'][g] else ''}"
                                        for g in (3, 4, 6) if g in r["ratio"]))
            P(f"| {W['task']} | " + " | ".join(cells) + " |")
    # full tau tables
    for W in ws.values():
        pt = f"B16_ctx{W['ctx']}"
        X = W["points"].get(pt, {}).get("rows", {})
        P(f"\n### tau, {W['task']} ctx {W['ctx']} (n={W['n']}, mean P {W['mean_P']:.0f}); c and speedup at {pt}")
        P("| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in sorted(W["tau"], key=lambda n: (X[n]["c"] if n in X else 9, W["kv_frac"].get(n, 1))):
            t = W["tau"][n]
            r = X.get(n)
            cs = (f"{r['c']:.3f}{' [I]' if r['estimated_c'] else ''} | {r['speedup']:.3f} (g{r['gamma']}) | "
                  f"{r['vs_best_snapkv']:.3f}") if r else "- | - | -"
            P(f"| {n} | {family(n)} | {W['kv_frac'].get(n, 0):.4f} | {W['alpha1'][n]:.3f} | "
              + " | ".join(f"{t.get(f'g{g}', float('nan')):.2f}" for g in (1, 2, 3, 4, 6)) + f" | {cs} |")
    if out.get("e2e"):
        P("\n## e2e loop check (e2e_loop.py, TP=4 GPUs 4-7, synthetic weights, acceptance replayed from the syn_quote 32k "
          "v4b match traces (24 new samples), fixed cache length; median of 3 reps)")
        P("| B | draft (tau draft) | gamma | spec tok/s | AR tok/s | e2e speedup | tau replayed |")
        P("|---|---|---|---|---|---|---|")
        import statistics as st
        for e in out["e2e"]:
            P(f"| {e['args']['B']} | {e['args']['draft']} ({e['args']['tau_draft']}) | {e['args']['gamma']} | "
              f"{st.median(e['spec_tok_per_s']):.0f} | {st.median(e['ar_tok_per_s']):.0f} | {e['speedup_e2e_median']:.3f} | "
              f"{e['tau_replayed']:.2f} |")
    # SnapKV drop vs PG-19
    if out.get("no_v3_ref"):
        return "\n".join(L)
    P("\n## SnapKV / StreamingLLM tau(g4) on each workload relative to PG-19 (and chat) at the same ctx (v3 tables)")
    names = ["snapkv512", "snapkv1024", "snapkv2048", "snapkv4096", "evict1024", "vqpL1+r512", "exact"]
    P("| task | ctx | " + " | ".join(names) + " |")
    P("|---|---|" + "---|" * len(names))
    for ref in ("pg19", "chat"):
        for c in (16384, 32768):
            R = out["v3_ref"][ref][c]
            P(f"| [{ref} v3 abs tau g4] | {c} | " + " | ".join(f"{R[n]['tau_g4']:.2f}" if n in R else "-" for n in names) + " |")
    for W in ws.values():
        R = out["v3_ref"]["pg19"][W["ctx"]]
        P(f"| {W['task']} | {W['ctx']} | " + " | ".join(
            f"{W['tau'][n]['g4']:.2f} ({W['tau'][n]['g4'] / R[n]['tau_g4']:.2f}x)" if n in W["tau"] and n in R else
            (f"{W['tau'][n]['g4']:.2f}" if n in W["tau"] else "-") for n in names) + " |")
    return "\n".join(L)


COL = {"SnapKV": "#1f77b4", "StreamingLLM": "#aec7e8", "RQ1": "#ff7f0e", "RQ1+win": "#ffbb78", "RQ2": "#2ca02c",
       "RQ2+win": "#98df8a", "KIVI+win": "#9467bd", "hybrid SnapKV+RQ0.5": "#d62728", "hybrid SnapKV+RQ1": "#e377c2",
       "hybrid SnapKV+RQ2": "#8c564b"}
MK = {"SnapKV": "o", "StreamingLLM": "s", "RQ1": "v", "RQ1+win": "v", "RQ2": "^", "RQ2+win": "^", "KIVI+win": "D",
      "hybrid SnapKV+RQ0.5": "*", "hybrid SnapKV+RQ1": "P", "hybrid SnapKV+RQ2": "X"}


def plots(out, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for pt in ("B16_ctx32768", "B16_ctx16384", "B8_ctx32768"):
        ws = [W for W in out["workloads"].values() if pt in W["points"]]
        if not ws:
            continue
        for ykey, ylab, sfx in (("tau", "tau at gamma = 4 (tokens / round, measured)", "tau_g4"),
                                ("speedup", "speedup vs AR (best gamma) [computed from measured parts]", "speedup")):
            nc = 3
            nr = (len(ws) + nc - 1) // nc
            fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.0 * nr), squeeze=False)
            for i, W in enumerate(ws):
                ax = axs[i // nc][i % nc]
                X = W["points"][pt]
                pts = []
                for fam in COL:
                    rr = sorted([(r["c"], (W["tau"][n]["g4"] if ykey == "tau" else r["speedup"]), r["estimated_c"], n)
                                 for n, r in X["rows"].items() if family(n) == fam])
                    if not rr:
                        continue
                    est = rr[0][2]
                    ax.plot([p[0] for p in rr], [p[1] for p in rr], marker=MK[fam], color=COL[fam], lw=1,
                            ms=8 if MK[fam] == "*" else 5, ls="--" if est else "-", mfc="none" if est else COL[fam],
                            label=fam + (" (c estimated [I])" if est else ""))
                    pts += rr
                pts.sort()
                fr, b = [], -1
                for x, y, est, _ in pts:
                    if y > b and not est:
                        fr.append((x, y)); b = y
                ax.step([p[0] for p in fr], [p[1] for p in fr], where="post", color="k", lw=2, alpha=0.25,
                        label="Pareto front (measured c only)")
                if ykey == "speedup":
                    ax.axhline(X["best_snapkv_speedup"], color="#1f77b4", lw=0.8, ls=":")
                    ax.axhline(1.0, color="r", lw=0.8, ls="--")
                ax.set_title(f"{W['task']}  (ctx {W['ctx'] // 1024}k, n={W['n']})", fontsize=10)
                ax.set_xlabel(f"c = T_draft / T_target, measured TP=4 {pt.replace('_', ' ')}", fontsize=8)
                ax.set_ylabel(ylab, fontsize=8)
                ax.grid(alpha=0.3)
            for j in range(len(ws), nr * nc):
                axs[j // nc][j % nc].axis("off")
            h, l = axs[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
            fig.suptitle(f"Llama-3.1-8B self-spec drafts on far-context workloads: {ylab.split(' (')[0]} vs draft cost "
                         f"({pt.replace('_', ' ')}; timing {'+'.join(out['tp_runs'])}); hollow/dashed = c estimated [I], "
                         f"solid = measured", fontsize=11)
            fig.tight_layout()
            fn = f"{base}_{sfx}_{pt}.png"
            fig.savefig(fn, dpi=120, bbox_inches="tight")
            plt.close(fig)
            files.append(fn)
    print("plots:", *files, sep="\n  ")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v4")
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--runs", nargs="+", default=None, help="tau runs merged as disjoint samples (default: --run)")
    ap.add_argument("--patch", nargs="*", default=[], help="tau runs adding drafts to the same sample ids")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--tp", nargs="+", default=["v1", "v1b"], help="bench_tp_<run>.json timing runs")
    ap.add_argument("--no_v3_ref", action="store_true")
    ap.add_argument("--e2e", default=None, help="tag of e2e_loop.py result files to list (mgpu/e2e_<tag>_*.json)")
    a = ap.parse_args()
    out = analyse(a.run, a.runs, a.patch, a.tasks, a.tp)
    out["tag"], out["no_v3_ref"] = a.tag, a.no_v3_ref
    if a.e2e:
        out["e2e"] = [json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/e2e_{a.e2e}_*.json"))]
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
        meas = any(r.get("hyb") for r in tp_records(a.tp).values())
        print(save_result("E7", f"llama8b_magicvq_{a.tag}", {
            "model": "Llama-3.1-8B-Instruct bf16, bf16 KV target; drafts = same weights + compressed KV",
            "tau_runs": out["runs"], "tau_patch_runs": out["patch"], "tasks": a.tasks,
            "workloads": "LongBench (THUDM/LongBench data.zip) + synthetic PG-19 haystack tasks",
            "timing": f"TP=4 bench_tp_{'+'.join(a.tp)} (synthetic weights, GPUs 4-7); hybrid c "
                      + ("measured (kvq_hybrid kernel, no per-round encode)" if meas else "estimated")},
            {k: v for k, v in out.items() if k != "v3_ref"}, "/data/models/llama3.1_8b_instruct",
            notes="speedup computed from measured step parts x measured tau; hybrid c "
                  + ("measured where a hyb record exists" if meas else "estimated [I]")))


if __name__ == "__main__":
    main()
