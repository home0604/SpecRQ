"""E7 magicvq round 3: speedup in the KV-dominated regime from the multi-GPU step timings (mgpu_bench.py) x measured
tau (v1/v2 tables), per (B, ctx), per method family; [computed from measured parts].

speedup = max_gamma tau(gamma) * T_AR / (gamma * T_draft + T_verify(gamma+1) + T_enc(gamma+1))
  T_draft(KIVI) includes the amortized K group quantization (1/32 of a 32-token group per step).
  ctx 64k: tau taken from ctx 32k (marked tau_extrapolated; not used for the headline).
tau mapping (timing kernel -> tau draft):
  evict<W> -> evict<W> and snapkv<W> (same bf16-window kernel); rq1/rq2/rq05[+rR] -> post-RoPE plain RQ
  vqpL1/vqpL2/vqqL1[+rR]; kivi<b>+r<R> -> kivi<b>g32+r<R> (kivi4+r32 -> kivi4g32, r0 tau); gs1/gs2/gs05[+rR] ->
  attention-trained GSRQ with V gain vqgkv8L1/vqgkv8L2/vqgkv16L1[+rR]; cvq1 -> vqcv1L1+r128 (pp only).
Usage: /hai/anaconda3/bin/python experiments/26_magicvq/mgpu_summarize.py --tp v1 [--pp v1] --tag v3
"""
import argparse
import glob
import json
import os

ROOT = "/data/lsh/results_accept/magicvq/mgpu"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GAM = [1, 2, 3, 4, 6, 8]
RQMAP = {"rq1": "vqpL1", "rq2": "vqpL2", "rq05": "vqqL1"}
GSMAP = {"gs1": "vqgkv8L1", "gs2": "vqgkv8L2", "gs05": "vqgkv16L1"}


def candidates(rec):
    """-> list of (timing name, family, T_draft_us, enc dict {M: us} or None, [tau names])"""
    out = []
    for W, r in rec.get("evict", {}).items():
        out.append((f"evict{W}", "evict", r["full"]["us"], None, [f"evict{W}"]))
        out.append((f"snapkv{W}", "snapkv", r["full"]["us"], None, [f"snapkv{W}"]))
    for name, r in rec.get("rq", {}).items():
        enc = {k[5:]: v["us"] for k, v in r.items() if k.startswith("enc_M")}
        for w, x in r.items():
            if w == "base" or w.startswith("r"):
                sfx = "" if w == "base" else f"+{w}"
                out.append((f"{name}{sfx}", "rq_post" + ("+win" if sfx else ""), x["full"]["us"], enc,
                            [RQMAP[name] + sfx]))
    for bits, r in rec.get("kivi", {}).items():
        enc = {k[5:]: v["us"] for k, v in r.items() if k.startswith("enc_M")}
        kq = r["kgroup_quant_per_layer"]["us"] / 32
        for w, x in r.items():
            if w.startswith("r") and isinstance(x, dict) and "full" in x:
                R = int(w[1:])
                taus = [f"kivi{bits}g32+r{R}"] + ([f"kivi{bits}g32"] if R == 32 else [])
                out.append((f"kivi{bits}+{w}", "kivi", x["full"]["us"] + kq, enc, taus))
    for name, r in rec.get("gs", {}).items():
        enc = {k[5:]: v["us"] for k, v in r.items() if k.startswith("enc_M")}
        for w, x in r.items():
            if w == "base" or w.startswith("r"):
                sfx = "" if w == "base" else f"+{w}"
                out.append((f"{name}{sfx}", "gsrq" + ("+win" if sfx else ""), x["full"]["us"], enc,
                            [GSMAP[name] + sfx]))
    if "cvq1" in rec.get("cvq", {}):
        r = rec["cvq"]["cvq1"]
        enc = {"2": r["enc_M1"]["us"]}  # official encoder of 1 token (x gamma+1 tokens below)
        out.append(("cvq1", "commvq", r["base"]["full"]["us"], {"per_token": r["enc_M1"]["us"]}, ["vqcv1L1+r128"]))
    return out


def load_tau(tag):
    d = json.load(open(os.path.join(REPO, "results", "E7", f"llama8b_magicvq_{tag}_table.json")))
    tau = {}
    for sname, by in d["tau"].items():
        for c, rows in by.items():
            tau.setdefault(sname, {})[int(c)] = {r["draft"]: r for r in rows}
    return tau


def speedups(recs, tau, sname, deploy):
    res = {}
    for rec in recs:
        B, ctx = rec["B"], rec["ctx"]
        tctx = ctx if ctx in tau.get(sname, {}) else 32768
        if tctx not in tau.get(sname, {}):
            continue
        T = tau[sname][tctx]
        tgt = rec["target"]
        T_AR = tgt["1"]["full"]["us"]
        rows = []
        for name, fam, Td, enc, tnames in candidates(rec):
            tn = next((x for x in tnames if x in T), None)
            if tn is None:
                continue
            best = None
            for g in GAM:
                Tv = tgt[str(g + 1)]["full"]["us"]
                if enc is None:
                    Te = 0.0
                elif "per_token" in enc:
                    Te = enc["per_token"] * (g + 1)
                else:
                    Te = enc.get(str(g + 1), 0.0)
                su = T[tn][f"tau_g{g}"] * T_AR / (g * Td + Tv + Te)
                if best is None or su > best["speedup"]:
                    best = {"gamma": g, "speedup": su, "tau": T[tn][f"tau_g{g}"], "T_enc_us": Te}
            rows.append({"draft": name, "tau_draft": tn, "family": fam, "T_draft_ms": Td / 1e3, "c": Td / T_AR,
                         "kv_frac_B1": T[tn]["kv_frac"], "kv_frac_amort": T[tn].get("kv_frac_amort"), **best})
        rows.sort(key=lambda r: -r["speedup"])
        fam_best = {}
        for r in rows:
            fam_best.setdefault(r["family"], r)
        res[f"B{B}_ctx{ctx}"] = {"B": B, "ctx": ctx, "deploy": deploy, "T_AR_ms": T_AR / 1e3,
                                 "T_verify_ms": {g: tgt[str(g + 1)]["full"]["us"] / 1e3 for g in GAM},
                                 "tau_ctx": tctx, "tau_extrapolated": tctx != ctx, "rows": rows,
                                 "best_per_family": fam_best}
    return res


def pp_records(run):
    """sum the 4 per-GPU stage timings (only the last stage includes the lm_head) + 3 activation hops."""
    files, stages = [], None
    for rn in run.split(","):  # several runs: records appended per GPU
        fs = sorted(glob.glob(f"{ROOT}/bench_pp_{rn}_gpu*.json"))
        files += fs
        st = [json.load(open(f)) for f in fs]
        if stages is None:
            stages = st
        else:
            for s0, s1 in zip(stages, st):
                s0["records"] += s1["records"]
    if not stages:
        return [], {}
    hops = json.load(open(f"{ROOT}/hops_v1.json"))["hops"]

    def hop(R):
        ks = sorted(int(k) for k in hops)
        k = min((k for k in ks if k >= R), default=ks[-1])
        return hops[str(k)]["us"]
    recs = []
    for i, r0 in enumerate(stages[0]["records"]):
        rs = [s["records"][i] for s in stages]
        B = r0["B"]

        def tot(get, M=1):
            vals = [get(r) for r in rs]
            if any(v is None for v in vals):
                return None
            return sum(vals) + 3 * hop(B * M)
        rec = {"B": B, "ctx": r0["ctx"], "target": {}, "evict": {}, "rq": {}, "kivi": {}, "gs": {}, "cvq": {}}
        for M in r0["target"]:
            rec["target"][M] = {"full": {"us": tot(lambda r, M=M: r["target"][M]["full"]["us"], int(M))}}
        for W in r0.get("evict", {}):
            rec["evict"][W] = {"full": {"us": tot(lambda r, W=W: r["evict"][W]["full"]["us"])}}
        for fam in ("rq", "kivi", "gs", "cvq"):
            for n, x in r0.get(fam, {}).items():
                if not isinstance(x, dict):
                    continue
                y = {}
                for k, v in x.items():
                    if isinstance(v, dict) and "full" in v:
                        y[k] = {"full": {"us": tot(lambda r, n=n, k=k, fam=fam: r[fam][n][k]["full"]["us"])}}
                    elif k.startswith("enc_M") or k == "kgroup_quant_per_layer":
                        y[k] = {"us": sum(r[fam][n][k]["us"] for r in rs)}
                rec[fam][n] = y
        recs.append(rec)
    return recs, {"files": files, "cotenants": [(s.get("cotenants_start"), s.get("cotenants_end")) for s in stages]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", default="v1")
    ap.add_argument("--pp", default=None)
    ap.add_argument("--tau_tag", default="v2")
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    tau = load_tau(a.tau_tag)
    out = {"tau_tag": a.tau_tag}
    TP = None
    for run in a.tp.split(","):  # merge runs (later runs add grid points)
        p = f"{ROOT}/bench_tp_{run}.json"
        if not os.path.exists(p):
            continue
        X = json.load(open(p))
        if TP is None:
            TP = X
        else:
            have = {(r["B"], r["ctx"]) for r in TP["records"]}
            TP["records"] += [r for r in X["records"] if (r["B"], r["ctx"]) not in have]
            TP["fit"] = [f for f in TP["fit"] if (f["B"], f["ctx"]) not in {(g["B"], g["ctx"]) for g in X["fit"]}] + X["fit"]
            TP["cotenants_end"] = X.get("cotenants_end")
    TP["records"].sort(key=lambda r: (r["B"], r["ctx"]))
    out["fit_tp"] = TP["fit"]
    out["tp_cotenants"] = {"start": TP.get("cotenants_start"), "end": TP.get("cotenants_end"),
                           "per_record": [(r["B"], r["ctx"], r.get("co_before"), r.get("co_after")) for r in TP["records"]]}
    lines = []
    P = lines.append
    for sname in ("pg19", "chat"):
        out[f"tp_{sname}"] = speedups(TP["records"], tau, sname, "tp4")
    if a.pp:
        recs, meta = pp_records(a.pp)
        out["pp_meta"] = meta
        out["pp_records"] = recs
        for sname in ("pg19", "chat"):
            out[f"pp_{sname}"] = speedups(recs, tau, sname, "pp4")
        P("\n## TP=4 vs PP=4 step times (target, M=1), ms")
        P("| B | ctx | TP4 T_AR | PP4 T_AR | TP4 evict512 | PP4 evict512 |")
        P("|---|---|---|---|---|---|")
        pp = {(r["B"], r["ctx"]): r for r in recs}
        for r in TP["records"]:
            q = pp.get((r["B"], r["ctx"]))
            if q:
                e1 = r["evict"].get("512", {}).get("full", {}).get("us", float("nan")) / 1e3
                e2 = q["evict"].get("512", {}).get("full", {}).get("us", float("nan")) / 1e3
                P(f"| {r['B']} | {r['ctx']} | {r['target']['1']['full']['us'] / 1e3:.2f} | "
                  f"{q['target']['1']['full']['us'] / 1e3:.2f} | {e1:.2f} | {e2:.2f} |")
    for key in [k for k in out if k.startswith(("tp_pg19", "tp_chat", "pp_pg19", "pp_chat"))]:
        P(f"\n## best speedup per family, {key} [computed from measured parts]")
        fams = ["evict", "snapkv", "kivi", "rq_post", "rq_post+win", "gsrq", "gsrq+win", "commvq"]
        P("| B | ctx | T_AR ms | " + " | ".join(fams) + " | best |")
        P("|---|---|---|" + "---|" * (len(fams) + 1))
        for k, v in out[key].items():
            fb = v["best_per_family"]
            cells = [f"{fb[f]['speedup']:.2f} ({fb[f]['draft']}, g{fb[f]['gamma']})" if f in fb else "-" for f in fams]
            best = v["rows"][0]
            P(f"| {v['B']} | {v['ctx']}{'*' if v['tau_extrapolated'] else ''} | {v['T_AR_ms']:.1f} | "
              + " | ".join(cells) + f" | {best['draft']} {best['speedup']:.2f} |")
    e2e = []
    for f in sorted(glob.glob(f"{ROOT}/e2e_v1_*.json")):
        d = json.load(open(f))
        A = d["args"]
        key = f"B{A['B']}_ctx{A['ctx']}"
        comp = next((r for r in out.get("tp_pg19", {}).get(key, {}).get("rows", []) if r["tau_draft"] == A["tau_draft"]
                     and r["tau_draft"] == A["tau_draft"]), None)
        e2e.append({"B": A["B"], "ctx": A["ctx"], "draft": A["draft"], "tau_draft": A["tau_draft"], "gamma": A["gamma"],
                    "speedup_e2e": d["speedup_e2e_median"], "tau_replayed": d["tau_replayed"],
                    "speedup_computed_best": comp and comp["speedup"], "computed_gamma": comp and comp["gamma"],
                    "file": f})
    out["e2e"] = e2e
    P("\n## end-to-end TP=4 spec loop (synthetic weights, replayed acceptance) vs computed")
    P("| B | ctx | draft (tau) | gamma | e2e speedup | computed |")
    P("|---|---|---|---|---|---|")
    for x in e2e:
        P(f"| {x['B']} | {x['ctx']} | {x['draft']} ({x['tau_draft']}) | {x['gamma']} | {x['speedup_e2e']:.3f} | "
          f"{x['speedup_computed_best'] or float('nan'):.3f} |")
    txt = "\n".join(lines)
    print(txt)
    base = os.path.join(REPO, "results", "E7", f"llama8b_magicvq_{a.tag}_speed")
    json.dump(out, open(base + "_table.json", "w"), indent=1)
    open(base + "_tables.md", "w").write("(* = tau from ctx 32k, extrapolated)\n" + txt + "\n")
    if a.plots:
        out["plots"] = plots(out, base)
    if a.save:
        import sys
        sys.path.insert(0, REPO)
        from tools.results import save_result
        print(save_result("E7", f"llama8b_magicvq_{a.tag}_speed", {
            "model": "Llama-3.1-8B-Instruct bf16 (synthetic weights for timing)", "deploy": "TP=4 (and PP=4) on "
            "RTX 4090 GPUs 4-7, NCCL P2P disabled", "tp_run": a.tp, "pp_run": a.pp, "tau_tag": a.tau_tag},
            out, "/data/models/llama3.1_8b_instruct",
            notes="speedup computed from measured step parts x measured tau (64k: tau extrapolated from 32k)"))


def plots(out, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for key in [k for k in out if k.startswith(("tp_pg19", "tp_chat"))]:
        recs = out[key]
        ks = list(recs)
        if not ks:
            continue
        nc = 4
        nr = (len(ks) + nc - 1) // nc
        for xk, xl, sfx in (("kv_frac_B1", "draft KV bytes / full bf16 KV (B=1 incl. codebooks)", ""),
                            ("kv_frac_amort", "draft KV bytes / full bf16 KV (codebooks amortised)", "_amort"),
                            ("c", "measured T_draft / T_target", "_c")):
            fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4.2 * nr), squeeze=False)
            for i, k in enumerate(ks):
                ax = axs[i // nc][i % nc]
                rows = recs[k]["rows"]
                fams = sorted({r["family"] for r in rows})
                for j, f in enumerate(fams):
                    rr = sorted([r for r in rows if r["family"] == f and r.get(xk) is not None], key=lambda r: r[xk])
                    ax.plot([r[xk] for r in rr], [r["speedup"] for r in rr], marker="ovs^Dx*P<>"[j % 10], lw=1, ms=4,
                            label=f)
                pts = sorted([(r[xk], r["speedup"]) for r in rows if r.get(xk) is not None])
                fr, b = [], -1
                for x, y in pts:
                    if y > b:
                        fr.append((x, y)); b = y
                ax.plot([p[0] for p in fr], [p[1] for p in fr], color="k", lw=2, alpha=0.3, label="Pareto front")
                ax.axhline(1.0, color="r", lw=0.8, ls="--")
                if xk != "c":
                    ax.set_xscale("log")
                ax.set_title(f"{key} B{recs[k]['B']} ctx{recs[k]['ctx']}{' (tau extrap.)' if recs[k]['tau_extrapolated'] else ''}",
                             fontsize=9)
                ax.set_xlabel(xl, fontsize=8)
                ax.set_ylabel("speedup vs AR [computed from measured parts]", fontsize=8)
                ax.grid(alpha=0.3)
            for j in range(len(ks), nr * nc):
                axs[j // nc][j % nc].axis("off")
            h, l = axs[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
            fig.tight_layout()
            fn = f"{base}_{key}{sfx}.png"
            fig.savefig(fn, dpi=110, bbox_inches="tight")
            plt.close(fig)
            files.append(fn)
    print("plots:", *files, sep="\n  ")
    return files


if __name__ == "__main__":
    main()
