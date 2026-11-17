"""E7 magicvq step 3: merged tau / quality tables, byte accounting, speedup computed from measured parts, plots.

Runs on any python with matplotlib (no torch), e.g. /hai/anaconda3/bin/python:
  python experiments/26_magicvq/summarize.py --tau_runs v1,v1b,v2,v3 --qual_runs v1,v2,v3 --timing_run v3 [--save]

Draft KV bytes per step (B = 1 view, in bf16-token units "tok-eq", fraction = tok-eq / ctx):
  eviction / SnapKV / TopK : budget (SnapKV: at decode start; generated tokens are appended on top)
  VQ / GSRQ                : (ctx - 16 - R) * bits / 16 + 16 sinks + R window + codebook bytes (all layers, stages read)
  KIVI                     : (ctx - 16 - R) * (b + 32 / group) / 16 + 16 + R (fp16 scale + zero per group)
speedup(draft, B, ctx) = max_gamma tau(gamma) * T_AR / (gamma * T_draft + T_verify(gamma + 1) + T_enc(gamma + 1))
  [computed from measured parts; round overheads (accept logic, rollback, window bookkeeping) not included]
"""
import argparse
import json
import math
import os
import sys

ROOT = "/data/lsh/results_accept/magicvq"
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUTDIR = os.path.join(REPO, "results", "E7")
SINKS = 16
# codebook bytes per stage, in bf16-token units (128 KiB / token for Llama-3.1-8B)
CB_TOK = {"a": 16, "p": 16, "c": 1, "e": 32, "q": 32, "g8": 288, "g16": 272}
FMT_BITS = {"a": 1.0, "p": 1.0, "c": 0.5, "e": 0.5, "q": 0.5, "g8": 1.0, "g16": 0.5}  # per stage
# timing record mapping
POSTMAP = {"aL2": "vq2_d8K256x2", "aL1": "vq1_d8K256", "eL1": "vq05_d16K256", "eL2": "vq1_d16K256x2",
           "cL1": "vq05_d8K16"}
GSMAP = {"g8L2": "gs2_D8C256x2", "g8L1": "gs1_D8C256", "g16L2": "gs1_D16C256x2", "g16L1": "gs05_D16C256"}
FAMILY_LABEL = {
    "evict": "StreamingLLM (16 sinks + window)", "snapkv": "SnapKV (MagicDec cfg)", "topk": "TopK oracle (untimed)",
    "kivi": "KIVI (g32)", "kivi+res": "KIVI (g32) + bf16 residual", "kivi128": "KIVI-like g128",
    "gsrq": "GSRQ (VQLLM port)", "gsrq+win": "GSRQ + bf16 window", "rq": "plain RQ ours (pre-RoPE, cn)",
    "rq+win": "plain RQ ours + bf16 window", "rqpost": "plain RQ ours post-RoPE", "rqpost+win": "plain RQ post + window",
    "exact": "exact (full bf16 KV)", "commvq": "CommVQ (official codebooks)", "commvq+win": "CommVQ + bf16 window",
    "gsrq-kl": "GSRQ attn-trained", "gsrq-kl+win": "GSRQ attn-trained + window", "gsrq-vn": "GSRQ V-gain",
    "gsrq-vn+win": "GSRQ V-gain + window", "gsrq-kv": "GSRQ K/V split (diag)", "rq-kv": "RQ K/V split (diag)"}


def parse(n):
    if n == "exact":
        return {"kind": "exact"}
    if n.startswith("evict"):
        return {"kind": "evict", "budget": int(n[5:])}
    if n.startswith("snapkv"):
        return {"kind": "snapkv", "budget": int(n[6:])}
    if n.startswith("topk"):
        return {"kind": "topk", "budget": int(n[4:])}
    base, _, r = n.partition("+r")
    R = int(r) if r else 0
    if n.startswith("kivi"):
        b, _, g = base[4:].partition("g")
        return {"kind": "kivi", "bits": int(b), "group": int(g) if g else 128, "R": R}
    import re
    m = re.fullmatch(r"vq([a-z0-9_]+?)(?:L(\d)|K(\d)V(\d))", base)
    lk, lv = (int(m.group(2)),) * 2 if m.group(2) else (int(m.group(3)), int(m.group(4)))
    return {"kind": "vq", "fmt": m.group(1), "level": max(lk, lv), "lk": lk, "lv": lv, "R": R}


def family(n):
    s = parse(n)
    k = s["kind"]
    if k in ("exact", "evict", "snapkv", "topk"):
        return k
    if k == "kivi":
        return ("kivi128" if s["group"] == 128 else "kivi+res" if s["R"] else "kivi")
    f = s["fmt"]
    if s["lk"] != s["lv"]:
        return "gsrq-kv" if f.startswith("g") else "rq-kv"  # K / V attribution or mixed bits (diagnostic)
    base = ("commvq" if f.startswith("cv") else "gsrq-kl" if f.startswith("gk") else "gsrq-vn" if f.startswith("gv")
            else "gsrq" if f.startswith("g") else "rqpost" if f in ("p", "q") else "rq")
    return base + ("+win" if s["R"] else "")


def series(n):
    """curve id: one knob swept per curve (budget / bits / window R)."""
    s, f = parse(n), family(n)
    if s["kind"] == "kivi":
        return f"{f} {s['bits']}b" if s["R"] else f
    if s["kind"] == "vq":
        if s["lk"] != s["lv"]:
            return f"{f} {s['fmt']} K{s['lk']}V{s['lv']}"
        return f"{f} {s['fmt']}L{s['level']}" if s["R"] else f"{f} {s['fmt']}"
    return f


CFG = {"bits": {}, "tok": {}}  # per-draft values recorded by tau_eval (used for formats without static entries)


def bits(n):
    s = parse(n)
    if s["kind"] == "vq":
        if s["fmt"] not in FMT_BITS or s["lk"] != s["lv"]:
            return CFG["bits"][n]
        return FMT_BITS[s["fmt"]] * s["level"]
    if s["kind"] == "kivi":
        return s["bits"] + 32 / s["group"]
    return 16.0


def tok_eq(n, ctx):
    s = parse(n)
    if s["kind"] == "exact":
        return ctx
    if s["kind"] in ("evict", "snapkv", "topk"):
        return min(s["budget"], ctx)
    R = s["R"]
    if s["kind"] == "vq" and (s["fmt"] not in CB_TOK or s["lk"] != s["lv"]):
        return CFG["tok"][(n, ctx)]
    cb = CB_TOK[s["fmt"]] * s["level"] if s["kind"] == "vq" else 0
    return (ctx - SINKS - R) * bits(n) / 16 + SINKS + R + cb


def kv_only(n, ctx):
    """draft KV bytes without codebook / decoder-table bytes (codebooks amortised over a large batch)."""
    s = parse(n)
    if s["kind"] != "vq":
        return tok_eq(n, ctx)
    R = s["R"]
    return (ctx - SINKS - R) * bits(n) / 16 + SINKS + R


def draft_time(n, rec):
    """-> (T_draft_us, T_enc(M) dict or None, note)."""
    s = parse(n)
    k = s["kind"]
    if k == "topk":
        return None, None, "oracle"
    if k == "exact":
        return rec["target"]["1"]["full"]["us"], None, "target"
    if k in ("evict", "snapkv"):
        e = rec["evict"].get(str(s["budget"]))
        return (e["full"]["us"] if e else None), None, "bf16 attention over budget"
    if k == "kivi":
        mode = f"int{s['bits']}"
        if s["R"]:
            h = rec["hyb"].get(f"{mode}+r{s['R']}")
            return (h["full"]["us"] if h else None), None, "kvq INT per-token-head kernel + bf16 window (proxy)"
        return rec["int"][mode]["full"]["us"], None, "kvq INT per-token-head kernel (proxy, no encode)"
    f, lv, R = s["fmt"], s["level"], s["R"]
    key = f"{f}L{lv}"
    if f.startswith("g"):
        tn = GSMAP.get(key)
        base = rec.get("gs", {}).get(tn)
        hyb = rec.get("hyb_gs", {})
        note = "GSRQ pre-RoPE Triton kernel (ours)"
    elif f in ("p", "q"):
        tn = POSTMAP.get(("a" if f == "p" else "e") + f"L{lv}")
        base = rec["vq"].get(tn)
        hyb = rec["hyb"]
        note = "kvq post-RoPE kernels (LUT/WMMA)"
    else:
        tn = POSTMAP.get(key)
        base = rec.get("vq_pre", {}).get(tn)
        hyb = rec.get("hyb_pre", {})
        note = "pre-RoPE standardized Triton kernel (ours)"
        if base is None:  # d8K16 (c): no pre-RoPE kernel -> post-RoPE kvq time (lower bound)
            base, hyb, note = rec["vq"].get(tn), rec["hyb"], "kvq post-RoPE kernel (lower bound; pre-RoPE not timed)"
    if base is None:
        return None, None, "no timing"
    enc = {k_[5:]: v["us"] for k_, v in base.items() if k_.startswith("enc_M")}
    if R:
        h = hyb.get(f"{tn}+r{R}")
        return (h["full"]["us"] if h else None), enc, note + " + bf16 window, LSE merge"
    return base["full"]["us"], enc, note


def pareto(points):
    """points [(x, y, label)] -> labels on the front (min x, max y)."""
    front, best = [], -1e9
    for x, y, lab in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            front.append(lab)
            best = y
    return set(front)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau_runs", default="v1,v1b,v2,v3")
    ap.add_argument("--qual_runs", default="v1,v2,v3")
    ap.add_argument("--timing_run", default="v3")
    ap.add_argument("--ctxs", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--set", default="pg19")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    gam = [1, 2, 3, 4, 6, 8]
    tau, qual, meta = {}, {}, {}
    for c in a.ctxs:
        for run in a.tau_runs.split(","):
            p = os.path.join(ROOT, "tau", f"tau_{run}_ctx{c}.json")
            if not os.path.exists(p):
                continue
            T = json.load(open(p))
            CFG["bits"].update(T["config"]["bits"])
            CFG["tok"].update({(n, c): v for n, v in T["config"]["kv_tokens_equiv"].items()})
            for sname, A in T["agg"].items():
                d = tau.setdefault(sname, {}).setdefault(c, {})
                meta.setdefault(sname, {}).setdefault(c, {})[run] = {k: A[k] for k in ("n_prompts", "mean_N",
                                                                                         "target_tf_vs_gen")}
                for n, v in A["drafts"].items():
                    if n == "exact" and "exact" in d:
                        d.setdefault("_exact_reruns", []).append(v[f"tau_g4"])
                        continue
                    d[n] = {"run": run, **v}
        for run in a.qual_runs.split(","):
            p = os.path.join(ROOT, "tau", f"qual_{run}_ctx{c}.json")
            if os.path.exists(p):
                Q = json.load(open(p))
                CFG["bits"].update(Q["config"]["bits"])
                CFG["tok"].update({(n, c): v for n, v in Q["config"]["kv_tokens_equiv"].items()})
                for n, v in Q["agg"].items():
                    qual.setdefault(c, {}).setdefault(n, {"run": run, **v})
    out = {"tau": {}, "quality": {}, "speedup": {}, "pareto": {}, "meta": meta}
    lines = []
    P = lines.append
    # ---- tables: tau + quality per ctx
    for sname in tau:
        for c in sorted(tau[sname]):
            rows = []
            for n, v in tau[sname][c].items():
                if n.startswith("_"):
                    continue
                q = qual.get(c, {}).get(n, {}) if sname == "pg19" else {}
                rows.append({"draft": n, "family": family(n), "bits": bits(n), "tok_eq": tok_eq(n, c),
                             "kv_frac": tok_eq(n, c) / c, "kv_frac_amort": kv_only(n, c) / c,
                             "alpha1": v["alpha_step"][0],
                             **{f"tau_g{g}": v[f"tau_g{g}"] for g in gam},
                             "tv": q.get("tv"), "kl": q.get("kl_target_draft"), "top1": q.get("top1_agree"),
                             "ppl": q.get("ppl")})
            rows.sort(key=lambda r: r["tok_eq"])
            out["tau"].setdefault(sname, {})[c] = rows
            if sname == "pg19":
                P(f"\n### tau + quality, PG-19, ctx {c} (target PPL on the true continuation: "
                  f"{qual.get(c, {}).get('target', {}).get('ppl', float('nan')):.3f})")
                P("| draft | family | bits | KV frac | alpha1 | tau g2 | tau g4 | tau g8 | TV | KL | top-1 | PPL |")
                P("|---|---|---|---|---|---|---|---|---|---|---|---|")
                f = lambda x, k=3: "-" if x is None else f"{x:.{k}f}"  # noqa: E731
                for r in rows:
                    P(f"| {r['draft']} | {r['family']} | {r['bits']:.2f} | {r['kv_frac']:.4f} | {r['alpha1']:.3f} | "
                      f"{r['tau_g2']:.2f} | {r['tau_g4']:.2f} | {r['tau_g8']:.2f} | {f(r['tv'])} | {f(r['kl'])} | "
                      f"{f(r['top1'])} | {f(r['ppl'], 2)} |")
                out["pareto"][c] = {f"tau_g{g}": sorted(pareto([(r["kv_frac"], r[f"tau_g{g}"], r["draft"])
                                                                 for r in rows if r["family"] not in ("topk", "exact")]))
                                    for g in (2, 4)}
    # ---- speedup
    tp = os.path.join(ROOT, "timing", f"timing_{a.timing_run}.json")
    TM = json.load(open(tp)) if os.path.exists(tp) else None
    if TM:
        out["fit"] = TM["fit"]
        out["timing_cotenants"] = {"start": TM.get("cotenants_start"), "end": TM.get("cotenants_end"),
                                   "per_record": [r.get("cotenants_after") for r in TM["records"]]}
        for rec in TM["records"]:
            B, c = rec["B"], rec["ctx"]
            if c not in out["tau"].get(a.set, {}):
                continue
            tgt = rec["target"]
            T_AR = tgt["1"]["full"]["us"]
            rows = []
            for r in out["tau"][a.set][c]:
                n = r["draft"]
                Td, enc, note = draft_time(n, rec)
                if Td is None:
                    continue
                per = {}
                for g in gam:
                    Tv = tgt[str(g + 1)]["full"]["us"]
                    Te = enc.get(str(g + 1), 0.0) if enc else 0.0
                    per[g] = r[f"tau_g{g}"] * T_AR / (g * Td + Tv + Te)
                bg = max(per, key=per.get)
                rows.append({"draft": n, "family": r["family"], "kv_frac": r["kv_frac"],
                             "kv_frac_amort": r["kv_frac_amort"], "T_draft_ms": Td / 1e3,
                             "c": Td / T_AR, "best_gamma": bg, "speedup": per[bg], "tau_at_best": r[f"tau_g{bg}"],
                             "tau_g4": r["tau_g4"], "speedup_per_gamma": per, "timing_note": note})
            rows.sort(key=lambda x: -x["speedup"])
            out["speedup"][f"B{B}_ctx{c}"] = {"B": B, "ctx": c, "T_AR_ms": T_AR / 1e3,
                                              "T_verify_ms": {g: tgt[str(g + 1)]["full"]["us"] / 1e3 for g in gam},
                                              "rows": rows}
            P(f"\n### speedup B={B} ctx={c}: T_AR {T_AR / 1e3:.2f} ms, T_verify(g4) {tgt['5']['full']['us'] / 1e3:.2f}"
              f" ms [computed from measured parts; tau = PG-19 B=1]")
            P("| draft | family | KV frac | T_draft ms | c | best g | tau | speedup |")
            P("|---|---|---|---|---|---|---|---|")
            for x in rows[:10]:
                P(f"| {x['draft']} | {x['family']} | {x['kv_frac']:.4f} | {x['T_draft_ms']:.2f} | {x['c']:.3f} | "
                  f"{x['best_gamma']} | {x['tau_at_best']:.2f} | {x['speedup']:.3f} |")
            fams = {}
            for x in rows:
                fams.setdefault(x["family"], x)
            P("best per family: " + ", ".join(f"{k}={v['draft']}:{v['speedup']:.3f}" for k, v in fams.items()))
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(OUTDIR, exist_ok=True)
    base = os.path.join(OUTDIR, f"llama8b_magicvq_{a.tag}")
    json.dump(out, open(base + "_table.json", "w"), indent=1)
    open(base + "_tables.md", "w").write(txt + "\n")
    if a.plots:
        out["plots"] = []
        for xk, sx in (("kv_frac", ""), ("kv_frac_amort", "_amort")):
            out["plots"] += plots(out, base, a.set, sx, xk=xk)
            if "chat" in out["tau"]:
                out["plots"] += plots(out, base, "chat", "_chat" + sx, (("tau_g2", "tau (gamma=2)"),
                                                                        ("tau_g4", "tau (gamma=4)")), speed=False, xk=xk)
    if a.save:
        sys.path.insert(0, REPO)
        from tools.results import save_result
        cfg = {"model": "Llama-3.1-8B-Instruct (bf16 W, bf16 KV target)",
               "setting": "MagicDec-style self-speculation, compressed-KV drafts (greedy)", "tau_runs": a.tau_runs,
               "qual_runs": a.qual_runs, "timing_run": a.timing_run, "ctxs": a.ctxs, "gammas": gam, "tau_set": a.set,
               "raw_dir": ROOT, "scripts": "experiments/26_magicvq"}
        print(save_result("E7", f"llama8b_magicvq_{a.tag}", cfg, out, "/data/models/llama3.1_8b_instruct",
                          notes="tau/quality measured (teacher-forced exact round semantics); speedup computed from "
                                "measured parts (synthetic-weight step timings x measured tau), not end-to-end"))


def figlegend(fig, axs):
    hs = {}
    for ax in axs.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    fig.legend(list(hs.values()), list(hs), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    fig.tight_layout()


XL = {"kv_frac": "draft KV bytes / full bf16 KV (B=1, incl. codebooks)",
      "kv_frac_amort": "draft KV bytes / full bf16 KV (codebooks amortised, large B)"}


def plots(out, base, sname, suffix="", ykeys=None, speed=True, xk="kv_frac"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = list(FAMILY_LABEL)
    cmap = plt.get_cmap("tab20")
    col = {f: cmap(i % 20) for i, f in enumerate(fams)}
    mks = "ovs^Dx*P<>hH8dp1234+X|_"
    mk = {f: mks[i % len(mks)] for i, f in enumerate(fams)}
    files = []
    ctxs = sorted(out["tau"].get(sname, {}))
    # (1) tau / quality vs bytes
    for ykey, ylab in (ykeys or (("tau_g2", "tau (gamma=2)"), ("tau_g4", "tau (gamma=4)"),
                                  ("tv", "TV(target, draft)"), ("top1", "top-1 agreement"))):
        fig, axs = plt.subplots(1, len(ctxs), figsize=(5 * len(ctxs), 4.2), squeeze=False)
        for ax, c in zip(axs[0], ctxs):
            rows = [r for r in out["tau"][sname][c] if r.get(ykey) is not None and r["family"] not in ("exact", "gsrq-kv")]
            for sr in sorted({series(r["draft"]) for r in rows}):
                rr = sorted([r for r in rows if series(r["draft"]) == sr], key=lambda r: r[xk])
                f = rr[0]["family"]
                ax.plot([r[xk] for r in rr], [r[ykey] for r in rr], marker=mk[f], color=col[f], lw=1,
                        ms=4, label=FAMILY_LABEL[f] + sr[len(f):])
            better_low = ykey == "tv"
            pts = [(r[xk], -r[ykey] if better_low else r[ykey], r["draft"]) for r in rows if r["family"] != "topk"]
            front = pareto(pts)
            fr = sorted([r for r in rows if r["draft"] in front], key=lambda r: r[xk])
            ax.plot([r[xk] for r in fr], [r[ykey] for r in fr], color="k", lw=2, alpha=0.3, label="Pareto front")
            ax.set_xscale("log")
            ax.set_title(f"ctx {c}")
            ax.set_xlabel(XL[xk])
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
        figlegend(fig, axs)
        fig.suptitle(f"Llama-3.1-8B-Instruct self-spec drafts, {sname}: {ylab} vs draft KV bytes "
                     f"(K/V-split diagnostics omitted; see table)", y=1.03)
        fn = f"{base}{suffix}_{ykey}_vs_bytes.png"
        fig.savefig(fn, dpi=130, bbox_inches="tight")
        plt.close(fig)
        files.append(fn)
    # (2) tau vs c and (3) speedup vs bytes, per (B, ctx)
    keys = list(out["speedup"])
    if keys and speed:
        for xkey, ykey, xl, yl, logx, tag in (("c", "tau_g4", "measured T_draft / T_target", "tau (gamma=4)", False,
                                               "tau_vs_c"),
                                              (xk, "speedup", XL[xk],
                                               "speedup vs AR [computed from measured parts]", True,
                                               "speedup_vs_bytes")):
            nc = min(4, len(keys))
            nr = math.ceil(len(keys) / nc)
            fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 4.2 * nr), squeeze=False)
            for i, k in enumerate(keys):
                ax = axs[i // nc][i % nc]
                rows = out["speedup"][k]["rows"]
                for sr in sorted({series(r["draft"]) for r in rows}):
                    rr = sorted([r for r in rows if series(r["draft"]) == sr], key=lambda r: r[xkey])
                    f = rr[0]["family"]
                    ax.plot([r[xkey] for r in rr], [r[ykey] for r in rr], marker=mk[f], color=col[f], lw=1, ms=4,
                            label=FAMILY_LABEL[f] + sr[len(f):])
                pts = [(r[xkey], r[ykey], r["draft"]) for r in rows if r["family"] != "exact"]
                front = pareto(pts)
                fr = sorted([r for r in rows if r["draft"] in front], key=lambda r: r[xkey])
                ax.plot([r[xkey] for r in fr], [r[ykey] for r in fr], color="k", lw=2, alpha=0.3, label="Pareto front")
                if ykey == "speedup":
                    ax.axhline(1.0, color="r", lw=0.8, ls="--")
                if logx:
                    ax.set_xscale("log")
                ax.set_title(k.replace("_", " "))
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.grid(alpha=0.3)
            for j in range(len(keys), nr * nc):
                axs[j // nc][j % nc].axis("off")
            figlegend(fig, axs)
            fn = f"{base}{suffix}_{tag}.png"
            fig.savefig(fn, dpi=120, bbox_inches="tight")
            plt.close(fig)
            files.append(fn)
    print("plots:", *files, sep="\n  ")
    return files


if __name__ == "__main__":
    main()
