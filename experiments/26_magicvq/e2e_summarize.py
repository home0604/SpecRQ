"""E7 magicvq v5: summarize the fully real e2e runs (e2e_real.py) -> results/E7/llama8b_magicvq_v5_* .

Per (point, config): decode-only tokens/s and speedup vs the AR baseline of the same run; prefill + compression +
decode tokens/s; time breakdown; measured tau (accepted + 1 per active round; all rounds and rounds that start before
the first EOS of the AR output, the region v4b tau covers); token match vs the AR output and first divergence;
v4b tau of the same samples (teacher-forced walk, tau_eval) and the v4b COMPUTED speedup for the same gamma
(v4b tau on all 36 samples x v4b synthetic-weight TP=4 step times, bench_tp_v4b.json).
Usage: python experiments/26_magicvq/e2e_summarize.py --tag v5 [--save]
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
OUT = "/data/lsh/results_accept/magicvq/e2e_real"
RES = os.path.join(REPO, "results", "E7")
EOS = {128001, 128008, 128009}
LABEL = {"snapkv2048": "SnapKV-2048", "snapkv4096": "SnapKV-4096", "vqpL2+r128": "RQ2+r128",
         "hyb512pL2": "SnapKV512+RQ2 hybrid"}


def med(xs):
    return statistics.median(xs)


def v4b_computed(task, ctx, B, draft, g, table, bench):
    """v4b method: tau (36 samples) * T_AR / (g T_draft + T_verify(g+1) + T_enc) from the synthetic-weight bench."""
    wl = table["workloads"].get(f"{task}_ctx{ctx}")
    rec = [r for r in bench["records"] if r["B"] == B and r["ctx"] == ctx]
    if not wl or not rec or draft not in wl["tau"] or f"g{g}" not in wl["tau"][draft]:
        return None
    r = rec[0]
    tau = wl["tau"][draft][f"g{g}"]
    tar = r["target"]["1"]["full"]["us"]
    tv = r["target"][str(g + 1)]["full"]["us"]
    enc = 0.0
    if draft.startswith("snapkv"):
        td = r["evict"][draft[6:]]["full"]["us"]
    elif draft.startswith("hyb"):
        td = r["hyb"]["rq2"][draft[3:].split("p")[0]]["full"]["us"]
    else:
        td = r["rq"]["rq2"]["r128"]["full"]["us"]
        enc = r["rq"]["rq2"][f"enc_M{g + 1}"]["us"]
    return {"tau36": tau, "speedup": tau * tar / (g * td + tv + enc), "T_AR_ms": tar / 1e3, "T_draft_ms": td / 1e3,
            "T_verify_ms": tv / 1e3, "T_enc_ms": enc / 1e3}


def cof(r):
    """co-tenant flag of one config record (GPUs 4-7, other processes) before / after its timing."""
    b, a = r.get("co_before") or {}, r.get("co_after") or {}
    return {"cotenant": bool(b.get("cotenant") or a.get("cotenant")),
            "cotenant_apps": (b.get("others") or []) + (a.get("others") or []),
            "gpus_before": (b.get("nvsmi") or {}).get("gpus"), "gpus_after": (a.get("nvsmi") or {}).get("gpus")}


def analyze(path, table, bench):
    R = json.load(open(path))
    a = R["args"]
    task, ctx, B, gen = a["task"], a["ctx"], a["B"], a["gen"]
    prompts = json.load(open(f"{OUT}/prompts_{task}_ctx{ctx}.json"))[a["skip"]:a["skip"] + B]
    ar = R["configs"]["ar"]
    ar_out = ar["out"]
    ar_tps = med([x["tok_per_s"] for x in ar["runs"]])
    pre = R["prefill"]["total_s"]
    ar_dec = med([x["wall_s"] for x in ar["runs"]])
    eos_at = []
    for o in ar_out:
        e = [i for i, t in enumerate(o) if t in EOS]
        eos_at.append(e[0] if e else gen)
    rows = [{"config": "ar", "draft": "AR", "gamma": None, "decode_tok_s": ar_tps,
             "decode_tok_s_min_max": [min(x["tok_per_s"] for x in ar["runs"]), max(x["tok_per_s"] for x in ar["runs"])],
             "speedup_decode": 1.0, "e2e_tok_s": B * gen / (pre + ar_dec), "speedup_e2e": 1.0,
             "prefill_s": pre, "compress_s": 0.0, "decode_s": ar_dec, "step_ms": ar["step_ms"], **cof(ar)}]
    for c, r in R["configs"].items():
        if c == "ar":
            continue
        name, g = c.split(":")
        g = int(g)
        runs = r["runs"]
        dec = med([x["wall_s"] for x in runs])
        tps = med([x["tok_per_s"] for x in runs])
        comp = r["compress_s"]["median"]
        mr = sorted(runs, key=lambda x: x["wall_s"])[len(runs) // 2]
        lg = r["acc_log"]
        acc_all, acc_pre = [], []
        for b in range(B):
            n = 0
            for row in lg:
                x = row[b]
                if x < 0:
                    continue
                acc_all.append(x + 1)
                if n < eos_at[b]:
                    acc_pre.append(x + 1)
                n += x + 1
        out = r["out"]
        match = [sum(int(p == q) for p, q in zip(o, ao)) / gen for o, ao in zip(out, ar_out)]
        fdiv = []
        for o, ao in zip(out, ar_out):
            d = [i for i, (p, q) in enumerate(zip(o, ao)) if p != q]
            fdiv.append(d[0] if d else None)
        tr = [p["tau_rec"].get(name, {}).get(f"g{g}") for p in prompts]
        tau_v4b_same = (sum(t[0] for t in tr) / sum(t[1] for t in tr)) if tr and all(tr) else None
        e2e = B * gen / (pre + comp + dec)
        rows.append({
            "config": c, "draft": name, "gamma": g, "decode_tok_s": tps,
            "decode_tok_s_min_max": [min(x["tok_per_s"] for x in runs), max(x["tok_per_s"] for x in runs)],
            "speedup_decode": tps / ar_tps, "e2e_tok_s": e2e, "speedup_e2e": e2e / rows[0]["e2e_tok_s"],
            "prefill_s": pre, "compress_s": comp, "decode_s": dec, "rounds": mr["rounds"],
            "breakdown_s": {k: mr[k] for k in ("draft_s", "verify_s", "accept_s", "encode_s", "host_s")},
            "per_round_ms": {k: mr[k] / mr["rounds"] * 1e3 for k in ("draft_s", "verify_s", "accept_s", "encode_s",
                                                                      "host_s")},
            "tau_measured": sum(acc_all) / len(acc_all), "tau_measured_pre_eos": sum(acc_pre) / max(1, len(acc_pre)),
            "tau_v4b_same_samples": tau_v4b_same, "tau_eff_batch": gen / mr["rounds"],
            "match_vs_ar_mean": sum(match) / B, "n_seq_identical": sum(1 for f in fdiv if f is None),
            "first_divergence": fdiv, "peak_GiB": r.get("peak_GiB"),
            "draft_plan": r["draft_plan"], "verify_plan": r["verify_plan"], **cof(r),
            "v4b_computed": v4b_computed(task, ctx, B, name, g, table, bench)})
    pt = {"task": task, "ctx": ctx, "B": B, "gen": gen, "P": R["prefill"]["P"], "prefill_s": pre,
          "prefill_per_seq_s": R["prefill"]["per_seq_s"], "ar_eos_at": eos_at, "rows": rows,
          "cotenants_start": R["cotenants_start"], "cotenants_end": R["cotenants_end"],
          "nvsmi_apps_start": R["nvsmi_apps_start"], "nvsmi_apps_end": R["nvsmi_apps_end"], "raw": path,
          "cotenant_any": R.get("cotenant_any", bool(R["cotenants_start"] or R["cotenants_end"]))}
    return pt


def fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def tables(pts, tag):
    L = [f"# E7 magicvq {tag}: FULLY MEASURED end-to-end self-speculative decoding (TP=4, GPUs 4-7, bf16 "
         "Llama-3.1-8B-Instruct, real weights / prompts / prefill / compression / decode loop)", "",
         "All numbers below are wall-clock measurements of the real loop (no replayed acceptance, no modeled step "
         "times), except the column 'v4b computed', which is the previous estimate (v4b tau x synthetic-weight step "
         "times) for the same gamma. Speedup = tokens/s of the config / tokens/s of the AR baseline in the same run "
         "(same kernels, CUDA graphs, TP). tau meas = mean accepted+1 per active (sequence, round); batch tau_eff = "
         "gen / rounds (the static batch runs until its slowest sequence has gen tokens). gen = generated tokens per sequence (fixed, EOS ignored); static batch: a "
         "sequence that reached gen waits for the slowest one.", ""]
    for p in pts:
        L.append(f"## {p['task']} ctx {p['ctx']}, B={p['B']}, gen {p['gen']} (mean P {sum(p['P']) / len(p['P']):.0f}; "
                 f"prefill {p['prefill_s']:.1f} s total, {med(p['prefill_per_seq_s']):.2f} s / sequence)")
        L.append(f"co-tenants on GPUs 4-7 at start: {p['cotenants_start'] or 'none'}; at end: {p['cotenants_end'] or 'none'}; "
                 f"any config with a co-tenant before/after its timing: {'YES (CONTAMINATED rows marked *)' if p['cotenant_any'] else 'no (clean)'}")
        L.append("")
        L.append("| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | "
                 "v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | "
                 "ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in p["rows"]:
            if r["config"] == "ar":
                L.append(f"| AR{' *' if r['cotenant'] else ''} | - | {r['decode_tok_s']:.1f} ({r['decode_tok_s_min_max'][0]:.1f}-"
                         f"{r['decode_tok_s_min_max'][1]:.1f}) | 1.000 | - | - | - | - | - | {r['e2e_tok_s']:.1f} | 1.000 | "
                         f"step {r['step_ms']:.2f} ms | - | - | - |")
                continue
            v = r["v4b_computed"]
            pr = r["per_round_ms"]
            L.append(f"| {LABEL.get(r['draft'], r['draft'])}{' *' if r['cotenant'] else ''} | {r['gamma']} | {r['decode_tok_s']:.1f} "
                     f"({r['decode_tok_s_min_max'][0]:.1f}-{r['decode_tok_s_min_max'][1]:.1f}) | "
                     f"**{r['speedup_decode']:.3f}** | {fmt(v['speedup'] if v else None)} | "
                     f"{r['tau_measured']:.2f} ({r['tau_measured_pre_eos']:.2f}) | {fmt(r['tau_v4b_same_samples'], 2)} | "
                     f"{r['tau_eff_batch']:.2f} | {r['compress_s']:.2f} | {r['e2e_tok_s']:.1f} | {r['speedup_e2e']:.3f} | "
                     f"{pr['draft_s']:.1f} / {pr['verify_s']:.1f} / {pr['accept_s']:.2f} / {pr['encode_s']:.2f} / "
                     f"{pr['host_s']:.2f} | {r['rounds']} | {r['match_vs_ar_mean']:.3f} | {r['n_seq_identical']}/{p['B']} |")
        # best per draft family and VQ vs SnapKV
        best = {}
        for r in p["rows"][1:]:
            fam = "snapkv" if r["draft"].startswith("snapkv") else r["draft"]
            if fam not in best or r["speedup_decode"] > best[fam]["speedup_decode"]:
                best[fam] = r
        if "snapkv" in best:
            s = best["snapkv"]
            L.append("")
            for fam, r in best.items():
                if fam == "snapkv":
                    continue
                L.append(f"- best {LABEL.get(fam, fam)} (g{r['gamma']}) / best SnapKV ({s['draft']} g{s['gamma']}): "
                         f"decode {r['speedup_decode'] / s['speedup_decode']:.3f}, prefill+compress+decode "
                         f"{r['speedup_e2e'] / s['speedup_e2e']:.3f}")
        L.append("")
    return "\n".join(L)


def plot(pts, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    drafts = ["snapkv", "vqpL2+r128", "hyb512pL2"]
    names = {"snapkv": "SnapKV (best budget)", "vqpL2+r128": "RQ2 + 128 bf16 window", "hyb512pL2": "SnapKV-512 + RQ2 hybrid"}
    cols = {"snapkv": "#2a78d6", "vqpL2+r128": "#eb6834", "hyb512pL2": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    for i, p in enumerate(pts):
        for j, dname in enumerate(drafts):
            rs = [r for r in p["rows"][1:] if (r["draft"].startswith("snapkv") if dname == "snapkv" else r["draft"] == dname)]
            if not rs:
                continue
            r = max(rs, key=lambda x: x["speedup_decode"])
            x = i + (j - 1) * w
            ax.bar(x, r["speedup_decode"], w * 0.92, color=cols[dname], label=names[dname] if i == 0 else None)
            top = max(r["speedup_decode"], r["v4b_computed"]["speedup"] if r["v4b_computed"] else 0)
            ax.text(x, top + 0.012, f"{r['speedup_decode']:.2f}\ng{r['gamma']}", ha="center",
                    va="bottom", fontsize=7.5)
            if r["v4b_computed"]:
                ax.plot([x - w * 0.42, x + w * 0.42], [r["v4b_computed"]["speedup"]] * 2, color="k", lw=1.2, ls=":",
                        label="v4b computed estimate (same gamma)" if (i == 0 and j == 0) else None)
    ax.axhline(1.0, color="#888", lw=0.8)
    ax.set_xticks(range(len(pts)))
    ax.set_xticklabels([f"{p['task']} {p['ctx'] // 1024}k\nB={p['B']}, gen {p['gen']}" for p in pts])
    ax.set_ylabel("decode speedup vs AR (measured wall clock)")
    ax.set_title("Llama-3.1-8B, TP=4 RTX 4090: FULLY MEASURED self-speculative decoding (real weights, prompts, "
                 "prefill, compression, loop)", fontsize=9.5)
    ax.set_ylim(0.8, max(1.8, ax.get_ylim()[1]))
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--plot", action="store_true", help="only the plot, from the saved table json (any python with "
                    "matplotlib; nestvq has none)")
    a = ap.parse_args()
    if a.plot:
        return plot(json.load(open(f"{RES}/llama8b_magicvq_{a.tag}_table.json")),
                    f"{RES}/llama8b_magicvq_{a.tag}_speedup_measured.png")
    table = json.load(open(f"{RES}/llama8b_magicvq_v4b_table.json"))
    bench = json.load(open("/data/lsh/results_accept/magicvq/mgpu/bench_tp_v4b.json"))
    order = {"syn_quote": 0, "hotpotqa_q": 1, "pg19": 2}
    files = sorted(glob.glob(f"{OUT}/run_{a.tag}_*.json")) + sorted(glob.glob(f"{OUT}/run_{a.tag}g512_*.json"))
    pts = [analyze(f, table, bench) for f in files]
    pts.sort(key=lambda p: (order.get(p["task"], 9), -p["ctx"], -p["B"], p["gen"]))
    md = tables(pts, a.tag)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_{a.tag}_tables.md", "w").write(md + "\n")
        json.dump(pts, open(f"{RES}/llama8b_magicvq_{a.tag}_table.json", "w"), indent=1)
        from tools.results import save_result
        for p in pts:
            save_result("E7", f"llama8b_magicvq_{a.tag}_{p['task']}_ctx{p['ctx']}_B{p['B']}_gen{p['gen']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7), fully real "
                                "e2e self-spec (e2e_real.py)", "B": p["B"], "T": p["ctx"], "gen": p["gen"],
                                "task": p["task"], "P": p["P"], "configs": [r["config"] for r in p["rows"]]},
                        metrics={"rows": p["rows"], "prefill_s": p["prefill_s"],
                                 "cotenants_start": p["cotenants_start"], "cotenants_end": p["cotenants_end"],
                                 "nvsmi_apps_start": p["nvsmi_apps_start"], "nvsmi_apps_end": p["nvsmi_apps_end"]},
                        model_dir="/data/models/llama3.1_8b_instruct",
                        notes=f"raw: {p['raw']}; all speedups measured wall clock vs AR of the same run")


if __name__ == "__main__":
    main()
