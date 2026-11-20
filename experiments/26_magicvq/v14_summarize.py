"""E7 v14 summary: bf16 vs Marlin INT4 draft weights, measured e2e (run_v14w4{off,on}{1,2}_<task>_ctx32768_B16.json).
Per config and run: all-active tok/s (median of reps), speedup vs the AR of the same run, draft ms/round, verify ms/round,
measured c = (draft ms/round / gamma) / AR ms/step, tau, AR-overlap match, rep consistency, co-tenants, peak memory.
Usage: python experiments/26_magicvq/v14_summarize.py <tagprefix e.g. v14w4> syn_quote [pg19 ...]  -> markdown on stdout, JSON next to raw."""
import json
import statistics as S
import sys

OUT = "/data/lsh/results_accept/magicvq/e2e_real"
import os
BS = int(os.environ.get("V14_B", "16"))  # batch of the runs


def load(tag, task):
    """final JSON, else the .partial (a run that died later, e.g. OOM: its finished configs); 'partial' flag set"""
    f = f"{OUT}/run_{tag}_{task}_ctx32768_B{BS}.json"
    try:
        return json.load(open(f))
    except FileNotFoundError:
        r = json.load(open(f + ".partial"))
        r["partial"] = True
        r.setdefault("cotenant_any", bool(r["co_start"]["cotenant"] or any(
            (x.get("co_before") or {}).get("cotenant") or (x.get("co_after") or {}).get("cotenant")
            for x in r["configs"].values())))
        return r


def summ(r):
    ar = r["configs"]["ar"]
    ar_tps = S.median(x["tok_per_s"] for x in ar["runs"])
    out = {"ar": {"tok_s": ar_tps, "step_ms": ar["step_ms"]}}
    ar_out = ar["out"]
    for c, v in r["configs"].items():
        if c == "ar":
            continue
        g = int(c.split(":")[1])
        aa = v["all_active"]
        rr = aa["runs"]
        tps = S.median(x["tok_per_s"] for x in rr)
        dr = S.median(x["draft_s"] / x["rounds"] for x in rr) * 1e3
        vr = S.median(x["verify_s"] / x["rounds"] for x in rr) * 1e3
        oth = S.median((x["accept_s"] + x["encode_s"] + x["host_s"]) / x["rounds"] for x in rr) * 1e3
        fd = aa.get("first_divergence_vs_ar")
        out[c] = {"gamma": g, "tok_s": tps, "tok_s_min": min(x["tok_per_s"] for x in rr),
                  "tok_s_max": max(x["tok_per_s"] for x in rr), "speedup": tps / ar_tps,
                  "draft_ms_round": dr, "draft_ms_step": dr / g, "verify_ms_round": vr, "other_ms_round": oth,
                  "c": dr / g / ar["step_ms"], "verify_over_ar_step": vr / ar["step_ms"], "tau": aa["tau"],
                  "rounds": rr[0]["rounds"], "tokens": aa["total_tokens"],
                  "draft_attn_ms_step": v["draft_tune"]["attn_ms"],
                  "rows_diverged_vs_ar": sum(x is not None for x in fd) if fd else None,
                  "first_div": [x for x in fd if x is not None] if fd else None,
                  "rep_out_differs": "rep_out_differs" in aa, "rep_ngen_differs": aa["rep_ngen_differs"],
                  "peak_GiB": v["peak_GiB"], "out": aa["out"]}
    out["_ar_out"] = ar_out
    return out


def main():
    md, js = [], {}
    P = sys.argv[1]  # tag prefix, e.g. v14w4
    for task in sys.argv[2:]:
        runs = {}
        for t in (P + "off1", P + "on1", P + "off2", P + "on2"):
            try:
                r = load(t, task)
            except FileNotFoundError:
                continue
            runs[t] = (r, summ(r))
        js[task] = {t: {"cotenant_any": r["cotenant_any"], "draft_w4": r.get("draft_w4"),
                        "prefill_peak_GiB": r["prefill"]["peak_GiB"], "partial": r.get("partial", False),
                        "configs": {c: {k: v for k, v in x.items() if k != "out"} for c, x in s.items() if c != "_ar_out"}}
                    for t, (r, s) in runs.items()}
        md.append(f"## {task} 32k, B{BS}, gen 256, all-active (median of 3 reps per run)\n")
        md.append("runs: " + ", ".join(f"{t} (co-tenant {r['cotenant_any']}{', PARTIAL: ' + str(list(r['configs'])) if r.get('partial') else ''})" for t, (r, _) in runs.items()) + "\n")
        md.append("| config | weights | tok/s r1 / r2 | speedup vs AR | draft ms/round | draft ms/step | verify ms/round | "
                  "other ms/round | c | tau | peak GiB | rows != AR (first div) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        cfgs = list(next(iter(runs.values()))[1].keys())
        for c in cfgs:
            if c == "_ar_out":
                continue
            for w in ("off", "on"):
                xs = [runs[t][1][c] for t in (f"{P}{w}1", f"{P}{w}2") if t in runs and c in runs[t][1]]
                if not xs:
                    continue
                wn = "bf16" if w == "off" else "INT4"
                if c == "ar":
                    md.append(f"| AR | ({w} runs) | {' / '.join('%.1f' % x['tok_s'] for x in xs)} | 1 | - | - | - | - | - | - | - | - |")
                    continue
                f = lambda k, p=2: " / ".join(f"{x[k]:.{p}f}" for x in xs)  # noqa: E731
                md.append(f"| {c} | {wn} | {f('tok_s', 1)} | {f('speedup', 3)} | {f('draft_ms_round')} | "
                          f"{f('draft_ms_step')} | {f('verify_ms_round')} | {f('other_ms_round')} | {f('c', 3)} | "
                          f"{f('tau', 3)} | {f('peak_GiB')} | "
                          f"{' / '.join(str(x['rows_diverged_vs_ar']) + ' ' + str(x['first_div']) for x in xs)} |")
        # INT4 vs bf16 gain per config (mean of the two runs' medians)
        md.append("\n| config | tok/s bf16 -> INT4 | gain | draft ms/round bf16 -> INT4 | saved ms/step | tau bf16 -> INT4 | outputs same across reps |")
        md.append("|---|---|---|---|---|---|---|")
        for c in cfgs:
            if c in ("_ar_out", "ar"):
                continue
            off = [runs[t][1][c] for t in (P + "off1", P + "off2") if t in runs and c in runs[t][1]]
            on = [runs[t][1][c] for t in (P + "on1", P + "on2") if t in runs and c in runs[t][1]]
            if not off or not on:
                continue
            m = lambda xs, k: S.mean(x[k] for x in xs)  # noqa: E731
            same = all(x["out"] == off[0]["out"] for x in off) and all(x["out"] == on[0]["out"] for x in on)
            md.append(f"| {c} | {m(off, 'tok_s'):.1f} -> {m(on, 'tok_s'):.1f} | {m(on, 'tok_s') / m(off, 'tok_s') - 1:+.1%} | "
                      f"{m(off, 'draft_ms_round'):.2f} -> {m(on, 'draft_ms_round'):.2f} | "
                      f"{m(off, 'draft_ms_step') - m(on, 'draft_ms_step'):.2f} | {m(off, 'tau'):.3f} -> {m(on, 'tau'):.3f} | {same} |")
        md.append("")
    print("\n".join(md))
    json.dump(js, open(f"{OUT}/{P}_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
