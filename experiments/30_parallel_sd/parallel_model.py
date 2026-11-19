"""E9 DERIVED (not measured end-to-end): upper-bound speedup of an idealized PEARL-style parallel schedule vs the serial
self-speculative schedule, per draft config, from the measured co-run microbench (corun_bench.py) + measured tau /
acceptance logs of the v6e2e / v8 e2e runs (syn_quote 32k B16, all-active, TP4 4090).

Serial round   : T_ser = T_dG + T_v + T_o                          tokens = tau
Parallel (U)   : T = T_cor + (1 - p_row) * T_dG + T_o             (task formula: drafting the next gamma during the
                 verify; on rejection the pre-drafted tokens are wasted and gamma fresh draft steps run serially;
                 p_row = P(all gamma accepted) per row and round)
Parallel (L)   : T = T_cor + (1 - p_batch) * T_dG + T_o           (lock-step batch: any rejecting row stalls the batch;
                 p_batch = P(all B rows accept all gamma) per round)
Ideal (U, no contention): T_cor replaced by max(T_dG, T_v)
tokens / round = tau for every schedule (optimistic: the verify's bonus token is assumed to be usable as in the serial
schedule; in a real PEARL loop it must also match the pre-drafted continuation).
T_dG, T_v, T_cor: corun_bench iso / co-run medians (T_cor = best priority of 'dG || vG'); T_o = per-round
non-draft/verify time of the source e2e run (accept + encode + host); T_AR = that run's AR step.
"""
from __future__ import annotations

import argparse
import json
import statistics

E2E = "/data/lsh/results_accept/magicvq/e2e_real"
SRC = {  # (draft, gamma) -> run file with that all-active config (syn_quote ctx32768 B16)
    ("snapkv4096", 4): "run_v8e2ea", ("hyb512pL2@t2w", 6): "run_v8e2ea", ("hyb512pL2@t2w", 4): "run_v6e2e",
    ("hyb512pL1@t1w", 4): "run_v8e2ea", ("hyb512pL1@t1w", 6): "run_v8e2ea",
    ("vqpL2+r128@t2w", 4): "run_v8e2eb", ("vqpL2+r128@t2w", 6): "run_v6e2e", ("vqpL1+r128@t2w", 4): "run_v8e2eb",
}
LABEL = {"snapkv4096": "SnapKV-4096", "hyb512pL2@t2w": "hyb512+RQ2 t2w", "hyb512pL1@t1w": "hyb512+RQ1 t1w",
         "vqpL2+r128@t2w": "RQ2+r128 t2w", "vqpL1+r128@t2w": "RQ1+r128 t2w-stage1 (nested)", "kvfree": "KV-free"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corun", required=True)
    ap.add_argument("--task", default="syn_quote")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--diag", default=None, help="concurrency_diag.py output (synthetic stream-concurrency check)")
    a = ap.parse_args()
    C = json.load(open(a.corun))
    rows, md = [], []
    md.append(f"# E9 parallel draft||verify: co-run microbench + DERIVED PEARL-style bound ({a.task} ctx{a.ctx} B{a.B}, "
              "TP4 RTX 4090, NCCL_P2P_DISABLE=1)\n")
    md.append(f"source: `{a.corun}`; co-tenant any: {C['cotenant_any']}; launch ms {C['launch_ms']}\n")
    md.append("## Measured: isolated vs co-run (median ms over 3x10 iters, max over ranks; spread = min-max)\n")
    md.append("overlap eff = (T_d + T_v - T_cor) / min(T_d, T_v); slowdown = side time in co-run / isolated; "
              "priorities: eq / vhi (verify stream priority -1) / dhi (draft high)\n")
    md.append("| draft | n draft steps | gamma | T_d iso | T_v iso | serial sum | T_cor eq (min-max) | T_cor vhi | T_cor dhi | "
              "draft slowdown eq | verify slowdown eq | overlap eff eq |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for dn, D in C["drafts"].items():
        for g in C["args"]["gammas"]:
            tv = D["iso"][f"v{g}"]["total_ms"]
            for n in (1, g):
                td = D["iso"][f"d{n}"]["total_ms"]
                cs = {p: D["corun"][f"d{n}|v{g}|{p}"] for p in ("eq", "vhi", "dhi")}
                e = cs["eq"]
                ov = (td + tv - e["total_ms"]) / min(td, tv)
                md.append(f"| {LABEL.get(dn, dn)} | {n} | {g} | {td:.2f} | {tv:.2f} | {td + tv:.2f} | "
                          f"{e['total_ms']:.2f} ({e['total_min']:.2f}-{e['total_max']:.2f}) | {cs['vhi']['total_ms']:.2f} | "
                          f"{cs['dhi']['total_ms']:.2f} | {e['side_ms'][0] / td:.3f} | {e['side_ms'][1] / tv:.3f} | {ov:.3f} |")
    if a.diag:
        Dg = json.load(open(a.diag))["res"]
        md.append("\n## Measured: synthetic stream-concurrency check (concurrency_diag.py, B16, same streams / communicators)\n")
        md.append("sleep = GPU sleep kernel (latency only); gemv = memory-bound bf16 GEMV chain (~1 TP4 rank of weights); "
                  "comm / comm7 = one forward's 66 all_reduces at 1 / 7 rows per sequence.\n")
        md.append("| pair | iso A | iso B | sum | max | co-run total | co-run total (B high priority) |")
        md.append("|---|---|---|---|---|---|---|")
        for k, v in Dg.items():
            if "|" in k and not k.endswith("vhi"):
                x, y = k.split("|")
                ia, ib = Dg[f"iso {x}"]["total_ms"], Dg[f"iso {y}"]["total_ms"]
                md.append(f"| {x} \\|\\| {y} | {ia:.2f} | {ib:.2f} | {ia + ib:.2f} | {max(ia, ib):.2f} | "
                          f"{v['total_ms']:.2f} | {Dg[k + ' vhi']['total_ms']:.2f} |")
    # components
    md.append("\n## Measured: components (gamma = %d), each co-run with the full other side\n" % C["args"]["parts_gamma"])
    md.append("d_comm = the 66 all_reduces of one draft step only; d_gemm = draft step without comm and attention; "
              "d_attn = draft attention of all layers only; v_* the same for the verify. 'hidden' = (T_part + T_full - "
              "T_cor) / T_part.\n")
    md.append("| draft | pair | T_part iso | T_full iso | T_cor | hidden fraction of the part |")
    md.append("|---|---|---|---|---|---|")
    g = C["args"]["parts_gamma"]
    first = next(iter(C["drafts"].values()))
    vparts = {k: first["parts"].get(f"v_{k}", {}).get("total_ms") for k in ("comm", "gemm", "attn")}
    for dn, D in C["drafts"].items():
        P = D["parts"]
        tv = D["iso"][f"v{g}"]["total_ms"]
        td = D["iso"]["d1"]["total_ms"]
        for k in ("comm", "gemm", "attn"):
            if f"d_{k}" in P:
                tp, tc = P[f"d_{k}"]["total_ms"], P[f"d_{k}|v{g}"]["total_ms"]
                md.append(f"| {LABEL.get(dn, dn)} | d_{k} \\|\\| v{g} | {tp:.2f} | {tv:.2f} | {tc:.2f} | {(tp + tv - tc) / tp:.3f} |")
        for k in ("comm", "gemm", "attn"):
            if f"d1|v_{k}" in P and vparts[k]:
                tp, tc = vparts[k], P[f"d1|v_{k}"]["total_ms"]
                md.append(f"| {LABEL.get(dn, dn)} | d1 \\|\\| v_{k} | {tp:.2f} | {td:.2f} | {tc:.2f} | {(tp + td - tc) / tp:.3f} |")
    # derived model
    md.append("\n## DERIVED: idealized PEARL-style schedule vs serial (tau / acceptance from v6e2e / v8 measured all-active runs)\n")
    md.append(__doc__.split("\n\n", 1)[1].strip().replace("\n", "  \n") + "\n")
    md.append("| draft | gamma | tau aa | p_row | p_batch | T_AR | T_dG | T_v | T_o | T_cor best | S serial (model) | "
              "S serial meas (e2e) | S par U | S par L | S par U ideal | U / serial |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (dn, g), src in SRC.items():
        if dn not in C["drafts"] or g not in C["args"]["gammas"]:
            continue
        try:
            r = json.load(open(f"{E2E}/{src}_{a.task}_ctx{a.ctx}_B{a.B}.json"))
        except FileNotFoundError:
            continue
        cfg = r["configs"][f"{dn}:{g}"]["all_active"]
        tau = cfg["tau"]
        lg = [[x for x in rw if x >= 0] for rw in cfg["acc_log"]]
        p_row = sum(x == g for rw in lg for x in rw) / sum(len(rw) for rw in lg)
        p_batch = statistics.mean(all(x == g for x in rw) for rw in lg)
        runs = cfg["runs"]
        t_o = statistics.median((x["wall_s"] - x["draft_s"] - x["verify_s"]) / x["rounds"] for x in runs) * 1e3
        s_meas = statistics.median(x["tok_per_s"] for x in runs) / statistics.median(
            x["tok_per_s"] for x in r["configs"]["ar"]["runs"])
        t_ar = r["configs"]["ar"]["step_ms"]
        D = C["drafts"][dn]
        tdg, tv = D["iso"][f"d{g}"]["total_ms"], D["iso"][f"v{g}"]["total_ms"]
        tcor = min(D["corun"][f"d{g}|v{g}|{p}"]["total_ms"] for p in ("eq", "vhi", "dhi"))
        s_ser = tau * t_ar / (tdg + tv + t_o)
        s_u = tau * t_ar / (tcor + (1 - p_row) * tdg + t_o)
        s_l = tau * t_ar / (tcor + (1 - p_batch) * tdg + t_o)
        s_i = tau * t_ar / (max(tdg, tv) + (1 - p_row) * tdg + t_o)
        rows.append({"draft": dn, "gamma": g, "tau": tau, "p_row": p_row, "p_batch": p_batch, "T_AR": t_ar,
                     "T_dG": tdg, "T_v": tv, "T_o": t_o, "T_cor": tcor, "S_serial_model": s_ser,
                     "S_serial_measured": s_meas, "S_par_U": s_u, "S_par_L": s_l, "S_par_U_ideal": s_i, "src": src})
        md.append(f"| {LABEL[dn]} | {g} | {tau:.2f} | {p_row:.3f} | {p_batch:.3f} | {t_ar:.2f} | {tdg:.2f} | {tv:.2f} | "
                  f"{t_o:.2f} | {tcor:.2f} | {s_ser:.3f} | {s_meas:.3f} | {s_u:.3f} | {s_l:.3f} | {s_i:.3f} | "
                  f"{s_u / s_ser:.3f} |")
    best = {}
    for x in rows:
        for k in ("S_serial_model", "S_par_U", "S_par_L", "S_par_U_ideal"):
            if x[k] > best.get(k, (0,))[0]:
                best[k] = (x[k], f"{LABEL[x['draft']]} g{x['gamma']}")
    md.append("\nbest per schedule: " + "; ".join(f"{k} {v[0]:.3f} ({v[1]})" for k, v in best.items()))
    open(a.out_md, "w").write("\n".join(md) + "\n")
    json.dump({"rows": rows, "best": best, "corun": a.corun}, open(a.out_json, "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
