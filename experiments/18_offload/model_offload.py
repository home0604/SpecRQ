"""Offload-regime model for nested weight-VQ self-speculative decoding (prefix planes on GPU, residual planes on CPU).

Inputs (all measured, see bench_cpu.py / bench_gpu.py / results/E4/q32b_L2_synth_*.json):
  T_cpu(M)   : one decode step of one residual plane (d8 K256, 64 layers x 7 linears of Qwen2.5-32B) on one socket
  T_pcie(M)  : per-layer sum of the 4 split-stage round trips (qkv, o, gate+up, down) x layers, GPU kernels excluded
  T_gpu_k(M) : synthetic 32B forward graph with k planes, full-vocab lm_head
Model (all outputs are MODELED, not measured end to end):
  T_split(M) for the split forward (GPU k_gpu planes + CPU residual planes), three variants
    ideal  = max(T_gpu(M), T_cpu(M) + T_pcie(M))                       (perfect overlap, lower bound)
    par    = T_cpu(M) + T_pcie(M) + T_gpu_other(M)                     (linears overlap; attention/norms/lm_head/glue
                                                                         on the GPU stay on the critical path)
    serial = T_gpu(M) + T_cpu(M) + T_pcie(M)                           (no overlap, upper bound)
  T_gpu_other(M) = T_gpu_k1(M) - T_plane(M),  T_plane(M) = T_gpu_k2(M) - T_gpu_k1(M)
  speed-up(gamma, tau) = tau * T_split(1) / (gamma * T_draft + T_split(gamma + 1))
S2 (hypothetical Qwen2.5-72B shapes) scales CPU time and GPU plane time by linear-weight bytes, GPU "other" by 1.6
(lm_head 2.49 vs 1.56 GB), PCIe by a latency + bytes fit applied to the 72B stage sizes and 80 layers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TAUS = [1.5, 2.0, 2.5, 3.0]
GAMMAS = [1, 2, 3, 4]
Q32 = dict(h=5120, i=27648, L=64, kv=1024, V=152064)
Q72 = dict(h=8192, i=29568, L=80, kv=1024, V=152064)


def lin_params(c):
    h, i, kv = c["h"], c["i"], c["kv"]
    return c["L"] * (2 * h * h + 2 * h * kv + 3 * h * i)


def stage_sizes(c):  # (d2h elems, h2d elems) per token row, 4 stages per layer
    h, i, kv = c["h"], c["i"], c["kv"]
    return [(h, h + 2 * kv), (h, h), (h, 2 * i), (i, h)]


def interp(tab, M):
    ks = sorted(tab)
    if M in tab:
        return tab[M]
    lo = max(k for k in ks if k < M)
    hi = min(k for k in ks if k > M)
    return tab[lo] + (tab[hi] - tab[lo]) * (M - lo) / (hi - lo)


def load_inputs(cpu_json, gpu_json, kernel, pcie_jsons=()):
    cpu = json.load(open(cpu_json))
    T_cpu = {r["M"]: r["wall_s"]["median"] * 1e3 for r in cpu["sweep"]["runs"] if r["kernel"] == kernel}
    g = json.load(open(gpu_json))
    rows = g["fwd"]["rows"]
    T_gpu = {k: {r["M"]: r["us"]["median"] / 1e3 for r in rows if r["k"] == k and r["vd"] is None} for k in (1, 2)}
    T_draft_fwd_vd = [r["us"]["median"] / 1e3 for r in rows if r["k"] == 1 and r["vd"]][0]
    # PCIe: per stage round trip minus the two GPU kernels; fit latency + bytes. With several runs, take the median
    # over runs of each stage's median overhead (host jitter showed up as one-off 2x outliers at random positions).
    runs = [json.load(open(p))["pcie"]["stages"] for p in pcie_jsons] or [g["pcie"]["stages"]]
    st = {}
    for key in runs[0]:
        ovs = sorted(r[key]["roundtrip_us"]["median"] - r[key]["gpu_kernels_only_us"]["median"] for r in runs)
        st[key] = {"roundtrip_us": {"median": ovs[len(ovs) // 2]}, "gpu_kernels_only_us": {"median": 0.0},
                   "bytes_d2h": runs[0][key]["bytes_d2h"], "bytes_h2d": runs[0][key]["bytes_h2d"],
                   "overheads_all_runs_us": ovs}
    pts = []
    for key, r in st.items():
        ov = r["roundtrip_us"]["median"] - r["gpu_kernels_only_us"]["median"]
        pts.append((r["bytes_d2h"] + r["bytes_h2d"], ov))
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)
    a = my - b * mx
    T_pcie32 = {}
    for M in sorted({int(k.split("_M")[1]) for k in st}):
        per_layer = sum(st[f"{s}_M{M}"]["roundtrip_us"]["median"] - st[f"{s}_M{M}"]["gpu_kernels_only_us"]["median"]
                        for s in ("qkv", "o", "gateup", "down"))
        T_pcie32[M] = per_layer * Q32["L"] / 1e3
    return dict(T_cpu=T_cpu, T_gpu=T_gpu, T_draft_fwd_vd=T_draft_fwd_vd, T_pcie32=T_pcie32, pcie_stage_overhead_us=st,
                pcie_fit={"lat_us": a, "us_per_byte": b, "GBps_eff": 1e-3 / b if b > 0 else None})


def pcie_fit_ms(fit, c, M):
    per_layer = sum(fit["lat_us"] + fit["us_per_byte"] * 2 * M * (d + h) for d, h in stage_sizes(c))
    return per_layer * c["L"] / 1e3


def scenario(name, T_draft, T_gpu_split, T_gpu_other, T_cpu, T_pcie, T_ref_gpu_ar=None, note=""):
    """T_*: functions of M (ms). Returns rows for gamma x tau and the split times."""
    out = {"name": name, "note": note, "T_draft_ms": T_draft, "split": {}, "rows": []}
    for M in (1, 2, 3, 4, 5):
        g, o, cp, pc = T_gpu_split(M), T_gpu_other(M), T_cpu(M), T_pcie(M)
        out["split"][M] = {"gpu": g, "gpu_other": o, "cpu": cp, "pcie": pc,
                           "ideal": max(g, cp + pc), "par": cp + pc + o, "serial": g + cp + pc}
    for var in ("ideal", "par", "serial"):
        T_ar = out["split"][1][var]
        out[f"T_AR_split_{var}_ms"] = T_ar
        out[f"c_{var}"] = T_draft / T_ar
        for gm in GAMMAS:
            T_v = out["split"][gm + 1][var]
            rnd = gm * T_draft + T_v
            row = {"variant": var, "gamma": gm, "T_verify_ms": T_v, "round_ms": rnd,
                   "tau_breakeven": rnd / T_ar, "tau_for_1.3x": 1.3 * rnd / T_ar}
            for tau in TAUS:
                if tau <= gm + 1:
                    row[f"speedup_tau{tau}"] = tau * T_ar / rnd
                    if T_ref_gpu_ar:
                        row[f"vs_gpu_ar_tau{tau}"] = tau * T_ref_gpu_ar / rnd
            out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, help="bench_cpu json (16 threads)")
    ap.add_argument("--gpu", required=True, help="bench_gpu json")
    ap.add_argument("--pcie", nargs="*", default=[], help="bench_gpu jsons with a pcie part (median over runs)")
    ap.add_argument("--cpu-alt", default=None, help="second bench_cpu json (e.g. 32 threads) for an extra S1 row")
    ap.add_argument("--kernel", default="perm_i8_tile")
    ap.add_argument("--exact-kernel", default="gather_f32")
    ap.add_argument("--draft-ms", type=float, default=None, help="draft step (default: E4 measured AR k1 vd32K)")
    ap.add_argument("--save", default=None, help="save_result run name (E4)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra json files to embed as raw inputs")
    a = ap.parse_args()
    inp = load_inputs(a.cpu, a.gpu, a.kernel, a.pcie)
    inp_exact = load_inputs(a.cpu, a.gpu, a.exact_kernel, a.pcie)
    e4 = json.load(open(REPO / "results/E4/q32b_L2_synth_spec_upperbound.json"))
    ar_k2_gpu = e4["metrics"]["ar"][0]["step_us"]["median"] / 1e3
    ar_k1_vd = [r for r in e4["metrics"]["ar"] if r["k"] == 1 and r["vd"]][0]["step_us"]["median"] / 1e3
    T_draft = a.draft_ms or ar_k1_vd
    Tg = inp["T_gpu"]
    plane32 = lambda M: Tg[2][M] - Tg[1][M]
    other32 = lambda M: Tg[1][M] - plane32(M)
    r_lin = lin_params(Q72) / lin_params(Q32)
    res = {"inputs": {k: v for k, v in inp.items()}, "inputs_exact_kernel_T_cpu": inp_exact["T_cpu"],
           "T_ar_gpu_resident_k2_ms": ar_k2_gpu, "T_draft_k1_vd32K_ms_E4": ar_k1_vd, "T_draft_used_ms": T_draft,
           "lin_param_ratio_72_32": r_lin, "scenarios": []}

    def cpu_fn(tab, f=1.0, flat=False):
        return lambda M: f * (tab[1] if flat else interp(tab, M))
    pc32 = lambda M: interp(inp["T_pcie32"], M)
    # ---------------- S1: 32B d8K256L2, GPU plane 1, CPU plane 2
    for kern, tab in ((a.kernel, inp["T_cpu"]), (a.exact_kernel, inp_exact["T_cpu"])):
        res["scenarios"].append(scenario(f"S1_32B_L2_cpu1plane_{kern}", T_draft, lambda M: Tg[1][M], other32,
                                         cpu_fn(tab), pc32, ar_k2_gpu,
                                         "GPU k1 (prefix) + CPU plane 2; draft = GPU k1 AR step (vd32K, E4)"))
    if a.cpu_alt:
        alt = load_inputs(a.cpu_alt, a.gpu, a.kernel, a.pcie)["T_cpu"]
        res["inputs_alt_T_cpu"] = alt
        res["scenarios"].append(scenario(f"S1_32B_L2_cpu1plane_{a.kernel}_alt", T_draft, lambda M: Tg[1][M], other32,
                                         cpu_fn(alt), pc32, ar_k2_gpu, f"CPU T(M) from {a.cpu_alt}"))
    # sensitivity: 2 sockets (CPU x0.5), flat CPU M-scaling
    res["scenarios"].append(scenario("S1_sens_2socket", T_draft, lambda M: Tg[1][M], other32,
                                     cpu_fn(inp["T_cpu"], 0.5), pc32, ar_k2_gpu, "[hypothetical] CPU time halved"))
    res["scenarios"].append(scenario("S1_sens_flatM", T_draft, lambda M: Tg[1][M], other32,
                                     cpu_fn(inp["T_cpu"], flat=True), pc32, ar_k2_gpu,
                                     "[hypothetical] CPU T(M) = T(1) (memory-bound kernel up to M=5)"))
    # ---------------- S2: Qwen2.5-72B shapes
    plane72 = lambda M: r_lin * plane32(M)
    other72 = lambda M: 1.6 * other32(M)
    pc72 = lambda M: pcie_fit_ms(inp["pcie_fit"], Q72, M)
    draft72_k1 = T_draft * r_lin  # whole 32B k1 (vd32K) AR step scaled by linear bytes (conservative)
    for L_, ncpu, kgpu in ((3, 1, 2), (4, 2, 2)):
        gpu72 = (lambda kg: (lambda M: other72(M) + kg * plane72(M)))(kgpu)
        cpu72 = cpu_fn(inp["T_cpu"], r_lin * ncpu)
        for dname, Td in (("draft_k1", draft72_k1), ("draft_k2", gpu72(1))):
            res["scenarios"].append(scenario(
                f"S2_72B_L{L_}_gpu{kgpu}_cpu{ncpu}_{dname}", Td, gpu72, other72, cpu72, pc72, None,
                f"[extrapolated] 72B shapes: GPU planes 1..{kgpu}, CPU {ncpu} plane(s); {dname}"))
        res["scenarios"].append(scenario(
            f"S2_72B_L{L_}_gpu{kgpu}_cpu{ncpu}_draft_k1_2socket", draft72_k1, gpu72, other72,
            cpu_fn(inp["T_cpu"], 0.5 * r_lin * ncpu), pc72, None, "[hypothetical] + 2 sockets"))
    # print compact tables
    for s in res["scenarios"]:
        sp = s["split"]
        print(f"\n== {s['name']}  T_draft {s['T_draft_ms']:.2f} ms  c(par) {s['c_par']:.3f}  {s['note']}")
        print("  M  gpu   other  cpu    pcie   ideal  par    serial")
        for M, v in sp.items():
            print(f"  {M}  {v['gpu']:.2f}  {v['gpu_other']:.2f}  {v['cpu']:.2f}  {v['pcie']:.2f}  {v['ideal']:.2f}  "
                  f"{v['par']:.2f}  {v['serial']:.2f}")
        for var in ("ideal", "par", "serial"):
            line = [f"  [{var}] T_AR {s[f'T_AR_split_{var}_ms']:.1f}"]
            for r in s["rows"]:
                if r["variant"] != var:
                    continue
                sps = " ".join(f"{r.get(f'speedup_tau{t}', float('nan')):.2f}" for t in TAUS)
                line.append(f"g{r['gamma']}: tau1.3x={r['tau_for_1.3x']:.2f} sp[{sps}]")
            print(" | ".join(line))
    if a.save:
        from tools.results import save_result
        extra = {p: json.load(open(p)) for p in a.extra}
        print("saved", save_result("E4", a.save, config={
            "setting": "offload regime: CPU residual-plane kernel micro-benchmark + PCIe round trips + GPU k1/k2 "
                       "synthetic 32B forward; split-forward and speculative speed-up are MODELED",
            "cpu": "Xeon Gold 6526Y, one socket (NUMA node1, CPUs 16-31 [+48-63 HT]), first-touch local memory",
            "gpu": "RTX 4090 on NUMA node1", "model_shapes": {"S1": Q32, "S2": Q72},
            "plane": "d8 K256, 1 bit/weight per plane, AQLM-style codebook per linear [256, 8], per-row scale",
            "kernel_main": a.kernel, "kernel_exact": a.exact_kernel, "taus": TAUS, "gammas": GAMMAS,
            "formula": "speedup = tau * T_split(1) / (gamma * T_draft + T_split(gamma+1))"},
            metrics={"model": res, "raw": extra},
            model_dir="/data/models/qwen2.5_32b_instruct"))


if __name__ == "__main__":
    main()
