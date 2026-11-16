"""Pre-training cost sweep over codebook configurations on a SYNTHETIC Llama-3.1-8B (real embed / norms / lm_head,
random nested-VQ linears + Hadamard signs B=256; timing only, no quality).

Per configuration (d, per-plane bits, L, q):
  target AR step (k = L, full vocab) at P=128; draft step for every prefix k < L with the FR-Spec draft vocab
  (V_d = 32k); target forward cost at M in {1, 2, 3, 4, 5, 8} (verify uses M = gamma+1); draft catch-up (M=2) per k;
  kernel launch plans per linear shape and k (smem bytes, smem vs global fallback, replication R, blocks/SM).
Protocol as v5 (l1_llama31_synthetic_cost_v5_frspec): CUDA graphs, deterministic split-K, 3 repeats, median.
Spec rounds are measured only for --measure-spec configs (C1 baseline, model validation); for every configuration
the round time is COMPOSED from measured parts: R(g, k) = g * t_draft(k) + t_fwd(M = g+1) + t_ovh, with t_ovh the
median overhead of the measured C1 rounds (or --ovh-ms). tau*, alpha*, and the tau = g+1 upper bound follow as in
llama_cost; they are [I] computed values, tau is an assumed variable.
Per-plane codebook precision: --qpats 16,8 16,4 (one comma list per run, length L; run name suffix q16-8).
Usage: python -m specrq.infer.config_sweep --configs C1 ... --qs 16 8 4 [--qpats 16,8 ...] --tag NAME
"""
from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

import torch

from specrq.infer.decode import ARDecoder, SpecDecoder
from specrq.infer.llama_cost import graph_us, req_alpha, rss_gb
from specrq.infer.model import GenLinear, NestModel

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {  # name -> (d, per-plane bits)
    "C1": (8, [8] * 5), "d16K256L10": (16, [8] * 10), "d8K16L10": (8, [4] * 10),
    "d8K4096L5": (8, [12] * 5), "d8K4096L3": (8, [12] * 3), "d16K4096L6": (16, [12] * 6),
    "d16K4096L7": (16, [12] * 7), "BF": (8, [12, 12, 8, 8]), "BL": (8, [8, 8, 12, 12]),
    # E4 cachefit: 2.0 bit/weight index targets
    "d8K256L2": (8, [8, 8]), "d16K256L4": (16, [8] * 4), "d8K16L4": (8, [4] * 4), "d8K4096K16": (8, [12, 4]),
    "d16K4096x2K256": (16, [12, 12, 8]), "d16K65536L2": (16, [16, 16]),
    # E4 cachefit2: per-plane 4/8/12 mixes (nestg B = 0); per-plane codebook q via --qpats
    "d8K16K4096": (8, [4, 12]), "d8K256K16x2": (8, [8, 4, 4]),
}
SHAPES = {"q": (4096, 4096), "k": (1024, 4096), "o": (4096, 4096), "gate": (14336, 4096), "down": (4096, 14336)}


def run_config(model_path, name, q, a, specialized=False):
    d, bits = CONFIGS[name]
    L = len(bits)
    t0, rss0 = time.time(), rss_gb()
    if specialized:  # existing C1 kernels (nest_gemv / nest_mma), regression vs v5
        m = NestModel.synthetic(model_path, verbose=False)
    else:
        m = NestModel.synthetic(model_path, vq={"d": d, "bits": bits, "L": L, "q": q}, verbose=False)
    from specrq.infer.frspec import coverage, default_freq_dir, load_subset
    fd = default_freq_dir(m.cfg)
    ids = load_subset(fd, a.vd, m.lm_head.shape[0])
    vd = m.set_draft_vocab(ids)
    torch.cuda.synchronize()
    res = {"name": name, "d": d, "bits": bits, "L": L, "q": q, "specialized_c1": specialized,
           "target_bpw_index": sum(bits) / d,
           "draft_bpw_index": {k: sum(bits[:k]) / d for k in range(1, L + 1)},
           "load": {"load_s": time.time() - t0, "peak_rss_gb_before": rss0, "peak_rss_gb_after": rss_gb(),
                    "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9},
           "linear_bytes_per_token": {k: m.linear_bytes(k) for k in range(1, L + 1)},
           "vd": vd, "vd_coverage": coverage(fd, ids), "ar": {}, "fwd": {}, "catchup": {}, "plans": {}, "spec": []}
    V = m.cfg.vocab_size
    gen = torch.Generator(device="cuda").manual_seed(0)
    P = a.P
    T = P + a.n_new + 16
    prompt = torch.randint(0, V, (P,), device="cuda", generator=gen)
    for k in [L] + list(range(1, L)):
        v = None if k == L else vd
        ar = ARDecoder(m, T, k=k, vd=v)
        ar.generate(prompt, 8)
        st = [ar.generate(prompt, a.n_new)[1] for _ in range(a.repeats)]
        step = [1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st]
        res["ar"][k] = {"vd": v or V, "step_us": statistics.median(step), "step_us_range": [min(step), max(step)]}
        if k == L:
            res["prefill_s"] = statistics.median(s["t_prefill_s"] for s in st)
        print(f"[{name} q{q}] AR k={k} vd={v or V}: {res['ar'][k]['step_us']:.1f} us", flush=True)
        del ar
        torch.cuda.empty_cache()
    cache = m.new_cache(T)
    m.prefill(prompt, cache, L)
    pos = torch.tensor(P, device="cuda")
    toks = torch.randint(0, V, (16,), device="cuda", generator=gen)
    for M in (1, 2, 3, 4, 5, 8):
        res["fwd"][M] = graph_us(lambda: m.forward(toks[:M], pos, cache, L, last_only=(M == 1)))
    for k in range(1, L):
        res["catchup"][k] = graph_us(lambda: m.forward(toks[:2], pos, cache, k, last_only=True, vd=vd))
    print(f"[{name} q{q}] fwd {res['fwd']} catchup {res['catchup']}", flush=True)
    del cache
    if not specialized:
        lay = m.layers[0]
        lins = {"q": lay["q_proj"], "k": lay["k_proj"], "o": lay["o_proj"], "gate": lay["gate_proj"],
                "down": lay["down_proj"]}
        for sn, lin in lins.items():
            res["plans"][sn] = {k: {"gemv": lin.plan("gemv", k), "mma_M5": lin.plan("mma", k, 5),
                                    "m1_path": lin.m1.get(k)} for k in range(1, L + 1)}
    if a.measure_spec:
        for g in a.gammas:
            for kd in range(1, L):
                sd = SpecDecoder(m, T, g, kd, vd=vd)
                sd.generate(prompt, 8)
                st = [sd.generate(prompt, a.n_new)[1] for _ in range(a.repeats)]
                r = {"gamma": g, "kd": kd}
                for key in ("t_draft_round_ms", "t_verify_ms", "t_overhead_ms", "tokens_per_s"):
                    r[key] = statistics.median(s[key] for s in st)
                res["spec"].append(r)
                print(f"[{name} q{q}] spec", r, flush=True)
                del sd
                torch.cuda.empty_cache()
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return res


def compose(res, gammas, ovh_ms):
    T_ar = res["ar"][res["L"]]["step_us"] / 1e3
    out = []
    for k in range(1, res["L"]):
        td = res["ar"][k]["step_us"] / 1e3
        for g in gammas:
            R = g * td + res["fwd"][min(g + 1, 8)] / 1e3 + ovh_ms
            tau = R / T_ar
            out.append({"kd": k, "draft_bpw_index": res["draft_bpw_index"][k], "gamma": g, "T_ar_ms": T_ar,
                        "t_draft_ms": td, "t_verify_ms": res["fwd"][min(g + 1, 8)] / 1e3, "ovh_ms": ovh_ms,
                        "round_ms": R, "tau_required": tau, "alpha_required_iid": req_alpha(tau, g),
                        "speedup_upper_bound": (g + 1) * T_ar / R})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--qs", type=int, nargs="*", default=[16, 8, 4])
    ap.add_argument("--qpats", nargs="*", default=[], help="per-plane codebook q lists, e.g. 16,8 (len = L)")
    ap.add_argument("--specialized-c1", action="store_true", help="also run C1 on the specialized kernels")
    ap.add_argument("--measure-spec", action="store_true")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 7])
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ovh-ms", type=float, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run-prefix", default=None, help="result name prefix (default l1_llama31_config_sweep_<tag>)")
    ap.add_argument("--no-save", action="store_true", help="functional smoke run (no JSON, co-tenant allowed)")
    a = ap.parse_args(argv)
    from specrq.nestkern.gpucheck import cotenants, require_alone
    cotenant_start = (cotenants() or "none") if a.no_save else require_alone()
    sys.path.insert(0, str(REPO))
    from tools.results import save_result
    qpats = [[int(v) for v in p.split(",")] for p in a.qpats]
    runs = ([("C1", 16, True)] if a.specialized_c1 else []) + [(c, q, False) for c in a.configs for q in a.qs + qpats]
    for name, q, spec in runs:
        cfg_start = cotenants() or "none"
        res = run_config(a.model, name, q, a, spec)
        ovh = a.ovh_ms if a.ovh_ms is not None else (
            statistics.median(r["t_overhead_ms"] for r in res["spec"]) if res["spec"] else 0.1)
        res["model"] = compose(res, a.gammas, ovh)
        if res["spec"]:  # validation of the composed round time against measured rounds
            res["compose_check"] = [{"gamma": r["gamma"], "kd": r["kd"],
                                     "measured_round_ms": r["t_draft_round_ms"] + r["t_verify_ms"] + r["t_overhead_ms"],
                                     "composed_round_ms": next(x["round_ms"] for x in res["model"]
                                                               if x["gamma"] == r["gamma"] and x["kd"] == r["kd"])}
                                    for r in res["spec"]]
        qn = "-".join(map(str, q)) if isinstance(q, list) else q
        run = f"{a.run_prefix or 'l1_llama31_config_sweep_' + a.tag}_{name}_q{qn}" + ("_specialized" if spec else "")
        if a.no_save:
            print("[smoke] model rows", len(res["model"]), "plans", list(res["plans"]), "ar", res["ar"], flush=True)
            continue
        print("saved", save_result("E4", run, config={"co_tenant_start": cfg_start, "co_tenant_sweep_start": cotenant_start, "co_tenant_end": cotenants() or "none",
            "setting": "SYNTHETIC Llama-3.1-8B: real embed/norm/lm_head, random nested-VQ linears + Hadamard signs "
                       "(B=256); timing only, no quality", "config": name, "d": res["d"], "bits": res["bits"],
            "q": q, "specialized_c1_kernels": spec, "P": a.P, "n_new": a.n_new, "repeats": a.repeats, "vd": a.vd,
            "gammas": a.gammas, "deterministic_splitk": True, "smem_max": GenLinear.SMEM_MAX,
            "m1_path_policy": GenLinear.M1_PATH,
            "compose_note": "round_ms = g*t_draft(k) + t_fwd(M=g+1) + ovh; tau/alpha are assumed, [I]"},
            metrics=res, model_dir=a.model), flush=True)


if __name__ == "__main__":
    main()
