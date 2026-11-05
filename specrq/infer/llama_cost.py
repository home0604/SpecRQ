"""L1: Llama-3.1-8B cost structure on a SYNTHETIC C1 checkpoint (real embed / norms / lm_head, random C1 linears with
Hadamard signs; NestModel.synthetic). Timing only -- codebook values do not affect kernel time; no quality numbers.

Measures (bf16, deterministic split-K, CUDA graphs):
  AR step k_t=5 and k in {1,2,3} (= draft step cost), at prompt lengths P (context dependence)
  forward cost at M in {2,3,5,8} (target k5; draft M=2 catch-up for each kd), lm_head GEMV, prefill
  spec rounds for gamma in {1,2,4,7} x kd in {1,2,3,5}: draft round / verify / overhead per round
  (acceptance of a random model is meaningless: tau is stored only for kd=5, the draft==target sanity bound)
Model: per round time R(g, kd) = t_draft_round + t_verify + t_overhead (measured); speedup(tau) = tau * T_AR / R;
required tau* = R / T_AR; with i.i.d. acceptance alpha, tau = (1 - alpha^(g+1)) / (1 - alpha) -> required alpha*.
tau / alpha are ASSUMED variables in this model, not measurements.
Usage: python -m specrq.infer.llama_cost --run NAME
"""
from __future__ import annotations

import argparse
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from specrq.infer.decode import ARDecoder, SpecDecoder, _Graph
from specrq.infer.model import NestModel

REPO = Path(__file__).resolve().parents[2]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is KB on Linux


def graph_us(fn, iters=10, reps=5):
    g = _Graph(lambda: [fn() for _ in range(iters)])
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    del g
    return statistics.median(ts)


def req_alpha(tau, g):
    """Smallest i.i.d. acceptance alpha with (1 - a^(g+1)) / (1 - a) >= tau (None if tau > g + 1)."""
    if tau > g + 1:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t = (1 - mid ** (g + 1)) / (1 - mid) if mid < 1 else g + 1
        lo, hi = (mid, hi) if t < tau else (lo, mid)
    return hi


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--prompts", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4, 7])
    ap.add_argument("--kds", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    rss0 = rss_gb()
    m = NestModel.synthetic(a.model)
    torch.cuda.synchronize()
    load = {"load_s": time.time() - t0, "peak_rss_gb_before": rss0, "peak_rss_gb_after_load": rss_gb(),
            "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "linear_bytes_per_token": {f"k{k}": m.linear_bytes(k) for k in range(1, 6)},
            "lm_head_bytes": m.lm_head.numel() * 2}
    print("[load]", load, flush=True)
    V = m.cfg.vocab_size
    gen = torch.Generator(device="cuda").manual_seed(0)
    res = {"load": load, "ar": [], "prefill": [], "forward_M": [], "lm_head_us": {}, "spec": []}
    Pmax = max(a.prompts)
    T = Pmax + a.n_new + max(a.gammas) + 8
    prompts = {P: torch.randint(0, V, (P,), device="cuda", generator=gen) for P in a.prompts}

    # AR step at k in {5, 1, 2, 3} and both prompt lengths
    for k in (5, 1, 2, 3):
        ar = ARDecoder(m, T, k=k)
        ar.generate(prompts[a.prompts[0]], 8)
        for P in a.prompts:
            st = [ar.generate(prompts[P], a.n_new)[1] for _ in range(a.repeats)]
            step = [1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st]
            res["ar"].append({"k": k, "P": P, "step_us": statistics.median(step), "step_us_range": [min(step), max(step)],
                              "tokens_per_s": statistics.median(s["tokens_per_s"] for s in st)})
            if k == 5:
                res["prefill"].append({"P": P, "k": 5, "prefill_s": statistics.median(s["t_prefill_s"] for s in st)})
            print("[ar]", res["ar"][-1], flush=True)
        del ar
        torch.cuda.empty_cache()

    # forward cost vs M (context = first prompt length)
    cache = m.new_cache(T)
    m.prefill(prompts[a.prompts[0]], cache, 5)
    pos = torch.tensor(a.prompts[0], device="cuda")
    toks = torch.randint(0, V, (16,), device="cuda", generator=gen)
    for k in (5, 3, 2, 1):
        for M in ((1, 2, 3, 5, 8) if k == 5 else (1, 2)):
            us = graph_us(lambda: m.forward(toks[:M], pos, cache, k, last_only=(M == 1 or k != 5)))
            res["forward_M"].append({"k": k, "M": M, "P": a.prompts[0], "us": us})
            print("[fwd]", res["forward_M"][-1], flush=True)
    del cache
    h = {M: torch.randn(M, m.cfg.hidden_size, device="cuda", dtype=m.dtype) for M in (1, 2, 3, 5, 8)}
    for M in h:
        res["lm_head_us"][f"M{M}"] = graph_us(lambda: F.linear(h[M], m.lm_head))
    print("[lm_head]", res["lm_head_us"], flush=True)

    # spec rounds
    P = a.prompts[0]
    for g in a.gammas:
        for kd in a.kds:
            sd = SpecDecoder(m, T, g, kd)
            sd.generate(prompts[P], 8)
            st = [sd.generate(prompts[P], a.n_new)[1] for _ in range(a.repeats)]
            r = {"gamma": g, "kd": kd, "P": P}
            for key in ("t_draft_round_ms", "t_verify_ms", "t_overhead_ms", "tokens_per_s", "t_prefill_s"):
                r[key] = statistics.median(s[key] for s in st)
            r["tau"] = statistics.median(s["tau"] for s in st) if kd == 5 else None  # random-model acceptance: n/a
            rows = [s["rounds"] for s in st]
            r["rounds"] = statistics.median(rows)
            res["spec"].append(r)
            print("[spec]", r, flush=True)
            del sd
            torch.cuda.empty_cache()

    # model: required tau for kd < 5 (measured costs, assumed tau)
    T_ar = next(x["step_us"] for x in res["ar"] if x["k"] == 5 and x["P"] == P) / 1e3  # ms
    fwd = {(x["k"], x["M"]): x["us"] / 1e3 for x in res["forward_M"]}
    model = []
    for r in res["spec"]:
        if r["kd"] == 5:
            continue
        g, kd = r["gamma"], r["kd"]
        R_meas = r["t_draft_round_ms"] + r["t_verify_ms"] + r["t_overhead_ms"]
        # catch-up bound: every round starts with an M=2 draft step instead of M=1
        R_catch = R_meas + (fwd[(kd, 2)] - fwd[(kd, 1)])
        tau_req = R_meas / T_ar
        model.append({"gamma": g, "kd": kd, "T_ar_ms": T_ar, "round_ms": R_meas, "round_ms_catchup_every_round": R_catch,
                      "tau_required": tau_req, "tau_required_catchup": R_catch / T_ar,
                      "alpha_required_iid": req_alpha(tau_req, g), "alpha_required_iid_catchup": req_alpha(R_catch / T_ar, g),
                      "speedup_at_tau": {f"{t:.1f}": t * T_ar / R_meas for t in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
                                         if t <= g + 1},
                      "speedup_upper_bound_tau_gamma_plus_1": (g + 1) * T_ar / R_meas})
        print("[model]", model[-1], flush=True)
    res["model"] = model
    res["peak_rss_gb_end"] = rss_gb()
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": "SYNTHETIC checkpoint: real Llama-3.1-8B-Instruct embed/norm/lm_head, random C1 linears "
                       "(L=5, 8 bit, d=8) + random Hadamard signs (B=256); timing only, no quality",
            "model": a.model, "prompts": a.prompts, "n_new": a.n_new, "gammas": a.gammas, "kds": a.kds,
            "repeats": a.repeats, "dtype": "bf16", "deterministic_splitk": True, "decode_attn": "fp32 compiled",
            "T_max": T, "tau_note": "tau/alpha in `model` are assumed variables; only kd=5 tau is measured (sanity)"},
            metrics=res, model_dir=a.model))


if __name__ == "__main__":
    main()
