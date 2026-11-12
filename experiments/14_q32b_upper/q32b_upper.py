"""E4 upper bound: Qwen2.5-32B-Instruct, SYNTHETIC nested 2-bit VQ (d8 K256 L2, Hadamard B=256, qkv bias; real
embed / norms / untied lm_head from the HF safetensors, random codes / codebooks / scales / signs generated on the GPU)
on our runner (NestModel + nestkern C1 kernels, CUDA graphs, fused glue, fp32 decode attention). Timing only.

Target = k2 (2 bit), draft = prefix k1 (1 bit, target-copy codebook = same kernel cost as a TV draft codebook).
Mode "vq" (default):
  parts (D1 protocol: CUDA graph, warm-up excluded, --reps repeats -> median [min, max]):
    target AR step k2 (ARDecoder loop, n_new tokens) and forward graph M=1 k2
    draft step k1 with full-vocab lm_head and with the FR-Spec V_d lm_head (ARDecoder loop + forward graph)
    verify forward k2 M = gamma+1 (gamma 1..4), lm_head(+argmax) full / V_d
  spec loop (draft_kv shared, real decode loop wall time):
    (a) kd=2 (draft == target), natural acceptance, and kd=2 forced full acceptance (len = gamma)
    (b) kd=1 with forced acceptance (SpecDecoder force_accept): i.i.d. p in --ps and fixed a in 0..gamma
  Output tokens of forced runs are NOT the target greedy output (timing of the loop only).
Mode "skip": prefix (k1) + draft layer skip. For each skip pattern (--skips) x draft vocab (--grid-vds): draft step
  k1 with the skipped decoder layers removed (residual passes through; skipped layers neither compute nor write KV),
  timed as an ARDecoder loop and as an M=1 forward graph; spec loop with draft_kv shared and forced full acceptance
  (len = gamma) for each gamma -> round_ms, tau_breakeven = round_ms / T_AR. Target AR k2 is measured at the start and
  at the end of the job (same GPU) as the reference. Verify always runs all layers with k2, so the target KV of every
  layer at confirmed positions is written by verify exactly as in the no-skip loop.
Mode "marlin": same runner, Marlin FP16 x INT4 g128 random linears (+ random fp16 qkv bias), fp16 embed / lm_head;
  AR step + forward M=1..5 (verify-sized) only.
Usage: python experiments/14_q32b_upper/q32b_upper.py --mode vq --run NAME
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from specrq.infer.decode import ARDecoder, SpecDecoder, _Graph  # noqa: E402
from specrq.infer.llama_cost import req_alpha  # noqa: E402
from specrq.infer.model import MarlinLinear, NestModel  # noqa: E402
from specrq.nestkern.gpucheck import cotenants  # noqa: E402


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def graph_us(fn, iters=20, reps=5):
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
    return med(ts)


def gpu_util():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


class _MarlinB(MarlinLinear):
    def __call__(self, x, k=None, use_draft=False, rotated=False):
        y = MarlinLinear.__call__(self, x, k, use_draft, rotated)
        return y if self.bias is None else y + self.bias


def ar_rows(m, prompt, T, k, vd, n_new, reps):
    ar = ARDecoder(m, T, k=k, vd=vd)
    ar.generate(prompt, 16)  # warm-up
    st = [ar.generate(prompt, n_new)[1] for _ in range(reps)]
    del ar
    torch.cuda.empty_cache()
    step = [1e6 * s["t_decode_s"] / s["n_decode_tokens"] for s in st]
    return {"k": k, "vd": vd, "step_us": med(step), "tokens_per_s": med([s["tokens_per_s"] for s in st]),
            "prefill_s": med([s["t_prefill_s"] for s in st])}


def spec_row(m, prompt, T, g, kd, vd, n_new, reps, force, tag):
    sd = SpecDecoder(m, T, g, kd, vd=vd, draft_kv="shared", force_accept=force)
    sd.generate(prompt, 16)  # warm-up
    st = [sd.generate(prompt, n_new)[1] for _ in range(reps)]
    del sd
    torch.cuda.empty_cache()
    r = {"tag": tag, "gamma": g, "kd": kd, "vd": vd, "force": list(force) if force else None, "draft_kv": "shared"}
    for key in ("tokens_per_s", "tau", "accept_rate", "t_draft_round_ms", "t_verify_ms", "t_overhead_ms", "rounds",
                "t_decode_s"):
        r[key] = med([s[key] for s in st])
    r["round_ms"] = med([1e3 * s["t_decode_s"] / s["rounds"] for s in st])
    return r


class SkipView:
    """Proxy over NestModel: forward calls with a draft prefix k < L run only the kept layers (layer skip); target
    calls (k None or L) run all layers. Everything else is delegated."""

    def __init__(self, m, skip):
        self._m = m
        self._ids = [i for i in range(m.n_layers) if i not in set(skip)] if skip else None

    def __getattr__(self, name):
        return getattr(self._m, name)

    def forward(self, tokens, pos, cache, k=None, use_draft=False, **kw):
        ids = self._ids if (k is not None and k < self._m.L) else None
        return self._m.forward(tokens, pos, cache, k, use_draft, layer_ids=ids, **kw)


def even_idx(n, lo, hi):
    """n layer indices spread evenly over [lo, hi)."""
    span = hi - lo
    return sorted({lo + int((i + 0.5) * span / n) for i in range(n)})


def skip_patterns(nL, names):
    pats = {}
    for nm in names:
        if nm == "none":
            pats[nm] = []
        elif nm.startswith("back"):  # last n layers (contiguous)
            n = int(nm[4:])
            pats[nm] = list(range(nL - n, nL))
        elif nm.startswith("mid"):  # contiguous middle block, first/last (nL-n)/2 kept
            n = int(nm[3:])
            lo = (nL - n) // 2
            pats[nm] = list(range(lo, lo + n))
        elif nm.startswith("even") and "_keep" in nm:  # n evenly spaced inside [keep, nL-keep)
            n, keep = nm[4:].split("_keep")
            pats[nm] = even_idx(int(n), int(keep), nL - int(keep))
        elif nm.startswith("even"):  # n evenly spaced over all layers (n = nL/2 -> every 2nd)
            pats[nm] = even_idx(int(nm[4:]), 0, nL)
        else:
            raise ValueError(nm)
        assert len(pats[nm]) == len(set(pats[nm])) and all(0 <= i < nL for i in pats[nm]), nm
    return pats


def main_skip(a, m, res, prompt, T, gen):
    V = m.lm_head.shape[0]
    pats = skip_patterns(m.n_layers, a.skips)
    res["skip_patterns"] = pats
    res["ar"].append(ar_rows(m, prompt, T, None, None, a.n_new, a.reps))
    print("[ar target start]", res["ar"][-1], flush=True)
    toks = torch.randint(0, m.cfg.vocab_size, (8,), device="cuda", generator=gen)
    pos = torch.tensor(a.P, device="cuda")
    rows = []
    for nm, sk in pats.items():
        sv = SkipView(m, sk)
        for vdg in a.grid_vds:
            vd = vdg or None
            r = {"skip": nm, "n_skip": len(sk), "n_run": m.n_layers - len(sk), "vd": vd or V}
            r["ar_draft"] = ar_rows(sv, prompt, T, 1, vd, a.n_new, a.reps)
            cache = m.new_cache(T)
            m.prefill(prompt, cache, m.L)
            r["fwd_M1_us"] = graph_us(lambda: m.to_token_ids(sv.forward(toks[:1], pos, cache, 1, last_only=True,
                                                                         vd=vd).argmax(-1), vd))
            del cache
            torch.cuda.empty_cache()
            r["spec"] = []
            for g in a.gammas:
                s = spec_row(sv, prompt, T, g, 1, vd, a.n_new, a.reps, ("len", g), f"skip_{nm}_len{g}")
                r["spec"].append(s)
            rows.append(r)
            print(f"[skip {nm} n={len(sk)} vd={vd}] draft step {r['ar_draft']['step_us']['median']:.0f} us, fwd "
                  f"{r['fwd_M1_us']['median']:.0f} us; rounds " +
                  " ".join(f"g{s['gamma']}={s['round_ms']['median']:.2f}ms(d{s['t_draft_round_ms']['median']:.2f}/"
                           f"v{s['t_verify_ms']['median']:.2f})" for s in r["spec"]), flush=True)
    res["skip"] = rows
    res["ar"].append(ar_rows(m, prompt, T, None, None, a.n_new, a.reps))
    print("[ar target end]", res["ar"][-1], flush=True)
    T0, T1 = (res["ar"][i]["step_us"]["median"] / 1e3 for i in (0, -1))
    T_ar = 0.5 * (T0 + T1)
    summ = []
    for r in rows:
        d = r["ar_draft"]["step_us"]["median"] / 1e3
        row = {"skip": r["skip"], "n_skip": r["n_skip"], "vd": r["vd"], "T_ar_ms": T_ar,
               "T_draft_step_ms": d, "c": d / T_ar,
               "c_range": [r["ar_draft"]["step_us"]["min"] / 1e3 / T_ar, r["ar_draft"]["step_us"]["max"] / 1e3 / T_ar],
               "c_fwd_graph": r["fwd_M1_us"]["median"] / 1e3 / T_ar, "gamma": {}}
        for s in r["spec"]:
            g, R = s["gamma"], s["round_ms"]["median"]
            row["gamma"][g] = {"round_ms": R, "round_ms_range": [s["round_ms"]["min"], s["round_ms"]["max"]],
                               "draft_ms": s["t_draft_round_ms"]["median"], "verify_ms": s["t_verify_ms"]["median"],
                               "overhead_ms": s["t_overhead_ms"]["median"],
                               "c_in_loop": s["t_draft_round_ms"]["median"] / g / T_ar,
                               "tau_breakeven": R / T_ar, "alpha_breakeven_iid": req_alpha(R / T_ar, g),
                               "speedup_at_full_accept": (g + 1) * T_ar / R}
        summ.append(row)
        print("[summary]", row, flush=True)
    res["summary"] = summ
    res["T_ar_start_end_ms"] = [T0, T1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_32b_instruct")
    ap.add_argument("--mode", choices=["vq", "marlin", "skip"], default="vq")
    ap.add_argument("--skips", nargs="+", default=["none", "back16", "back24", "back32", "even16", "even24", "even32",
                                                   "mid32", "even32_keep4"],
                    help="mode skip: draft layer-skip patterns (none, back<n>, mid<n>, even<n>, even<n>_keep<m>)")
    ap.add_argument("--P", type=int, default=128)
    ap.add_argument("--n-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--grid-vds", type=int, nargs="+", default=[32768, 0], help="draft vocab for the p-grid (0=full)")
    ap.add_argument("--quick", action="store_true", help="smoke: 1 gamma, 1 p, small n_new")
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    if a.quick:
        a.gammas, a.ps, a.n_new, a.grid_vds = [2], [0.7], 64, [a.vd]
        if a.mode == "skip":
            a.skips = ["none", "even32"]
    co0, util0 = cotenants() or "none", gpu_util()
    print(f"[q32b] co-tenants at start: {co0}; util {util0}", flush=True)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if a.mode in ("vq", "skip"):
        m = NestModel.synthetic(a.model, L=2, hadamard=True, verbose=True)
    else:
        m = NestModel.synthetic(a.model, dtype=torch.float16, linear_impl="marlin", verbose=True)
        gen = torch.Generator(device="cuda").manual_seed(1)
        for lay in m.layers:
            for n in ("q_proj", "k_proj", "v_proj"):
                lay[n].__class__ = _MarlinB
                lay[n].bias = (torch.randn(lay[n].N, device="cuda", generator=gen) * 0.01).half()
            for n in ("o_proj", "gate_proj", "up_proj", "down_proj"):
                lay[n].__class__ = _MarlinB
    torch.cuda.synchronize()
    V = m.lm_head.shape[0]
    load = {"load_s": time.time() - t0, "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1e9,
            "lm_head_bytes": m.lm_head.numel() * m.lm_head.element_size(),
            "embed_bytes": m.embed.numel() * m.embed.element_size(),
            "tied_lm_head": m.lm_head.data_ptr() == m.embed.data_ptr(),
            "linear_bytes_per_token": {f"k{k}": m.linear_bytes(k) for k in range(1, m.L + 1)}}
    if a.mode in ("vq", "skip"):
        from specrq.infer.frspec import coverage, default_freq_dir, load_subset
        fd = default_freq_dir(m.cfg)  # Qwen2.5-0.5B pile-val table; tokenizer files identical to 32B (md5 checked)
        ids = load_subset(fd, a.vd, None)
        m.set_draft_vocab(ids)
        load.update({"freq_dir": fd, "vd": a.vd, "vd_coverage_pileval": coverage(fd, ids),
                     "draft_vocab_bytes": m.draft_vocab_bytes()})
    print("[load]", load, flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    prompt = torch.randint(0, m.cfg.vocab_size, (a.P,), device="cuda", generator=gen)
    T = a.P + a.n_new + max(a.gammas) + 24
    res = {"load": load, "ar": [], "fwd": [], "lm_head": {}, "spec": []}
    kt = m.L
    if a.mode == "skip":
        main_skip(a, m, res, prompt, T, gen)
        res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        co1, util1 = cotenants() or "none", gpu_util()
        print(f"[q32b] co-tenants at end: {co1}; util {util1}; peak mem {res['peak_mem_gib']:.2f} GiB", flush=True)
        if a.run:
            from tools.results import save_result
            print("saved", save_result("E4", a.run, config={
                "setting": ("SYNTHETIC Qwen2.5-32B-Instruct: real embed/norms/untied lm_head, random nested-VQ C1-kernel "
                            "linears d8 K256 L2 + random Hadamard signs B=256 + random qkv bias; draft = prefix k1 + "
                            "decoder layer skip (skipped layers: residual pass-through, no KV write in the draft); "
                            "verify = k2 all layers; timing only, no quality; forced full acceptance (len=gamma)"),
                "model": a.model, "mode": a.mode, "d": 8, "K": 256, "L": m.L, "kt": kt, "kd": 1, "P": a.P,
                "n_new": a.n_new, "repeats": a.reps, "gammas": a.gammas, "grid_vds": a.grid_vds, "skips": a.skips,
                "draft_kv": "shared", "dtype": str(m.dtype), "deterministic_splitk": True,
                "decode_attn": "fp32 compiled", "T_max": T,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "co_tenant_start": co0, "co_tenant_end": co1, "gpu_util_start": util0, "gpu_util_end": util1,
                "T_ar_definition": "mean of target AR k2 step medians measured at job start and end",
                "force_accept_note": "SpecDecoder(force_accept=('len', gamma)): every round accepts gamma drafts; "
                                     "draft/verify graphs and loop are the real ones; output is not target-greedy"},
                metrics=res, model_dir=a.model))
        return

    # ---------------- parts
    res["ar"].append(ar_rows(m, prompt, T, None, None, a.n_new, a.reps))
    print("[ar target]", res["ar"][-1], flush=True)
    cache = m.new_cache(T)
    m.prefill(prompt, cache, kt)
    pos = torch.tensor(a.P, device="cuda")
    toks = torch.randint(0, m.cfg.vocab_size, (8,), device="cuda", generator=gen)
    for M in (1, 2, 3, 4, 5):
        r = {"k": kt, "vd": None, "M": M,
             "us": graph_us(lambda: m.forward(toks[:M], pos, cache, kt, last_only=(M == 1)).argmax(-1))}
        res["fwd"].append(r)
        print("[fwd]", r, flush=True)
    hid = m.cfg.hidden_size
    for M in (1, 2, 3, 4, 5):
        h = torch.randn(M, hid, device="cuda", dtype=m.dtype)
        res["lm_head"][f"full_M{M}"] = graph_us(lambda: F.linear(h, m.lm_head).argmax(-1))
    if a.mode == "vq":
        h = torch.randn(1, hid, device="cuda", dtype=m.dtype)
        w = m.dvocab[a.vd][1]
        res["lm_head"][f"vd{a.vd}_M1"] = graph_us(lambda: m.to_token_ids(F.linear(h, w).argmax(-1), a.vd))
        for vd in (None, a.vd):
            res["fwd"].append({"k": 1, "vd": vd, "M": 1,
                               "us": graph_us(lambda: m.to_token_ids(m.forward(toks[:1], pos, cache, 1, last_only=True,
                                                                               vd=vd).argmax(-1), vd))})
            print("[fwd]", res["fwd"][-1], flush=True)
    print("[lm_head]", res["lm_head"], flush=True)
    del cache
    torch.cuda.empty_cache()
    if a.mode == "vq":
        for vd in (None, a.vd):
            res["ar"].append(ar_rows(m, prompt, T, 1, vd, a.n_new, a.reps))
            print("[ar draft k1]", res["ar"][-1], flush=True)

    # ---------------- spec loop
    if a.mode == "vq":
        for g in a.gammas:
            runs = [(2, None, None, "a_kd2_natural"), (2, None, ("len", g), "a_kd2_forced_full")]
            for vdg in a.grid_vds:
                vd = vdg or None
                runs += [(1, vd, ("len", acc), f"b_kd1_len{acc}") for acc in sorted({0, g})]
                runs += [(1, vd, ("p", p, 1234), f"b_kd1_p{p}") for p in a.ps]
            for kd, vd, force, tag in runs:
                r = spec_row(m, prompt, T, g, kd, vd, a.n_new, a.reps, force, tag)
                res["spec"].append(r)
                print(f"[spec g={g} kd={kd} vd={vd} {tag}] {r['tokens_per_s']['median']:.1f} tok/s tau="
                      f"{r['tau']['median']:.2f} round={r['round_ms']['median']:.2f}ms draft="
                      f"{r['t_draft_round_ms']['median']:.2f} verify={r['t_verify_ms']['median']:.2f} ovh="
                      f"{r['t_overhead_ms']['median']:.3f}", flush=True)

        # ---------------- summary: measured tok/s vs AR; tau* from the measured round time
        T_ar = res["ar"][0]["step_us"]["median"] / 1e3
        ar_tps = res["ar"][0]["tokens_per_s"]["median"]
        summ = []
        for vdg in a.grid_vds:
            vd = vdg or None
            for g in a.gammas:
                sel = [r for r in res["spec"] if r["gamma"] == g and r["kd"] == 1 and r["vd"] == vd]
                Rs = [r["round_ms"]["median"] for r in sel]
                R = statistics.median(Rs)
                tau_star = R / T_ar
                row = {"gamma": g, "vd": vd or V, "round_ms_median_over_configs": R, "round_ms_range": [min(Rs), max(Rs)],
                       "T_ar_ms": T_ar, "tau_breakeven": tau_star, "alpha_breakeven_iid": req_alpha(tau_star, g),
                       "tokens_per_s_per_tau": 1e3 / R, "upper_bound_tau_g1": (g + 1) * T_ar / R, "grid": {}}
                for r in sel:
                    if r["force"][0] == "p":
                        p = r["force"][1]
                        row["grid"][f"p{p}"] = {"tau_expected": (1 - p ** (g + 1)) / (1 - p),
                                                "tau_measured": r["tau"]["median"],
                                                "tokens_per_s": r["tokens_per_s"]["median"],
                                                "tokens_per_s_range": [r["tokens_per_s"]["min"], r["tokens_per_s"]["max"]],
                                                "speedup_vs_ar": r["tokens_per_s"]["median"] / ar_tps}
                summ.append(row)
                print("[summary]", row, flush=True)
        res["summary"] = summ
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    co1, util1 = cotenants() or "none", gpu_util()
    print(f"[q32b] co-tenants at end: {co1}; util {util1}; peak mem {res['peak_mem_gib']:.2f} GiB", flush=True)
    if a.run:
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": ("SYNTHETIC Qwen2.5-32B-Instruct: real embed/norms/untied lm_head, random nested-VQ C1-kernel "
                        "linears d8 K256 L2 (2 bit target, k1 = 1 bit draft prefix) + random Hadamard signs B=256 + "
                        "random qkv bias; timing only, no quality; forced acceptance = assumed tau")
            if a.mode == "vq" else
            "SYNTHETIC Qwen2.5-32B-Instruct: Marlin FP16xINT4 g128 random linears + random fp16 qkv bias, fp16 embed/"
            "lm_head; AR only; timing only",
            "model": a.model, "mode": a.mode, "d": 8, "K": 256, "L": m.L if a.mode == "vq" else None, "kt": kt,
            "kd": 1, "P": a.P, "n_new": a.n_new, "repeats": a.reps, "gammas": a.gammas, "ps": a.ps, "vd": a.vd,
            "grid_vds": a.grid_vds, "draft_kv": "shared", "dtype": str(m.dtype), "deterministic_splitk": True,
            "decode_attn": "fp32 compiled", "T_max": T, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "co_tenant_start": co0, "co_tenant_end": co1, "gpu_util_start": util0, "gpu_util_end": util1,
            "force_accept_note": "SpecDecoder(force_accept=...): acceptance length forced on the host; draft/verify "
                                 "graphs and the loop are the real ones; output is not target-greedy"},
            metrics=res, model_dir=a.model))


if __name__ == "__main__":
    main()
