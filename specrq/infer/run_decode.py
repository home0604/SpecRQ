"""I3/I4: greedy AR baseline vs greedy self-speculative decoding on NestModel (CUDA graphs, static KV).

For each prompt: AR target (k_t) and spec (gamma, k_d) runs; output checked against AR with first_divergence.
Timing (Notion 03): warm-up run per decoder excluded, `repeats` timed runs per (prompt, config), median and
[min, max]; decode throughput = confirmed tokens / decode wall time (draft + verify + overhead), prefill separate.
Also reports the bf16 lm_head GEMV time (M=1 and M=gamma+1) and its share of an AR step.
FR-Spec: --vds V_d ... (0 = full vocab) runs every (gamma, kd) with the draft restricted to the V_d most frequent
tokens (frequency table --freq-dir, default specrq.infer.frspec.FREQ_DIRS); verify stays full-vocab.
Main metric: fixed length (EOS ignored, n_new tokens). --eos-stop adds a secondary pass with EOS stop (stop ids =
generation_config eos; a request ends at the first confirmed stop token or n_new) stored as rows_eos_stop.
Near-tie check (untimed, default on): every AR/spec divergence is re-run with logit recording; the AR (M=1) top gap
l[ar_tok] - l[spec_tok] <= --tie-ulps bf16 ulps -> near_tie, else counted as a bug candidate (WARNING in tie_check).
--chat adds INSTRUCT_PROMPTS (chat template) to the 3 text prompts. Draft vocab keeps chat specials (--no-chat-forced
= stop tokens only, as in results/E4/frspec_qwen05_*). Default V_d = frspec.DEFAULT_VD (32768).
Usage: python -m specrq.infer.run_decode --ckpt DIR [--gammas 1 2 4] [--kds 3 2] [--vds 0 32768] [--n-new 128] [--run NAME]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from specrq.infer.decode import ARDecoder, SpecDecoder, ar_token_logits, first_divergence, tie_report
from specrq.infer.model import NestLinear, NestModel

REPO = Path(__file__).resolve().parents[2]
PROMPTS = [
    "The history of natural language processing generally started in the 1950s, although work can be found from "
    "earlier periods. In 1950, Alan Turing published an article titled Computing Machinery and Intelligence which",
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods "
    "from carbon dioxide and water. In plants, photosynthesis generally involves the green pigment chlorophyll and",
    "def quicksort(arr):\n    \"\"\"Sort a list of integers using the quicksort algorithm.\"\"\"\n    if len(arr) <= 1:\n",
]
INSTRUCT_PROMPTS = [  # same as experiments/06_accept/draft_target_eval.py
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean at night.",
    "What are three common causes of high blood pressure?",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Give step-by-step instructions to make a cup of green tea.",
    "Why is the sky blue? Answer for a ten-year-old.",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "List five tips for writing clear technical documentation.",
]


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}


def lm_head_us(model, M, iters=50):
    h = torch.randn(M, model.cfg.hidden_size, device=model.device, dtype=model.dtype)
    for _ in range(3):
        F.linear(h, model.lm_head)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            F.linear(h, model.lm_head)
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / iters)
    return statistics.median(ts)


def aggregate(rows):
    """Per (gamma, kd, draft_src, vd) and prompt kind (+ 'all'): median over prompts of the per-prompt medians of
    tau, tokens/s, speedup vs AR and the per-round parts; AR tokens/s per kind."""
    out = []
    kinds = sorted({r["kind"] for r in rows}) + ["all"]
    for kind in kinds:
        ar = [r for r in rows if r["mode"] == "ar" and (kind == "all" or r["kind"] == kind)]
        if ar:
            out.append({"mode": "ar", "kind": kind, "n_prompts": len(ar),
                        "tokens_per_s": statistics.median(r["tokens_per_s"]["median"] for r in ar),
                        "step_us": statistics.median(r["step_us"]["median"] for r in ar)})
        keys = sorted({(r["gamma"], r["kd"], r["draft_src"], r["vd"]) for r in rows if r["mode"] == "spec"})
        for g, kd, src, vd in keys:
            sel = [r for r in rows if r["mode"] == "spec" and (r["gamma"], r["kd"], r["draft_src"], r["vd"]) ==
                   (g, kd, src, vd) and (kind == "all" or r["kind"] == kind)]
            if not sel:
                continue
            d = {"mode": "spec", "kind": kind, "gamma": g, "kd": kd, "draft_src": src, "vd": vd, "n_prompts": len(sel)}
            for key in ("tau", "tokens_per_s", "t_draft_round_ms", "t_verify_ms", "t_overhead_ms", "accept_rate"):
                d[key] = statistics.median(r[key]["median"] for r in sel)
            d["speedup_vs_ar_median"] = statistics.median(r["speedup_vs_ar"] for r in sel)
            d["speedup_vs_ar_min"] = min(r["speedup_vs_ar"] for r in sel)
            d["speedup_vs_ar_max"] = max(r["speedup_vs_ar"] for r in sel)
            d["n_greedy_match_all_repeats"] = sum(all(r["greedy_match"]) for r in sel)
            out.append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/smoke/qwen05_c1_35w_tiny")
    ap.add_argument("--draft-sd", default=None, help="DbD draft_state_dict file (use_draft=True)")
    ap.add_argument("--draft-sds", nargs="+", default=[], help="several draft_state_dict files (e.g. kd2 and kd3 "
                    "TV sets); each installs its k_d set, selected per spec config with --specs kd:tv")
    ap.add_argument("--specs", nargs="+", default=None, help="spec configs 'kd:copy' (target codebook prefix) or "
                    "'kd:tv' (installed draft set for kd); overrides --kds")
    ap.add_argument("--wt2", type=int, default=0, help="add N wikitext2 prompts (64 tokens at (100+i)*512 of the "
                    "test split, as experiments/06_accept/draft_target_eval.py)")
    ap.add_argument("--no-text-prompts", action="store_true", help="drop the 3 built-in text prompts")
    ap.add_argument("--kt", type=int, default=None)
    ap.add_argument("--kds", type=int, nargs="+", default=[3, 2])
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--vds", type=int, nargs="+", default=[32768], help="FR-Spec draft vocab sizes (0 = full)")
    ap.add_argument("--eos-stop", action="store_true",
                    help="also measure with EOS stop (secondary; main = fixed n_new tokens, EOS ignored)")
    ap.add_argument("--no-eos-stop", action="store_true", help="no-op (fixed length is the default)")
    ap.add_argument("--no-tie-check", action="store_true", help="skip the untimed near-tie diagnosis")
    ap.add_argument("--tie-ulps", type=float, default=2.0)
    ap.add_argument("--div-logits-dir", default=None, help="(tie check) save AR / verify logits top-k at each first "
                    "divergence (same prefix) to DIR/<run>_divlogits.pt and add the top-k to the divergence summary")
    ap.add_argument("--div-topk", type=int, default=8)
    ap.add_argument("--no-chat-forced", action="store_true", help="draft vocab forces stop tokens only")
    ap.add_argument("--chat", action="store_true", help="also run the chat-template INSTRUCT_PROMPTS")
    ap.add_argument("--freq-dir", default=None)
    ap.add_argument("--n-new", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--run", default=None)
    ap.add_argument("--nondet", action="store_true", help="fp32-atomic split-K (faster, not bitwise reproducible)")
    a = ap.parse_args(argv)
    NestLinear.DETERMINISTIC = not a.nondet
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = NestModel(a.model, a.ckpt)
    from specrq.nestkern.gpucheck import cotenants, require_alone
    cotenant_start = require_alone() if a.run else (cotenants() or "none")
    use_draft = False
    for path in ([a.draft_sd] if a.draft_sd else []) + list(a.draft_sds):
        n = m.load_draft(torch.load(path, map_location="cpu", weights_only=False))
        use_draft = n > 0
        print(f"[run_decode] installed draft codebooks from {path} on {n} linears")
    if a.specs:
        spec_list = []
        for sp in a.specs:
            kd_s, src = sp.split(":")
            assert src in ("copy", "tv"), sp
            spec_list.append((int(kd_s), src == "tv", src))
    else:
        spec_list = [(kd, use_draft, "tv" if use_draft else "copy") for kd in a.kds]
    from specrq.infer.frspec import coverage, default_freq_dir, load_subset
    freq_dir = a.freq_dir or default_freq_dir(m.cfg)
    vd_cov = {}
    for vd in a.vds:
        if vd:
            ids = load_subset(freq_dir, vd, m.lm_head.shape[0], chat=not a.no_chat_forced)
            m.set_draft_vocab(ids)
            vd_cov[vd] = coverage(freq_dir, ids)
    prompt_texts = [] if a.no_text_prompts else list(PROMPTS)
    prompts = [] if a.no_text_prompts else [tok(p, return_tensors="pt").input_ids[0].cuda() for p in PROMPTS]
    kinds = ["text"] * len(prompts)
    if a.chat:
        for p in INSTRUCT_PROMPTS:
            s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            prompt_texts.append(s)
            prompts.append(tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda())
            kinds.append("chat")
    if a.wt2:
        sys.path.insert(0, str(REPO / "specrq" / "dbd"))
        from src.datautils import get_loaders
        wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=512, eval_mode=True)
        for i in range(a.wt2):
            ids = wt[0, (100 + i) * 512:(100 + i) * 512 + 64]
            prompt_texts.append(tok.decode(ids))
            prompts.append(ids.cuda())
            kinds.append("wikitext2")
    from transformers import GenerationConfig
    e = GenerationConfig.from_pretrained(a.model).eos_token_id
    eos_ids = sorted(set(e if isinstance(e, list) else [e]))
    T = max(p.numel() for p in prompts) + a.n_new + max(a.gammas) + 8

    def measure(eos, tag):
        """Timed AR + spec rows (eos=None: fixed n_new tokens) and near-tie diagnostics of every divergence."""
        print(f"[run_decode] ===== {tag}: eos stop ids {eos}", flush=True)
        rows = []
        ar = ARDecoder(m, T, k=a.kt, eos=eos)
        ar.generate(prompts[0], 16)  # warm-up
        ar_ref = {}
        for pi, p in enumerate(prompts):
            runs = [ar.generate(p, a.n_new) for _ in range(a.repeats)]
            toks = runs[0][0]
            ar_ref[pi] = toks
            det = all(r[0] == toks for r in runs)
            st = [r[1] for r in runs]
            rows.append({"mode": "ar", "prompt": pi, "kind": kinds[pi], "kt": a.kt or m.L, "reproducible": det,
                         "tokens_per_s": med([s["tokens_per_s"] for s in st]),
                         "t_decode_s": med([s["t_decode_s"] for s in st]),
                         "t_prefill_s": med([s["t_prefill_s"] for s in st]),
                         "step_us": med([1e6 * s["t_decode_s"] / max(s["n_decode_tokens"], 1) for s in st]),
                         "n_tokens": len(toks), "eos_hit": runs[0][1].get("eos_hit")})
            print(f"[{tag} AR] prompt {pi}: {rows[-1]['tokens_per_s']['median']:.1f} tok/s, step "
                  f"{rows[-1]['step_us']['median']:.1f} us, reproducible={det}, n_tokens={len(toks)} "
                  f"eos_hit={rows[-1]['eos_hit']}", flush=True)
        del ar
        ar_tps = {r["prompt"]: r["tokens_per_s"]["median"] for r in rows}
        ar_diag, diag_out = None, {}
        div_saved = []

        def diagnose(pi, g, kd, vd, spec_toks, use_draft):
            nonlocal ar_diag
            if ar_diag is None:
                ar_diag = ARDecoder(m, T, k=a.kt, eos=eos, n_max=T, record_logits=True)
            if pi not in diag_out:
                diag_out[pi] = ar_diag.generate(prompts[pi], a.n_new)[0]
                diag_out[pi] = (diag_out[pi], [ar_token_logits(ar_diag, i).clone()
                                               for i in range(len(diag_out[pi]))])
            ar_toks, ar_logs = diag_out[pi]
            sdd = SpecDecoder(m, T, g, kd, a.kt, use_draft, vd=vd or None, eos=eos, record_logits=True)
            sp = sdd.generate(prompts[pi], a.n_new)[0]
            vl = sdd.rec_logits
            del sdd
            i = first_divergence(sp, ar_toks)
            r = {"diag_ar_equals_timed_ar": ar_toks == ar_ref[pi], "diag_spec_equals_timed_spec": sp == spec_toks}
            if i is None or i >= min(len(sp), len(ar_toks)):  # no divergence (or only a length difference)
                r.update({"index": i, "near_tie": None})
                return r
            r.update(tie_report(i, ar_toks[i], sp[i], ar_logs[i], vl[i] if vl is not None and i < len(vl) else None,
                                a.tie_ulps))
            if a.div_logits_dir:  # AR row (M=1 path) and verify row that produced token i; prefix tokens[:i] equal
                tk = {"ar": ar_logs[i].float()}
                if vl is not None and i < len(vl):
                    tk["verify"] = vl[i].float()
                rec = {"prompt": pi, "gamma": g, "kd": kd, "vd": vd, "index": i, "ar_tok": int(ar_toks[i]),
                       "spec_tok": int(sp[i]), "prefix": ar_toks[:i]}
                for nm, lg in tk.items():
                    tv, ti = lg.topk(a.div_topk)
                    r[f"{nm}_topk_ids"], r[f"{nm}_topk_logits"] = ti.tolist(), tv.tolist()
                    rec[f"{nm}_topk_ids"], rec[f"{nm}_topk_logits"] = ti.cpu(), tv.cpu()
                    rec[f"{nm}_logit_of"] = {"ar_tok": float(lg[ar_toks[i]]), "spec_tok": float(lg[sp[i]])}
                div_saved.append(rec)
            return r

        for g in a.gammas:
            for kd, use_draft, src, vd in [(kd, ud, src, vd) for kd, ud, src in spec_list for vd in a.vds]:
                sd = SpecDecoder(m, T, g, kd, a.kt, use_draft, vd=vd or None, eos=eos)
                sd.generate(prompts[0], 16)  # warm-up
                for pi, p in enumerate(prompts):
                    runs = [sd.generate(p, a.n_new) for _ in range(a.repeats)]
                    st = [r[1] for r in runs]
                    fd = [first_divergence(r[0], ar_ref[pi]) for r in runs]
                    r = {"mode": "spec", "prompt": pi, "kind": kinds[pi], "gamma": g, "kd": kd, "draft_src": src,
                         "vd": vd or m.lm_head.shape[0], "kt": a.kt or m.L, "use_draft": use_draft,
                         "first_divergence": fd, "greedy_match": [f is None for f in fd],
                         "n_tokens": [len(x[0]) for x in runs], "eos_hit": [x[1].get("eos_hit") for x in runs],
                         "_toks": runs[0][0]}
                    for key in ("tokens_per_s", "t_decode_s", "t_prefill_s", "tau", "accept_rate", "t_draft_round_ms",
                                "t_verify_ms", "t_overhead_ms"):
                        r[key] = med([s[key] for s in st])
                    r["speedup_vs_ar"] = r["tokens_per_s"]["median"] / ar_tps[pi] if ar_tps[pi] else None
                    r["share"] = {k2: statistics.median(s[k2] for s in st) / statistics.median(s["t_decode_s"] for s in st)
                                  for k2 in ("t_draft_s", "t_verify_s", "t_overhead_s")}
                    rows.append(r)
                    print(f"[{tag} spec g={g} kd={kd}:{src} vd={vd}] prompt {pi} ({kinds[pi]}): {r['tokens_per_s']['median']:.1f} tok/s "
                          f"(x{r['speedup_vs_ar'] or 0:.2f} AR) n={r['n_tokens']} tau={r['tau']['median']:.2f} "
                          f"acc={r['accept_rate']['median']:.2f} draft {r['t_draft_round_ms']['median']:.2f}ms verify "
                          f"{r['t_verify_ms']['median']:.2f}ms ovh {r['t_overhead_ms']['median']:.2f}ms first_div={fd}",
                          flush=True)
                del sd
                torch.cuda.empty_cache()
                if not a.no_tie_check:  # untimed, after the timed runs of this config
                    for r in rows:
                        if r["mode"] == "spec" and r["gamma"] == g and r["kd"] == kd and r["draft_src"] == src and \
                                r["vd"] == (vd or m.lm_head.shape[0]) and any(f is not None for f in r["first_divergence"]):
                            r["divergence"] = diagnose(r["prompt"], g, kd, vd, r["_toks"], use_draft)
                            d = r["divergence"]
                            print(f"[{tag} tie] g={g} kd={kd} vd={vd} prompt {r['prompt']}: {d}", flush=True)
        del ar_diag
        torch.cuda.empty_cache()
        if a.div_logits_dir and div_saved:
            Path(a.div_logits_dir).mkdir(parents=True, exist_ok=True)
            fp = Path(a.div_logits_dir) / f"{a.run or 'run'}_{tag}_divlogits.pt"
            torch.save(div_saved, fp)
            print(f"[{tag} tie] saved {len(div_saved)} divergence logit records to {fp}", flush=True)
        for r in rows:
            r.pop("_toks", None)
        divs = [r["divergence"] for r in rows if "divergence" in r]
        bad = [dict(d, prompt=r["prompt"], gamma=r["gamma"], kd=r["kd"], vd=r["vd"]) for r in rows
               if "divergence" in r for d in [r["divergence"]] if d.get("near_tie") is False]
        summ = {"n_spec_rows": sum(r["mode"] == "spec" for r in rows), "n_diverged": len(divs),
                "n_near_tie": sum(d.get("near_tie") is True for d in divs), "n_bug_candidates": len(bad),
                "bug_candidates": bad, "tie_ulps": a.tie_ulps,
                "n_diag_ar_mismatch": sum(not d["diag_ar_equals_timed_ar"] for d in divs),
                "n_diag_spec_mismatch": sum(not d["diag_spec_equals_timed_spec"] for d in divs)}
        if bad:
            summ["WARNING"] = f"{len(bad)} AR/spec divergences with gap > {a.tie_ulps} bf16 ulp: bug candidates"
            print("[WARNING]", summ["WARNING"], flush=True)
        print(f"[{tag} tie summary] {summ}", flush=True)
        return rows, summ

    rows, tie = measure(None, "fixed")
    rows_eos, tie_eos = measure(eos_ids, "eos") if a.eos_stop else (None, None)
    lm = {f"M{M}": lm_head_us(m, M) for M in sorted({1, *[g + 1 for g in a.gammas]})}
    ar_step = statistics.median(r["step_us"]["median"] for r in rows if r["mode"] == "ar")
    print(f"lm_head bf16 GEMV us: {lm}; share of AR step {lm['M1'] / ar_step:.2%}")
    agg = aggregate(rows)
    for x in agg:
        print("[agg]", x, flush=True)
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={"co_tenant_start": cotenant_start,
            "co_tenant_end": cotenants() or "none", "specs": a.specs, "draft_sds": a.draft_sds, "wt2": a.wt2,
            "prompt_kinds": kinds,
            "setting": "greedy AR vs self-spec, NestModel CUDA graphs, batch 1", "model": a.model, "ckpt": a.ckpt,
            "draft_sd": a.draft_sd, "kt": a.kt or m.L, "kds": a.kds, "gammas": a.gammas, "n_new": a.n_new,
            "repeats": a.repeats, "prompts": prompt_texts,
            "main_metric": "fixed length n_new (EOS ignored)", "eos_stop_ids_secondary": eos_ids if a.eos_stop else None,
            "chat_forced_draft_vocab": not a.no_chat_forced, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"), "deterministic_splitk": not a.nondet, "dtype": "bf16", "T_max": T,
            "linear_bytes_per_token": {f"k{k}": m.linear_bytes(k) for k in range(1, m.L + 1)},
            "lm_head_bytes": m.lm_head.numel() * 2, "vds": a.vds, "freq_dir": freq_dir,
            "draft_vocab_bytes": m.draft_vocab_bytes(), "draft_vocab_pileval_coverage": vd_cov},
            metrics={"rows": rows, "aggregate": agg, "tie_check": tie, "rows_eos_stop": rows_eos, "tie_check_eos_stop": tie_eos,
                     "lm_head_us": lm, "lm_head_share_of_ar_step": lm["M1"] / ar_step},
            model_dir=a.model))


if __name__ == "__main__":
    main()
