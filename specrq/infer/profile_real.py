"""D1: per-op cost breakdown of decode-sized NestModel forwards on the REAL Llama-3.1-8B C1 checkpoint.

Cases (one forward + argmax (+ FR-Spec id map) + output copy, exactly the body of a decode / verify graph):
  draft kd2-TV M=1 (V_d), draft kd2-TV M=2 catch-up, draft kd3-copy M=1, draft kd3-copy M=2, target AR k5 M=1,
  verify k5 M=2, M=3 (full vocab).
Three separate passes per case:
  (A) throughput: the step captured in a CUDA graph, replayed `--iters` times back-to-back, CUDA-event timed; `--reps`
      repeats -> median [min, max] per step. No profiler attached.
  (B) eager torch.profiler pass with record_function tags (wrappers on the 7 linears, add_rms_rot / act_rot glue,
      rope, attention). Each kernel is tagged with the innermost annotation around its launch (correlation id ->
      cudaLaunchKernel / cuLaunchKernel time); untagged kernels are classified by name and position
      (pre-layer setup / embedding, in-layer KV write / attention layout copies, post-layer lm_head / argmax / tail).
  (C) graph torch.profiler pass: kernels inside `--prof-replays` graph replays (CUPTI reports graph kernel nodes),
      aligned 1:1 by order with the eager kernel list (names checked). Per category: summed in-graph kernel time,
      idle gap before each kernel (attributed to "gap_in_graph"), and the replay span. gap_between_replays =
      (A) step time - median span (graph launch + host).
Context: the cache is prefilled with the chosen prompt (target k5 or the case's draft), pos = prompt_len + ctx_offset.
Attention runs over the full static cache T (masked), so its cost depends on T, not on pos; both are recorded.
Usage: python -m specrq.infer.profile_real --run NAME [--out-dir /data/lsh/profiles/d1]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import statistics
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import specrq.infer.model as MM
from specrq.infer.model import LINEARS, NestLinear, NestModel

REPO = Path(__file__).resolve().parents[2]
CAT_OF_TAG = {"lin.q_proj": "lin_q", "lin.k_proj": "lin_k", "lin.v_proj": "lin_v", "lin.o_proj": "lin_o",
              "lin.gate_proj": "lin_gate", "lin.up_proj": "lin_up", "lin.down_proj": "lin_down",
              "glue.add_rms_rot": "glue_add_rms_fwht", "glue.act_rot": "glue_act_fwht", "rope": "rope",
              "attention": "attention"}


class _TagLin:
    def __init__(self, lin, tag):
        self.lin, self.tag = lin, tag

    def __call__(self, *a, **kw):
        with record_function(self.tag):
            return self.lin(*a, **kw)

    def __getattr__(self, n):
        return getattr(self.lin, n)


def _tagfn(fn, tag):
    def w(*a, **kw):
        with record_function(tag):
            return fn(*a, **kw)
    return w


@contextmanager
def tagged(m):
    """Swap in record_function wrappers (eager profiling only; graphs are captured from the untagged model)."""
    saved = [(lay, {n.split(".")[1]: lay[n.split(".")[1]] for n in LINEARS}) for lay in m.layers]
    for lay in m.layers:
        for n in LINEARS:
            s = n.split(".")[1]
            lay[s] = _TagLin(lay[s], f"lin.{s}")
    o_ar, o_act, o_fused = MM.add_rms_rot, MM.act_rot, MM.fused
    MM.add_rms_rot = _tagfn(o_ar, "glue.add_rms_rot")
    MM.act_rot = _tagfn(o_act, "glue.act_rot")
    MM.fused = lambda name, uc: _tagfn(o_fused(name, uc), {"attn": "attention"}.get(name, name))
    try:
        yield
    finally:
        for lay, d in saved:
            lay.update(d)
        MM.add_rms_rot, MM.act_rot, MM.fused = o_ar, o_act, o_fused


def make_step(m, case, cache, tok, pos, out):
    k, ud, M, vd, last_only = case["k"], case["use_draft"], case["M"], case["vd"], case["last_only"]

    def step():
        logits = m.forward(tok[:M], pos, cache, k, ud, last_only=last_only, vd=vd)
        nxt = m.to_token_ids(logits.argmax(-1), vd)
        out[:nxt.numel()].copy_(nxt)
    return step


def kernels_from_trace(path):
    """Kernels (ordered by GPU start) with the innermost user annotation around their CPU launch."""
    ev = json.load(open(path))["traceEvents"]
    ann = sorted([(e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") == "user_annotation"
                  and e["name"] in CAT_OF_TAG])
    launch = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")
              and "correlation" in e.get("args", {})}
    ks = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
    starts = [a[0] for a in ann]
    out = []
    for e in ks:
        t = launch.get(e["args"].get("correlation"))
        tag = None
        if t is not None:
            i = bisect.bisect_right(starts, t) - 1
            best = None
            while i >= 0:  # innermost = latest-starting annotation that still contains t (annotations do not nest)
                if ann[i][0] <= t <= ann[i][1]:
                    best = ann[i]
                    break
                if ann[i][1] < t and t - ann[i][1] > 1e5:
                    break
                i -= 1
            tag = best[2] if best else None
        out.append({"name": e["name"], "ts": e["ts"], "dur": e["dur"], "tag": tag, "corr": e["args"].get("correlation"),
                    "memop": e.get("cat") != "kernel"})
    return out


def classify(kern):
    """Category per kernel: tag if tagged, else by name and position relative to the tagged layer kernels."""
    tagged_idx = [i for i, k in enumerate(kern) if k["tag"]]
    first, last = tagged_idx[0], tagged_idx[-1]
    for i, k in enumerate(kern):
        n = k["name"]
        if "fwht" in n:  # standalone FWHT (o_proj input rotation, launched inside the o_proj wrapper)
            k["cat"] = "fwht_standalone"
        elif k["tag"]:
            k["cat"] = CAT_OF_TAG[k["tag"]]
        elif i < first:
            k["cat"] = "embedding" if ("indexSelect" in n or "embedding" in n) else "setup_pos_mask_rope_idx"
        elif i > last:
            if "gemv" in n or "gemm" in n or "sm80" in n or "cutlass" in n or "ampere" in n:
                k["cat"] = "lm_head"
            elif "reduce_kernel" in n or "argmax" in n.lower():
                k["cat"] = "argmax"
            else:
                k["cat"] = "tail_idmap_copy"
        else:
            k["cat"] = "kv_write" if "index_copy" in n or "index_put" in n else "attn_layout_copy"
    return kern


def graph_kernels_per_replay(path, n_rep):
    ev = json.load(open(path))["traceEvents"]
    ks = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")], key=lambda e: e["ts"])
    assert len(ks) % n_rep == 0, (len(ks), n_rep)
    n = len(ks) // n_rep
    return [ks[i:i + n] for i in range(0, len(ks), n)]


def _key(name):
    return "memcpy" if ("memcpy" in name.lower()) else name[:80]


def align(eager, rk):
    """Category per graph kernel: matched (difflib on names) to the eager list; graph-only kernels (fill / memcpy
    nodes) -> setup before the first match, tail after the last, else graph_only_mid."""
    import difflib
    a, b = [_key(k["name"]) for k in eager], [_key(e["name"]) for e in rk]
    cats = [None] * len(rk)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1):
                cats[j1 + d] = eager[i1 + d]["cat"]
    matched = [j for j, c in enumerate(cats) if c]
    for j, c in enumerate(cats):
        if c is None:
            cats[j] = "setup_pos_mask_rope_idx" if j < matched[0] else ("tail_idmap_copy" if j > matched[-1]
                                                                          else "graph_only_mid")
    return cats, len(rk) - len(matched), len(eager) - len(matched)


def bytes_model(m, case, T_cache, ctx):
    """Bytes each category must read (lower bound on DRAM traffic) for one forward."""
    k, M, vd = case["k"], case["M"], case["vd"]
    lay = m.layers[0]
    nb = {}
    for n in LINEARS:
        s = n.split(".")[1]
        nb[f"lin_{ {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'o_proj': 'o', 'gate_proj': 'gate', 'up_proj': 'up', 'down_proj': 'down'}[s]}"] = \
            sum(L[s].nbytes(k) for L in m.layers)
    kv_layer_full = 2 * m.Hkv * T_cache * m.D * 2
    kv_layer_ctx = 2 * m.Hkv * ctx * m.D * 2
    nb["attention"] = m.n_layers * (kv_layer_full + 2 * M * m.H * m.D * 2)
    nb["attention_ctx_only"] = m.n_layers * (kv_layer_ctx + 2 * M * m.H * m.D * 2)
    nb["kv_write"] = m.n_layers * 2 * M * m.Hkv * m.D * 2
    V = m.lm_head.shape[0] if vd is None else vd
    nb["lm_head"] = V * m.cfg.hidden_size * 2
    nb["embedding"] = M * m.cfg.hidden_size * 2
    hid, inter = m.cfg.hidden_size, m.cfg.intermediate_size
    sg = sum(x.numel() * 4 for x in (lay["g_qkv"][1], lay["g_gu"][1], lay["g_down"][1]) if x is not None)
    nb["glue_signs_and_acts"] = m.n_layers * (sg + M * 2 * (2 * hid * 2 + 3 * hid + 2 * hid + 2 * inter + inter))
    return nb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/llama31_8b/c1_had_uniform_pile_fp32_ft5")
    ap.add_argument("--draft-sds", nargs="+", default=[
        "/data/lsh/results_accept/llama31_c1_uniform_kd2/train_B_bf16/draft_codebooks.pt",
        "/data/lsh/results_accept/llama31_c1_uniform_kd3/train_B_bf16/draft_codebooks.pt"])
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--T-max", type=int, default=204, help="cache length (D0 / v1 runs: 204 -> padded 208)")
    ap.add_argument("--prompt-idx", type=int, default=0, help="INSTRUCT_PROMPTS index (chat template)")
    ap.add_argument("--ctx-offset", type=int, default=64, help="pos = prompt_len + ctx_offset (mid-generation)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--prof-eager", type=int, default=3)
    ap.add_argument("--prof-replays", type=int, default=5)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--out-dir", default="/data/lsh/profiles/d1")
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    NestLinear.DETERMINISTIC = True
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from specrq.infer.frspec import default_freq_dir, load_subset
    from specrq.infer.run_decode import INSTRUCT_PROMPTS
    from specrq.nestkern.gpucheck import require_alone
    tok_ = AutoTokenizer.from_pretrained(a.model)
    m = NestModel(a.model, a.ckpt)
    for p in a.draft_sds:
        print(f"[profile_real] draft {p}: {m.load_draft(torch.load(p, map_location='cpu', weights_only=False))} linears")
    freq_dir = default_freq_dir(m.cfg)
    m.set_draft_vocab(load_subset(freq_dir, a.vd, m.lm_head.shape[0], chat=True))
    s = tok_.apply_chat_template([{"role": "user", "content": INSTRUCT_PROMPTS[a.prompt_idx]}], tokenize=False,
                                 add_generation_prompt=True)
    prompt = tok_(s, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda()
    P = prompt.numel()
    ctx = P + a.ctx_offset
    cases_all = [
        {"name": "draft_kd2_tv_M1", "k": 2, "use_draft": True, "M": 1, "vd": a.vd, "last_only": True},
        {"name": "draft_kd2_tv_M2_catchup", "k": 2, "use_draft": True, "M": 2, "vd": a.vd, "last_only": True},
        {"name": "draft_kd3_copy_M1", "k": 3, "use_draft": False, "M": 1, "vd": a.vd, "last_only": True},
        {"name": "draft_kd3_copy_M2_catchup", "k": 3, "use_draft": False, "M": 2, "vd": a.vd, "last_only": True},
        {"name": "target_ar_k5_M1", "k": 5, "use_draft": False, "M": 1, "vd": None, "last_only": True},
        {"name": "verify_k5_M2", "k": 5, "use_draft": False, "M": 2, "vd": None, "last_only": False},
        {"name": "verify_k5_M3", "k": 5, "use_draft": False, "M": 3, "vd": None, "last_only": False},
    ]
    cases = [c for c in cases_all if a.cases is None or c["name"] in a.cases]
    # lm_head reference bandwidth (full-vocab bf16 GEMV, M=1, graph)
    from specrq.infer.run_decode import lm_head_us
    lm_us = lm_head_us(m, 1)
    bw_ref = m.lm_head.numel() * 2 / (lm_us * 1e-6)
    print(f"[profile_real] prompt len {P}, pos {ctx}, lm_head M1 {lm_us:.1f} us -> {bw_ref / 1e9:.0f} GB/s", flush=True)
    results = []
    with torch.no_grad():
        for case in cases:
            cache = m.new_cache(a.T_max)
            T_cache = cache.k.shape[2]
            m.prefill(prompt, cache, case["k"], case["use_draft"], vd=case["vd"])
            tok = torch.full((4,), int(prompt[-1]), dtype=torch.long, device=m.device)
            pos = torch.tensor(ctx - (case["M"] - 1 if case["name"].endswith("catchup") else 0), device=m.device)
            out = torch.zeros(4, dtype=torch.long, device=m.device)
            step = make_step(m, case, cache, tok, pos, out)
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            torch.cuda.synchronize()
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            # (A) throughput, no profiler
            require_alone()
            ts = []
            for _ in range(a.reps):
                s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s0.record()
                for _ in range(a.iters):
                    g.replay()
                e0.record()
                torch.cuda.synchronize()
                ts.append(s0.elapsed_time(e0) * 1e3 / a.iters)
            t_graph = {"median": statistics.median(ts), "min": min(ts), "max": max(ts)}
            # (B) eager tagged profile
            eager_runs = []
            for r in range(a.prof_eager):
                with tagged(m):
                    step()
                    torch.cuda.synchronize()
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                        step()
                        torch.cuda.synchronize()
                fp = od / f"{a.run or 'dbg'}_{case['name']}_eager{r}.json"
                prof.export_chrome_trace(str(fp))
                eager_runs.append(classify(kernels_from_trace(fp)))
            ker = eager_runs[0]
            n_per = len(ker)
            assert all(len(x) == n_per for x in eager_runs), [len(x) for x in eager_runs]
            eager_cat = defaultdict(list)
            for run in eager_runs:
                c = defaultdict(float)
                for kk in run:
                    c[kk["cat"]] += kk["dur"]
                for kc, v in c.items():
                    eager_cat[kc].append(v)
            # (C) graph profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(a.prof_replays):
                    g.replay()
                torch.cuda.synchronize()
            fpg = od / f"{a.run or 'dbg'}_{case['name']}_graph.json"
            prof.export_chrome_trace(str(fpg))
            reps = graph_kernels_per_replay(fpg, a.prof_replays)
            al = [align(ker, rk) for rk in reps]
            mism = {"graph_only": al[0][1], "eager_only": al[0][2], "n_graph_kernels": len(reps[0])}
            gcats = al[0][0]
            assert all(x[0] == gcats for x in al)
            g_cat, g_gap, spans, ksum = defaultdict(list), defaultdict(list), [], []
            for rk in reps:
                c, gp = defaultdict(float), defaultdict(float)
                for i, e in enumerate(rk):
                    c[gcats[i]] += e["dur"]
                    if i:
                        gp[gcats[i]] += max(0.0, e["ts"] - (rk[i - 1]["ts"] + rk[i - 1]["dur"]))
                for kc in set(gcats):
                    g_cat[kc].append(c[kc])
                    g_gap[kc].append(gp[kc])
                spans.append(rk[-1]["ts"] + rk[-1]["dur"] - rk[0]["ts"])
                ksum.append(sum(e["dur"] for e in rk))
            span = statistics.median(spans)
            ks_med = statistics.median(ksum)
            nb = bytes_model(m, case, T_cache, ctx)
            n_kern = defaultdict(int)
            for kc in gcats:
                n_kern[kc] += 1
            rows = {}
            for kc in sorted(g_cat):
                t = statistics.median(g_cat[kc])
                b = nb.get(kc) if kc != "glue_add_rms_fwht" else None
                if kc.startswith("glue"):
                    b = None
                rows[kc] = {"us_graph": t, "us_gap_before": statistics.median(g_gap[kc]),
                            "us_eager_median": statistics.median(eager_cat[kc]) if kc in eager_cat else None,
                            "n_kernels": n_kern[kc],
                            "bytes": b, "us_floor_bw": (b / bw_ref * 1e6) if b else None}
            glue_b = nb["glue_signs_and_acts"]
            res = {"case": case, "prompt_len": P, "pos": int(pos), "T_cache": T_cache, "t_step_graph_us": t_graph,
                   "n_kernels": n_per, "graph_name_mismatches": mism, "graph_span_us": span,
                   "graph_kernel_sum_us": ks_med, "gap_in_graph_us": span - ks_med,
                   "gap_between_replays_us": t_graph["median"] - span, "categories": rows,
                   "glue_bytes_all": glue_b, "bytes_model": nb,
                   "eager_trace": str(od / f"{a.run or 'dbg'}_{case['name']}_eager0.json"), "graph_trace": str(fpg),
                   "kernel_list_graph": [{"name": e["name"][:90], "cat": gcats[i],
                                          "dur_graph_med": statistics.median(rk[i]["dur"] for rk in reps)}
                                         for i, e in enumerate(reps[0])]}
            results.append(res)
            print(f"=== {case['name']}: step(graph) {t_graph['median']:.1f} us [{t_graph['min']:.1f},{t_graph['max']:.1f}]"
                  f" kernels {n_per} align {mism} span {span:.1f} ksum {ks_med:.1f} gap_in {span - ks_med:.1f}"
                  f" gap_between {t_graph['median'] - span:.1f}", flush=True)
            for kc, r in sorted(rows.items(), key=lambda x: -x[1]["us_graph"]):
                fl = f"{r['us_floor_bw']:.1f}" if r["us_floor_bw"] else "-"
                eg = f"{r['us_eager_median']:.1f}" if r["us_eager_median"] is not None else "-"
                print(f"   {kc:28s} n={r['n_kernels']:4d} graph {r['us_graph']:8.1f} gap {r['us_gap_before']:7.1f}"
                      f" eager {eg:>8s} floor {fl}", flush=True)
            del g, cache
            torch.cuda.empty_cache()
    if a.run:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E4", a.run, config={
            "setting": "REAL Llama-3.1-8B C1 uniform ckpt, per-op breakdown of decode-sized forwards (D1)",
            "model": a.model, "ckpt": a.ckpt, "draft_sds": a.draft_sds, "vd": a.vd, "T_max": a.T_max,
            "prompt": s, "prompt_len": P, "pos": ctx, "ctx_offset": a.ctx_offset, "iters": a.iters, "reps": a.reps,
            "prof_eager": a.prof_eager, "prof_replays": a.prof_replays, "deterministic_splitk": True, "dtype": "bf16",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "out_dir": str(od),
            "method": "A: graph replay event timing (no profiler); B: eager torch.profiler with record_function tags; "
                      "C: torch.profiler over graph replays (kernel nodes), aligned to B by order",
            "bw_ref": "full lm_head bf16 GEMV M=1 bytes / time"},
            metrics={"lm_head_M1_us": lm_us, "bw_ref_GBps": bw_ref / 1e9, "cases": results}, model_dir=a.model))


if __name__ == "__main__":
    main()
