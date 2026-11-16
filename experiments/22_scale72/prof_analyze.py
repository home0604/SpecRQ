"""Per-operation breakdown of the decode step from an nsys sqlite export of prof_synth.py.
Each profiled step is an NVTX range "step:<cfg>:<i>" (graph replay + device sync). Kernels are classified by name
and position (single stream, no overlap):
  setup      : kernels before the first add_rms_rot except the embedding gather (positions, mask, RoPE-table gather)
  embedding  : indexSelectSmallIndex (F.embedding)
  vq.<lin>   : nest_* kernels, in order q,k,v,o,gate,up,down per layer
  rot.o_in   : fwht_kernel (Hadamard of the o_proj input; the other rotations are fused, see below)
  norm.fused : add_rms_rot (residual add + RMSNorm + block FWHT of the next linear group's input), per layer x2
  act.fused  : act_rot (SiLU*mul + FWHT of the down_proj input)
  rope       : the first 2 triton kernels after v_proj (compiled rotate-half RoPE)
  kv_write   : index_copy kernels inside the layer (K and V cache writes)
  attn       : remaining triton kernels before fwht (compiled fp32 decode attention over the cache)
  final_norm : last add_rms_rot (after the last down_proj)
  lm_head    : gemv/gemm after the final norm;  argmax : ArgMax reduce;  vd_map : to_token_ids gather (FR-Spec)
  bookkeep   : index_copy / add / copy after argmax (out buffer, next token, pos, idx)
  bias       : bf16 add after q/k/v (Qwen2 qkv bias)
  other      : anything else
idle = (last kernel end - first kernel start) - sum(kernel time); memcpy nodes are counted separately.
Usage: python prof_analyze.py --sqlite X.sqlite --meta X_meta.json --name TAG [--run RESULT_RUN]
"""
import argparse
import collections
import json
import os
import sqlite3
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
LIN = ["q", "k", "v", "o", "gate", "up", "down"]


def classify(rows):
    """rows: [(start, end, short, demangled)] of one step, time ordered -> list of (category, dur_ns)."""
    out = []
    first_norm = next(i for i, r in enumerate(rows) if "add_rms_rot" in r[2])
    vq_idx = [i for i, r in enumerate(rows) if r[2].startswith("nest_")]
    last_vq = vq_idx[-1]
    nvq = 0
    after_v = False
    tri_after_v = 0
    after_head = False
    for i, (s, e, sh, dm) in enumerate(rows):
        d = e - s
        if i < first_norm:
            out.append(("embedding" if "indexSelect" in sh else "setup", d))
            continue
        if i > last_vq:
            if "add_rms_rot" in sh:
                out.append(("final_norm", d))
            elif "gemv" in dm or "gemm" in dm.lower() or "cutlass" in dm or sh in ("kernel",) or "sm80" in dm or "sm90" in dm:
                out.append(("lm_head", d))
                after_head = True
            elif "ArgMax" in dm:
                out.append(("argmax", d))
            elif "index_kernel_impl" in dm and after_head:
                out.append(("vd_map", d))
            else:
                out.append(("bookkeep", d))
            continue
        if sh.startswith("nest_"):
            lin = LIN[nvq % 7] if "reduce" not in sh else LIN[(nvq - 1) % 7]
            if "reduce" not in sh:
                nvq += 1
            out.append((f"vq.{lin}", d))
            after_v = lin == "v"
            if after_v:
                tri_after_v = 0
            continue
        if "CUDAFunctor_add<c10::BFloat16>" in dm:
            out.append(("bias", d))  # q/k/v bias add (Qwen2) after the VQ linear
        elif "add_rms_rot" in sh:
            out.append(("norm.fused", d))
        elif "act_rot" in sh:
            out.append(("act.fused", d))
        elif "fwht" in sh:
            out.append(("rot.o_in", d))
            after_v = False
        elif after_v and "triton" in sh:
            out.append(("rope" if tri_after_v < 2 else "attn", d))
            tri_after_v += 1
        elif after_v and "index_copy" in dm:
            out.append(("kv_write", d))
        elif after_v:
            out.append(("attn", d))
        else:
            out.append(("other", d))
    assert nvq % 7 == 0, nvq
    return out, nvq // 7


def lin_bytes(cfg, k):
    H, Hkv, hid, inter = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"], cfg["intermediate_size"]
    D = hid // H
    shp = {"q": (H * D, hid), "k": (Hkv * D, hid), "v": (Hkv * D, hid), "o": (hid, H * D), "gate": (inter, hid),
           "up": (inter, hid), "down": (hid, inter)}
    return {n: k * N * K // 8 + k * 256 * 8 * 2 + 2 * N for n, (N, K) in shp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    cfgj = json.load(open(os.path.join(meta["model"], "config.json")))
    c = sqlite3.connect(a.sqlite)
    steps = c.execute("select start,end,text from NVTX_EVENTS where text like 'step:%' order by start").fetchall()
    res = {}
    for cfgname in meta["cfg"]:
        per = collections.defaultdict(list)
        spans, ksums, nvtxs, memc, nl = [], [], [], [], None
        for (s0, s1, txt) in steps:
            if txt.split(":")[1] != cfgname:
                continue
            rows = c.execute("select k.start,k.end,s.value,d.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on "
                             "k.shortName=s.id join StringIds d on k.demangledName=d.id where k.start>=? and k.end<=? "
                             "order by k.start", (s0, s1)).fetchall()
            mc = c.execute("select coalesce(sum(end-start),0), count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? "
                           "and end<=?", (s0, s1)).fetchone()
            cl, nl = classify(rows)
            agg = collections.defaultdict(float)
            for cat, d in cl:
                agg[cat] += d / 1e3
            for cat in set(agg) | set(per):
                per[cat].append(agg.get(cat, 0.0))
            span = (rows[-1][1] - rows[0][0]) / 1e3
            spans.append(span)
            ksums.append(sum(d for _, d in cl) / 1e3)
            nvtxs.append((s1 - s0) / 1e3)
            memc.append(mc[0] / 1e3)
        md = {cat: statistics.median(v) for cat, v in per.items()}
        span, ksum = statistics.median(spans), statistics.median(ksums)
        wall = meta["cfg"][cfgname]["wall_step_us"]
        k = meta["cfg"][cfgname]["k"]
        vq = {cat: v for cat, v in md.items() if cat.startswith("vq.")}
        vq_sum = sum(vq.values())
        lb = lin_bytes(cfgj, k)
        tab = {}
        groups = {
            "VQ linears (sum)": vq_sum,
            "q/k/v bias add (Qwen2)": md.get("bias", 0),
            "rotation: separate FWHT (o_proj input)": md.get("rot.o_in", 0),
            "fused residual+RMSNorm+FWHT (2/layer)": md.get("norm.fused", 0),
            "fused SiLU*mul+FWHT": md.get("act.fused", 0),
            "RoPE": md.get("rope", 0), "KV cache write": md.get("kv_write", 0), "attention (fp32, compiled)": md.get("attn", 0),
            "step setup (pos/mask/RoPE tables)": md.get("setup", 0), "embedding": md.get("embedding", 0),
            "final norm": md.get("final_norm", 0), "lm_head": md.get("lm_head", 0), "argmax": md.get("argmax", 0),
            "FR-Spec id map": md.get("vd_map", 0), "bookkeeping (out/tok/pos)": md.get("bookkeep", 0),
            "other kernels": md.get("other", 0), "memcpy nodes": statistics.median(memc),
            "idle: wall - kernel sum (unprofiled wall)": wall - ksum,
            "[profiled] idle gaps inside traced step": span - ksum,
        }
        for g, v in groups.items():
            tab[g] = {"us": v, "pct_of_wall": 100 * v / wall}
        vq_detail = {}
        for n in LIN:
            t = md.get(f"vq.{n}", 0)
            bytes_ = lb[n] * nl
            vq_detail[n] = {"us": t, "pct_of_wall": 100 * t / wall, "bytes": bytes_, "GBps": bytes_ / (t * 1e-6) / 1e9 if t else None}
        vq_bytes = sum(lb.values()) * nl
        lm_bytes = meta["vd_bytes"] if meta["cfg"][cfgname]["vd"] else meta["lm_head_bytes"]
        r = {"k": k, "vd": meta["cfg"][cfgname]["vd"], "ctx": meta["cfg"][cfgname]["ctx"], "n_layers": nl,
             "wall_step_us": wall, "gpu_span_us": span, "kernel_sum_us": ksum, "table": tab, "vq_by_linear": vq_detail,
             "vq_GBps": vq_bytes / (vq_sum * 1e-6) / 1e9, "vq_bytes": vq_bytes,
             "lm_head_GBps": lm_bytes / (md.get("lm_head", 0) * 1e-6) / 1e9 if md.get("lm_head") else None,
             "fixed_us": wall - vq_sum, "fixed_pct": 100 * (wall - vq_sum) / wall, "n_steps": len(spans),
             "span_range": [min(spans), max(spans)]}
        res[cfgname] = r
        print(f"\n=== {a.name} {cfgname}: wall {wall:.0f} us, span {span:.0f}, kernel sum {ksum:.0f}, VQ {vq_sum:.0f} us "
              f"({100 * vq_sum / wall:.1f}%), fixed {wall - vq_sum:.0f} us ({r['fixed_pct']:.1f}%), VQ {r['vq_GBps']:.0f} GB/s, "
              f"lm_head {r['lm_head_GBps'] or 0:.0f} GB/s")
        for g, v in tab.items():
            print(f"  {g:42s} {v['us']:9.1f} us {v['pct_of_wall']:6.2f} %")
        print("  VQ by linear: " + ", ".join(f"{n} {v['us']:.0f}us ({v['GBps'] or 0:.0f} GB/s)" for n, v in vq_detail.items()))
    if a.run:
        from tools.results import save_result
        print(save_result("E4", a.run, config={
            "setting": "SYNTHETIC (all random weights) C1 nested-VQ d8 K256 model, batch 1 M=1 decode step, CUDA graph "
                       "(ARDecoder step), nsys --cuda-graph-trace=node; each profiled step = graph replay + device sync",
            "model_shapes": meta["model"], "P": meta["P"], "pre": meta["pre"], "steps": meta["steps"], "vd": meta["vd"],
            "cuda_visible_devices": meta["cuda_visible_devices"], "co_tenants_start": meta["co_tenants_start"],
            "co_tenants_end": meta["co_tenants_end"],
            "notes": "wall_step_us = pipelined graph replays without sync (median of 3 x steps) measured in the same process "
                     "before the profiled replays; rotation of q/k/v, gate/up and down inputs is fused into add_rms_rot / "
                     "act_rot and cannot be separated; only the o_proj input FWHT is a separate kernel",
            "scripts": ["experiments/22_scale72/prof_synth.py", "experiments/22_scale72/prof_analyze.py"]},
            metrics={"configs": res, "meta": meta}, model_dir=meta["model"]))


if __name__ == "__main__":
    main()
