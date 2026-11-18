"""Summarize maxreuse.py (draft-max reuse as unified max in verify) -> results/E7/llama8b_magicvq_maxreuse_*.
Usage: python experiments/26_magicvq/maxreuse_summarize.py --tag mr1 [--save]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
D = "/data/lsh/results_accept/magicvq/maxreuse"
RES = os.path.join(REPO, "results", "E7")
LABEL = {"snapkv4096": "SnapKV-4096", "vqpL2+r128": "RQ2+r128", "hyb512pL2": "SnapKV512+RQ2 hybrid"}


def row(task, c, k, s):
    n = s["n"]
    return (f"| {task} | {LABEL.get(c.split(':')[0], c)} g{c.split(':')[1]} | {k} | {n} | {s['mean']:.3f} | {s['p1']:.3f} | "
            f"{s['p50']:.3f} | {s['p99']:.3f} | {s['p99.9']:.3f} | {s['min']:.2f} | {s['max']:.2f} | "
            f"{round(s['frac_gt20'] * n)} ({s['frac_gt20']:.1e}) | {round(s['frac_lt_m20'] * n)} | "
            f"{s['frac_abs_gt5']:.1e} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="mr1")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    st = {f: json.load(open(f)) for f in sorted(glob.glob(f"{D}/stats_{a.tag}_*.json"))}
    pr = {f: json.load(open(f)) for f in sorted(glob.glob(f"{D}/prof_{a.tag}_*.json"))}
    L = ["# E7 phase 2: reuse the draft's softmax max as a fixed (unified) max in verify attention: gap statistics and "
         "best-case time saving", "",
         "Llama-3.1-8B-Instruct bf16, TP=4 RTX 4090 (GPUs 4-7), e2e_real.py model / drafts / kernels "
         "(experiments/26_magicvq/maxreuse.py). Analysis only, no new kernel.", "",
         "## (a) gap = m_target - m_hat (natural units, score = q.k/sqrt(128))", "",
         "Per layer x sequence x q-head x verify row x round (active rows), B=4 prompts, gen 64, gamma 4, real prefill, "
         "eager spec loop. m_target = max over the exact bf16 cache (verify query q_t, from the verify kernel's per-split "
         "maxima). m_hat (q_d) = the draft step's own max (draft query, draft cache), rows 0..gamma-1 (the bonus row "
         "has no draft query). m_hat (q_t) = verify query over the draft cache (all gamma+1 rows; bonus row also "
         "listed separately). Positive gap: true max above the estimate (exp(s - m_hat) can reach e^gap); negative: "
         "estimate too high (sum shrinks by e^gap). m_target row = the distribution of m_target itself.", "",
         "| task | draft | quantity | n | mean | p1 | p50 | p99 | p99.9 | min | max | # gap > 20 (frac) | # gap < -20 | "
         "frac abs(gap) > 5 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in st.items():
        task = r["args"]["task"]
        for c, x in r["results"].items():
            for k, lab in (("m_hat_qd", "gap, m_hat with q_d"), ("m_hat_qt", "gap, m_hat with q_t"),
                           ("m_hat_qt_bonus_row", "gap, q_t, bonus row only"), ("m_target", "m_target (value)")):
                L.append(row(task, c, lab, x[k]))
    L += ["", "Co-tenant flags (stats runs): " + ", ".join(f"{os.path.basename(f)}: {r['cotenant_any']}" for f, r in st.items()), ""]
    L += ["## (b) best-case saving in the verify step (syn_quote 32k prompt lengths, B=16, TP=4)", "",
          "Verify attention = split-K flash-decoding partial kernel (_bf_partial, online max + alpha rescale inside the "
          "kernel) + a SEPARATE log-sum-exp merge kernel (_combine) per layer. A unified max would remove the merge's "
          "rescale (exp2(m_i - m_max) weights; a sum across splits is still needed) and the in-kernel alpha rescale. "
          "Upper bound = whole _combine time + (partial - fixed-max partial). Fixed-max partial = the same kernel with "
          "the running max / alpha removed (timing probe, cache contents random). Times: CUDA-graph replay, median of "
          "20 (min-max); profile: eager, torch.profiler, 5 steps (NCCL time in eager includes rank skew, so shares are "
          "of the eager GPU total).", "",
          "| gamma | verify plan (splits, chunk) | verify step ms (graph) | attention all layers ms | partial ms | "
          "fixed-max partial ms | combine ms | upper-bound saving ms | share of verify | eager profile share: partial / "
          "combine / NCCL / GEMM / other |", "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in pr.items():
        for g, x in r["results"].items():
            sh = x["profile_class_share"]
            L.append(f"| {x['gamma']} | {x['verify_plan'][0]} | {x['graph_verify_ms'][0]:.2f} ({x['graph_verify_ms'][1]:.2f}-"
                     f"{x['graph_verify_ms'][2]:.2f}) | {x['graph_attn_all_layers_ms'][0]:.2f} | {x['graph_bf_partial_ms'][0]:.2f} | "
                     f"{x['graph_bf_partial_fixed_max_ms'][0]:.2f} | {x['graph_combine_ms'][0]:.3f} | "
                     f"{x['upper_bound_saving_ms']:.3f} | {100 * x['upper_bound_saving_share_of_verify']:.2f}% | "
                     f"{100 * sh['bf_partial']:.1f}% / {100 * sh['combine']:.2f}% / {100 * sh['nccl']:.1f}% / "
                     f"{100 * sh['gemm']:.1f}% / {100 * (sh['other'] + sh['append']):.1f}% |")
        L.append("")
        L.append(f"Co-tenant flag (prof run): {r['cotenant_any']}")
    md = "\n".join(L) + "\n"
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicvq_maxreuse_tables.md", "w").write(md)
        from tools.results import save_result
        for f, r in list(st.items()) + list(pr.items()):
            ar = r["args"]
            save_result("E7", f"llama8b_magicvq_maxreuse_{ar['mode']}_{ar['task']}_ctx{ar['ctx']}_B{ar['B']}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "TP=4 RTX 4090 (GPUs 4-7)", **ar},
                        metrics={"results": r["results"], "cotenant_any": r["cotenant_any"]},
                        model_dir="/data/models/llama3.1_8b_instruct", notes=f"raw: {f}; analysis only, no new kernel")


if __name__ == "__main__":
    main()
