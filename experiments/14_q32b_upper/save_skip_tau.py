"""Merge skip_tau.py outputs and save results/E5/<run>.json with break-even tau and the expected speed-up
tau * T_AR / round_ms from the synthetic timing record (results/E4/q32b_L2_synth_prefix_layerskip.json).
Patterns without a timing row (e.g. even16_keep4) use the mean round_ms of the timed patterns with the same number of
skipped layers (timing is pattern-independent there: ~0.143 ms per layer).
Usage: python experiments/14_q32b_upper/save_skip_tau.py --run NAME --jsons A.json B.json ... [--print_only]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TIMING = REPO / "results/E4/q32b_L2_synth_prefix_layerskip.json"


def timing_table():
    d = json.load(open(TIMING))
    pats = d["metrics"]["skip_patterns"]
    rows = d["metrics"]["summary"]
    T_ar = rows[0]["T_ar_ms"]
    tab = {}
    for r in rows:
        for g, v in r["gamma"].items():
            tab[(r["skip"], r["vd"], int(g))] = v["round_ms"]
    by_n = {}
    for (sk, vd, g), x in tab.items():
        by_n.setdefault((len(pats[sk]), vd, g), []).append(x)
    return T_ar, tab, {k: statistics.mean(v) for k, v in by_n.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--jsons", nargs="+", required=True)
    ap.add_argument("--print_only", action="store_true")
    a = ap.parse_args()
    T_ar, tab, by_n = timing_table()
    rows, meta = [], None
    for p in a.jsons:
        d = json.load(open(p))
        meta = meta or {k: d[k] for k in ("eos", "path_lens", "kinds", "target_tf_consistency", "freq_dir",
                                          "vd_coverage_pileval")}
        for r in d["rows"]:
            r["skip_layers"] = d["skip_patterns"][r["skip"]]
            r["source"] = p
            rows.append(r)
    table = []
    for r in rows:
        for key, dec in r["decode"].items():
            kv, vds, gs = key.split("_")
            vd, g = int(vds[2:]), int(gs[1:])
            rm = tab.get((r["skip"], vd, g))
            src = "pattern"
            if rm is None:
                rm, src = by_n.get((r["n_skip"], vd, g)), f"mean of n_skip={r['n_skip']}"
            e = {"draft": r["draft"], "skip": r["skip"], "n_skip": r["n_skip"], "kv": kv, "vd": vd, "gamma": g,
                 "tau_chat": dec["instruct"]["tau"], "tau_wt2": dec["wikitext2"]["tau"],
                 "tau_all": (dec["instruct"]["tokens"] + dec["wikitext2"]["tokens"]) /
                            (dec["instruct"]["rounds"] + dec["wikitext2"]["rounds"]),
                 "exact_prompts": dec["exact_prompts"],
                 "sim_tau_chat": r["sim"][str(vd)]["instruct"].get(f"gamma{g}"),
                 "sim_tau_wt2": r["sim"][str(vd)]["wikitext2"].get(f"gamma{g}"),
                 "round_ms_timing": rm, "round_ms_source": src}
            if rm:
                e["tau_breakeven"] = rm / T_ar
                for k in ("chat", "wt2", "all"):
                    e[f"exp_speedup_{k}"] = e[f"tau_{k}"] * T_ar / rm
            table.append(e)
    for e in sorted(table, key=lambda e: (e["kv"], e["vd"], e["draft"], e["n_skip"], e["skip"], e["gamma"])):
        print(f"{e['kv']:6s} vd{e['vd']:6d} {e['draft']} {e['skip']:13s} g{e['gamma']} tau chat {e['tau_chat']:.2f} "
              f"wt2 {e['tau_wt2']:.2f} (sim {e['sim_tau_chat']:.2f}/{e['sim_tau_wt2']:.2f}) BE "
              f"{e.get('tau_breakeven', float('nan')):.2f} speedup {e.get('exp_speedup_chat', float('nan')):.2f}/"
              f"{e.get('exp_speedup_wt2', float('nan')):.2f} exact {e['exact_prompts']}/16")
    if a.print_only:
        return
    from tools.results import save_result
    cfg = {"setting": "REAL Qwen2.5-32B-Instruct 2-bit nested checkpoint d8 K256 L2 noFT (Hadamard, alpha .5/.5); "
                      "draft = prefix k1 + decoder-layer skip (SkipView: skipped layers pass the residual, no KV write); "
                      "target/verify = k2 all layers; greedy",
           "ckpt": "/data/lsh/ckpts/weightvq/scale/qwen32b_d8K256L2_noft",
           "draft_A": "target codebook plane 1", "draft_B": "/data/lsh/results_accept/scale/qwen32b_d8K256L2_noft/kd1/"
                                                            "train_B_bf16/draft_codebooks.pt (TV-trained, no skip)",
           "prompts": "8 chat (chat template) + 8 wikitext2-test 64-token cuts, target greedy path <= 128 tokens, EOS stop",
           "tau_decode": "real SpecDecoder rounds on our runner: confirmed decode tokens / rounds (prefill token excluded); "
                         "kv=shared as in the timing run (draft attends to target k2 KV of committed context); kv=own "
                         "only for no-skip (bridge)",
           "tau_sim": "teacher-forced draft (own KV) argmax vs target path, greedy simulation (draft_target_eval protocol, "
                      "path incl. prefill token)",
           "corpus": "teacher-forced TV/top-1 on 8 x 512 windows each of wikitext2 test / c4 val",
           "expected_speedup": "[I] measured tau x synthetic-timing T_AR / round_ms (results/E4/q32b_L2_synth_prefix_"
                               "layerskip.json), not a measured end-to-end speed-up",
           "T_ar_ms_timing": T_ar, "scripts": ["experiments/14_q32b_upper/skip_tau.py",
                                              "experiments/14_q32b_upper/save_skip_tau.py"], "sources": a.jsons, **meta}
    print(save_result("E5", a.run, cfg, {"table": table, "rows": rows}, model_dir="/data/models/qwen2.5_32b_instruct"))


if __name__ == "__main__":
    main()
