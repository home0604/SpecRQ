"""Collect one kd12 study (experiments/07_kd12/run_kd12.sh) into results/E3/<run>.json via save_result.

Usage: python experiments/07_kd12/save_kd12.py --run NAME --ckpt DIR --cbw "a1 .. a5" --log JOBLOG \
         --suite kd:DIR [--suite kd:DIR ...] [--extra_log LOG] [--notes TEXT]
PPL @4096 / @512 for k=1..5 are parsed from '[specrq] RESULT dataset=.. k=.. ppl=..' lines in the given logs, in order:
the first occurrence of each (dataset, k) is @4096 (training run / --load @4096), a second one is @512 (run_full.sh's
--load eval at seqlen 512).
"""
import argparse, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from tools.results import save_result

RE = re.compile(r"\[specrq\] RESULT dataset=(\S+) k=(\S+) ppl=([0-9.]+)")


def parse_ppl(logs):
    out = {"ppl_4096": {}, "ppl_512": {}}
    for lg in logs:
        for line in open(lg, errors="replace"):
            m = RE.search(line)
            if not m:
                continue
            ds, k, p = m.group(1), m.group(2), float(m.group(3))
            key = f"{ds}_k{k}"
            tgt = "ppl_4096" if key not in out["ppl_4096"] else "ppl_512"
            out[tgt].setdefault(key, p)
    return out


def suite_rows(d):
    rows = {}
    for v in ("A_bf16", "B_bf16"):
        p = f"{d}/eval/{v}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        row = {}
        for ds in ("wikitext2", "c4"):
            x = r[ds]
            row[ds] = {k: x[k] for k in ("tv_mean", "accept_prob_1mTV", "top1_agree", "kl_t_d", "kl_d_t", "ppl_target",
                                         "ppl_draft", "ppl_fp16", "kl_fp16_target", "top1_fp16_target") if k in x}
        g = r["greedy_sim"]
        row["greedy"] = {kind: {"match_rate": g[kind]["draft_match_rate_on_target_path"],
                                **{f"g{gm}_tok_per_iter": g[kind][f"gamma{gm}"]["tokens_per_iter"] for gm in (1, 2, 4)}}
                         for kind in ("instruct", "wikitext2")}
        row["greedy"]["target_consistency"] = g["target_teacher_forced_vs_generated_consistency"]
        row["eval_json"] = p
        rows[v] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cbw", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--extra_log", action="append", default=[])
    ap.add_argument("--suite", action="append", default=[], help="kd:dir")
    ap.add_argument("--tmux_job", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    metrics = parse_ppl([a.log] + a.extra_log)
    metrics["kd"] = {}
    for s in a.suite:
        kd, d = s.split(":", 1)
        metrics["kd"][kd] = {"dir": d, **suite_rows(d)}
    config = {
        "model": "Qwen2.5-0.5B-Instruct", "ckpt": a.ckpt,
        "setting": "C1 d=8 K=256 M=5, Hadamard, wikitext2 1024x4096 calib, accum fp64, alpha=0 prefixes skipped",
        "codebook_weights": [float(x) for x in a.cbw.split()],
        "draft": "A = target-copy codebooks (bf16); B = draft-only codebooks+scales TV-trained (pile_val_heldout 1024x512, "
                 "300 steps, batch 4, lr 1e-4)",
        "eval": {"tool": "experiments/06_accept/draft_target_eval.py", "kt": 5,
                 "windows": "40 x 512 wikitext2 test + 40 x 512 c4 val",
                 "greedy": "8 chat prompts + 8 wikitext2 prompts, 128 new tokens; tokens/iter simulated from target greedy path",
                 "ppl": "specrq/dbd/main.py perplexity_eval @4096 (and @512 for trained ckpts)"},
        "scripts": ["experiments/07_kd12/run_kd12.sh", "experiments/06_accept/run_q_suite.sh",
                    "experiments/05_qwen_full/run_full.sh", "experiments/07_kd12/save_kd12.py"],
        "gpu": a.gpu, "logs": [a.log] + a.extra_log,
    }
    config["tmux_job"] = a.tmux_job
    p = save_result("E3", a.run, config=config, metrics=metrics, model_dir="/data/models/qwen2.5_0.5b_instruct",
                    notes=a.notes)
    print(p)


if __name__ == "__main__":
    main()
