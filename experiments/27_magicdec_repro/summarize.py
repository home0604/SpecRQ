"""E7 MagicDec repro: tables of the official MagicDec runs (results/E7/llama8b_magicdec_official_*.json) next to our
e2e_real.py runs on the same PG-19 rows (run_md27g{80,256}_pg19md_ctx32768_B{8,16}.json).
Usage: python experiments/27_magicdec_repro/summarize.py [--save]   (any env; no GPU)
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "26_magicvq"))
from e2e_summarize import analyze, OUT  # noqa: E402

RES = os.path.join(REPO, "results", "E7")
PAPER = ("Paper (arXiv 2408.11049, Llama-3.1-8B base, PG-19, prefill 32k, 8xH100): SnapKV gamma3 Omega 3.41-3.42 at B16 "
         "(speedup 1.17-1.18), gamma4 Omega 4.03-4.10 at B32; StreamingLLM-512 gamma3 Omega 2.82-3.09.")


def f(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def official():
    R = {}
    for p in sorted(glob.glob(f"{RES}/llama8b_magicdec_official_*.json")):
        r = json.load(open(p))
        R[r["run"].replace("llama8b_magicdec_official_", "")] = r["metrics"] | {"kind": r["config"]["kind"]}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    R = official()
    L = ["# E7: official MagicDec on our 4x RTX 4090 (TP=4, GPUs 4-7) vs our SnapKV self-spec (e2e_real.py)", "",
         "Model: Llama-3.1-8B-Instruct bf16 (rev 0e9e39f) for both. PG-19 prompts from the official loader "
         "(Data/pg19, first 50 docs, skip 8000 tokens, 32672-token chunks, BOS first). " + PAPER, "",
         "## 1. Official MagicDec (unmodified code @09cd671, torch 2.5.1 + flashinfer 0.1.6, --compile)", "",
         "Exactly as the scripts report: 10 batches, batches 0-4 are warm-up, numbers = cumulative over batches 5-9. "
         "Prefill excluded (timer starts after encode). SD generation stops when the longest sequence in the batch has "
         ">= 80 new tokens or any sequence emits EOS; AR generates max_len - prefix_len = 96 tokens (stops on EOS). "
         "Omega = generated tokens / (verify steps x B). Speedup = SD tok/s / AR tok/s at the same B (paper's "
         "definition). Spread = min-max over the 5 measured batches (different prompts).", "",
         "| B | draft | gamma | Omega (batch min-max) | SD tok/s (batch min-max) | AR tok/s | speedup | ms/round draft / verify / accept+host | AR step ms | gen/seq | co-tenant |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for B in (8, 16, 32):
        ar = R.get(f"ar_b{B}")
        for k, r in R.items():
            if r["kind"] == "AR" or r["B"] != B:
                continue
            sp = r["final_tok_per_s"] / ar["final_tok_per_s"] if ar else None
            L.append(f"| {B} | {r['kind']}-{r['draft_budget']} | {r['gamma']} | {f(r['omega'])} ({f(r['omega_batch']['min'])}-"
                     f"{f(r['omega_batch']['max'])}) | {f(r['final_tok_per_s'], 1)} ({f(r['tok_per_s_batch']['min'], 1)}-"
                     f"{f(r['tok_per_s_batch']['max'], 1)}) | {f(ar['final_tok_per_s'], 1) if ar else '-'} | **{f(sp, 3)}** | "
                     f"{f(r['draft_ms'], 1)} / {f(r['verify_ms'], 1)} / {f(r['accept_loop_ms'], 2)} | "
                     f"{f(ar['step_ms']) if ar else '-'} | {f(r['gen_per_seq'], 1)} | {r['cotenant_before'] or r['cotenant_after']} |")
        if ar:
            L.append(f"| {B} | AR | - | - | - | {f(ar['final_tok_per_s'], 1)} | 1.000 | - | {f(ar['step_ms'])} | "
                     f"{ar['gen_per_seq']} | {ar['cotenant_before'] or ar['cotenant_after']} |")
    L += ["", "## 2. Ours (e2e_real.py, unchanged) on the same PG-19 rows", "",
          "B16 = official rows 80-95 (official B16 batch 5), B8 = rows 40-47 (official B8 batch 5). Fixed gen tokens/seq, "
          "EOS ignored, static batch (a finished sequence waits). Decode-only tok/s, median of 3 reps; speedup vs our AR "
          "in the same run. tau = mean accepted+1 per active (seq, round) (= Omega definition); tau_eff = gen/rounds.", "",
          "| B | gen | draft | gamma | tau | tau_eff | decode tok/s | speedup | speedup if every row stayed active (tau x T_AR / T_round, derived) | ms/round draft / verify / accept / host | AR step ms | official batch-5 Omega / tok/s (same rows) | co-tenant |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(f"{OUT}/run_md27g*_pg19md_ctx32768_B*.json")):
        if p.endswith(".partial"):
            continue
        pt = analyze(p, {"workloads": {}}, {"records": []})
        ar = pt["rows"][0]
        for r in pt["rows"][1:]:
            o = R.get(f"snap{int(r['draft'][6:]) + 1}_g{r['gamma']}_b{pt['B']}")
            ob = o["per_batch"][0] if o else None
            pr = r["per_round_ms"]
            L.append(f"| {pt['B']} | {pt['gen']} | {r['draft']} | {r['gamma']} | {f(r['tau_measured'])} | {f(r['tau_eff_batch'])} | "
                     f"{f(r['decode_tok_s'], 1)} | **{f(r['speedup_decode'], 3)}** | {f(r['tau_measured'] * ar['step_ms'] / sum(pr.values()), 3)} | {f(pr['draft_s'], 1)} / {f(pr['verify_s'], 1)} / "
                     f"{f(pr['accept_s'], 2)} / {f(pr['host_s'], 2)} | {f(ar['step_ms'])} | "
                     f"{(f(ob['omega']) + ' / ' + f(ob['tok_per_s'], 1)) if ob else '-'} | {r['cotenant'] or pt['cotenant_any']} |")
        L.append(f"| {pt['B']} | {pt['gen']} | AR | - | - | - | {f(ar['decode_tok_s'], 1)} | 1.000 | - | - | {f(ar['step_ms'])} | - | {ar['cotenant']} |")
    L += ["", "## 3. Failed / not run official points (memory, 4x24 GB)", "",
          "- B16 SnapKV-4097 (official default budget) gamma 3 and 4: CUDA OOM in the first prefill (model.py _norm), "
          "23.6 GiB in use per GPU (/data/lsh/logs/magicdec/snap4097_g{3,4}_b16.out).",
          "- B16 SnapKV-2049 gamma 3 / 4: OOM in prefill (gen_draft_kv softmax) after 2 / 1 batches "
          "(snap2049_g{3,4}_b16.out). B16 SnapKV-513 gamma 3 ran with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
          "(allocator setting only, code unchanged).",
          "- B32: not attempted; the target KV alone is 32 layers x 2 x 2 kv-heads x 128 x 2 B x 32 x 32768 = 34 GiB per GPU.",
          "- B16 StreamingLLM-513 gamma 4: stopped to save GPU time (each official B16 run takes 15-30 min: eager 128-token "
          "chunked prefill, StreamingLLM re-prefills the whole prompt for the draft cache).", "",
          "## 4. Reading", "",
          "- Same rows, same model: official Omega and our tau agree within ~2% (B8 SnapKV-4096: 3.64 vs 3.68 at gamma 3, "
          "4.38 vs 4.46 at gamma 4; B16 SnapKV-512 gamma 3: 3.39 vs 3.33 (gen 80) / 3.45 (gen 256)). Our SnapKV draft is "
          "not weaker than the official one in acceptance.",
          "- Step times: official kernels (flashinfer + torch.compile max-autotune) are slightly faster: AR step 15.96 vs "
          "16.62 ms (B8), 25.93 vs 26.8 ms (B16); SnapKV draft step 8.2 vs 8.9 ms (B8, 4096), 8.1 vs 9.1 ms (B16, 512); "
          "verify 19.5 vs 20.0 ms (B8 g3), 30.0 vs 31.0 ms (B16 g3).",
          "- The decode speedup gap (official 1.30 / 1.54 vs our 1.19 / 1.22 at B8 SnapKV-4096 g3 / B16 SnapKV-512 g3, "
          "gen 80) is the batch accounting, not the draft: the official loop keeps every sequence generating until the "
          "longest one has >= 80 tokens (and stops the whole batch at the first EOS) and counts all tokens, so throughput "
          "~ Omega x T_AR / T_round; ours gives every sequence exactly gen tokens and lets finished rows idle (tau_eff < "
          "tau). Recomputing ours with the official accounting from our measured tau / step times gives 1.31 / 1.53, "
          "matching the official 1.30 / 1.54.",
          "- Official-vs-paper: StreamingLLM-513 g3 Omega 3.01 (B8) / 2.98 (B16) is inside the paper range 2.82-3.09; "
          "SnapKV g3 Omega 3.63 (budget 4097, B8) / 3.27 (budget 513, B16) vs paper 3.41-3.42 (base model, budget not "
          "stated in the table we quote)."]
    md = "\n".join(L)
    print(md)
    if a.save:
        open(f"{RES}/llama8b_magicdec_official_tables.md", "w").write(md + "\n")
        sys.path.insert(0, REPO)
        from tools.results import save_result
        for p in sorted(glob.glob(f"{OUT}/run_md27g*_pg19md_ctx32768_B*.json")):
            pt = analyze(p, {"workloads": {}}, {"records": []})
            tag = os.path.basename(p)[4:-5]
            save_result("E7", f"llama8b_magicdec_repro_ours_{tag}",
                        config={"model": "Llama-3.1-8B-Instruct bf16", "setting": "our e2e_real.py (TP=4, GPUs 4-7) on the "
                                "official MagicDec PG-19 rows (prompts_pg19md_ctx32768.json)", "B": pt["B"], "T": 32672,
                                "gen": pt["gen"], "configs": [r["config"] for r in pt["rows"]]},
                        metrics={"rows": pt["rows"], "prefill_s": pt["prefill_s"], "cotenant_any": pt["cotenant_any"],
                                 "cotenants_start": pt["cotenants_start"], "cotenants_end": pt["cotenants_end"]},
                        model_dir="/data/models/llama3.1_8b_instruct", notes=f"raw: {p}")


if __name__ == "__main__":
    main()
