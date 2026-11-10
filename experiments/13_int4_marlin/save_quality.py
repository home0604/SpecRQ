"""Collect the INT4 Marlin quality evals (eval_quality.py JSONs) with the cited C1 VQ target numbers into one record:
results/E4/<run>.json. C1 values are copied from results/E3/llama31_uniform_c1_had_pile_fp32_ft5_kd123.json
(ppl_4096 k5; target fidelity from the kd=3 A_bf16 eval, which ran with the bf16 reference).
Usage: python experiments/13_int4_marlin/save_quality.py --run NAME --main JSON [--int4dense JSON] [--variant JSON]
"""
import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from tools.results import save_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--int4dense", default=None)
    ap.add_argument("--variant", default=None)
    a = ap.parse_args()
    e3 = json.load(open(os.path.join(REPO, "results/E3/llama31_uniform_c1_had_pile_fp32_ft5_kd123.json")))
    p = e3["metrics"]["ppl_4096"]
    acc = json.load(open("/data/lsh/results_accept/llama31_c1_uniform_kd3/eval/A_bf16.json"))
    c1 = {"ppl_4096": {"wikitext2": p["wikitext2_k5"], "c4": p["c4_k5"]},
          "kl_bf16_vs_target": {n: {"kl_ref_q": acc[n]["kl_fp16_target"], "top1_ref_q": acc[n]["top1_fp16_target"],
                                    "ppl_q": acc[n]["ppl_target"], "ppl_ref": acc[n]["ppl_fp16"]} for n in ("wikitext2", "c4")},
          "source": "results/E3/llama31_uniform_c1_had_pile_fp32_ft5_kd123.json + "
                    "/data/lsh/results_accept/llama31_c1_uniform_kd3/eval/A_bf16.json (cited, not re-run)"}
    files = {"main": a.main, "int4dense": a.int4dense, "variant": a.variant}
    data = {k: json.load(open(v)) for k, v in files.items() if v}
    metrics = {"c1_vq_target_cited": c1}
    for k, d in data.items():
        metrics[k] = {kk: vv for kk, vv in d.items() if kk.startswith(("ppl_", "kl_"))}
    ck = data["main"]["marlin_ckpt"]
    meta = json.load(open(os.path.join(os.path.dirname(ck), "meta.json")))
    meta.pop("pack_check_rel_err", None)
    cfg = {"setting": "Llama-3.1-8B-Instruct quality: original (bf16/fp16) vs real INT4 Marlin (GPTQ g128 sym) vs C1 VQ",
           "model": data["main"]["model"], "files": files, "gptq_meta_main": meta,
           "ppl_protocol": "PPL@4096 wikitext2 test (70 windows) / c4 val (256 windows), specrq/dbd get_loaders + "
                           "perplexity_eval math (final norm + lm_head in model dtype, CE)",
           "kl_protocol": "40 x 512 windows wikitext2 test + c4 val (draft_target_eval.py), ref = original bf16",
           "notes": "INT4 quality of record = int4dense (the same INT4 weights dequantized through the Marlin "
                    "kernel at M=16 into dense fp16 linears; deterministic). main.ppl_4096_marlin / "
                    "kl_bf16_vs_marlin are INVALID as quality numbers: they ran the original Marlin kernel at "
                    "M=512/4096, where it returns wrong, non-reproducible outputs on our 4090 setup (rel err "
                    "3-7% vs the dequant reference for M>=48, exact ~3e-4 for M<=16; log "
                    "/data/lsh/logs/e4_int4_followup2.log); kept only as evidence (wt2 7.6676 in a first run vs "
                    "7.6466 here)."}
    if a.variant:
        vck = data["variant"]["marlin_ckpt"]
        vm = json.load(open(os.path.join(os.path.dirname(vck), "meta.json")))
        vm.pop("pack_check_rel_err", None)
        cfg["gptq_meta_variant"] = vm
    print("saved", save_result("E4", a.run, config=cfg, metrics=metrics, model_dir=data["main"]["model"]))


if __name__ == "__main__":
    main()
