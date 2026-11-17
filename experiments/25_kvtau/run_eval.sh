#!/usr/bin/env bash
# E6 kvtau eval chain for one weight mode (after calib_kv.py plain + --chnorm --suffix _cn).
#   1 PPL @4096 (wt2 / c4_new, 40 windows each): chnorm + sink(keep 1) codebooks, greedy + refine
#   2 PPL plain codebooks (spec granularity, no normalization), keep 0 and keep 1 (20 windows each)
#   3 tau: chnorm + keep 1, greedy codebooks, ctx 6144 and 512, pg19 / wt2 / chat x 8 prompts x 128 tokens
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/25_kvtau/run_eval.sh <bf16|w2>
set -euo pipefail
W=${1:?wmode}
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EV="python experiments/25_kvtau/eval_kv.py --wmode $W"
$EV --parts ppl --cb_suffix _cn --keep_first 1 --variants greedy refine
$EV --parts tau --cb_suffix _cn --keep_first 1 --variants greedy
$EV --parts ppl --keep_first 0 --variants greedy --ppl_windows 20
$EV --parts ppl --keep_first 1 --variants greedy --ppl_windows 20
echo "[run_eval] done $(date -Is)"
