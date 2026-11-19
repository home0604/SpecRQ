#!/usr/bin/env bash
# E7 magicvq v8 (1-bit KV drafts) tau: same samples as v6 (--only_run on the v6 records, same candidate params), new
# 1-bit drafts + anchors (snapkv4096, vqpt2wL2+r128, vqpt1wL1+r128 must reproduce the v6 values sample by sample).
#   pt2wL1   : nested prefix = stage 1 of the 2-bit attention-trained t2w codebook
#   pt1wL1   : dedicated 1-stage attention-trained codebook (v6 t1w = t2w recipe at level 1)
#   pt12w    : v8 nested-prefix trained codebook (loss stage 1 + stages 1+2), read at L1 and L2
# Usage: experiments/26_magicvq/v8_tau.sh <job> ; jobs: sq32 gov32hq16 gov16 pg32
set -euo pipefail
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
DR="snapkv4096 vqpt2wL2+r128 vqpt1wL1+r128 vqpt1wL1+r512 vqpt2wL1+r128 hyb512pt2wL1 vqpt12wL1+r128 vqpt12wL1+r512 vqpt12wL2+r128 hyb512pt12wL1 hyb512pt12wL2"
LB="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/lb_tau.py --n 99 --G 6 --drafts $DR"
case $1 in
  sq32) for r in "v8a v6a" "v8b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks syn_quote --run $1 --only_run $2; done ;;
  gov32hq16)
    for r in "v8a v6a" "v8b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks gov_report --run $1 --only_run $2; done
    $LB --n_cand 100 --ctx 16384 --tasks hotpotqa_q --run v8a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run v8b --only_run v6b
    $LB --n_cand 60 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run v8a2 --only_run v6a2 ;;
  gov16)
    $LB --n_cand 100 --ctx 16384 --tasks gov_report --run v8a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks gov_report --run v8b --only_run v6b ;;
  pg32) nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 6 --run v8 --drafts $DR ;;
esac
