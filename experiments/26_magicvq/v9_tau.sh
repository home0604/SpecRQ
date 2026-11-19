#!/usr/bin/env bash
# E7 magicvq v9 (nested-prefix codebook retry) tau: same samples as v6 (--only_run on the v6 records), drafts of one
# v9 codebook TAG (mv_common name without the 'p', e.g. t12w_v2lag3) at L1 and L2 (r128 and SnapKV512 hybrid) +
# anchor vqpt2wL2+r128 (must reproduce the v6 values sample by sample). References t1w / t2w come from the v6 records.
# Usage: TAG=t12w_v2lag3 experiments/26_magicvq/v9_tau.sh <job> ; jobs: sq32 gov32hq16 gov16 pg32
set -euo pipefail
: "${TAG:?set TAG}"
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
DR="vqpt2wL2+r128 vqp${TAG}L1+r128 vqp${TAG}L2+r128 hyb512p${TAG}L1 hyb512p${TAG}L2"
LB="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/lb_tau.py --n 99 --G 6 --drafts $DR"
case $1 in
  sq32) for r in "v9a v6a" "v9b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks syn_quote --run $1 --only_run $2; done ;;
  gov32hq16)
    for r in "v9a v6a" "v9b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks gov_report --run $1 --only_run $2; done
    $LB --n_cand 100 --ctx 16384 --tasks hotpotqa_q --run v9a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run v9b --only_run v6b
    $LB --n_cand 60 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run v9a2 --only_run v6a2 ;;
  gov16)
    $LB --n_cand 100 --ctx 16384 --tasks gov_report --run v9a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks gov_report --run v9b --only_run v6b ;;
  pg32) nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 6 --run v9 --drafts $DR ;;
esac
