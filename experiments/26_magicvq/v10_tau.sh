#!/usr/bin/env bash
# E7 magicvq v10 (end-to-end output-distribution fine-tuned KV draft codebooks, rqd_e2e.py) tau: same samples as v6
# (--only_run on the v6 records), drafts of the v10 codebook TAGS at L2 (+ L1 if LV1=1) as r128 and SnapKV512 hybrid,
# + anchors vqpt2wL2+r128 / hyb512pt2wL2 (must reproduce the v6 values sample by sample). t1w / v9 references come from
# the v6 / v9 records.
# Usage: TAGS="e2ekl2" [LV1=1] [LV2=0] [RUN=v10] experiments/26_magicvq/v10_tau.sh <job> ; jobs: sq32 gov32hq16 gov16 pg32
set -euo pipefail
: "${TAGS:?set TAGS}"
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false V10_TAGS="$TAGS"
DR="vqpt2wL2+r128 hyb512pt2wL2"
for t in $TAGS; do
  [[ "${LV2:-1}" == 1 ]] && DR="$DR vqp${t}L2+r128 hyb512p${t}L2"
  [[ "${LV1:-0}" == 1 ]] && DR="$DR vqp${t}L1+r128 hyb512p${t}L1"
done
W="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/v10_tau.py"
LB="$W lb_tau.py --n 99 --G 6 --drafts $DR"
R=${RUN:-v10}
case $1 in
  sq32) for r in "${R}a v6a" "${R}b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks syn_quote --run $1 --only_run $2; done ;;
  gov32hq16)
    for r in "${R}a v6a" "${R}b v6b"; do set -- $r; $LB --n_cand 100 --ctx 32768 --tasks gov_report --run $1 --only_run $2; done
    $LB --n_cand 100 --ctx 16384 --tasks hotpotqa_q --run ${R}a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run ${R}b --only_run v6b
    $LB --n_cand 60 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run ${R}a2 --only_run v6a2 ;;
  gov16)
    $LB --n_cand 100 --ctx 16384 --tasks gov_report --run ${R}a --only_run v6a
    $LB --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks gov_report --run ${R}b --only_run v6b ;;
  pg32) $W tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 6 --run $R --drafts $DR ;;
esac
