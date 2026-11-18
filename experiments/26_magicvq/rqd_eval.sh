#!/usr/bin/env bash
# E7 magicvq v6 tau eval: old 'p' vs draft-trained codebooks (+ SnapKV reference), paired in one process.
# Usage: experiments/26_magicvq/rqd_eval.sh <job> ; jobs: lb32a lb32h lb16 pg32 pg16
set -euo pipefail
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
DR="snapkv1024 snapkv2048 snapkv4096 vqpL1 vqpL1+r128 vqpL2 vqpL2+r128 hyb512pL1 hyb512pL2 vqpt1L1 vqpt1wL1+r128 hyb512pt1wL1 vqpt2L2 vqpt2wL2+r128 hyb512pt2wL2"
LB="nice -n 10 python experiments/26_magicvq/lb_tau.py --n 99 --n_cand 100 --G 6 --drafts $DR"
case $1 in
  lb32a) for r in "v6a v4" "v6b v4b"; do set -- $r; $LB --ctx 32768 --tasks syn_quote gov_report --run $1 --only_run $2; done ;;
  lb32h) for r in "v6a v4" "v6b v4b"; do set -- $r; $LB --ctx 32768 --tasks hotpotqa_q --run $1 --only_run $2; done ;;
  lb16)  for r in "v6a v4" "v6b v4b"; do set -- $r; $LB --ctx 16384 --tasks syn_quote hotpotqa_q gov_report --run $1 --only_run $2; done ;;
  pg32)  nice -n 10 python experiments/26_magicvq/tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 6 --run v6 --drafts $DR ;;
  pg16)  nice -n 10 python experiments/26_magicvq/tau_eval.py --ctx 16384 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 6 --run v6 --drafts $DR ;;
esac
