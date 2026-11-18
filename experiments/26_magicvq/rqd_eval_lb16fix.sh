#!/usr/bin/env bash
set -euo pipefail
cd /hai/home/lsh/SpecRQ
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
DR="snapkv1024 snapkv2048 snapkv4096 vqpL1 vqpL1+r128 vqpL2 vqpL2+r128 hyb512pL1 hyb512pL2 vqpt1L1 vqpt1wL1+r128 hyb512pt1wL1 vqpt2L2 vqpt2wL2+r128 hyb512pt2wL2"
# 16k LongBench candidates as v4 (n_cand 60) / v4b (n_cand 80), min_frac 0.9
nice -n 10 python experiments/26_magicvq/lb_tau.py --n 99 --G 6 --drafts $DR --ctx 16384 --tasks hotpotqa_q gov_report --min_frac 0.9 --n_cand 80 --run v6b --only_run v4b
nice -n 10 python experiments/26_magicvq/lb_tau.py --n 99 --G 6 --drafts $DR --ctx 16384 --tasks hotpotqa_q --min_frac 0.9 --n_cand 60 --run v6a2 --only_run v4 --skip_run v6a
