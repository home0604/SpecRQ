#!/usr/bin/env bash
# E7 magicvq v10: end-to-end output-distribution fine-tune of the 'p' KV draft codebooks (rqd_e2e.py), sequential runs.
# Usage: v10_train.sh "TAG ARGS..." ["TAG ARGS..." ...]   (each quoted spec = one run, on CUDA_VISIBLE_DEVICES)
cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for spec in "$@"; do
  set -- $spec
  tag=$1; shift
  nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/rqd_e2e.py train --tag $tag "$@" 2>&1 | grep -v "Loading checkpoint"
done
