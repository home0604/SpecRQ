#!/usr/bin/env bash
# FR-Spec draft vocab: correctness/acceptance on Qwen full C1 ckpt, then synthetic Llama-3.1-8B cost v5.
# Launch: tools/run_tmux.sh frspec_gpu <gpu> nestvq -- bash experiments/08_frspec/run_frspec.sh
# Prereq: frequency tables in /data/lsh/frspec/* (experiments/08_frspec/token_freq.py).
set -e
CK=/data/lsh/ckpts/weightvq/qwen05/c1_had35w_wt2_fp64
python experiments/08_frspec/vocab_accept.py --ckpt $CK --run frspec_qwen05_had35w_vocab_accept
python -m specrq.infer.run_decode --ckpt $CK --kds 3 2 1 --gammas 1 2 4 --vds 0 8192 16384 32768 65536 \
  --run frspec_qwen05_had35w_spec_vd
python -m specrq.infer.llama_cost --kds 1 2 3 5 --vds 0 16384 32768 65536 --run l1_llama31_synthetic_cost_v5_frspec
