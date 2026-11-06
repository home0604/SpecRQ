#!/usr/bin/env bash
# EOS-stop + chat-forced draft vocab correctness on Qwen full C1 (AR vs spec outputs), plus a --no-eos-stop
# --no-chat-forced replay of one frspec_qwen05_had35w_spec_vd config (regression vs the committed result).
set -e
CK=/data/lsh/ckpts/weightvq/qwen05/c1_had35w_wt2_fp64
python -m specrq.infer.run_decode --ckpt $CK --chat --kds 3 --gammas 2 4 --vds 32768 0 --run frspec_qwen05_had35w_eos_stop_check
python -m specrq.infer.run_decode --ckpt $CK --kds 3 --gammas 2 --vds 32768 --no-eos-stop --no-chat-forced \
  --run frspec_qwen05_had35w_noeos_regress
