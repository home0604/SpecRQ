#!/usr/bin/env bash
# Draft-only codebook precision q in {8,4} (q=16 = existing screening evals) on Qwen no-FT screening ckpts.
# Rounding: experiments/06_accept/quantize_draft.py --gran codebook = one fp16 scale per draft codebook (plane) and
# linear, symmetric INT, clip ratio MSE-searched in [0.25,1] x max|v|/qmax, int8 levels +-127, int4 +-7.
# (nestkern pack: per-plane max-abs scale, int4 [-8,7] -- same granularity, slightly different rounding.)
# A = target-codebook copy rounded; B = TV-trained draft copy (screening train_B_bf16) rounded. Target stays q16.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/11_qcmp/run_qcmp.sh "<cfg>:<L>:<kd> ..."
set -euo pipefail
ITEMS=${1:?items}
MODEL=/data/models/qwen2.5_0.5b_instruct
cd /hai/home/lsh/SpecRQ
for it in $ITEMS; do
  IFS=: read -r cfg L kd <<<"$it"
  CKPT=/data/lsh/ckpts/weightvq/qwen05/screen/${cfg}_noft
  SCR=/data/lsh/results_accept/screen/${cfg}_noft/kd$kd
  OUT=/data/lsh/results_accept/qcmp/${cfg}_kd$kd; mkdir -p $OUT/draft_q $OUT/eval
  for fmt in int8 int4; do
    for v in A B; do
      init=""; [[ $v == B ]] && init="--init_draft $SCR/train_B_bf16/draft_codebooks.pt"
      pt=$OUT/draft_q/${v}_${fmt}.pt; js=$OUT/eval/${v}_${fmt}.json
      echo "[qcmp] $cfg kd=$kd $v $fmt $(date -Is)"
      [[ -e $pt ]] || CUDA_VISIBLE_DEVICES="" python experiments/06_accept/quantize_draft.py --model $MODEL --ckpt $CKPT \
        --kd $kd --fmt $fmt --gran codebook $init --out $pt
      [[ -e $js ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
        --n_windows 40 --no_fp16 --draft $pt --out $js
    done
  done
done
echo "[qcmp] done $(date -Is)"
