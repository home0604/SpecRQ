#!/usr/bin/env bash
# 4-bit (and 3-bit) draft codebook formats beyond INT: rounding-only, same pipeline as run_qcmp.sh.
# quantize_draft.py --gran codebook --fmt <f>: one fp16 scale per (linear, draft plane) x a 16-level (or 8-level) LUT;
# scale MSE-searched over max|v| -> 2^-6..2^0.5 x largest level (int3/int4 keep the qcmp [0.25,1] clip grid).
# A = target-codebook copy rounded; B = TV-trained draft copy (screening train_B_bf16) rounded. Target stays q16.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/11_qcmp/run_fmt4.sh "<cfg>:<L>:<kd> ..." "<fmt ...>"
set -euo pipefail
ITEMS=${1:?items}; FMTS=${2:-"nf4 fp4_e2m1 posit4_es0 posit4_es1"}
MODEL=/data/models/qwen2.5_0.5b_instruct
cd /hai/home/lsh/SpecRQ
for it in $ITEMS; do
  IFS=: read -r cfg L kd <<<"$it"
  CKPT=/data/lsh/ckpts/weightvq/qwen05/screen/${cfg}_noft
  SCR=/data/lsh/results_accept/screen/${cfg}_noft/kd$kd
  OUT=/data/lsh/results_accept/qcmp_fmt4/${cfg}_kd$kd; mkdir -p $OUT/draft_q $OUT/eval
  for fmt in $FMTS; do
    for v in A B; do
      init=""; [[ $v == B ]] && init="--init_draft $SCR/train_B_bf16/draft_codebooks.pt"
      pt=$OUT/draft_q/${v}_${fmt}.pt; js=$OUT/eval/${v}_${fmt}.json
      echo "[fmt4] $cfg kd=$kd $v $fmt $(date -Is)"
      [[ -e $pt ]] || CUDA_VISIBLE_DEVICES="" python experiments/06_accept/quantize_draft.py --model $MODEL --ckpt $CKPT \
        --kd $kd --fmt $fmt --gran codebook $init --out $pt
      [[ -e $js ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
        --n_windows 40 --no_fp16 --draft $pt --out $js
    done
  done
done
echo "[fmt4] done $(date -Is)"
