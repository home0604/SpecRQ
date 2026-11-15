#!/usr/bin/env bash
# E5 fmt2b: rounding-only codebook FORMAT sweep on the Llama-3.1-8B-Instruct 2-bit nested d-sweep ckpts
# (adapted from experiments/11_qcmp/run_fmt4.sh; Llama flags from experiments/15_dsweep/run_llama_eval.sh).
# Draft A = copy of the first kd target codebooks rounded to <fmt> (quantize_draft.py --gran codebook; no B for Llama).
# kd < kt: prefix-draft quality at that codebook format. kd == kt: target-side probe (all stages rounded) -- the
# draft-vs-target TV/top1 then measures how much the target itself would degrade if its codebooks were stored in <fmt>
# (draft_target_eval.py target calls never take the draft path, set_k(use_draft=False)).
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/19_fmt2b/run_fmt2b.sh "<ckpt>:<L>:<kd> ..." "<fmt ...>"
set -euo pipefail
ITEMS=${1:?items}; FMTS=${2:?fmts}
MODEL=/data/models/llama3.1_8b_instruct
cd /hai/home/lsh/SpecRQ
for it in $ITEMS; do
  IFS=: read -r cfg L kd <<<"$it"
  CKPT=/data/lsh/ckpts/weightvq/dsweep_llama/$cfg
  OUT=/data/lsh/results_accept/fmt2b/${cfg}_kd$kd; mkdir -p $OUT/draft_q $OUT/eval
  for fmt in $FMTS; do
    pt=$OUT/draft_q/A_${fmt}.pt; js=$OUT/eval/A_${fmt}.json
    echo "[fmt2b] $cfg L=$L kd=$kd A $fmt $(date -Is)"
    [[ -e $pt ]] || CUDA_VISIBLE_DEVICES="" /usr/bin/time -f "[mem] quantize_draft maxrss_kb=%M wall_s=%e" \
      python experiments/06_accept/quantize_draft.py --model $MODEL --ckpt $CKPT --kd $kd --fmt $fmt --gran codebook --out $pt
    [[ -e $js ]] || /usr/bin/time -f "[mem] draft_target_eval maxrss_kb=%M wall_s=%e" \
      python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
      --n_windows 40 --codes_uint8 --no_fp16 --draft $pt --out $js
  done
done
echo "[fmt2b] done $(date -Is)"
