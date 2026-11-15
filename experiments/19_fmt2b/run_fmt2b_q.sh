#!/usr/bin/env bash
# E5 fmt2b queue worker (several GPUs share one queue; an item is claimed atomically with mkdir <OUT>/lock/<label>).
# Queue file lines: <ckpt>:<L>:<kd>:<label>:<fmtspec>  fmtspec = one format (--fmt) or a comma list of kd formats
# (--fmt_stages, mixed per-stage target-side probe). Outputs as run_fmt2b.sh: <OUT>/{draft_q,eval}/A_<label>.*
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/19_fmt2b/run_fmt2b_q.sh <queue file>
set -euo pipefail
Q=${1:?queue}; MODEL=/data/models/llama3.1_8b_instruct
cd /hai/home/lsh/SpecRQ
while IFS=: read -r cfg L kd label spec; do
  [[ -z "$cfg" || "$cfg" == \#* ]] && continue
  CKPT=/data/lsh/ckpts/weightvq/dsweep_llama/$cfg
  OUT=/data/lsh/results_accept/fmt2b/${cfg}_kd$kd; mkdir -p $OUT/draft_q $OUT/eval $OUT/lock
  pt=$OUT/draft_q/A_${label}.pt; js=$OUT/eval/A_${label}.json
  [[ -e $js ]] && continue
  mkdir $OUT/lock/$label 2>/dev/null || continue
  echo "$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)" > $OUT/lock/$label/owner
  if [[ $spec == *,* ]]; then FA="--fmt_stages $spec"; else FA="--fmt $spec"; fi
  echo "[fmt2b] $cfg L=$L kd=$kd A $label ($spec) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
  [[ -e $pt ]] || CUDA_VISIBLE_DEVICES="" /usr/bin/time -f "[mem] quantize_draft maxrss_kb=%M wall_s=%e" \
    python experiments/06_accept/quantize_draft.py --model $MODEL --ckpt $CKPT --kd $kd $FA --gran codebook --out $pt
  /usr/bin/time -f "[mem] draft_target_eval maxrss_kb=%M wall_s=%e" \
    python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
    --n_windows 40 --codes_uint8 --no_fp16 --draft $pt --out $js
done < <(cat "$Q")
echo "[fmt2b] worker done $(date -Is)"
