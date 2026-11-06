#!/usr/bin/env bash
# Draft k=1/2 study on Qwen2.5-0.5B C1 (Had, wt2 1024x4096, fp64, alpha=0 skip).
#   [optional] train a new nested ckpt with CBW="a1 .. a5" (run_full.sh, eval k=5..1 @4096 and @512)
#   PPL @4096 for k=1..5 via --load (only when no training was done here; training already evaluates them)
#   per kd in KDS: q-suite stages A (target-copy draft codebooks, + FP16 fidelity) and B (TV-trained draft copy)
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- bash experiments/07_kd12/run_kd12.sh <ckpt> <outprefix> "<kds>" [CBW]
set -euo pipefail
CKPT=${1:?ckpt}; OUTP=${2:?out prefix}; KDS=${3:-"1 2"}; CBW=${4:-}
MODEL=/data/models/qwen2.5_0.5b_instruct
cd /hai/home/lsh/SpecRQ
if [[ -n "$CBW" ]]; then
  [[ -e "$CKPT" ]] && { echo "refusing to overwrite $CKPT"; exit 1; }
  echo "[kd12] train $CKPT CBW=$CBW $(date -Is)"
  CBW="$CBW" EVALK="5 4 3 2 1" HAD=1 bash experiments/05_qwen_full/run_full.sh "$CKPT"
else
  echo "[kd12] ppl@4096 k=5..1 $(date -Is)"
  (cd specrq/dbd && python main.py $MODEL wikitext2 --load "$CKPT" --model_seqlen=4096 --eval_num_codebooks 5 4 3 2 1 --devices cuda:0)
fi
for kd in $KDS; do
  out=${OUTP}_kd${kd}
  [[ -e "$out" ]] && { echo "refusing to overwrite $out"; exit 1; }
  echo "[kd12] q-suite kd=$kd -> $out $(date -Is)"
  KD=$kd bash experiments/06_accept/run_q_suite.sh "$CKPT" "$out" A,B
  python experiments/06_accept/summarize_q_suite.py "$out"
done
echo "[kd12] done $(date -Is)"
