#!/usr/bin/env bash
# Timing probe for C2-L5 on Qwen2.5-0.5B-Instruct: d=8, K=4096 (12-bit codes), M=5, uniform alpha, Hadamard,
# wikitext2 1024x4096, accum fp64, Qwen full FT settings; first N layers only (no eval).
# Every log line is prefixed with a wall-clock timestamp (epoch seconds) so stage times can be read off the log:
# XTX (layer header -> first 'Quantizing module'), k-means init ('Quantizing module' -> aq_epoch_start 0),
# AQ epochs (Adam steps + beam search), FT ('PREPARING TO FINETUNE' -> 'FINISHED FINETUNING').
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/09_c2_timing/run_c2_timing.sh <save> [N] [nbits]
set -euo pipefail
SAVE=${1:?save dir}; NL=${2:-2}; NBITS=${3:-12}
[[ -e "$SAVE" ]] && { echo "refusing to overwrite $SAVE"; exit 1; }
mkdir -p "$(dirname "$SAVE")"
MODEL=/data/models/qwen2.5_0.5b_instruct
cd /hai/home/lsh/SpecRQ/specrq/dbd
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}; sleep 5; done ) > $SAVE.gpumem.csv 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
/usr/bin/time -v python main.py $MODEL wikitext2 \
  --nsamples=1024 --val_size=32 --model_seqlen=4096 \
  --num_codebooks=5 --nbits_per_codebook=$NBITS --in_group_size=8 --out_group_size=1 \
  --codebook_weights 0.2 0.2 0.2 0.2 0.2 --hadamard --accum_dtype float64 \
  --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=10 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --offload_activations --devices cuda:0 \
  --quantize_num_layers $NL --save $SAVE 2>&1 \
  | python3 -u -c 'import sys,time
for l in sys.stdin: sys.stdout.write(f"{time.time():.2f} {l}"); sys.stdout.flush()'
exit ${PIPESTATUS[0]}
