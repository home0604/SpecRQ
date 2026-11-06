#!/usr/bin/env bash
# Llama-3.1-8B-Instruct C1 main run (E3): d=8, K=256, M=5, Hadamard, uniform alpha (all prefixes in the loss, so the
# alpha=0 skip has no effect), pile_val 1024x4096 calibration, accum fp32, finetune 5 epochs (else = Qwen full settings:
# val 32, rel_tol 0.01, finetune batch 32 / early stop 3 / keep_best, offload activations).
# Eval: wikitext2/c4 PPL @4096 for k=5..1. GPU memory is logged to $SAVE.gpumem.csv.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- bash experiments/08_llama_full/run_llama_full.sh <save dir>
set -euo pipefail
SAVE=${1:?save dir}
[[ -e "$SAVE" ]] && { echo "refusing to overwrite $SAVE"; exit 1; }
MODEL=${MODEL:-/data/models/llama3.1_8b_instruct}
mkdir -p "$(dirname "$SAVE")"
CBW=${CBW:-"0.2 0.2 0.2 0.2 0.2"}; EVALK=${EVALK:-"5 4 3 2 1"}; FT_EPOCHS=${FT_EPOCHS:-5}
cd /hai/home/lsh/SpecRQ/specrq/dbd
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}; sleep 30; done ) > $SAVE.gpumem.csv 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
/usr/bin/time -v python main.py $MODEL pile_val \
  --nsamples=1024 --val_size=32 --model_seqlen=4096 \
  --num_codebooks=5 --nbits_per_codebook=8 --in_group_size=8 --out_group_size=1 \
  --codebook_weights $CBW --hadamard --accum_dtype float32 \
  --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=$FT_EPOCHS --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --offload_activations --devices cuda:0 \
  --eval_num_codebooks $EVALK --save $SAVE
