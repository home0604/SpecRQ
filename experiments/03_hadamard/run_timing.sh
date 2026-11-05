#!/usr/bin/env bash
# Timing probe: DbD quantize.sh settings (calib 1024 x 4096, val 32, steps/epoch 100, rel_tol 0.01, finetune 10/3/keep_best,
# offload activations) + Hadamard + 35W on the first N layers of Qwen2.5-0.5B-Instruct. No evaluation.
set -euo pipefail
SAVE=${1:?save dir}; NL=${2:-2}
cd /hai/home/lsh/SpecRQ/specrq/dbd
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}; sleep 10; done ) > $SAVE.gpumem.csv 2>&1 &
MON=$!
/usr/bin/time -v python main.py ${MODEL:-/data/models/qwen2.5_0.5b_instruct} wikitext2 \
  --nsamples=${NSAMPLES:-1024} --val_size=32 --model_seqlen=4096 \
  --num_codebooks=5 --nbits_per_codebook=8 --in_group_size=8 --out_group_size=1 \
  --codebook_weights 0 0 0.5 0 0.5 --hadamard \
  --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=10 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --offload_activations --devices cuda:0 ${EXTRA_ARGS:-} \
  --quantize_num_layers $NL --save $SAVE
kill $MON
