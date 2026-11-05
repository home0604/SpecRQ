#!/usr/bin/env bash
# E2-prep smoke: DbD (AQLM + Matryoshka 35W) C1 = 5 codebooks x 8 bit, group 8, on Qwen2.5-0.5B-Instruct.
# Tiny settings: 16 calib seq x 512 tokens, 2 AQ epochs x 20 steps, no fine-tune. Then prefix-k weight rel-MSE.
# Launch: tools/run_tmux.sh <name> <gpu> nestvq -- bash experiments/02_smoke/run_smoke.sh <save_dir>
set -euo pipefail
SAVE=${1:?save dir}
MODEL=/data/models/qwen2.5_0.5b_instruct
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}
cd /hai/home/lsh/SpecRQ/specrq/dbd
nvidia-smi --query-gpu=index,name,memory.used --format=csv
/usr/bin/time -v python main.py $MODEL wikitext2 \
  --nsamples=16 --val_size=0 --model_seqlen=512 \
  --num_codebooks=5 --nbits_per_codebook=8 --in_group_size=8 --out_group_size=1 \
  --codebook_weights 0 0 0.5 0 0.5 \
  --init_max_iter=20 --max_epochs=2 --steps_per_epoch=20 --relative_mse_tolerance=0.01 \
  --finetune_max_epochs=0 --local_batch_size=1 --devices cuda:0 \
  --save $SAVE
PYTHONPATH=. python ../../experiments/02_smoke/prefix_mse.py --save $SAVE --model $MODEL --ks 1 2 3 4 5 --out $SAVE/prefix_rel_mse.json
