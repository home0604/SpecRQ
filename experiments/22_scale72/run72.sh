#!/usr/bin/env bash
# E5 scale72: L2 nested prefix (d8 K256, 2 bit/weight, alpha [0.5,0.5], Hadamard, accum fp32, wikitext2 NSxSEQ, val 32,
# AQ max_epochs 1000 / rel_tol 0.01, no block FT) = experiments/12_scale/run_scale.sh's recipe, via the layer-streaming
# driver quant72.py (weights streamed from safetensors, train activations on /data/lsh disk). Resumable (--resume + state).
# Usage: tools/run_tmux.sh <name> 6,7 nestvq -- <cpu prefix> env TAG=qwen72b MODEL=/data/models/qwen2.5_72b_instruct \
#          [NDEV=2] [QNL=n] [CKPT=..] [ACT_DIR=..] bash experiments/22_scale72/run72.sh
set -euo pipefail
TAG=${TAG:?tag}; MODEL=${MODEL:?model}; NS=${NS:-512}; SEQ=${SEQ:-4096}; QNL=${QNL:-}; MAXEP=${MAXEP:-1000}; NDEV=${NDEV:-2}
CKPT=${CKPT:-/data/lsh/ckpts/weightvq/scale/${TAG}_d8K256L2_noft}
export ACT_DIR=${ACT_DIR:-/data/lsh/temp/act72/${TAG}_d8K256L2_noft}
DEVS=$(seq -s ' ' -f 'cuda:%g' 0 $((NDEV-1)))
export HF_DATASETS_OFFLINE=1
cd /hai/home/lsh/SpecRQ
mkdir -p "$(dirname "$CKPT")" "$ACT_DIR"
echo "[q72] $TAG ns=$NS seq=$SEQ maxep=$MAXEP devs=$DEVS qnl=${QNL:-all} ckpt=$CKPT act=$ACT_DIR $(date -Is)"
/usr/bin/time -v python experiments/22_scale72/quant72.py $MODEL wikitext2 \
  --nsamples=$NS --val_size=32 --model_seqlen=$SEQ \
  --num_codebooks=2 --nbits_per_codebook=8 --in_group_size=8 --out_group_size=1 \
  --codebook_weights 0.5 0.5 --hadamard --accum_dtype float32 \
  --relative_mse_tolerance=0.01 --max_epochs=$MAXEP \
  --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --offload_activations --devices $DEVS --resume \
  ${QNL:+--quantize_num_layers $QNL} --save $CKPT
echo "[q72] done $TAG $(date -Is)"
