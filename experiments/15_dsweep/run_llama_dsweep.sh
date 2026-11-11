#!/usr/bin/env bash
# Claim-1 d-sweep on Llama-3.1-8B-Instruct, 2-bit index, target-only (alpha = 0..0,1), Hadamard, no FT.
# Recipe = experiments/08_llama_full/run_llama_full.sh (pile_val calib, val 32, seqlen 4096, accum fp32, rel_tol 0.01,
# beam 1) except: --finetune_max_epochs 0, activations on the GPU (no --offload_activations; host RAM limit, see
# below) and NSAMPLES (default 256 instead of 1024): 2 x NSAMPLES x 4096 x 4096 bf16 must fit next to AQ on a 24 GB GPU.
# The offloaded alternative needs pinned host RAM (rounded up to powers of two) that two jobs + the 32B job exceed.
# Main eval (training process): PPL@4096 wt2/c4 at k=L only (EVALK env to add prefixes).
# LAYERS=n: quantize only the first n layers (probe), separate ckpt dir.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/15_dsweep/run_llama_dsweep.sh <cfg> <d> <nbits> <L>
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}
MODEL=/data/models/llama3.1_8b_instruct
ROOT=/data/lsh/ckpts/weightvq/dsweep_llama
NS=${NSAMPLES:-256}
cd /hai/home/lsh/SpecRQ
CBW=$(python3 -c "print(' '.join(['0']*($L-1)+['1']))")
EVALK=${EVALK:-$L}
if [[ -n "${LAYERS:-}" ]]; then
  CKPT=$ROOT/probe/${CFG}_T_n${NS}_l${LAYERS}; rm -rf "$CKPT"; mkdir -p "$(dirname "$CKPT")"
  EXTRA="--quantize_num_layers $LAYERS"
else
  CKPT=$ROOT/${CFG}_T; EXTRA="--eval_num_codebooks $EVALK"
  [[ -e "$CKPT" ]] && { echo "refusing to overwrite $CKPT"; exit 1; }
  mkdir -p "$ROOT"
fi
echo "[llama-dsweep] $CFG d=$D nbits=$NB L=$L CBW=$CBW nsamples=$NS layers=${LAYERS:-all} $(date -Is)"
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}; sleep 30; done ) > $CKPT.gpumem.csv 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
(cd specrq/dbd && /usr/bin/time -v python main.py $MODEL pile_val \
  --nsamples=$NS --val_size=32 --model_seqlen=4096 \
  --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
  --codebook_weights $CBW --hadamard --accum_dtype float32 --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --devices cuda:0 $EXTRA --save $CKPT)
echo "[llama-dsweep] done $CFG $(date -Is)"
