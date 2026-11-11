#!/usr/bin/env bash
# Claim-1 d-sweep on Llama-3.1-8B-Instruct, 2-bit index, Hadamard, no FT.
# Recipe = experiments/08_llama_full/run_llama_full.sh (pile_val calib, val 32, seqlen 4096, accum fp32, rel_tol 0.01,
# beam 1) except --finetune_max_epochs 0, and:
#   OBJ   T (target-only alpha 0..0,1, default) | RQ (stage-wise calibration-aware RQ) | RQw (weight-MSE residual k-means);
#         all run through experiments/16_rq/rq_main.py (= plain main.py for T).
#   NSAMPLES (default 192; 1024 = C1 recipe)
#   OFFLOAD  gpu (default; activations on the GPU, only fits ~192 samples) | pinned (--offload_activations, recipe) |
#            file (--offload_activations with the large host buffers as file-backed mmaps in /data/lsh/temp/offload;
#            same values/dtype, pageable instead of pinned RAM)
#   SFX   ckpt / run suffix (e.g. _n1024)
#   EVALK prefixes evaluated by the training process (default: L..1)
#   LAYERS=n: quantize only the first n layers (probe), separate ckpt dir.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/15_dsweep/run_llama_dsweep.sh <cfg> <d> <nbits> <L> [OBJ]
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}; OBJ=${5:-T}
MODEL=/data/models/llama3.1_8b_instruct
ROOT=/data/lsh/ckpts/weightvq/dsweep_llama
NS=${NSAMPLES:-192}; OFF=${OFFLOAD:-gpu}; SFX=${SFX:-}
cd /hai/home/lsh/SpecRQ
CBW=$(python3 -c "print(' '.join(['0']*($L-1)+['1']))")
case $OBJ in T) export SPECRQ_RQ="";; RQ) export SPECRQ_RQ=act;; RQw) export SPECRQ_RQ=wmse;; *) echo "bad OBJ"; exit 2;; esac
case $OFF in
  gpu) OFFL="";;
  pinned) OFFL="--offload_activations";;
  file) OFFL="--offload_activations"; mkdir -p /data/lsh/temp/offload; export SPECRQ_OFFLOAD_FILE_DIR=/data/lsh/temp/offload;;
  *) echo "bad OFFLOAD"; exit 2;;
esac
EVALK=${EVALK:-$(seq $L -1 1 | tr '\n' ' ')}
if [[ -n "${LAYERS:-}" ]]; then
  CKPT=$ROOT/probe/${CFG}_${OBJ}_n${NS}_${OFF}_l${LAYERS}; rm -rf "$CKPT"; mkdir -p "$(dirname "$CKPT")"
  EXTRA="--quantize_num_layers $LAYERS"
else
  CKPT=$ROOT/${CFG}_${OBJ}${SFX}; EXTRA="--eval_num_codebooks $EVALK"
  [[ -e "$CKPT" ]] && { echo "refusing to overwrite $CKPT"; exit 1; }
  mkdir -p "$ROOT"
fi
echo "[llama-dsweep] $CFG obj=$OBJ d=$D nbits=$NB L=$L CBW=$CBW nsamples=$NS offload=$OFF layers=${LAYERS:-all} $(date -Is)"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}) cg=$(( $(cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current) >> 30 ))G $(grep -E '^(anon|shmem|file) ' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.stat | awk '{printf "%s=%dG ",$1,$2/2^30}')"; sleep 30; done ) > $CKPT.mem.log 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
(cd specrq/dbd && /usr/bin/time -v python ../../experiments/16_rq/rq_main.py $MODEL pile_val \
  --nsamples=$NS --val_size=32 --model_seqlen=4096 \
  --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
  --codebook_weights $CBW --hadamard --accum_dtype float32 --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 $OFFL --devices cuda:0 $EXTRA --save $CKPT)
echo "[llama-dsweep] done $CFG $OBJ $(date -Is)"
