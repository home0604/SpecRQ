#!/usr/bin/env bash
# E5 sub1b8: train a 2.0-bpw Llama-3.1-8B-Instruct target with a sub-1-bit FRONT stage.
# Recipe = experiments/15_dsweep/run_llama_dsweep.sh <cfg> <d> <nb> <L> T with NSAMPLES=192 FT_EPOCHS=5 SEED=0
# (pile_val 192x4096 calib, val 32, Hadamard, target-only alpha 0..0,1, accum fp32, rel_tol 0.01, beam 1, block FT 5 epochs
# batch 32 early stop 3 keep best, activations on GPU, eval k=L only), except
#   * ckpt root /data/lsh/ckpts/weightvq/sub1b8
#   * <nb> may be a per-stage list b1,b2,.. -> --nbits_per_codebook max(b) --codebook_bits b1 b2 .. (plain AQ main.py path)
# env CBW="a1,..,aL" (nested AQ alpha = main.py --codebook_weights, the DbD/32B mechanism; default target-only 0..0,1),
#     FT_EPOCHS (default 5; 0 = no block FT, the AQ alpha is then the only nested training), SFX (ckpt suffix, default _T_FT_n192)
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/23_sub1b8/train_sub1b.sh <cfg> <d> <nb> <L>
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}
MODEL=/data/models/llama3.1_8b_instruct
ROOT=/data/lsh/ckpts/weightvq/sub1b8
FT=${FT_EPOCHS:-5}
CKPT=$ROOT/${CFG}${SFX:-_T_FT_n192}
cd /hai/home/lsh/SpecRQ
export SPECRQ_RQ=""
CBW=${CBW:-$(python3 -c "print(' '.join(['0']*($L-1)+['1']))")}; CBW=${CBW//,/ }
[[ $(wc -w <<<"$CBW") == "$L" ]] || { echo "CBW needs $L values"; exit 2; }
if [[ "$NB" == *,* ]]; then
  NBMAX=$(tr ',' '\n' <<<"$NB" | sort -n | tail -1); BITSFLAG="--codebook_bits ${NB//,/ }"
else NBMAX=$NB; BITSFLAG=""; fi
[[ -e "$CKPT" ]] && { echo "refusing to overwrite $CKPT"; exit 1; }
mkdir -p "$ROOT"
echo "[sub1b] $CFG d=$D nbits=$NB L=$L CBW=$CBW nsamples=192 ft=$FT seed=0 $(date -Is)"
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*})"; sleep 30; done ) > $CKPT.mem.log 2>&1 &
MON=$!
trap "kill $MON 2>/dev/null || true" EXIT
(cd specrq/dbd && /usr/bin/time -v python ../../experiments/16_rq/rq_main.py $MODEL pile_val \
  --nsamples=192 --val_size=32 --model_seqlen=4096 \
  --num_codebooks=$L --nbits_per_codebook=$NBMAX $BITSFLAG --in_group_size=$D --out_group_size=1 \
  --codebook_weights $CBW --hadamard --accum_dtype float32 --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=$FT --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --devices cuda:0 --seed 0 --eval_num_codebooks $L --save $CKPT)
echo "[sub1b] done $CFG $(date -Is)"
