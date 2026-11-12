#!/usr/bin/env bash
# KL(bf16||target)/top-1 for a Llama d-sweep ckpt (protocol of experiments/08_llama_full/run_llama_accept.sh A<kd> with
# the reference: draft_target_eval --codes_uint8, bf16 reference on the second visible GPU) + results/E5 record.
# env SFX (run/ckpt suffix, e.g. _n1024), NSAMPLES (for the record), TRAIN_LOG (default /data/lsh/logs/e5_lds_<cfg>_<obj><sfx>.log)
# Usage: tools/run_tmux.sh <name> <gpu,gpu_ref> nestvq -- <cpu prefix> bash experiments/15_dsweep/run_llama_eval.sh <cfg> <d> <nbits> <L> [OBJ]
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}; OBJ=${5:-T}; SFX=${SFX:-}
MODEL=/data/models/llama3.1_8b_instruct
CKPT=/data/lsh/ckpts/weightvq/dsweep_llama/${CFG}_${OBJ}${SFX}
OUT=/data/lsh/results_accept/dsweep_llama/${CFG}_${OBJ}${SFX}
cd /hai/home/lsh/SpecRQ
mkdir -p $OUT/eval
# reference placement: second visible GPU (default) or, with REF_MAXMEM (e.g. "0:5GiB,cpu:40GiB"), accelerate dispatch
# on the same GPU + CPU (single-GPU eval; experiments/06_accept/draft_target_eval.py --ref_max_memory)
if [[ -n "${REF_MAXMEM:-}" ]]; then REFARG="--ref_max_memory $REF_MAXMEM"; else REFARG="--ref_device cuda:1"; fi
[[ -e $OUT/eval/A_bf16.json ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kt $L --kd 1 \
  --n_windows 40 --codes_uint8 $REFARG --out $OUT/eval/A_bf16.json
if [[ $OBJ == N_* ]]; then CBW=${CBW:?nested run needs env CBW}; CBW=${CBW//,/ }
else CBW=$(python3 -c "print(' '.join(['0']*($L-1)+['1']))"); fi
KDS=${KDS:-}; KDS=${KDS//,/ }
for kd in $KDS; do  # copy-draft A at prefix kd (TV, top-1, greedy tau g1/2/4 chat + wt2), no FP16 reference
  mkdir -p $OUT/kd$kd/eval
  [[ -e $OUT/kd$kd/eval/A_bf16.json ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT \
    --kt $L --kd $kd --n_windows 40 --codes_uint8 --no_fp16 --out $OUT/kd$kd/eval/A_bf16.json
done
python experiments/15_dsweep/save_dsweep.py --arch llama31 --cfg $CFG --d $D --nbits $NB --L $L --obj $OBJ --run_suffix "$SFX" --cbw "$CBW" --kds "$KDS" --kd_root $OUT \
  --ckpt $CKPT --eval $OUT/eval/A_bf16.json --log ${TRAIN_LOG:-/data/lsh/logs/e5_lds_${CFG}_${OBJ}${SFX}.log} \
  --calib_note "pile_val ${NSAMPLES:-192}x4096 calib (val 32; C1 recipe 1024), accum fp32, seed ${SEED:-0}" --ft_epochs ${FT_EPOCHS:-0}
