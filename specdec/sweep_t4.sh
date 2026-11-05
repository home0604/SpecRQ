#!/bin/bash
# Quantized-target baseline (target ≈ 4 BPA, also 3): acceptance vs draft bits. Llama-2-7B-32K-Instruct, PG-19 8k,
# 50 prompts, batch 1, greedy, γ ∈ {1,2,4}, recent window 128 exact.
#   RQ D32 nested: target t16 (4 bit) with draft m ∈ {4,6,10,12} (1/1.5/2.5/3 bit; m=8 is rq/d32_m8_t16),
#                  target t12 (3 bit) with draft m ∈ {4,6,8}.
#   Scalar, separate copies (not nested): plain RTN 4-bit target + b-bit RTN draft (b = 1,2,3), and 3-bit target + 2-bit draft.
#   Nested scalar 2+2 is lowbit/kv2_full. Target PPL / KL of the plain RTN targets go to target_curve/.
# Usage: bash specdec/sweep_t4.sh "2 7" "0 1" "3 4"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
CB_D32="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[32]_Value_K[16]C[256]D[32]_SubVec1_Stages1"
OUT=results/specdec/t4
TC=results/specdec/target_curve
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $TC $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 50 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/t4_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/t4_$name.log)"
}
ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $TC/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --ref --gpus $gpus --out $TC/$name.json "$@" > $LOG/tc_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/tc_$name.log)"
}
rq() { run "$1" "$2" --rep rq --codebook "$CB_D32" "${@:3}"; }

lane_a() {
    for m in 4 6; do for g in 1 2 4; do rq "$1" d32_m${m}_t16_g$g --m $m --target full --target-stages 16 --gamma $g; done; done
    for g in 1 2 4; do run "$1" s2_sep3_g$g --rep quantspec --kv-bits 2 --target sep3 --gamma $g; done
    for g in 1 2 4; do run "$1" s1_sep4_g$g --rep quantspec --kv-bits 1 --target sep4 --gamma $g; done
}
lane_b() {
    for m in 10 12; do for g in 1 2 4; do rq "$1" d32_m${m}_t16_g$g --m $m --target full --target-stages 16 --gamma $g; done; done
    for g in 1 2 4; do run "$1" s2_sep4_g$g --rep quantspec --kv-bits 2 --target sep4 --gamma $g; done
    ppl "$1" scalar_rtn4 --rep quantspec --kv-bits 2 --target sep4
    ppl "$1" scalar_rtn3 --rep quantspec --kv-bits 2 --target sep3
}
lane_c() {
    for m in 4 6 8; do for g in 1 2 4; do rq "$1" d32_m${m}_t12_g$g --m $m --target full --target-stages 12 --gamma $g; done; done
    for g in 1 2 4; do run "$1" s3_sep4_g$g --rep quantspec --kv-bits 3 --target sep4 --gamma $g; done
}

lane_a "${1:-2 7}" & lane_b "${2:-0 1}" & lane_c "${3:-3 4}" & wait
echo ALL_DONE
