#!/bin/bash
# Low-BPA targets (≤ 2 bit), where VQ should beat scalar. Llama-2-7B-32K-Instruct, PG-19 8k, 50 prompts, batch 1,
# greedy, γ ∈ {1,2,4}, recent window 128 exact.
#   Target quality (target_curve/, --ref): scalar plain RTN2, D32 t4/t6 (1/1.5 bit), D16 SV8 t2/t4.
#   Acceptance (t2/): D32 target t8 (2 bit) with draft m ∈ {2,4,6} (0.5/1/1.5 bit), target t6 (1.5 bit) with m ∈ {2,4},
#                     D16 SV1 target t4 (2 bit) with m=2 (1 bit).
# No scalar acceptance baseline: a nested scalar draft below 2 bit is 1-bit RTN, which collapses (t4/s1_sep4: 7%).
# Usage: bash specdec/sweep_t2.sh "0 6"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
CB_D16="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[16]_Value_K[16]C[256]D[16]_SubVec1_Stages1"
CB_SV8="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[16]_Value_K[16]C[256]D[16]_SubVec8_Stages1"
CB_D32="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[32]_Value_K[16]C[256]D[32]_SubVec1_Stages1"
OUT=results/specdec/t2
TC=results/specdec/target_curve
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $TC $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 50 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/t2_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/t2_$name.log)"
}
ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $TC/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --ref --gpus $gpus --out $TC/$name.json "$@" > $LOG/tc_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/tc_$name.log)"
}

lane() {
    ppl "$1" scalar_rtn2 --rep quantspec --kv-bits 1 --target sep2
    for t in 4 6; do ppl "$1" d32_t$t --rep rq --codebook "$CB_D32" --m 2 --target full --target-stages $t; done
    for t in 2 4; do ppl "$1" sv8_t$t --rep rq --codebook "$CB_SV8" --m 2 --target full --target-stages $t; done
    for m in 4 2 6; do for g in 1 2 4; do
        run "$1" d32_m${m}_t8_g$g --rep rq --codebook "$CB_D32" --m $m --target full --target-stages 8 --gamma $g; done; done
    for m in 2 4; do for g in 1 2 4; do
        run "$1" d32_m${m}_t6_g$g --rep rq --codebook "$CB_D32" --m $m --target full --target-stages 6 --gamma $g; done; done
    for g in 1 2 4; do run "$1" d16_m2_t4_g$g --rep rq --codebook "$CB_D16" --m 2 --target full --target-stages 4 --gamma $g; done
}

lane "${1:-0 6}"
echo ALL_DONE
