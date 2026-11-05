#!/bin/bash
# Target fidelity vs target bits: PPL, KL(exact || target) and top-1 agreement with the exact model
# (teacher-forced, specdec.ppl --ref). Llama-2-7B-32K-Instruct, PG-19 8k, 10 prompts, recent window 128.
#   RQ D16: t = 4..16 stages (2..8 bit), RQ D32: t = 8..16 stages (2..4 bit),
#   scalar hierarchy (target = upper+lower): 2+2, 3+3, 4+4.
# Usage: bash specdec/sweep_target_curve.sh "0 1" "3 4"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
CB_D16="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[16]_Value_K[16]C[256]D[16]_SubVec1_Stages1"
CB_D32="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[32]_Value_K[16]C[256]D[32]_SubVec1_Stages1"
OUT=results/specdec/target_curve
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG

ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --ref --gpus $gpus --out $OUT/$name.json "$@" > $LOG/tc_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/tc_$name.log)"
}

lane_a() {
    for t in 4 6 8 10 12 16; do ppl "$1" d16_t$t --rep rq --codebook "$CB_D16" --m 4 --target full --target-stages $t; done
    ppl "$1" fp16 --rep fp16
}
lane_b() {
    for t in 8 10 12 14 16; do ppl "$1" d32_t$t --rep rq --codebook "$CB_D32" --m 8 --target full --target-stages $t; done
    for b in 2 3 4; do ppl "$1" scalar_${b}p${b} --rep quantspec --kv-bits $b --target full; done
}

lane_a "${1:-0 1}" & lane_b "${2:-3 4}" & wait
echo ALL_DONE
