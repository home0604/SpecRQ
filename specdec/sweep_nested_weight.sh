#!/bin/bash
# Nested weights, step 1: QuantSpec-style hierarchical INT8 weights (draft = upper 4 bits, target = INT8).
# Llama-2-7B-32K-Instruct, PG-19 8k, batch 1, γ ∈ {1,2,4,6}. Compared with results/specdec/quantspec_repro:
#   qs_w4   : separate RTN INT4 draft weights + hierarchical KV (QuantSpec)
#   qs_w16  : KV-only
# New:
#   n8_kvqs : nested weights + hierarchical KV        (both nested; no separate draft copy)
#   n8_kvfp : nested weights, exact KV                (weight effect alone, nested)
#   w4_kvfp : separate RTN INT4 weights, exact KV     (weight effect alone, separate copy)
# Plus target PPL for each target configuration.
# Usage: bash specdec/sweep_nested_weight.sh "1 2" "6 7"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
OUT=results/specdec/nested_weight
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 10 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/nw_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/nw_$name.log)"
}
ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --gpus $gpus --out $OUT/$name.json "$@" > $LOG/nw_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/nw_$name.log)"
}

lane_a() {
    for g in 1 2 4 6; do run "$1" n8_kvqs_g$g --rep quantspec --target full --kv-scale paper --draft-wbits n8 --gamma $g; done
    ppl "$1" ppl_fp16 --rep fp16 --draft-wbits 16
    ppl "$1" ppl_w8 --rep fp16 --draft-wbits n8
}
lane_b() {
    for g in 1 2 4 6; do run "$1" n8_kvfp_g$g --rep fp16 --draft-wbits n8 --gamma $g; done
    for g in 1 2 4 6; do run "$1" w4_kvfp_g$g --rep fp16 --draft-wbits 4 --gamma $g; done
    ppl "$1" ppl_kvqs --rep quantspec --target full --kv-scale paper --draft-wbits 16
    ppl "$1" ppl_w8_kvqs --rep quantspec --target full --kv-scale paper --draft-wbits n8
}

lane_a "${1:-1 2}" &
lane_b "${2:-6 7}" &
wait
echo ALL_DONE
