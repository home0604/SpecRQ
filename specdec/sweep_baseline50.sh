#!/bin/bash
# Baseline re-run with 50 prompts (10-prompt runs had ~±2%p noise). Llama-2-7B-32K-Instruct, PG-19 8k, batch 1.
# Same configs as quantspec_repro (8k) + nested_weight; PPL is not repeated (already stable over 80k tokens).
# Usage: bash specdec/sweep_baseline50.sh "1 2" "3 5"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
OUT=results/specdec/baseline50
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 50 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/b50_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/b50_$name.log)"
}

lane_a() {
    for g in 1 2 4 6; do run "$1" qs_w4_g$g  --rep quantspec --target full --kv-scale paper --draft-wbits 4  --gamma $g; done
    for g in 1 2 4 6; do run "$1" n8_kvqs_g$g --rep quantspec --target full --kv-scale paper --draft-wbits n8 --gamma $g; done
    for g in 1 2 3; do run "$1" stream_g$g --rep streaming --draft-wbits 16 --gamma $g; done
    for g in 1 2 4 6; do run "$1" w4_kvfp_g$g --rep fp16 --draft-wbits 4 --gamma $g; done
}
lane_b() {
    for g in 1 2 4 6; do run "$1" qs_w16_g$g --rep quantspec --target full --kv-scale paper --draft-wbits 16 --gamma $g; done
    for g in 1 2 4 6; do run "$1" n8_kvfp_g$g --rep fp16 --draft-wbits n8 --gamma $g; done
    for g in 1 2 3; do run "$1" snapkv_g$g --rep snapkv --draft-wbits 16 --gamma $g; done
}

lane_a "${1:-1 2}" &
lane_b "${2:-3 5}" &
wait
echo ALL_DONE
