#!/bin/bash
# Reproduce QuantSpec (Tiwari et al., ICML'25) acceptance rates on Llama-2-7B-32K-Instruct, PG-19, batch 1.
#   Table 6a (8k): QuantSpec γ∈{1,2,4,6} (W4 + hierarchical KV), StreamingLLM / SnapKV γ∈{1,2,3} (budget = ctx/4)
#   Table 3  (4k): same methods
# Plus our ablation: QuantSpec KV-only (draft weights fp16).
# Usage: bash specdec/sweep_quantspec.sh "0 1" "4 5" "6 7"     (one GPU pair per lane)
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
OUT=results/specdec/quantspec_repro
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 10"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/$name.log)"
}

lane_a() {
    for g in 1 2 4 6; do run "$1" qs_w4_ctx8k_g$g --rep quantspec --target full --kv-scale paper --draft-wbits 4 --ctx 8192 --gamma $g; done
    for g in 1 2 4; do run "$1" qs_w4_ctx4k_g$g --rep quantspec --target full --kv-scale paper --draft-wbits 4 --ctx 4096 --gamma $g; done
}
lane_b() {
    for g in 1 2 3; do run "$1" stream_ctx8k_g$g --rep streaming --ctx 8192 --gamma $g; done
    for g in 1 2 3; do run "$1" snapkv_ctx8k_g$g --rep snapkv --ctx 8192 --gamma $g; done
    for g in 1 2; do run "$1" stream_ctx4k_g$g --rep streaming --ctx 4096 --gamma $g; done
    for g in 1 2; do run "$1" snapkv_ctx4k_g$g --rep snapkv --ctx 4096 --gamma $g; done
}
lane_c() {
    for g in 1 2 4 6; do run "$1" qs_w16_ctx8k_g$g --rep quantspec --target full --kv-scale paper --draft-wbits 16 --ctx 8192 --gamma $g; done
    run "$1" fp16_ctx8k_g4 --rep fp16 --ctx 8192 --gamma 4
}

lane_a "${1:-0 1}" &
lane_b "${2:-4 5}" &
lane_c "${3:-6 7}" &
wait
echo ALL_DONE
