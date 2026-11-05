#!/bin/bash
# KV RQ prefix draft vs scalar hierarchy (results/specdec/lowbit). Llama-2-7B-32K-Instruct, PG-19 8k, 50 prompts,
# batch 1, greedy, γ ∈ {1,2,4}. Recent 128..256 tokens exact (--rq-window 128, same as QuantSpec).
# Stage 1 (target exact): d16_m{4,6,8} = 2/3/4-bit draft. kv4_tfp = scalar 4-bit draft reference.
# Stage 2 (target = RQ prefix + residual): d16_m4_t8 (2-bit draft, 4-bit target), PPL of 4/8-bit RQ targets.
# Usage: bash specdec/sweep_rq.sh d16 "0 2" "3 4" "5 6"   (d32: third lane "-" runs lane_c after lane_b)
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
CB_D16="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[16]_Value_K[16]C[256]D[16]_SubVec1_Stages1"
CB_SV8="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[16]_Value_K[16]C[256]D[16]_SubVec8_Stages1"
CB_D32="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[32]_Value_K[16]C[256]D[32]_SubVec1_Stages1"
OUT=results/specdec/rq
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 50 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/rq_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/rq_$name.log)"
}
ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --gpus $gpus --out $OUT/$name.json "$@" > $LOG/rq_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/rq_$name.log)"
}

d16() {
    lane_a() { for g in 1 2 4; do run "$1" d16_m4_g$g --rep rq --codebook "$CB_D16" --m 4 --gamma $g; done
               ppl "$1" ppl_d16_t8 --rep rq --codebook "$CB_D16" --m 4 --target full --target-stages 8; }
    lane_b() { for g in 1 2 4; do run "$1" d16_m6_g$g --rep rq --codebook "$CB_D16" --m 6 --gamma $g; done
               ppl "$1" ppl_d16_t16 --rep rq --codebook "$CB_D16" --m 4 --target full --target-stages 16; }
    lane_c() { for g in 1 2 4; do run "$1" d16_m8_g$g --rep rq --codebook "$CB_D16" --m 8 --gamma $g; done
               for g in 1 2 4; do run "$1" kv4_tfp_g$g --rep quantspec --kv-bits 4 --target fp16 --gamma $g; done
               for g in 1 2 4; do run "$1" d16_m4_t8_g$g --rep rq --codebook "$CB_D16" --m 4 --target full --target-stages 8 --gamma $g; done; }
    lane_a "$1" & lane_b "$2" & lane_c "$3" & wait
}
d32() {
    lane_a() { for g in 1 2 4; do run "$1" d32_m8_g$g --rep rq --codebook "$CB_D32" --m 8 --gamma $g; done
               ppl "$1" ppl_d32_t16 --rep rq --codebook "$CB_D32" --m 8 --target full --target-stages 16; }
    lane_b() { for g in 1 2 4; do run "$1" d32_m12_g$g --rep rq --codebook "$CB_D32" --m 12 --gamma $g; done; }
    lane_c() { for g in 1 2 4; do run "$1" d32_m8_t16_g$g --rep rq --codebook "$CB_D32" --m 8 --target full --target-stages 16 --gamma $g; done; }
    if [ "$3" = "-" ]; then lane_a "$1" & { lane_b "$2"; lane_c "$2"; } & wait  # two lanes only
    else lane_a "$1" & lane_b "$2" & lane_c "$3" & wait; fi
}

sv8() {  # head-shared shape codebooks (SubVec8), same grid as d16; a lane given as "-" is skipped
    lane_a() { for g in 1 2 4; do run "$1" sv8_m4_g$g --rep rq --codebook "$CB_SV8" --m 4 --gamma $g; done
               ppl "$1" ppl_sv8_t8 --ref --rep rq --codebook "$CB_SV8" --m 4 --target full --target-stages 8
               ppl "$1" ppl_sv8_t16 --ref --rep rq --codebook "$CB_SV8" --m 4 --target full --target-stages 16; }
    lane_b() { for g in 1 2 4; do run "$1" sv8_m4_t8_g$g --rep rq --codebook "$CB_SV8" --m 4 --target full --target-stages 8 --gamma $g; done; }
    lane_c() { for g in 1 2 4; do run "$1" sv8_m6_g$g --rep rq --codebook "$CB_SV8" --m 6 --gamma $g; done
               for g in 1 2 4; do run "$1" sv8_m8_g$g --rep rq --codebook "$CB_SV8" --m 8 --gamma $g; done; }
    [ "$1" != "-" ] && lane_a "$1" &
    [ "$2" != "-" ] && lane_b "$2" &
    [ "$3" != "-" ] && lane_c "$3" &
    wait
}

which=$1; shift
$which "${1:-0 2}" "${2:-3 4}" "${3:-5 6}"
echo ALL_DONE
