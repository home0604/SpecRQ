#!/bin/bash
# Lower-precision nesting: does the bit-plane baseline hold up below INT8 targets?
# Stage 1 (default, target = exact original; isolates the draft's low-bit error):
#   kv{b}_tfp : KV draft = upper b bits of the b+b hierarchy, target KV exact, W16
#   w{b}_kvfp : separate RTN b-bit draft weights, target bf16, KV exact
#   w2_kv2tfp : both (W RTN2 + KV upper 2), target fully exact
# Stage 2 (target = full upper + lower reconstruction, i.e. how much the residual recovers):
#   kv{b}_full: KV b+b, target 2b-bit (--target full);  n{2b}_kvfp : nested weights, target 2b-bit;  n4_kv2 : both
# Llama-2-7B-32K-Instruct, PG-19 8k, 50 prompts, batch 1, γ ∈ {1,2,4}. 8-bit references: results/specdec/baseline50.
# Hierarchy scale: upper = RTN over the full range, lower = mid-rise inside one upper cell (QuantSpecRep scale="rtn").
# Runs with QuantSpec's own formula (only sound at 4 + 4) are kept in results/specdec/lowbit/_paper_scale/.
# Usage: bash specdec/sweep_lowbit.sh "1 2"   (one lane)
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
OUT=results/specdec/lowbit
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG
COMMON="--model /data/models/llama2_7b_32k_instruct --dataset pg19 --n-new 90 --num-prompts 50 --ctx 8192"

run() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.run $COMMON --gpus $gpus --out $OUT/$name.json "$@" > $LOG/lb_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/lb_$name.log)"
}
ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --gpus $gpus --out $OUT/$name.json "$@" > $LOG/lb_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/lb_$name.log)"
}

stage1() {
    for g in 1 2 4; do run "$1" kv2_tfp_g$g --rep quantspec --kv-bits 2 --target fp16 --draft-wbits 16 --gamma $g; done
    for g in 1 2 4; do run "$1" kv3_tfp_g$g --rep quantspec --kv-bits 3 --target fp16 --draft-wbits 16 --gamma $g; done
    for g in 1 2 4; do run "$1" w2_kvfp_g$g --rep fp16 --draft-wbits 2 --gamma $g; done
    for g in 1 2 4; do run "$1" w3_kvfp_g$g --rep fp16 --draft-wbits 3 --gamma $g; done
    for g in 1 2 4; do run "$1" w2_kv2tfp_g$g --rep quantspec --kv-bits 2 --target fp16 --draft-wbits 2 --gamma $g; done
}
stage2() {
    ppl "$1" ppl_kv3_full --rep quantspec --kv-bits 3 --target full
    ppl "$1" ppl_kv2_full --rep quantspec --kv-bits 2 --target full
    ppl "$1" ppl_n6 --rep fp16 --draft-wbits n6
    ppl "$1" ppl_n4 --rep fp16 --draft-wbits n4
    ppl "$1" ppl_n4_kv2_full --rep quantspec --kv-bits 2 --target full --draft-wbits n4
    for g in 1 2 4; do run "$1" kv3_full_g$g --rep quantspec --kv-bits 3 --target full --draft-wbits 16 --gamma $g; done
    for g in 1 2 4; do run "$1" kv2_full_g$g --rep quantspec --kv-bits 2 --target full --draft-wbits 16 --gamma $g; done
    for g in 1 2 4; do run "$1" n6_kvfp_g$g --rep fp16 --draft-wbits n6 --gamma $g; done
    for g in 1 2 4; do run "$1" n4_kvfp_g$g --rep fp16 --draft-wbits n4 --gamma $g; done
    for g in 1 2 4; do run "$1" n4_kv2_full_g$g --rep quantspec --kv-bits 2 --target full --draft-wbits n4 --gamma $g; done
}

stage1 "${1:-1 2}"
stage2 "${1:-1 2}"
echo ALL_DONE
