#!/bin/bash
# E1 (pure format effect): post-hoc shape-codeword format on the trained D32 codebook (all stages, gains fp16),
# teacher-forced target quality vs the exact model (PPL / KL / top-1, specdec.ppl --ref), PG-19 8k, 10 prompts.
# t = 4 / 8 / 16 stages = 1 / 2 / 4 bit (t=4 is also the 1-bit draft prefix).  fp16 = as stored (target_curve/d32_t*).
# Usage: bash specdec/sweep_cwfmt.sh "2 3" "4 5" "6 7"
set -u
cd "$(dirname "$0")/.."
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
CB_D32="/data/lsh/ckpts/wikitext/both_quant_Key_K[16]C[256]D[32]_Value_K[16]C[256]D[32]_SubVec1_Stages1"
OUT=results/specdec/cwfmt
LOG=/data/lsh/logs/specdec
mkdir -p $OUT $LOG

ppl() {  # gpus name args...
    local gpus=$1 name=$2; shift 2
    [ -f $OUT/$name.json ] && { echo "skip $name"; return; }
    python -m specdec.ppl --ctx 8192 --ref --gpus $gpus --out $OUT/$name.json "$@" > $LOG/cw_$name.log 2>&1 \
        && echo "done $name" || echo "FAILED $name (see $LOG/cw_$name.log)"
}
lane() {  # gpus t
    for f in fp8 int8 nf4 fp4 int4 int3; do
        ppl "$1" d32_t$2_$f --rep rq --codebook "$CB_D32" --m 4 --target full --target-stages $2 --cw-format $f
    done
}

lane "${1:-2 3}" 4 & lane "${2:-4 5}" 8 & lane "${3:-6 7}" 16 & wait
echo ALL_DONE
