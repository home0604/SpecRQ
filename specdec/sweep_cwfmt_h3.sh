#!/bin/bash
# H3 preview on D32 (post-hoc): prefix stages in a low-precision format, refinement stages fp16, target t16 (4 bit).
# Plus window check: exact recent window off (--rq-window 0) for fp16 vs nf4 at t8.
# Usage: bash specdec/sweep_cwfmt_h3.sh "3 4" "5 6"
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
RQ="--rep rq --codebook $CB_D32 --m 4 --target full"
lane1() {
    for f in nf4 int3; do for s in 4 8; do
        ppl "$1" d32_t16_${f}x${s} $RQ --target-stages 16 --cw-format $f --cw-stages $s; done; done
}
lane2() {
    ppl "$1" d32_t8_w0 $RQ --target-stages 8 --rq-window 0
    ppl "$1" d32_t8_nf4_w0 $RQ --target-stages 8 --cw-format nf4 --rq-window 0
    ppl "$1" d32_t8_int3_w0 $RQ --target-stages 8 --cw-format int3 --rq-window 0
}
lane1 "${1:-3 4}" & lane2 "${2:-5 6}" & wait
echo ALL_DONE
