#!/usr/bin/env bash
# PPL@4096 (wt2, c4) of an existing Llama d-sweep checkpoint in a fresh process (specrq/dbd/main.py --load, activations on
# the GPU; EVALK prefixes, default L). Used when the training process ran out of GPU memory during its own eval.
# Then writes <train log>.combined.log = training log + this log (save_dsweep.py takes the first RESULT per (dataset,k),
# so values already printed by the training process win) for run_llama_eval.sh (TRAIN_LOG=...combined.log).
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/15_dsweep/run_llama_ppl_load.sh <ckpt dir> <train log> <EVALK...>
set -euo pipefail
CKPT=${1:?ckpt}; TLOG=${2:?train log}; shift 2; EVALK=${*:?evalk}
MODEL=/data/models/llama3.1_8b_instruct
cd /hai/home/lsh/SpecRQ/specrq/dbd
python main.py $MODEL pile_val --load $CKPT --model_seqlen=4096 --devices cuda:0 --eval_num_codebooks $EVALK
cd /hai/home/lsh/SpecRQ
SELF=/data/lsh/logs/${RUNTMUX_NAME:-$(basename $CKPT)_ppl}.log
cat $TLOG $SELF > ${TLOG%.log}.combined.log 2>/dev/null || cat $TLOG > ${TLOG%.log}.combined.log
echo "[ppl-load] wrote ${TLOG%.log}.combined.log"
