#!/usr/bin/env bash
# Llama-3.1-8B C1 acceptance evals (same protocol as the Qwen q-suite A/B), memory layout for 24 GB GPUs:
# codes stored uint8 (--codes_uint8, outputs unchanged), FP16 reference on a second visible GPU (cuda:1) and only for
# the kd listed in REFKD (target fidelity does not depend on kd).
# Steps (space-separated list, run in order): fp16ppl | A<kd> | B<kd>   e.g. "fp16ppl A3 A2 B2"
# Usage: tools/run_tmux.sh <name> <gpu[,gpu_ref]> nestvq -- <cpu prefix> bash experiments/08_llama_full/run_llama_accept.sh <ckpt> <outprefix> "<steps>" [REFKD]
set -euo pipefail
CKPT=${1:?ckpt}; OUTP=${2:?outprefix}; STEPS=${3:?steps}; REFKD=${4:-3}
MODEL=/data/models/llama3.1_8b_instruct
cd /hai/home/lsh/SpecRQ
EV="python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kt 5 --n_windows 40 --codes_uint8"
TR="python experiments/06_accept/train_draft_codebooks.py --model $MODEL --ckpt $CKPT --kt 5 --steps 300 --batch 4 --nsamples 1024 --seqlen 512 --codes_uint8 --checkpoint_linears --micro_batch 1"
for s in $STEPS; do
  echo "[llama-accept] $s $(date -Is)"
  case $s in
    fp16ppl) (cd specrq/dbd && python main.py $MODEL pile_val --no_quant --model_seqlen=4096 --devices cuda:0) ;;
    A*) kd=${s#A}; o=${OUTP}_kd$kd; mkdir -p $o/eval
        if [[ "$kd" == "$REFKD" ]]; then $EV --kd $kd --ref_device cuda:1 --out $o/eval/A_bf16.json
        else $EV --kd $kd --no_fp16 --out $o/eval/A_bf16.json; fi ;;
    B*) kd=${s#B}; o=${OUTP}_kd$kd; mkdir -p $o/eval
        $TR --kd $kd --out $o/train_B_bf16
        $EV --kd $kd --no_fp16 --draft $o/train_B_bf16/draft_codebooks.pt --out $o/eval/B_bf16.json ;;
  esac
done
echo "[llama-accept] done $(date -Is)"
