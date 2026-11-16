#!/usr/bin/env bash
# E5 sub1b8: draft A / B evaluation of one 2.0-bpw Llama-3.1-8B-Instruct ckpt (protocol of
# experiments/08_llama_full/run_llama_accept.sh + experiments/15_dsweep/run_llama_eval.sh):
#   A<kd>: draft_target_eval.py kt=L kd (target-copy prefix draft): TV/top-1 + greedy tau g1/2/4 (8 chat + 8 wt2 prompts x
#          128 tok); the first kd in the list also measures target fidelity vs bf16 (reference on GPU+CPU, REF_MAXMEM)
#   B<kd>: train_draft_codebooks.py (TV, pile_val_heldout 1024x512, 300 steps, batch 4, lr 1e-4 = Llama C1 B runs;
#          teacher = full target, use_draft=False path, --check_teacher) then draft_target_eval --draft
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/23_sub1b8/eval_sub1b.sh <ckpt> <L> <outdir> "<steps>"
#   steps e.g. "A1 A2 B1 B2"
set -euo pipefail
CKPT=${1:?ckpt}; L=${2:?L}; OUT=${3:?outdir}; STEPS=${4:?steps}
MODEL=/data/models/llama3.1_8b_instruct
REF_MAXMEM=${REF_MAXMEM:-0:5GiB,cpu:40GiB}
cd /hai/home/lsh/SpecRQ
EV="python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kt $L --n_windows 40 --codes_uint8"
TR="python experiments/06_accept/train_draft_codebooks.py --model $MODEL --ckpt $CKPT --kt $L --steps 300 --batch 4 --nsamples 1024 --seqlen 512 --lr 1e-4 --codes_uint8 --checkpoint_linears --micro_batch 1 --check_teacher"
REFDONE=${REFDONE:-0}
for s in $STEPS; do
  echo "[sub1b-eval] $s $(date -Is)"
  case $s in
    A*) kd=${s#A}; o=$OUT/kd$kd; mkdir -p $o/eval
        if [[ -e $o/eval/A_bf16.json ]]; then echo "exists $o/eval/A_bf16.json"
        elif [[ $REFDONE == 0 ]]; then $EV --kd $kd --ref_max_memory $REF_MAXMEM --out $o/eval/A_bf16.json
        else $EV --kd $kd --no_fp16 --out $o/eval/A_bf16.json; fi
        REFDONE=1 ;;
    B*) kd=${s#B}; o=$OUT/kd$kd; mkdir -p $o/eval
        [[ -e $o/train_B_bf16/draft_codebooks.pt ]] || { /usr/bin/time -v $TR --kd $kd --out $o/train_B_bf16; }
        [[ -e $o/eval/B_bf16.json ]] || $EV --kd $kd --no_fp16 --draft $o/train_B_bf16/draft_codebooks.pt --out $o/eval/B_bf16.json ;;
    *) echo "bad step $s"; exit 2;;
  esac
done
echo "[sub1b-eval] done $(date -Is)"
