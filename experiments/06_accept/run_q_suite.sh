#!/usr/bin/env bash
# Phase 5 q-suite on one nested DbD ckpt (fixed C1: d=8, K=256, M=5, draft k=3):
#   A    baseline acceptance/fidelity (draft = target codebooks, q=16 bf16)
#   D(i) rounding-only draft codebooks: int8 / int4 / fp8_e4m3 x {codebook, dim} scales
#   B    draft-only codebook training (TV, pile_val_heldout, fixed steps) from the bf16 copy
#   D(ii) the same training started from each rounded copy, kept q-bit by STE fake-quant (scales re-searched every 10
#        steps, final values baked to the q-bit grid)
# KD env (default 3) sets the draft prefix k_d for eval and draft-codebook training.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- bash experiments/06_accept/run_q_suite.sh <ckpt> <outdir> [stages]
set -euo pipefail
CKPT=${1:?ckpt}; OUT=${2:?outdir}; STAGES=${3:-A,Di,B,Dii}
MODEL=${MODEL:-/data/models/qwen2.5_0.5b_instruct}
KD=${KD:-3}; STEPS=${STEPS:-300}; BATCH=${BATCH:-4}; NS=${NS:-1024}; SEQ=${SEQ:-512}; NWIN=${NWIN:-40}
EV="python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $KD --kt 5 --n_windows $NWIN"
mkdir -p $OUT/draft_q $OUT/eval
VARIANTS=${VARIANTS:-"int8_codebook int8_dim int4_codebook int4_dim fp8_e4m3_codebook fp8_e4m3_dim"}
has() { [[ ",$STAGES," == *",$1,"* ]]; }
if has A; then $EV --out $OUT/eval/A_bf16.json; fi
if has Di; then
  for v in $VARIANTS; do
    fmt=${v%_*}; gran=${v##*_}
    CUDA_VISIBLE_DEVICES="" python experiments/06_accept/quantize_draft.py --model $MODEL --ckpt $CKPT --fmt $fmt --gran $gran --out $OUT/draft_q/$v.pt
    $EV --no_fp16 --draft $OUT/draft_q/$v.pt --out $OUT/eval/Di_$v.json
  done
fi
TR="python experiments/06_accept/train_draft_codebooks.py --model $MODEL --ckpt $CKPT --kd $KD --steps $STEPS --batch $BATCH --nsamples $NS --seqlen $SEQ"
if has B; then
  $TR --out $OUT/train_B_bf16
  $EV --no_fp16 --draft $OUT/train_B_bf16/draft_codebooks.pt --out $OUT/eval/B_bf16.json
fi
if has Dii; then
  for v in $VARIANTS; do
    fmt=${v%_*}; gran=${v##*_}
    $TR --init_draft $OUT/draft_q/$v.pt --qat_fmt $fmt --qat_gran $gran --out $OUT/train_Dii_$v
    $EV --no_fp16 --draft $OUT/train_Dii_$v/draft_codebooks.pt --out $OUT/eval/Dii_$v.json
  done
fi
echo "[q-suite] done $STAGES"
