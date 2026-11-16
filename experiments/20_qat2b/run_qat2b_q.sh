#!/usr/bin/env bash
# E5 qat2b queue worker: target-side QAT of codebook value formats on the Llama-3.1-8B 2-bit nested ckpts (kd == kt).
# Student = draft copy of ALL kt codebooks with STE fake-quant at <fmtspec> (train_draft_codebooks.py --qat_fmt /
# --qat_fmt_stages), init = the fmt2b rounding-only copy A_<label>.pt; teacher = unmodified bf16-codebook target
# (set_k(use_draft=False), checked bit-exact by --check_teacher). Codes frozen; codebooks + per-row scales trained.
# Budget = Llama B-draft runs (results_accept/llama31_c1_uniform_kd*/train_B_bf16): TV loss, 300 steps x batch 4 x 512
# tok, pile_val_heldout nsamples 1024; Adam lr 1e-5 (B-draft used 1e-4; a 40-step probe, lrprobe/, showed 1e-4 raises
# train-batch TV from the rounding init for the first ~20 steps, 1e-5 lowers it). Eval = fmt2b flags (draft_target_eval --n_windows 40 --no_fp16).
# Queue lines: <ckpt>:<L>:<kd>:<label>:<fmtspec>  (fmtspec = one format or a comma list of kd per-stage formats).
# Several GPUs share one queue; an item is claimed atomically with mkdir <OUT>/lock/<label>.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/20_qat2b/run_qat2b_q.sh <queue file>
set -euo pipefail
Q=${1:?queue}; MODEL=/data/models/llama3.1_8b_instruct
STEPS=${STEPS:-300}; NS=${NS:-1024}; LR=${LR:-1e-5}
# QREF: --qat_refresh (10 = re-search the fake-quant scales every 10 steps and at save; 0 = search once at init, fixed
# grid through training and bake). VAR: output sub-dir of the variant ("" for QREF=10, "fixgrid/" for QREF=0).
QREF=${QREF:-10}; VAR=${VAR:-}
cd /hai/home/lsh/SpecRQ
while IFS=: read -r cfg L kd label spec; do
  [[ -z "$cfg" || "$cfg" == \#* ]] && continue
  CKPT=/data/lsh/ckpts/weightvq/dsweep_llama/$cfg
  INIT=/data/lsh/results_accept/fmt2b/${cfg}_kd$kd/draft_q/A_${label}.pt
  OUT=/data/lsh/results_accept/qat2b/${VAR}${cfg}_kd$kd; mkdir -p $OUT/eval $OUT/lock
  TR=$OUT/train_$label; js=$OUT/eval/Q_${label}.json
  [[ -e $js ]] && continue
  mkdir $OUT/lock/$label 2>/dev/null || continue
  echo "$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)" > $OUT/lock/$label/owner
  if [[ $spec == *,* ]]; then FA="--qat_fmt_stages $spec"; else FA="--qat_fmt $spec"; fi
  echo "[qat2b] $cfg L=$L kd=$kd QAT $label ($spec) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
  [[ -e $TR/draft_codebooks.pt ]] || /usr/bin/time -f "[mem] train maxrss_kb=%M wall_s=%e" \
    python experiments/06_accept/train_draft_codebooks.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
    --div tv --steps $STEPS --batch 4 --micro_batch 1 --lr $LR --nsamples $NS --seqlen 512 \
    --codes_uint8 --checkpoint_linears --check_teacher $FA --qat_gran codebook --qat_refresh $QREF --init_draft $INIT --out $TR
  /usr/bin/time -f "[mem] draft_target_eval maxrss_kb=%M wall_s=%e" \
    python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kd $kd --kt $L \
    --n_windows 40 --codes_uint8 --no_fp16 --draft $TR/draft_codebooks.pt --out $js
done < <(cat "$Q")
echo "[qat2b] worker done $(date -Is)"
