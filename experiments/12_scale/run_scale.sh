#!/usr/bin/env bash
# E5 scale check: does the 1-bit prefix of a nested 2-bit target work as a draft on a large model?
# Same no-FT screening recipe as experiments/10_screen/run_screen.sh except: L=2 (d8, K=256 -> target 2 bit, kd1 = 1 bit),
# uniform alpha [0.5,0.5], Hadamard, accum fp32, wikitext2 ${NS}x${SEQ} calibration (val 32 of them), --resume so a
# partial run (e.g. the layer-0 timing run, QNL=1) is continued, AQ epoch cap MAXEP (default 1000 = DbD default), and
# large-model memory flags when BIG=1.
#   1) quantize (+ PPL @4096 wt2/c4 for k=2,1)  2) FP16 PPL @4096 (main.py --no_quant)
#   3) kd1 A (target-copy draft + FP16 fidelity), B (TV-trained draft, pile_val_heldout 1024x512, 300 steps, batch 4)
# Usage: tools/run_tmux.sh <name> <gpu[,gpu2]> nestvq -- <cpu prefix> env TAG=.. MODEL=.. [BIG=1] [QNL=1] \
#          bash experiments/12_scale/run_scale.sh
#   BIG=1: second visible GPU (cuda:1) + CPU hold the FP16 reference; codes uint8; decoder-layer checkpointing and
#          micro-batch 1 in TV training (same update as batch 4).
set -euo pipefail
TAG=${TAG:?tag}; MODEL=${MODEL:?model}; NS=${NS:-512}; SEQ=${SEQ:-4096}; QNL=${QNL:-}; BIG=${BIG:-0}; MAXEP=${MAXEP:-1000}
L=2; D=8; NB=8; KD=1
CKPT=/data/lsh/ckpts/weightvq/scale/${TAG}_d8K256L2_noft
OUT=/data/lsh/results_accept/scale/${TAG}_d8K256L2_noft
cd /hai/home/lsh/SpecRQ
mkdir -p "$(dirname "$CKPT")" "$OUT"
if [[ ! -e "$CKPT/args.pt" ]]; then
  echo "[scale] train $TAG ns=$NS seq=$SEQ maxep=$MAXEP qnl=${QNL:-all} $(date -Is)"
  (cd specrq/dbd && /usr/bin/time -v python main.py $MODEL wikitext2 \
    --nsamples=$NS --val_size=32 --model_seqlen=$SEQ \
    --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
    --codebook_weights 0.5 0.5 --hadamard --accum_dtype float32 \
    --relative_mse_tolerance=0.01 --max_epochs=$MAXEP \
    --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
    --local_batch_size=1 --offload_activations --devices cuda:0 --resume \
    ${QNL:+--quantize_num_layers $QNL} \
    --eval_num_codebooks 2 1 --save $CKPT)
  echo "[scale] train done $(date -Is)"
  [[ -n "$QNL" ]] && { echo "[scale] QNL=$QNL partial run; stop"; exit 0; }
fi
# quantized PPL @4096 (k=2,1): only if the training run's in-process eval did not finish (e.g. c4 load failure)
JOBLOG=/data/lsh/logs/${SCALE_JOB:-none}.log
if ! grep -qa "RESULT dataset=c4 k=1 " "$JOBLOG" 2>/dev/null && [[ ! -e $OUT/q_ppl.log ]]; then
  echo "[scale] quantized ppl (--load) $(date -Is)"
  (cd specrq/dbd && python main.py $MODEL wikitext2 --load $CKPT --model_seqlen=4096 --offload_activations \
    --devices cuda:0 --eval_num_codebooks 2 1) 2>&1 | tee $OUT/q_ppl.log.tmp
  mv $OUT/q_ppl.log.tmp $OUT/q_ppl.log
fi
if [[ ! -e $OUT/fp16_ppl.log ]]; then
  echo "[scale] fp16 ppl $(date -Is)"
  (cd specrq/dbd && python main.py $MODEL wikitext2 --no_quant --model_seqlen=4096 --offload_activations \
    --devices cuda:0) 2>&1 | tee $OUT/fp16_ppl.log.tmp
  mv $OUT/fp16_ppl.log.tmp $OUT/fp16_ppl.log
fi
EVX=""; TRX=""
if [[ "$BIG" == 1 ]]; then
  EVX="--codes_uint8 --ref_max_memory 1:21GiB,cpu:150GiB"
  TRX="--codes_uint8 --checkpoint_layers --micro_batch 1"
fi
EV="python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT --kt $L --n_windows 40 $EVX"
TR="python experiments/06_accept/train_draft_codebooks.py --model $MODEL --ckpt $CKPT --kt $L --steps 300 --batch 4 --nsamples 1024 --seqlen 512 $TRX"
o=$OUT/kd$KD; mkdir -p $o/eval
echo "[scale] kd=$KD A $(date -Is)"
[[ -e $o/eval/A_bf16.json ]] || $EV --kd $KD --out $o/eval/A_bf16.json
echo "[scale] kd=$KD B train $(date -Is)"
[[ -e $o/train_B_bf16/draft_codebooks.pt ]] || $TR --kd $KD --out $o/train_B_bf16
echo "[scale] kd=$KD B eval $(date -Is)"
[[ -e $o/eval/B_bf16.json ]] || $EV --kd $KD --no_fp16 --draft $o/train_B_bf16/draft_codebooks.pt --out $o/eval/B_bf16.json
echo "[scale] done $TAG $(date -Is)"
