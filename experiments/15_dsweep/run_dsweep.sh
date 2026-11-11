#!/usr/bin/env bash
# Claim-1 d-sweep (E5): same index bits, larger d (= more K=256 stages). Recipe = experiments/10_screen/run_screen.sh
# (Hadamard, wikitext2 1024x4096 calib, val 32, accum fp64, AQ max_epochs 1000 / rel_tol 0.01, no block fine-tune),
# only the codebook_weights differ:  T = target-only (alpha = 0..0,1), N = nested uniform alpha 1/L,
#   N_<tag> = custom alphas from env CBW="a1 .. aL" (e.g. N_k4 = 0 0 0 .5 0 0 0 .5).
#   1) quantize + PPL@4096 wt2/c4 for every prefix k=L..1   2) draft_target_eval kd=1 kt=L (target KL/top-1 vs FP16)
#   2b) env KDS="2 4": per draft prefix kd, draft_target_eval kd kt=L (copy draft A; TV, top-1, greedy tau g1/2/4;
#       --no_fp16, as the screening's A eval minus the FP16 reference) -> $OUT/kd<kd>/eval/A_bf16.json
#   3) results/E5/qwen05_dsweep_<cfg>_<obj><RUN_SUFFIX>.json (env RUN_SUFFIX, e.g. _kd24 for an eval-only add-on of a
#      finished run; then env TRAIN_LOG must point at the training log)
# LAYERS=n env: quantize only the first n layers (timing probe; no eval, separate ckpt dir).
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/15_dsweep/run_dsweep.sh <cfg> <d> <nbits> <L> <T|N>
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}; OBJ=${5:?T|N}
MODEL=/data/models/qwen2.5_0.5b_instruct
ROOT=/data/lsh/ckpts/weightvq/dsweep
cd /hai/home/lsh/SpecRQ
CBW=${CBW:-}; CBW=${CBW//,/ }; KDS=${KDS:-}; KDS=${KDS//,/ }  # queue lines pass comma lists
case $OBJ in
  T) CBW=$(python3 -c "print(' '.join(['0']*($L-1)+['1']))");;
  N) CBW=$(python3 -c "print(' '.join(['%.10g' % (1/$L)]*$L))");;
  N_*) [[ -n "${CBW:-}" ]] || { echo "obj $OBJ needs env CBW"; exit 2; }
       [[ $(wc -w <<<"$CBW") == "$L" ]] || { echo "CBW needs $L values"; exit 2; };;
  *) echo "obj must be T, N or N_<tag>"; exit 2;;
esac
EVALK=$(seq $L -1 1 | tr '\n' ' ')
# activations stay on the GPU by default (2 x 1024x4096x896 bf16 = 15 GB): host RAM ~19 -> few GiB (user cgroup OOM
# killed d8K256L2_T with offload on 2026-11-10). Same tensors/ops, only placement differs. OFFLOAD=1 = screen recipe.
OFFL=$([[ "${OFFLOAD:-0}" == 1 ]] && echo --offload_activations || true)
if [[ -n "${LAYERS:-}" ]]; then
  CKPT=$ROOT/probe/${CFG}_${OBJ}_l${LAYERS}; rm -rf "$CKPT"; mkdir -p "$(dirname "$CKPT")"
  echo "[dsweep] PROBE $CFG obj=$OBJ layers=$LAYERS CBW=$CBW $(date -Is)"
  (cd specrq/dbd && /usr/bin/time -v python main.py $MODEL wikitext2 \
    --nsamples=1024 --val_size=32 --model_seqlen=4096 \
    --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
    --codebook_weights $CBW --hadamard --accum_dtype float64 --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
    --local_batch_size=1 $OFFL --devices cuda:0 --quantize_num_layers $LAYERS --save $CKPT)
  echo "[dsweep] PROBE done $(date -Is)"; exit 0
fi
CKPT=$ROOT/${CFG}_${OBJ}
OUT=/data/lsh/results_accept/dsweep/${CFG}_${OBJ}
mkdir -p "$ROOT" "$OUT/eval"
if [[ ! -e "$CKPT/args.pt" ]]; then
  [[ -e "$CKPT" ]] && { echo "partial $CKPT exists; refusing"; exit 1; }
  echo "[dsweep] train $CFG d=$D nbits=$NB L=$L obj=$OBJ CBW=$CBW $(date -Is)"
  (cd specrq/dbd && /usr/bin/time -v python main.py $MODEL wikitext2 \
    --nsamples=1024 --val_size=32 --model_seqlen=4096 \
    --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
    --codebook_weights $CBW --hadamard --accum_dtype float64 --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
    --local_batch_size=1 $OFFL --devices cuda:0 \
    --eval_num_codebooks $EVALK --save $CKPT)
  echo "[dsweep] train done $(date -Is)"
fi
echo "[dsweep] eval kd=1 kt=$L $(date -Is)"
[[ -e $OUT/eval/A_bf16.json ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT \
  --kt $L --kd 1 --n_windows 40 --out $OUT/eval/A_bf16.json
for kd in ${KDS:-}; do
  mkdir -p $OUT/kd$kd/eval
  echo "[dsweep] draft eval kd=$kd kt=$L $(date -Is)"
  [[ -e $OUT/kd$kd/eval/A_bf16.json ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT \
    --kt $L --kd $kd --n_windows 40 --no_fp16 --out $OUT/kd$kd/eval/A_bf16.json
done
python experiments/15_dsweep/save_dsweep.py --cfg $CFG --d $D --nbits $NB --L $L --obj $OBJ --cbw "$CBW" --ckpt $CKPT \
  --eval $OUT/eval/A_bf16.json --log ${TRAIN_LOG:-/data/lsh/logs/${DSWEEP_JOB:-e5_dsweep_${CFG}_${OBJ}}.log} \
  --kds "${KDS:-}" --kd_root $OUT --run_suffix "${RUN_SUFFIX:-}"
echo "[dsweep] done $CFG $OBJ $(date -Is)"
