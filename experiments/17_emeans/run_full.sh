#!/usr/bin/env bash
# Full Qwen2.5-0.5B run, run_dsweep.sh recipe (Hadamard, wikitext2 1024x4096 calib, val 32, accum fp64, AQ rel_tol 0.01,
# no block FT, T obj), with a selectable codebook init and main.py --seed.
# Env: INIT=emeans (default; experiments/17_emeans/emeans_main.py, SPECRQ_EMEANS_EPOCHS default 20, e-means seed = SEED)
#        | torch (plain main.py = AQLM torch k-means)
#      SEED (default 0): main.py --seed = calibration sampling + val split + Hadamard signs + torch RNG (k-means init),
#        and the e-means seed.   SUFFIX (default _emeans): ckpt <cfg>_T<SUFFIX>, result qwen05_dsweep_<cfg>_T<SUFFIX>.json
#      Do not set HF_DATASETS_OFFLINE (main.py's c4 eval needs the hub lookup for the cached allenai/c4).
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- <cpu prefix> bash experiments/17_emeans/run_full.sh <cfg> <d> <nbits> <L>
#   (the training log is /data/lsh/logs/<name>.log; pass EMEANS_JOB=<name> if <name> != e5_emeans_full_<cfg>)
set -euo pipefail
CFG=${1:?cfg}; D=${2:?d}; NB=${3:?nbits}; L=${4:?L}; OBJ=T
INIT=${INIT:-emeans}; SEED=${SEED:-0}; SUFFIX=${SUFFIX:-_emeans}
export SPECRQ_EMEANS_EPOCHS=${SPECRQ_EMEANS_EPOCHS:-20} SPECRQ_EMEANS_SEED=$SEED
export PYTHONPATH=/data/lsh/temp/pylib_faiss116${PYTHONPATH:+:$PYTHONPATH}
case $INIT in
  emeans) MAINPY=../../experiments/17_emeans/emeans_main.py;;
  torch) MAINPY=main.py;;
  *) echo "INIT must be emeans or torch"; exit 2;;
esac
MODEL=/data/models/qwen2.5_0.5b_instruct
ROOT=/data/lsh/ckpts/weightvq/dsweep
cd /hai/home/lsh/SpecRQ
[[ -e results/E5/qwen05_dsweep_${CFG}_${OBJ}${SUFFIX}.json ]] && { echo "result for suffix $SUFFIX exists"; exit 1; }
CBW=$(python3 -c "print(' '.join(['0']*($L-1)+['1']))")
EVALK=$(seq $L -1 1 | tr '\n' ' ')
CKPT=$ROOT/${CFG}_${OBJ}${SUFFIX}
OUT=/data/lsh/results_accept/dsweep/${CFG}_${OBJ}${SUFFIX}
mkdir -p "$ROOT" "$OUT/eval"
if [[ ! -e "$CKPT/args.pt" ]]; then
  [[ -e "$CKPT" ]] && { echo "partial $CKPT exists; refusing"; exit 1; }
  echo "[full] train $CFG d=$D nbits=$NB L=$L CBW=$CBW init=$INIT seed=$SEED nepochs=$SPECRQ_EMEANS_EPOCHS $(date -Is)"
  (cd specrq/dbd && /usr/bin/time -v python $MAINPY $MODEL wikitext2 \
    --nsamples=1024 --val_size=32 --model_seqlen=4096 --seed $SEED \
    --num_codebooks=$L --nbits_per_codebook=$NB --in_group_size=$D --out_group_size=1 \
    --codebook_weights $CBW --hadamard --accum_dtype float64 --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 --finetune_max_epochs=0 --finetune_early_stop=3 --finetune_keep_best \
    --local_batch_size=1 --devices cuda:0 \
    --eval_num_codebooks $EVALK --save $CKPT)
  echo "[full] train done $(date -Is)"
fi
[[ -e $OUT/eval/A_bf16.json ]] || python experiments/06_accept/draft_target_eval.py --model $MODEL --ckpt $CKPT \
  --kt $L --kd 1 --n_windows 40 --out $OUT/eval/A_bf16.json
SAVE_OUT=$(python experiments/15_dsweep/save_dsweep.py --cfg $CFG --d $D --nbits $NB --L $L --obj $OBJ --cbw "$CBW" --ckpt $CKPT \
  --eval $OUT/eval/A_bf16.json --log ${TRAIN_LOG:-/data/lsh/logs/${EMEANS_JOB:-e5_emeans_full_${CFG}}.log} \
  --run_suffix "$SUFFIX")
echo "$SAVE_OUT"
P=$(grep -m1 '\.json$' <<<"$SAVE_OUT")
python3 - "$P" "$INIT" "$SEED" <<'PY'
import json, os, sys
p, init, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); r = json.load(open(p))
r["config"]["seed"] = seed
r["config"]["init"] = ((f"e-means (faiss 1.16.0 faiss.contrib.e_means.QuantizedEmeans, quantizer=None fp32, native torch "
                        f"backend on GPU, CUDA-graph replay off, nepochs={os.environ['SPECRQ_EMEANS_EPOCHS']}, seed {seed}) "
                        "replaces torch k-means in init_aq_kmeans; codes = exact nearest centroid") if init == "emeans"
                       else "torch k-means (AQLM default fit_kmeans, 100 iters)")
if seed != 0:
    r["config"]["setting"] += f"; main.py --seed {seed} (calib sampling, val split, Hadamard signs, init RNG)"
r["config"]["scripts"] += ["experiments/17_emeans/run_full.sh"] + (["experiments/17_emeans/emeans_main.py"] if init == "emeans" else [])
json.dump(r, open(p, "w"), indent=1)
print("annotated", p)
PY
echo "[full] done $CFG $(date -Is)"
