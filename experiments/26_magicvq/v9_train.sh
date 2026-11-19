#!/usr/bin/env bash
# E7 magicvq v9: retry of the nested-prefix KV draft codebook (t12w_v2*). rqd_train.py, R 128, attention loss.
#   w03 / w10 : warm start from t2w, nested loss  W * attn(stage 1) + attn(stages 1+2), W = 0.3 / 1.0
#   fz        : stage 1 + mu / sd frozen = t1w, stage 2 trained with attn(stages 1+2) (dedicated prefix + residual)
#   sc<W>     : from the greedy 'p' init (as v8 t12w), nested loss, longer
#   nn1 / w01 : warm t2w, per-term normalised W = 1 (W L1/L1_0 + L2/L2_0) / raw W = 0.1
#   lag3      : warm t2w, min L2 s.t. L1 <= 1.03 L1_t1w (same batch), Lagrange multiplier W from 0, eta 0.1
# Usage: v9_train.sh pilot|pilot2 STEPS          (variants w03 w10 fz on the pilot layers, 2 procs per GPU on 4-7)
#        v9_train.sh full TAG STEPS ARGS  (all 32 layers of one variant, 2 procs per GPU on 4-7, then merge)
#        v9_train.sh eval LV TAG CB=PATH ...   (held-out eval, all 32 layers, 8 procs on 4-7)
cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CBD=/data/lsh/results_accept/magicvq/cb
T="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/rqd_train.py --R 128"
W="--init $CBD/cb_d8K256x2_post_cn_t2w.pt"
FZ="--init $CBD/cb_d8K256x2_post_cn_t1w.pt --freeze_s1"
case $1 in
  pilot)
    S=$2; PL1="1 8 16 24"; PL2="4 12 20 28"
    CUDA_VISIBLE_DEVICES=4 $T --level 2 --steps $S --tag t12w_v2w03 --nest 0.3 $W --layer_list $PL1 > /data/lsh/logs/v9_pilot_w03a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 $T --level 2 --steps $S --tag t12w_v2w03 --nest 0.3 $W --layer_list $PL2 > /data/lsh/logs/v9_pilot_w03b.log 2>&1 &
    CUDA_VISIBLE_DEVICES=5 $T --level 2 --steps $S --tag t12w_v2w10 --nest 1.0 $W --layer_list $PL1 > /data/lsh/logs/v9_pilot_w10a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=5 $T --level 2 --steps $S --tag t12w_v2w10 --nest 1.0 $W --layer_list $PL2 > /data/lsh/logs/v9_pilot_w10b.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6 $T --level 2 --steps $S --tag t12w_v2fz $FZ --layer_list $PL1 > /data/lsh/logs/v9_pilot_fza.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 $T --level 2 --steps $S --tag t12w_v2fz $FZ --layer_list $PL2 > /data/lsh/logs/v9_pilot_fzb.log 2>&1 &
    wait
    grep -hE "^\{|Error|Traceback" /data/lsh/logs/v9_pilot_*.log ;;
  pilot2)  # supervisor follow-up: per-term normalised nested loss (nn1: W = 1) and small raw W (w01), warm start t2w
    S=$2; PL1="1 8 16 24"; PL2="4 12 20 28"
    CUDA_VISIBLE_DEVICES=6 $T --level 2 --steps $S --tag t12w_v2nn1 --nest 1.0 --nest_norm $W --layer_list $PL1 > /data/lsh/logs/v9_pilot2_nn1a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6 $T --level 2 --steps $S --tag t12w_v2nn1 --nest 1.0 --nest_norm $W --layer_list $PL2 > /data/lsh/logs/v9_pilot2_nn1b.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 $T --level 2 --steps $S --tag t12w_v2w01 --nest 0.1 $W --layer_list $PL1 > /data/lsh/logs/v9_pilot2_w01a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 $T --level 2 --steps $S --tag t12w_v2w01 --nest 0.1 $W --layer_list $PL2 > /data/lsh/logs/v9_pilot2_w01b.log 2>&1 &
    wait
    grep -hE "^\{|Error|Traceback" /data/lsh/logs/v9_pilot2_*.log ;;
  pilot3)  # supervisor priority update: constrained joint (min L2 s.t. L1 <= 1.03 L1_t1w, Lagrangian W), warm start t2w
    S=$2; PL1="1 8 16 24"; PL2="4 12 20 28"
    CUDA_VISIBLE_DEVICES=4 $T --level 2 --steps $S --tag t12w_v2lag3 --nest 0 --nest_lag 0.03 $W --layer_list $PL1 > /data/lsh/logs/v9_pilot3_lag3a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=5 $T --level 2 --steps $S --tag t12w_v2lag3 --nest 0 --nest_lag 0.03 $W --layer_list $PL2 > /data/lsh/logs/v9_pilot3_lag3b.log 2>&1 &
    wait
    grep -hE "^\{|Error|Traceback" /data/lsh/logs/v9_pilot3_*.log ;;
  t1long)  # equal-compute L1 baseline: t1w continued at level 1 for the same extra steps (pilot layers, one GPU)
    S=$2; G=$3
    CUDA_VISIBLE_DEVICES=$G $T --level 1 --steps $S --tag t1w_long --init $CBD/cb_d8K256x2_post_cn_t1w.pt --layer_list 1 4 8 12 > /data/lsh/logs/v9_t1long_a.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$G $T --level 1 --steps $S --tag t1w_long --init $CBD/cb_d8K256x2_post_cn_t1w.pt --layer_list 16 20 24 28 > /data/lsh/logs/v9_t1long_b.log 2>&1 &
    wait
    grep -hE "^\{|Error|Traceback" /data/lsh/logs/v9_t1long_*.log ;;
  peval)  # pilot held-out eval: partial codebooks (pilot layers from the train dir, rest = init), L1 and L2, pilot layers
    G=$2; shift 2; CBS=""
    for t in "$@"; do  # t = TAG:INIT
      python experiments/26_magicvq/v9_partial_merge.py ${t%%:*} ${t#*:} || exit 1
      CBS="$CBS ${t%%:*}=/data/lsh/temp/v9/${t%%:*}_partial.pt"
    done
    for lv in 1 2; do
      CUDA_VISIBLE_DEVICES=$G $T --level $lv --tag v9p_$(date +%H%M)_lv$lv --layer_list 1 4 8 12 16 20 24 28 --eval_cb $CBS \
        > /data/lsh/logs/v9_peval_lv$lv.$(date +%H%M%S).log 2>&1 &
    done
    wait ;;
  full)
    TAG=$2; S=$3; shift 3
    i=0
    for lo in 0 4 8 12 16 20 24 28; do
      CUDA_VISIBLE_DEVICES=$((4 + i / 2)) $T --level 2 --steps $S --tag $TAG "$@" --layers $lo $((lo + 4)) \
        > /data/lsh/logs/v9_full_${TAG}_L$lo.log 2>&1 &
      i=$((i + 1))
    done
    wait
    grep -hE "^\{|Error|Traceback" /data/lsh/logs/v9_full_${TAG}_L*.log
    $T --level 2 --tag $TAG --merge "$@" ;;
  eval)
    LV=$2; TAG=$3; shift 3
    i=0
    for lo in 0 4 8 12 16 20 24 28; do
      CUDA_VISIBLE_DEVICES=$((4 + i / 2)) $T --level $LV --tag ${TAG}_L$lo --layers $lo $((lo + 4)) --eval_cb "$@" \
        > /data/lsh/logs/v9_eval_${TAG}_L$lo.log 2>&1 &
      i=$((i + 1))
    done
    wait
    tail -qn 1 /data/lsh/logs/v9_eval_${TAG}_L*.log ;;
esac
