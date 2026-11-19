#!/usr/bin/env bash
# E7 magicvq v8 (1-bit drafts): nested-prefix codebook t12w (rqd_train.py --nest 1: attn loss of stage 1 alone + of
# stages 1+2, same recipe as t2w: --level 2 --R 128 --steps 300 --lr 1e-3, from the greedy 'p' codebook), layers split
# over GPUs 4-7, merge, then held-out diagnostics at level 1 (R 128) and level 2 for def / t1w / t2w / t12w.
cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CBD=/data/lsh/results_accept/magicvq/cb
T="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/rqd_train.py --R 128"
i=0
for lo in 0 8 16 24; do
  CUDA_VISIBLE_DEVICES=$((4 + i)) $T --level 2 --tag t12w --nest 1 --layers $lo $((lo + 8)) > /data/lsh/logs/v8_train_L$lo.log 2>&1 &
  i=$((i + 1))
done
wait
cat /data/lsh/logs/v8_train_L*.log | grep -E "done|Error|Traceback"
CUDA_VISIBLE_DEVICES=4 $T --level 2 --tag t12w --nest 1 --merge || exit 1
CBS="def=$CBD/cb_d8K256x2_post_cn_greedy.pt t1w=$CBD/cb_d8K256x2_post_cn_t1w.pt t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t12w=$CBD/cb_d8K256x2_post_cn_t12w.pt"
for lv in 1 2; do
  i=0
  for lo in 0 8 16 24; do
    CUDA_VISIBLE_DEVICES=$((4 + i)) $T --level $lv --tag v8_lv${lv}_L$lo --layers $lo $((lo + 8)) --eval_cb $CBS \
      > /data/lsh/logs/v8_eval_lv${lv}_L$lo.log 2>&1 &
    i=$((i + 1))
  done
  wait
done
tail -n 1 /data/lsh/logs/v8_eval_lv*_L*.log
