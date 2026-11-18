#!/usr/bin/env bash
# E7 magicvq v7mse: MSE-objective control codebook t2mw (same data / params / steps / lr / STE / RoPE offset / held-out
# rule as t2w: --level 2 --R 128 --steps 600 --lr 3e-3 --tf32), layers split over GPUs 4-7, then merge, then the
# held-out diagnostic (attention-output loss + K/V MSE) for default / t2mw / t2w.
cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CBD=/data/lsh/results_accept/magicvq/cb
T="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/rqd_train.py --level 2 --R 128 --steps 600 --lr 3e-3 --tf32"
i=0
for lo in 0 8 16 24; do
  CUDA_VISIBLE_DEVICES=$((4 + i)) $T --tag t2mw --loss mse --layers $lo $((lo + 8)) > /data/lsh/logs/v7mse_train_L$lo.log 2>&1 &
  i=$((i + 1))
done
wait
cat /data/lsh/logs/v7mse_train_L*.log | grep -E "done|Error|Traceback"
CUDA_VISIBLE_DEVICES=4 $T --tag t2mw --loss mse --merge || exit 1
i=0
for lo in 0 8 16 24; do
  CUDA_VISIBLE_DEVICES=$((4 + i)) $T --tag v7mse_L$lo --layers $lo $((lo + 8)) \
    --eval_cb def=$CBD/cb_d8K256x2_post_cn_greedy.pt t2mw=$CBD/cb_d8K256x2_post_cn_t2mw.pt t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt \
    > /data/lsh/logs/v7mse_eval_L$lo.log 2>&1 &
  i=$((i + 1))
done
wait
tail -n 2 /data/lsh/logs/v7mse_eval_L*.log
