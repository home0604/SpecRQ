#!/usr/bin/env bash
# E7/E10: measured e2e of the lazy-max LUT kernel flag (ce9668d): same point / configs twice, --lazy_max off then on.
# TP=4 GPUs 4-7, syn_quote 32k B16 gen 256, all-active accounting. Order off, on, off, on (drift check).
cd /hai/home/lsh/SpecRQ
CBD=/data/lsh/results_accept/magicvq/cb
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting all_active --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t1w=$CBD/cb_d8K256x2_post_cn_t1w.pt --gen 256 --soft_rep_check --task syn_quote --ctx 32768 --B 16"
C="ar snapkv4096:4 vqpL2+r128@t2w:4 hyb512pL2@t2w:6 vqpL1+r128@t1w:4"
for i in 1 2; do
  echo "[lazy] off $i"; $R --tag lazyoff$i --configs $C; echo "[lazy] rc=$?"
  echo "[lazy] on $i";  $R --tag lazyon$i --lazy_max --configs $C; echo "[lazy] rc=$?"
done
