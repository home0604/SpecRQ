#!/usr/bin/env bash
# E7 MagicDec repro: OUR fully measured TP=4 loop (experiments/26_magicvq/e2e_real.py, unchanged) on the SAME PG-19 rows
# the official MagicDec benchmark feeds (prompts_pg19md_ctx32768.json from make_prompts.py, P = 32672):
# B16 -> rows 80..95 (= official B16 first measured batch), B8 -> rows 40..47 (= official B8 first measured batch).
# Run via: tools/run_tmux.sh magicdec_ours 4,5,6,7 nestvq -- bash experiments/27_magicdec_repro/run_ours.sh
set -u
cd /hai/home/lsh/SpecRQ
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 3600 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --task pg19md --ctx 32768 --reps 3"
for g in 80 256; do
  $R --gen $g --tag md27g$g --B 16 --skip 80 --configs ar snapkv4096:3 snapkv4096:4 snapkv2048:3 snapkv2048:4
  $R --gen $g --tag md27g$g --B 8 --skip 40 --configs ar snapkv4096:3 snapkv4096:4
done
