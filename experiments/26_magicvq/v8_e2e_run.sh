#!/usr/bin/env bash
# E7 magicvq v8 phase 2: measured e2e of 1-bit VQ KV drafts (TP=4, GPUs 4-7, B16, gen 256, --accounting both).
# Same config strings (incl. gamma) at the 16k and 32k points so v8_e2e_summarize.py can fit per-round components in ctx.
#   a: AR + best SnapKV of the point + RQ2 t2w hybrid g6 (v7 best) + 1-bit hybrid t1w g4 / g6
#   b: AR + best SnapKV + plain RQ2+r128 t2w g4 + nested-prefix 1-bit (t2w stage 1) plain g4 and hybrid g4
# A point is skipped if /data/lsh/temp/tmp/v8_stop exists (budget stop between points).
cd /hai/home/lsh/SpecRQ
CBD=/data/lsh/results_accept/magicvq/cb
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting both --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t1w=$CBD/cb_d8K256x2_post_cn_t1w.pt --gen 256 --soft_rep_check"
pt() { [ -e /data/lsh/temp/tmp/v8_stop ] && { echo "[v8] stop file, skip $*"; return; }; echo "[v8] point $*"; $R "$@"; echo "[v8] rc=$?"; }
A="hyb512pL2@t2w:6 hyb512pL1@t1w:4 hyb512pL1@t1w:6"
Bq="vqpL2+r128@t2w:4 vqpL1+r128@t2w:4 hyb512pL1@t2w:4"
pt --tag v8e2ea --task syn_quote --ctx 32768 --B 16 --configs ar snapkv4096:4 $A
pt --tag v8e2eb --task syn_quote --ctx 32768 --B 16 --configs ar snapkv4096:4 $Bq
pt --tag v8e2ea --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 $A
pt --tag v8e2eb --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 $Bq
pt --tag v8e2ea --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:6 $A
pt --tag v8e2eb --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:6 $Bq
pt --tag v8e2ea --task gov_report --ctx 16384 --B 16 --configs ar snapkv4096:3 $A
pt --tag v8e2eb --task gov_report --ctx 16384 --B 16 --configs ar snapkv4096:3 $Bq
