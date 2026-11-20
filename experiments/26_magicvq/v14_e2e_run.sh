#!/usr/bin/env bash
# E7 v14: measured e2e with Marlin INT4 draft weights (--draft_w4) vs bf16 draft weights; target verify bf16.
# TP=4 GPUs 4-7, <task> 32k B16 gen 256, all-active accounting. Order off, on, off, on (drift check).
# Usage: bash experiments/26_magicvq/v14_e2e_run.sh [task=syn_quote]
cd /hai/home/lsh/SpecRQ
TASK=${1:-syn_quote}
CBD=/data/lsh/results_accept/magicvq/cb
W4=/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting all_active --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt --gen 256 --soft_rep_check --task $TASK --ctx 32768 --B 16"
C="ar snapkv4096:4 vqpL2+r128@t2w:4 hyb512pL2@t2w:6"
for i in 1 2; do
  echo "[v14] nvsmi before w4off$i"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
  echo "[v14] off $i"; $R --tag v14w4off$i --configs $C; echo "[v14] rc=$?"
  echo "[v14] nvsmi before w4on$i"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
  echo "[v14] on $i";  $R --tag v14w4on$i --draft_w4 $W4 --configs $C; echo "[v14] rc=$?"
done
