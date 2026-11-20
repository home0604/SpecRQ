#!/usr/bin/env bash
# E10 phase 3 -> E7 e2e: MV_LUT_VARIANT (p3 LUT VQ-KV partial kernels, e2e_kern.py) off vs on, measured e2e.
# TP=4 GPUs 4-7 (shark 4090), syn_quote 32k B16, all-active accounting.
#   bash experiments/26_magicvq/p3_e2e_run.sh smoke   gen 64, reps 1: off / on(4090 preset) / on(a100 preset, KST2
#                                                     falls back on the 4090) -> outputs must match AR / each other
#   bash experiments/26_magicvq/p3_e2e_run.sh timing  gen 256, reps 3: off, on, off, on (4090 preset; drift check)
cd /hai/home/lsh/SpecRQ
MODE=${1:-smoke}
CBD=/data/lsh/results_accept/magicvq/cb
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --accounting all_active --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t1w=$CBD/cb_d8K256x2_post_cn_t1w.pt --soft_rep_check --task syn_quote --ctx 32768 --B 16"
C="ar vqpL2+r128@t2w:4 hyb512pL2@t2w:6 vqpL1+r128@t1w:4"
nv() { echo "[p3e2e] nvsmi $1"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader; }
if [[ $MODE == smoke ]]; then
  S="--gen 64 --reps 1"
  nv off; echo "[p3e2e] smoke off"; MV_LUT_VARIANT= $R $S --tag p3smoke_off --configs $C; echo "[p3e2e] rc=$?"
  nv on4090; echo "[p3e2e] smoke on 4090"; MV_LUT_VARIANT=4090 $R $S --tag p3smoke_on4090 --configs $C; echo "[p3e2e] rc=$?"
  nv ona100; echo "[p3e2e] smoke on a100"; MV_LUT_VARIANT=a100 $R $S --tag p3smoke_ona100 --configs $C; echo "[p3e2e] rc=$?"
else
  S="--gen 256 --reps 3"
  for i in 1 2; do
    nv off$i; echo "[p3e2e] off $i"; MV_LUT_VARIANT= $R $S --tag p3off$i --configs $C; echo "[p3e2e] rc=$?"
    nv on$i; echo "[p3e2e] on $i"; MV_LUT_VARIANT=4090 $R $S --tag p3on$i --configs $C; echo "[p3e2e] rc=$?"
  done
fi
nv end
