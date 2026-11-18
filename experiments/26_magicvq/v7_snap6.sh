#!/usr/bin/env bash
# E7 magicvq v7g6s: SnapKV at gamma 5/6 (fairness check for the t2w gamma-6 rows of v7g6), same-process hyb t2w g6 reference.
cd /hai/home/lsh/SpecRQ
CB=/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting both --cbx t2w=$CB --gen 256 --soft_rep_check"
$R --tag v7g6s --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:5 snapkv4096:6 hyb512pL2@t2w:6; echo "[v7] rc=$?"
$R --tag v7g6s --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:5 snapkv2048:6 snapkv4096:6 hyb512pL2@t2w:6; echo "[v7] rc=$?"
