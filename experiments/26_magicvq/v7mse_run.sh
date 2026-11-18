#!/usr/bin/env bash
# E7 magicvq v7mse e2e: default ('p' greedy) vs t2mw (MSE-trained control) vs t2w (attention-loss trained) codebooks.
# Gammas: t2w-best of v6e2e / v7 (hybrid g6; RQ2+r128 g6 or g4). <= ~6 drafts per process; 32k B16 split in two
# processes (hybrid / RQ2), each with its own AR + best SnapKV. Skip a point if /data/lsh/temp/tmp/v7mse_stop exists.
cd /hai/home/lsh/SpecRQ
CBD=/data/lsh/results_accept/magicvq/cb
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting both --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t2mw=$CBD/cb_d8K256x2_post_cn_t2mw.pt --soft_rep_check"
pt() { [ -e /data/lsh/temp/tmp/v7mse_stop ] && { echo "[v7mse] stop file, skip $*"; return; }; echo "[v7mse] point $*"; $R "$@"; echo "[v7mse] rc=$?"; }
H() { echo "hyb512pL2:$1 hyb512pL2@t2mw:$1 hyb512pL2@t2w:$1"; }
V() { echo "vqpL2+r128:$1 vqpL2+r128@t2mw:$1 vqpL2+r128@t2w:$1"; }
pt --tag v7mseh --gen 256 --task syn_quote --ctx 32768 --B 16 --configs ar snapkv4096:4 $(H 6)
pt --tag v7msev --gen 256 --task syn_quote --ctx 32768 --B 16 --configs ar snapkv4096:4 $(V 6)
pt --tag v7mse --gen 256 --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 $(H 6) $(V 6)
pt --tag v7mseh --gen 256 --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:4 $(H 6)
pt --tag v7msev --gen 256 --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:4 $(V 4)
pt --tag v7mse --gen 256 --task syn_quote --ctx 32768 --B 8 --configs ar snapkv4096:3 $(H 6) $(V 4)
pt --tag v7mseg512 --gen 512 --task syn_quote --ctx 32768 --B 16 --configs ar snapkv4096:3 $(H 4)
