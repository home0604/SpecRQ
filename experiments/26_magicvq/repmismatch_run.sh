#!/usr/bin/env bash
# E7 gamma6 rep-mismatch investigation (TP=4, GPUs 4-7, --accounting both --soft_rep_check).
# Root cause: static-mode frozen rows keep running draft + verify at their final length (<= P + gen + gamma), writing
# cache positions up to P + gen + 2 gamma; the old T = ceil64(Pmax + gen + gmax + 2) is too short at gamma 6 when
# rows have P close to Pmax (16k LongBench points: slack T - Pmax - gen = 9 < 12) -> the append wraps into the next
# (seq, kv-head)'s first positions (BOS / sinks) of the target cache and persists for the rest of the process.
#   repro : --legacy_T, worst point (gov16), reps 1  -> expect prompt-head changes + rep mismatches
#   F*    : fixed T, rerun of every v7 / v8 config that ran in or after an overflowing gamma-6 config (reps 3)
# Skip a point if /data/lsh/temp/tmp/rm_stop exists.
cd /hai/home/lsh/SpecRQ
CBD=/data/lsh/results_accept/magicvq/cb
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --accounting both --soft_rep_check --gen 256 --cbx t2w=$CBD/cb_d8K256x2_post_cn_t2w.pt t1w=$CBD/cb_d8K256x2_post_cn_t1w.pt t2mw=$CBD/cb_d8K256x2_post_cn_t2mw.pt"
pt() { [ -e /data/lsh/temp/tmp/rm_stop ] && { echo "[rm] stop file, skip $*"; return; }; echo "[rm] point $*"; $R "$@"; echo "[rm] rc=$?"; }
PART=${1:-all}
if [ "$PART" = repro ] || [ "$PART" = all ]; then
pt --reps 1 --legacy_T --tag rmlegacy --task gov_report --ctx 16384 --B 16 --configs ar hyb512pL2@t2w:6 hyb512pL1@t1w:6 hyb512pL1@t1w:4
fi
if [ "$PART" = fix ] || [ "$PART" = all ]; then
pt --reps 3 --tag rmfixa --task gov_report --ctx 16384 --B 16 --configs ar snapkv4096:3 hyb512pL2@t2w:6 hyb512pL1@t1w:4 hyb512pL1@t1w:6
pt --reps 3 --tag rmfixa --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 hyb512pL2@t2w:6 hyb512pL1@t1w:4 hyb512pL1@t1w:6
pt --reps 3 --tag rmfix7 --task gov_report --ctx 16384 --B 16 --configs ar snapkv4096:3 hyb512pL2@t2w:6 vqpL2+r128@t2w:3 vqpL2+r128@t2w:4 vqpL2+r128@t2w:6
pt --reps 3 --tag rmfix7 --task qmsum --ctx 16384 --B 16 --configs ar snapkv2048:3 hyb512pL2@t2w:6 vqpL2+r128@t2w:3 vqpL2+r128@t2w:4 vqpL2+r128@t2w:6
pt --reps 3 --tag rmfix7 --task repobench-p --ctx 16384 --B 16 --configs ar snapkv2048:3 hyb512pL2@t2w:6 vqpL2+r128@t2w:3 vqpL2+r128@t2w:4 vqpL2+r128@t2w:6
pt --reps 3 --tag rmfix7g6 --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 snapkv2048:6 snapkv4096:6 hyb512pL2@t2w:6 hyb512pL2@t2w:5 vqpL2+r128@t2w:6 vqpL2+r128@t2w:5
pt --reps 3 --tag rmfixmse --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 hyb512pL2:6 hyb512pL2@t2mw:6 hyb512pL2@t2w:6 vqpL2+r128:6 vqpL2+r128@t2mw:6 vqpL2+r128@t2w:6
fi
