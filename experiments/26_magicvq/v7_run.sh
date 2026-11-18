#!/usr/bin/env bash
# E7 magicvq v7: more e2e points with the v6 t2w codebook (run via tools/run_tmux.sh on GPUs 4-7).
# A point is skipped if /data/lsh/temp/tmp/v7_stop exists (budget stop between points).
cd /hai/home/lsh/SpecRQ
CB=/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt
R="env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 16-31,48-63 timeout 4500 torchrun --standalone --nproc_per_node 4 experiments/26_magicvq/e2e_real.py --reps 3 --accounting both --cbx t2w=$CB --gen 256 --soft_rep_check"
SNAP="snapkv4096:3 snapkv4096:4 snapkv2048:3 snapkv2048:4 stream512:3"
VQ="hyb512pL2@t2w:3 hyb512pL2@t2w:4 hyb512pL2@t2w:6 vqpL2+r128@t2w:3 vqpL2+r128@t2w:4 vqpL2+r128@t2w:6"
pt() { [ -e /data/lsh/temp/tmp/v7_stop ] && { echo "[v7] stop file, skip $*"; return; }; echo "[v7] point $*"; $R "$@"; echo "[v7] rc=$?"; }
case ${1:-all} in
  g6|all)
    # v7g6 (first attempt, partial records kept): hyb t2w g6/g5 done; vq t2w g6 aborted (hotpotqa: static outputs
    # differed between reps; pg19: OOM in the 6-draft process) -> vq g6/g5 in their own small processes (v7g6v)
    pt --tag v7g6v --task hotpotqa_q --ctx 16384 --B 16 --configs ar snapkv2048:4 vqpL2+r128@t2w:6 vqpL2+r128@t2w:5
    pt --tag v7g6v --task pg19 --ctx 32768 --B 16 --configs ar snapkv4096:4 vqpL2+r128@t2w:6 vqpL2+r128@t2w:5
    ;;&
  lb|all)
    pt --tag v7 --task gov_report --ctx 32768 --B 8 --configs ar $SNAP $VQ
    for t in gov_report qmsum repobench-p multi_news; do pt --tag v7 --task $t --ctx 16384 --B 16 --configs ar $SNAP $VQ; done
    pt --tag v7 --task hotpotqa_q --ctx 32768 --B 16 --configs ar $SNAP
    pt --tag v7b --task hotpotqa_q --ctx 32768 --B 16 --configs ar snapkv4096:4 hyb512pL2@t2w:3 hyb512pL2@t2w:4 hyb512pL2@t2w:6 vqpL2+r128@t2w:4 vqpL2+r128@t2w:6
    ;;
esac
