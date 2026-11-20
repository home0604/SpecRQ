#!/usr/bin/env bash
# E7 magicvq v13: tau of drafts with weights below INT4 (C1 weight-VQ nested prefix k=3, C1 k=2 + draft-only codebooks
# (E3 'B'), GPTQ INT3 g128; weights-only also C1 k=2 plain prefix) + compressed KV, on the v6 samples, same points /
# gammas as v12_tau.sh. Drafts per source: exact (= weights-only), SnapKV, vqpt2wL2+r128, hyb512pt2wL2 (v12's 1-bit
# vqpe2enkl_w10L1+r128 dropped for GPU time). GPTQ INT2 (wt2 PPL in the thousands) is not run. bf16 and INT4 rows are reused from the v12 records (not rerun); snapkv4096@w4 is
# rerun as anchor (must reproduce v12 per sample). v13_tau.py = v12_w4.py with an LRU pool of 1 GPU shadow layer.
# Usage: experiments/26_magicvq/v13_tau.sh <job> ; jobs: sq32 gov32hq16 gov16 pg32 smoke ; env RUN (default v13)
set -euo pipefail
cd /hai/home/lsh/SpecRQ
D=/data/lsh/ckpts/weightvq/llama31_8b
WS="w4=$D/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors"
WS="$WS,k3=$D/c1ft5_k3_dense/w_bf16.safetensors,k2B=$D/c1ft5_k2B_dense/w_bf16.safetensors"
WS="$WS,g3=$D/gptq3_g128_pile128x2048_dense/w_fp16.safetensors,k2=$D/c1ft5_k2_dense/w_bf16.safetensors"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false V12_TAGS="e2enkl_w10" V12_WSRC="$WS" V13_NSHADOW=1
base() { local d="exact $1 vqpt2wL2+r128 hyb512pt2wL2" o="snapkv4096@w4"
  for s in k3 k2B g3; do for x in $d; do o="$o $x@$s"; done; done
  echo $o exact@k2; }
D32=$(base "snapkv4096")
D16=$(base "snapkv2048 snapkv4096")
W="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/v13_tau.py"
LB="$W lb_tau.py --n 99 --G 6"
R=${RUN:-v13}
case $1 in
  sq32) for r in "${R}a v6a" "${R}b v6b"; do set -- $r; $LB --drafts $D32 --n_cand 100 --ctx 32768 --tasks syn_quote --run $1 --only_run $2; done ;;
  gov32hq16)
    for r in "${R}a v6a" "${R}b v6b"; do set -- $r; $LB --drafts $D32 --n_cand 100 --ctx 32768 --tasks gov_report --run $1 --only_run $2; done
    $LB --drafts $D16 --n_cand 100 --ctx 16384 --tasks hotpotqa_q --run ${R}a --only_run v6a
    $LB --drafts $D16 --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run ${R}b --only_run v6b
    $LB --drafts $D16 --n_cand 60 --min_frac 0.9 --ctx 16384 --tasks hotpotqa_q --run ${R}a2 --only_run v6a2 ;;
  gov16)
    $LB --drafts $D16 --n_cand 100 --ctx 16384 --tasks gov_report --run ${R}a --only_run v6a
    $LB --drafts $D16 --n_cand 80 --min_frac 0.9 --ctx 16384 --tasks gov_report --run ${R}b --only_run v6b ;;
  pg32) $W tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 5 6 --run $R --drafts $D32 ;;
  smoke) $W tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 5 6 --run ${R}smoke --limit 1 --drafts $D32 ;;
esac
