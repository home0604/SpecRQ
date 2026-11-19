#!/usr/bin/env bash
# E7 magicvq v12: tau of drafts with INT4 (GPTQ g128, E4 Marlin checkpoint dequantized to dense) WEIGHTS + compressed KV,
# next to the same drafts with the target's bf16 weights (anchors: must reproduce v6 / v11 sample by sample).
# Same samples as v6 (--only_run on the v6 records), same jobs as v10_tau.sh. Drafts (each plain and @w4):
#   exact (weights-only when @w4), SnapKV (4096; + 2048 at 16k), vqpt2wL2+r128, hyb512pt2wL2, vqpe2enkl_w10L1+r128.
# Usage: experiments/26_magicvq/v12_tau.sh <job> ; jobs: sq32 gov32hq16 gov16 pg32 ; env RUN (default v12)
set -euo pipefail
cd /hai/home/lsh/SpecRQ
W4=/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0 PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false V12_TAGS="e2enkl_w10" V12_WSRC="w4=$W4"
base() { local d="exact $1 vqpt2wL2+r128 hyb512pt2wL2 vqpe2enkl_w10L1+r128" o=""; for x in $d; do o="$o $x $x@w4"; done; echo $o; }
D32=$(base "snapkv4096")
D16=$(base "snapkv2048 snapkv4096")
W="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/v12_w4.py"
LB="$W lb_tau.py --n 99 --G 6"
R=${RUN:-v12}
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
  smoke) V12_WSRC="w4=$W4,id=:target" $W tau_eval.py --ctx 32768 --n 36 --sets pg19 --G 6 --gammas 1 2 3 4 5 6 --run ${R}smoke --limit 1 --drafts exact exact@w4 exact@id vqpt2wL2+r128 vqpt2wL2+r128@id vqpt2wL2+r128@w4 snapkv4096 snapkv4096@id ;;
esac
