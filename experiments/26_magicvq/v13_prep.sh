#!/usr/bin/env bash
# E7 magicvq v13: build the dense draft-weight sources below INT4 (v13_prep.py) and their PPL@4096.
# Usage: tools/run_tmux.sh <name> <gpu> nestvq -- experiments/26_magicvq/v13_prep.sh {c1|gptq|ppl_ref}
set -euo pipefail
cd /hai/home/lsh/SpecRQ
export PYTHONWARNINGS=ignore HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
D=/data/lsh/ckpts/weightvq/llama31_8b
P="nice -n 10 taskset -c 16-31,48-63 python experiments/26_magicvq/v13_prep.py"
PPL=results/E7/llama8b_magicvq_v13_lowbit_ppl.json
KD2=/data/lsh/results_accept/llama31_c1_uniform_kd2/train_B_bf16/draft_codebooks.pt
case $1 in
  c1)
    for k in 5 3 2; do $P c1 --k $k --out $D/c1ft5_k${k}_dense/w_bf16.safetensors; done
    $P c1 --k 2 --draft_cb $KD2 --out $D/c1ft5_k2B_dense/w_bf16.safetensors
    for k in 5 3 2 2B; do $P ppl --src $D/c1ft5_k${k}_dense/w_bf16.safetensors --tag c1k$k --out $PPL; done ;;
  gptq)
    $P gptq --wbits 4 --n_layers 2 --out /data/lsh/temp/tmp/v13smoke/gptq4_l2.safetensors
    $P cmp /data/lsh/temp/tmp/v13smoke/gptq4_l2.safetensors $D/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors
    $P gptq --wbits 3 --out $D/gptq3_g128_pile128x2048_dense/w_fp16.safetensors
    $P ppl --src $D/gptq3_g128_pile128x2048_dense/w_fp16.safetensors --tag g3 --out $PPL
    $P gptq --wbits 2 --out $D/gptq2_g128_pile128x2048_dense/w_fp16.safetensors
    $P ppl --src $D/gptq2_g128_pile128x2048_dense/w_fp16.safetensors --tag g2 --out $PPL ;;
  ppl_ref)
    $P ppl --src bf16 --tag bf16 --out ${PPL%.json}_ref.json
    $P ppl --src $D/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors --tag w4 --out ${PPL%.json}_ref.json ;;
esac
