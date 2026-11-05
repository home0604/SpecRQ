#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct C1 full run: DbD quantize.sh settings (wikitext2 1024x4096 calib, val 32, rel_tol 0.01,
# finetune 32/10/3/keep_best, offload) + 35W (+ --hadamard if HAD=1), alpha=0 prefixes skipped (bit-identical loss),
# accum fp64. Eval at seqlen 4096 (quantize.sh default) for k=5,4,3, then a --load eval at seqlen 512.
set -euo pipefail
SAVE=${1:?save dir}; HAD=${HAD:-1}
MODEL=/data/models/qwen2.5_0.5b_instruct
HADFLAG=""; [[ "$HAD" == 1 ]] && HADFLAG="--hadamard"
cd /hai/home/lsh/SpecRQ/specrq/dbd
/usr/bin/time -v python main.py $MODEL wikitext2 \
  --nsamples=1024 --val_size=32 --model_seqlen=4096 \
  --num_codebooks=5 --nbits_per_codebook=8 --in_group_size=8 --out_group_size=1 \
  --codebook_weights 0 0 0.5 0 0.5 $HADFLAG --accum_dtype float64 \
  --relative_mse_tolerance=0.01 \
  --finetune_batch_size=32 --finetune_max_epochs=10 --finetune_early_stop=3 --finetune_keep_best \
  --local_batch_size=1 --offload_activations --devices cuda:0 \
  --eval_num_codebooks 5 4 3 --save $SAVE
python main.py $MODEL wikitext2 --load $SAVE --model_seqlen=512 --eval_num_codebooks 5 4 3 --devices cuda:0
