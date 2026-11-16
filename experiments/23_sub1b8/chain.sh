#!/usr/bin/env bash
# Wait for a training job to exit 0, then launch the A/B eval job on the same GPU (tools/run_tmux.sh).
# Usage: tools/run_tmux.sh <name> none nestvq -- bash experiments/23_sub1b8/chain.sh <train job> <gpu> <cpus> <eval job> <ckpt> <L> <outdir> "<steps>"
set -uo pipefail
TJ=$1; GPU=$2; CPUS=$3; EJ=$4; CKPT=$5; L=$6; OUT=$7; STEPS=$8
cd /hai/home/lsh/SpecRQ
ec() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('exit_code',''))" /data/lsh/logs/$1.meta.json 2>/dev/null; }
until [[ -n "$(ec $TJ)" ]]; do sleep 60; done
echo "[chain] $TJ exit=$(ec $TJ) $(date -Is)"
[[ "$(ec $TJ)" == 0 && -e $CKPT/args.pt ]] || { echo "[chain] training failed; not launching eval"; exit 1; }
until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) < 2000 )); do sleep 60; done
tools/run_tmux.sh $EJ $GPU nestvq -- env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c $CPUS \
  bash experiments/23_sub1b8/eval_sub1b.sh $CKPT $L $OUT "$STEPS"
echo "[chain] launched $EJ $(date -Is)"
