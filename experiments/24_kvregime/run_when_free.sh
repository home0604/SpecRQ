#!/usr/bin/env bash
# E6 kvregime: wait until one of the allowed GPUs has no compute process and < 500 MiB in use, then run the sweep
# pinned to it (timing requires an exclusive GPU; GPU 0 is never used, GPU 6 is reserved for the 72B job).
# Usage (inside tools/run_tmux.sh <name> none nestvq -- bash experiments/24_kvregime/run_when_free.sh <run> [args...])
set -euo pipefail
RUN=$1; shift
ALLOWED=${KVR_GPUS:-"1 7 5 2 3 4"}
pick() {
  for g in $ALLOWED; do
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    napps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
    if [[ "$napps" == 0 && "$used" -lt 500 ]]; then echo "$g"; return 0; fi
  done
  return 1
}
while true; do
  if g=$(pick); then break; fi
  echo "$(date +%T) no free GPU among [$ALLOWED]; waiting"
  sleep 120
done
echo "$(date +%T) using GPU $g"
export CUDA_VISIBLE_DEVICES=$g
exec env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 0-15,32-47 \
  python experiments/24_kvregime/kvregime.py --run "$RUN" --jsonl "/data/lsh/logs/$RUN.jsonl" "$@"
