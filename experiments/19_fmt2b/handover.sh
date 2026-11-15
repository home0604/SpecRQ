#!/usr/bin/env bash
# One-off: stop an old run_fmt2b.sh session after a given eval json exists, remove the half-written draft_q files it
# may have started next (no eval json, not claimed by a queue lock), then start a queue worker on its GPU.
# Usage: handover.sh <old session name> <json to wait for> <gpu> <cpu list>
set -uo pipefail
OLD=$1; JS=$2; GPU=$3; CPUS=$4
until [[ -e $JS ]]; do sleep 20; done
sleep 2; tmux kill-session -t "=sr-$OLD" && echo "killed sr-$OLD $(date -Is)"
sleep 20
D=$(dirname $(dirname $JS))
for pt in $D/draft_q/A_*.pt; do
  lab=$(basename $pt .pt); lab=${lab#A_}
  [[ -e $D/eval/A_$lab.json || -e $D/lock/$lab ]] && continue
  echo "remove partial $pt"; rm -f $pt $D/draft_q/A_$lab.meta.json
done
cd /hai/home/lsh/SpecRQ
tools/run_tmux.sh e5_fmt2b_q$GPU $GPU nestvq -- env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  nice -n 10 taskset -c $CPUS bash experiments/19_fmt2b/run_fmt2b_q.sh experiments/19_fmt2b/queue_ts.txt
