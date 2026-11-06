#!/usr/bin/env bash
# Screening worker for E5 (one per GPU; several workers share one queue file). Each config runs as its own tmux job
# e5_screen_<cfg> (tools/run_tmux.sh, CPU-rule prefix). Claims are atomic mkdir's in $Q.claims/<cfg>, so two workers
# never take the same config. Queue file lines: "<cfg> <d> <nbits|b1,b2,..> <L> <kd,kd,...>"; '#' lines skipped; the
# file is re-read before every claim, so it can be reordered/edited. Checkpoint: touch $Q.PAUSE holds every worker
# before its next claim. A failed config is logged and the worker continues.
# Usage: tools/run_tmux.sh <name> none nestvq -- bash experiments/10_screen/queue.sh <queue> <gpu> [WAIT_FOR job] [FIRST cfg]
#   WAIT_FOR: wait for this job's meta exit_code before the first claim; FIRST: claim this cfg first (ignores PAUSE)
set -uo pipefail
Q=${1:?queue}; GPU=${2:?gpu}; WAIT_FOR=${3:-}; FIRST=${4:-}
LOGS=/data/lsh/logs; cd /hai/home/lsh/SpecRQ; unset TMUX; mkdir -p $Q.claims
exitcode() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('exit_code',''))" "$LOGS/$1.meta.json" 2>/dev/null; }
if [[ -n "$WAIT_FOR" ]]; then
  echo "[worker gpu$GPU] waiting for $WAIT_FOR $(date -Is)"
  until [[ -n "$(exitcode $WAIT_FOR)" ]]; do sleep 60; done
  echo "[worker gpu$GPU] $WAIT_FOR exit=$(exitcode $WAIT_FOR)"
fi
claim() {  # print the first claimable line (or FIRST), claiming it
  local cfg rest
  while read -r cfg rest; do
    [[ -z "$cfg" || "$cfg" == \#* ]] && continue
    [[ -n "$FIRST" && "$cfg" != "$FIRST" ]] && continue
    mkdir "$Q.claims/$cfg" 2>/dev/null && { echo "gpu$GPU $(date -Is)" > "$Q.claims/$cfg/by"; echo "$cfg $rest"; return; }
  done < "$Q"
}
while true; do
  [[ -z "$FIRST" ]] && while [[ -e $Q.PAUSE ]]; do sleep 60; done
  line=$(claim); FIRST=""
  [[ -z "$line" ]] && { echo "[worker gpu$GPU] nothing left $(date -Is)"; exit 0; }
  read -r cfg d nb L kds <<<"$line"; kds=${kds//,/ }
  until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) < 2000 )); do sleep 60; done
  echo "[worker gpu$GPU] launch $cfg d=$d nbits=$nb L=$L kds=$kds $(date -Is)"
  tools/run_tmux.sh e5_screen_$cfg $GPU nestvq -- env SCREEN_JOB=e5_screen_$cfg OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 0-15,32-47 bash experiments/10_screen/run_screen.sh $cfg $d $nb $L "$kds" \
    || { echo "[worker gpu$GPU] launch of $cfg failed"; continue; }
  until [[ -n "$(exitcode e5_screen_$cfg)" ]]; do sleep 60; done
  rc=$(exitcode e5_screen_$cfg); echo "[worker gpu$GPU] $cfg exit=$rc $(date -Is)"
  [[ "$rc" == 0 ]] || echo "[worker gpu$GPU] WARNING $cfg failed (rc=$rc); continuing"
done
