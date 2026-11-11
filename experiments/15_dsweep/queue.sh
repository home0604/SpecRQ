#!/usr/bin/env bash
# Claim-1 d-sweep worker (E5; copy of experiments/10_screen/queue.sh) (one per GPU; several workers share one queue file). Each config runs as its own tmux job
# e5_screen_<cfg> (tools/run_tmux.sh, CPU-rule prefix). Claims are atomic mkdir's in $Q.claims/<cfg>, so two workers
# never take the same config. Queue file lines: "<cfg> <d> <nbits> <L> <T|N|N_tag> [VAR=a,b ...]"; '#' lines skipped; the
# file is re-read before every claim, so it can be reordered/edited. Checkpoint: touch $Q.PAUSE holds every worker
# before its next claim. A failed config is logged and the worker continues.
# env CPUS (taskset list, default 0-15,32-47 = NUMA0), MEMCAP_GIB (default 130).
# Usage: tools/run_tmux.sh <name> none nestvq -- [env CPUS=16-31,48-63] bash experiments/15_dsweep/queue.sh <queue> <gpu> [WAIT_FOR job] [FIRST cfg_obj]
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
    [[ -n "$FIRST" && "${cfg}_$(awk '{print $4}' <<<"$rest")" != "$FIRST" ]] && continue
    local key="${cfg}_$(awk '{print $4}' <<<"$rest")$(grep -o 'RUN_SUFFIX=[^ ]*' <<<"$rest" | cut -d= -f2)$(grep -o 'JOB_SUFFIX=[^ ]*' <<<"$rest" | cut -d= -f2)"  # <cfg>_<obj><sfx>
    mkdir "$Q.claims/$key" 2>/dev/null && { echo "gpu$GPU $(date -Is)" > "$Q.claims/$key/by"; echo "$cfg $rest"; return; }
  done < "$Q"
}
while true; do
  [[ -z "$FIRST" ]] && while [[ -e $Q.PAUSE ]]; do sleep 60; done
  line=$(claim); FIRST=""
  [[ -z "$line" ]] && { echo "[worker gpu$GPU] nothing left $(date -Is)"; exit 0; }
  read -r cfg d nb L obj extra <<<"$line"  # extra: optional VAR=v,v env assignments (CBW, KDS, RUN_SUFFIX, TRAIN_LOG)
  sfx=$(grep -o 'RUN_SUFFIX=[^ ]*' <<<"$extra" | cut -d= -f2 || true); job=e5_dsweep_${cfg}_$obj$sfx$(grep -o 'JOB_SUFFIX=[^ ]*' <<<"$extra" | cut -d= -f2 || true)  # JOB_SUFFIX: job/claim name only (re-runs)
  until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) < 2000 )); do sleep 60; done
  # host RAM guard: no new job while the user cgroup is above MEMCAP_GIB (default 130)
  CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.current
  until [[ ! -r $CG ]] || (( $(cat $CG) < ${MEMCAP_GIB:-130} * 1073741824 )); do
    echo "[worker gpu$GPU] cgroup $(( $(cat $CG) / 1073741824 )) GiB > ${MEMCAP_GIB:-130}; waiting $(date -Is)"; sleep 120; done
  echo "[worker gpu$GPU] launch $cfg d=$d nbits=$nb L=$L obj=$obj $(date -Is)"
  if [[ "$cfg" == gptq_int* ]]; then  # scalar baseline line: "gptq_int<b> <b> - - S"
    tools/run_tmux.sh e5_dsweep_${cfg}_$obj $GPU nestvq -- env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
      TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c ${CPUS:-0-15,32-47} python experiments/15_dsweep/gptq_fake.py --wbits $d --save_result \
      || { echo "[worker gpu$GPU] launch of $cfg failed"; continue; }
  else
  tools/run_tmux.sh $job $GPU nestvq -- env DSWEEP_JOB=$job $extra OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c ${CPUS:-0-15,32-47} bash experiments/15_dsweep/run_dsweep.sh $cfg $d $nb $L $obj \
    || { echo "[worker gpu$GPU] launch of $cfg failed"; continue; }
  fi
  until [[ -n "$(exitcode $job)" ]]; do sleep 60; done
  rc=$(exitcode $job); echo "[worker gpu$GPU] $cfg exit=$rc $(date -Is)"
  [[ "$rc" == 0 ]] || echo "[worker gpu$GPU] WARNING $cfg failed (rc=$rc); continuing"
done
