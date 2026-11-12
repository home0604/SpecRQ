#!/usr/bin/env bash
# Run Llama d-sweep steps one after another on ONE GPU (activations on the GPU: n192-type runs, a few GiB host RAM).
# Each step = training job e5_lds_<cfg>_<obj><sfx> (run_llama_dsweep.sh) then single-GPU eval job
# e5_lds_eval_<cfg>_<obj><sfx> (run_llama_eval.sh with REF_MAXMEM: bf16 reference split GPU/CPU). Every job is its own
# tmux session (tools/run_tmux.sh); this script only launches and waits. Re-running re-attaches to launched jobs.
# env WAIT_FOR=<job>: wait for that job to exit before the first step (keeps a GPU reserved for another seq).
# Host-RAM guard before each launch: cgroup anon+shmem < MEM_GIB (default 110).
# Usage: tools/run_tmux.sh <name> none nestvq -- bash experiments/15_dsweep/llama_gpu_seq.sh <gpu> <step> [<step> ...]
#   step = cfg:d:nbits:L:obj:nsamples:ft_epochs:sfx[:seed]   e.g. d8K256L2:8:8:2:T:192:5:_FT_n192:2
#   env CBW (nested alphas, passed through), KDS (draft prefixes for copy-draft TV/tau in the eval), EVALK_OVERRIDE
#   training evaluates only k=L (EVALK=L; prefix evals grew GPU/pinned memory and caused OOMs)
set -uo pipefail
cd /hai/home/lsh/SpecRQ
GPU=${1:?gpu}; shift
LOGS=/data/lsh/logs; MEM_GIB=${MEM_GIB:-110}
CPU="env OMP_NUM_THREADS=${OMP:-8} MKL_NUM_THREADS=${OMP:-8} TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c ${CPUS:-0-15,32-47}"
exitcode() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('exit_code',''))" "$LOGS/$1.meta.json" 2>/dev/null; }
waitjob() { until [[ -n "$(exitcode $1)" ]]; do sleep 60; done; echo "[seq gpu$GPU] $1 exit=$(exitcode $1) $(date -Is)"; }
pinned_gib() { grep -E '^(anon|shmem) ' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.stat | awk '{s+=$2} END {print int(s/2^30)}'; }
waitmem() { until (( $(pinned_gib) < MEM_GIB )); do echo "[seq gpu$GPU] anon+shmem $(pinned_gib) GiB >= $MEM_GIB; waiting"; sleep 120; done; }
waitgpu() { until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) < 2000 )); do sleep 60; done; }
launch() {  # name, env..., script args
  local name=$1; shift
  if [[ -e $LOGS/$name.meta.json ]]; then echo "[seq gpu$GPU] $name already launched; waiting"
  else waitmem; waitgpu; tools/run_tmux.sh $name $GPU nestvq -- "$@" || { echo "[seq gpu$GPU] launch $name failed"; return 1; }; fi
  waitjob $name
}
if [[ -n "${WAIT_FOR:-}" ]]; then echo "[seq gpu$GPU] waiting for $WAIT_FOR $(date -Is)"; waitjob $WAIT_FOR; fi
for st in "$@"; do
  IFS=: read -r cfg d nb L obj ns ft sfx seed <<<"$st"; seed=${seed:-0}
  tj=e5_lds_${cfg}_${obj}${sfx}
  launch $tj env NSAMPLES=$ns FT_EPOCHS=$ft SFX=$sfx SEED=$seed "EVALK=${EVALK_OVERRIDE:-$L}" "CBW=${CBW:-}" $CPU bash experiments/15_dsweep/run_llama_dsweep.sh $cfg $d $nb $L $obj || continue
  if [[ "$(exitcode $tj)" != 0 ]]; then  # training finished but its own GPU eval may have OOMed: re-measure PPL with --load
    ck=/data/lsh/ckpts/weightvq/dsweep_llama/${cfg}_${obj}${sfx}
    [[ -e $ck/args.pt ]] || { echo "[seq gpu$GPU] $tj failed before saving; skipping"; continue; }
    launch e5_lds_ppl_${cfg}_${obj}${sfx} env RUNTMUX_NAME=e5_lds_ppl_${cfg}_${obj}${sfx} $CPU \
      bash experiments/15_dsweep/run_llama_ppl_load.sh $ck $LOGS/$tj.log ${EVALK_OVERRIDE:-$L} || continue
    tlog=$LOGS/$tj.combined.log
  else tlog=$LOGS/$tj.log; fi
  launch e5_lds_eval_${cfg}_${obj}${sfx} env NSAMPLES=$ns FT_EPOCHS=$ft SFX=$sfx SEED=$seed TRAIN_LOG=$tlog "REF_MAXMEM=0:5GiB,cpu:40GiB" "CBW=${CBW:-}" "KDS=${KDS:-}" \
    $CPU bash experiments/15_dsweep/run_llama_eval.sh $cfg $d $nb $L $obj
done
echo "[seq gpu$GPU] done $(date -Is)"
