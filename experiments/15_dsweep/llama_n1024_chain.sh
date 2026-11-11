#!/usr/bin/env bash
# Sequential Llama-3.1-8B n1024 chain (one 1024x4096 pinned-offload training at a time: ~64 GiB pinned activations).
# Each step is its own tmux job (tools/run_tmux.sh); this script only launches and waits.
#   0) wait for WAIT_JOB (32B eval) to exit and for cgroup anon+shmem < MEM_START_GIB
#   1) probe: layer 0, d8K256L2 T, NSAMPLES=1024 OFFLOAD=pinned (GPU TRAIN_GPU)
#   env TRAIN_GPU / TRAIN_CPUS (taskset list) / EVAL_GPUS (gpu,gpu_ref | gpu with REF_MAXMEM) / RUNS (cfg:d:nbits:L:obj ...)
#   REF_MAXMEM e.g. "0:5GiB,cpu:40GiB": single-GPU eval, bf16 reference partly on the CPU (run_llama_eval.sh)
#   Re-running the chain re-attaches to already launched jobs (waits for them, then evals).
#   2) per run in RUNS: train on TRAIN_GPU, then KL/top-1 eval on EVAL_GPUS (async; next training starts right away)
# Usage: tools/run_tmux.sh e5_lds_chain none nestvq -- bash experiments/15_dsweep/llama_n1024_chain.sh
set -uo pipefail
cd /hai/home/lsh/SpecRQ
LOGS=/data/lsh/logs
WAIT_JOB=${WAIT_JOB:-e5_scale_q32_L2_eval}; TRAIN_GPU=${TRAIN_GPU:-6}; EVAL_GPUS=${EVAL_GPUS:-0,1}
MEM_START_GIB=${MEM_START_GIB:-50}
RUNS=${RUNS:-"d8K256L2:8:8:2:T d32K256L8:32:8:8:T d32K256L8:32:8:8:RQ d8K256L2:8:8:2:RQ"}
NUMA1="env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c ${TRAIN_CPUS:-16-31,48-63}"
NUMA0="env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nice -n 10 taskset -c 0-15,32-47"
exitcode() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('exit_code',''))" "$LOGS/$1.meta.json" 2>/dev/null; }
waitjob() { until [[ -n "$(exitcode $1)" ]]; do sleep 60; done; echo "[chain] $1 exit=$(exitcode $1) $(date -Is)"; }
pinned_gib() { grep -E '^(anon|shmem) ' /sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.stat | awk '{s+=$2} END {print int(s/2^30)}'; }
waitmem() { until (( $(pinned_gib) < MEM_START_GIB )); do echo "[chain] anon+shmem $(pinned_gib) GiB >= $MEM_START_GIB; waiting $(date -Is)"; sleep 120; done; }
waitgpu() { for g in ${1//,/ }; do until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g) < 2000 )); do sleep 60; done; done; }

[[ -n "$WAIT_JOB" ]] && { echo "[chain] waiting for $WAIT_JOB $(date -Is)"; waitjob $WAIT_JOB; }
waitmem; waitgpu $TRAIN_GPU
if [[ -z "$(exitcode e5_lds_probe_d8K256L2_n1024)" ]]; then
  tools/run_tmux.sh e5_lds_probe_d8K256L2_n1024 $TRAIN_GPU nestvq -- env LAYERS=1 NSAMPLES=1024 OFFLOAD=pinned \
    $NUMA1 bash experiments/15_dsweep/run_llama_dsweep.sh d8K256L2 8 8 2 T
  waitjob e5_lds_probe_d8K256L2_n1024
fi
for r in $RUNS; do
  IFS=: read -r cfg d nb L obj <<<"$r"
  job=e5_lds_${cfg}_${obj}_n1024
  if [[ -e $LOGS/$job.meta.json ]]; then  # started by an earlier chain instance: re-attach (wait), eval if not done yet
    echo "[chain] $job already launched; waiting"
  else
    waitmem; waitgpu $TRAIN_GPU
    tools/run_tmux.sh $job $TRAIN_GPU nestvq -- env NSAMPLES=1024 OFFLOAD=pinned SFX=_n1024 \
      $NUMA1 bash experiments/15_dsweep/run_llama_dsweep.sh $cfg $d $nb $L $obj || { echo "[chain] launch $job failed"; continue; }
  fi
  waitjob $job
  [[ -e $LOGS/e5_lds_eval_${cfg}_${obj}_n1024.meta.json ]] && { echo "[chain] eval of $job already launched"; continue; }
  if [[ "$(exitcode $job)" == 0 ]]; then
    ( waitgpu $EVAL_GPUS
      tools/run_tmux.sh e5_lds_eval_${cfg}_${obj}_n1024 $EVAL_GPUS nestvq -- env NSAMPLES=1024 SFX=_n1024 REF_MAXMEM=${REF_MAXMEM:-} \
        $NUMA0 bash experiments/15_dsweep/run_llama_eval.sh $cfg $d $nb $L $obj ) &
  fi
done
wait
echo "[chain] done $(date -Is)"
