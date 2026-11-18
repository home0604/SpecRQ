#!/usr/bin/env bash
# E7 MagicDec repro: run one OFFICIAL MagicDec benchmark (unmodified code at /data/lsh/src/MagicDec) with TP=4 on the
# visible GPUs (CUDA_VISIBLE_DEVICES=4,5,6,7 set by tools/run_tmux.sh), with a GPU co-tenant check before and after.
# Usage: run_official.sh <tag> <tests/...py> <args...>      (env /data/lsh/envs/magicdec must be active)
# Writes /data/lsh/logs/magicdec/<tag>.out (stdout of rank 0) and <tag>.co.json (nvidia-smi snapshots before/after).
# SMOKE=1 skips the wait (correctness runs only); NPROC overrides the TP size.
# If another process is on GPUs 4-7 before the start, waits (poll every 15 min, max 6 h) instead of timing.
set -uo pipefail
TAG=$1; SCRIPT=$2; shift 2
OD=/data/lsh/logs/magicdec; mkdir -p "$OD"
snap() { python3 - <<'PY'
import json, subprocess, time
q = lambda a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
idx = {l.split(", ")[1]: l.split(", ")[0] for l in q(["--query-gpu=index,uuid"])}
apps = [dict(zip(("pid", "mem", "gpu"), (p, m, idx.get(u.strip())))) for p, m, u in (l.split(", ") for l in q(["--query-compute-apps=pid,used_memory,gpu_uuid"]) if l)]
gpus = q(["--query-gpu=index,memory.used,utilization.gpu"])
oth = [a for a in apps if a["gpu"] in ("4", "5", "6", "7")]
print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cotenant": bool(oth), "others": oth, "gpus": gpus}))
PY
}
waited=0; BEFORE=$(snap)
while [[ -z "${SMOKE:-}" ]]; do
  BEFORE=$(snap)
  echo "$BEFORE" | grep -q '"cotenant": false' && break
  (( waited >= 21600 )) && { echo "[run_official] GPUs 4-7 busy for >6h, abort: $BEFORE"; exit 3; }
  echo "[run_official] co-tenant on GPUs 4-7, waiting 15 min: $BEFORE"; sleep 900; waited=$((waited + 900))
done
cd /data/lsh/src/MagicDec
T0=$(date +%s)
env NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 \
  nice -n 10 taskset -c ${CPUSET:-16-31,48-63} torchrun --standalone --nproc_per_node=${NPROC:-4} "$SCRIPT" "$@" 2>&1 | tee "$OD/$TAG.out"
rc=${PIPESTATUS[0]}
AFTER=$(snap)
python3 -c "import json,sys; json.dump({'tag':sys.argv[1],'script':sys.argv[2],'args':sys.argv[3],'rc':int(sys.argv[4]),'wall_s':int(sys.argv[5]),'waited_s':int(sys.argv[6]),'before':json.loads(sys.argv[7]),'after':json.loads(sys.argv[8])},open(sys.argv[9],'w'),indent=1)" \
  "$TAG" "$SCRIPT" "$*" "$rc" "$(( $(date +%s) - T0 ))" "$waited" "$BEFORE" "$AFTER" "$OD/$TAG.co.json"
echo "[run_official] $TAG rc=$rc"
exit $rc
