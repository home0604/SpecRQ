#!/usr/bin/env bash
# Launch a long job in a detached tmux session with a log and run metadata.
# Usage: tools/run_tmux.sh <name> <gpu-ids|none> <env> -- <command...>
#   name : short id, e.g. e2_qwen_c1_smoke  -> tmux session "sr-<name>"
#   gpus : CUDA_VISIBLE_DEVICES value (e.g. 0 or 0,1) or "none"; refuses GPUs with >2 GB in use
#   env  : conda env name (e.g. nestvq)
# Files: /data/lsh/logs/<name>.{log,meta.json,sh}. meta gets end/exit_code on finish.
# Monitor: tmux attach -t sr-<name> | tail -f /data/lsh/logs/<name>.log | tools/jobs.sh
set -euo pipefail
[[ $# -ge 5 && "$4" == "--" ]] || { echo "usage: $0 <name> <gpus|none> <env> -- <cmd...>"; exit 2; }
NAME=$1; GPUS=$2; ENV=$3; shift 4
REPO=/hai/home/lsh/SpecRQ; LOGDIR=/data/lsh/logs; mkdir -p "$LOGDIR"
SESS="sr-$NAME"; LOG="$LOGDIR/$NAME.log"; META="$LOGDIR/$NAME.meta.json"; RUNNER="$LOGDIR/$NAME.sh"
tmux has-session -t "=$SESS" 2>/dev/null && { echo "session $SESS already exists"; exit 1; }
[[ -e "$LOG" || -e "$META" ]] && { echo "$LOG exists; pick a new name"; exit 1; }
if [[ "$GPUS" != none ]]; then
  for g in ${GPUS//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
    (( used > 2000 )) && { echo "GPU $g busy (${used} MiB used); aborting"; exit 1; }
  done
fi
CMD=$(printf '%q ' "$@")
python3 - "$META" "$NAME" "$SESS" "$GPUS" "$ENV" "$CMD" "$REPO" <<'PY'
import json,sys,datetime,subprocess,socket
meta,name,sess,gpus,env,cmd,repo=sys.argv[1:]
g=lambda *a: subprocess.run(["git","-C",repo,*a],capture_output=True,text=True).stdout.strip()
json.dump({"name":name,"session":sess,"gpus":gpus,"env":env,"cmd":cmd,"commit":g("rev-parse","--short","HEAD") or None,
 "dirty_files":len(g("status","--porcelain").splitlines()),"host":socket.gethostname(),
 "start":datetime.datetime.now().isoformat(timespec="seconds")},open(meta,"w"),indent=1)
PY
cat > "$RUNNER" <<RUN
#!/usr/bin/env bash
source /hai/anaconda3/etc/profile.d/conda.sh
conda activate $ENV
cd $REPO
export HF_HOME=/data/lsh/hf_cache PYTHONUNBUFFERED=1
T=/data/lsh/temp; export TMPDIR=\$T/tmp TORCHINDUCTOR_CACHE_DIR=\$T/inductor_cache TRITON_CACHE_DIR=\$T/triton_cache TORCH_EXTENSIONS_DIR=\$T/torch_extensions PIP_CACHE_DIR=\$T/pip_cache XDG_CACHE_HOME=\$T/xdg_cache
$( [[ "$GPUS" != none ]] && echo "export CUDA_VISIBLE_DEVICES=$GPUS" )
$CMD 2>&1 | tee -a $LOG
rc=\${PIPESTATUS[0]}
python3 -c "import json,datetime,sys;p='$META';m=json.load(open(p));m['end']=datetime.datetime.now().isoformat(timespec='seconds');m['exit_code']=int(sys.argv[1]);json.dump(m,open(p,'w'),indent=1)" \$rc
echo "[run_tmux] exit=\$rc" | tee -a $LOG
RUN
tmux new-session -d -s "$SESS" "bash $RUNNER; exec bash"
echo "started $SESS  log=$LOG"
