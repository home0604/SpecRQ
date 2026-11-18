#!/usr/bin/env bash
# List SpecRQ tmux jobs with status from their meta files.
for m in $(ls -t ${SPECRQ_DATA:-/data/lsh}/logs/*.meta.json 2>/dev/null | head -${1:-15}); do
  python3 - "$m" <<'PY'
import json,sys,subprocess
m=json.load(open(sys.argv[1]))
alive=subprocess.run(["tmux","has-session","-t",m["session"]],capture_output=True).returncode==0
st="running" if "exit_code" not in m else f"exit={m['exit_code']}"
print(f"{m['name']:32s} {st:9s} tmux={'up' if alive else '--'}  gpu={m['gpus']:5s} start={m['start']} end={m.get('end','-')} commit={m['commit']}")
PY
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
