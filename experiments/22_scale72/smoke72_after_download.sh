#!/usr/bin/env bash
# Wait for the 72B download (REVISION.txt is written only after `hf download` succeeds), then quantize layers 0..QNL-1
# of Qwen2.5-72B with run72.sh into the real ckpt dir (resumable later with the same TAG). Single GPU (NDEV=1).
set -euo pipefail
DST=/data/models/qwen2.5_72b_instruct
until [[ -e $DST/REVISION.txt ]]; do sleep 60; done
echo "[smoke72] download done: $(cat $DST/REVISION.txt | tr '\n' ' ') $(date -Is)"
exec env TAG=qwen72b MODEL=$DST QNL=${QNL:-2} NDEV=1 bash /hai/home/lsh/SpecRQ/experiments/22_scale72/run72.sh
