#!/usr/bin/env bash
# Download Qwen/Qwen2.5-72B-Instruct to /data/models/qwen2.5_72b_instruct and record the revision.
set -euo pipefail
REPO=Qwen/Qwen2.5-72B-Instruct; DST=/data/models/qwen2.5_72b_instruct
export HF_HOME=/data/lsh/hf_cache HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
REV=$(python -c "from huggingface_hub import HfApi;print(HfApi().model_info('$REPO').sha)")
echo "[dl] $REPO rev=$REV $(date -Is)"
hf download $REPO --revision $REV --local-dir $DST --max-workers ${MAXW:-8}
printf 'repo=%s\nrevision=%s\ndownloaded=%s\n' $REPO $REV "$(date -Is)" > $DST/REVISION.txt
du -sh $DST; echo "[dl] done $(date -Is)"
