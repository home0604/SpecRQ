"""Co-tenant check for timing runs: other compute processes on the GPUs visible to this process (by UUID).
Usage: from specrq.nestkern.gpucheck import cotenants, require_alone
"""
from __future__ import annotations

import os
import subprocess


def _visible_uuids():
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip().splitlines()
    idx = {r.split(", ")[0]: r.split(", ")[1] for r in rows}
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [idx[i.strip()] for i in vis.split(",")] if vis else list(idx.values())


def cotenants():
    """[(pid, used_memory)] of compute apps on our visible GPUs, excluding this process."""
    uu = set(_visible_uuids())
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    me = os.getpid()
    res = []
    for line in out:
        pid, mem, uuid = [x.strip() for x in line.split(",")]
        if uuid in uu and int(pid) != me:
            res.append((int(pid), mem))
    return res


def require_alone():
    c = cotenants()
    if c:
        raise SystemExit(f"[gpucheck] co-tenant on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}: {c}; timing aborted")
    return "none"
