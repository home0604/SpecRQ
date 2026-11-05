"""Standard result record for SpecRQ experiments (maps onto the Notion 03 results table).

Usage:
    from tools.results import save_result
    save_result("E0", "qwen_shapes_m1-8", config={...}, metrics={...}, model_dir="/data/models/qwen2.5_0.5b_instruct")

Writes results/<exp>/<run>.json with provenance (git commit, dirty flag, GPU, library versions,
model revision, tmux job name). Refuses to overwrite an existing record.

Notion 03 columns -> fields:
  모델·설정 -> config.model, config.setting | B·T·γ -> config.{B,T,gamma} | d·K·L·k·q -> config.{d,K,L,k,q}
  Target 품질 -> metrics.target_quality | Acceptance/τ -> metrics.{acceptance,tau}
  Draft/verify/overhead -> metrics.{t_draft_round_ms,t_verify_ms,t_overhead_ms}
  tokens/s / AR 대비 -> metrics.{tokens_per_s,speedup_vs_ar} | Peak memory -> metrics.peak_mem_gib
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout.strip()


def _versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "triton", "aqlm"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
                capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return out


def _model_revision(model_dir: str | None) -> str | None:
    if not model_dir:
        return None
    p = Path(model_dir) / "REVISION.txt"
    return p.read_text().strip() if p.exists() else None


def save_result(exp: str, run: str, config: dict, metrics: dict, model_dir: str | None = None,
                notes: str = "") -> Path:
    path = REPO / "results" / exp / f"{run}.json"
    if path.exists():
        raise FileExistsError(f"{path} exists; use a new run name")
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "exp": exp,
        "run": run,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD") or None,
        "dirty_files": len(_git("status", "--porcelain").splitlines()),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tmux_job": os.environ.get("TMUX") and subprocess.run(
            ["tmux", "display-message", "-p", "#S"], capture_output=True, text=True).stdout.strip(),
        "versions": _versions(),
        "model_dir": model_dir,
        "model_revision": _model_revision(model_dir),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return path
