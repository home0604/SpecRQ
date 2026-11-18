#!/usr/bin/env bash
# Set up SpecRQ (E7 MagicDec-VQ) on a new server. See HANDOFF_A100.md.
# Usage: tools/setup_server.sh <DATA_ROOT> <MODELS_ROOT> [ARTIFACT_TAR]
#   DATA_ROOT   : heavy outputs / datasets / temp (replaces /data/lsh on the original server "shark")
#   MODELS_ROOT : model dir root (replaces /data/models); the model goes to $MODELS_ROOT/llama3.1_8b_instruct
#   ARTIFACT_TAR: specrq_e7_artifacts.tar copied from shark:/data/lsh/handoff/ (codebooks, prompts, PG-19, LongBench, raw tau/timing)
# Steps: dirs -> artifacts -> path mapping (symlink /data/lsh,/data/models if possible, else rewrite literals in the code)
#        -> conda env nestvq -> model download (needs your own HF access to the gated meta-llama repo).
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n 2,9p "$0"; exit 2; }
DATA=$(realpath -m "$1"); MODELS=$(realpath -m "$2"); TAR=${3:-}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$DATA"/{logs,temp/{tmp,inductor_cache,triton_cache,torch_extensions,pip_cache,xdg_cache},hf_cache,datasets,results_accept} "$MODELS"

# 1) artifacts (paths inside the tar are relative to /data/lsh)
if [[ -n "$TAR" ]]; then
  [[ -f "$TAR.sha256" ]] && (cd "$(dirname "$TAR")" && sha256sum -c "$(basename "$TAR").sha256")
  tar -xf "$TAR" -C "$DATA"
  (cd "$DATA/datasets/longbench" && [[ -d data ]] || unzip -q data.zip)
fi

# 2) path mapping. The code hardcodes /data/lsh, /data/models and /hai/home/lsh/SpecRQ.
map() {  # $1 = original path, $2 = new path
  [[ "$1" == "$2" ]] && return 0
  if [[ -e "$1" ]]; then echo "[setup] $1 exists; leaving it"; return 0; fi
  if mkdir -p "$(dirname "$1")" 2>/dev/null && ln -s "$2" "$1" 2>/dev/null; then echo "[setup] symlink $1 -> $2"; return 0; fi
  echo "[setup] cannot create $1; rewriting the literal in code files (local change, do not commit it)"
  grep -rlI --include=*.py --include=*.sh --include=*.md -F "$1" "$REPO/experiments" "$REPO/specrq" "$REPO/tools" 2>/dev/null \
    | xargs -r sed -i "s#$1#$2#g"
}
map /data/lsh "$DATA"
map /data/models "$MODELS"
map /hai/home/lsh/SpecRQ "$REPO"
echo "export SPECRQ_DATA=$DATA" > "$REPO/.specrq_env"; echo "[setup] wrote $REPO/.specrq_env (source it; also set SPECRQ_CONDA_SH if conda is not at /hai/anaconda3)"

# 3) env
if command -v conda >/dev/null && ! conda env list | grep -q '^nestvq '; then
  conda env create -n nestvq -f "$REPO/env/nestvq.yml" || echo "[setup] conda env create failed; see env/nestvq_pip_freeze.txt"
fi

# 4) model (gated: your own HF login / token must have access; no workarounds)
M="$MODELS/llama3.1_8b_instruct"
if [[ ! -f "$M/model.safetensors.index.json" ]]; then
  echo "[setup] downloading meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659 -> $M"
  HF_HOME="$DATA/hf_cache" huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --revision 0e9e39f249a16976918f6564b8830bc894c89659 --local-dir "$M" --exclude "original/*" \
    && printf 'repo=meta-llama/Llama-3.1-8B-Instruct\nrevision=0e9e39f249a16976918f6564b8830bc894c89659\n' > "$M/REVISION.txt"
fi
echo "[setup] done. For A100 builds: export TORCH_CUDA_ARCH_LIST=8.0 (code defaults to 8.9 via setdefault)."
