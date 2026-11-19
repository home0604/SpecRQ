#!/usr/bin/env bash
# E10 phase 3 A100 handoff: correctness + timing of the original LUT VQ-KV kernel vs the phase-1 struct vs the phase-3
# variants (tensor-core P.V, packed HFMA2, V-codebook replicas, half2 scores) + timing-only ablations.
# Run from the repo root in the env that runs experiments/26_magicvq (torch + triton + ninja + CUDA toolkit), on an
# otherwise EMPTY A100:   GPU=<id> bash experiments/32_kernel_tiling/a100_p3_bench.sh
# Writes results/E10/a100_p3_{check,bench}.json and a100_p3_tables.md (prefix: OUTP=a100_p3).
# Approx. time: build ~3-5 min (60 variants), check ~5 min, timing ~15-25 min.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${GPU:?set GPU=<free A100 index>}"
export CUDA_VISIBLE_DEVICES=$GPU TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0} PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MAX_JOBS=${MAX_JOBS:-16}
# separate source / build dirs so a 4090 (sm_89) build is never reused
export E10_SRC_DIR=${E10_SRC_DIR:-${TMPDIR:-/tmp}/e10_p3_src_a100}
D=experiments/32_kernel_tiling
O=results/E10/${OUTP:-a100_p3}   # output prefix
echo "== nvidia-smi before (the timing GPU must have no other process)"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
nvidia-smi -i "$GPU" --query-gpu=name,uuid,memory.used,clocks.sm,clocks.max.sm --format=csv
VARS=$(cd $D && python p3_sets.py a100)
EXACT=$(cd $D && python p3_sets.py exact)
echo "== 1. correctness (p3 variants vs original vs fp64 reference)"
python $D/check_p3.py --out ${O}_check.json --variants $EXACT
echo "== 2. timing at the reference shape B16 x Hkv2 (bh 32), n 32k/64k/128k, KST 1/2:"
echo "   splits {4, 8, 16} + the chunk set / split_plan / rule candidates (oracle = min); long warm-up (bimodal first replays)"
python $D/bench_p3.py --out ${O}_bench.json --orig --orig_lazy --p1 --no_part \
  --bh 32 --n 32768 65536 131072 --kst 1 2 --splits 4 8 16 --warm_ms 500 --reps 15 --variants $VARS
echo "== 3. tables"
python $D/analyze_p3.py --raw ${O}_bench.json --check ${O}_check.json \
  --out ${O}_tables.md
echo "== nvidia-smi after"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
