#!/usr/bin/env bash
# CPU part of the offload micro-benchmark: node1 (CPUs 16-31 physical, 48-63 HT siblings), first-touch local memory.
set -euo pipefail
cd /hai/home/lsh/SpecRQ
OUT=${1:-/data/lsh/temp/tmp/offload}
mkdir -p "$OUT"
uptime | tee "$OUT/uptime_start.txt"
export OMP_PROC_BIND=close OMP_PLACES=cores
for T in 8 16 32; do
  if [[ $T == 32 ]]; then export OMP_PLACES=threads; else export OMP_PLACES=cores; fi
  OMP_NUM_THREADS=$T nice -n 10 taskset -c 16-31,48-63 python experiments/18_offload/bench_cpu.py \
    --part stream check sweep --kinds 0 1 3 --Ms 1 2 3 4 5 8 16 --reps 5 --out "$OUT/cpu_t$T.json"
done
OMP_NUM_THREADS=16 OMP_PLACES=cores nice -n 10 taskset -c 16-31 python experiments/18_offload/bench_cpu.py \
  --part torch --Ms 1 4 16 --reps 5 --out "$OUT/cpu_torch_t16.json"
uptime | tee "$OUT/uptime_end.txt"
