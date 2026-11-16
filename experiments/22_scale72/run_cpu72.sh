#!/usr/bin/env bash
# CPU residual-plane bench for 72B shapes (2 planes = stages 3-4) on NUMA node1 + load record (uptime, top on node1).
set -eu
cd /hai/home/lsh/SpecRQ
OUT=${1:-/data/lsh/temp/tmp/offload72}
mkdir -p "$OUT"
rec() { { date -Is; uptime; ps -eo pid,user,psr,pcpu,rss,etime,args --sort=-pcpu | awk 'NR==1 || ($3>=16 && $3<=31) || ($3>=48)' | head -12 | cut -c1-160 || true; } > "$OUT/load_$1.txt"; cat "$OUT/load_$1.txt"; }
rec start
export OMP_PROC_BIND=close
for T in 16 32; do
  if [[ $T == 32 ]]; then export OMP_PLACES=threads; else export OMP_PLACES=cores; fi
  OMP_NUM_THREADS=$T nice -n 10 taskset -c 16-31,48-63 python experiments/22_scale72/bench_offload72.py --target cpu \
    --part stream sweep --kinds 3 --Ms 1 2 3 4 5 8 --reps 5 --planes 2 --out "$OUT/cpu72_t$T.json"
  rec "after_t$T"
done
