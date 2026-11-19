# E10 phase 0: Nsight Compute on representative points (RTX 4090, ncu 2023 / CUDA 12.1, --set full)

Point name = k{KST}_n{tokens}_bh{B*Hkv}_c{chunk}; single launch (ncu serializes and flushes caches, so times differ from the graph-timed sweep).

| point | time us | DRAM %peak | DRAM rd MB | DRAM wr MB | achieved occ % | issue active % | L1/smem thr % | FMA pipe % | smem ld wavefronts M | smem ld bank-conflict M | inst M | top stalls (cycles per issued inst) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1_n131072_bh256_c8192 | 1969.8 | 62.3 | 1076.7 | 128.4 | 32.8 | 63.9 | 74.6 | 41.3 | 329.4 | 167.9 | 1418.9 | short_scoreboard 1.60, wait 0.73, long_scoreboard 0.63, dispatch_stall 0.33 |
| k1_n32768_bh128_c1024 | 432.6 | 56.8 | 134.4 | 106.9 | 32.0 | 48.5 | 58.4 | 29.4 | 48.9 | 21.0 | 230.7 | short_scoreboard 1.56, long_scoreboard 0.88, barrier 0.87, wait 0.82 |
| k1_n32768_bh32_c8192 | 67.1 | 51.1 | 33.6 | 0.0 | 32.7 | 62.5 | 72.9 | 40.3 | 10.3 | 5.2 | 44.3 | short_scoreboard 1.60, wait 0.73, long_scoreboard 0.71, dispatch_stall 0.33 |
| k2_n131072_bh256_c8192 | 2921.8 | 79.6 | 2152.0 | 131.7 | 32.6 | 58.9 | 87.6 | 36.4 | 623.8 | 335.9 | 1936.8 | short_scoreboard 1.99, long_scoreboard 1.14, wait 0.57, dispatch_stall 0.30 |
| k2_n32768_bh32_c8192 | 98.8 | 69.2 | 67.2 | 0.0 | 32.5 | 56.6 | 84.3 | 35.0 | 19.5 | 10.5 | 60.5 | short_scoreboard 1.98, long_scoreboard 1.41, wait 0.57, dispatch_stall 0.29 |
