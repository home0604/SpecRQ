# E10 phase 2: Nsight Compute, DEC 0 vs DEC 1 at identical decomposition (KST1): Nsight Compute on representative points (RTX 4090, ncu 2023 / CUDA 12.1, --set full)

Point name = k{KST}_n{tokens}_bh{B*Hkv}_c{chunk}; single launch (ncu serializes and flushes caches, so times differ from the graph-timed sweep).

| point | time us | DRAM %peak | DRAM rd MB | DRAM wr MB | achieved occ % | issue active % | L1/smem thr % | FMA pipe % | smem ld wavefronts M | smem ld bank-conflict M | inst M | top stalls (cycles per issued inst) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1_w16_s0_d0_p0_f0_n131072_bh32_c32768 | 189.2 | 72.9 | 134.3 | 1.1 | 33.3 | 72.7 | 83.9 | 49.6 | 40.2 | 21.0 | 153.0 | short_scoreboard 0.90, dispatch_stall 0.55, wait 0.40, math_pipe_throttle 0.36 |
| k1_w16_s0_d1_p0_f0_n131072_bh32_c128 | 179.3 | 77.0 | 134.3 | 1.3 | 33.3 | 76.6 | 89.0 | 52.6 | 40.2 | 21.0 | 152.3 | short_scoreboard 1.14, wait 0.41, math_pipe_throttle 0.29, dispatch_stall 0.28 |
| k1_w16_s0_d1_p0_f0_n4096_bh32_c128 | 17.3 | 25.9 | 4.2 | 0.0 | 33.3 | 46.2 | 52.7 | 27.7 | 1.6 | 0.7 | 7.3 | long_scoreboard 1.83, barrier 1.12, short_scoreboard 0.95, wait 0.76 |
| k1_w8_s0_d0_p0_f0_n131072_bh32_c32768 | 225.4 | 60.9 | 134.3 | 0.6 | 16.6 | 60.4 | 69.7 | 41.3 | 40.2 | 21.0 | 152.5 | short_scoreboard 0.71, wait 0.39, dispatch_stall 0.30, math_pipe_throttle 0.12 |
| k1_w8_s0_d0_p0_f0_n4096_bh32_c1024 | 16.4 | 27.3 | 4.2 | 0.0 | 16.5 | 42.5 | 48.8 | 26.4 | 1.6 | 0.7 | 6.5 | long_scoreboard 0.78, short_scoreboard 0.76, wait 0.61, barrier 0.31 |
