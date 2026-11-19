# E10 phase 1: Nsight Compute on representative points (RTX 4090, ncu 2023 / CUDA 12.1, --set full)

Point name = k{KST}_n{tokens}_bh{B*Hkv}_c{chunk}; single launch (ncu serializes and flushes caches, so times differ from the graph-timed sweep).

| point | time us | DRAM %peak | DRAM rd MB | DRAM wr MB | achieved occ % | issue active % | L1/smem thr % | FMA pipe % | smem ld wavefronts M | smem ld bank-conflict M | inst M | top stalls (cycles per issued inst) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1_w16_b1_r0_p0_l0_s0_n131072_bh256_c131072 | 1731.4 | 63.4 | 1074.4 | 4.5 | 33.0 | 68.9 | 79.6 | 45.4 | 319.5 | 167.9 | 1354.4 | short_scoreboard 1.55, wait 0.57, long_scoreboard 0.53, dispatch_stall 0.28 |
| k1_w16_b1_r0_p0_l0_s2_n131072_bh256_c131072 | 1440.4 | 76.2 | 1074.3 | 4.4 | 32.3 | 73.4 | 85.4 | 50.7 | 319.5 | 167.9 | 1199.9 | short_scoreboard 0.95, long_scoreboard 0.65, dispatch_stall 0.39, wait 0.37 |
| k1_w16_b1_r1_p0_l0_s0_n131072_bh256_c131072 | 1746.5 | 62.7 | 1074.1 | 2.9 | 33.3 | 68.3 | 78.9 | 45.0 | 319.7 | 167.9 | 1354.7 | short_scoreboard 1.49, wait 0.61, long_scoreboard 0.53, dispatch_stall 0.36 |
| k1_w16_b1_r1_p0_l0_s1_n131072_bh256_c131072 | 1455.9 | 75.3 | 1074.1 | 2.9 | 33.3 | 71.6 | 84.6 | 50.0 | 319.9 | 167.9 | 1182.8 | short_scoreboard 0.94, long_scoreboard 0.65, dispatch_stall 0.51, wait 0.38 |
| k1_w8_b1_r1_p1_l0_s2_n131072_bh256_c131072 | 1439.6 | 76.1 | 1074.3 | 1.7 | 30.6 | 73.9 | 85.5 | 50.7 | 319.5 | 167.9 | 1209.5 | short_scoreboard 0.92, dispatch_stall 0.53, wait 0.39, math_pipe_throttle 0.34 |
| k1_w8_b1_r1_p1_l0_s2_n32768_bh64_c16384 | 118.0 | 58.4 | 67.2 | 0.1 | 16.6 | 59.4 | 68.4 | 40.4 | 20.3 | 10.5 | 77.2 | short_scoreboard 0.71, wait 0.40, dispatch_stall 0.30, math_pipe_throttle 0.12 |
| k1_w8_b1_r1_p1_l0_s2_n32768_bh8_c2048 | 23.4 | 37.9 | 8.4 | 0.0 | 16.5 | 49.1 | 56.4 | 31.8 | 2.8 | 1.3 | 11.2 | short_scoreboard 0.73, wait 0.51, long_scoreboard 0.45, dispatch_stall 0.28 |
| k2_w16_b1_r0_p0_l0_s0_n131072_bh256_c131072 | 2609.1 | 84.0 | 2147.9 | 4.3 | 33.1 | 62.8 | 94.2 | 39.3 | 607.9 | 335.8 | 1857.1 | short_scoreboard 2.08, mio_throttle 0.49, wait 0.46, long_scoreboard 0.45 |
| k2_w16_b1_r0_p0_l0_s2_n131072_bh256_c131072 | 2352.9 | 93.2 | 2147.8 | 4.4 | 32.9 | 63.9 | 98.2 | 41.2 | 608.3 | 335.8 | 1703.7 | short_scoreboard 1.54, mio_throttle 0.80, long_scoreboard 0.70, wait 0.31 |
| k2_w16_b1_r1_p0_l0_s0_n131072_bh256_c131072 | 2594.5 | 84.3 | 2147.9 | 2.9 | 33.3 | 63.0 | 94.5 | 39.4 | 608.2 | 335.8 | 1859.5 | short_scoreboard 2.00, wait 0.49, long_scoreboard 0.44, dispatch_stall 0.42 |
| k2_w16_b1_r1_p0_l0_s1_n131072_bh256_c131072 | 2353.8 | 93.3 | 2147.9 | 2.9 | 33.3 | 63.1 | 97.9 | 41.1 | 607.9 | 335.8 | 1685.4 | short_scoreboard 1.44, long_scoreboard 0.70, mio_throttle 0.64, dispatch_stall 0.45 |
| k2_w8_b1_r1_p1_l0_s2_n131072_bh256_c131072 | 2463.5 | 88.7 | 2148.1 | 1.6 | 16.7 | 61.2 | 93.4 | 39.2 | 608.1 | 335.8 | 1718.5 | short_scoreboard 0.94, wait 0.31, dispatch_stall 0.19, math_pipe_throttle 0.15 |
| k2_w8_b1_r1_p1_l0_s2_n32768_bh64_c16384 | 167.1 | 82.6 | 134.3 | 0.6 | 16.6 | 59.5 | 90.3 | 37.9 | 38.5 | 21.0 | 109.7 | short_scoreboard 0.94, wait 0.33, dispatch_stall 0.19, math_pipe_throttle 0.14 |
| k2_w8_b1_r1_p1_l0_s2_n32768_bh8_c2048 | 32.8 | 53.5 | 16.8 | 0.0 | 16.6 | 50.1 | 74.1 | 31.1 | 5.3 | 2.6 | 16.0 | short_scoreboard 0.91, long_scoreboard 0.48, wait 0.44, dispatch_stall 0.19 |
| orig_k1_n131072_bh256_c131072 | 1772.4 | 61.8 | 1074.1 | 4.4 | 33.1 | 68.2 | 79.3 | 44.6 | 321.2 | 167.9 | 1361.7 | short_scoreboard 1.59, wait 0.71, long_scoreboard 0.52, dispatch_stall 0.34 |
| orig_k1_n32768_bh64_c16384 | 122.2 | 55.9 | 67.2 | 0.0 | 32.9 | 65.6 | 76.3 | 42.6 | 20.3 | 10.5 | 86.8 | short_scoreboard 1.58, wait 0.72, long_scoreboard 0.61, dispatch_stall 0.34 |
| orig_k1_n32768_bh8_c2048 | 26.5 | 32.7 | 8.4 | 0.0 | 31.9 | 49.5 | 58.7 | 31.0 | 2.8 | 1.3 | 12.5 | short_scoreboard 1.59, long_scoreboard 1.18, wait 0.77, barrier 0.66 |
| orig_k2_n131072_bh256_c131072 | 2670.4 | 81.9 | 2147.9 | 4.4 | 33.1 | 61.4 | 92.0 | 38.4 | 607.7 | 335.8 | 1857.1 | short_scoreboard 2.13, long_scoreboard 0.90, wait 0.55, dispatch_stall 0.31 |
| orig_k2_n32768_bh64_c16384 | 184.2 | 74.9 | 134.3 | 0.0 | 32.8 | 59.0 | 88.0 | 36.7 | 38.5 | 21.0 | 118.4 | short_scoreboard 2.07, long_scoreboard 1.12, wait 0.56, dispatch_stall 0.31 |
| orig_k2_n32768_bh8_c2048 | 36.5 | 47.9 | 16.8 | 0.0 | 31.6 | 46.9 | 69.8 | 28.8 | 5.3 | 2.6 | 17.1 | long_scoreboard 2.61, short_scoreboard 1.68, wait 0.62, barrier 0.60 |
