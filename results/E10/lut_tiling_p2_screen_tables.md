# E10 phase 2: new structural axes of the LUT VQ-KV draft attention (NVIDIA GeForce RTX 4090)

- raw: `/data/lsh/results_accept/e10/lut_tiling_p2_screen_k1_raw.json`, `/data/lsh/results_accept/e10/lut_tiling_p2_screen_k2_raw.json`; metric `t_tot_us` (LUT build if any + partial + combine, us per call, median of 9 CUDA-graph reps, codes stream from DRAM); oracle runtime knob per struct.
- limits: {'sm_count': 128, 'smem_per_sm': 102400, 'smem_per_block_optin': 101376, 'reserved_smem_per_block': 1024, 'regs_per_sm': 65536, 'threads_per_sm': 1536, 'blocks_per_sm': 24, 'l2_bytes': 75497472, 'cc_major': 8, 'cc_minor': 9}; ref read BW 940, 941 GB/s
- `/data/lsh/results_accept/e10/lut_tiling_p2_screen_k1_raw.json`: GPU 6 uuid GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f; sweep 444 s; apps before ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 4036 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3673609, 21986 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 4036 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3673607, 6098 MiB']; after ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 4036 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3673609, 21986 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 4036 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3673607, 6098 MiB', 'GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f, 3731710, 1564 MiB']
- `/data/lsh/results_accept/e10/lut_tiling_p2_screen_k2_raw.json`: GPU 7 uuid GPU-babe0ba5-7c14-6768-fce4-c7515d249f25; sweep 214 s; apps before ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 4036 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3673609, 21986 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 4036 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3673607, 6098 MiB']; after ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 4036 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3673609, 21986 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 4036 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3673607, 6098 MiB', 'GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f, 3731710, 742 MiB', 'GPU-babe0ba5-7c14-6768-fce4-c7515d249f25, 3731722, 1562 MiB']
- struct name: w = warps/block, s = LUT source (0 build per block, 1 separate build kernel + copy from L2), d = decomposition (0 (bh, split) grid, 1 stream-K over flattened tokens), p = code pipeline (0 register prefetch = phase 1, 2/3 = cp.async ring stages issued before the LUT prologue), f = fused combine

## 0. Variant resources

| variant | regs | local B | smem B | blocks/SM | build regs |
|---|---|---|---|---|---|
| k1_w16_s0_d0_p0_f0 | 87 | 32 | 53328 | 1 | 0 |
| k1_w16_s0_d0_p0_f1 | 88 | 32 | 53328 | 1 | 0 |
| k1_w16_s0_d0_p2_f0 | 82 | 32 | 79952 | 1 | 0 |
| k1_w16_s0_d0_p2_f1 | 83 | 32 | 79952 | 1 | 0 |
| k1_w16_s0_d0_p3_f0 | 83 | 32 | 96336 | 1 | 0 |
| k1_w16_s0_d0_p3_f1 | 84 | 32 | 96336 | 1 | 0 |
| k1_w16_s0_d1_p0_f0 | 108 | 48 | 53328 | 1 | 0 |
| k1_w16_s0_d1_p0_f1 | 108 | 32 | 53328 | 1 | 0 |
| k1_w16_s0_d1_p2_f0 | 106 | 48 | 79952 | 1 | 0 |
| k1_w16_s0_d1_p2_f1 | 104 | 48 | 79952 | 1 | 0 |
| k1_w16_s0_d1_p3_f0 | 106 | 48 | 96336 | 1 | 0 |
| k1_w16_s0_d1_p3_f1 | 104 | 48 | 96336 | 1 | 0 |
| k1_w16_s1_d0_p0_f0 | 85 | 16 | 53328 | 1 | 51 |
| k1_w16_s1_d0_p0_f1 | 87 | 16 | 53328 | 1 | 51 |
| k1_w16_s1_d0_p2_f0 | 82 | 16 | 79952 | 1 | 51 |
| k1_w16_s1_d0_p2_f1 | 84 | 16 | 79952 | 1 | 51 |
| k1_w16_s1_d0_p3_f0 | 83 | 16 | 96336 | 1 | 51 |
| k1_w16_s1_d0_p3_f1 | 85 | 16 | 96336 | 1 | 51 |
| k1_w16_s1_d1_p0_f0 | 112 | 16 | 53328 | 1 | 51 |
| k1_w16_s1_d1_p0_f1 | 110 | 16 | 53328 | 1 | 51 |
| k1_w16_s1_d1_p2_f0 | 115 | 16 | 79952 | 1 | 51 |
| k1_w16_s1_d1_p2_f1 | 111 | 16 | 79952 | 1 | 51 |
| k1_w16_s1_d1_p3_f0 | 115 | 16 | 96336 | 1 | 51 |
| k1_w16_s1_d1_p3_f1 | 111 | 16 | 96336 | 1 | 51 |
| k1_w4_s0_d0_p0_f0 | 88 | 32 | 41040 | 2 | 0 |
| k1_w4_s0_d0_p0_f1 | 88 | 32 | 41040 | 2 | 0 |
| k1_w4_s0_d0_p2_f0 | 84 | 32 | 49232 | 2 | 0 |
| k1_w4_s0_d0_p2_f1 | 85 | 32 | 49232 | 2 | 0 |
| k1_w4_s0_d0_p3_f0 | 85 | 32 | 53328 | 1 | 0 |
| k1_w4_s0_d0_p3_f1 | 86 | 32 | 53328 | 1 | 0 |
| k1_w4_s0_d1_p0_f0 | 102 | 48 | 41040 | 2 | 0 |
| k1_w4_s0_d1_p0_f1 | 103 | 32 | 41040 | 2 | 0 |
| k1_w4_s0_d1_p2_f0 | 105 | 48 | 49232 | 2 | 0 |
| k1_w4_s0_d1_p2_f1 | 104 | 48 | 49232 | 2 | 0 |
| k1_w4_s0_d1_p3_f0 | 104 | 48 | 53328 | 1 | 0 |
| k1_w4_s0_d1_p3_f1 | 104 | 48 | 53328 | 1 | 0 |
| k1_w4_s1_d0_p0_f0 | 89 | 16 | 41040 | 2 | 51 |
| k1_w4_s1_d0_p0_f1 | 91 | 16 | 41040 | 2 | 51 |
| k1_w4_s1_d0_p2_f0 | 84 | 16 | 49232 | 2 | 51 |
| k1_w4_s1_d0_p2_f1 | 86 | 16 | 49232 | 2 | 51 |
| k1_w4_s1_d0_p3_f0 | 85 | 16 | 53328 | 1 | 51 |
| k1_w4_s1_d0_p3_f1 | 87 | 16 | 53328 | 1 | 51 |
| k1_w4_s1_d1_p0_f0 | 111 | 16 | 41040 | 2 | 51 |
| k1_w4_s1_d1_p0_f1 | 107 | 16 | 41040 | 2 | 51 |
| k1_w4_s1_d1_p2_f0 | 114 | 16 | 49232 | 2 | 51 |
| k1_w4_s1_d1_p2_f1 | 113 | 16 | 49232 | 2 | 51 |
| k1_w4_s1_d1_p3_f0 | 116 | 16 | 53328 | 1 | 51 |
| k1_w4_s1_d1_p3_f1 | 113 | 16 | 53328 | 1 | 51 |
| k1_w8_s0_d0_p0_f0 | 88 | 32 | 45136 | 2 | 0 |
| k1_w8_s0_d0_p0_f1 | 89 | 32 | 45136 | 2 | 0 |
| k1_w8_s0_d0_p2_f0 | 84 | 32 | 59472 | 1 | 0 |
| k1_w8_s0_d0_p2_f1 | 84 | 32 | 59472 | 1 | 0 |
| k1_w8_s0_d0_p3_f0 | 85 | 32 | 67664 | 1 | 0 |
| k1_w8_s0_d0_p3_f1 | 85 | 32 | 67664 | 1 | 0 |
| k1_w8_s0_d1_p0_f0 | 102 | 48 | 45136 | 2 | 0 |
| k1_w8_s0_d1_p0_f1 | 103 | 32 | 45136 | 2 | 0 |
| k1_w8_s0_d1_p2_f0 | 105 | 48 | 59472 | 1 | 0 |
| k1_w8_s0_d1_p2_f1 | 104 | 48 | 59472 | 1 | 0 |
| k1_w8_s0_d1_p3_f0 | 104 | 48 | 67664 | 1 | 0 |
| k1_w8_s0_d1_p3_f1 | 104 | 48 | 67664 | 1 | 0 |
| k1_w8_s1_d0_p0_f0 | 89 | 16 | 45136 | 2 | 51 |
| k1_w8_s1_d0_p0_f1 | 91 | 16 | 45136 | 2 | 51 |
| k1_w8_s1_d0_p2_f0 | 83 | 16 | 59472 | 1 | 51 |
| k1_w8_s1_d0_p2_f1 | 85 | 16 | 59472 | 1 | 51 |
| k1_w8_s1_d0_p3_f0 | 84 | 16 | 67664 | 1 | 51 |
| k1_w8_s1_d0_p3_f1 | 86 | 16 | 67664 | 1 | 51 |
| k1_w8_s1_d1_p0_f0 | 112 | 16 | 45136 | 2 | 51 |
| k1_w8_s1_d1_p0_f1 | 107 | 16 | 45136 | 2 | 51 |
| k1_w8_s1_d1_p2_f0 | 114 | 16 | 59472 | 1 | 51 |
| k1_w8_s1_d1_p2_f1 | 113 | 16 | 59472 | 1 | 51 |
| k1_w8_s1_d1_p3_f0 | 116 | 16 | 67664 | 1 | 51 |
| k1_w8_s1_d1_p3_f1 | 113 | 16 | 67664 | 1 | 51 |
| orig_k1 | 76 | 0 | 53312 | 1 | - |
| p1:k1_w8_b1_r1_p1_l0_s2 | 88 | 48 | 45120 | 2 | - |

## raw 1 (/data/lsh/results_accept/e10/lut_tiling_p2_screen_k1_raw.json)

- one global struct (minimax over 9 conditions, 72 structs): `w8_s0_d0_p0_f0` -> max 5.0% / median 1.3% (n=9)
- phase-1 struct re-implemented (`w8_s0_d0_p0_f0`): max 5.0% / median 1.3% (n=9)
- phase-1 struct (own p1 extension, `p1`): max 4.8% / median 1.4% (n=9)

| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | p1 t us (regret) | orig t us | best vs p1 | best vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 9.3 | w16_s0_d0_p0_f0 (0.4%) | 0.0% | 9.3 (-0.2%) | 14.7 | 0.998 | 1.574 |
| 1 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 13.5 | w8_s0_d0_p2_f0 (0.5%) | 0.0% | 13.5 (-0.0%) | 18.3 | 1.000 | 1.352 |
| 1 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 53.1 | w8_s0_d0_p0_f0 (0.1%) | 0.1% | 53.9 (1.4%) | 67.6 | 1.014 | 1.273 |
| 1 | 32768 | 8 | w16_s0_d1_p0_f0 | 128 | 2048 | 128 | 19.2 | w16_s0_d1_p2_f0 (0.4%) | 1.3% | 19.4 (1.0%) | 31.1 | 1.010 | 1.617 |
| 1 | 32768 | 32 | w16_s0_d0_p0_f0 | 8192 | 8192 | 128 | 52.0 | w16_s0_d0_p3_f1 (0.6%) | 1.5% | 51.8 (-0.4%) | 61.0 | 0.996 | 1.173 |
| 1 | 32768 | 256 | w16_s0_d1_p0_f0 | 256 | 32768 | 256 | 351.3 | w16_s0_d0_p3_f1 (2.2%) | 5.0% | 368.3 (4.8%) | 447.3 | 1.048 | 1.273 |
| 1 | 131072 | 8 | w16_s0_d1_p0_f0 | 128 | 8192 | 128 | 52.7 | w8_s0_d0_p0_f0 (0.6%) | 0.6% | 54.5 (3.4%) | 67.0 | 1.034 | 1.272 |
| 1 | 131072 | 32 | w16_s0_d1_p0_f0 | 128 | 32768 | 128 | 181.0 | w16_s1_d0_p0_f1 (0.3%) | 4.4% | 188.8 (4.3%) | 224.5 | 1.043 | 1.240 |
| 1 | 131072 | 256 | w16_s1_d1_p2_f1 | 256 | 131072 | 256 | 1341.4 | w16_s1_d1_p0_f0 (0.2%) | 3.5% | 1405.6 (4.8%) | 1674.8 | 1.048 | 1.249 |

- KST1: best-per-cond vs p1 geomean 1.021, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.003 (min 0.981, max 1.028); global vs orig geomean 1.305

### Per axis: fix the axis value, other axes free per condition (regret vs best struct)

| axis | value | regret over conditions | conditions where this value is the axis argmax |
|---|---|---|---|
| NW (warps/block) | 4 | max 16.0% / median 12.3% (n=9) | 0 / 9 |
| NW (warps/block) | 8 | max 5.0% / median 1.3% (n=9) | 3 / 9 |
| NW (warps/block) | 16 (best fixed) | max 2.2% / median 0.0% (n=9) | 6 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 0 (best fixed) | max 0.9% / median 0.0% (n=9) | 8 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 1 | max 18.9% / median 4.0% (n=9) | 1 / 9 |
| decomposition (0 split grid, 1 stream-K) | 0 (best fixed) | max 2.2% / median 0.3% (n=9) | 4 / 9 |
| decomposition (0 split grid, 1 stream-K) | 1 | max 3.6% / median 0.0% (n=9) | 5 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 0 (best fixed) | max 0.2% / median 0.0% (n=9) | 8 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 2 | max 5.1% / median 1.1% (n=9) | 1 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 3 | max 9.0% / median 3.4% (n=9) | 0 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 0 (best fixed) | max 0.2% / median 0.0% (n=9) | 7 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 1 | max 3.3% / median 0.6% (n=9) | 2 / 9 |

Per-condition axis argmax (value; regret of the best-fixed value at that condition):

| KST | n | B*Hkv | NW | LUT source | decomposition | code pipeline | combine |
|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | 8 (0.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 32 | 8 (1.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 256 | 8 (2.2%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| 1 | 32768 | 8 | 16 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 32 | 16 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 256 | 16 (0.0%) | 0 (0.0%) | 1 (2.2%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 8 | 16 (0.0%) | 0 (0.0%) | 1 (0.6%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 32 | 16 (0.0%) | 0 (0.0%) | 1 (0.3%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 256 | 16 (0.0%) | 1 (0.9%) | 1 (2.0%) | 2 (0.2%) | 1 (0.2%) |

### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)

| condition axis | regret |
|---|---|
| KST | n/a |
| n | max 14.3% / median 3.2% (n=18) |
| B*Hkv | max 13.1% / median 0.5% (n=18) |

Full transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:

| i \ j | k1n4k b8 | k1n4k b32 | k1n4k b256 | k1n32k b8 | k1n32k b32 | k1n32k b256 | k1n128k b8 | k1n128k b32 | k1n128k b256 |
|---|---|---|---|---|---|---|---|---|---|
| k1n4k b8 | 0.0 | 0.0 | 13.1 | 2.6 | 0.4 | 2.6 | 2.6 | 2.6 | 37.7 |
| k1n4k b32 | 0.0 | 0.0 | 0.6 | 8.5 | 5.5 | 8.5 | 8.5 | 8.5 | 18.3 |
| k1n4k b256 | 0.1 | 0.1 | 0.0 | 6.3 | 8.8 | 6.3 | 6.3 | 6.3 | 14.3 |
| k1n32k b8 | 1.3 | 1.3 | 9.2 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 17.3 |
| k1n32k b32 | 1.5 | 1.5 | 6.0 | 1.7 | 0.0 | 1.7 | 1.7 | 1.7 | 4.5 |
| k1n32k b256 | 5.0 | 5.0 | 6.5 | 0.0 | 6.4 | 0.0 | 0.0 | 0.0 | 4.6 |
| k1n128k b8 | 0.6 | 0.6 | 6.7 | 0.0 | 1.1 | 0.0 | 0.0 | 0.0 | 4.9 |
| k1n128k b32 | 4.4 | 4.4 | 5.1 | 0.0 | 3.9 | 0.0 | 0.0 | 0.0 | 0.6 |
| k1n128k b256 | 3.5 | 3.5 | 3.8 | 0.9 | 3.9 | 0.9 | 0.9 | 0.9 | 0.0 |

## raw 2 (/data/lsh/results_accept/e10/lut_tiling_p2_screen_k2_raw.json)

- one global struct (minimax over 9 conditions, 32 structs): `w8_s0_d0_p0_f0` -> max 4.7% / median 0.0% (n=9)
- phase-1 struct re-implemented (`w8_s0_d0_p0_f0`): max 4.7% / median 0.0% (n=9)
- phase-1 struct (own p1 extension, `p1`): max 5.6% / median 1.1% (n=9)

| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | p1 t us (regret) | orig t us | best vs p1 | best vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 12.0 | w16_s0_d0_p0_f0 (0.6%) | 0.0% | 12.2 (1.1%) | 17.5 | 1.011 | 1.456 |
| 2 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 18.8 | w8_s0_d0_p0_f1 (0.1%) | 0.0% | 18.9 (0.5%) | 23.3 | 1.005 | 1.243 |
| 2 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 87.9 | w8_s0_d1_p0_f1 (3.7%) | 4.7% | 92.8 (5.6%) | 103.6 | 1.056 | 1.179 |
| 2 | 32768 | 8 | w8_s0_d0_p0_f0 | 2048 | 2048 | 128 | 29.0 | w8_s0_d1_p0_f0 (2.0%) | 0.0% | 29.0 (0.0%) | 40.8 | 1.000 | 1.406 |
| 2 | 32768 | 32 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 85.1 | w8_s0_d0_p0_f1 (1.7%) | 0.0% | 85.5 (0.5%) | 96.0 | 1.005 | 1.128 |
| 2 | 32768 | 256 | w8_s0_d1_p0_f0 | 256 | 32768 | 256 | 603.3 | w8_s0_d0_p0_f1 (0.4%) | 0.6% | 609.9 (1.1%) | 672.4 | 1.011 | 1.114 |
| 2 | 131072 | 8 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 86.7 | w8_s0_d1_p0_f0 (1.0%) | 0.0% | 87.6 (1.1%) | 102.7 | 1.011 | 1.185 |
| 2 | 131072 | 32 | w8_s0_d1_p0_f0 | 128 | 32768 | 128 | 306.4 | w8_s0_d1_p0_f1 (0.2%) | 0.5% | 307.4 (0.3%) | 344.2 | 1.003 | 1.123 |
| 2 | 131072 | 256 | w8_s0_d0_p0_f0 | 131072 | 131072 | 256 | 2292.2 | w8_s0_d1_p0_f0 (0.0%) | 0.0% | 2379.3 (3.8%) | 2488.3 | 1.038 | 1.086 |

- KST2: best-per-cond vs p1 geomean 1.015, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.009 (min 0.998, max 1.038); global vs orig geomean 1.200

### Per axis: fix the axis value, other axes free per condition (regret vs best struct)

| axis | value | regret over conditions | conditions where this value is the axis argmax |
|---|---|---|---|
| NW (warps/block) | 4 | max 12.1% / median 9.1% (n=9) | 0 / 9 |
| NW (warps/block) | 8 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| NW (warps/block) | 16 | max 14.5% / median 5.4% (n=9) | 0 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 0 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 1 | max 21.7% / median 5.5% (n=9) | 0 / 9 |
| decomposition (0 split grid, 1 stream-K) | 0 (best fixed) | max 0.4% / median 0.0% (n=9) | 7 / 9 |
| decomposition (0 split grid, 1 stream-K) | 1 | max 3.9% / median 2.0% (n=9) | 2 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 0 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 2 | max 13.0% / median 9.1% (n=9) | 0 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 0 (best fixed) | max 4.7% / median 0.0% (n=9) | 8 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 1 | max 5.2% / median 0.4% (n=9) | 1 / 9 |

Per-condition axis argmax (value; regret of the best-fixed value at that condition):

| KST | n | B*Hkv | NW | LUT source | decomposition | code pipeline | combine |
|---|---|---|---|---|---|---|---|
| 2 | 4096 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.7%) |
| 2 | 32768 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 256 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 32 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)

| condition axis | regret |
|---|---|
| KST | n/a |
| n | max 6.1% / median 0.1% (n=18) |
| B*Hkv | max 10.4% / median 0.5% (n=18) |

Full transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:

| i \ j | k2n4k b8 | k2n4k b32 | k2n4k b256 | k2n32k b8 | k2n32k b32 | k2n32k b256 | k2n128k b8 | k2n128k b32 | k2n128k b256 |
|---|---|---|---|---|---|---|---|---|---|
| k2n4k b8 | 0.0 | 0.0 | 10.4 | 0.0 | 0.0 | 3.6 | 0.0 | 3.6 | 0.0 |
| k2n4k b32 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 3.9 | 0.0 | 3.9 | 0.0 |
| k2n4k b256 | 4.7 | 4.7 | 0.0 | 4.7 | 4.7 | 6.1 | 4.7 | 6.1 | 4.7 |
| k2n32k b8 | 0.0 | 0.0 | 5.2 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b32 | 0.0 | 0.0 | 1.7 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b256 | 0.6 | 0.6 | 0.4 | 0.6 | 0.6 | 0.0 | 0.6 | 0.0 | 0.6 |
| k2n128k b8 | 0.0 | 0.0 | 2.3 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| k2n128k b32 | 0.5 | 0.5 | 0.4 | 0.5 | 0.5 | 0.0 | 0.5 | 0.0 | 0.5 |
| k2n128k b256 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## pooled (min over raws)

- one global struct (minimax over 18 conditions, 32 structs): `w8_s0_d0_p0_f0` -> max 5.0% / median 0.3% (n=18)
- phase-1 struct re-implemented (`w8_s0_d0_p0_f0`): max 5.0% / median 0.3% (n=18)
- phase-1 struct (own p1 extension, `p1`): max 5.6% / median 1.1% (n=18)

| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | p1 t us (regret) | orig t us | best vs p1 | best vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 9.3 | w16_s0_d0_p0_f0 (0.4%) | 0.0% | 9.3 (-0.2%) | 14.7 | 0.998 | 1.574 |
| 1 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 13.5 | w8_s0_d0_p0_f1 (0.6%) | 0.0% | 13.5 (-0.0%) | 18.3 | 1.000 | 1.352 |
| 1 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 53.1 | w8_s0_d0_p0_f0 (0.1%) | 0.1% | 53.9 (1.4%) | 67.6 | 1.014 | 1.273 |
| 1 | 32768 | 8 | w16_s0_d1_p0_f0 | 128 | 2048 | 128 | 19.2 | w16_s0_d0_p0_f0 (0.4%) | 1.3% | 19.4 (1.0%) | 31.1 | 1.010 | 1.617 |
| 1 | 32768 | 32 | w16_s0_d0_p0_f0 | 8192 | 8192 | 128 | 52.0 | w8_s0_d0_p0_f0 (1.5%) | 1.5% | 51.8 (-0.4%) | 61.0 | 0.996 | 1.173 |
| 1 | 32768 | 256 | w16_s0_d1_p0_f0 | 256 | 32768 | 256 | 351.3 | w16_s1_d1_p0_f0 (4.0%) | 5.0% | 368.3 (4.8%) | 447.3 | 1.048 | 1.273 |
| 1 | 131072 | 8 | w16_s0_d1_p0_f0 | 128 | 8192 | 128 | 52.7 | w8_s0_d0_p0_f0 (0.6%) | 0.6% | 54.5 (3.4%) | 67.0 | 1.034 | 1.272 |
| 1 | 131072 | 32 | w16_s0_d1_p0_f0 | 128 | 32768 | 128 | 181.0 | w16_s1_d0_p0_f1 (0.3%) | 4.4% | 188.8 (4.3%) | 224.5 | 1.043 | 1.240 |
| 1 | 131072 | 256 | w16_s1_d1_p0_f0 | 128 | 262144 | 128 | 1343.5 | w16_s0_d1_p0_f0 (0.7%) | 3.4% | 1405.6 (4.6%) | 1674.8 | 1.046 | 1.247 |
| 2 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 12.0 | w16_s0_d0_p0_f0 (0.6%) | 0.0% | 12.2 (1.1%) | 17.5 | 1.011 | 1.456 |
| 2 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 18.8 | w8_s0_d0_p0_f1 (0.1%) | 0.0% | 18.9 (0.5%) | 23.3 | 1.005 | 1.243 |
| 2 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 87.9 | w8_s0_d1_p0_f1 (3.7%) | 4.7% | 92.8 (5.6%) | 103.6 | 1.056 | 1.179 |
| 2 | 32768 | 8 | w8_s0_d0_p0_f0 | 2048 | 2048 | 128 | 29.0 | w8_s0_d1_p0_f0 (2.0%) | 0.0% | 29.0 (0.0%) | 40.8 | 1.000 | 1.406 |
| 2 | 32768 | 32 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 85.1 | w8_s0_d0_p0_f1 (1.7%) | 0.0% | 85.5 (0.5%) | 96.0 | 1.005 | 1.128 |
| 2 | 32768 | 256 | w8_s0_d1_p0_f0 | 256 | 32768 | 256 | 603.3 | w8_s0_d0_p0_f1 (0.4%) | 0.6% | 609.9 (1.1%) | 672.4 | 1.011 | 1.114 |
| 2 | 131072 | 8 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 86.7 | w8_s0_d1_p0_f0 (1.0%) | 0.0% | 87.6 (1.1%) | 102.7 | 1.011 | 1.185 |
| 2 | 131072 | 32 | w8_s0_d1_p0_f0 | 128 | 32768 | 128 | 306.4 | w8_s0_d1_p0_f1 (0.2%) | 0.5% | 307.4 (0.3%) | 344.2 | 1.003 | 1.123 |
| 2 | 131072 | 256 | w8_s0_d0_p0_f0 | 131072 | 131072 | 256 | 2292.2 | w8_s0_d1_p0_f0 (0.0%) | 0.0% | 2379.3 (3.8%) | 2488.3 | 1.038 | 1.086 |

- KST1: best-per-cond vs p1 geomean 1.021, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.003 (min 0.981, max 1.028); global vs orig geomean 1.305

- KST2: best-per-cond vs p1 geomean 1.015, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.009 (min 0.998, max 1.038); global vs orig geomean 1.200

### Per axis: fix the axis value, other axes free per condition (regret vs best struct)

| axis | value | regret over conditions | conditions where this value is the axis argmax |
|---|---|---|---|
| NW (warps/block) | 4 | max 16.0% / median 9.9% (n=18) | 0 / 18 |
| NW (warps/block) | 8 (best fixed) | max 5.0% / median 0.0% (n=18) | 12 / 18 |
| NW (warps/block) | 16 | max 14.5% / median 0.6% (n=18) | 6 / 18 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 0 (best fixed) | max 0.7% / median 0.0% (n=18) | 17 / 18 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 1 | max 21.7% / median 5.3% (n=18) | 1 / 18 |
| decomposition (0 split grid, 1 stream-K) | 0 (best fixed) | max 4.7% / median 0.0% (n=18) | 11 / 18 |
| decomposition (0 split grid, 1 stream-K) | 1 | max 5.2% / median 1.3% (n=18) | 7 / 18 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 0 (best fixed) | max 0.0% / median 0.0% (n=18) | 18 / 18 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 2 | max 16.0% / median 9.9% (n=18) | 0 / 18 |
| combine (0 separate Triton, 1 fused last-block) | 0 (best fixed) | max 4.7% / median 0.0% (n=18) | 16 / 18 |
| combine (0 separate Triton, 1 fused last-block) | 1 | max 5.2% / median 1.2% (n=18) | 2 / 18 |

Per-condition axis argmax (value; regret of the best-fixed value at that condition):

| KST | n | B*Hkv | NW | LUT source | decomposition | code pipeline | combine |
|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| 1 | 32768 | 8 | 16 (1.3%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 32 | 16 (1.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 256 | 16 (5.0%) | 0 (0.0%) | 1 (4.7%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 8 | 16 (0.6%) | 0 (0.0%) | 1 (0.6%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 32 | 16 (4.4%) | 0 (0.0%) | 1 (0.3%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 256 | 16 (3.4%) | 1 (0.7%) | 1 (1.9%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.7%) |
| 2 | 32768 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 256 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 32 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)

| condition axis | regret |
|---|---|
| KST | max 9.3% / median 1.4% (n=18) |
| n | max 15.0% / median 1.0% (n=36) |
| B*Hkv | max 13.1% / median 0.5% (n=36) |

Full transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:

| i \ j | k1n4k b8 | k1n4k b32 | k1n4k b256 | k1n32k b8 | k1n32k b32 | k1n32k b256 | k1n128k b8 | k1n128k b32 | k1n128k b256 | k2n4k b8 | k2n4k b32 | k2n4k b256 | k2n32k b8 | k2n32k b32 | k2n32k b256 | k2n128k b8 | k2n128k b32 | k2n128k b256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k1n4k b8 | 0.0 | 0.0 | 13.1 | 2.6 | 0.4 | 2.6 | 2.6 | 2.6 | 34.6 | 0.0 | 0.0 | 13.1 | 0.0 | 0.0 | 4.9 | 0.0 | 4.9 | 0.0 |
| k1n4k b32 | 0.0 | 0.0 | 0.6 | 8.5 | 5.5 | 8.5 | 8.5 | 8.5 | 22.6 | 0.0 | 0.0 | 0.6 | 0.0 | 0.0 | 5.2 | 0.0 | 5.2 | 0.0 |
| k1n4k b256 | 0.1 | 0.1 | 0.0 | 6.3 | 8.8 | 6.3 | 6.3 | 6.3 | 15.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 | 5.6 | 0.1 | 5.6 | 0.1 |
| k1n32k b8 | 1.3 | 1.3 | 9.2 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 11.8 | 1.3 | 1.3 | 9.2 | 1.3 | 1.3 | 4.0 | 1.3 | 4.0 | 1.3 |
| k1n32k b32 | 1.5 | 1.5 | 6.0 | 1.7 | 0.0 | 1.7 | 1.7 | 1.7 | 6.0 | 1.5 | 1.5 | 6.0 | 1.5 | 1.5 | 6.6 | 1.5 | 6.6 | 1.5 |
| k1n32k b256 | 5.0 | 5.0 | 6.5 | 0.0 | 6.4 | 0.0 | 0.0 | 0.0 | 4.0 | 5.0 | 5.0 | 6.5 | 5.0 | 5.0 | 7.8 | 5.0 | 7.8 | 5.0 |
| k1n128k b8 | 0.6 | 0.6 | 6.7 | 0.0 | 1.1 | 0.0 | 0.0 | 0.0 | 3.8 | 0.6 | 0.6 | 6.7 | 0.6 | 0.6 | 4.6 | 0.6 | 4.6 | 0.6 |
| k1n128k b32 | 4.4 | 4.4 | 5.1 | 0.0 | 3.9 | 0.0 | 0.0 | 0.0 | 1.0 | 4.4 | 4.4 | 5.1 | 4.4 | 4.4 | 8.4 | 4.4 | 8.4 | 4.4 |
| k1n128k b256 | 3.4 | 3.4 | 3.6 | 0.7 | 3.7 | 0.7 | 0.7 | 0.7 | 0.0 | 3.4 | 3.4 | 3.6 | 3.4 | 3.4 | 6.2 | 3.4 | 6.2 | 3.4 |
| k2n4k b8 | 0.0 | 0.0 | 10.4 | 3.9 | 0.6 | 3.9 | 3.9 | 3.9 | 26.7 | 0.0 | 0.0 | 10.4 | 0.0 | 0.0 | 3.6 | 0.0 | 3.6 | 0.0 |
| k2n4k b32 | 0.0 | 0.0 | 0.1 | 17.5 | 16.0 | 17.5 | 17.5 | 17.5 | 33.9 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 3.9 | 0.0 | 3.9 | 0.0 |
| k2n4k b256 | 4.7 | 4.7 | 0.0 | 16.5 | 15.5 | 16.5 | 16.5 | 16.5 | 36.7 | 4.7 | 4.7 | 0.0 | 4.7 | 4.7 | 6.1 | 4.7 | 6.1 | 4.7 |
| k2n32k b8 | 0.0 | 0.0 | 5.2 | 9.3 | 8.4 | 9.3 | 9.3 | 9.3 | 18.0 | 0.0 | 0.0 | 5.2 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b32 | 0.0 | 0.0 | 1.7 | 6.7 | 6.3 | 6.7 | 6.7 | 6.7 | 13.0 | 0.0 | 0.0 | 1.7 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b256 | 0.6 | 0.6 | 0.4 | 1.8 | 1.6 | 1.8 | 1.8 | 1.8 | 8.5 | 0.6 | 0.6 | 0.4 | 0.6 | 0.6 | 0.0 | 0.6 | 0.0 | 0.6 |
| k2n128k b8 | 0.0 | 0.0 | 2.3 | 5.4 | 5.5 | 5.4 | 5.4 | 5.4 | 9.3 | 0.0 | 0.0 | 2.3 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| k2n128k b32 | 0.5 | 0.5 | 0.4 | 1.0 | 1.5 | 1.0 | 1.0 | 1.0 | 3.2 | 0.5 | 0.5 | 0.4 | 0.5 | 0.5 | 0.0 | 0.5 | 0.0 | 0.5 |
| k2n128k b256 | 0.0 | 0.0 | 0.2 | 1.0 | 0.9 | 1.0 | 1.0 | 1.0 | 2.7 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## pooled, KST1 only

- one global struct (minimax over 9 conditions, 72 structs): `w8_s0_d0_p0_f0` -> max 5.0% / median 1.3% (n=9)
- phase-1 struct re-implemented (`w8_s0_d0_p0_f0`): max 5.0% / median 1.3% (n=9)
- phase-1 struct (own p1 extension, `p1`): max 4.8% / median 1.4% (n=9)

| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | p1 t us (regret) | orig t us | best vs p1 | best vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 9.3 | w16_s0_d0_p0_f0 (0.4%) | 0.0% | 9.3 (-0.2%) | 14.7 | 0.998 | 1.574 |
| 1 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 13.5 | w8_s0_d0_p2_f0 (0.5%) | 0.0% | 13.5 (-0.0%) | 18.3 | 1.000 | 1.352 |
| 1 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 53.1 | w8_s0_d0_p0_f0 (0.1%) | 0.1% | 53.9 (1.4%) | 67.6 | 1.014 | 1.273 |
| 1 | 32768 | 8 | w16_s0_d1_p0_f0 | 128 | 2048 | 128 | 19.2 | w16_s0_d1_p2_f0 (0.4%) | 1.3% | 19.4 (1.0%) | 31.1 | 1.010 | 1.617 |
| 1 | 32768 | 32 | w16_s0_d0_p0_f0 | 8192 | 8192 | 128 | 52.0 | w16_s0_d0_p3_f1 (0.6%) | 1.5% | 51.8 (-0.4%) | 61.0 | 0.996 | 1.173 |
| 1 | 32768 | 256 | w16_s0_d1_p0_f0 | 256 | 32768 | 256 | 351.3 | w16_s0_d0_p3_f1 (2.2%) | 5.0% | 368.3 (4.8%) | 447.3 | 1.048 | 1.273 |
| 1 | 131072 | 8 | w16_s0_d1_p0_f0 | 128 | 8192 | 128 | 52.7 | w8_s0_d0_p0_f0 (0.6%) | 0.6% | 54.5 (3.4%) | 67.0 | 1.034 | 1.272 |
| 1 | 131072 | 32 | w16_s0_d1_p0_f0 | 128 | 32768 | 128 | 181.0 | w16_s1_d0_p0_f1 (0.3%) | 4.4% | 188.8 (4.3%) | 224.5 | 1.043 | 1.240 |
| 1 | 131072 | 256 | w16_s1_d1_p2_f1 | 256 | 131072 | 256 | 1341.4 | w16_s1_d1_p0_f0 (0.2%) | 3.5% | 1405.6 (4.8%) | 1674.8 | 1.048 | 1.249 |

- KST1: best-per-cond vs p1 geomean 1.021, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.003 (min 0.981, max 1.028); global vs orig geomean 1.305

### Per axis: fix the axis value, other axes free per condition (regret vs best struct)

| axis | value | regret over conditions | conditions where this value is the axis argmax |
|---|---|---|---|
| NW (warps/block) | 4 | max 16.0% / median 12.3% (n=9) | 0 / 9 |
| NW (warps/block) | 8 | max 5.0% / median 1.3% (n=9) | 3 / 9 |
| NW (warps/block) | 16 (best fixed) | max 2.2% / median 0.0% (n=9) | 6 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 0 (best fixed) | max 0.9% / median 0.0% (n=9) | 8 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 1 | max 18.9% / median 4.0% (n=9) | 1 / 9 |
| decomposition (0 split grid, 1 stream-K) | 0 (best fixed) | max 2.2% / median 0.3% (n=9) | 4 / 9 |
| decomposition (0 split grid, 1 stream-K) | 1 | max 3.6% / median 0.0% (n=9) | 5 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 0 (best fixed) | max 0.2% / median 0.0% (n=9) | 8 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 2 | max 5.1% / median 1.1% (n=9) | 1 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 3 | max 9.0% / median 3.4% (n=9) | 0 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 0 (best fixed) | max 0.2% / median 0.0% (n=9) | 7 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 1 | max 3.3% / median 0.6% (n=9) | 2 / 9 |

Per-condition axis argmax (value; regret of the best-fixed value at that condition):

| KST | n | B*Hkv | NW | LUT source | decomposition | code pipeline | combine |
|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | 8 (0.4%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 32 | 8 (1.5%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 4096 | 256 | 8 (2.2%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (0.1%) |
| 1 | 32768 | 8 | 16 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 32 | 16 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 32768 | 256 | 16 (0.0%) | 0 (0.0%) | 1 (2.2%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 8 | 16 (0.0%) | 0 (0.0%) | 1 (0.6%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 32 | 16 (0.0%) | 0 (0.0%) | 1 (0.3%) | 0 (0.0%) | 0 (0.0%) |
| 1 | 131072 | 256 | 16 (0.0%) | 1 (0.9%) | 1 (2.0%) | 2 (0.2%) | 1 (0.2%) |

### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)

| condition axis | regret |
|---|---|
| KST | n/a |
| n | max 14.3% / median 3.2% (n=18) |
| B*Hkv | max 13.1% / median 0.5% (n=18) |

Full transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:

| i \ j | k1n4k b8 | k1n4k b32 | k1n4k b256 | k1n32k b8 | k1n32k b32 | k1n32k b256 | k1n128k b8 | k1n128k b32 | k1n128k b256 |
|---|---|---|---|---|---|---|---|---|---|
| k1n4k b8 | 0.0 | 0.0 | 13.1 | 2.6 | 0.4 | 2.6 | 2.6 | 2.6 | 37.7 |
| k1n4k b32 | 0.0 | 0.0 | 0.6 | 8.5 | 5.5 | 8.5 | 8.5 | 8.5 | 18.3 |
| k1n4k b256 | 0.1 | 0.1 | 0.0 | 6.3 | 8.8 | 6.3 | 6.3 | 6.3 | 14.3 |
| k1n32k b8 | 1.3 | 1.3 | 9.2 | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 17.3 |
| k1n32k b32 | 1.5 | 1.5 | 6.0 | 1.7 | 0.0 | 1.7 | 1.7 | 1.7 | 4.5 |
| k1n32k b256 | 5.0 | 5.0 | 6.5 | 0.0 | 6.4 | 0.0 | 0.0 | 0.0 | 4.6 |
| k1n128k b8 | 0.6 | 0.6 | 6.7 | 0.0 | 1.1 | 0.0 | 0.0 | 0.0 | 4.9 |
| k1n128k b32 | 4.4 | 4.4 | 5.1 | 0.0 | 3.9 | 0.0 | 0.0 | 0.0 | 0.6 |
| k1n128k b256 | 3.5 | 3.5 | 3.8 | 0.9 | 3.9 | 0.9 | 0.9 | 0.9 | 0.0 |

## pooled, KST2 only

- one global struct (minimax over 9 conditions, 32 structs): `w8_s0_d0_p0_f0` -> max 4.7% / median 0.0% (n=9)
- phase-1 struct re-implemented (`w8_s0_d0_p0_f0`): max 4.7% / median 0.0% (n=9)
- phase-1 struct (own p1 extension, `p1`): max 5.6% / median 1.1% (n=9)

| KST | n | B*Hkv | best struct | knob | chunk | blocks | t us | 2nd best (regret) | global struct regret | p1 t us (regret) | orig t us | best vs p1 | best vs orig |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 4096 | 8 | w8_s0_d0_p0_f0 | 256 | 256 | 128 | 12.0 | w16_s0_d0_p0_f0 (0.6%) | 0.0% | 12.2 (1.1%) | 17.5 | 1.011 | 1.456 |
| 2 | 4096 | 32 | w8_s0_d0_p0_f0 | 1024 | 1024 | 128 | 18.8 | w8_s0_d0_p0_f1 (0.1%) | 0.0% | 18.9 (0.5%) | 23.3 | 1.005 | 1.243 |
| 2 | 4096 | 256 | w8_s0_d0_p0_f1 | 4096 | 4096 | 256 | 87.9 | w8_s0_d1_p0_f1 (3.7%) | 4.7% | 92.8 (5.6%) | 103.6 | 1.056 | 1.179 |
| 2 | 32768 | 8 | w8_s0_d0_p0_f0 | 2048 | 2048 | 128 | 29.0 | w8_s0_d1_p0_f0 (2.0%) | 0.0% | 29.0 (0.0%) | 40.8 | 1.000 | 1.406 |
| 2 | 32768 | 32 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 85.1 | w8_s0_d0_p0_f1 (1.7%) | 0.0% | 85.5 (0.5%) | 96.0 | 1.005 | 1.128 |
| 2 | 32768 | 256 | w8_s0_d1_p0_f0 | 256 | 32768 | 256 | 603.3 | w8_s0_d0_p0_f1 (0.4%) | 0.6% | 609.9 (1.1%) | 672.4 | 1.011 | 1.114 |
| 2 | 131072 | 8 | w8_s0_d0_p0_f0 | 8192 | 8192 | 128 | 86.7 | w8_s0_d1_p0_f0 (1.0%) | 0.0% | 87.6 (1.1%) | 102.7 | 1.011 | 1.185 |
| 2 | 131072 | 32 | w8_s0_d1_p0_f0 | 128 | 32768 | 128 | 306.4 | w8_s0_d1_p0_f1 (0.2%) | 0.5% | 307.4 (0.3%) | 344.2 | 1.003 | 1.123 |
| 2 | 131072 | 256 | w8_s0_d0_p0_f0 | 131072 | 131072 | 256 | 2292.2 | w8_s0_d1_p0_f0 (0.0%) | 0.0% | 2379.3 (3.8%) | 2488.3 | 1.038 | 1.086 |

- KST2: best-per-cond vs p1 geomean 1.015, global struct `w8_s0_d0_p0_f0` vs p1 geomean 1.009 (min 0.998, max 1.038); global vs orig geomean 1.200

### Per axis: fix the axis value, other axes free per condition (regret vs best struct)

| axis | value | regret over conditions | conditions where this value is the axis argmax |
|---|---|---|---|
| NW (warps/block) | 4 | max 12.1% / median 9.1% (n=9) | 0 / 9 |
| NW (warps/block) | 8 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| NW (warps/block) | 16 | max 14.5% / median 5.4% (n=9) | 0 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 0 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| LUT source (0 per-block build, 1 shared prebuilt LUT) | 1 | max 21.7% / median 5.5% (n=9) | 0 / 9 |
| decomposition (0 split grid, 1 stream-K) | 0 (best fixed) | max 0.4% / median 0.0% (n=9) | 7 / 9 |
| decomposition (0 split grid, 1 stream-K) | 1 | max 3.9% / median 2.0% (n=9) | 2 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 0 (best fixed) | max 0.0% / median 0.0% (n=9) | 9 / 9 |
| code pipeline (0 reg prefetch, 2/3 cp.async stages) | 2 | max 13.0% / median 9.1% (n=9) | 0 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 0 (best fixed) | max 4.7% / median 0.0% (n=9) | 8 / 9 |
| combine (0 separate Triton, 1 fused last-block) | 1 | max 5.2% / median 0.4% (n=9) | 1 / 9 |

Per-condition axis argmax (value; regret of the best-fixed value at that condition):

| KST | n | B*Hkv | NW | LUT source | decomposition | code pipeline | combine |
|---|---|---|---|---|---|---|---|
| 2 | 4096 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 4096 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.7%) |
| 2 | 32768 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 32 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 32768 | 256 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 8 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 32 | 8 (0.0%) | 0 (0.0%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) |
| 2 | 131072 | 256 | 8 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### Transfer regret of the full struct along condition axes (cond j's best struct used on cond i)

| condition axis | regret |
|---|---|
| KST | n/a |
| n | max 6.1% / median 0.1% (n=18) |
| B*Hkv | max 10.4% / median 0.5% (n=18) |

Full transfer matrix R[i, j] (%), rows i = condition evaluated, cols j = condition whose best struct is used:

| i \ j | k2n4k b8 | k2n4k b32 | k2n4k b256 | k2n32k b8 | k2n32k b32 | k2n32k b256 | k2n128k b8 | k2n128k b32 | k2n128k b256 |
|---|---|---|---|---|---|---|---|---|---|
| k2n4k b8 | 0.0 | 0.0 | 10.4 | 0.0 | 0.0 | 3.6 | 0.0 | 3.6 | 0.0 |
| k2n4k b32 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 3.9 | 0.0 | 3.9 | 0.0 |
| k2n4k b256 | 4.7 | 4.7 | 0.0 | 4.7 | 4.7 | 6.1 | 4.7 | 6.1 | 4.7 |
| k2n32k b8 | 0.0 | 0.0 | 5.2 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b32 | 0.0 | 0.0 | 1.7 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | 0.0 |
| k2n32k b256 | 0.6 | 0.6 | 0.4 | 0.6 | 0.6 | 0.0 | 0.6 | 0.0 | 0.6 |
| k2n128k b8 | 0.0 | 0.0 | 2.3 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| k2n128k b32 | 0.5 | 0.5 | 0.4 | 0.5 | 0.5 | 0.0 | 0.5 | 0.0 | 0.5 |
| k2n128k b256 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
