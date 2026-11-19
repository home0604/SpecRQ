# E10 phase 1: structural knobs of the LUT VQ-KV draft attention (NVIDIA GeForce RTX 4090)

- raw: `../../results/E10/lut_tiling_p1_rep_r1_raw.json`, `../../results/E10/lut_tiling_p1_rep_r2_raw.json`; metric `t_tot_us` (partial + combine, us per call, median of 9 CUDA-graph reps, codes stream from DRAM).
- limits: {'sm_count': 128, 'smem_per_sm': 102400, 'smem_per_block_optin': 101376, 'reserved_smem_per_block': 1024, 'regs_per_sm': 65536, 'threads_per_sm': 1536, 'blocks_per_sm': 24, 'l2_bytes': 75497472, 'cc_major': 8, 'cc_minor': 9}; ref read BW 942 GB/s
- `results/E10/lut_tiling_p1_rep_r1_raw.json`: GPU 4 uuid GPU-ce80dfbd-cb70-c9ce-b491-4a2146bc402c; apps before ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3940 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5156 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f, 3521126, 792 MiB']; after ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3944 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5156 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-ce80dfbd-cb70-c9ce-b491-4a2146bc402c, 3525125, 516 MiB', 'GPU-8522ca3c-3d2f-69c2-407a-02b1a73bcb22, 3525135, 3590 MiB']
- `results/E10/lut_tiling_p1_rep_r2_raw.json`: GPU 5 uuid GPU-8522ca3c-3d2f-69c2-407a-02b1a73bcb22; apps before ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3940 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5156 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f, 3521126, 792 MiB']; after ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3944 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5156 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-8522ca3c-3d2f-69c2-407a-02b1a73bcb22, 3525135, 516 MiB', 'GPU-babe0ba5-7c14-6768-fce4-c7515d249f25, 3534863, 2576 MiB']
- struct name: w = warps/block, b = launch-bounds minBlocks, r = in-block partial reduction, p = code prefetch, l = LUT (0 fp16, 1 int8 per-(stage,j,head) scale, 2 int8 per-head packed), s = max (0 online, 1 static LUT bound [unsafe], 2 lazy online max)

## 0. Variant resources (per KST)

| struct | KST1 regs / local B / smem B / blocks/SM / slots | KST2 regs / local B / smem B / blocks/SM / slots |
|---|---|---|
| orig | 76 / 0 / 53312 / 1 / 16 | 104 / 0 / 98368 / 1 / 16 |
| w16_b1_r0_p0_l0_s0 | 101 / 0 / 53312 / 1 / 16 | 107 / 0 / 98368 / 1 / 16 |
| w16_b1_r0_p0_l0_s2 | 103 / 0 / 53312 / 1 / 16 | 107 / 0 / 98368 / 1 / 16 |
| w16_b1_r1_p0_l0_s0 | 81 / 48 / 53312 / 1 / 1 | 88 / 48 / 98368 / 1 / 1 |
| w16_b1_r1_p0_l0_s1 | 83 / 48 / 53568 / 1 / 1 | 91 / 48 / 98880 / 1 / 1 |
| w16_b1_r1_p0_l0_s2 | 83 / 48 / 53312 / 1 / 1 | 90 / 48 / 98368 / 1 / 1 |
| w16_b1_r1_p1_l0_s2 | 87 / 48 / 53312 / 1 / 1 | 105 / 48 / 98368 / 1 / 1 |
| w8_b1_r0_p0_l0_s0 | 108 / 0 / 45120 / 2 / 8 | 112 / 0 / 86080 / 1 / 8 |
| w8_b1_r1_p0_l0_s2 | 85 / 48 / 45120 / 2 / 1 | 92 / 48 / 86080 / 1 / 1 |
| w8_b1_r1_p1_l0_s2 | 88 / 48 / 45120 / 2 / 1 | 104 / 48 / 86080 / 1 / 1 |
| w8_b1_r1_p1_l2_s2 | 92 / 48 / 28752 / 2 / 1 | 109 / 48 / 53328 / 1 / 1 |
| w8_b2_r1_p0_l0_s2 | 83 / 48 / 45120 / 2 / 1 | 90 / 48 / 86080 / 1 / 1 |
| w8_b2_r1_p1_l0_s2 | 87 / 48 / 45120 / 2 / 1 | 105 / 48 / 86080 / 1 / 1 |

## 1. Per-struct speed vs the original kernel (oracle splits for both): geomean / min / max of T_orig / T_struct

| struct | class | KST1 geomean | KST1 min..max | KST2 geomean | KST2 min..max | #conds |
|---|---|---|---|---|---|---|
| orig | exact | 1.000 | 1.00..1.00 | 1.000 | 1.00..1.00 | 18 |
| w16_b1_r0_p0_l0_s0 | exact | 1.022 | 1.01..1.04 | 1.034 | 1.01..1.08 | 18 |
| w16_b1_r0_p0_l0_s2 | exact | 1.127 | 1.08..1.21 | 1.062 | 1.02..1.09 | 18 |
| w16_b1_r1_p0_l0_s0 | exact | 1.164 | 1.04..1.51 | 1.123 | 1.03..1.38 | 18 |
| w16_b1_r1_p0_l0_s1 | sb1 | 1.256 | 1.13..1.54 | 1.124 | 1.04..1.28 | 18 |
| w16_b1_r1_p0_l0_s2 | exact | 1.283 | 1.16..1.58 | 1.153 | 1.05..1.39 | 18 |
| w16_b1_r1_p1_l0_s2 | exact | 1.284 | 1.14..1.61 | 1.137 | 1.01..1.47 | 18 |
| w8_b1_r0_p0_l0_s0 | exact | 1.035 | 0.97..1.16 | 1.015 | 0.92..1.17 | 18 |
| w8_b1_r1_p0_l0_s2 | exact | 1.266 | 1.11..1.58 | 1.163 | 1.05..1.46 | 18 |
| w8_b1_r1_p1_l0_s2 | exact | 1.299 | 1.11..1.60 | 1.199 | 1.10..1.45 | 18 |
| w8_b1_r1_p1_l2_s2 | int8 | 1.024 | 0.79..1.23 | 0.863 | 0.57..1.11 | 18 |
| w8_b2_r1_p0_l0_s2 | exact | 1.271 | 1.15..1.52 | 1.144 | 1.05..1.39 | 18 |
| w8_b2_r1_p1_l0_s2 | exact | 1.291 | 1.12..1.59 | 1.201 | 1.11..1.46 | 18 |

## 2a. Optimum per condition, class `exact` (oracle splits)

| KST | n | B*Hkv | best struct | chunk | splits | t us | orig best us | speedup vs orig | 2nd best (regret) | best with rule splits (regret vs oracle opt) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | w8_b1_r1_p1_l0_s2 | 256 | 16 | 9.2 | 14.6 | 1.584 | w16_b1_r1_p1_l0_s2 (0.0%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 4096 | 32 | w8_b1_r1_p1_l0_s2 | 1024 | 4 | 13.5 | 18.0 | 1.336 | w8_b2_r1_p1_l0_s2 (0.2%) | w16_b1_r1_p0_l0_s2 (3.7%) |
| 1 | 4096 | 256 | w8_b1_r1_p1_l0_s2 | 4096 | 1 | 54.2 | 66.8 | 1.234 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 32768 | 8 | w16_b1_r1_p1_l0_s2 | 2048 | 16 | 19.0 | 30.5 | 1.606 | w8_b1_r1_p1_l0_s2 (0.3%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 32768 | 32 | w16_b1_r1_p0_l0_s2 | 8192 | 4 | 51.9 | 60.2 | 1.162 | w8_b2_r1_p0_l0_s2 (1.1%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 32768 | 256 | w8_b1_r1_p1_l0_s2 | 32768 | 1 | 353.8 | 438.1 | 1.238 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 131072 | 8 | w16_b1_r1_p0_l0_s2 | 8192 | 16 | 51.9 | 66.3 | 1.277 | w16_b1_r1_p1_l0_s2 (2.7%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 131072 | 32 | w8_b1_r1_p1_l0_s2 | 16384 | 8 | 182.4 | 219.9 | 1.206 | w16_b1_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 131072 | 256 | w8_b1_r1_p1_l0_s2 | 65536 | 2 | 1368.7 | 1685.5 | 1.232 | w8_b2_r1_p0_l0_s2 (1.1%) | w8_b2_r1_p0_l0_s2 (1.1%) |
| 2 | 4096 | 8 | w16_b1_r1_p1_l0_s2 | 256 | 16 | 11.9 | 17.5 | 1.468 | w8_b1_r1_p0_l0_s2 (0.6%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 4096 | 32 | w8_b1_r1_p1_l0_s2 | 1024 | 4 | 18.7 | 23.4 | 1.254 | w8_b2_r1_p1_l0_s2 (0.4%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 4096 | 256 | w8_b2_r1_p1_l0_s2 | 4096 | 1 | 91.5 | 102.1 | 1.116 | w8_b1_r1_p1_l0_s2 (0.3%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 32768 | 8 | w8_b1_r1_p0_l0_s2 | 2048 | 16 | 28.6 | 40.3 | 1.406 | w8_b1_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p0_l0_s2 (0.0%) |
| 2 | 32768 | 32 | w8_b2_r1_p1_l0_s2 | 8192 | 4 | 85.0 | 95.0 | 1.117 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 32768 | 256 | w8_b2_r1_p1_l0_s2 | 32768 | 1 | 600.4 | 666.4 | 1.110 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 8 | w8_b2_r1_p1_l0_s2 | 8192 | 16 | 87.3 | 102.6 | 1.175 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 32 | w8_b2_r1_p1_l0_s2 | 32768 | 4 | 302.8 | 339.7 | 1.122 | w8_b1_r1_p1_l0_s2 (0.5%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 256 | w8_b1_r1_p1_l0_s2 | 131072 | 1 | 2288.1 | 2582.0 | 1.128 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p1_l0_s2 (0.0%) |

## 2b. Optimum per condition, class `exact+int8` (oracle splits)

| KST | n | B*Hkv | best struct | chunk | splits | t us | orig best us | speedup vs orig | 2nd best (regret) | best with rule splits (regret vs oracle opt) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 8 | w8_b1_r1_p1_l0_s2 | 256 | 16 | 9.2 | 14.6 | 1.584 | w16_b1_r1_p1_l0_s2 (0.0%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 4096 | 32 | w8_b1_r1_p1_l0_s2 | 1024 | 4 | 13.5 | 18.0 | 1.336 | w8_b2_r1_p1_l0_s2 (0.2%) | w16_b1_r1_p0_l0_s2 (3.7%) |
| 1 | 4096 | 256 | w8_b1_r1_p1_l0_s2 | 4096 | 1 | 54.2 | 66.8 | 1.234 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 32768 | 8 | w16_b1_r1_p1_l0_s2 | 2048 | 16 | 19.0 | 30.5 | 1.606 | w8_b1_r1_p1_l0_s2 (0.3%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 32768 | 32 | w16_b1_r1_p0_l0_s2 | 8192 | 4 | 51.9 | 60.2 | 1.162 | w8_b2_r1_p0_l0_s2 (1.1%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 32768 | 256 | w8_b1_r1_p1_l0_s2 | 32768 | 1 | 353.8 | 438.1 | 1.238 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 131072 | 8 | w16_b1_r1_p0_l0_s2 | 8192 | 16 | 51.9 | 66.3 | 1.277 | w16_b1_r1_p1_l0_s2 (2.7%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 131072 | 32 | w8_b1_r1_p1_l0_s2 | 16384 | 8 | 182.4 | 219.9 | 1.206 | w16_b1_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 131072 | 256 | w8_b1_r1_p1_l0_s2 | 65536 | 2 | 1368.7 | 1685.5 | 1.232 | w8_b1_r1_p1_l2_s2 (0.1%) | w8_b2_r1_p0_l0_s2 (1.1%) |
| 2 | 4096 | 8 | w16_b1_r1_p1_l0_s2 | 256 | 16 | 11.9 | 17.5 | 1.468 | w8_b1_r1_p0_l0_s2 (0.6%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 4096 | 32 | w8_b1_r1_p1_l0_s2 | 1024 | 4 | 18.7 | 23.4 | 1.254 | w8_b2_r1_p1_l0_s2 (0.4%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 4096 | 256 | w8_b2_r1_p1_l0_s2 | 4096 | 1 | 91.5 | 102.1 | 1.116 | w8_b1_r1_p1_l0_s2 (0.3%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 32768 | 8 | w8_b1_r1_p0_l0_s2 | 2048 | 16 | 28.6 | 40.3 | 1.406 | w8_b1_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p0_l0_s2 (0.0%) |
| 2 | 32768 | 32 | w8_b2_r1_p1_l0_s2 | 8192 | 4 | 85.0 | 95.0 | 1.117 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 32768 | 256 | w8_b2_r1_p1_l0_s2 | 32768 | 1 | 600.4 | 666.4 | 1.110 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 8 | w8_b2_r1_p1_l0_s2 | 8192 | 16 | 87.3 | 102.6 | 1.175 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 32 | w8_b2_r1_p1_l0_s2 | 32768 | 4 | 302.8 | 339.7 | 1.122 | w8_b1_r1_p1_l0_s2 (0.5%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 131072 | 256 | w8_b1_r1_p1_l0_s2 | 131072 | 1 | 2288.1 | 2582.0 | 1.128 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b1_r1_p1_l0_s2 (0.0%) |

## 3a. Structural-config regret, class `exact`

Regret = T(policy struct, splits by oracle | rule) / T(best struct at oracle splits) - 1.

- [oracle splits] one global struct: `w8_b1_r1_p1_l0_s2` -> max 4.2% / median 0.3% (n=18)
  - [oracle splits] best struct for KST1 only: `w16_b1_r1_p0_l0_s2` -> on KST1: max 4.0% / median 3.0% (n=9); transferred to the other KST: max 7.0% / median 4.1% (n=9)
  - [oracle splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=9); transferred to the other KST: max 4.5% / median 0.7% (n=9)
  - [oracle splits] original kernel: max 60.6% / median 23.3% (n=18)
- [rule splits] one global struct: `w16_b1_r1_p0_l0_s2` -> max 7.0% / median 3.5% (n=18)
  - [rule splits] best struct for KST1 only: `w16_b1_r1_p0_l0_s2` -> on KST1: max 4.0% / median 3.0% (n=9); transferred to the other KST: max 7.0% / median 4.1% (n=9)
  - [rule splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=9); transferred to the other KST: max 18.9% / median 3.9% (n=9)
  - [rule splits] original kernel: max 65.0% / median 23.3% (n=18)

| axis | splits | regret of using cond j's best struct on cond i (i, j differ only in that axis) |
|---|---|---|
| KST | oracle | max 9.6% / median 0.6% (n=18) |
| KST | rule | max 9.8% / median 1.0% (n=18) |
| n | oracle | max 9.6% / median 0.4% (n=36) |
| n | rule | max 9.8% / median 0.6% (n=36) |
| B*Hkv | oracle | max 16.0% / median 0.6% (n=36) |
| B*Hkv | rule | max 18.5% / median 1.2% (n=36) |

KST transfer per (n, B*Hkv): best struct KST1 / KST2; regret of KST2-best on KST1 / KST1-best on KST2 (oracle splits)

| n \ B*Hkv | 8 | 32 | 256 |
|---|---|---|---|
| 4096 | w8_b1_r1_p1_l0_s2 / w16_b1_r1_p1_l0_s2: 0.0% / 1.3% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.2% / 0.3% |
| 32768 | w16_b1_r1_p1_l0_s2 / w8_b1_r1_p0_l0_s2: 6.7% / 9.6% | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 3.9% / 6.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.6% / 0.4% |
| 131072 | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 4.5% / 4.3% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.7% / 0.5% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% |

## 3b. Structural-config regret, class `exact+int8`

Regret = T(policy struct, splits by oracle | rule) / T(best struct at oracle splits) - 1.

- [oracle splits] one global struct: `w8_b1_r1_p1_l0_s2` -> max 4.2% / median 0.3% (n=18)
  - [oracle splits] best struct for KST1 only: `w16_b1_r1_p0_l0_s2` -> on KST1: max 4.0% / median 3.0% (n=9); transferred to the other KST: max 7.0% / median 4.1% (n=9)
  - [oracle splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=9); transferred to the other KST: max 4.5% / median 0.7% (n=9)
  - [oracle splits] original kernel: max 60.6% / median 23.3% (n=18)
- [rule splits] one global struct: `w16_b1_r1_p0_l0_s2` -> max 7.0% / median 3.5% (n=18)
  - [rule splits] best struct for KST1 only: `w16_b1_r1_p0_l0_s2` -> on KST1: max 4.0% / median 3.0% (n=9); transferred to the other KST: max 7.0% / median 4.1% (n=9)
  - [rule splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=9); transferred to the other KST: max 18.9% / median 3.9% (n=9)
  - [rule splits] original kernel: max 65.0% / median 23.3% (n=18)

| axis | splits | regret of using cond j's best struct on cond i (i, j differ only in that axis) |
|---|---|---|
| KST | oracle | max 9.6% / median 0.6% (n=18) |
| KST | rule | max 9.8% / median 1.0% (n=18) |
| n | oracle | max 9.6% / median 0.4% (n=36) |
| n | rule | max 9.8% / median 0.6% (n=36) |
| B*Hkv | oracle | max 16.0% / median 0.6% (n=36) |
| B*Hkv | rule | max 18.5% / median 1.2% (n=36) |

KST transfer per (n, B*Hkv): best struct KST1 / KST2; regret of KST2-best on KST1 / KST1-best on KST2 (oracle splits)

| n \ B*Hkv | 8 | 32 | 256 |
|---|---|---|---|
| 4096 | w8_b1_r1_p1_l0_s2 / w16_b1_r1_p1_l0_s2: 0.0% / 1.3% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.2% / 0.3% |
| 32768 | w16_b1_r1_p1_l0_s2 / w8_b1_r1_p0_l0_s2: 6.7% / 9.6% | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 3.9% / 6.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.6% / 0.4% |
| 131072 | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 4.5% / 4.3% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.7% / 0.5% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% |

## 3c. One knob at a time around a base struct (oracle splits)

For each knob, the other fields are fixed at the base; per condition the best value of the knob; regret = T(value) / T(best value of that knob at that condition) - 1. 'x-KST' = regret of the other KST's best value at the same (n, B*Hkv); 'one value' = best single value of the knob over all conditions.


base `w8_b1_r1_p1_l0_s2`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 4k/8:8,4k/32:8,4k/256:8,32k/8:16,32k/32:16,32k/256:8,128k/8:16,128k/32:8,128k/256:8 ; 4k/8:16,4k/32:8,4k/256:8,32k/8:8,32k/32:8,32k/256:8,128k/8:8,128k/32:8,128k/256:8 | 8.9% | 2.1% (8) | 2.1% ({1: 8, 2: 8}) |
| b | [1, 2] | 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:2,32k/256:1,128k/8:1,128k/32:1,128k/256:1 ; 4k/8:2,4k/32:1,4k/256:2,32k/8:1,32k/32:2,32k/256:2,128k/8:2,128k/32:2,128k/256:1 | 0.7% | 0.6% (1) | 0.6% ({1: 1, 2: 1}) |
| p | [0, 1] | 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:1,32k/256:1,128k/8:1,128k/32:1,128k/256:1 ; 4k/8:0,4k/32:1,4k/256:1,32k/8:0,32k/32:1,32k/256:1,128k/8:1,128k/32:1,128k/256:1 | 6.3% | 0.7% (1) | 0.7% ({1: 1, 2: 1}) |
| l | [0, 2] | 4k/8:0,4k/32:0,4k/256:0,32k/8:0,32k/32:0,32k/256:0,128k/8:0,128k/32:0,128k/256:0 ; 4k/8:0,4k/32:0,4k/256:0,32k/8:0,32k/32:0,32k/256:0,128k/8:0,128k/32:0,128k/256:0 | 0.0% | 0.0% (0) | 0.0% ({1: 0, 2: 0}) |

base `w16_b1_r1_p0_l0_s2`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 4k/8:8,4k/32:8,4k/256:8,32k/8:16,32k/32:16,32k/256:16,128k/8:16,128k/32:16,128k/256:16 ; 4k/8:8,4k/32:8,4k/256:8,32k/8:8,32k/32:8,32k/256:16,128k/8:8,128k/32:16,128k/256:16 | 5.3% | 5.3% (8) | 4.0% ({1: 16, 2: 8}) |
| r | [0, 1] | 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:1,32k/256:0,128k/8:1,128k/32:1,128k/256:1 ; 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:1,32k/256:1,128k/8:1,128k/32:1,128k/256:1 | 1.5% | 0.6% (1) | 0.6% ({1: 1, 2: 1}) |
| p | [0, 1] | 4k/8:1,4k/32:0,4k/256:1,32k/8:1,32k/32:0,32k/256:1,128k/8:0,128k/32:1,128k/256:0 ; 4k/8:1,4k/32:0,4k/256:0,32k/8:0,32k/32:0,32k/256:1,128k/8:0,128k/32:1,128k/256:1 | 6.7% | 5.7% (0) | 5.7% ({1: 1, 2: 0}) |
| s | [0, 1, 2] | 4k/8:2,4k/32:2,4k/256:2,32k/8:2,32k/32:2,32k/256:1,128k/8:2,128k/32:2,128k/256:2 ; 4k/8:2,4k/32:2,4k/256:2,32k/8:2,32k/32:2,32k/256:1,128k/8:2,128k/32:1,128k/256:1 | 0.9% | 0.9% (2) | 0.9% ({1: 2, 2: 2}) |

base `w16_b1_r0_p0_l0_s0`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 4k/8:8,4k/32:8,4k/256:8,32k/8:8,32k/32:16,32k/256:16,128k/8:16,128k/32:16,128k/256:16 ; 4k/8:8,4k/32:8,4k/256:16,32k/8:8,32k/32:16,32k/256:16,128k/8:16,128k/32:16,128k/256:16 | 5.0% | 11.6% (8) | 11.6% ({1: 8, 2: 8}) |
| r | [0, 1] | 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:1,32k/256:1,128k/8:1,128k/32:1,128k/256:1 ; 4k/8:1,4k/32:1,4k/256:1,32k/8:1,32k/32:1,32k/256:1,128k/8:1,128k/32:1,128k/256:1 | 0.0% | 0.0% (1) | 0.0% ({1: 1, 2: 1}) |
| s | [0, 2] | 4k/8:2,4k/32:2,4k/256:2,32k/8:2,32k/32:2,32k/256:2,128k/8:2,128k/32:2,128k/256:2 ; 4k/8:2,4k/32:2,4k/256:2,32k/8:2,32k/32:2,32k/256:2,128k/8:2,128k/32:2,128k/256:2 | 0.0% | 0.0% (2) | 0.0% ({1: 2, 2: 2}) |

## 4. KST1 / KST2 time ratio: original vs best struct per KST (oracle splits)

| n | B*Hkv | orig ratio | best exact ratio (KST1 / KST2 struct) | best exact+int8 ratio | best incl. sb1 ratio |
|---|---|---|---|---|---|
| 4096 | 8 | 0.830 | 0.769 (w8_b1_r1_p1_l0_s2 / w16_b1_r1_p1_l0_s2) | 0.769 | 0.769 |
| 4096 | 32 | 0.770 | 0.723 (w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2) | 0.723 | 0.723 |
| 4096 | 256 | 0.654 | 0.592 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.592 | 0.592 |
| 32768 | 8 | 0.758 | 0.663 (w16_b1_r1_p1_l0_s2 / w8_b1_r1_p0_l0_s2) | 0.663 | 0.663 |
| 32768 | 32 | 0.634 | 0.610 (w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.610 | 0.610 |
| 32768 | 256 | 0.657 | 0.589 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.589 | 0.589 |
| 131072 | 8 | 0.646 | 0.595 (w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.595 | 0.595 |
| 131072 | 32 | 0.647 | 0.602 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.602 | 0.602 |
| 131072 | 256 | 0.653 | 0.598 (w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2) | 0.598 | 0.598 |

## 5. Oracle-splits time per struct and condition (us; * = class-exact optimum, + = overall optimum)

| struct | K1 4k 8 | K1 4k 32 | K1 4k 256 | K1 32k 8 | K1 32k 32 | K1 32k 256 | K1 128k 8 | K1 128k 32 | K1 128k 256 | K2 4k 8 | K2 4k 32 | K2 4k 256 | K2 32k 8 | K2 32k 32 | K2 32k 256 | K2 128k 8 | K2 128k 32 | K2 128k 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig | 15 | 18 | 67 | 30 | 60 | 438 | 66 | 220 | 1686 | 18 | 23 | 102 | 40 | 95 | 666 | 103 | 340 | 2582 |
| w16_b1_r0_p0_l0_s0 | 14 | 17 | 66 | 30 | 59 | 430 | 65 | 216 | 1638 | 17 | 22 | 100 | 38 | 92 | 649 | 99 | 329 | 2516 |
| w16_b1_r0_p0_l0_s2 | 13 | 17 | 60 | 28 | 55 | 364 | 59 | 188 | 1422 | 17 | 21 | 96 | 37 | 91 | 627 | 98 | 319 | 2361 |
| w16_b1_r1_p0_l0_s0 | 10 | 15 | 63 | 21 | 58 | 418 | 58 | 212 | 1587 | 13 | 21 | 97 | 30 | 91 | 648 | 93 | 327 | 2415 |
| w16_b1_r1_p0_l0_s1 | 10 | 15 | 56 | 20 | 53 | 363 | 53 | 185 | 1421 | 14 | 21 | 97 | 31 | 91 | 618 | 92 | 315 | 2327 |
| w16_b1_r1_p0_l0_s2 | 10 | 14 | 56 | 19 | 52*+ | 366 | 52*+ | 184 | 1410 | 13 | 20 | 95 | 30 | 90 | 618 | 91 | 315 | 2349 |
| w16_b1_r1_p1_l0_s2 | 9 | 14 | 56 | 19*+ | 53 | 358 | 53 | 183 | 1417 | 12*+ | 22 | 101 | 31 | 90 | 616 | 92 | 309 | 2333 |
| w8_b1_r0_p0_l0_s0 | 13 | 17 | 63 | 28 | 62 | 440 | 67 | 225 | 1642 | 15 | 21 | 105 | 35 | 98 | 717 | 102 | 363 | 2808 |
| w8_b1_r1_p0_l0_s2 | 9 | 14 | 54 | 20 | 54 | 369 | 55 | 190 | 1411 | 12 | 19 | 93 | 29*+ | 90 | 636 | 90 | 323 | 2443 |
| w8_b1_r1_p1_l0_s2 | 9*+ | 13*+ | 54*+ | 19 | 54 | 354*+ | 54 | 182*+ | 1369*+ | 12 | 19*+ | 92 | 29 | 85 | 603 | 88 | 304 | 2288*+ |
| w8_b1_r1_p1_l2_s2 | 18 | 23 | 64 | 28 | 62 | 368 | 62 | 196 | 1370 | 31 | 38 | 127 | 48 | 104 | 633 | 105 | 321 | 2322 |
| w8_b2_r1_p0_l0_s2 | 10 | 14 | 54 | 20 | 52 | 366 | 54 | 184 | 1383 | 13 | 20 | 95 | 29 | 91 | 636 | 91 | 324 | 2446 |
| w8_b2_r1_p1_l0_s2 | 9 | 14 | 54 | 19 | 54 | 356 | 54 | 184 | 1406 | 12 | 19 | 92*+ | 29 | 85*+ | 600*+ | 87*+ | 303*+ | 2303 |
