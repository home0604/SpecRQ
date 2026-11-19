# E10 phase 1: structural knobs of the LUT VQ-KV draft attention (NVIDIA GeForce RTX 4090)

- raw: `../../results/E10/lut_tiling_p1_waveq_raw.json`; metric `t_tot_us` (partial + combine, us per call, median of 9 CUDA-graph reps, codes stream from DRAM).
- limits: {'sm_count': 128, 'smem_per_sm': 102400, 'smem_per_block_optin': 101376, 'reserved_smem_per_block': 1024, 'regs_per_sm': 65536, 'threads_per_sm': 1536, 'blocks_per_sm': 24, 'l2_bytes': 75497472, 'cc_major': 8, 'cc_minor': 9}; ref read BW 942 GB/s
- `results/E10/lut_tiling_p1_waveq_raw.json`: GPU 6 uuid GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f; apps before ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3940 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5044 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-ce80dfbd-cb70-c9ce-b491-4a2146bc402c, 3519084, 734 MiB']; after ['GPU-4c14c121-b681-d11e-3f9f-4183c3b576f0, 3510999, 3942 MiB', 'GPU-e1269272-d9f8-436f-f9b1-19c4e73bf388, 3417523, 5156 MiB', 'GPU-170c8ed4-3e29-2bc3-9029-68662abdc849, 3491858, 3944 MiB', 'GPU-ea94c586-2a95-d769-6af9-e4921dc56a0f, 3417526, 5158 MiB', 'GPU-ce80dfbd-cb70-c9ce-b491-4a2146bc402c, 3525125, 774 MiB', 'GPU-8522ca3c-3d2f-69c2-407a-02b1a73bcb22, 3525135, 774 MiB', 'GPU-ed76c2dd-ea7b-b6b4-0bbc-8a085b4d4f9f, 3521126, 492 MiB']
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
| orig | exact | 1.000 | 1.00..1.00 | 1.000 | 1.00..1.00 | 20 |
| w16_b1_r0_p0_l0_s0 | exact | 1.022 | 1.01..1.04 | 1.033 | 1.02..1.07 | 20 |
| w16_b1_r0_p0_l0_s2 | exact | 1.155 | 1.10..1.20 | 1.080 | 1.05..1.14 | 20 |
| w16_b1_r1_p0_l0_s0 | exact | 1.055 | 1.03..1.11 | 1.058 | 1.03..1.10 | 20 |
| w16_b1_r1_p0_l0_s1 | sb1 | 1.190 | 1.13..1.23 | 1.101 | 1.07..1.13 | 20 |
| w16_b1_r1_p0_l0_s2 | exact | 1.205 | 1.16..1.26 | 1.110 | 1.08..1.15 | 20 |
| w16_b1_r1_p1_l0_s2 | exact | 1.207 | 1.14..1.29 | 1.104 | 1.04..1.15 | 20 |
| w8_b1_r0_p0_l0_s0 | exact | 1.004 | 0.96..1.09 | 0.954 | 0.92..0.99 | 20 |
| w8_b1_r1_p0_l0_s2 | exact | 1.200 | 1.16..1.25 | 1.084 | 1.05..1.16 | 20 |
| w8_b1_r1_p1_l0_s2 | exact | 1.232 | 1.19..1.29 | 1.142 | 1.11..1.23 | 20 |
| w8_b1_r1_p1_l2_s2 | int8 | 1.081 | 0.90..1.17 | 0.968 | 0.81..1.06 | 20 |
| w8_b2_r1_p0_l0_s2 | exact | 1.201 | 1.16..1.26 | 1.075 | 1.04..1.13 | 20 |
| w8_b2_r1_p1_l0_s2 | exact | 1.229 | 1.19..1.28 | 1.143 | 1.11..1.23 | 20 |

## 2a. Optimum per condition, class `exact` (oracle splits)

| KST | n | B*Hkv | best struct | chunk | splits | t us | orig best us | speedup vs orig | 2nd best (regret) | best with rule splits (regret vs oracle opt) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16384 | 24 | w16_b1_r1_p1_l0_s2 | 3328 | 5 | 24.7 | 31.7 | 1.286 | w8_b1_r1_p1_l0_s2 (2.3%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 48 | w8_b1_r1_p1_l0_s2 | 3328 | 5 | 43.9 | 56.1 | 1.278 | w8_b2_r1_p1_l0_s2 (0.0%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 96 | w8_b1_r1_p1_l0_s2 | 4096 | 4 | 81.4 | 98.4 | 1.209 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (2.8%) |
| 1 | 16384 | 160 | w8_b1_r1_p1_l0_s2 | 5504 | 3 | 128.1 | 165.5 | 1.292 | w8_b2_r1_p1_l0_s2 (2.1%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 192 | w8_b1_r1_p1_l0_s2 | 8192 | 2 | 150.1 | 179.2 | 1.194 | w8_b2_r1_p1_l0_s2 (0.1%) | w16_b1_r1_p0_l0_s2 (2.5%) |
| 1 | 65536 | 24 | w16_b1_r1_p0_l0_s2 | 13120 | 5 | 75.9 | 93.9 | 1.237 | w16_b1_r1_p1_l0_s2 (0.5%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 65536 | 48 | w8_b2_r1_p1_l0_s2 | 8192 | 8 | 150.1 | 181.5 | 1.209 | w16_b1_r1_p0_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.2%) |
| 1 | 65536 | 96 | w8_b1_r1_p1_l0_s2 | 8192 | 8 | 279.5 | 344.2 | 1.231 | w8_b2_r1_p1_l0_s2 (0.4%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 65536 | 160 | w8_b1_r1_p1_l0_s2 | 16384 | 4 | 461.1 | 573.4 | 1.244 | w8_b2_r1_p1_l0_s2 (1.6%) | w8_b1_r1_p1_l0_s2 (0.2%) |
| 1 | 65536 | 192 | w8_b2_r1_p1_l0_s2 | 16384 | 4 | 537.9 | 661.9 | 1.231 | w16_b1_r1_p1_l0_s2 (1.2%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 24 | w8_b2_r1_p1_l0_s2 | 3328 | 5 | 37.4 | 46.2 | 1.233 | w8_b1_r1_p1_l0_s2 (0.0%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 48 | w8_b2_r1_p1_l0_s2 | 3328 | 5 | 71.1 | 83.7 | 1.178 | w8_b1_r1_p1_l0_s2 (0.8%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 96 | w8_b2_r1_p1_l0_s2 | 4096 | 4 | 135.9 | 152.7 | 1.124 | w8_b1_r1_p1_l0_s2 (0.2%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 160 | w8_b2_r1_p1_l0_s2 | 5504 | 3 | 218.8 | 253.1 | 1.157 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 192 | w8_b1_r1_p1_l0_s2 | 8192 | 2 | 251.3 | 280.2 | 1.115 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 24 | w8_b2_r1_p1_l0_s2 | 13120 | 5 | 123.8 | 141.8 | 1.145 | w8_b1_r1_p1_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 48 | w8_b1_r1_p1_l0_s2 | 13120 | 5 | 241.9 | 276.3 | 1.142 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b2_r1_p1_l0_s2 (3.6%) |
| 2 | 65536 | 96 | w8_b2_r1_p1_l0_s2 | 16384 | 4 | 467.4 | 523.4 | 1.120 | w8_b1_r1_p1_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 160 | w8_b1_r1_p1_l0_s2 | 16384 | 4 | 770.8 | 866.8 | 1.125 | w8_b2_r1_p1_l0_s2 (0.7%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 192 | w8_b2_r1_p1_l0_s2 | 32768 | 2 | 901.4 | 1003.5 | 1.113 | w8_b1_r1_p1_l0_s2 (0.5%) | w8_b2_r1_p1_l0_s2 (0.0%) |

## 2b. Optimum per condition, class `exact+int8` (oracle splits)

| KST | n | B*Hkv | best struct | chunk | splits | t us | orig best us | speedup vs orig | 2nd best (regret) | best with rule splits (regret vs oracle opt) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16384 | 24 | w16_b1_r1_p1_l0_s2 | 3328 | 5 | 24.7 | 31.7 | 1.286 | w8_b1_r1_p1_l0_s2 (2.3%) | w16_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 48 | w8_b1_r1_p1_l0_s2 | 3328 | 5 | 43.9 | 56.1 | 1.278 | w8_b2_r1_p1_l0_s2 (0.0%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 96 | w8_b1_r1_p1_l0_s2 | 4096 | 4 | 81.4 | 98.4 | 1.209 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (2.8%) |
| 1 | 16384 | 160 | w8_b1_r1_p1_l0_s2 | 5504 | 3 | 128.1 | 165.5 | 1.292 | w8_b2_r1_p1_l0_s2 (2.1%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 16384 | 192 | w8_b1_r1_p1_l0_s2 | 8192 | 2 | 150.1 | 179.2 | 1.194 | w8_b2_r1_p1_l0_s2 (0.1%) | w16_b1_r1_p0_l0_s2 (2.5%) |
| 1 | 65536 | 24 | w16_b1_r1_p0_l0_s2 | 13120 | 5 | 75.9 | 93.9 | 1.237 | w16_b1_r1_p1_l0_s2 (0.5%) | w16_b1_r1_p0_l0_s2 (0.0%) |
| 1 | 65536 | 48 | w8_b2_r1_p1_l0_s2 | 8192 | 8 | 150.1 | 181.5 | 1.209 | w16_b1_r1_p0_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.2%) |
| 1 | 65536 | 96 | w8_b1_r1_p1_l0_s2 | 8192 | 8 | 279.5 | 344.2 | 1.231 | w8_b2_r1_p1_l0_s2 (0.4%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 1 | 65536 | 160 | w8_b1_r1_p1_l0_s2 | 16384 | 4 | 461.1 | 573.4 | 1.244 | w8_b2_r1_p1_l0_s2 (1.6%) | w8_b1_r1_p1_l0_s2 (0.2%) |
| 1 | 65536 | 192 | w8_b2_r1_p1_l0_s2 | 16384 | 4 | 537.9 | 661.9 | 1.231 | w16_b1_r1_p1_l0_s2 (1.2%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 24 | w8_b2_r1_p1_l0_s2 | 3328 | 5 | 37.4 | 46.2 | 1.233 | w8_b1_r1_p1_l0_s2 (0.0%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 48 | w8_b2_r1_p1_l0_s2 | 3328 | 5 | 71.1 | 83.7 | 1.178 | w8_b1_r1_p1_l0_s2 (0.8%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 96 | w8_b2_r1_p1_l0_s2 | 4096 | 4 | 135.9 | 152.7 | 1.124 | w8_b1_r1_p1_l0_s2 (0.2%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 160 | w8_b2_r1_p1_l0_s2 | 5504 | 3 | 218.8 | 253.1 | 1.157 | w8_b1_r1_p1_l0_s2 (0.4%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 16384 | 192 | w8_b1_r1_p1_l0_s2 | 8192 | 2 | 251.3 | 280.2 | 1.115 | w8_b2_r1_p1_l0_s2 (0.2%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 24 | w8_b2_r1_p1_l0_s2 | 13120 | 5 | 123.8 | 141.8 | 1.145 | w8_b1_r1_p1_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 48 | w8_b1_r1_p1_l0_s2 | 13120 | 5 | 241.9 | 276.3 | 1.142 | w8_b2_r1_p1_l0_s2 (0.6%) | w8_b2_r1_p1_l0_s2 (3.6%) |
| 2 | 65536 | 96 | w8_b2_r1_p1_l0_s2 | 16384 | 4 | 467.4 | 523.4 | 1.120 | w8_b1_r1_p1_l0_s2 (0.1%) | w8_b2_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 160 | w8_b1_r1_p1_l0_s2 | 16384 | 4 | 770.8 | 866.8 | 1.125 | w8_b2_r1_p1_l0_s2 (0.7%) | w8_b1_r1_p1_l0_s2 (0.0%) |
| 2 | 65536 | 192 | w8_b2_r1_p1_l0_s2 | 32768 | 2 | 901.4 | 1003.5 | 1.113 | w8_b1_r1_p1_l0_s2 (0.5%) | w8_b2_r1_p1_l0_s2 (0.0%) |

## 3a. Structural-config regret, class `exact`

Regret = T(policy struct, splits by oracle | rule) / T(best struct at oracle splits) - 1.

- [oracle splits] one global struct: `w8_b2_r1_p1_l0_s2` -> max 2.6% / median 0.1% (n=20)
  - [oracle splits] best struct for KST1 only: `w8_b2_r1_p1_l0_s2` -> on KST1: max 2.6% / median 0.3% (n=10); transferred to the other KST: max 0.7% / median 0.0% (n=10)
  - [oracle splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=10); transferred to the other KST: max 2.6% / median 0.3% (n=10)
  - [oracle splits] original kernel: max 29.2% / median 20.1% (n=20)
- [rule splits] one global struct: `w8_b2_r1_p1_l0_s2` -> max 4.4% / median 0.2% (n=20)
  - [rule splits] best struct for KST1 only: `w8_b2_r1_p1_l0_s2` -> on KST1: max 4.4% / median 2.5% (n=10); transferred to the other KST: max 3.6% / median 0.0% (n=10)
  - [rule splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 3.6% / median 0.0% (n=10); transferred to the other KST: max 4.4% / median 2.5% (n=10)
  - [rule splits] original kernel: max 29.4% / median 20.1% (n=20)

| axis | splits | regret of using cond j's best struct on cond i (i, j differ only in that axis) |
|---|---|---|
| KST | oracle | max 10.2% / median 0.2% (n=20) |
| KST | rule | max 10.2% / median 0.6% (n=20) |
| n | oracle | max 3.7% / median 0.1% (n=20) |
| n | rule | max 3.7% / median 0.4% (n=20) |
| B*Hkv | oracle | max 6.1% / median 0.1% (n=80) |
| B*Hkv | rule | max 6.1% / median 0.4% (n=80) |

KST transfer per (n, B*Hkv): best struct KST1 / KST2; regret of KST2-best on KST1 / KST1-best on KST2 (oracle splits)

| n \ B*Hkv | 24 | 48 | 96 | 160 | 192 |
|---|---|---|---|---|---|
| 16384 | w16_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 2.6% / 10.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.0% / 0.8% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.2% / 0.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 2.1% / 0.4% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% |
| 65536 | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 2.4% / 5.4% | w8_b2_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.2% / 0.6% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.4% / 0.1% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% | w8_b2_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.0% / 0.0% |

## 3b. Structural-config regret, class `exact+int8`

Regret = T(policy struct, splits by oracle | rule) / T(best struct at oracle splits) - 1.

- [oracle splits] one global struct: `w8_b2_r1_p1_l0_s2` -> max 2.6% / median 0.1% (n=20)
  - [oracle splits] best struct for KST1 only: `w8_b2_r1_p1_l0_s2` -> on KST1: max 2.6% / median 0.3% (n=10); transferred to the other KST: max 0.7% / median 0.0% (n=10)
  - [oracle splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 0.7% / median 0.0% (n=10); transferred to the other KST: max 2.6% / median 0.3% (n=10)
  - [oracle splits] original kernel: max 29.2% / median 20.1% (n=20)
- [rule splits] one global struct: `w8_b2_r1_p1_l0_s2` -> max 4.4% / median 0.2% (n=20)
  - [rule splits] best struct for KST1 only: `w8_b2_r1_p1_l0_s2` -> on KST1: max 4.4% / median 2.5% (n=10); transferred to the other KST: max 3.6% / median 0.0% (n=10)
  - [rule splits] best struct for KST2 only: `w8_b2_r1_p1_l0_s2` -> on KST2: max 3.6% / median 0.0% (n=10); transferred to the other KST: max 4.4% / median 2.5% (n=10)
  - [rule splits] original kernel: max 29.4% / median 20.1% (n=20)

| axis | splits | regret of using cond j's best struct on cond i (i, j differ only in that axis) |
|---|---|---|
| KST | oracle | max 10.2% / median 0.2% (n=20) |
| KST | rule | max 10.2% / median 0.6% (n=20) |
| n | oracle | max 3.7% / median 0.1% (n=20) |
| n | rule | max 3.7% / median 0.4% (n=20) |
| B*Hkv | oracle | max 6.1% / median 0.1% (n=80) |
| B*Hkv | rule | max 6.1% / median 0.4% (n=80) |

KST transfer per (n, B*Hkv): best struct KST1 / KST2; regret of KST2-best on KST1 / KST1-best on KST2 (oracle splits)

| n \ B*Hkv | 24 | 48 | 96 | 160 | 192 |
|---|---|---|---|---|---|
| 16384 | w16_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 2.6% / 10.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.0% / 0.8% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.2% / 0.2% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 2.1% / 0.4% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% |
| 65536 | w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2: 2.4% / 5.4% | w8_b2_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.2% / 0.6% | w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.4% / 0.1% | w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2: 0.0% / 0.0% | w8_b2_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2: 0.0% / 0.0% |

## 3c. One knob at a time around a base struct (oracle splits)

For each knob, the other fields are fixed at the base; per condition the best value of the knob; regret = T(value) / T(best value of that knob at that condition) - 1. 'x-KST' = regret of the other KST's best value at the same (n, B*Hkv); 'one value' = best single value of the knob over all conditions.


base `w8_b1_r1_p1_l0_s2`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 16k/24:16,16k/48:8,16k/96:8,16k/160:8,16k/192:8,64k/24:16,64k/48:8,64k/96:8,64k/160:8,64k/192:16 ; 16k/24:8,16k/48:8,16k/96:8,16k/160:8,16k/192:8,64k/24:8,64k/48:8,64k/96:8,64k/160:8,64k/192:8 | 10.2% | 2.5% (8) | 2.5% ({1: 8, 2: 8}) |
| b | [1, 2] | 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:2,64k/96:1,64k/160:1,64k/192:2 ; 16k/24:2,16k/48:2,16k/96:2,16k/160:2,16k/192:1,64k/24:2,64k/48:1,64k/96:2,64k/160:1,64k/192:2 | 2.1% | 2.1% (2) | 2.1% ({1: 2, 2: 2}) |
| p | [0, 1] | 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:0 ; 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:1 | 5.9% | 1.8% (1) | 1.8% ({1: 1, 2: 1}) |
| l | [0, 2] | 16k/24:0,16k/48:0,16k/96:0,16k/160:0,16k/192:0,64k/24:0,64k/48:0,64k/96:0,64k/160:0,64k/192:0 ; 16k/24:0,16k/48:0,16k/96:0,16k/160:0,16k/192:0,64k/24:0,64k/48:0,64k/96:0,64k/160:0,64k/192:0 | 0.0% | 0.0% (0) | 0.0% ({1: 0, 2: 0}) |

base `w16_b1_r1_p0_l0_s2`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 16k/24:16,16k/48:8,16k/96:8,16k/160:8,16k/192:16,64k/24:16,64k/48:16,64k/96:16,64k/160:8,64k/192:16 ; 16k/24:8,16k/48:16,16k/96:16,16k/160:16,16k/192:16,64k/24:16,64k/48:16,64k/96:16,64k/160:16,64k/192:16 | 4.4% | 3.0% (16) | 3.0% ({1: 16, 2: 16}) |
| r | [0, 1] | 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:1 ; 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:1 | 0.0% | 0.0% (1) | 0.0% ({1: 1, 2: 1}) |
| p | [0, 1] | 16k/24:1,16k/48:1,16k/96:0,16k/160:1,16k/192:0,64k/24:0,64k/48:0,64k/96:0,64k/160:1,64k/192:1 ; 16k/24:0,16k/48:0,16k/96:0,16k/160:1,16k/192:0,64k/24:1,64k/48:0,64k/96:0,64k/160:1,64k/192:0 | 3.0% | 2.3% (0) | 1.9% ({1: 1, 2: 0}) |
| s | [0, 1, 2] | 16k/24:2,16k/48:2,16k/96:2,16k/160:2,16k/192:1,64k/24:2,64k/48:2,64k/96:2,64k/160:2,64k/192:2 ; 16k/24:2,16k/48:2,16k/96:2,16k/160:2,16k/192:2,64k/24:2,64k/48:2,64k/96:2,64k/160:2,64k/192:2 | 0.8% | 0.6% (2) | 0.6% ({1: 2, 2: 2}) |

base `w16_b1_r0_p0_l0_s0`

| knob | values measured | per-condition best value (KST1 ; KST2) | max x-KST regret | max regret of one value (value) | max regret of per-KST value (values) |
|---|---|---|---|---|---|
| w | [8, 16] | 16k/24:16,16k/48:8,16k/96:16,16k/160:8,16k/192:16,64k/24:16,64k/48:16,64k/96:16,64k/160:16,64k/192:16 ; 16k/24:16,16k/48:16,16k/96:16,16k/160:16,16k/192:16,64k/24:16,64k/48:16,64k/96:16,64k/160:16,64k/192:16 | 8.6% | 5.9% (16) | 5.3% ({1: 8, 2: 16}) |
| r | [0, 1] | 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:0 ; 16k/24:1,16k/48:1,16k/96:1,16k/160:1,16k/192:1,64k/24:1,64k/48:1,64k/96:1,64k/160:1,64k/192:1 | 1.5% | 1.5% (1) | 1.5% ({1: 1, 2: 1}) |
| s | [0, 2] | 16k/24:2,16k/48:2,16k/96:2,16k/160:2,16k/192:2,64k/24:2,64k/48:2,64k/96:2,64k/160:2,64k/192:2 ; 16k/24:2,16k/48:2,16k/96:2,16k/160:2,16k/192:2,64k/24:2,64k/48:2,64k/96:2,64k/160:2,64k/192:2 | 0.0% | 0.0% (2) | 0.0% ({1: 2, 2: 2}) |

## 4. KST1 / KST2 time ratio: original vs best struct per KST (oracle splits)

| n | B*Hkv | orig ratio | best exact ratio (KST1 / KST2 struct) | best exact+int8 ratio | best incl. sb1 ratio |
|---|---|---|---|---|---|
| 16384 | 24 | 0.687 | 0.659 (w16_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.659 | 0.659 |
| 16384 | 48 | 0.670 | 0.618 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.618 | 0.618 |
| 16384 | 96 | 0.644 | 0.599 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.599 | 0.599 |
| 16384 | 160 | 0.654 | 0.585 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.585 | 0.585 |
| 16384 | 192 | 0.639 | 0.597 (w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2) | 0.597 | 0.597 |
| 65536 | 24 | 0.662 | 0.613 (w16_b1_r1_p0_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.613 | 0.613 |
| 65536 | 48 | 0.657 | 0.621 (w8_b2_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2) | 0.621 | 0.621 |
| 65536 | 96 | 0.658 | 0.598 (w8_b1_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.598 | 0.598 |
| 65536 | 160 | 0.662 | 0.598 (w8_b1_r1_p1_l0_s2 / w8_b1_r1_p1_l0_s2) | 0.598 | 0.598 |
| 65536 | 192 | 0.660 | 0.597 (w8_b2_r1_p1_l0_s2 / w8_b2_r1_p1_l0_s2) | 0.597 | 0.597 |

## 5. Oracle-splits time per struct and condition (us; * = class-exact optimum, + = overall optimum)

| struct | K1 16k 24 | K1 16k 48 | K1 16k 96 | K1 16k 160 | K1 16k 192 | K1 64k 24 | K1 64k 48 | K1 64k 96 | K1 64k 160 | K1 64k 192 | K2 16k 24 | K2 16k 48 | K2 16k 96 | K2 16k 160 | K2 16k 192 | K2 64k 24 | K2 64k 48 | K2 64k 96 | K2 64k 160 | K2 64k 192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| orig | 32 | 56 | 98 | 165 | 179 | 94 | 181 | 344 | 573 | 662 | 46 | 84 | 153 | 253 | 280 | 142 | 276 | 523 | 867 | 1004 |
| w16_b1_r0_p0_l0_s0 | 31 | 55 | 97 | 161 | 177 | 93 | 179 | 338 | 565 | 635 | 43 | 80 | 149 | 247 | 273 | 139 | 269 | 508 | 846 | 979 |
| w16_b1_r0_p0_l0_s2 | 28 | 48 | 89 | 142 | 159 | 78 | 155 | 300 | 500 | 566 | 42 | 73 | 137 | 238 | 266 | 134 | 257 | 494 | 809 | 940 |
| w16_b1_r1_p0_l0_s0 | 29 | 52 | 93 | 154 | 173 | 91 | 172 | 330 | 545 | 644 | 42 | 78 | 144 | 233 | 266 | 137 | 264 | 501 | 831 | 975 |
| w16_b1_r1_p0_l0_s1 | 26 | 47 | 87 | 137 | 153 | 76 | 151 | 292 | 483 | 555 | 41 | 74 | 137 | 226 | 261 | 131 | 250 | 484 | 794 | 921 |
| w16_b1_r1_p0_l0_s2 | 25 | 46 | 85 | 135 | 154 | 76*+ | 150 | 290 | 480 | 544 | 40 | 73 | 137 | 222 | 259 | 131 | 249 | 482 | 791 | 920 |
| w16_b1_r1_p1_l0_s2 | 25*+ | 46 | 86 | 135 | 155 | 76 | 151 | 291 | 472 | 544 | 41 | 73 | 137 | 221 | 268 | 128 | 249 | 489 | 784 | 922 |
| w8_b1_r0_p0_l0_s0 | 32 | 52 | 100 | 156 | 186 | 96 | 181 | 354 | 579 | 649 | 46 | 86 | 155 | 261 | 293 | 154 | 293 | 556 | 924 | 1080 |
| w8_b1_r1_p0_l0_s2 | 26 | 45 | 85 | 133 | 154 | 79 | 153 | 292 | 474 | 547 | 40 | 75 | 138 | 228 | 261 | 135 | 260 | 492 | 814 | 959 |
| w8_b1_r1_p1_l0_s2 | 25 | 44*+ | 81*+ | 128*+ | 150*+ | 77 | 150 | 280*+ | 461*+ | 558 | 37 | 72 | 136 | 220 | 251*+ | 124 | 242*+ | 468 | 771*+ | 906 |
| w8_b1_r1_p1_l2_s2 | 35 | 55 | 98 | 149 | 165 | 87 | 158 | 301 | 490 | 566 | 57 | 89 | 150 | 285 | 306 | 142 | 269 | 507 | 844 | 946 |
| w8_b2_r1_p0_l0_s2 | 27 | 45 | 85 | 131 | 154 | 78 | 151 | 294 | 469 | 562 | 41 | 76 | 140 | 231 | 264 | 136 | 261 | 494 | 816 | 952 |
| w8_b2_r1_p1_l0_s2 | 25 | 44 | 82 | 131 | 150 | 78 | 150*+ | 281 | 468 | 538*+ | 37*+ | 71*+ | 136*+ | 219*+ | 252 | 124*+ | 243 | 467*+ | 776 | 901*+ |
