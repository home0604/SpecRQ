# E9 parallel draft||verify: co-run microbench + DERIVED PEARL-style bound (syn_quote ctx32768 B16, TP4 RTX 4090, NCCL_P2P_DISABLE=1)

source: `results/E9/parallel_corun_v1_syn_quote_ctx32768_B16.json`; co-tenant any: False; launch ms {'median': 0.3513393457978964, 'max': 8.356689009815454, 'sleep_ms': 8.0}

## Measured: isolated vs co-run (median ms over 3x10 iters, max over ranks; spread = min-max)

overlap eff = (T_d + T_v - T_cor) / min(T_d, T_v); slowdown = side time in co-run / isolated; priorities: eq / vhi (verify stream priority -1) / dhi (draft high)

| draft | n draft steps | gamma | T_d iso | T_v iso | serial sum | T_cor eq (min-max) | T_cor vhi | T_cor dhi | draft slowdown eq | verify slowdown eq | overlap eff eq |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb512+RQ2 t2w | 1 | 4 | 12.17 | 31.79 | 43.96 | 45.20 (43.26-45.71) | 43.27 | 45.22 | 1.166 | 1.422 | -0.102 |
| hyb512+RQ2 t2w | 4 | 4 | 49.69 | 31.79 | 81.48 | 79.68 (79.51-82.32) | 79.59 | 79.67 | 1.003 | 2.507 | 0.057 |
| hyb512+RQ2 t2w | 1 | 6 | 12.17 | 33.55 | 45.73 | 45.04 (44.93-45.61) | 45.01 | 45.04 | 1.108 | 1.342 | 0.057 |
| hyb512+RQ2 t2w | 6 | 6 | 72.82 | 33.55 | 106.37 | 105.69 (105.31-106.44) | 105.68 | 105.70 | 1.018 | 3.150 | 0.020 |
| KV-free | 1 | 4 | 8.39 | 31.75 | 40.14 | 39.62 (39.57-40.14) | 39.63 | 39.62 | 1.137 | 1.248 | 0.062 |
| KV-free | 4 | 4 | 33.49 | 31.75 | 65.24 | 64.72 (64.57-65.60) | 64.74 | 64.69 | 1.030 | 2.039 | 0.016 |
| KV-free | 1 | 6 | 8.39 | 33.56 | 41.95 | 41.44 (41.35-41.89) | 41.44 | 41.43 | 1.171 | 1.235 | 0.061 |
| KV-free | 6 | 6 | 50.43 | 33.56 | 83.99 | 83.32 (83.07-83.89) | 83.34 | 83.39 | 1.025 | 2.483 | 0.020 |
| SnapKV-4096 | 1 | 4 | 11.28 | 31.74 | 43.02 | 42.37 (42.32-42.68) | 42.36 | 42.38 | 1.125 | 1.335 | 0.058 |
| SnapKV-4096 | 4 | 4 | 45.01 | 31.74 | 76.75 | 76.06 (75.97-76.69) | 76.10 | 76.07 | 1.030 | 2.396 | 0.022 |
| SnapKV-4096 | 1 | 6 | 11.28 | 33.56 | 44.84 | 44.16 (44.06-44.39) | 44.18 | 44.19 | 1.128 | 1.316 | 0.060 |
| SnapKV-4096 | 6 | 6 | 67.62 | 33.56 | 101.18 | 100.48 (100.27-101.03) | 100.48 | 100.54 | 1.019 | 2.994 | 0.021 |
| hyb512+RQ1 t1w | 1 | 4 | 11.17 | 31.78 | 42.95 | 42.29 (42.23-42.61) | 42.25 | 42.30 | 1.116 | 1.331 | 0.059 |
| hyb512+RQ1 t1w | 4 | 4 | 44.46 | 31.78 | 76.23 | 75.49 (75.38-75.82) | 75.48 | 75.54 | 1.028 | 2.376 | 0.023 |
| hyb512+RQ1 t1w | 1 | 6 | 11.17 | 33.59 | 44.76 | 44.10 (43.95-44.17) | 44.09 | 44.11 | 1.076 | 1.313 | 0.059 |
| hyb512+RQ1 t1w | 6 | 6 | 66.79 | 33.59 | 100.38 | 99.53 (99.37-100.28) | 99.60 | 99.56 | 1.010 | 2.963 | 0.025 |
| RQ2+r128 t2w | 1 | 4 | 11.94 | 31.79 | 43.73 | 43.05 (42.98-43.37) | 43.01 | 43.04 | 1.107 | 1.354 | 0.057 |
| RQ2+r128 t2w | 4 | 4 | 47.47 | 31.79 | 79.26 | 78.56 (78.45-79.65) | 78.59 | 78.59 | 1.027 | 2.471 | 0.022 |
| RQ2+r128 t2w | 1 | 6 | 11.94 | 33.61 | 45.55 | 44.84 (44.66-45.22) | 44.84 | 44.84 | 1.112 | 1.334 | 0.059 |
| RQ2+r128 t2w | 6 | 6 | 71.38 | 33.61 | 104.99 | 104.19 (103.93-104.78) | 104.13 | 104.16 | 1.017 | 3.100 | 0.024 |
| RQ1+r128 t2w-stage1 (nested) | 1 | 4 | 10.86 | 31.79 | 42.64 | 41.99 (41.92-42.42) | 41.96 | 41.99 | 1.120 | 1.321 | 0.060 |
| RQ1+r128 t2w-stage1 (nested) | 4 | 4 | 43.20 | 31.79 | 74.99 | 74.30 (74.17-75.01) | 74.27 | 74.32 | 1.029 | 2.337 | 0.022 |
| RQ1+r128 t2w-stage1 (nested) | 1 | 6 | 10.86 | 33.62 | 44.48 | 43.80 (43.69-44.10) | 43.82 | 43.80 | 1.088 | 1.303 | 0.062 |
| RQ1+r128 t2w-stage1 (nested) | 6 | 6 | 64.91 | 33.62 | 98.54 | 97.66 (97.43-98.15) | 97.68 | 97.74 | 1.017 | 2.905 | 0.026 |

## Measured: synthetic stream-concurrency check (concurrency_diag.py, B16, same streams / communicators)

sleep = GPU sleep kernel (latency only); gemv = memory-bound bf16 GEMV chain (~1 TP4 rank of weights); comm / comm7 = one forward's 66 all_reduces at 1 / 7 rows per sequence.

| pair | iso A | iso B | sum | max | co-run total | co-run total (B high priority) |
|---|---|---|---|---|---|---|
| sleep \|\| sleep | 6.89 | 6.89 | 13.79 | 6.89 | 6.90 | 6.90 |
| gemv \|\| gemv | 4.27 | 4.27 | 8.54 | 4.27 | 7.62 | 7.62 |
| comm \|\| comm | 2.96 | 2.96 | 5.92 | 2.96 | 3.48 | 3.48 |
| comm \|\| comm7 | 2.96 | 9.52 | 12.48 | 9.52 | 8.53 | 8.54 |
| gemv \|\| comm7 | 4.27 | 9.52 | 13.79 | 9.52 | 9.94 | 9.43 |
| comm \|\| gemv | 2.96 | 4.27 | 7.23 | 4.27 | 4.93 | 4.92 |
| sleep \|\| gemv | 6.89 | 4.27 | 11.17 | 6.89 | 6.91 | 6.91 |
| sleep \|\| comm7 | 6.89 | 9.52 | 16.41 | 9.52 | 7.52 | 7.51 |
| gemv \|\| sleep | 4.27 | 6.89 | 11.17 | 6.89 | 6.91 | 6.90 |

## Measured: components (gamma = 4), each co-run with the full other side

d_comm = the 66 all_reduces of one draft step only; d_gemm = draft step without comm and attention; d_attn = draft attention of all layers only; v_* the same for the verify. 'hidden' = (T_part + T_full - T_cor) / T_part.

| draft | pair | T_part iso | T_full iso | T_cor | hidden fraction of the part |
|---|---|---|---|---|---|
| hyb512+RQ2 t2w | d_comm \|\| v4 | 2.33 | 31.79 | 33.09 | 0.443 |
| hyb512+RQ2 t2w | d_gemm \|\| v4 | 5.75 | 31.79 | 36.55 | 0.172 |
| hyb512+RQ2 t2w | d_attn \|\| v4 | 3.57 | 31.79 | 34.63 | 0.205 |
| hyb512+RQ2 t2w | d1 \|\| v_comm | 6.29 | 12.17 | 17.75 | 0.114 |
| hyb512+RQ2 t2w | d1 \|\| v_gemm | 6.46 | 12.17 | 18.08 | 0.087 |
| hyb512+RQ2 t2w | d1 \|\| v_attn | 18.05 | 12.17 | 29.74 | 0.027 |
| KV-free | d_comm \|\| v4 | 2.31 | 31.75 | 32.73 | 0.577 |
| KV-free | d_gemm \|\| v4 | 5.75 | 31.75 | 36.58 | 0.159 |
| KV-free | d1 \|\| v_comm | 6.29 | 8.39 | 14.19 | 0.079 |
| KV-free | d1 \|\| v_gemm | 6.46 | 8.39 | 14.42 | 0.068 |
| KV-free | d1 \|\| v_attn | 18.05 | 8.39 | 26.05 | 0.022 |
| SnapKV-4096 | d_comm \|\| v4 | 2.31 | 31.74 | 32.98 | 0.465 |
| SnapKV-4096 | d_gemm \|\| v4 | 5.75 | 31.74 | 36.55 | 0.163 |
| SnapKV-4096 | d_attn \|\| v4 | 2.73 | 31.74 | 33.59 | 0.323 |
| SnapKV-4096 | d1 \|\| v_comm | 6.29 | 11.28 | 16.93 | 0.103 |
| SnapKV-4096 | d1 \|\| v_gemm | 6.46 | 11.28 | 17.23 | 0.080 |
| SnapKV-4096 | d1 \|\| v_attn | 18.05 | 11.28 | 28.88 | 0.025 |
| hyb512+RQ1 t1w | d_comm \|\| v4 | 2.35 | 31.78 | 33.39 | 0.316 |
| hyb512+RQ1 t1w | d_gemm \|\| v4 | 5.75 | 31.78 | 36.55 | 0.170 |
| hyb512+RQ1 t1w | d_attn \|\| v4 | 2.38 | 31.78 | 33.51 | 0.272 |
| hyb512+RQ1 t1w | d1 \|\| v_comm | 6.29 | 11.17 | 16.83 | 0.101 |
| hyb512+RQ1 t1w | d1 \|\| v_gemm | 6.46 | 11.17 | 17.13 | 0.079 |
| hyb512+RQ1 t1w | d1 \|\| v_attn | 18.05 | 11.17 | 28.73 | 0.027 |
| RQ2+r128 t2w | d_comm \|\| v4 | 2.38 | 31.79 | 33.39 | 0.326 |
| RQ2+r128 t2w | d_gemm \|\| v4 | 5.75 | 31.79 | 36.58 | 0.166 |
| RQ2+r128 t2w | d_attn \|\| v4 | 3.31 | 31.79 | 34.39 | 0.216 |
| RQ2+r128 t2w | d1 \|\| v_comm | 6.29 | 11.94 | 17.54 | 0.109 |
| RQ2+r128 t2w | d1 \|\| v_gemm | 6.46 | 11.94 | 17.86 | 0.083 |
| RQ2+r128 t2w | d1 \|\| v_attn | 18.05 | 11.94 | 29.50 | 0.027 |
| RQ1+r128 t2w-stage1 (nested) | d_comm \|\| v4 | 2.36 | 31.79 | 33.40 | 0.318 |
| RQ1+r128 t2w-stage1 (nested) | d_gemm \|\| v4 | 5.75 | 31.79 | 36.58 | 0.166 |
| RQ1+r128 t2w-stage1 (nested) | d_attn \|\| v4 | 2.15 | 31.79 | 33.25 | 0.319 |
| RQ1+r128 t2w-stage1 (nested) | d1 \|\| v_comm | 6.29 | 10.86 | 16.52 | 0.100 |
| RQ1+r128 t2w-stage1 (nested) | d1 \|\| v_gemm | 6.46 | 10.86 | 16.82 | 0.077 |
| RQ1+r128 t2w-stage1 (nested) | d1 \|\| v_attn | 18.05 | 10.86 | 28.42 | 0.027 |

## DERIVED: idealized PEARL-style schedule vs serial (tau / acceptance from v6e2e / v8 measured all-active runs)

Serial round   : T_ser = T_dG + T_v + T_o                          tokens = tau  
Parallel (U)   : T = T_cor + (1 - p_row) * T_dG + T_o             (task formula: drafting the next gamma during the  
                 verify; on rejection the pre-drafted tokens are wasted and gamma fresh draft steps run serially;  
                 p_row = P(all gamma accepted) per row and round)  
Parallel (L)   : T = T_cor + (1 - p_batch) * T_dG + T_o           (lock-step batch: any rejecting row stalls the batch;  
                 p_batch = P(all B rows accept all gamma) per round)  
Ideal (U, no contention): T_cor replaced by max(T_dG, T_v)  
tokens / round = tau for every schedule (optimistic: the verify's bonus token is assumed to be usable as in the serial  
schedule; in a real PEARL loop it must also match the pre-drafted continuation).  
T_dG, T_v, T_cor: corun_bench iso / co-run medians (T_cor = best priority of 'dG || vG'); T_o = per-round  
non-draft/verify time of the source e2e run (accept + encode + host); T_AR = that run's AR step.

| draft | gamma | tau aa | p_row | p_batch | T_AR | T_dG | T_v | T_o | T_cor best | S serial (model) | S serial meas (e2e) | S par U | S par L | S par U ideal | U / serial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | 4 | 4.30 | 0.752 | 0.154 | 26.41 | 45.01 | 31.74 | 0.04 | 76.06 | 1.480 | 1.468 | 1.303 | 0.995 | 2.022 | 0.880 |
| hyb512+RQ2 t2w | 6 | 6.90 | 0.976 | 0.676 | 26.41 | 72.82 | 33.55 | 0.04 | 105.68 | 1.713 | 1.714 | 1.696 | 1.409 | 2.444 | 0.990 |
| hyb512+RQ2 t2w | 4 | 4.94 | 0.975 | 0.673 | 26.41 | 49.69 | 31.79 | 0.06 | 79.59 | 1.602 | 1.615 | 1.614 | 1.362 | 2.560 | 1.008 |
| hyb512+RQ1 t1w | 4 | 4.26 | 0.690 | 0.093 | 26.41 | 44.46 | 31.78 | 0.05 | 75.48 | 1.475 | 1.467 | 1.259 | 0.971 | 1.930 | 0.854 |
| hyb512+RQ1 t1w | 6 | 5.25 | 0.526 | 0.047 | 26.41 | 66.79 | 33.59 | 0.05 | 99.53 | 1.382 | 1.383 | 1.058 | 0.850 | 1.409 | 0.765 |
| RQ2+r128 t2w | 4 | 4.86 | 0.938 | 0.442 | 26.41 | 47.47 | 31.79 | 0.47 | 78.56 | 1.610 | 1.607 | 1.565 | 1.216 | 2.521 | 0.972 |
| RQ2+r128 t2w | 6 | 6.70 | 0.919 | 0.378 | 26.41 | 71.38 | 33.61 | 0.51 | 104.13 | 1.677 | 1.686 | 1.602 | 1.187 | 2.277 | 0.955 |
| RQ1+r128 t2w-stage1 (nested) | 4 | 3.63 | 0.446 | 0.032 | 26.41 | 43.20 | 31.79 | 0.25 | 74.27 | 1.274 | 1.271 | 0.974 | 0.824 | 1.423 | 0.764 |

best per schedule: S_serial_model 1.713 (hyb512+RQ2 t2w g6); S_par_U 1.696 (hyb512+RQ2 t2w g6); S_par_L 1.409 (hyb512+RQ2 t2w g6); S_par_U_ideal 2.560 (hyb512+RQ2 t2w g4)
