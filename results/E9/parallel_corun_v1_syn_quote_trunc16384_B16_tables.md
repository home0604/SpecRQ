# E9 parallel draft||verify: co-run microbench + DERIVED PEARL-style bound (syn_quote ctx16384 B16, TP4 RTX 4090, NCCL_P2P_DISABLE=1)

source: `results/E9/parallel_corun_v1_syn_quote_trunc16384_B16.json`; co-tenant any: False; launch ms {'median': 0.3925145138055086, 'max': 8.036519400775433, 'sleep_ms': 8.0}

## Measured: isolated vs co-run (median ms over 3x10 iters, max over ranks; spread = min-max)

overlap eff = (T_d + T_v - T_cor) / min(T_d, T_v); slowdown = side time in co-run / isolated; priorities: eq / vhi (verify stream priority -1) / dhi (draft high)

| draft | n draft steps | gamma | T_d iso | T_v iso | serial sum | T_cor eq (min-max) | T_cor vhi | T_cor dhi | draft slowdown eq | verify slowdown eq | overlap eff eq |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hyb512+RQ2 t2w | 1 | 4 | 10.93 | 23.66 | 34.59 | 34.03 (33.95-34.36) | 33.96 | 34.03 | 1.073 | 1.438 | 0.051 |
| hyb512+RQ2 t2w | 4 | 4 | 43.65 | 23.66 | 67.32 | 66.71 (66.62-67.21) | 66.68 | 66.75 | 1.018 | 2.819 | 0.026 |
| hyb512+RQ2 t2w | 1 | 6 | 10.93 | 25.14 | 36.07 | 35.47 (35.23-36.11) | 35.44 | 35.48 | 1.080 | 1.411 | 0.055 |
| hyb512+RQ2 t2w | 6 | 6 | 65.58 | 25.14 | 90.72 | 90.01 (89.80-91.27) | 89.99 | 90.19 | 1.012 | 3.581 | 0.028 |
| KV-free | 1 | 4 | 8.35 | 23.68 | 32.03 | 31.55 (31.48-31.65) | 31.50 | 31.55 | 1.100 | 1.332 | 0.057 |
| KV-free | 4 | 4 | 33.42 | 23.68 | 57.10 | 56.54 (56.44-57.06) | 56.54 | 56.55 | 1.022 | 2.388 | 0.024 |
| KV-free | 1 | 6 | 8.35 | 25.15 | 33.49 | 32.99 (32.86-33.53) | 32.95 | 32.97 | 1.108 | 1.312 | 0.060 |
| KV-free | 6 | 6 | 50.26 | 25.15 | 75.41 | 74.69 (74.45-75.46) | 74.92 | 74.81 | 1.014 | 2.970 | 0.029 |
| SnapKV-4096 | 1 | 4 | 11.24 | 23.69 | 34.93 | 34.27 (34.20-34.89) | 34.29 | 34.26 | 1.082 | 1.447 | 0.058 |
| SnapKV-4096 | 4 | 4 | 44.94 | 23.69 | 68.63 | 68.00 (67.85-68.61) | 67.96 | 67.98 | 1.021 | 2.871 | 0.027 |
| SnapKV-4096 | 1 | 6 | 11.24 | 25.14 | 36.38 | 35.71 (35.62-35.79) | 35.73 | 35.71 | 1.089 | 1.421 | 0.059 |
| SnapKV-4096 | 6 | 6 | 67.56 | 25.14 | 92.70 | 91.90 (91.69-92.41) | 91.97 | 91.93 | 1.012 | 3.656 | 0.032 |
| hyb512+RQ1 t1w | 1 | 4 | 10.32 | 23.69 | 34.01 | 33.40 (33.31-33.89) | 33.42 | 33.44 | 1.080 | 1.410 | 0.059 |
| hyb512+RQ1 t1w | 4 | 4 | 41.25 | 23.69 | 64.94 | 64.30 (64.18-65.10) | 64.31 | 64.35 | 1.021 | 2.714 | 0.027 |
| hyb512+RQ1 t1w | 1 | 6 | 10.32 | 25.15 | 35.46 | 34.87 (34.78-35.11) | 34.85 | 34.85 | 1.086 | 1.387 | 0.057 |
| hyb512+RQ1 t1w | 6 | 6 | 61.98 | 25.15 | 87.13 | 86.42 (86.24-87.17) | 86.56 | 86.54 | 1.012 | 3.436 | 0.028 |
| RQ2+r128 t2w | 1 | 4 | 10.79 | 23.72 | 34.51 | 33.90 (33.83-34.11) | 33.88 | 33.90 | 1.077 | 1.429 | 0.057 |
| RQ2+r128 t2w | 4 | 4 | 43.07 | 23.72 | 66.79 | 66.14 (65.98-66.95) | 66.14 | 66.19 | 1.019 | 2.788 | 0.027 |
| RQ2+r128 t2w | 1 | 6 | 10.79 | 25.18 | 35.97 | 35.33 (35.19-35.39) | 35.30 | 35.33 | 1.084 | 1.403 | 0.059 |
| RQ2+r128 t2w | 6 | 6 | 64.56 | 25.18 | 89.74 | 89.08 (88.80-89.94) | 89.07 | 89.06 | 1.014 | 3.538 | 0.026 |
| RQ1+r128 t2w-stage1 (nested) | 1 | 4 | 10.15 | 23.68 | 33.83 | 33.29 (33.20-33.38) | 33.24 | 33.26 | 1.082 | 1.406 | 0.053 |
| RQ1+r128 t2w-stage1 (nested) | 4 | 4 | 40.60 | 23.68 | 64.28 | 63.71 (63.56-64.27) | 63.65 | 63.75 | 1.020 | 2.690 | 0.024 |
| RQ1+r128 t2w-stage1 (nested) | 1 | 6 | 10.15 | 25.15 | 35.30 | 34.71 (34.57-35.13) | 34.69 | 34.70 | 1.086 | 1.380 | 0.058 |
| RQ1+r128 t2w-stage1 (nested) | 6 | 6 | 61.06 | 25.15 | 86.20 | 85.52 (85.32-87.05) | 85.44 | 85.48 | 1.013 | 3.401 | 0.027 |

## Measured: components (gamma = 4), each co-run with the full other side

d_comm = the 66 all_reduces of one draft step only; d_gemm = draft step without comm and attention; d_attn = draft attention of all layers only; v_* the same for the verify. 'hidden' = (T_part + T_full - T_cor) / T_part.

| draft | pair | T_part iso | T_full iso | T_cor | hidden fraction of the part |
|---|---|---|---|---|---|

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

best per schedule: 
