# E7 magicvq v8 phase 2: measured e2e of 1-bit VQ KV drafts (TP=4 RTX 4090 GPUs 4-7, bf16 Llama-3.1-8B-Instruct, B16, gen 256, all-active headline)

- e2e_real.py --accounting both --soft_rep_check; codebooks via --cbx t2w / t1w; '@t2w' at L1 = stage 1 of the 2-stage t2w codebook (nested prefix, LUT kernel KST=1). Speedups vs the AR of the same process; ratios across processes use each process's own AR.

## Summary per point (all-active measured, best gamma per draft kind; rows whose outputs differ across reps (--soft_rep_check) are excluded from 'best', their aa tau is not reproducible)

| point | AR tok/s | best SnapKV | best RQ2 t2w | best 1-bit | 1-bit / RQ2 | 1-bit / SnapKV | RQ2 / SnapKV | higher but excluded (outputs differ across reps) | co-tenant |
|---|---|---|---|---|---|---|---|---|---|
| gov_report ctx 16384 B16 | 899.2 / 899.9 | 1.021 (SnapKV-4096 g3, v8e2eb) | 1.140 (RQ2+r128 t2w g4, v8e2eb) | 1.057 (hyb512+RQ1 t1w g4, v8e2ea) | **0.927** | **1.036** | **1.117** | hyb512+RQ2 t2w g6 1.187; hyb512+RQ1 t1w g6 1.253 | no |
| hotpotqa_q ctx 16384 B16 | 906.4 / 904.9 | 1.196 (SnapKV-2048 g4, v8e2eb) | 1.302 (hyb512+RQ2 t2w g6, v8e2ea) | 1.189 (hyb512+RQ1 t1w g4, v8e2ea) | **0.913** | **0.994** | **1.089** | - | no |
| pg19 ctx 32768 B16 | 591.9 / 591.7 | 1.622 (SnapKV-4096 g6, v8e2eb) | 1.623 (hyb512+RQ2 t2w g6, v8e2ea) | 1.577 (hyb512+RQ1 t1w g4, v8e2ea) | **0.972** | **0.972** | **1.000** | - | no |
| syn_quote ctx 32768 B16 | 605.8 / 605.8 | 1.470 (SnapKV-4096 g4, v8e2eb) | 1.714 (hyb512+RQ2 t2w g6, v8e2ea) | 1.467 (hyb512+RQ1 t1w g4, v8e2ea) | **0.856** | **0.998** | **1.166** | - | no |

## PROJECTED all-active speedup at 64k / 128k (B16; per-round components linear in ctx, tau = measured 32k)

Not a measurement (64k / 128k at B16 does not fit on 4x24 GB): draft step fitted per draft over its 16k and 32k points (pooled over gamma), verify fitted per gamma over all drafts, AR step over all AR baselines; contexts = mean prompt + gen/2. tau held at its measured 32k value: optimistic for SnapKV (fixed budget covers a shrinking fraction of the context), neutral for VQ. c = T_draft_step / T_AR. Rows whose outputs differ across reps are skipped.

AR step ms = 8.84 + 0.5540 * ctx/1000; draft step ms (a + b * ctx/1000): hyb512+RQ1 t1w 9.70 + 0.0458; hyb512+RQ1 t2w-stage1 (nested) 9.71 + 0.0459; hyb512+RQ2 t2w 9.83 + 0.0723; SnapKV-4096 11.37 + -0.0015; RQ1+r128 t2w-stage1 (nested) 9.40 + 0.0464; RQ2+r128 t2w 9.51 + 0.0763; verify ms: g4 14.34 + 0.5511; g6 15.80 + 0.5514

| task | draft | gamma | tau aa @32k | c @32k / 64k / 128k | S meas 32k | S proj 32k / 64k / 128k | S proj / best SnapKV proj @32k / 64k / 128k | S proj / best RQ2 t2w proj |
|---|---|---|---|---|---|---|---|---|
| pg19 | hyb512+RQ1 t1w | 4 | 4.51 | 0.415 / 0.281 / 0.193 | 1.577 | 1.574 / 2.008 / 2.456 | 0.974 / 0.873 / 0.768 | 0.973 / 0.965 / 0.958 |
| pg19 | hyb512+RQ1 t1w | 6 | 5.89 | 0.415 / 0.281 / 0.193 | 1.575 | 1.571 / 2.073 / 2.630 | 0.972 / 0.901 / 0.822 | 0.971 / 0.997 / 1.025 |
| pg19 | hyb512+RQ1 t2w-stage1 (nested) | 4 | 4.43 | 0.415 / 0.282 / 0.193 | 1.550 | 1.546 / 1.972 / 2.412 | 0.957 / 0.857 / 0.754 | 0.955 / 0.948 / 0.940 |
| pg19 | RQ1+r128 t2w-stage1 (nested) | 4 | 4.05 | 0.405 / 0.276 / 0.190 | 1.431 | 1.432 / 1.819 / 2.217 | 0.886 / 0.791 / 0.693 | 0.885 / 0.875 / 0.864 |
| pg19 | hyb512+RQ2 t2w | 6 | 6.42 | 0.452 / 0.323 / 0.237 | 1.623 | 1.618 / 2.080 / 2.565 | 1.002 / 0.904 / 0.802 | 1.000 / 1.000 / 1.000 |
| pg19 | RQ2+r128 t2w | 4 | 4.43 | 0.445 / 0.321 / 0.239 | 1.477 | 1.478 / 1.835 / 2.185 | 0.915 / 0.798 / 0.683 | 0.913 / 0.882 / 0.852 |
| pg19 | SnapKV-4096 | 6 | 6.10 | 0.419 / 0.250 / 0.137 | 1.622 | 1.616 / 2.301 / 3.200 | 1.000 / 1.000 / 1.000 | 0.998 / 1.106 / 1.248 |
| syn_quote | hyb512+RQ1 t1w | 4 | 4.26 | 0.415 / 0.281 / 0.193 | 1.467 | 1.488 / 1.898 / 2.322 | 0.996 / 0.934 / 0.870 | 0.856 / 0.849 / 0.843 |
| syn_quote | hyb512+RQ1 t1w | 6 | 5.25 | 0.415 / 0.281 / 0.193 | 1.383 | 1.402 / 1.850 / 2.347 | 0.939 / 0.911 / 0.879 | 0.807 / 0.828 / 0.852 |
| syn_quote | hyb512+RQ1 t2w-stage1 (nested) | 4 | 4.15 | 0.415 / 0.282 / 0.193 | 1.430 | 1.450 / 1.849 / 2.262 | 0.970 / 0.910 / 0.847 | 0.834 / 0.827 / 0.821 |
| syn_quote | RQ1+r128 t2w-stage1 (nested) | 4 | 3.63 | 0.405 / 0.276 / 0.190 | 1.271 | 1.284 / 1.631 / 1.988 | 0.859 / 0.803 / 0.745 | 0.738 / 0.730 / 0.721 |
| syn_quote | hyb512+RQ2 t2w | 6 | 6.90 | 0.452 / 0.323 / 0.237 | 1.714 | 1.739 / 2.235 / 2.756 | 1.164 / 1.100 / 1.032 | 1.000 / 1.000 / 1.000 |
| syn_quote | RQ2+r128 t2w | 4 | 4.86 | 0.445 / 0.321 / 0.239 | 1.607 | 1.621 / 2.014 / 2.398 | 1.085 / 0.991 / 0.898 | 0.932 / 0.901 / 0.870 |
| syn_quote | SnapKV-4096 | 4 | 4.30 | 0.419 / 0.250 / 0.137 | 1.470 | 1.494 / 2.032 / 2.669 | 1.000 / 1.000 / 1.000 | 0.859 / 0.909 / 0.969 |

## gov_report ctx 16384 B16

processes: v8e2ea: AR 899.2 tok/s (step 17.79 ms), co-tenant no (start none, end none), run_v8e2ea_gov_report_ctx16384_B16.json; v8e2eb: AR 899.9 tok/s (step 17.78 ms), co-tenant no (start none, end none), run_v8e2eb_gov_report_ctx16384_B16.json

| draft | proc | gamma | **all-active** (min-max) | static | tau aa / static | ms/round aa | draft step ms | verify ms | other ms | aa rows identical to AR | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v8e2ea | 3 | **1.020** (1.020-1.020) | 0.872 | 3.22 / 3.18 | 56.1 | 11.34 | 22.01 | 0.06 | 16/16 | no |
| hyb512+RQ2 t2w | v8e2ea | 6 | **1.187** (1.186-1.188) | 1.043 | 6.06 / 5.72 | 91.0 | 11.00 | 24.93 | 0.06 | 4/16 | yes |
| hyb512+RQ1 t1w | v8e2ea | 4 | **1.057** (1.056-1.057) | 0.831 | 3.86 / 3.81 | 65.0 | 10.43 | 23.17 | 0.06 | 3/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 6 | **1.253** (1.209-1.297) | 0.931 | 6.14 / 4.45 | 87.2 | 10.38 | 24.91 | 0.06 | 0/16 | yes |
| SnapKV-4096 | v8e2eb | 3 | **1.021** (1.021-1.021) | 0.873 | 3.22 / 3.18 | 56.0 | 11.34 | 21.94 | 0.06 | 16/16 | no |
| RQ2+r128 t2w | v8e2eb | 4 | **1.140** (1.140-1.140) | 1.050 | 4.28 / 4.28 | 66.7 | 10.74 | 23.29 | 0.44 | 2/16 | no |
| RQ1+r128 t2w-stage1 (nested) | v8e2eb | 4 | **0.905** (0.905-0.905) | 0.739 | 3.26 / 3.24 | 64.1 | 10.15 | 23.29 | 0.26 | 3/16 | no |
| hyb512+RQ1 t2w-stage1 (nested) | v8e2eb | 4 | **0.945** (0.945-0.945) | 0.805 | 3.46 / 3.43 | 65.1 | 10.43 | 23.29 | 0.06 | 2/16 | no |

## hotpotqa_q ctx 16384 B16

processes: v8e2ea: AR 906.4 tok/s (step 17.65 ms), co-tenant no (start none, end none), run_v8e2ea_hotpotqa_q_ctx16384_B16.json; v8e2eb: AR 904.9 tok/s (step 17.68 ms), co-tenant no (start none, end none), run_v8e2eb_hotpotqa_q_ctx16384_B16.json

| draft | proc | gamma | **all-active** (min-max) | static | tau aa / static | ms/round aa | draft step ms | verify ms | other ms | aa rows identical to AR | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-2048 | v8e2ea | 4 | **1.194** (1.194-1.195) | 0.644 | 4.30 / 4.05 | 63.6 | 10.10 | 23.16 | 0.05 | 7/16 | no |
| hyb512+RQ2 t2w | v8e2ea | 6 | **1.302** (1.301-1.302) | 1.216 | 6.68 / 6.69 | 90.5 | 10.98 | 24.59 | 0.05 | 7/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 4 | **1.189** (1.188-1.189) | 0.952 | 4.37 / 4.33 | 64.8 | 10.43 | 23.07 | 0.05 | 8/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 6 | **1.180** (1.179-1.181) | 0.837 | 5.80 / 5.63 | 86.8 | 10.35 | 24.59 | 0.06 | 8/16 | yes |
| SnapKV-2048 | v8e2eb | 4 | **1.196** (1.196-1.196) | 0.645 | 4.30 / 4.05 | 63.6 | 10.11 | 23.10 | 0.06 | 7/16 | no |
| RQ2+r128 t2w | v8e2eb | 4 | **1.253** (1.253-1.253) | 1.135 | 4.71 / 4.71 | 66.5 | 10.72 | 23.11 | 0.45 | 7/16 | no |
| RQ1+r128 t2w-stage1 (nested) | v8e2eb | 4 | **1.026** (1.026-1.027) | 0.956 | 3.71 / 3.70 | 64.0 | 10.15 | 23.12 | 0.25 | 8/16 | no |
| hyb512+RQ1 t2w-stage1 (nested) | v8e2eb | 4 | **1.161** (1.160-1.161) | 0.880 | 4.27 / 4.20 | 65.0 | 10.45 | 23.11 | 0.06 | 8/16 | no |

## pg19 ctx 32768 B16

processes: v8e2ea: AR 591.9 tok/s (step 27.03 ms), co-tenant no (start none, end none), run_v8e2ea_pg19_ctx32768_B16.json; v8e2eb: AR 591.7 tok/s (step 27.04 ms), co-tenant no (start none, end none), run_v8e2eb_pg19_ctx32768_B16.json

| draft | proc | gamma | **all-active** (min-max) | static | tau aa / static | ms/round aa | draft step ms | verify ms | other ms | aa rows identical to AR | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v8e2ea | 6 | **1.622** (1.622-1.622) | 1.448 | 6.10 / 6.11 | 101.6 | 11.26 | 33.98 | 0.06 | 1/16 | no |
| hyb512+RQ2 t2w | v8e2ea | 6 | **1.623** (1.623-1.623) | 1.406 | 6.42 / 6.41 | 107.0 | 12.16 | 33.99 | 0.06 | 1/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 4 | **1.577** (1.577-1.577) | 1.400 | 4.51 / 4.50 | 77.2 | 11.18 | 32.47 | 0.06 | 1/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 6 | **1.575** (1.574-1.577) | 1.317 | 5.89 / 5.91 | 101.0 | 11.16 | 33.99 | 0.06 | 1/16 | no |
| SnapKV-4096 | v8e2eb | 6 | **1.622** (1.622-1.622) | 1.447 | 6.10 / 6.11 | 101.6 | 11.28 | 33.91 | 0.06 | 1/16 | no |
| RQ2+r128 t2w | v8e2eb | 4 | **1.477** (1.477-1.477) | 1.273 | 4.43 / 4.42 | 81.0 | 12.04 | 32.46 | 0.44 | 1/16 | no |
| RQ1+r128 t2w-stage1 (nested) | v8e2eb | 4 | **1.431** (1.430-1.431) | 1.273 | 4.05 / 4.08 | 76.5 | 10.95 | 32.46 | 0.25 | 1/16 | no |
| hyb512+RQ1 t2w-stage1 (nested) | v8e2eb | 4 | **1.550** (1.550-1.552) | 1.399 | 4.43 / 4.43 | 77.2 | 11.17 | 32.45 | 0.06 | 1/16 | no |

## syn_quote ctx 32768 B16

processes: v8e2ea: AR 605.8 tok/s (step 26.41 ms), co-tenant no (start none, end none), run_v8e2ea_syn_quote_ctx32768_B16.json; v8e2eb: AR 605.8 tok/s (step 26.41 ms), co-tenant no (start none, end none), run_v8e2eb_syn_quote_ctx32768_B16.json

| draft | proc | gamma | **all-active** (min-max) | static | tau aa / static | ms/round aa | draft step ms | verify ms | other ms | aa rows identical to AR | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v8e2ea | 4 | **1.468** (1.468-1.468) | 1.118 | 4.30 / 4.22 | 77.4 | 11.38 | 31.84 | 0.05 | 16/16 | no |
| hyb512+RQ2 t2w | v8e2ea | 6 | **1.714** (1.713-1.714) | 1.628 | 6.90 / 6.90 | 106.3 | 12.18 | 33.23 | 0.04 | 15/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 4 | **1.467** (1.467-1.468) | 1.258 | 4.26 / 4.24 | 76.7 | 11.20 | 31.82 | 0.05 | 15/16 | no |
| hyb512+RQ1 t1w | v8e2ea | 6 | **1.383** (1.382-1.383) | 1.140 | 5.25 / 5.21 | 100.4 | 11.18 | 33.23 | 0.05 | 16/16 | no |
| SnapKV-4096 | v8e2eb | 4 | **1.470** (1.469-1.470) | 1.120 | 4.30 / 4.22 | 77.3 | 11.38 | 31.77 | 0.05 | 16/16 | no |
| RQ2+r128 t2w | v8e2eb | 4 | **1.607** (1.607-1.607) | 1.539 | 4.86 / 4.86 | 79.9 | 11.90 | 31.79 | 0.48 | 15/16 | no |
| RQ1+r128 t2w-stage1 (nested) | v8e2eb | 4 | **1.271** (1.271-1.272) | 1.164 | 3.63 / 3.61 | 75.4 | 10.84 | 31.77 | 0.25 | 15/16 | no |
| hyb512+RQ1 t2w-stage1 (nested) | v8e2eb | 4 | **1.430** (1.430-1.430) | 1.296 | 4.15 / 4.12 | 76.7 | 11.21 | 31.78 | 0.05 | 15/16 | no |

