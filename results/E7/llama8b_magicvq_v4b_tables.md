# E7 magicvq v4b - far-context workloads: tau and break-even vs SnapKV
tau runs ['v4', 'v4b'] (disjoint samples merged) + patch runs ['v4bh'] (extra drafts on the same samples); TP=4 timing runs ['v4b'] (synthetic weights, GPUs 4-7).
speedup = [computed from measured parts]: measured tau x measured TP=4 step times. hyb* c: measured where the timing run has a hybrid kernel record (hyb:<rq>[ctx-W]+bf16[W], no per-round encode: generated tokens stay in the bf16 part), else ESTIMATED [I]. Drafts with no step time in the timing runs have tau only. CommVQ: no TP timing.

## workloads
| task | ctx | n | n per run | skipped (<128 tok) | mean P | mean N | target task score (synthetic) |
|---|---|---|---|---|---|---|---|
| gov_report | 16384 | 36 | {'v4': 18, 'v4b': 18} | 0 | 16125 | 255 | - |
| hotpotqa_q | 16384 | 36 | {'v4': 18, 'v4b': 18} | 23 | 15945 | 202 | - |
| syn_quote | 16384 | 36 | {'v4': 18, 'v4b': 18} | 1 | 15696 | 206 | 0.99 |
| gov_report | 32768 | 13 | {'v4': 12, 'v4b': 1} | 0 | 27833 | 255 | - |
| hotpotqa_q | 32768 | 36 | {'v4': 12, 'v4b': 24} | 19 | 32757 | 220 | - |
| syn_quote | 32768 | 36 | {'v4': 12, 'v4b': 24} | 2 | 31560 | 215 | 0.99 |
target task score: syn_quote = char similarity of the target output to the true paragraph; syn_kv = fraction of the 16 codes the target output contains; syn_cont = fraction of the generated ledger records that match the true records verbatim.

## B16_ctx32768: best SnapKV vs best other draft per family (speedup, gamma) [computed from measured parts; hyb c as in the header]
| task | best SnapKV | StreamingLLM | RQ1+win | RQ2+win | RQ1 | RQ2 | KIVI+win | hybrid SnapKV+RQ0.5 | hybrid SnapKV+RQ1 | hybrid SnapKV+RQ2 | best non-SnapKV / SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gov_report | 1.268 (snapkv4096, g2) | 1.210 (evict512, g1) | - | 1.297 (vqpL2+r128, g3) | - | 1.258 (vqpL2, g3) | - | - | - | 1.287 (hyb1024pL2, g3) | 1.022 (vqpL2+r128) |
| hotpotqa_q | 1.487 (snapkv4096, g3) | 1.209 (evict512, g1) | - | 1.444 (vqpL2+r128, g3) | - | 1.299 (vqpL2, g3) | - | - | - | 1.499 (hyb512pL2, g4) | 1.008 (hyb512pL2) |
| syn_quote | 1.420 (snapkv4096, g3) | 1.174 (evict512, g1) | - | 1.546 (vqpL2+r128, g6) | - | 1.361 (vqpL2, g3) | - | - | - | 1.587 (hyb512pL2, g6) | 1.117 (hyb512pL2) |

## B8_ctx32768: best SnapKV vs best other draft per family (speedup, gamma) [computed from measured parts; hyb c as in the header]
| task | best SnapKV | StreamingLLM | RQ1+win | RQ2+win | RQ1 | RQ2 | KIVI+win | hybrid SnapKV+RQ0.5 | hybrid SnapKV+RQ1 | hybrid SnapKV+RQ2 | best non-SnapKV / SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gov_report | 1.148 (snapkv1024, g1) | 1.122 (evict512, g1) | - | 1.133 (vqpL2+r128, g1) | - | 1.122 (vqpL2, g1) | - | - | - | 1.139 (hyb1024pL2, g1) | 0.993 (hyb1024pL2) |
| hotpotqa_q | 1.291 (snapkv4096, g3) | 1.121 (evict512, g1) | - | 1.234 (vqpL2+r128, g3) | - | 1.133 (vqpL2, g1) | - | - | - | 1.271 (hyb1024pL2, g3) | 0.985 (hyb1024pL2) |
| syn_quote | 1.233 (snapkv4096, g3) | 1.088 (evict512, g1) | - | 1.287 (vqpL2+r128, g3) | - | 1.175 (vqpL2, g2) | - | - | - | 1.322 (hyb512pL2, g6) | 1.072 (hyb512pL2) |

## B16_ctx16384: best SnapKV vs best other draft per family (speedup, gamma) [computed from measured parts; hyb c as in the header]
| task | best SnapKV | StreamingLLM | RQ1+win | RQ2+win | RQ1 | RQ2 | KIVI+win | hybrid SnapKV+RQ0.5 | hybrid SnapKV+RQ1 | hybrid SnapKV+RQ2 | best non-SnapKV / SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gov_report | 1.076 (snapkv2048, g1) | 1.044 (evict512, g1) | - | 1.079 (vqpL2+r128, g1) | - | 1.075 (vqpL2, g1) | - | - | - | 1.093 (hyb512pL2, g1) | 1.015 (hyb512pL2) |
| hotpotqa_q | 1.122 (snapkv2048, g2) | 1.031 (evict512, g1) | - | 1.187 (vqpL2+r128, g3) | - | 1.122 (vqpL2, g2) | - | - | - | 1.217 (hyb512pL2, g3) | 1.085 (hyb512pL2) |
| syn_quote | 1.208 (snapkv4096, g3) | 1.019 (evict512, g1) | - | 1.268 (vqpL2+r128, g6) | - | 1.207 (vqpL2, g3) | - | - | - | 1.288 (hyb512pL2, g6) | 1.066 (hyb512pL2) |

## B8_ctx16384: best SnapKV vs best other draft per family (speedup, gamma) [computed from measured parts; hyb c as in the header]
| task | best SnapKV | StreamingLLM | RQ1+win | RQ2+win | RQ1 | RQ2 | KIVI+win | hybrid SnapKV+RQ0.5 | hybrid SnapKV+RQ1 | hybrid SnapKV+RQ2 | best non-SnapKV / SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|

## drafts that beat the best SnapKV (computed speedup ratio > 1), with paired-bootstrap 90 % CI over prompts
c from the timing runs above; [I] = hybrid with c estimated (no kernel record). StreamingLLM rows with the same c and tau as SnapKV (ties) are omitted. CI: paired prompt bootstrap (2000 resamples), each draft at its own full-data best gamma.
| point | task | draft | c | speedup (g) | best SnapKV (g) | ratio | 90 % CI |
|---|---|---|---|---|---|---|---|
| B16_ctx32768 | gov_report | vqpL2+r128 | 0.453 | 1.297 (g3) | 1.268 (snapkv4096, g2) | 1.022 | [1.007, 1.039] |
| B16_ctx32768 | gov_report | hyb1024pL2 | 0.472 | 1.287 (g3) | 1.268 (snapkv4096, g2) | 1.015 | [0.998, 1.033] |
| B16_ctx32768 | gov_report | hyb512pL2 | 0.458 | 1.286 (g3) | 1.268 (snapkv4096, g2) | 1.014 | [1.000, 1.029] |
| B16_ctx32768 | gov_report | hyb2048pL2 | 0.495 | 1.277 (g3) | 1.268 (snapkv4096, g2) | 1.007 | [0.990, 1.025] |
| B16_ctx32768 | hotpotqa_q | hyb512pL2 | 0.458 | 1.499 (g4) | 1.487 (snapkv4096, g3) | 1.008 | [0.983, 1.032] |
| B16_ctx32768 | hotpotqa_q | hyb1024pL2 | 0.472 | 1.496 (g6) | 1.487 (snapkv4096, g3) | 1.006 | [0.979, 1.031] |
| B16_ctx32768 | syn_quote | hyb512pL2 | 0.458 | 1.587 (g6) | 1.420 (snapkv4096, g3) | 1.117 | [1.081, 1.152] |
| B16_ctx32768 | syn_quote | hyb1024pL2 | 0.472 | 1.573 (g6) | 1.420 (snapkv4096, g3) | 1.108 | [1.071, 1.142] |
| B16_ctx32768 | syn_quote | hyb2048pL2 | 0.495 | 1.550 (g6) | 1.420 (snapkv4096, g3) | 1.091 | [1.059, 1.122] |
| B16_ctx32768 | syn_quote | vqpL2+r128 | 0.453 | 1.546 (g6) | 1.420 (snapkv4096, g3) | 1.089 | [1.050, 1.127] |
| B16_ctx32768 | syn_quote | vqpL2+r512 | 0.467 | 1.514 (g6) | 1.420 (snapkv4096, g3) | 1.066 | [1.027, 1.105] |
| B8_ctx32768 | syn_quote | hyb512pL2 | 0.583 | 1.322 (g6) | 1.233 (snapkv4096, g3) | 1.072 | [1.038, 1.106] |
| B8_ctx32768 | syn_quote | hyb1024pL2 | 0.593 | 1.320 (g6) | 1.233 (snapkv4096, g3) | 1.071 | [1.036, 1.104] |
| B8_ctx32768 | syn_quote | hyb2048pL2 | 0.610 | 1.317 (g6) | 1.233 (snapkv4096, g3) | 1.068 | [1.037, 1.099] |
| B8_ctx32768 | syn_quote | vqpL2+r128 | 0.583 | 1.287 (g3) | 1.233 (snapkv4096, g3) | 1.044 | [1.009, 1.078] |
| B8_ctx32768 | syn_quote | vqpL2+r512 | 0.588 | 1.280 (g3) | 1.233 (snapkv4096, g3) | 1.038 | [1.004, 1.072] |
| B16_ctx16384 | gov_report | hyb512pL2 | 0.609 | 1.093 (g1) | 1.076 (snapkv2048, g1) | 1.015 | [1.007, 1.024] |
| B16_ctx16384 | gov_report | hyb1024pL2 | 0.627 | 1.092 (g2) | 1.076 (snapkv2048, g1) | 1.015 | [1.006, 1.023] |
| B16_ctx16384 | gov_report | hyb2048pL2 | 0.660 | 1.080 (g2) | 1.076 (snapkv2048, g1) | 1.004 | [0.995, 1.012] |
| B16_ctx16384 | gov_report | vqpL2+r128 | 0.601 | 1.079 (g1) | 1.076 (snapkv2048, g1) | 1.002 | [0.994, 1.011] |
| B16_ctx16384 | hotpotqa_q | hyb512pL2 | 0.609 | 1.217 (g3) | 1.122 (snapkv2048, g2) | 1.085 | [1.049, 1.121] |
| B16_ctx16384 | hotpotqa_q | hyb1024pL2 | 0.627 | 1.215 (g3) | 1.122 (snapkv2048, g2) | 1.082 | [1.048, 1.115] |
| B16_ctx16384 | hotpotqa_q | hyb2048pL2 | 0.660 | 1.190 (g3) | 1.122 (snapkv2048, g2) | 1.060 | [1.028, 1.091] |
| B16_ctx16384 | hotpotqa_q | vqpL2+r128 | 0.601 | 1.187 (g3) | 1.122 (snapkv2048, g2) | 1.057 | [1.015, 1.100] |
| B16_ctx16384 | hotpotqa_q | vqpL2+r512 | 0.615 | 1.171 (g3) | 1.122 (snapkv2048, g2) | 1.043 | [1.003, 1.084] |
| B16_ctx16384 | syn_quote | hyb512pL2 | 0.609 | 1.288 (g6) | 1.208 (snapkv4096, g3) | 1.066 | [1.047, 1.087] |
| B16_ctx16384 | syn_quote | hyb1024pL2 | 0.627 | 1.281 (g6) | 1.208 (snapkv4096, g3) | 1.060 | [1.042, 1.080] |
| B16_ctx16384 | syn_quote | vqpL2+r128 | 0.601 | 1.268 (g6) | 1.208 (snapkv4096, g3) | 1.049 | [1.026, 1.073] |
| B16_ctx16384 | syn_quote | vqpL2+r512 | 0.615 | 1.250 (g4) | 1.208 (snapkv4096, g3) | 1.035 | [1.014, 1.057] |
| B16_ctx16384 | syn_quote | hyb2048pL2 | 0.660 | 1.248 (g6) | 1.208 (snapkv4096, g3) | 1.033 | [1.015, 1.052] |

## B16_ctx32768: tau ratio vs SnapKV (highest tau among budgets with c <= c_X) and break-even ratio, gamma 3 / 4 / 6
cell = ratio / BE (pass if ratio >= BE); * = passes at that gamma
| task | vqpL2 | vqpL2+r128 | vqpL2+r512 | hyb512pL2 | hyb1024pL2 | hyb2048pL2 |
|---|---|---|---|---|---|---|
| gov_report | 1.04:1.03* / 1.02:1.04 / 1.06:1.04* | 1.08:1.04* / 1.06:1.05* / 1.08:1.06* | 1.07:1.06* / 1.05:1.07 / 1.08:1.08 | 1.07:1.05* / 1.08:1.05* / 1.09:1.06* | 1.09:1.06* / 1.10:1.07* / 1.12:1.08* | 1.11:1.09* / 1.13:1.10* / 1.17:1.12* |
| hotpotqa_q | 0.90:1.03 / 0.89:1.04 / 0.82:1.04 | 1.01:1.04 / 1.02:1.05 / 1.02:1.06 | 1.01:1.06 / 1.03:1.07 / 1.02:1.08 | 1.04:1.05 / 1.06:1.05* / 1.09:1.06* | 1.05:1.06 / 1.08:1.07* / 1.11:1.08* | - |
| syn_quote | 0.99:1.03 / 0.99:1.04 / 0.97:1.04 | 1.11:1.04* / 1.15:1.05* / 1.23:1.06* | 1.11:1.06* / 1.15:1.07* / 1.23:1.08* | 1.12:1.05* / 1.17:1.05* / 1.27:1.06* | 1.13:1.06* / 1.19:1.07* / 1.28:1.08* | 1.13:1.09* / 1.19:1.10* / 1.31:1.12* |

## B16_ctx16384: tau ratio vs SnapKV (highest tau among budgets with c <= c_X) and break-even ratio, gamma 3 / 4 / 6
cell = ratio / BE (pass if ratio >= BE); * = passes at that gamma
| task | vqpL2 | vqpL2+r128 | vqpL2+r512 | hyb512pL2 | hyb1024pL2 | hyb2048pL2 |
|---|---|---|---|---|---|---|
| gov_report | 1.08:1.04* / 1.10:1.04* / 1.14:1.05* | 1.11:1.05* / 1.15:1.06* / 1.20:1.06* | 1.12:1.07* / 1.15:1.07* / 1.20:1.08* | 1.14:1.06* / 1.18:1.06* / 1.26:1.07* | 1.05:1.01* / 1.06:1.01* / 1.09:1.01* | 1.08:1.04* / 1.10:1.04* / 1.15:1.05* |
| hotpotqa_q | 1.06:1.04* / 1.08:1.04* / 1.12:1.05* | 1.14:1.05* / 1.19:1.06* / 1.29:1.06* | 1.14:1.07* / 1.19:1.07* / 1.29:1.08* | 1.17:1.06* / 1.23:1.06* / 1.38:1.07* | 1.10:1.01* / 1.14:1.01* / 1.22:1.01* | 1.11:1.04* / 1.15:1.04* / 1.25:1.05* |
| syn_quote | 1.05:1.04* / 1.06:1.04* / 1.10:1.05* | 1.11:1.05* / 1.15:1.06* / 1.25:1.06* | 1.11:1.07* / 1.15:1.07* / 1.25:1.08* | 1.11:1.06* / 1.16:1.06* / 1.27:1.07* | 1.04:1.01* / 1.06:1.01* / 1.10:1.01* | 1.05:1.04* / 1.07:1.04* / 1.11:1.05* |

## B8_ctx32768: tau ratio vs SnapKV (highest tau among budgets with c <= c_X) and break-even ratio, gamma 3 / 4 / 6
cell = ratio / BE (pass if ratio >= BE); * = passes at that gamma
| task | vqpL2 | vqpL2+r128 | vqpL2+r512 | hyb512pL2 | hyb1024pL2 | hyb2048pL2 |
|---|---|---|---|---|---|---|
| gov_report | 1.04:1.05 / 1.02:1.05 / 1.06:1.06 | 1.08:1.06* / 1.06:1.07 / 1.08:1.07* | 1.07:1.07* / 1.05:1.07 / 1.08:1.08 | 1.07:1.06* / 1.08:1.06* / 1.09:1.07* | 1.09:1.07* / 1.10:1.07* / 1.12:1.08* | 1.11:1.09* / 1.13:1.09* / 1.17:1.11* |
| hotpotqa_q | 0.90:1.05 / 0.89:1.05 / 0.82:1.06 | 1.01:1.06 / 1.02:1.07 / 1.02:1.07 | 1.01:1.07 / 1.03:1.07 / 1.02:1.08 | 1.04:1.06 / 1.06:1.06* / 1.09:1.07* | 1.05:1.07 / 1.08:1.07* / 1.11:1.08* | - |
| syn_quote | 0.99:1.05 / 0.99:1.05 / 0.97:1.06 | 1.11:1.06* / 1.15:1.07* / 1.23:1.07* | 1.11:1.07* / 1.15:1.07* / 1.23:1.08* | 1.12:1.06* / 1.17:1.06* / 1.27:1.07* | 1.13:1.07* / 1.19:1.07* / 1.28:1.08* | 1.13:1.09* / 1.19:1.09* / 1.31:1.11* |

### tau, gov_report ctx 16384 (n=36, mean P 16125); c and speedup at B16_ctx16384
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0312 | 0.750 | 1.73 | 2.23 | 2.59 | 2.82 | 3.13 | 0.507 | 1.063 (g1) | 0.988 |
| evict512 | StreamingLLM | 0.0312 | 0.716 | 1.70 | 2.16 | 2.48 | 2.68 | 2.91 | 0.507 | 1.044 (g1) | 0.969 |
| snapkv1024 | SnapKV | 0.0625 | 0.784 | 1.76 | 2.32 | 2.72 | 3.01 | 3.38 | 0.529 | 1.065 (g1) | 0.990 |
| evict1024 | StreamingLLM | 0.0625 | 0.725 | 1.70 | 2.17 | 2.50 | 2.72 | 2.96 | 0.529 | 1.032 (g1) | 0.958 |
| snapkv2048 | SnapKV | 0.1250 | 0.825 | 1.80 | 2.42 | 2.90 | 3.25 | 3.76 | 0.555 | 1.076 (g1) | 1.000 |
| evict2048 | StreamingLLM | 0.1250 | 0.737 | 1.71 | 2.21 | 2.55 | 2.78 | 3.07 | 0.555 | 1.023 (g1) | 0.950 |
| vqpL2 | RQ2 | 0.1278 | 0.849 | 1.85 | 2.54 | 3.13 | 3.57 | 4.30 | 0.590 | 1.075 (g1) | 0.998 |
| vqpL2+r128 | RQ2+win | 0.1346 | 0.874 | 1.87 | 2.61 | 3.22 | 3.72 | 4.52 | 0.601 | 1.079 (g1) | 1.002 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1543 | 0.890 | 1.89 | 2.65 | 3.30 | 3.83 | 4.72 | 0.609 | 1.093 (g1) | 1.015 |
| vqpL2+r512 | RQ2+win | 0.1552 | 0.876 | 1.87 | 2.62 | 3.23 | 3.73 | 4.52 | 0.615 | 1.072 (g1) | 0.996 |
| snapkv4096 | SnapKV | 0.2500 | 0.882 | 1.87 | 2.59 | 3.19 | 3.69 | 4.47 | 0.621 | 1.073 (g1) | 0.997 |
| evict4096 | StreamingLLM | 0.2500 | 0.753 | 1.73 | 2.24 | 2.60 | 2.85 | 3.16 | 0.621 | 0.991 (g1) | 0.921 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1816 | 0.899 | 1.89 | 2.69 | 3.37 | 3.91 | 4.86 | 0.627 | 1.092 (g2) | 1.015 |
| hyb2048pL2 | hybrid SnapKV+RQ2 | 0.2363 | 0.916 | 1.91 | 2.73 | 3.43 | 4.08 | 5.13 | 0.660 | 1.080 (g2) | 1.004 |
| vqpL1+r512 | RQ1+win | 0.0937 | 0.752 | 1.73 | 2.24 | 2.60 | 2.84 | 3.11 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3185 | 0.993 | 1.99 | 2.96 | 3.94 | 4.90 | 6.79 | - | - | - |

### tau, hotpotqa_q ctx 16384 (n=36, mean P 15945); c and speedup at B16_ctx16384
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0312 | 0.795 | 1.77 | 2.30 | 2.67 | 2.94 | 3.27 | 0.507 | 1.085 (g1) | 0.966 |
| evict512 | StreamingLLM | 0.0312 | 0.702 | 1.68 | 2.08 | 2.32 | 2.47 | 2.66 | 0.507 | 1.031 (g1) | 0.918 |
| snapkv1024 | SnapKV | 0.0625 | 0.842 | 1.81 | 2.44 | 2.89 | 3.24 | 3.71 | 0.529 | 1.100 (g1) | 0.980 |
| evict1024 | StreamingLLM | 0.0625 | 0.704 | 1.68 | 2.09 | 2.33 | 2.49 | 2.67 | 0.529 | 1.018 (g1) | 0.907 |
| snapkv2048 | SnapKV | 0.1250 | 0.894 | 1.88 | 2.60 | 3.19 | 3.67 | 4.39 | 0.555 | 1.122 (g2) | 1.000 |
| evict2048 | StreamingLLM | 0.1250 | 0.756 | 1.72 | 2.20 | 2.51 | 2.72 | 3.00 | 0.555 | 1.029 (g1) | 0.917 |
| vqpL2 | RQ2 | 0.1278 | 0.908 | 1.90 | 2.70 | 3.38 | 3.96 | 4.92 | 0.590 | 1.122 (g2) | 1.000 |
| vqpL2+r128 | RQ2+win | 0.1346 | 0.954 | 1.94 | 2.82 | 3.64 | 4.38 | 5.66 | 0.601 | 1.187 (g3) | 1.057 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1543 | 0.968 | 1.96 | 2.87 | 3.75 | 4.53 | 6.03 | 0.609 | 1.217 (g3) | 1.085 |
| vqpL2+r512 | RQ2+win | 0.1552 | 0.955 | 1.94 | 2.82 | 3.64 | 4.37 | 5.65 | 0.615 | 1.171 (g3) | 1.043 |
| snapkv4096 | SnapKV | 0.2500 | 0.935 | 1.92 | 2.74 | 3.45 | 4.07 | 5.08 | 0.621 | 1.119 (g2) | 0.997 |
| evict4096 | StreamingLLM | 0.2500 | 0.786 | 1.76 | 2.28 | 2.65 | 2.90 | 3.24 | 0.621 | 1.009 (g1) | 0.898 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1816 | 0.976 | 1.97 | 2.90 | 3.80 | 4.64 | 6.21 | 0.627 | 1.215 (g3) | 1.082 |
| hyb2048pL2 | hybrid SnapKV+RQ2 | 0.2363 | 0.982 | 1.98 | 2.91 | 3.84 | 4.69 | 6.36 | 0.660 | 1.190 (g3) | 1.060 |
| vqpL1+r512 | RQ1+win | 0.0937 | 0.775 | 1.75 | 2.24 | 2.56 | 2.76 | 3.00 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3185 | 0.997 | 1.99 | 2.98 | 3.96 | 4.93 | 6.82 | - | - | - |

### tau, syn_quote ctx 16384 (n=36, mean P 15696); c and speedup at B16_ctx16384
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0312 | 0.840 | 1.82 | 2.42 | 2.81 | 3.12 | 3.48 | 0.507 | 1.120 (g1) | 0.927 |
| evict512 | StreamingLLM | 0.0312 | 0.665 | 1.66 | 1.98 | 2.11 | 2.19 | 2.24 | 0.507 | 1.019 (g1) | 0.844 |
| snapkv1024 | SnapKV | 0.0625 | 0.899 | 1.89 | 2.61 | 3.17 | 3.62 | 4.28 | 0.529 | 1.150 (g2) | 0.952 |
| evict1024 | StreamingLLM | 0.0625 | 0.665 | 1.66 | 1.98 | 2.11 | 2.19 | 2.25 | 0.529 | 1.007 (g1) | 0.833 |
| snapkv2048 | SnapKV | 0.1250 | 0.943 | 1.93 | 2.77 | 3.50 | 4.13 | 5.15 | 0.555 | 1.201 (g3) | 0.993 |
| evict2048 | StreamingLLM | 0.1250 | 0.673 | 1.67 | 2.00 | 2.14 | 2.22 | 2.29 | 0.555 | 0.998 (g1) | 0.826 |
| vqpL2 | RQ2 | 0.1278 | 0.959 | 1.95 | 2.83 | 3.66 | 4.38 | 5.67 | 0.590 | 1.207 (g3) | 0.999 |
| vqpL2+r128 | RQ2+win | 0.1346 | 0.990 | 1.99 | 2.94 | 3.87 | 4.74 | 6.41 | 0.601 | 1.268 (g6) | 1.049 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1543 | 0.990 | 1.99 | 2.95 | 3.90 | 4.81 | 6.56 | 0.609 | 1.288 (g6) | 1.066 |
| vqpL2+r512 | RQ2+win | 0.1552 | 0.990 | 1.99 | 2.95 | 3.87 | 4.76 | 6.43 | 0.615 | 1.250 (g4) | 1.035 |
| snapkv4096 | SnapKV | 0.2500 | 0.976 | 1.97 | 2.89 | 3.76 | 4.58 | 6.04 | 0.621 | 1.208 (g3) | 1.000 |
| evict4096 | StreamingLLM | 0.2500 | 0.670 | 1.67 | 2.00 | 2.14 | 2.23 | 2.29 | 0.621 | 0.961 (g1) | 0.795 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1816 | 0.993 | 1.99 | 2.96 | 3.91 | 4.84 | 6.66 | 0.627 | 1.281 (g6) | 1.060 |
| hyb2048pL2 | hybrid SnapKV+RQ2 | 0.2363 | 0.994 | 1.99 | 2.97 | 3.94 | 4.88 | 6.73 | 0.660 | 1.248 (g6) | 1.033 |
| vqpL1+r512 | RQ1+win | 0.0937 | 0.793 | 1.78 | 2.24 | 2.48 | 2.63 | 2.72 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3185 | 1.000 | 1.99 | 2.99 | 3.97 | 4.97 | 6.89 | - | - | - |

### tau, gov_report ctx 32768 (n=13, mean P 27833); c and speedup at B16_ctx32768
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0156 | 0.754 | 1.75 | 2.25 | 2.60 | 2.81 | 3.11 | 0.341 | 1.235 (g2) | 0.974 |
| evict512 | StreamingLLM | 0.0156 | 0.733 | 1.73 | 2.20 | 2.52 | 2.74 | 2.97 | 0.341 | 1.210 (g1) | 0.954 |
| snapkv1024 | SnapKV | 0.0312 | 0.780 | 1.78 | 2.31 | 2.73 | 2.96 | 3.32 | 0.362 | 1.238 (g2) | 0.976 |
| evict1024 | StreamingLLM | 0.0312 | 0.739 | 1.73 | 2.22 | 2.54 | 2.76 | 3.03 | 0.362 | 1.198 (g1) | 0.945 |
| snapkv2048 | SnapKV | 0.0625 | 0.811 | 1.79 | 2.38 | 2.84 | 3.15 | 3.65 | 0.373 | 1.261 (g2) | 0.994 |
| evict2048 | StreamingLLM | 0.0625 | 0.745 | 1.74 | 2.23 | 2.58 | 2.78 | 3.04 | 0.373 | 1.197 (g1) | 0.943 |
| snapkv4096 | SnapKV | 0.1250 | 0.856 | 1.84 | 2.51 | 3.04 | 3.49 | 4.16 | 0.420 | 1.268 (g2) | 1.000 |
| evict4096 | StreamingLLM | 0.1250 | 0.756 | 1.75 | 2.26 | 2.62 | 2.88 | 3.12 | 0.420 | 1.164 (g1) | 0.917 |
| vqpL2 | RQ2 | 0.1264 | 0.851 | 1.85 | 2.56 | 3.15 | 3.55 | 4.40 | 0.444 | 1.258 (g3) | 0.992 |
| vqpL2+r128 | RQ2+win | 0.1298 | 0.878 | 1.88 | 2.62 | 3.28 | 3.70 | 4.50 | 0.453 | 1.297 (g3) | 1.022 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1396 | 0.884 | 1.88 | 2.63 | 3.27 | 3.77 | 4.52 | 0.458 | 1.286 (g3) | 1.014 |
| vqpL2+r512 | RQ2+win | 0.1401 | 0.879 | 1.88 | 2.62 | 3.25 | 3.66 | 4.47 | 0.467 | 1.263 (g3) | 0.996 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1533 | 0.892 | 1.90 | 2.65 | 3.32 | 3.84 | 4.66 | 0.472 | 1.287 (g3) | 1.015 |
| hyb2048pL2 | hybrid SnapKV+RQ2 | 0.1807 | 0.906 | 1.91 | 2.68 | 3.38 | 3.93 | 4.88 | 0.495 | 1.277 (g3) | 1.007 |
| vqpL1+r512 | RQ1+win | 0.0781 | 0.751 | 1.75 | 2.24 | 2.56 | 2.77 | 3.04 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3155 | 0.990 | 1.99 | 2.96 | 3.90 | 4.82 | 6.71 | - | - | - |

### tau, hotpotqa_q ctx 32768 (n=36, mean P 32757); c and speedup at B16_ctx32768
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0156 | 0.829 | 1.80 | 2.41 | 2.84 | 3.17 | 3.62 | 0.341 | 1.323 (g2) | 0.889 |
| evict512 | StreamingLLM | 0.0156 | 0.752 | 1.72 | 2.20 | 2.50 | 2.69 | 2.97 | 0.341 | 1.209 (g1) | 0.813 |
| snapkv1024 | SnapKV | 0.0312 | 0.875 | 1.86 | 2.55 | 3.08 | 3.50 | 4.11 | 0.362 | 1.368 (g2) | 0.920 |
| evict1024 | StreamingLLM | 0.0312 | 0.753 | 1.73 | 2.20 | 2.49 | 2.68 | 2.96 | 0.362 | 1.192 (g1) | 0.801 |
| snapkv2048 | SnapKV | 0.0625 | 0.910 | 1.89 | 2.67 | 3.32 | 3.84 | 4.72 | 0.373 | 1.455 (g3) | 0.978 |
| evict2048 | StreamingLLM | 0.0625 | 0.769 | 1.75 | 2.25 | 2.56 | 2.79 | 3.09 | 0.373 | 1.197 (g1) | 0.805 |
| snapkv4096 | SnapKV | 0.1250 | 0.950 | 1.94 | 2.80 | 3.61 | 4.30 | 5.56 | 0.420 | 1.487 (g3) | 1.000 |
| evict4096 | StreamingLLM | 0.1250 | 0.780 | 1.76 | 2.27 | 2.61 | 2.84 | 3.16 | 0.420 | 1.168 (g1) | 0.785 |
| vqpL2 | RQ2 | 0.1264 | 0.883 | 1.87 | 2.63 | 3.25 | 3.81 | 4.56 | 0.444 | 1.299 (g3) | 0.873 |
| vqpL2+r128 | RQ2+win | 0.1298 | 0.958 | 1.95 | 2.83 | 3.65 | 4.38 | 5.66 | 0.453 | 1.444 (g3) | 0.971 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1396 | 0.972 | 1.96 | 2.89 | 3.74 | 4.58 | 6.03 | 0.458 | 1.499 (g4) | 1.008 |
| vqpL2+r512 | RQ2+win | 0.1401 | 0.960 | 1.95 | 2.83 | 3.66 | 4.41 | 5.68 | 0.467 | 1.423 (g4) | 0.957 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1533 | 0.977 | 1.97 | 2.91 | 3.79 | 4.63 | 6.16 | 0.472 | 1.496 (g6) | 1.006 |
| vqpL1+r512 | RQ1+win | 0.0781 | 0.806 | 1.79 | 2.33 | 2.72 | 2.97 | 3.31 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3155 | 0.996 | 1.99 | 2.98 | 3.95 | 4.94 | 6.81 | - | - | - |

### tau, syn_quote ctx 32768 (n=36, mean P 31560); c and speedup at B16_ctx32768
| draft | family | KV frac | alpha1 | tau g1 | g2 | g3 | g4 | g6 | c | speedup (g) | vs best SnapKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| snapkv512 | SnapKV | 0.0156 | 0.808 | 1.79 | 2.33 | 2.69 | 2.91 | 3.20 | 0.341 | 1.280 (g2) | 0.901 |
| evict512 | StreamingLLM | 0.0156 | 0.677 | 1.67 | 2.00 | 2.17 | 2.24 | 2.30 | 0.341 | 1.174 (g1) | 0.826 |
| snapkv1024 | SnapKV | 0.0312 | 0.852 | 1.84 | 2.46 | 2.91 | 3.24 | 3.70 | 0.362 | 1.322 (g2) | 0.931 |
| evict1024 | StreamingLLM | 0.0312 | 0.677 | 1.68 | 2.00 | 2.17 | 2.26 | 2.32 | 0.362 | 1.158 (g1) | 0.815 |
| snapkv2048 | SnapKV | 0.0625 | 0.898 | 1.89 | 2.61 | 3.20 | 3.68 | 4.37 | 0.373 | 1.402 (g3) | 0.987 |
| evict2048 | StreamingLLM | 0.0625 | 0.679 | 1.67 | 2.01 | 2.18 | 2.25 | 2.31 | 0.373 | 1.148 (g1) | 0.808 |
| snapkv4096 | SnapKV | 0.1250 | 0.936 | 1.92 | 2.74 | 3.44 | 4.05 | 5.05 | 0.420 | 1.420 (g3) | 1.000 |
| evict4096 | StreamingLLM | 0.1250 | 0.683 | 1.68 | 2.02 | 2.19 | 2.27 | 2.33 | 0.420 | 1.113 (g1) | 0.784 |
| vqpL2 | RQ2 | 0.1264 | 0.919 | 1.91 | 2.73 | 3.41 | 4.00 | 4.92 | 0.444 | 1.361 (g3) | 0.958 |
| vqpL2+r128 | RQ2+win | 0.1298 | 0.981 | 1.98 | 2.92 | 3.81 | 4.67 | 6.22 | 0.453 | 1.546 (g6) | 1.089 |
| hyb512pL2 | hybrid SnapKV+RQ2 | 0.1396 | 0.987 | 1.98 | 2.94 | 3.86 | 4.75 | 6.41 | 0.458 | 1.587 (g6) | 1.117 |
| vqpL2+r512 | RQ2+win | 0.1401 | 0.981 | 1.98 | 2.92 | 3.81 | 4.66 | 6.22 | 0.467 | 1.514 (g6) | 1.066 |
| hyb1024pL2 | hybrid SnapKV+RQ2 | 0.1533 | 0.989 | 1.98 | 2.95 | 3.89 | 4.81 | 6.48 | 0.472 | 1.573 (g6) | 1.108 |
| hyb2048pL2 | hybrid SnapKV+RQ2 | 0.1807 | 0.991 | 1.99 | 2.96 | 3.91 | 4.83 | 6.60 | 0.495 | 1.550 (g6) | 1.091 |
| vqpL1+r512 | RQ1+win | 0.0781 | 0.765 | 1.76 | 2.18 | 2.42 | 2.54 | 2.64 | - | - | - |
| kivi4g32+r128 | KIVI+win | 0.3155 | 1.000 | 1.99 | 2.99 | 3.98 | 4.97 | 6.91 | - | - | - |

## e2e loop check (e2e_loop.py, TP=4 GPUs 4-7, synthetic weights, acceptance replayed from the syn_quote 32k v4b match traces (24 new samples), fixed cache length; median of 3 reps)
| B | draft (tau draft) | gamma | spec tok/s | AR tok/s | e2e speedup | tau replayed |
|---|---|---|---|---|---|---|
| 16 | evict4096 (snapkv4096) | 3 | 912 | 600 | 1.520 | 3.63 |
| 16 | rq2+r128 (vqpL2+r128) | 6 | 953 | 601 | 1.587 | 6.32 |
| 8 | evict4096 (snapkv4096) | 3 | 634 | 482 | 1.315 | 3.58 |
| 8 | hyb512rq2 (hyb512pL2) | 6 | 666 | 482 | 1.382 | 6.62 |
