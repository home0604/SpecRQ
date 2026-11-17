(* = tau from ctx 32k, extrapolated)

## TP=4 vs PP=4 step times (target, M=1), ms
| B | ctx | TP4 T_AR | PP4 T_AR | TP4 evict512 | PP4 evict512 |
|---|---|---|---|---|---|
| 1 | 32768 | 7.58 | 21.87 | 6.28 | 17.18 |
| 16 | 16384 | 17.90 | 54.24 | 9.20 | 19.36 |

## best speedup per family, tp_pg19 [computed from measured parts]
| B | ctx | T_AR ms | evict | snapkv | kivi | rq_post | rq_post+win | gsrq | gsrq+win | commvq | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 6.8 | 0.97 (evict4096, g1) | 0.97 (snapkv4096, g1) | 0.87 (kivi4+r32, g1) | 0.84 (rq2, g1) | 0.92 (rq2+r512, g1) | 0.66 (gs1, g1) | 0.78 (gs1+r512, g1) | - | snapkv4096 0.97 |
| 1 | 16384 | 7.1 | 0.98 (evict4096, g1) | 1.00 (snapkv4096, g1) | 0.88 (kivi4+r128, g1) | 0.88 (rq2, g1) | 0.94 (rq2+r512, g1) | 0.67 (gs1, g1) | 0.78 (gs05+r512, g1) | - | snapkv4096 1.00 |
| 1 | 32768 | 7.6 | 1.02 (evict1024, g1) | 1.02 (snapkv4096, g1) | 0.90 (kivi4+r32, g1) | 0.87 (rq2, g1) | 0.95 (rq2+r512, g1) | 0.64 (gs1, g1) | 0.75 (gs05+r512, g1) | - | snapkv4096 1.02 |
| 1 | 65536* | 8.7 | 1.10 (evict4096, g3) | 1.11 (snapkv4096, g3) | 0.94 (kivi4+r128, g3) | 0.89 (rq2, g1) | 0.98 (rq1+r512, g1) | 0.64 (gs1, g1) | 0.76 (gs05+r512, g1) | - | snapkv4096 1.11 |
| 8 | 8192 | 10.2 | 1.01 (evict1024, g1) | 1.03 (snapkv1024, g1) | 0.91 (kivi4+r128, g1) | 0.89 (rq2, g1) | 0.96 (rq2+r512, g1) | 0.62 (gs1, g1) | 0.74 (gs05+r512, g1) | - | snapkv1024 1.03 |
| 8 | 16384 | 12.3 | 1.10 (evict1024, g1) | 1.11 (snapkv2048, g3) | 0.96 (kivi4+r128, g1) | 0.98 (rq2, g1) | 1.04 (rq2+r512, g1) | 0.65 (gs1, g1) | 0.76 (gs05+r512, g1) | - | snapkv2048 1.11 |
| 8 | 32768 | 16.7 | 1.34 (evict2048, g3) | 1.34 (snapkv2048, g3) | 1.09 (kivi4+r128, g6) | 1.05 (rq2, g1) | 1.19 (rq1+r512, g3) | 0.65 (gs1, g1) | 0.78 (gs05+r512, g1) | - | snapkv2048 1.34 |
| 8 | 65536* | 25.6 | 1.80 (evict4096, g6) | 1.82 (snapkv4096, g6) | 1.26 (kivi4+r128, g6) | 1.18 (rq2, g1) | 1.47 (rq1+r512, g3) | 0.67 (gs1, g1) | 0.80 (gs05+r512, g1) | - | snapkv4096 1.82 |
| 16 | 8192 | 13.5 | 1.02 (evict1024, g1) | 1.03 (snapkv1024, g1) | 0.92 (kivi4+r128, g3) | 0.91 (rq2, g1) | 0.97 (rq2+r512, g1) | 0.62 (gs1, g1) | 0.73 (gs05+r512, g1) | - | snapkv1024 1.03 |
| 16 | 16384 | 17.9 | 1.21 (evict1024, g3) | 1.25 (snapkv2048, g4) | 1.04 (kivi4+r128, g6) | 1.03 (rq2, g1) | 1.14 (rq2+r512, g3) | 0.66 (gs1, g1) | 0.76 (gs05+r512, g1) | - | snapkv2048 1.25 |
| 16 | 32768 | 26.8 | 1.61 (evict1024, g4) | 1.65 (snapkv1024, g6) | 1.24 (kivi4+r128, g6) | 1.14 (rq2, g1) | 1.40 (rq1+r512, g3) | 0.67 (gs1, g1) | 0.79 (gs05+r512, g1) | - | snapkv1024 1.65 |
| 32 | 8192 | 20.2 | 1.12 (evict1024, g2) | 1.15 (snapkv1024, g2) | 1.01 (kivi4+r128, g2) | 1.00 (rq2, g1) | 1.07 (rq2+r128, g1) | 0.67 (gs1, g1) | 0.77 (gs05+r512, g1) | - | snapkv1024 1.15 |
| 32 | 16384 | 29.0 | 1.42 (evict512, g3) | 1.46 (snapkv512, g3) | 1.16 (kivi4+r128, g3) | 1.15 (rq2, g1) | 1.30 (rq1+r512, g3) | 0.70 (gs1, g1) | 0.81 (gs05+r512, g1) | - | snapkv512 1.46 |
| 64 | 8192 | 31.4 | 1.18 (evict1024, g3) | 1.22 (snapkv512, g3) | 1.06 (kivi4+r128, g3) | 1.06 (rq2, g1) | 1.13 (rq2+r128, g1) | 0.67 (gs1, g1) | 0.78 (gs05+r512, g1) | - | snapkv512 1.22 |

## best speedup per family, tp_chat [computed from measured parts]
| B | ctx | T_AR ms | evict | snapkv | kivi | rq_post | rq_post+win | gsrq | gsrq+win | commvq | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8192 | 6.8 | 0.93 (evict4096, g1) | 0.98 (snapkv4096, g1) | 0.87 (kivi4+r32, g1) | 0.88 (rq2, g1) | 0.92 (rq2+r512, g1) | 0.69 (gs1, g1) | 0.77 (gs1+r512, g1) | - | snapkv4096 0.98 |
| 1 | 16384 | 7.1 | 0.97 (evict4096, g1) | 0.99 (snapkv4096, g1) | 0.89 (kivi4+r32, g1) | 0.89 (rq2, g1) | 0.93 (rq2+r512, g1) | 0.68 (gs1, g1) | 0.77 (gs1+r512, g1) | - | snapkv4096 0.99 |
| 1 | 32768 | 7.6 | 0.96 (evict2048, g1) | 1.00 (snapkv4096, g1) | - | - | - | - | - | - | snapkv4096 1.00 |
| 1 | 65536* | 8.7 | 1.02 (evict1024, g1) | 1.06 (snapkv4096, g1) | - | - | - | - | - | - | snapkv4096 1.06 |
| 8 | 8192 | 10.2 | 0.98 (evict2048, g1) | 1.01 (snapkv2048, g1) | 0.91 (kivi4+r128, g1) | 0.93 (rq2, g1) | 0.97 (rq2+r512, g1) | 0.65 (gs1, g1) | 0.74 (gs05+r512, g1) | - | snapkv2048 1.01 |
| 8 | 16384 | 12.3 | 1.06 (evict1024, g1) | 1.08 (snapkv1024, g1) | 0.96 (kivi4+r128, g1) | 0.99 (rq2, g1) | 1.03 (rq2+r512, g1) | 0.66 (gs1, g1) | 0.74 (gs05+r512, g1) | - | snapkv1024 1.08 |
| 8 | 32768 | 16.7 | 1.18 (evict512, g1) | 1.24 (snapkv4096, g3) | - | - | - | - | - | - | snapkv4096 1.24 |
| 8 | 65536* | 25.6 | 1.44 (evict2048, g3) | 1.59 (snapkv4096, g3) | - | - | - | - | - | - | snapkv4096 1.59 |
| 16 | 8192 | 13.5 | 0.98 (evict512, g1) | 1.01 (snapkv1024, g1) | 0.93 (kivi4+r128, g3) | 0.95 (rq2, g1) | 0.98 (rq2+r512, g1) | 0.65 (gs1, g1) | 0.72 (gs05+r512, g1) | - | snapkv1024 1.01 |
| 16 | 16384 | 17.9 | 1.13 (evict1024, g2) | 1.19 (snapkv2048, g3) | 1.05 (kivi4+r128, g3) | 1.05 (rq2, g1) | 1.13 (rq2+r512, g3) | 0.67 (gs1, g1) | 0.75 (gs05+r512, g1) | - | snapkv2048 1.19 |
| 16 | 32768 | 26.8 | 1.35 (evict1024, g2) | 1.44 (snapkv2048, g3) | - | - | - | - | - | - | snapkv2048 1.44 |
| 32 | 8192 | 20.2 | 1.08 (evict256, g1) | 1.12 (snapkv512, g2) | 1.02 (kivi4+r32, g3) | 1.05 (rq2, g1) | 1.09 (rq2+r128, g2) | 0.70 (gs1, g1) | 0.76 (gs05+r512, g1) | - | snapkv512 1.12 |
| 32 | 16384 | 29.0 | 1.32 (evict512, g2) | 1.36 (snapkv1024, g2) | 1.17 (kivi4+r128, g3) | 1.20 (rq2, g2) | 1.29 (rq2+r512, g3) | 0.71 (gs1, g1) | 0.79 (gs05+r512, g1) | - | snapkv1024 1.36 |
| 64 | 8192 | 31.4 | 1.15 (evict256, g1) | 1.19 (snapkv512, g1) | 1.08 (kivi4+r32, g3) | 1.11 (rq2, g1) | 1.15 (rq2+r128, g2) | 0.71 (gs1, g1) | 0.77 (gs05+r512, g1) | - | snapkv512 1.19 |

## best speedup per family, pp_pg19 [computed from measured parts]
| B | ctx | T_AR ms | evict | snapkv | kivi | rq_post | rq_post+win | gsrq | gsrq+win | commvq | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32768 | 21.9 | 1.07 (evict4096, g2) | 1.08 (snapkv4096, g2) | 1.01 (kivi4+r128, g3) | 0.93 (rq2, g1) | 1.04 (rq2+r512, g1) | 0.69 (gs05, g1) | 0.86 (gs05+r512, g1) | 0.21 (cvq1, g1) | snapkv4096 1.08 |
| 16 | 16384 | 54.2 | 1.58 (evict512, g4) | 1.66 (snapkv1024, g4) | 1.31 (kivi4+r128, g6) | 1.24 (rq2, g2) | 1.46 (rq2+r512, g4) | 0.70 (gs05, g1) | 0.84 (gs05+r512, g1) | 0.11 (cvq1, g1) | snapkv1024 1.66 |

## best speedup per family, pp_chat [computed from measured parts]
| B | ctx | T_AR ms | evict | snapkv | kivi | rq_post | rq_post+win | gsrq | gsrq+win | commvq | best |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 32768 | 21.9 | 1.02 (evict2048, g1) | 1.05 (snapkv4096, g1) | - | - | - | - | - | - | snapkv4096 1.05 |
| 16 | 16384 | 54.2 | 1.44 (evict512, g2) | 1.53 (snapkv1024, g4) | 1.31 (kivi4+r128, g6) | 1.31 (rq2, g2) | 1.43 (rq2+r512, g3) | 0.73 (gs05, g1) | 0.82 (gs05+r512, g1) | 0.11 (cvq1, g1) | snapkv1024 1.53 |

## end-to-end TP=4 spec loop (synthetic weights, replayed acceptance) vs computed
| B | ctx | draft (tau) | gamma | e2e speedup | computed |
|---|---|---|---|---|---|
| 16 | 32768 | evict1024 (snapkv1024) | 6 | 1.656 | 1.648 |
| 8 | 32768 | evict1024 (snapkv1024) | 3 | 1.371 | 1.340 |
| 8 | 32768 | kivi4+r128 (kivi4g32+r128) | 6 | 1.099 | 1.085 |
| 8 | 32768 | rq1+r512 (vqpL1+r512) | 3 | 1.210 | 1.186 |
