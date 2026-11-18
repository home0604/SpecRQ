# E7 v5: decode speedup vs AR under two batch accountings

- **static (measured):** each sequence makes exactly gen tokens, finished rows idle until the batch ends (tau_eff = gen/rounds < tau).
- **all-active (derived, MagicDec official accounting; close to continuous batching):** speedup = tau x T_AR / T_round from the same measured tau and step times. Our recomputation matched the official MagicDec numbers within 1% (results/E7/llama8b_magicdec_official_tables.md).
- Headline for E7 from 2026-11-18: **all-active**; static is reported as a secondary result (eviction drafts lose more to stragglers).

## Summary (best per family)

| point | SnapKV static | VQ static | ratio | SnapKV all-active | VQ all-active | ratio |
|---|---|---|---|---|---|---|
| syn_quote ctx 32768, B=16, gen 256 | 1.200 | 1.510 | 1.258 | 1.449 | 1.624 | 1.121 |
| syn_quote ctx 32768, B=16, gen 512 | 1.138 | 1.417 | 1.245 | 1.453 | 1.547 | 1.065 |
| syn_quote ctx 32768, B=8, gen 256 | 1.145 | 1.228 | 1.072 | 1.269 | 1.327 | 1.046 |
| hotpotqa_q ctx 16384, B=16, gen 256 | 0.937 | 1.134 | 1.210 | 1.156 | 1.213 | 1.049 |
| pg19 ctx 32768, B=16, gen 256 | 1.434 | 1.320 | 0.921 | 1.578 | 1.476 | 0.935 |

## syn_quote ctx 32768, B=16, gen 256

| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |
|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 3.55 | 0.83 | 1.200 | 1.449 |
| SnapKV-4096 | 4 | 4.22 | 0.78 | 1.119 | 1.440 |
| RQ2+r128 | 6 | 6.18 | 0.88 | 1.360 | 1.544 |
| RQ2+r128 | 4 | 4.66 | 0.92 | 1.399 | 1.528 |
| SnapKV512+RQ2 hybrid | 6 | 6.61 | 0.92 | 1.498 | 1.624 |
| SnapKV512+RQ2 hybrid | 4 | 4.84 | 0.96 | 1.510 | 1.568 |

best VQ/hybrid ÷ best SnapKV: static **1.258**, all-active **1.121**

## syn_quote ctx 32768, B=16, gen 512

| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |
|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 3.57 | 0.78 | 1.138 | 1.453 |
| RQ2+r128 | 4 | 4.57 | 0.94 | 1.403 | 1.490 |
| SnapKV512+RQ2 hybrid | 4 | 4.78 | 0.92 | 1.417 | 1.547 |

best VQ/hybrid ÷ best SnapKV: static **1.245**, all-active **1.065**

## syn_quote ctx 32768, B=8, gen 256

| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |
|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 3.59 | 0.90 | 1.145 | 1.269 |
| SnapKV-4096 | 4 | 4.24 | 0.89 | 1.085 | 1.221 |
| RQ2+r128 | 3 | 3.83 | 0.96 | 1.228 | 1.286 |
| RQ2+r128 | 4 | 4.70 | 0.94 | 1.203 | 1.281 |
| SnapKV512+RQ2 hybrid | 6 | 6.53 | 0.89 | 1.182 | 1.327 |
| SnapKV512+RQ2 hybrid | 4 | 4.78 | 0.94 | 1.212 | 1.291 |

best VQ/hybrid ÷ best SnapKV: static **1.072**, all-active **1.046**

## hotpotqa_q ctx 16384, B=16, gen 256

| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |
|---|---|---|---|---|---|
| SnapKV-2048 | 2 | 2.70 | 0.81 | 0.937 | 1.156 |
| SnapKV-2048 | 4 | 4.05 | 0.58 | 0.646 | 1.124 |
| RQ2+r128 | 3 | 3.63 | 0.94 | 1.092 | 1.164 |
| RQ2+r128 | 4 | 4.41 | 0.91 | 1.043 | 1.151 |
| SnapKV512+RQ2 hybrid | 3 | 3.80 | 0.94 | 1.134 | 1.213 |
| SnapKV512+RQ2 hybrid | 4 | 4.66 | 0.92 | 1.109 | 1.212 |

best VQ/hybrid ÷ best SnapKV: static **1.210**, all-active **1.049**

## pg19 ctx 32768, B=16, gen 256

| draft | gamma | tau | tau_eff/tau | static (measured) | all-active (derived) |
|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 3.72 | 0.84 | 1.293 | 1.541 |
| SnapKV-4096 | 4 | 4.54 | 0.91 | 1.434 | 1.578 |
| RQ2+r128 | 3 | 3.32 | 0.84 | 1.099 | 1.310 |
| RQ2+r128 | 4 | 4.03 | 0.85 | 1.123 | 1.326 |
| SnapKV512+RQ2 hybrid | 3 | 3.58 | 0.84 | 1.192 | 1.416 |
| SnapKV512+RQ2 hybrid | 4 | 4.47 | 0.89 | 1.320 | 1.476 |

best VQ/hybrid ÷ best SnapKV: static **0.921**, all-active **0.935**

