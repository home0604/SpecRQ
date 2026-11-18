# E7 magicvq v5: FULLY MEASURED end-to-end self-speculative decoding (TP=4, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, real weights / prompts / prefill / compression / decode loop)

All numbers below are wall-clock measurements of the real loop (no replayed acceptance, no modeled step times), except the column 'v4b computed', which is the previous estimate (v4b tau x synthetic-weight step times) for the same gamma. Speedup = tokens/s of the config / tokens/s of the AR baseline in the same run (same kernels, CUDA graphs, TP). tau meas = mean accepted+1 per active (sequence, round); batch tau_eff = gen / rounds (the static batch runs until its slowest sequence has gen tokens). gen = generated tokens per sequence (fixed, EOS ignored); static batch: a sequence that reached gen waits for the slowest one.

## syn_quote ctx 32768, B=16, gen 256 (mean P 31534; prefill 75.9 s total, 4.73 s / sequence)
co-tenants on GPUs 4-7 at start: none; at end: none; any config with a co-tenant before/after its timing: no (clean)

| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | - | 605.5 (605.4-605.9) | 1.000 | - | - | - | - | - | 49.5 | 1.000 | step 26.43 ms | - | - | - |
| SnapKV-4096 | 3 | 726.8 (726.8-726.9) | **1.200** | 1.420 | 3.55 (3.52) | 3.50 | 2.94 | 0.17 | 50.1 | 1.012 | 34.3 / 30.4 / 0.04 / 0.00 / 0.02 | 87 | 1.000 | 16/16 |
| SnapKV-4096 | 4 | 677.8 (677.7-678.0) | **1.119** | 1.397 | 4.22 (4.15) | 4.10 | 3.28 | 0.17 | 49.9 | 1.007 | 45.6 / 31.8 / 0.04 / 0.00 / 0.02 | 78 | 0.990 | 15/16 |
| RQ2+r128 | 6 | 823.6 (805.8-823.9) | **1.360** | 1.546 | 6.18 (6.46) | 6.35 | 5.45 | 1.52 | 49.7 | 1.003 | 71.9 / 33.4 / 0.04 / 0.45 / 0.02 | 47 | 0.990 | 15/16 |
| RQ2+r128 | 4 | 847.0 (847.0-847.6) | **1.399** | 1.532 | 4.66 (4.79) | 4.74 | 4.27 | 1.52 | 49.8 | 1.005 | 48.3 / 31.8 / 0.04 / 0.42 / 0.02 | 60 | 0.990 | 15/16 |
| SnapKV512+RQ2 hybrid | 6 | 906.8 (906.6-907.4) | **1.498** | 1.587 | 6.61 (6.62) | 6.46 | 6.10 | 1.71 | 49.9 | 1.007 | 74.1 / 33.4 / 0.04 / 0.00 / 0.02 | 42 | 0.990 | 15/16 |
| SnapKV512+RQ2 hybrid | 4 | 914.0 (913.8-914.1) | **1.510** | 1.557 | 4.84 (4.84) | 4.78 | 4.65 | 1.71 | 49.9 | 1.007 | 49.6 / 31.9 / 0.04 / 0.00 / 0.02 | 55 | 0.990 | 15/16 |

- best RQ2+r128 (g4) / best SnapKV (snapkv4096 g3): decode 1.165, prefill+compress+decode 0.993
- best SnapKV512+RQ2 hybrid (g4) / best SnapKV (snapkv4096 g3): decode 1.257, prefill+compress+decode 0.995

## syn_quote ctx 32768, B=16, gen 512 (mean P 31534; prefill 75.3 s total, 4.70 s / sequence)
co-tenants on GPUs 4-7 at start: none; at end: none; any config with a co-tenant before/after its timing: no (clean)

| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | - | 605.2 (605.2-605.3) | 1.000 | - | - | - | - | - | 92.3 | 1.000 | step 26.44 ms | - | - | - |
| SnapKV-4096 | 3 | 688.8 (688.6-688.9) | **1.138** | 1.420 | 3.57 (3.48) | 3.50 | 2.80 | 0.17 | 93.8 | 1.017 | 34.4 / 30.5 / 0.04 / 0.00 / 0.02 | 183 | 1.000 | 16/16 |
| RQ2+r128 | 4 | 849.0 (834.8-849.3) | **1.403** | 1.532 | 4.57 (4.77) | 4.74 | 4.30 | 1.52 | 94.8 | 1.027 | 48.8 / 31.8 / 0.04 / 0.42 / 0.03 | 119 | 0.806 | 7/16 |
| SnapKV512+RQ2 hybrid | 4 | 857.7 (857.5-857.9) | **1.417** | 1.557 | 4.78 (4.83) | 4.78 | 4.38 | 1.71 | 94.7 | 1.026 | 49.7 / 31.9 / 0.04 / 0.00 / 0.03 | 117 | 0.806 | 7/16 |

- best RQ2+r128 (g4) / best SnapKV (snapkv4096 g3): decode 1.232, prefill+compress+decode 1.010
- best SnapKV512+RQ2 hybrid (g4) / best SnapKV (snapkv4096 g3): decode 1.245, prefill+compress+decode 1.009

## syn_quote ctx 32768, B=8, gen 256 (mean P 31821; prefill 38.5 s total, 4.79 s / sequence)
co-tenants on GPUs 4-7 at start: none; at end: none; any config with a co-tenant before/after its timing: no (clean)

| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | - | 485.3 (485.2-485.4) | 1.000 | - | - | - | - | - | 48.0 | 1.000 | step 16.49 ms | - | - | - |
| SnapKV-4096 | 3 | 555.5 (555.4-555.5) | **1.145** | 1.233 | 3.59 (3.57) | 3.57 | 3.24 | 0.08 | 48.5 | 1.011 | 26.8 / 19.8 / 0.04 / 0.00 / 0.02 | 79 | 0.968 | 6/8 |
| SnapKV-4096 | 4 | 526.3 (526.2-526.3) | **1.085** | 1.181 | 4.24 (4.21) | 4.20 | 3.76 | 0.08 | 48.2 | 1.006 | 35.8 / 21.4 / 0.04 / 0.00 / 0.02 | 68 | 0.968 | 6/8 |
| RQ2+r128 | 3 | 596.0 (596.0-596.1) | **1.228** | 1.287 | 3.83 (3.86) | 3.84 | 3.66 | 0.77 | 48.0 | 1.000 | 29.1 / 19.8 / 0.04 / 0.17 / 0.02 | 70 | 0.968 | 6/8 |
| RQ2+r128 | 4 | 583.7 (583.6-593.9) | **1.203** | 1.276 | 4.70 (4.76) | 4.71 | 4.41 | 0.76 | 47.9 | 0.999 | 38.8 / 21.4 / 0.04 / 0.24 / 0.02 | 58 | 0.968 | 6/8 |
| SnapKV512+RQ2 hybrid | 6 | 573.5 (573.3-586.9) | **1.182** | 1.322 | 6.53 (6.60) | 6.43 | 5.82 | 0.86 | 47.7 | 0.995 | 59.1 / 22.0 / 0.04 / 0.00 / 0.02 | 44 | 0.975 | 7/8 |
| SnapKV512+RQ2 hybrid | 4 | 588.1 (588.0-588.3) | **1.212** | 1.305 | 4.78 (4.82) | 4.76 | 4.49 | 0.86 | 47.8 | 0.997 | 39.6 / 21.4 / 0.04 / 0.00 / 0.02 | 57 | 0.968 | 6/8 |

- best RQ2+r128 (g3) / best SnapKV (snapkv4096 g3): decode 1.073, prefill+compress+decode 0.990
- best SnapKV512+RQ2 hybrid (g4) / best SnapKV (snapkv4096 g3): decode 1.059, prefill+compress+decode 0.987

## hotpotqa_q ctx 16384, B=16, gen 256 (mean P 15807; prefill 36.6 s total, 2.27 s / sequence)
co-tenants on GPUs 4-7 at start: none; at end: none; any config with a co-tenant before/after its timing: no (clean)

| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | - | 905.6 (904.8-906.6) | 1.000 | - | - | - | - | - | 99.6 | 1.000 | step 17.67 ms | - | - | - |
| SnapKV-2048 | 2 | 848.3 (847.7-848.4) | **0.937** | 1.122 | 2.70 (2.63) | 2.58 | 2.19 | 0.11 | 98.6 | 0.990 | 20.0 / 21.2 / 0.04 / 0.00 / 0.02 | 117 | 0.570 | 5/16 |
| SnapKV-2048 | 4 | 584.9 (583.9-585.4) | **0.646** | 1.036 | 4.05 (3.85) | 3.59 | 2.33 | 0.11 | 93.7 | 0.941 | 40.5 / 23.1 / 0.04 / 0.00 / 0.02 | 110 | 0.644 | 7/16 |
| RQ2+r128 | 3 | 989.2 (988.8-989.6) | **1.092** | 1.187 | 3.63 (3.61) | 3.60 | 3.41 | 0.76 | 98.7 | 0.991 | 33.0 / 21.8 / 0.04 / 0.26 / 0.02 | 75 | 1.000 | 16/16 |
| RQ2+r128 | 4 | 944.4 (944.0-946.1) | **1.043** | 1.168 | 4.41 (4.37) | 4.37 | 4.00 | 0.76 | 98.2 | 0.986 | 44.0 / 23.2 / 0.04 / 0.43 / 0.02 | 64 | 0.644 | 7/16 |
| SnapKV512+RQ2 hybrid | 3 | 1026.9 (1026.6-1041.4) | **1.134** | 1.217 | 3.80 (3.74) | 3.72 | 3.56 | 0.89 | 98.7 | 0.991 | 33.5 / 21.8 / 0.04 / 0.00 / 0.02 | 72 | 1.000 | 16/16 |
| SnapKV512+RQ2 hybrid | 4 | 1004.5 (1003.9-1006.4) | **1.109** | 1.204 | 4.66 (4.57) | 4.49 | 4.27 | 0.89 | 98.5 | 0.989 | 44.7 / 23.2 / 0.04 / 0.00 / 0.02 | 60 | 0.644 | 7/16 |

- best RQ2+r128 (g3) / best SnapKV (snapkv2048 g2): decode 1.166, prefill+compress+decode 1.001
- best SnapKV512+RQ2 hybrid (g3) / best SnapKV (snapkv2048 g2): decode 1.210, prefill+compress+decode 1.001

## pg19 ctx 32768, B=16, gen 256 (mean P 32768; prefill 79.5 s total, 4.94 s / sequence)
co-tenants on GPUs 4-7 at start: none; at end: none; any config with a co-tenant before/after its timing: no (clean)

| draft | gamma | decode tok/s (min-max) | decode speedup | v4b computed | tau meas (pre-EOS) | v4b tau same samples | batch tau_eff = gen/rounds | compress s | prefill+compress+decode tok/s | e2e speedup | ms/round draft / verify / accept / enc / host | rounds | match vs AR | identical seqs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | - | 592.0 (591.8-592.2) | 1.000 | - | - | - | - | - | 47.4 | 1.000 | step 27.03 ms | - | - | - |
| SnapKV-4096 | 3 | 765.7 (765.6-765.9) | **1.293** | - | 3.72 (3.72) | - | 3.12 | 0.17 | 48.2 | 1.016 | 34.0 / 31.2 / 0.04 / 0.00 / 0.02 | 82 | 1.000 | 16/16 |
| SnapKV-4096 | 4 | 849.0 (848.9-849.0) | **1.434** | - | 4.54 (4.54) | - | 4.13 | 0.17 | 48.5 | 1.023 | 45.3 / 32.4 / 0.04 / 0.00 / 0.02 | 62 | 0.220 | 1/16 |
| RQ2+r128 | 3 | 650.4 (650.3-665.4) | **1.099** | - | 3.32 (3.32) | - | 2.78 | 1.58 | 46.9 | 0.989 | 37.0 / 31.2 / 0.04 / 0.26 / 0.02 | 92 | 1.000 | 16/16 |
| RQ2+r128 | 4 | 664.8 (655.8-673.4) | **1.123** | - | 4.03 (4.03) | - | 3.41 | 1.58 | 47.0 | 0.991 | 49.3 / 32.4 / 0.04 / 0.40 / 0.02 | 75 | 0.220 | 1/16 |
| SnapKV512+RQ2 hybrid | 3 | 705.6 (705.2-713.4) | **1.192** | - | 3.58 (3.58) | - | 3.01 | 1.77 | 47.1 | 0.992 | 37.1 / 31.2 / 0.04 / 0.00 / 0.02 | 85 | 1.000 | 16/16 |
| SnapKV512+RQ2 hybrid | 4 | 781.3 (769.1-781.4) | **1.320** | - | 4.47 (4.47) | - | 4.00 | 1.77 | 47.4 | 0.999 | 49.4 / 32.4 / 0.04 / 0.00 / 0.02 | 64 | 0.220 | 1/16 |

- best RQ2+r128 (g4) / best SnapKV (snapkv4096 g4): decode 0.783, prefill+compress+decode 0.969
- best SnapKV512+RQ2 hybrid (g4) / best SnapKV (snapkv4096 g4): decode 0.920, prefill+compress+decode 0.977

