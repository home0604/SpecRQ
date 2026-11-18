# E7 magicvq v5aa: static vs MEASURED all-active batch accounting (TP=4 RTX 4090, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, v5 prompts / drafts / gammas / default codebooks)

- One run per point (e2e_real.py --accounting both): same AR baseline, plans, CUDA graphs; static and all-active reps interleaved (warm-up of each excluded, 3 timed reps each, median).
- **static (measured):** every row makes exactly gen tokens; finished rows idle until the slowest row is done. Speedup = B*gen/wall vs AR tok/s.
- **all-active (measured):** no row is frozen; the batch stops when the LONGEST row has >= gen new tokens (EOS ignored); throughput = all tokens of all rows / decode wall (MagicDec official accounting, minus EOS stop).
- **all-active derived:** tau_static x T_AR / T_round_static, (a) from this run, (b) v5 value from results/E7/llama8b_magicvq_v5_accounting.md (v5 run).
- AR: every row gen tokens (no stragglers). Speedups vs the AR tok/s of the same run.

## Summary: best VQ/hybrid / best SnapKV

| point | static meas: SnapKV / VQ / ratio | all-active meas: SnapKV / VQ / ratio | all-active derived v5: SnapKV / VQ / ratio |
|---|---|---|---|
| syn_quote ctx 32768, B=16, gen 256 | 1.200 (g3) / 1.513 (SnapKV512+RQ2 hybrid g4) / **1.260** | 1.468 (g4) / 1.626 (SnapKV512+RQ2 hybrid g6) / **1.108** | 1.449 (g3) / 1.624 (SnapKV512+RQ2 hybrid g6) / **1.121** |
| syn_quote ctx 32768, B=16, gen 512 | 1.136 (g3) / 1.415 (SnapKV512+RQ2 hybrid g4) / **1.246** | 1.454 (g3) / 1.548 (SnapKV512+RQ2 hybrid g4) / **1.065** | 1.453 (g3) / 1.547 (SnapKV512+RQ2 hybrid g4) / **1.065** |
| syn_quote ctx 32768, B=8, gen 256 | 1.146 (g3) / 1.228 (RQ2+r128 g3) / **1.072** | 1.276 (g3) / 1.338 (SnapKV512+RQ2 hybrid g6) / **1.049** | 1.269 (g3) / 1.327 (SnapKV512+RQ2 hybrid g6) / **1.046** |
| hotpotqa_q ctx 16384, B=16, gen 256 | 0.936 (g2) / 1.131 (SnapKV512+RQ2 hybrid g3) / **1.209** | 1.194 (g4) / 1.209 (SnapKV512+RQ2 hybrid g4) / **1.012** | 1.156 (g2) / 1.213 (SnapKV512+RQ2 hybrid g3) / **1.049** |
| pg19 ctx 32768, B=16, gen 256 | 1.430 (g4) / 1.317 (SnapKV512+RQ2 hybrid g4) / **0.921** | 1.579 (g4) / 1.470 (SnapKV512+RQ2 hybrid g4) / **0.931** | 1.578 (g4) / 1.476 (SnapKV512+RQ2 hybrid g4) / **0.935** |

## syn_quote ctx 32768, B=16, gen 256

AR 606.3 tok/s (step 26.39 ms); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 1.200 | **1.466** (1.465-1.467) | 1.449 | 1.449 | 3.55 / 3.59 | 87 / 64 | 3678 (193-256) | 64.7 / 64.6 | 16/16 | no |
| SnapKV-4096 | 4 | 1.119 | **1.468** (1.468-1.468) | 1.440 | 1.440 | 4.22 / 4.30 | 78 / 52 | 3580 (179-257) | 77.4 / 77.4 | 16/16 | no |
| RQ2+r128 | 6 | 1.334 | **1.560** (1.551-1.560) | 1.546 | 1.544 | 6.18 / 6.25 | 48 / 38 | 3798 (214-260) | 105.5 / 105.7 | 15/16 | yes |
| RQ2+r128 | 4 | 1.401 | **1.536** (1.533-1.537) | 1.527 | 1.528 | 4.65 / 4.68 | 60 / 52 | 3890 (227-259) | 80.4 / 80.3 | 15/16 | yes |
| SnapKV512+RQ2 hybrid | 6 | 1.500 | **1.626** (1.626-1.626) | 1.624 | 1.624 | 6.60 / 6.61 | 42 / 37 | 3915 (224-259) | 107.3 / 107.3 | 15/16 | yes |
| SnapKV512+RQ2 hybrid | 4 | 1.513 | **1.570** (1.568-1.571) | 1.572 | 1.568 | 4.84 / 4.83 | 55 / 52 | 4026 (244-259) | 81.2 / 81.3 | 15/16 | yes |

best VQ/hybrid / best SnapKV: static meas 1.260, all-active meas 1.108, all-active derived v5 1.121

## syn_quote ctx 32768, B=16, gen 512

AR 604.7 tok/s (step 26.46 ms); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 1.136 | **1.454** (1.454-1.455) | 1.448 | 1.453 | 3.57 / 3.57 | 183 / 129 | 7367 (367-513) | 65.2 / 64.9 | 16/16 | no |
| RQ2+r128 | 4 | 1.400 | **1.486** (1.482-1.486) | 1.486 | 1.490 | 4.56 / 4.56 | 119 / 107 | 7808 (454-515) | 81.3 / 81.2 | 7/16 | yes |
| SnapKV512+RQ2 hybrid | 4 | 1.415 | **1.548** (1.547-1.549) | 1.545 | 1.547 | 4.78 / 4.78 | 117 / 104 | 7959 (460-516) | 81.8 / 81.7 | 7/16 | yes |

best VQ/hybrid / best SnapKV: static meas 1.246, all-active meas 1.065, all-active derived v5 1.065

## syn_quote ctx 32768, B=8, gen 256

AR 485.2 tok/s (step 16.49 ms); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 1.146 | **1.276** (1.276-1.276) | 1.269 | 1.269 | 3.59 / 3.61 | 79 / 64 | 1847 (202-256) | 46.6 / 46.6 | 7/8 | no |
| SnapKV-4096 | 4 | 1.086 | **1.231** (1.231-1.231) | 1.223 | 1.221 | 4.24 / 4.27 | 68 / 53 | 1809 (193-258) | 57.2 / 57.1 | 8/8 | no |
| RQ2+r128 | 3 | 1.228 | **1.287** (1.284-1.287) | 1.283 | 1.286 | 3.82 / 3.83 | 70 / 65 | 1992 (241-259) | 49.1 / 49.1 | 7/8 | yes |
| RQ2+r128 | 4 | 1.225 | **1.284** (1.283-1.287) | 1.279 | 1.281 | 4.69 / 4.71 | 57 / 52 | 1963 (235-258) | 60.5 / 60.4 | 7/8 | yes |
| SnapKV512+RQ2 hybrid | 6 | 1.210 | **1.338** (1.336-1.345) | 1.324 | 1.327 | 6.51 / 6.59 | 43 / 37 | 1950 (226-258) | 81.1 / 81.1 | 7/8 | yes |
| SnapKV512+RQ2 hybrid | 4 | 1.213 | **1.297** (1.296-1.299) | 1.292 | 1.291 | 4.78 / 4.80 | 57 / 52 | 1997 (239-257) | 61.1 / 61.1 | 7/8 | yes |

best VQ/hybrid / best SnapKV: static meas 1.072, all-active meas 1.049, all-active derived v5 1.046

## hotpotqa_q ctx 16384, B=16, gen 256

AR 906.9 tok/s (step 17.64 ms); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-2048 | 2 | 0.936 | **1.168** (1.168-1.169) | 1.153 | 1.156 | 2.70 / 2.73 | 117 / 86 | 3756 (188-258) | 41.3 / 41.2 | 5/16 | no |
| SnapKV-2048 | 4 | 0.644 | **1.194** (1.194-1.194) | 1.121 | 1.124 | 4.05 / 4.30 | 110 / 52 | 3580 (138-260) | 63.8 / 63.6 | 7/16 | no |
| RQ2+r128 | 3 | 1.090 | **1.155** (1.152-1.155) | 1.158 | 1.164 | 3.63 / 3.61 | 75 / 68 | 3931 (229-259) | 55.3 / 55.2 | 16/16 | yes |
| RQ2+r128 | 4 | 1.041 | **1.146** (1.144-1.147) | 1.143 | 1.151 | 4.39 / 4.40 | 64 / 56 | 3937 (225-257) | 67.8 / 67.8 | 7/16 | yes |
| SnapKV512+RQ2 hybrid | 3 | 1.131 | **1.207** (1.206-1.207) | 1.210 | 1.213 | 3.80 / 3.79 | 72 / 64 | 3876 (228-256) | 55.4 / 55.4 | 16/16 | yes |
| SnapKV512+RQ2 hybrid | 4 | 1.126 | **1.209** (1.208-1.209) | 1.210 | 1.212 | 4.66 / 4.65 | 59 / 52 | 3872 (226-257) | 68.0 / 67.9 | 8/16 | yes |

best VQ/hybrid / best SnapKV: static meas 1.209, all-active meas 1.012, all-active derived v5 1.049

## pg19 ctx 32768, B=16, gen 256

AR 591.7 tok/s (step 27.04 ms); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | gamma | static meas | all-active meas (min-max) | aa derived (this run) | aa derived (v5) | tau static / aa | rounds static / aa | aa tokens (row min-max) | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | 3 | 1.293 | **1.544** (1.544-1.545) | 1.539 | 1.541 | 3.72 / 3.73 | 82 / 65 | 3875 (209-259) | 65.3 / 65.2 | 16/16 | no |
| SnapKV-4096 | 4 | 1.430 | **1.579** (1.578-1.580) | 1.573 | 1.578 | 4.54 / 4.55 | 62 / 52 | 3786 (215-260) | 78.1 / 77.9 | 1/16 | no |
| RQ2+r128 | 3 | 1.122 | **1.314** (1.311-1.319) | 1.308 | 1.310 | 3.32 / 3.33 | 90 / 65 | 3454 (186-258) | 68.6 / 68.5 | 16/16 | yes |
| RQ2+r128 | 4 | 1.120 | **1.304** (1.303-1.315) | 1.319 | 1.326 | 4.02 / 3.97 | 75 / 52 | 3329 (173-258) | 82.4 / 82.3 | 1/16 | yes |
| SnapKV512+RQ2 hybrid | 3 | 1.191 | **1.424** (1.423-1.429) | 1.418 | 1.416 | 3.59 / 3.60 | 85 / 66 | 3794 (205-258) | 68.4 / 68.3 | 16/16 | yes |
| SnapKV512+RQ2 hybrid | 4 | 1.317 | **1.470** (1.465-1.472) | 1.468 | 1.476 | 4.46 / 4.46 | 64 / 52 | 3696 (203-258) | 82.1 / 82.0 | 1/16 | yes |

best VQ/hybrid / best SnapKV: static meas 0.921, all-active meas 0.931, all-active derived v5 0.935

