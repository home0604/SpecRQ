# E7 magicvq v6e2e: v6 draft-trained codebook (t2w) vs default codebook, measured e2e, static and all-active (TP=4 RTX 4090, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, v5 prompts / drafts / gammas)

- Kernel: deterministic VQ LUT draft kernel (fixed-order q.mu reduction replaces the shared-memory float atomicAdd; csrc/vqattn_lut_rag.cu). Default-codebook rows are therefore re-measured; 'v5aa' columns are the pre-fix values.
- One run per point (e2e_real.py --accounting both --cbx t2w=cb_d8K256x2_post_cn_t2w.pt): one AR baseline, default ('def', cb_d8K256x2_post_cn_greedy.pt) and t2w drafts in the same process; static and all-active reps interleaved, warm-up excluded, 3 timed reps each, median.
- static: every row exactly gen tokens (finished rows idle). all-active (headline, MagicDec accounting): no row frozen, stop when the longest row has >= gen tokens, all tokens / decode wall.
- tau ratio meas = tau_static(t2w) / tau_static(def) for the same draft and gamma (same prompts, same AR target trajectory up to divergence); v6 computed = paired tau ratio of results/E7/llama8b_magicvq_v6_tables.md (same task / ctx / gamma; v6 used more samples and a replay-style tau eval, not this batch).

## Summary: best VQ/hybrid / best SnapKV (measured)

| point | AR tok/s | static: SnapKV / VQ def / VQ t2w | static ratio def / t2w | all-active: SnapKV / VQ def / VQ t2w | **all-active ratio def / t2w** | v5aa aa ratio (pre-fix) |
|---|---|---|---|---|---|---|
| syn_quote ctx 32768, B=16, gen 256 | 605.8 | 1.200 (g3) / 1.520 (SnapKV512+RQ2 hybrid g4) / 1.627 (SnapKV512+RQ2 hybrid g6) | 1.267 / 1.356 | 1.469 (g4) / 1.638 (SnapKV512+RQ2 hybrid g6) / 1.711 (SnapKV512+RQ2 hybrid g6) | **1.115 / 1.165** | 1.108 |
| syn_quote ctx 32768, B=16, gen 512 | 604.8 | 1.135 (g3) / 1.436 (SnapKV512+RQ2 hybrid g4) / 1.520 (RQ2+r128 g4) | 1.265 / 1.339 | 1.453 (g3) / 1.561 (SnapKV512+RQ2 hybrid g4) / 1.599 (SnapKV512+RQ2 hybrid g4) | **1.074 / 1.101** | 1.065 |
| syn_quote ctx 32768, B=8, gen 256 | 485.5 | 1.146 (g3) / 1.225 (SnapKV512+RQ2 hybrid g4) / 1.350 (SnapKV512+RQ2 hybrid g6) | 1.069 / 1.178 | 1.276 (g3) / 1.360 (SnapKV512+RQ2 hybrid g6) / 1.414 (SnapKV512+RQ2 hybrid g6) | **1.066 / 1.108** | 1.049 |
| hotpotqa_q ctx 16384, B=16, gen 256 | 903.8 | 0.935 (g2) / 1.157 (SnapKV512+RQ2 hybrid g3) / 1.208 (SnapKV512+RQ2 hybrid g3) | 1.237 / 1.292 | 1.197 (g4) / 1.221 (SnapKV512+RQ2 hybrid g4) / 1.266 (SnapKV512+RQ2 hybrid g4) | **1.020 / 1.058** | 1.012 |
| pg19 ctx 32768, B=16, gen 256 | 591.6 | 1.433 (g4) / 1.348 (SnapKV512+RQ2 hybrid g4) / 1.440 (SnapKV512+RQ2 hybrid g4) | 0.941 / 1.005 | 1.581 (g4) / 1.478 (SnapKV512+RQ2 hybrid g4) / 1.562 (SnapKV512+RQ2 hybrid g4) | **0.935 / 0.988** | 0.931 |

## tau: t2w vs default codebook (measured static tau ratio vs v6 computed ratio)

| point | draft | gamma | tau static def -> t2w | tau aa def -> t2w | meas ratio (static) | v6 computed ratio | aa speedup def -> t2w |
|---|---|---|---|---|---|---|---|
| syn_quote ctx 32768, B=16, gen 256 | RQ2+r128 | 6 | 6.17 -> 6.69 | 6.24 -> 6.70 | 1.085 | 1.086 | 1.569 -> 1.686 |
| syn_quote ctx 32768, B=16, gen 256 | RQ2+r128 | 4 | 4.65 -> 4.86 | 4.68 -> 4.86 | 1.045 | 1.049 | 1.547 -> 1.608 |
| syn_quote ctx 32768, B=16, gen 256 | SnapKV512+RQ2 hybrid | 6 | 6.61 -> 6.90 | 6.60 -> 6.90 | 1.042 | 1.067 | 1.638 -> 1.711 |
| syn_quote ctx 32768, B=16, gen 256 | SnapKV512+RQ2 hybrid | 4 | 4.83 -> 4.94 | 4.84 -> 4.94 | 1.024 | 1.036 | 1.580 -> 1.615 |
| syn_quote ctx 32768, B=16, gen 512 | RQ2+r128 | 4 | 4.56 -> 4.80 | 4.55 -> 4.80 | 1.054 | 1.049 | 1.501 -> 1.584 |
| syn_quote ctx 32768, B=16, gen 512 | SnapKV512+RQ2 hybrid | 4 | 4.78 -> 4.91 | 4.79 -> 4.91 | 1.026 | 1.036 | 1.561 -> 1.599 |
| syn_quote ctx 32768, B=8, gen 256 | RQ2+r128 | 3 | 3.81 -> 3.93 | 3.82 -> 3.94 | 1.031 | 1.034 | 1.296 -> 1.333 |
| syn_quote ctx 32768, B=8, gen 256 | RQ2+r128 | 4 | 4.69 -> 4.91 | 4.72 -> 4.91 | 1.047 | 1.049 | 1.298 -> 1.351 |
| syn_quote ctx 32768, B=8, gen 256 | SnapKV512+RQ2 hybrid | 6 | 6.55 -> 6.86 | 6.61 -> 6.88 | 1.047 | 1.067 | 1.360 -> 1.414 |
| syn_quote ctx 32768, B=8, gen 256 | SnapKV512+RQ2 hybrid | 4 | 4.78 -> 4.92 | 4.80 -> 4.92 | 1.028 | 1.036 | 1.309 -> 1.342 |
| hotpotqa_q ctx 16384, B=16, gen 256 | RQ2+r128 | 3 | 3.62 -> 3.86 | 3.61 -> 3.86 | 1.066 | 1.052 | 1.176 -> 1.256 |
| hotpotqa_q ctx 16384, B=16, gen 256 | RQ2+r128 | 4 | 4.40 -> 4.71 | 4.39 -> 4.71 | 1.071 | 1.069 | 1.169 -> 1.253 |
| hotpotqa_q ctx 16384, B=16, gen 256 | SnapKV512+RQ2 hybrid | 3 | 3.80 -> 3.92 | 3.79 -> 3.92 | 1.032 | 1.041 | 1.218 -> 1.258 |
| hotpotqa_q ctx 16384, B=16, gen 256 | SnapKV512+RQ2 hybrid | 4 | 4.66 -> 4.82 | 4.65 -> 4.82 | 1.036 | 1.066 | 1.221 -> 1.266 |
| pg19 ctx 32768, B=16, gen 256 | RQ2+r128 | 3 | 3.31 -> 3.64 | 3.30 -> 3.64 | 1.101 | 1.095 | 1.320 -> 1.453 |
| pg19 ctx 32768, B=16, gen 256 | RQ2+r128 | 4 | 4.05 -> 4.42 | 4.00 -> 4.43 | 1.092 | 1.132 | 1.334 -> 1.475 |
| pg19 ctx 32768, B=16, gen 256 | SnapKV512+RQ2 hybrid | 3 | 3.59 -> 3.79 | 3.61 -> 3.79 | 1.054 | 1.045 | 1.436 -> 1.507 |
| pg19 ctx 32768, B=16, gen 256 | SnapKV512+RQ2 hybrid | 4 | 4.46 -> 4.69 | 4.45 -> 4.70 | 1.052 | 1.063 | 1.478 -> 1.559 |

## syn_quote ctx 32768, B=16, gen 256

AR 605.8 tok/s (step 26.41 ms) (v5aa AR 606.3); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | aa change vs v5aa |
|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | def | 3 | 1.200 | **1.467** (1.466-1.467) | 3.55 / 3.59 | 64.8 / 64.7 | 16/16 | no | 1.200 / 1.466 | +0.05% |
| SnapKV-4096 | def | 4 | 1.119 | **1.469** (1.468-1.469) | 4.22 / 4.30 | 77.5 / 77.4 | 16/16 | no | 1.119 / 1.468 | +0.07% |
| RQ2+r128 | def | 6 | 1.370 | **1.569** (1.569-1.570) | 6.17 / 6.24 | 105.0 / 105.0 | 15/16 | no | 1.334 / 1.560 | +0.58% |
| RQ2+r128 | def | 4 | 1.411 | **1.547** (1.547-1.548) | 4.65 / 4.68 | 79.9 / 79.9 | 15/16 | no | 1.401 / 1.536 | +0.71% |
| SnapKV512+RQ2 hybrid | def | 6 | 1.509 | **1.638** (1.637-1.638) | 6.61 / 6.60 | 106.7 / 106.5 | 15/16 | no | 1.500 / 1.626 | +0.73% |
| SnapKV512+RQ2 hybrid | def | 4 | 1.520 | **1.580** (1.580-1.580) | 4.83 / 4.84 | 80.9 / 80.8 | 15/16 | no | 1.513 / 1.570 | +0.61% |
| RQ2+r128 | t2w | 6 | 1.572 | **1.686** (1.686-1.687) | 6.69 / 6.70 | 104.9 / 104.9 | 15/16 | no | - | - |
| RQ2+r128 | t2w | 4 | 1.540 | **1.608** (1.607-1.608) | 4.86 / 4.86 | 79.8 / 79.8 | 15/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 6 | 1.627 | **1.711** (1.710-1.712) | 6.90 / 6.90 | 106.5 / 106.5 | 15/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 4 | 1.548 | **1.615** (1.614-1.615) | 4.94 / 4.94 | 80.9 / 80.9 | 15/16 | no | - | - |

best VQ/hybrid / best SnapKV (def / t2w): static 1.267 / 1.356, all-active **1.115 / 1.165**

## syn_quote ctx 32768, B=16, gen 512

AR 604.8 tok/s (step 26.46 ms) (v5aa AR 604.7); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | aa change vs v5aa |
|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | def | 3 | 1.135 | **1.453** (1.453-1.453) | 3.57 / 3.57 | 65.2 / 65.0 | 16/16 | no | 1.136 / 1.454 | -0.10% |
| RQ2+r128 | def | 4 | 1.416 | **1.501** (1.500-1.501) | 4.56 / 4.55 | 80.4 / 80.3 | 7/16 | no | 1.400 / 1.486 | +1.02% |
| SnapKV512+RQ2 hybrid | def | 4 | 1.436 | **1.561** (1.561-1.561) | 4.78 / 4.79 | 81.3 / 81.1 | 7/16 | no | 1.415 / 1.548 | +0.80% |
| RQ2+r128 | t2w | 4 | 1.520 | **1.584** (1.584-1.585) | 4.80 / 4.80 | 80.3 / 80.2 | 7/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 4 | 1.485 | **1.599** (1.599-1.599) | 4.91 / 4.91 | 81.4 / 81.3 | 7/16 | no | - | - |

best VQ/hybrid / best SnapKV (def / t2w): static 1.265 / 1.339, all-active **1.074 / 1.101**

## syn_quote ctx 32768, B=8, gen 256

AR 485.5 tok/s (step 16.48 ms) (v5aa AR 485.2); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | aa change vs v5aa |
|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | def | 3 | 1.146 | **1.276** (1.276-1.277) | 3.59 / 3.61 | 46.6 / 46.6 | 7/8 | no | 1.146 / 1.276 | +0.05% |
| SnapKV-4096 | def | 4 | 1.085 | **1.230** (1.230-1.230) | 4.24 / 4.27 | 57.2 / 57.1 | 8/8 | no | 1.086 / 1.231 | -0.07% |
| RQ2+r128 | def | 3 | 1.222 | **1.296** (1.295-1.296) | 3.81 / 3.82 | 48.6 / 48.6 | 7/8 | no | 1.228 / 1.287 | +0.66% |
| RQ2+r128 | def | 4 | 1.214 | **1.298** (1.298-1.298) | 4.69 / 4.72 | 59.9 / 59.9 | 7/8 | no | 1.225 / 1.284 | +1.14% |
| SnapKV512+RQ2 hybrid | def | 6 | 1.224 | **1.360** (1.360-1.361) | 6.55 / 6.61 | 80.1 / 80.1 | 7/8 | no | 1.210 / 1.338 | +1.62% |
| SnapKV512+RQ2 hybrid | def | 4 | 1.225 | **1.309** (1.309-1.309) | 4.78 / 4.80 | 60.4 / 60.4 | 7/8 | no | 1.213 / 1.297 | +0.93% |
| RQ2+r128 | t2w | 3 | 1.294 | **1.333** (1.333-1.333) | 3.93 / 3.94 | 48.6 / 48.6 | 6/8 | no | - | - |
| RQ2+r128 | t2w | 4 | 1.305 | **1.351** (1.351-1.351) | 4.91 / 4.91 | 59.9 / 59.9 | 6/8 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 6 | 1.350 | **1.414** (1.414-1.414) | 6.86 / 6.88 | 80.1 / 80.1 | 7/8 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 4 | 1.269 | **1.342** (1.342-1.342) | 4.92 / 4.92 | 60.4 / 60.4 | 6/8 | no | - | - |

best VQ/hybrid / best SnapKV (def / t2w): static 1.069 / 1.178, all-active **1.066 / 1.108**

## hotpotqa_q ctx 16384, B=16, gen 256

AR 903.8 tok/s (step 17.70 ms) (v5aa AR 906.9); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | aa change vs v5aa |
|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-2048 | def | 2 | 0.935 | **1.167** (1.167-1.168) | 2.70 / 2.73 | 41.4 / 41.4 | 5/16 | no | 0.936 / 1.168 | -0.06% |
| SnapKV-2048 | def | 4 | 0.645 | **1.197** (1.196-1.197) | 4.05 / 4.30 | 63.9 / 63.7 | 7/16 | no | 0.644 / 1.194 | +0.21% |
| RQ2+r128 | def | 3 | 1.110 | **1.176** (1.176-1.176) | 3.62 / 3.61 | 54.4 / 54.4 | 16/16 | no | 1.090 / 1.155 | +1.85% |
| RQ2+r128 | def | 4 | 1.064 | **1.169** (1.169-1.169) | 4.40 / 4.39 | 66.5 / 66.6 | 7/16 | no | 1.041 / 1.146 | +2.03% |
| SnapKV512+RQ2 hybrid | def | 3 | 1.157 | **1.218** (1.217-1.218) | 3.80 / 3.79 | 55.2 / 55.1 | 16/16 | no | 1.131 / 1.207 | +0.91% |
| SnapKV512+RQ2 hybrid | def | 4 | 1.099 | **1.221** (1.220-1.221) | 4.66 / 4.65 | 67.6 / 67.4 | 8/16 | no | 1.126 / 1.209 | +1.01% |
| RQ2+r128 | t2w | 3 | 1.172 | **1.256** (1.254-1.256) | 3.86 / 3.86 | 54.5 / 54.4 | 16/16 | no | - | - |
| RQ2+r128 | t2w | 4 | 1.135 | **1.253** (1.253-1.254) | 4.71 / 4.71 | 66.6 / 66.5 | 7/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 3 | 1.208 | **1.258** (1.257-1.258) | 3.92 / 3.92 | 55.2 / 55.2 | 16/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w | 4 | 1.198 | **1.266** (1.265-1.266) | 4.82 / 4.82 | 67.5 / 67.5 | 7/16 | no | - | - |

best VQ/hybrid / best SnapKV (def / t2w): static 1.237 / 1.292, all-active **1.020 / 1.058**

## pg19 ctx 32768, B=16, gen 256

AR 591.6 tok/s (step 27.05 ms) (v5aa AR 591.7); co-tenants on GPUs 4-7 at start: none, end: none; any config flagged: no

| draft | cb | gamma | static meas | all-active meas (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | aa per-row lengths vary across reps | v5aa static / aa | aa change vs v5aa |
|---|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | def | 3 | 1.293 | **1.544** (1.544-1.545) | 3.72 / 3.73 | 65.3 / 65.3 | 16/16 | no | 1.293 / 1.544 | -0.02% |
| SnapKV-4096 | def | 4 | 1.433 | **1.581** (1.581-1.582) | 4.54 / 4.55 | 77.9 / 77.8 | 1/16 | no | 1.430 / 1.579 | +0.12% |
| RQ2+r128 | def | 3 | 1.111 | **1.320** (1.320-1.321) | 3.31 / 3.30 | 67.7 / 67.7 | 16/16 | no | 1.122 / 1.314 | +0.42% |
| RQ2+r128 | def | 4 | 1.124 | **1.334** (1.333-1.334) | 4.05 / 4.00 | 81.1 / 81.1 | 1/16 | no | 1.120 / 1.304 | +2.27% |
| SnapKV512+RQ2 hybrid | def | 3 | 1.212 | **1.436** (1.435-1.436) | 3.59 / 3.61 | 68.0 / 68.0 | 16/16 | no | 1.191 / 1.424 | +0.82% |
| SnapKV512+RQ2 hybrid | def | 4 | 1.348 | **1.478** (1.477-1.478) | 4.46 / 4.45 | 81.5 / 81.4 | 1/16 | no | 1.317 / 1.470 | +0.54% |
| SnapKV-4096 | def (sep. run) | 3 | 1.292 | **1.542** (1.541-1.543) | 3.72 / 3.73 | 65.3 / 65.3 | 16/16 | no | 1.293 / 1.544 | -0.13% |
| SnapKV-4096 | def (sep. run) | 4 | 1.431 | **1.579** (1.579-1.580) | 4.54 / 4.55 | 78.0 / 77.9 | 1/16 | no | 1.430 / 1.579 | -0.03% |
| RQ2+r128 | t2w (sep. run) | 3 | 1.381 | **1.453** (1.452-1.453) | 3.64 / 3.64 | 67.7 / 67.6 | 16/16 | no | - | - |
| RQ2+r128 | t2w (sep. run) | 4 | 1.273 | **1.475** (1.475-1.476) | 4.42 / 4.43 | 81.1 / 81.1 | 1/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w (sep. run) | 3 | 1.394 | **1.507** (1.507-1.507) | 3.79 / 3.79 | 68.0 / 67.9 | 16/16 | no | - | - |
| SnapKV512+RQ2 hybrid | t2w (sep. run) | 4 | 1.438 | **1.559** (1.559-1.560) | 4.69 / 4.70 | 81.5 / 81.4 | 1/16 | no | - | - |

*sep. run*: t2w drafts measured in a separate process (AR 592.2 tok/s there; speedups vs that AR); the summary t2w ratio uses the best SnapKV of that same process.


best VQ/hybrid / best SnapKV (def / t2w): static 0.941 / 1.005, all-active **0.935 / 0.988**

