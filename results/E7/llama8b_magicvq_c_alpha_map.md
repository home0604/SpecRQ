# E7 c–α theory map: measured e2e points on the analytical speculative-decoding map

Figure: `results/E7/llama8b_magicvq_c_alpha_map.{png,pdf}`, script `experiments/26_magicvq/plot_c_alpha.py`.
Data: `results/E7/llama8b_magicvq_{v6e2e,v7,v7_g6,v7mse}_*.json` (4×RTX 4090 TP=4, bf16 Llama-3.1-8B-Instruct, bf16 target KV, all-active accounting = headline; median-tok/s timed rep of 3).

- **Measured:** T_AR (= 1000·B / AR tok/s of the same process), draft time per round, round time, τ (= all-active tokens / (B·rounds)), S_meas (all-active tok/s ÷ AR tok/s).
- **Derived:** c = T_draft / (γ·T_AR); v = (T_round − T_draft) / T_AR (verify + accept + encode + host); α = root of τ(α, γ) = (1 − α^(γ+1)) / (1 − α) at the row's γ (i.i.d. acceptance model).
- S_model = τ / (γc + v) from the row's own c, v, τ. This equals S_meas by construction (same wall clock), so the error only checks record consistency: max |S_model/S_meas − 1| = 4.4e-16 over 146 rows.
- S_map = background value at (c, α): max over γ ∈ 1..8 of τ(α, γ) / (γc + v0 + v1·γ) with the pooled fit v = 1.098 + 0.0407·γ (max residual 0.085). Regime fits: 32k B16 v = 1.062 + 0.0350·γ (max res 0.020); 16k B16 and 32k B8 v = 1.096 + 0.0506·γ (max res 0.063). S_map can exceed S_meas because it also optimizes γ (and SnapKV was measured at γ ≤ 4 on LongBench).
- Best = max S_meas over all measured budgets / γ / processes of the point (VQ/hybrid: RQ2+r128 and the SnapKV512+RQ2 hybrid, t2w codebook; panel B: default codebook).

## Panel A: SnapKV vs VQ/hybrid t2w (and StreamingLLM-512)

| point | draft (best) | γ | T_AR ms | c | v | τ | α | S_meas | S_map (γ*) | VQ ÷ SnapKV meas |
|---|---|---|---|---|---|---|---|---|---|---|
| syn_quote 32k B16 | SnapKV-4096 | 4 | 26.41 | 0.431 | 1.207 | 4.30 | 0.925 | 1.469 | 1.442 (γ4) |  |
| syn_quote 32k B16 | hybrid SnapKV512+RQ2 [t2w] | 6 | 26.40 | 0.461 | 1.262 | 6.90 | 0.995 | 1.713 | 1.728 (γ8) | **1.167** |
| PG-19 32k B16 | SnapKV-4096 | 6 | 27.05 | 0.418 | 1.259 | 6.10 | 0.954 | 1.619 | 1.584 (γ6) |  |
| PG-19 32k B16 | hybrid SnapKV512+RQ2 [t2w] | 6 | 27.01 | 0.450 | 1.260 | 6.42 | 0.971 | 1.623 | 1.600 (γ8) | **1.002** |
| syn_quote 32k B8 | SnapKV-4096 | 3 | 16.51 | 0.541 | 1.201 | 3.61 | 0.931 | 1.277 | 1.273 (γ4) |  |
| syn_quote 32k B8 | hybrid SnapKV512+RQ2 [t2w] | 6 | 16.48 | 0.587 | 1.338 | 6.88 | 0.994 | 1.414 | 1.435 (γ8) | **1.107** |
| gov_report 32k B8 | StreamingLLM-512 | 3 | 15.28 | 0.508 | 1.221 | 2.54 | 0.701 | 0.924 | 1.033 (γ1) |  |
| gov_report 32k B8 | SnapKV-4096 | 3 | 15.28 | 0.583 | 1.222 | 3.14 | 0.840 | 1.056 | 1.085 (γ2) |  |
| gov_report 32k B8 | RQ2+r128 [t2w] | 3 | 15.28 | 0.622 | 1.233 | 3.54 | 0.920 | 1.144 | 1.149 (γ3) | **1.083** |
| hotpotqa_q 16k B16 | SnapKV-2048 | 4 | 17.70 | 0.572 | 1.306 | 4.30 | 0.925 | 1.197 | 1.216 (γ3) |  |
| hotpotqa_q 16k B16 | hybrid SnapKV512+RQ2 [t2w] | 6 | 17.67 | 0.622 | 1.394 | 6.68 | 0.984 | 1.302 | 1.320 (γ8) | **1.088** |
| gov_report 16k B16 | StreamingLLM-512 | 3 | 17.79 | 0.517 | 1.237 | 2.48 | 0.686 | 0.889 | 1.018 (γ1) |  |
| gov_report 16k B16 | SnapKV-4096 | 3 | 17.79 | 0.639 | 1.235 | 3.22 | 0.856 | 1.021 | 1.054 (γ2) |  |
| gov_report 16k B16 | hybrid SnapKV512+RQ2 [t2w] | 6 | 17.79 | 0.619 | 1.402 | 6.06 | 0.952 | 1.185 | 1.215 (γ4) | **1.161** |
| qmsum 16k B16 | StreamingLLM-512 | 3 | 17.84 | 0.516 | 1.234 | 2.62 | 0.723 | 0.944 | 1.042 (γ1) |  |
| qmsum 16k B16 | SnapKV-2048 | 3 | 17.84 | 0.568 | 1.237 | 3.25 | 0.863 | 1.104 | 1.126 (γ2) |  |
| qmsum 16k B16 | RQ2+r128 [t2w] | 6 | 17.84 | 0.599 | 1.415 | 6.01 | 0.949 | 1.199 | 1.233 (γ4) | **1.085** |
| repobench-p 16k B16 | StreamingLLM-512 | 3 | 17.92 | 0.511 | 1.238 | 3.22 | 0.856 | 1.161 | 1.177 (γ2) |  |
| repobench-p 16k B16 | SnapKV-2048 | 3 | 17.92 | 0.563 | 1.236 | 3.59 | 0.929 | 1.229 | 1.235 (γ3) |  |
| repobench-p 16k B16 | hybrid SnapKV512+RQ2 [t2w] | 6 | 17.92 | 0.614 | 1.393 | 6.49 | 0.975 | 1.279 | 1.292 (γ6) | **1.041** |
| syn_quote 32k B16 gen512 (table only) | SnapKV-4096 | 3 | 26.46 | 0.433 | 1.156 | 3.57 | 0.924 | 1.455 | 1.437 (γ4) |  |
| syn_quote 32k B16 gen512 (table only) | hybrid SnapKV512+RQ2 [t2w] | 4 | 26.46 | 0.465 | 1.208 | 4.92 | 0.992 | 1.604 | 1.695 (γ8) | **1.103** |

## Panel B: SnapKV vs VQ/hybrid default codebook

| point | draft (best) | γ | T_AR ms | c | v | τ | α | S_meas | S_map (γ*) | VQ ÷ SnapKV meas |
|---|---|---|---|---|---|---|---|---|---|---|
| syn_quote 32k B16 | SnapKV-4096 | 4 | 26.41 | 0.431 | 1.207 | 4.30 | 0.925 | 1.469 | 1.442 (γ4) |  |
| syn_quote 32k B16 | hybrid SnapKV512+RQ2 [def] | 6 | 26.40 | 0.461 | 1.262 | 6.60 | 0.981 | 1.641 | 1.631 (γ8) | **1.117** |
| PG-19 32k B16 | SnapKV-4096 | 6 | 27.05 | 0.418 | 1.259 | 6.10 | 0.954 | 1.619 | 1.584 (γ6) |  |
| PG-19 32k B16 | hybrid SnapKV512+RQ2 [def] | 4 | 27.05 | 0.452 | 1.200 | 4.45 | 0.942 | 1.478 | 1.456 (γ5) | **0.913** |
| syn_quote 32k B8 | SnapKV-4096 | 3 | 16.51 | 0.541 | 1.201 | 3.61 | 0.931 | 1.277 | 1.273 (γ4) |  |
| syn_quote 32k B8 | hybrid SnapKV512+RQ2 [def] | 6 | 16.48 | 0.587 | 1.337 | 6.61 | 0.981 | 1.360 | 1.363 (γ8) | **1.065** |
| hotpotqa_q 16k B16 | SnapKV-2048 | 4 | 17.70 | 0.572 | 1.306 | 4.30 | 0.925 | 1.197 | 1.216 (γ3) |  |
| hotpotqa_q 16k B16 | hybrid SnapKV512+RQ2 [def] | 4 | 17.70 | 0.626 | 1.306 | 4.65 | 0.964 | 1.221 | 1.237 (γ5) | **1.020** |
| syn_quote 32k B16 gen512 (table only) | SnapKV-4096 | 3 | 26.46 | 0.433 | 1.156 | 3.57 | 0.924 | 1.455 | 1.437 (γ4) |  |
| syn_quote 32k B16 gen512 (table only) | hybrid SnapKV512+RQ2 [def] | 4 | 26.46 | 0.466 | 1.203 | 4.79 | 0.978 | 1.561 | 1.602 (γ8) | **1.073** |

## Check: does one α predict another γ? (i.i.d.-acceptance model test)

For each draft / process with ≥ 2 measured γ: α from γ_a, τ and S predicted at γ_b with the measured c, v of γ_b, compared with measured S at γ_b. 122 pairs: median |S error| 0.7%, max 5.8%. Positive error = model overpredicts.

| point file | proc | draft | cb | γ_a → γ_b | α(γ_a) | α(γ_b) | τ pred / meas | S pred / meas | error |
|---|---|---|---|---|---|---|---|---|---|
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 3 → 4 | 0.965 | 0.964 | 4.66 / 4.65 | 1.223 / 1.221 | +0.2% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 4 → 3 | 0.964 | 0.965 | 3.79 / 3.79 | 1.216 / 1.218 | -0.1% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.986 | 0.982 | 4.87 / 4.82 | 1.276 / 1.266 | +0.8% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.982 | 0.986 | 3.89 / 3.92 | 1.250 / 1.258 | -0.6% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | SnapKV-2048 | def | 2 → 4 | 0.907 | 0.925 | 4.15 / 4.30 | 1.155 / 1.197 | -3.5% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | SnapKV-2048 | def | 4 → 2 | 0.925 | 0.907 | 2.78 / 2.73 | 1.189 / 1.167 | +1.9% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | RQ2+r128 | def | 3 → 4 | 0.933 | 0.935 | 4.37 / 4.39 | 1.162 / 1.169 | -0.6% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | RQ2+r128 | def | 4 → 3 | 0.935 | 0.933 | 3.63 / 3.61 | 1.181 / 1.176 | +0.4% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | RQ2+r128 | t2w | 3 → 4 | 0.976 | 0.970 | 4.77 / 4.71 | 1.268 / 1.253 | +1.2% |
| v6e2e_hotpotqa_q_ctx16384_B16_gen256 | main | RQ2+r128 | t2w | 4 → 3 | 0.970 | 0.976 | 3.82 / 3.86 | 1.244 / 1.256 | -0.9% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 3 → 4 | 0.932 | 0.942 | 4.36 / 4.45 | 1.449 / 1.478 | -1.9% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 4 → 3 | 0.942 | 0.932 | 3.66 / 3.61 | 1.457 / 1.436 | +1.5% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | SnapKV-4096 | def | 3 → 4 | 0.953 | 0.953 | 4.55 / 4.55 | 1.581 / 1.581 | -0.0% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | SnapKV-4096 | def | 4 → 3 | 0.953 | 0.953 | 3.73 / 3.73 | 1.544 / 1.544 | +0.0% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | RQ2+r128 | def | 3 → 4 | 0.873 | 0.888 | 3.89 / 4.00 | 1.296 / 1.334 | -2.9% |
| v6e2e_pg19_ctx32768_B16_gen256 | main | RQ2+r128 | def | 4 → 3 | 0.888 | 0.873 | 3.38 / 3.30 | 1.350 / 1.320 | +2.3% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.964 | 0.969 | 4.65 / 4.70 | 1.546 / 1.559 | -0.9% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.969 | 0.964 | 3.82 / 3.79 | 1.517 / 1.507 | +0.7% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | SnapKV-4096 | def | 3 → 4 | 0.953 | 0.953 | 4.55 / 4.55 | 1.579 / 1.579 | -0.0% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | SnapKV-4096 | def | 4 → 3 | 0.953 | 0.953 | 3.73 / 3.73 | 1.542 / 1.542 | +0.0% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | RQ2+r128 | t2w | 3 → 4 | 0.937 | 0.939 | 4.41 / 4.43 | 1.468 / 1.475 | -0.5% |
| v6e2e_pg19_ctx32768_B16_gen256 | t | RQ2+r128 | t2w | 4 → 3 | 0.939 | 0.937 | 3.65 / 3.64 | 1.459 / 1.453 | +0.4% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 4 → 6 | 0.983 | 0.981 | 6.66 / 6.60 | 1.651 / 1.638 | +0.8% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | def | 6 → 4 | 0.981 | 0.983 | 4.81 / 4.84 | 1.571 / 1.580 | -0.5% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.994 | 0.995 | 6.88 / 6.90 | 1.707 / 1.711 | -0.2% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.995 | 0.994 | 4.95 / 4.94 | 1.618 / 1.615 | +0.2% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | SnapKV-4096 | def | 3 → 4 | 0.929 | 0.925 | 4.34 / 4.30 | 1.480 / 1.469 | +0.8% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | SnapKV-4096 | def | 4 → 3 | 0.925 | 0.929 | 3.57 / 3.59 | 1.459 / 1.467 | -0.6% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | RQ2+r128 | def | 4 → 6 | 0.967 | 0.961 | 6.34 / 6.24 | 1.594 / 1.569 | +1.6% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | RQ2+r128 | def | 6 → 4 | 0.961 | 0.967 | 4.63 / 4.68 | 1.530 / 1.547 | -1.1% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | RQ2+r128 | t2w | 4 → 6 | 0.986 | 0.985 | 6.71 / 6.70 | 1.689 / 1.686 | +0.1% |
| v6e2e_syn_quote_ctx32768_B16_gen256 | main | RQ2+r128 | t2w | 6 → 4 | 0.985 | 0.986 | 4.85 / 4.86 | 1.606 / 1.608 | -0.1% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | hybrid SnapKV512+RQ2 | def | 4 → 6 | 0.980 | 0.981 | 6.59 / 6.61 | 1.355 / 1.360 | -0.4% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | hybrid SnapKV512+RQ2 | def | 6 → 4 | 0.981 | 0.980 | 4.81 / 4.80 | 1.312 / 1.309 | +0.3% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.992 | 0.994 | 6.83 / 6.88 | 1.406 / 1.414 | -0.6% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.994 | 0.992 | 4.94 / 4.92 | 1.347 / 1.342 | +0.4% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | SnapKV-4096 | def | 3 → 4 | 0.931 | 0.921 | 4.36 / 4.27 | 1.257 / 1.230 | +2.2% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | SnapKV-4096 | def | 4 → 3 | 0.921 | 0.931 | 3.55 / 3.61 | 1.255 / 1.276 | -1.6% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | RQ2+r128 | def | 3 → 4 | 0.970 | 0.971 | 4.71 / 4.72 | 1.295 / 1.298 | -0.2% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | RQ2+r128 | def | 4 → 3 | 0.971 | 0.970 | 3.83 / 3.82 | 1.298 / 1.296 | +0.2% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | RQ2+r128 | t2w | 3 → 4 | 0.989 | 0.991 | 4.89 / 4.91 | 1.346 / 1.351 | -0.4% |
| v6e2e_syn_quote_ctx32768_B8_gen256 | main | RQ2+r128 | t2w | 4 → 3 | 0.991 | 0.989 | 3.95 / 3.94 | 1.337 / 1.333 | +0.3% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6 | hybrid SnapKV512+RQ2 | t2w | 5 → 6 | 0.982 | 0.984 | 6.63 / 6.68 | 1.294 / 1.302 | -0.6% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6 | hybrid SnapKV512+RQ2 | t2w | 6 → 5 | 0.984 | 0.982 | 5.77 / 5.74 | 1.287 / 1.280 | +0.5% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6 | SnapKV-2048 | def | 2 → 4 | 0.907 | 0.925 | 4.15 / 4.30 | 1.155 / 1.197 | -3.5% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6 | SnapKV-2048 | def | 4 → 2 | 0.925 | 0.907 | 2.78 / 2.73 | 1.192 / 1.170 | +1.9% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6s | SnapKV-2048 | def | 5 → 6 | 0.927 | 0.921 | 5.65 / 5.54 | 1.175 / 1.152 | +2.0% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6s | SnapKV-2048 | def | 6 → 5 | 0.921 | 0.927 | 4.93 / 5.01 | 1.168 / 1.188 | -1.7% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6v | RQ2+r128 | t2w | 5 → 6 | 0.972 | 0.975 | 6.43 / 6.49 | 1.276 / 1.288 | -1.0% |
| v7_g6_hotpotqa_q_ctx16384_B16_gen256 | v7g6v | RQ2+r128 | t2w | 6 → 5 | 0.975 | 0.972 | 5.63 / 5.59 | 1.275 / 1.265 | +0.8% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6 | hybrid SnapKV512+RQ2 | t2w | 5 → 6 | 0.971 | 0.971 | 6.41 / 6.42 | 1.620 / 1.623 | -0.2% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6 | hybrid SnapKV512+RQ2 | t2w | 6 → 5 | 0.971 | 0.971 | 5.58 / 5.58 | 1.598 / 1.595 | +0.2% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6 | SnapKV-4096 | def | 3 → 4 | 0.953 | 0.953 | 4.55 / 4.55 | 1.582 / 1.582 | -0.0% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6 | SnapKV-4096 | def | 4 → 3 | 0.953 | 0.953 | 3.73 / 3.73 | 1.545 / 1.545 | +0.0% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6s | SnapKV-4096 | def | 5 → 6 | 0.957 | 0.954 | 6.17 / 6.10 | 1.638 / 1.619 | +1.2% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6s | SnapKV-4096 | def | 6 → 5 | 0.954 | 0.957 | 5.34 / 5.40 | 1.600 / 1.615 | -1.0% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6v | RQ2+r128 | t2w | 5 → 6 | 0.939 | 0.936 | 5.85 / 5.79 | 1.507 / 1.491 | +1.1% |
| v7_g6_pg19_ctx32768_B16_gen256 | v7g6v | RQ2+r128 | t2w | 6 → 5 | 0.936 | 0.939 | 5.11 / 5.16 | 1.491 / 1.504 | -0.9% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.940 | 0.931 | 4.44 / 4.36 | 1.166 / 1.146 | +1.8% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 6 | 0.940 | 0.952 | 5.86 / 6.06 | 1.146 / 1.185 | -3.3% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.931 | 0.940 | 3.61 / 3.66 | 1.160 / 1.176 | -1.3% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.931 | 0.952 | 5.71 / 6.06 | 1.116 / 1.185 | -5.8% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 3 | 0.952 | 0.940 | 3.72 / 3.66 | 1.197 / 1.176 | +1.8% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.952 | 0.931 | 4.54 / 4.36 | 1.194 / 1.146 | +4.2% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 3 → 4 | 0.792 | 0.802 | 3.31 / 3.38 | 0.922 / 0.941 | -2.1% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 4 → 3 | 0.802 | 0.792 | 2.96 / 2.91 | 1.006 / 0.990 | +1.6% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 3 → 4 | 0.856 | 0.859 | 3.75 / 3.77 | 0.970 / 0.975 | -0.6% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 4 → 3 | 0.859 | 0.856 | 3.23 / 3.22 | 1.025 / 1.021 | +0.4% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 4 | 0.936 | 0.934 | 4.40 / 4.39 | 1.170 / 1.166 | +0.4% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 6 | 0.936 | 0.941 | 5.79 / 5.87 | 1.152 / 1.167 | -1.3% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 3 | 0.934 | 0.936 | 3.62 / 3.63 | 1.180 / 1.183 | -0.3% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 6 | 0.934 | 0.941 | 5.76 / 5.87 | 1.146 / 1.167 | -1.8% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 3 | 0.941 | 0.936 | 3.66 / 3.63 | 1.191 / 1.183 | +0.7% |
| v7_gov_report_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 4 | 0.941 | 0.934 | 4.44 / 4.39 | 1.181 / 1.166 | +1.3% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.925 | 0.928 | 4.30 / 4.33 | 1.114 / 1.123 | -0.8% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 6 | 0.925 | 0.929 | 5.60 / 5.68 | 1.090 / 1.106 | -1.4% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.928 | 0.925 | 3.59 / 3.57 | 1.150 / 1.144 | +0.6% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.928 | 0.929 | 5.66 / 5.68 | 1.103 / 1.106 | -0.3% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 3 | 0.929 | 0.925 | 3.60 / 3.57 | 1.152 / 1.144 | +0.7% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.929 | 0.928 | 4.34 / 4.33 | 1.125 / 1.123 | +0.2% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | SnapKV-2048 | def | 3 → 4 | 0.786 | 0.785 | 3.27 / 3.27 | 0.935 / 0.935 | +0.1% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | SnapKV-2048 | def | 4 → 3 | 0.785 | 0.786 | 2.89 / 2.89 | 1.013 / 1.014 | -0.0% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | SnapKV-4096 | def | 3 → 4 | 0.840 | 0.850 | 3.64 / 3.71 | 0.994 / 1.013 | -1.9% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | SnapKV-4096 | def | 4 → 3 | 0.850 | 0.840 | 3.19 / 3.14 | 1.072 / 1.056 | +1.5% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 3 → 4 | 0.920 | 0.921 | 4.26 / 4.27 | 1.112 / 1.115 | -0.3% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 3 → 6 | 0.920 | 0.922 | 5.53 / 5.57 | 1.087 / 1.095 | -0.7% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 4 → 3 | 0.921 | 0.920 | 3.55 / 3.54 | 1.146 / 1.144 | +0.2% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 4 → 6 | 0.921 | 0.922 | 5.55 / 5.57 | 1.091 / 1.095 | -0.4% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 6 → 3 | 0.922 | 0.920 | 3.56 / 3.54 | 1.148 / 1.144 | +0.4% |
| v7_gov_report_ctx32768_B8_gen256 | v7 | RQ2+r128 | t2w | 6 → 4 | 0.922 | 0.921 | 4.28 / 4.27 | 1.118 / 1.115 | +0.3% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.934 | 0.936 | 4.38 / 4.40 | 1.154 / 1.160 | -0.5% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 6 | 0.934 | 0.953 | 5.75 / 6.08 | 1.128 / 1.193 | -5.5% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.936 | 0.934 | 3.63 / 3.62 | 1.172 / 1.168 | +0.4% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.936 | 0.953 | 5.79 / 6.08 | 1.136 / 1.193 | -4.8% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 3 | 0.953 | 0.934 | 3.73 / 3.62 | 1.202 / 1.168 | +3.0% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.953 | 0.936 | 4.55 / 4.40 | 1.200 / 1.160 | +3.4% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 3 → 4 | 0.863 | 0.859 | 3.80 / 3.77 | 1.063 / 1.055 | +0.8% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 4 → 3 | 0.859 | 0.863 | 3.23 / 3.25 | 1.098 / 1.104 | -0.6% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 3 → 4 | 0.903 | 0.905 | 4.12 / 4.14 | 1.069 / 1.072 | -0.3% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 4 → 3 | 0.905 | 0.903 | 3.46 / 3.46 | 1.100 / 1.098 | +0.3% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 4 | 0.940 | 0.934 | 4.43 / 4.38 | 1.185 / 1.171 | +1.2% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 6 | 0.940 | 0.949 | 5.85 / 6.01 | 1.168 / 1.199 | -2.6% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 3 | 0.934 | 0.940 | 3.62 / 3.65 | 1.182 / 1.193 | -0.9% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 6 | 0.934 | 0.949 | 5.75 / 6.01 | 1.148 / 1.199 | -4.2% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 3 | 0.949 | 0.940 | 3.70 / 3.65 | 1.209 / 1.193 | +1.3% |
| v7_qmsum_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 4 | 0.949 | 0.934 | 4.51 / 4.38 | 1.206 / 1.171 | +3.0% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 4 | 0.978 | 0.981 | 4.79 / 4.81 | 1.269 / 1.275 | -0.5% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 3 → 6 | 0.978 | 0.975 | 6.56 / 6.49 | 1.293 / 1.279 | +1.1% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 3 | 0.981 | 0.978 | 3.89 / 3.87 | 1.258 / 1.253 | +0.4% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 4 → 6 | 0.981 | 0.975 | 6.61 / 6.49 | 1.303 / 1.279 | +1.9% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 3 | 0.975 | 0.978 | 3.85 / 3.87 | 1.246 / 1.253 | -0.6% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | hybrid SnapKV512+RQ2 | t2w | 6 → 4 | 0.975 | 0.981 | 4.75 / 4.81 | 1.259 / 1.275 | -1.3% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 3 → 4 | 0.929 | 0.927 | 4.34 / 4.32 | 1.221 / 1.216 | +0.4% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | SnapKV-2048 | def | 4 → 3 | 0.927 | 0.929 | 3.58 / 3.59 | 1.225 / 1.229 | -0.3% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 3 → 4 | 0.957 | 0.961 | 4.59 / 4.62 | 1.196 / 1.204 | -0.7% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | SnapKV-4096 | def | 4 → 3 | 0.961 | 0.957 | 3.77 / 3.75 | 1.202 / 1.196 | +0.5% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 4 | 0.964 | 0.973 | 4.66 / 4.73 | 1.248 / 1.268 | -1.6% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 3 → 6 | 0.964 | 0.968 | 6.29 / 6.36 | 1.259 / 1.272 | -1.0% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 3 | 0.973 | 0.964 | 3.84 / 3.79 | 1.256 / 1.241 | +1.3% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 4 → 6 | 0.973 | 0.968 | 6.45 / 6.36 | 1.291 / 1.272 | +1.4% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 3 | 0.968 | 0.964 | 3.81 / 3.79 | 1.247 / 1.241 | +0.5% |
| v7_repobench-p_ctx16384_B16_gen256 | v7 | RQ2+r128 | t2w | 6 → 4 | 0.968 | 0.973 | 4.69 / 4.73 | 1.256 / 1.268 | -1.0% |

S_map / S_meas − 1 over the plotted markers: median +0.7%, range -2.2% … +3.2%.

