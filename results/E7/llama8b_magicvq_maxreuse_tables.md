# E7 phase 2: reuse the draft's softmax max as a fixed (unified) max in verify attention: gap statistics and best-case time saving

Llama-3.1-8B-Instruct bf16, TP=4 RTX 4090 (GPUs 4-7), e2e_real.py model / drafts / kernels (experiments/26_magicvq/maxreuse.py). Analysis only, no new kernel.

## (a) gap = m_target - m_hat (natural units, score = q.k/sqrt(128))

Per layer x sequence x q-head x verify row x round (active rows), B=4 prompts, gen 64, gamma 4, real prefill, eager spec loop. m_target = max over the exact bf16 cache (verify query q_t, from the verify kernel's per-split maxima). m_hat (q_d) = the draft step's own max (draft query, draft cache), rows 0..gamma-1 (the bonus row has no draft query). m_hat (q_t) = verify query over the draft cache (all gamma+1 rows; bonus row also listed separately). Positive gap: true max above the estimate (exp(s - m_hat) can reach e^gap); negative: estimate too high (sum shrinks by e^gap). m_target row = the distribution of m_target itself.

| task | draft | quantity | n | mean | p1 | p50 | p99 | p99.9 | min | max | # gap > 20 (frac) | # gap < -20 | frac abs(gap) > 5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pg19 | RQ2+r128 g4 | gap, m_hat with q_d | 294912 | 0.095 | -0.377 | 0.066 | 0.888 | 3.018 | -12.48 | 21.57 | 1 (3.4e-06) | 0 | 2.6e-04 |
| pg19 | RQ2+r128 g4 | gap, m_hat with q_t | 368640 | 0.011 | 0.000 | 0.000 | 0.214 | 2.162 | -1.20 | 6.51 | 0 (0.0e+00) | 0 | 4.1e-05 |
| pg19 | RQ2+r128 g4 | gap, q_t, bonus row only | 73728 | 0.009 | 0.000 | 0.000 | 0.137 | 2.022 | -0.95 | 6.51 | 0 (0.0e+00) | 0 | 4.1e-05 |
| pg19 | RQ2+r128 g4 | m_target (value) | 368640 | 3.923 | 1.809 | 3.790 | 7.753 | 22.442 | 0.60 | 63.58 | 622 (1.7e-03) | 0 | 1.0e-01 |
| pg19 | SnapKV512+RQ2 hybrid g4 | gap, m_hat with q_d | 258048 | 0.061 | -0.281 | 0.043 | 0.579 | 2.282 | -10.32 | 21.83 | 1 (3.9e-06) | 0 | 1.6e-04 |
| pg19 | SnapKV512+RQ2 hybrid g4 | gap, m_hat with q_t | 322560 | 0.004 | 0.000 | 0.000 | 0.000 | 1.146 | -0.86 | 6.44 | 0 (0.0e+00) | 0 | 1.2e-05 |
| pg19 | SnapKV512+RQ2 hybrid g4 | gap, q_t, bonus row only | 64512 | 0.004 | 0.000 | 0.000 | 0.000 | 1.237 | -0.83 | 6.44 | 0 (0.0e+00) | 0 | 1.6e-05 |
| pg19 | SnapKV512+RQ2 hybrid g4 | m_target (value) | 322560 | 3.914 | 1.805 | 3.782 | 7.748 | 22.647 | 0.56 | 63.58 | 554 (1.7e-03) | 0 | 9.7e-02 |
| pg19 | SnapKV-4096 g4 | gap, m_hat with q_d | 229376 | 0.135 | -0.096 | 0.123 | 0.498 | 1.397 | -6.18 | 7.56 | 0 (0.0e+00) | 0 | 2.2e-05 |
| pg19 | SnapKV-4096 g4 | gap, m_hat with q_t | 286720 | 0.001 | 0.000 | 0.000 | 0.000 | 0.135 | 0.00 | 4.93 | 0 (0.0e+00) | 0 | 0.0e+00 |
| pg19 | SnapKV-4096 g4 | gap, q_t, bonus row only | 57344 | 0.001 | 0.000 | 0.000 | 0.000 | 0.246 | 0.00 | 4.00 | 0 (0.0e+00) | 0 | 0.0e+00 |
| pg19 | SnapKV-4096 g4 | m_target (value) | 286720 | 3.916 | 1.808 | 3.785 | 7.732 | 22.742 | 0.58 | 62.88 | 491 (1.7e-03) | 0 | 9.6e-02 |
| syn_quote | RQ2+r128 g4 | gap, m_hat with q_d | 225280 | 0.081 | -1.079 | 0.032 | 2.177 | 5.416 | -12.45 | 9.92 | 0 (0.0e+00) | 0 | 1.5e-03 |
| syn_quote | RQ2+r128 g4 | gap, m_hat with q_t | 281600 | 0.058 | 0.000 | 0.000 | 2.100 | 4.376 | -2.15 | 8.37 | 0 (0.0e+00) | 0 | 5.8e-04 |
| syn_quote | RQ2+r128 g4 | gap, q_t, bonus row only | 56320 | 0.062 | 0.000 | 0.000 | 2.160 | 4.379 | -1.52 | 7.44 | 0 (0.0e+00) | 0 | 4.8e-04 |
| syn_quote | RQ2+r128 g4 | m_target (value) | 281600 | 3.968 | 1.815 | 3.794 | 8.748 | 22.156 | -0.09 | 42.11 | 463 (1.6e-03) | 0 | 1.1e-01 |
| syn_quote | SnapKV512+RQ2 hybrid g4 | gap, m_hat with q_d | 212992 | 0.055 | -0.953 | 0.026 | 1.595 | 4.497 | -8.46 | 9.90 | 0 (0.0e+00) | 0 | 8.5e-04 |
| syn_quote | SnapKV512+RQ2 hybrid g4 | gap, m_hat with q_t | 266240 | 0.024 | 0.000 | 0.000 | 0.895 | 3.563 | -2.21 | 8.05 | 0 (0.0e+00) | 0 | 2.4e-04 |
| syn_quote | SnapKV512+RQ2 hybrid g4 | gap, q_t, bonus row only | 53248 | 0.026 | 0.000 | 0.000 | 1.011 | 3.655 | -2.21 | 6.24 | 0 (0.0e+00) | 0 | 3.4e-04 |
| syn_quote | SnapKV512+RQ2 hybrid g4 | m_target (value) | 266240 | 3.971 | 1.817 | 3.797 | 8.768 | 22.130 | -0.09 | 42.11 | 436 (1.6e-03) | 0 | 1.1e-01 |
| syn_quote | SnapKV-4096 g4 | gap, m_hat with q_d | 217088 | 0.171 | -0.840 | 0.120 | 1.769 | 5.547 | -9.42 | 12.64 | 0 (0.0e+00) | 0 | 1.5e-03 |
| syn_quote | SnapKV-4096 g4 | gap, m_hat with q_t | 271360 | 0.012 | 0.000 | 0.000 | 0.000 | 3.024 | 0.00 | 9.80 | 0 (0.0e+00) | 0 | 2.8e-04 |
| syn_quote | SnapKV-4096 g4 | gap, q_t, bonus row only | 54272 | 0.013 | 0.000 | 0.000 | 0.000 | 3.131 | 0.00 | 8.25 | 0 (0.0e+00) | 0 | 3.7e-04 |
| syn_quote | SnapKV-4096 g4 | m_target (value) | 271360 | 3.974 | 1.817 | 3.799 | 8.786 | 22.130 | -0.09 | 45.44 | 445 (1.6e-03) | 0 | 1.1e-01 |

Co-tenant flags (stats runs): stats_mr1_pg19_ctx32768_B4.json: False, stats_mr1_syn_quote_ctx32768_B4.json: False

## (b) best-case saving in the verify step (syn_quote 32k prompt lengths, B=16, TP=4)

Verify attention = split-K flash-decoding partial kernel (_bf_partial, online max + alpha rescale inside the kernel) + a SEPARATE log-sum-exp merge kernel (_combine) per layer. A unified max would remove the merge's rescale (exp2(m_i - m_max) weights; a sum across splits is still needed) and the in-kernel alpha rescale. Upper bound = whole _combine time + (partial - fixed-max partial). Fixed-max partial = the same kernel with the running max / alpha removed (timing probe, cache contents random). Times: CUDA-graph replay, median of 20 (min-max); profile: eager, torch.profiler, 5 steps (NCCL time in eager includes rank skew, so shares are of the eager GPU total).

| gamma | verify plan (splits, chunk) | verify step ms (graph) | attention all layers ms | partial ms | fixed-max partial ms | combine ms | upper-bound saving ms | share of verify | eager profile share: partial / combine / NCCL / GEMM / other |
|---|---|---|---|---|---|---|---|---|---|
| 3 | [4, 8128] | 30.97 (30.93-31.07) | 17.69 | 17.56 | 17.56 | 0.098 | 0.098 | 0.32% | 46.6% / 0.23% / 36.1% / 13.1% / 4.0% |
| 4 | [4, 8128] | 32.26 (32.24-32.30) | 18.06 | 17.76 | 17.70 | 0.108 | 0.172 | 0.53% | 55.6% / 0.30% / 23.8% / 15.5% / 4.8% |

Co-tenant flag (prof run): False
