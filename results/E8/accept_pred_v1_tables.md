# E8a accept_pred v1: is per-sequence draft acceptance predictable at prefill? (go/no-go)

- sequences: 327 with features, 326 kept (tf_agree >= 0.97 and prompt length = record); excluded: [['repobench-p', 16384, 'bc83c326681f82acc4301980ec122cafad03de6b8a4d1ae5', 0.969]]
- tau = tau(gamma=4) per sequence from the E7 lb_tau records (v4/v4b/v6); features from one target prefill (observation window = last 32 prompt queries) and, as a non-causal control, from the target's attention at the generated positions ('oracle').
- within-task = Pearson of task-demeaned within-task ranks (pooled); mean task rho = n-weighted mean of per-task Spearman (tasks with n >= 6).

## Correlations, ctx 16384, prefill features

| feature | tau4_snapkv4096 | alpha1_snapkv4096 | tau4_hyb512pL2 | tau4_vqpL2+r128 | logratio4_hyb_vs_snap4096 | logratio4_vq_vs_snap4096 | logratio4_hybt2w_vs_snap4096 |
|---|---|---|---|---|---|---|---|
| obs_cov4096_mean | +0.19 / +0.07 / +0.07 | +0.19 / +0.05 / +0.04 | +0.20 / +0.05 / +0.05 | +0.15 / -0.02 / -0.02 | +0.11 / +0.06 / +0.06 | +0.07 / +0.02 / +0.02 | +0.08 / +0.01 / +0.01 |
| obs_cov512_mean | +0.22 / +0.08 / +0.08 | +0.25 / +0.06 / +0.06 | +0.34 / +0.09 / +0.09 | +0.30 / +0.03 / +0.03 | +0.19 / +0.09 / +0.09 | +0.14 / +0.04 / +0.04 | -0.08 / -0.11 / -0.11 |
| obs_far4096_mean | +0.20 / +0.08 / +0.07 | +0.20 / +0.05 / +0.05 | +0.20 / +0.05 / +0.06 | +0.15 / -0.02 / -0.02 | +0.10 / +0.05 / +0.05 | +0.06 / +0.00 / +0.00 | +0.09 / +0.00 / +0.00 |
| obs_covho4096_mean | +0.20 / +0.05 / +0.05 | +0.24 / +0.05 / +0.05 | +0.37 / +0.04 / +0.04 | +0.37 / +0.01 / +0.01 | +0.25 / +0.08 / +0.08 | +0.22 / +0.02 / +0.02 | -0.10 / -0.02 / -0.02 |
| obs_cov4096_q10 | +0.03 / +0.06 / +0.06 | +0.04 / +0.05 / +0.05 | +0.09 / +0.09 / +0.10 | +0.02 / +0.00 / +0.00 | +0.17 / +0.10 / +0.10 | +0.11 / +0.03 / +0.03 | +0.06 / -0.00 / -0.00 |
| obs_ent_mean | +0.01 / -0.08 / -0.07 | -0.04 / -0.07 / -0.07 | -0.13 / -0.14 / -0.15 | -0.19 / -0.06 / -0.06 | -0.23 / -0.10 / -0.10 | -0.29 / -0.05 / -0.05 | +0.05 / +0.10 / +0.10 |
| obs_logn90_mean | -0.29 / -0.13 / -0.13 | -0.31 / -0.12 / -0.11 | -0.32 / -0.09 / -0.09 | -0.31 / -0.05 / -0.05 | -0.08 / -0.02 / -0.02 | -0.08 / +0.00 / +0.00 | +0.10 / +0.12 / +0.12 |
| obs_mfar128_mean | -0.66 / -0.09 / -0.09 | -0.68 / -0.09 / -0.09 | -0.72 / -0.01 / -0.01 | -0.68 / -0.05 / -0.05 | +0.08 / +0.05 / +0.05 | +0.09 / +0.05 / +0.05 | +0.34 / +0.08 / +0.08 |
| obs_o_snap4096_mean | -0.34 / -0.07 / -0.07 | -0.34 / -0.05 / -0.05 | -0.32 / -0.04 / -0.05 | -0.27 / +0.03 / +0.03 | -0.03 / -0.04 / -0.04 | -0.01 / +0.01 / +0.01 | -0.03 / -0.07 / -0.07 |
| obs_o_snap512_mean | -0.36 / -0.11 / -0.11 | -0.38 / -0.08 / -0.08 | -0.47 / -0.07 / -0.07 | -0.43 / -0.01 / -0.01 | -0.13 / -0.02 / -0.02 | -0.11 / +0.01 / +0.01 | +0.15 / +0.07 / +0.07 |
| obs_o_vq_def_mean | -0.30 / +0.08 / +0.08 | -0.34 / +0.03 / +0.03 | -0.53 / -0.05 / -0.05 | -0.56 / -0.08 / -0.08 | -0.17 / -0.16 / -0.16 | -0.21 / -0.17 / -0.17 | +0.15 / -0.12 / -0.12 |
| obs_o_hyb_def_mean | -0.14 / -0.07 / -0.07 | -0.19 / -0.03 / -0.04 | -0.37 / -0.07 / -0.07 | -0.30 / +0.00 / +0.00 | -0.28 / -0.11 / -0.11 | -0.20 / -0.04 / -0.04 | +0.14 / +0.07 / +0.07 |
| obs_o_hyb_t2w_mean | -0.08 / -0.03 / -0.04 | -0.13 / -0.01 / -0.01 | -0.35 / -0.06 / -0.07 | -0.26 / -0.02 / -0.02 | -0.32 / -0.11 / -0.11 | -0.22 / -0.08 / -0.08 | +0.14 / +0.03 / +0.03 |
| obs_d_snap4096_hyb_def | -0.09 / +0.03 / +0.03 | -0.03 / +0.03 / +0.03 | +0.23 / +0.09 / +0.09 | +0.19 / +0.05 / +0.05 | +0.37 / +0.14 / +0.13 | +0.30 / +0.09 / +0.09 | -0.19 / -0.11 / -0.11 |
| obs_d_snap4096_vq_def | +0.16 / -0.08 / -0.09 | +0.21 / -0.03 / -0.03 | +0.43 / +0.04 / +0.04 | +0.49 / +0.08 / +0.08 | +0.18 / +0.15 / +0.15 | +0.25 / +0.15 / +0.15 | -0.18 / +0.09 / +0.09 |
| P | -0.33 / +0.08 / +0.07 | -0.35 / +0.11 / +0.11 | -0.44 / +0.06 / +0.06 | -0.46 / -0.02 / -0.02 | -0.04 / -0.03 / -0.03 | -0.07 / -0.05 / -0.05 | +0.11 / -0.12 / -0.12 |

cells: pooled Spearman / within-task (rank, pooled) / mean per-task Spearman

## Correlations, ctx 16384, oracle features

| feature | tau4_snapkv4096 | alpha1_snapkv4096 | tau4_hyb512pL2 | tau4_vqpL2+r128 | logratio4_hyb_vs_snap4096 | logratio4_vq_vs_snap4096 | logratio4_hybt2w_vs_snap4096 |
|---|---|---|---|---|---|---|---|
| orc_cov4096_mean | +0.88 / +0.76 / +0.76 | +0.86 / +0.74 / +0.73 | +0.65 / +0.47 / +0.47 | +0.51 / +0.22 / +0.22 | -0.44 / -0.37 / -0.37 | -0.52 / -0.53 / -0.53 | -0.80 / -0.76 / -0.76 |
| orc_cov512_mean | +0.71 / +0.68 / +0.68 | +0.70 / +0.64 / +0.63 | +0.57 / +0.49 / +0.49 | +0.46 / +0.28 / +0.28 | -0.27 / -0.25 / -0.25 | -0.35 / -0.39 / -0.39 | -0.71 / -0.67 / -0.67 |
| orc_far4096_mean | +0.88 / +0.75 / +0.74 | +0.87 / +0.72 / +0.71 | +0.66 / +0.44 / +0.44 | +0.52 / +0.19 / +0.19 | -0.44 / -0.38 / -0.38 | -0.52 / -0.55 / -0.55 | -0.79 / -0.74 / -0.74 |
| orc_cov4096_q10 | +0.90 / +0.78 / +0.77 | +0.88 / +0.73 / +0.72 | +0.67 / +0.46 / +0.45 | +0.54 / +0.22 / +0.22 | -0.44 / -0.40 / -0.39 | -0.51 / -0.54 / -0.54 | -0.79 / -0.75 / -0.75 |
| orc_ent_mean | -0.43 / -0.42 / -0.41 | -0.43 / -0.40 / -0.39 | -0.35 / -0.40 / -0.40 | -0.26 / -0.25 / -0.25 | +0.13 / +0.09 / +0.09 | +0.22 / +0.20 / +0.20 | +0.48 / +0.45 / +0.45 |
| orc_logn90_mean | -0.24 / -0.47 / -0.47 | -0.22 / -0.44 / -0.43 | -0.06 / -0.35 / -0.35 | +0.07 / -0.22 / -0.22 | +0.17 / +0.15 / +0.15 | +0.32 / +0.24 / +0.24 | +0.34 / +0.48 / +0.48 |
| orc_mfar128_mean | -0.01 / -0.26 / -0.26 | -0.02 / -0.28 / -0.28 | -0.11 / -0.21 / -0.21 | +0.02 / -0.08 / -0.08 | -0.05 / +0.07 / +0.07 | +0.10 / +0.15 / +0.15 | +0.31 / +0.32 / +0.32 |
| orc_o_snap4096_mean | -0.90 / -0.79 / -0.78 | -0.87 / -0.76 / -0.75 | -0.61 / -0.44 / -0.44 | -0.47 / -0.21 / -0.21 | +0.50 / +0.41 / +0.41 | +0.58 / +0.56 / +0.56 | +0.85 / +0.81 / +0.81 |
| orc_o_snap512_mean | -0.80 / -0.72 / -0.72 | -0.77 / -0.68 / -0.68 | -0.60 / -0.48 / -0.48 | -0.44 / -0.24 / -0.24 | +0.37 / +0.29 / +0.29 | +0.48 / +0.44 / +0.44 | +0.73 / +0.70 / +0.70 |
| orc_o_vq_def_mean | +0.38 / -0.16 / -0.15 | +0.39 / -0.18 / -0.18 | +0.31 / -0.30 / -0.30 | +0.27 / -0.31 / -0.31 | -0.06 / -0.02 / -0.02 | -0.09 / -0.03 / -0.03 | +0.07 / +0.21 / +0.21 |
| orc_o_hyb_def_mean | -0.63 / -0.65 / -0.64 | -0.61 / -0.61 / -0.61 | -0.56 / -0.48 / -0.48 | -0.38 / -0.27 / -0.27 | +0.22 / +0.23 / +0.23 | +0.37 / +0.36 / +0.36 | +0.68 / +0.65 / +0.65 |
| orc_o_hyb_t2w_mean | -0.50 / -0.64 / -0.64 | -0.46 / -0.60 / -0.59 | -0.34 / -0.45 / -0.45 | -0.13 / -0.24 / -0.24 | +0.27 / +0.24 / +0.24 | +0.44 / +0.36 / +0.36 | +0.61 / +0.68 / +0.68 |
| orc_d_snap4096_hyb_def | -0.69 / -0.42 / -0.40 | -0.69 / -0.44 / -0.43 | -0.31 / +0.02 / +0.02 | -0.30 / +0.19 / +0.19 | +0.63 / +0.51 / +0.51 | +0.57 / +0.60 / +0.60 | +0.74 / +0.52 / +0.52 |
| orc_d_snap4096_vq_def | -0.90 / -0.66 / -0.65 | -0.89 / -0.63 / -0.62 | -0.64 / -0.23 / -0.23 | -0.50 / +0.01 / +0.01 | +0.47 / +0.46 / +0.46 | +0.54 / +0.60 / +0.60 | +0.79 / +0.65 / +0.65 |

cells: pooled Spearman / within-task (rank, pooled) / mean per-task Spearman

### ctx 16384: strongest prefill features by |mean per-task Spearman|

| target | feature | pooled | within (ranks) | mean task rho | per task |
|---|---|---|---|---|---|
| tau4_snapkv4096 | obs_logn90_mean | -0.29 | -0.13 | -0.13 | {'gov_report': 0.273, 'hotpotqa_q': -0.193, 'lcc': -0.389, 'multi_news': -0.359, 'qmsum': 0.117, 'repobench-p': -0.262, 'syn_cont': -0.507, 'syn_kv': 0.353, 'syn_quote': -0.336} |
| tau4_snapkv4096 | obs_o_snap512_mean | -0.36 | -0.11 | -0.11 | {'gov_report': 0.226, 'hotpotqa_q': -0.189, 'lcc': -0.388, 'multi_news': -0.306, 'qmsum': 0.181, 'repobench-p': -0.232, 'syn_cont': -0.503, 'syn_kv': 0.165, 'syn_quote': -0.149} |
| tau4_snapkv4096 | obs_far512_mean | +0.24 | +0.10 | +0.09 | {'gov_report': -0.205, 'hotpotqa_q': 0.165, 'lcc': 0.226, 'multi_news': 0.161, 'qmsum': -0.146, 'repobench-p': 0.232, 'syn_cont': 0.553, 'syn_kv': -0.374, 'syn_quote': 0.281} |
| alpha1_snapkv4096 | obs_logn90_mean | -0.31 | -0.12 | -0.11 | {'gov_report': 0.249, 'hotpotqa_q': -0.188, 'lcc': -0.281, 'multi_news': -0.347, 'qmsum': 0.133, 'repobench-p': -0.306, 'syn_cont': -0.511, 'syn_kv': 0.388, 'syn_quote': -0.284} |
| alpha1_snapkv4096 | obs_mfar128_mean | -0.68 | -0.09 | -0.09 | {'gov_report': -0.268, 'hotpotqa_q': 0.163, 'lcc': -0.017, 'multi_news': -0.293, 'qmsum': -0.352, 'repobench-p': 0.009, 'syn_cont': -0.446, 'syn_kv': 0.302, 'syn_quote': -0.033} |
| alpha1_snapkv4096 | obs_o_snap512_mean | -0.38 | -0.08 | -0.08 | {'gov_report': 0.202, 'hotpotqa_q': -0.16, 'lcc': -0.267, 'multi_news': -0.282, 'qmsum': 0.271, 'repobench-p': -0.271, 'syn_cont': -0.484, 'syn_kv': 0.28, 'syn_quote': -0.129} |
| tau4_hyb512pL2 | obs_ent_mean | -0.13 | -0.14 | -0.15 | {'gov_report': 0.18, 'hotpotqa_q': -0.051, 'lcc': -0.31, 'multi_news': 0.012, 'qmsum': 0.062, 'repobench-p': -0.507, 'syn_cont': -0.661, 'syn_kv': -0.332, 'syn_quote': -0.146} |
| tau4_hyb512pL2 | obs_far512_mean | +0.35 | +0.10 | +0.10 | {'gov_report': -0.155, 'hotpotqa_q': 0.066, 'lcc': 0.217, 'multi_news': -0.053, 'qmsum': -0.028, 'repobench-p': 0.271, 'syn_cont': 0.717, 'syn_kv': 0.212, 'syn_quote': 0.055} |
| tau4_hyb512pL2 | obs_cov4096_q10 | +0.09 | +0.09 | +0.10 | {'gov_report': -0.158, 'hotpotqa_q': -0.032, 'lcc': 0.17, 'multi_news': 0.049, 'qmsum': -0.042, 'repobench-p': 0.423, 'syn_cont': 0.631, 'syn_kv': 0.279, 'syn_quote': 0.016} |
| tau4_vqpL2+r128 | obs_o_vq_def_mean | -0.56 | -0.08 | -0.08 | {'gov_report': 0.166, 'hotpotqa_q': -0.058, 'lcc': -0.361, 'multi_news': 0.129, 'qmsum': -0.04, 'repobench-p': -0.083, 'syn_cont': -0.395, 'syn_kv': 0.202, 'syn_quote': -0.334} |
| tau4_vqpL2+r128 | obs_d_snap4096_vq_def | +0.49 | +0.08 | +0.08 | {'gov_report': -0.106, 'hotpotqa_q': 0.204, 'lcc': 0.331, 'multi_news': -0.183, 'qmsum': -0.3, 'repobench-p': 0.206, 'syn_cont': 0.369, 'syn_kv': -0.211, 'syn_quote': 0.266} |
| tau4_vqpL2+r128 | obs_o_vq_t2w_mean | -0.58 | -0.06 | -0.06 | {'gov_report': 0.18, 'hotpotqa_q': -0.055, 'lcc': -0.398, 'multi_news': 0.149, 'qmsum': -0.02, 'repobench-p': -0.197, 'syn_cont': -0.44, 'syn_kv': 0.401, 'syn_quote': -0.245} |
| logratio4_hyb_vs_snap4096 | obs_o_vq_def_q90 | -0.30 | -0.20 | -0.20 | {'gov_report': -0.032, 'hotpotqa_q': -0.181, 'lcc': -0.156, 'multi_news': 0.404, 'qmsum': -0.128, 'repobench-p': -0.507, 'syn_cont': -0.655, 'syn_kv': -0.451, 'syn_quote': -0.251} |
| logratio4_hyb_vs_snap4096 | obs_e_vq_def_mean | -0.10 | -0.17 | -0.17 | {'gov_report': -0.032, 'hotpotqa_q': -0.198, 'lcc': -0.085, 'multi_news': 0.243, 'qmsum': -0.124, 'repobench-p': -0.358, 'syn_cont': -0.313, 'syn_kv': -0.255, 'syn_quote': -0.354} |
| logratio4_hyb_vs_snap4096 | obs_o_vq_def_mean | -0.17 | -0.16 | -0.16 | {'gov_report': -0.005, 'hotpotqa_q': -0.263, 'lcc': -0.1, 'multi_news': 0.35, 'qmsum': -0.009, 'repobench-p': -0.405, 'syn_cont': -0.42, 'syn_kv': -0.313, 'syn_quote': -0.272} |
| logratio4_vq_vs_snap4096 | obs_o_vq_def_mean | -0.21 | -0.17 | -0.17 | {'gov_report': -0.093, 'hotpotqa_q': -0.396, 'lcc': -0.278, 'multi_news': 0.188, 'qmsum': 0.261, 'repobench-p': -0.216, 'syn_cont': -0.378, 'syn_kv': -0.091, 'syn_quote': -0.284} |
| logratio4_vq_vs_snap4096 | obs_d_snap4096_vq_def | +0.25 | +0.15 | +0.15 | {'gov_report': 0.047, 'hotpotqa_q': 0.449, 'lcc': 0.263, 'multi_news': -0.126, 'qmsum': -0.416, 'repobench-p': 0.229, 'syn_cont': 0.386, 'syn_kv': 0.039, 'syn_quote': 0.231} |
| logratio4_vq_vs_snap4096 | obs_o_vq_def_q90 | -0.33 | -0.13 | -0.13 | {'gov_report': -0.109, 'hotpotqa_q': -0.168, 'lcc': -0.357, 'multi_news': 0.195, 'qmsum': 0.137, 'repobench-p': -0.256, 'syn_cont': -0.548, 'syn_kv': 0.316, 'syn_quote': -0.277} |
| logratio4_hybt2w_vs_snap4096 | obs_e_vq_def_mean | +0.19 | -0.15 | -0.15 | {'gov_report': 0.042, 'hotpotqa_q': -0.296, 'syn_quote': -0.193} |
| logratio4_hybt2w_vs_snap4096 | obs_far512_mean | -0.08 | -0.13 | -0.13 | {'gov_report': 0.071, 'hotpotqa_q': -0.145, 'syn_quote': -0.324} |
| logratio4_hybt2w_vs_snap4096 | obs_o_vq_t2w_mean | +0.19 | -0.13 | -0.13 | {'gov_report': 0.09, 'hotpotqa_q': -0.372, 'syn_quote': -0.105} |

## Correlations, ctx 32768, prefill features

| feature | tau4_snapkv4096 | alpha1_snapkv4096 | tau4_hyb512pL2 | tau4_vqpL2+r128 | logratio4_hyb_vs_snap4096 | logratio4_vq_vs_snap4096 | logratio4_hybt2w_vs_snap4096 |
|---|---|---|---|---|---|---|---|
| obs_cov4096_mean | +0.34 / +0.25 / +0.27 | +0.36 / +0.26 / +0.25 | +0.34 / +0.07 / +0.07 | +0.20 / +0.01 / +0.01 | +0.01 / -0.09 / -0.09 | -0.10 / -0.15 / -0.15 | -0.11 / -0.13 / -0.13 |
| obs_cov512_mean | +0.24 / +0.18 / +0.20 | +0.27 / +0.20 / +0.19 | +0.58 / +0.10 / +0.09 | +0.50 / -0.02 / -0.02 | +0.25 / -0.01 / -0.01 | +0.17 / -0.12 / -0.12 | -0.22 / -0.11 / -0.11 |
| obs_far4096_mean | +0.36 / +0.24 / +0.26 | +0.37 / +0.24 / +0.24 | +0.34 / +0.07 / +0.06 | +0.20 / +0.02 / +0.01 | +0.01 / -0.09 / -0.10 | -0.11 / -0.15 / -0.15 | -0.09 / -0.12 / -0.12 |
| obs_covho4096_mean | +0.12 / +0.04 / +0.05 | +0.17 / +0.06 / +0.07 | +0.59 / +0.05 / +0.04 | +0.58 / +0.01 / +0.01 | +0.37 / +0.12 / +0.12 | +0.31 / +0.05 / +0.05 | +0.01 / +0.21 / +0.21 |
| obs_cov4096_q10 | +0.14 / +0.22 / +0.24 | +0.15 / +0.21 / +0.20 | +0.34 / +0.13 / +0.13 | +0.16 / +0.02 / +0.02 | +0.20 / -0.01 / -0.01 | +0.04 / -0.12 / -0.11 | -0.02 / -0.08 / -0.08 |
| obs_ent_mean | +0.21 / -0.09 / -0.10 | +0.14 / -0.17 / -0.17 | -0.25 / -0.03 / -0.02 | -0.40 / +0.04 / +0.04 | -0.42 / -0.05 / -0.05 | -0.46 / +0.06 / +0.06 | +0.13 / +0.03 / +0.03 |
| obs_logn90_mean | -0.41 / -0.28 / -0.30 | -0.44 / -0.31 / -0.30 | -0.44 / -0.16 / -0.15 | -0.39 / -0.05 / -0.05 | +0.02 / +0.07 / +0.07 | +0.05 / +0.15 / +0.15 | +0.27 / +0.25 / +0.25 |
| obs_mfar128_mean | -0.51 / -0.04 / -0.05 | -0.56 / -0.04 / -0.03 | -0.61 / -0.16 / -0.16 | -0.54 / +0.04 / +0.04 | +0.06 / -0.01 / -0.01 | +0.11 / +0.13 / +0.13 | +0.14 / +0.09 / +0.09 |
| obs_o_snap4096_mean | -0.47 / -0.24 / -0.26 | -0.48 / -0.24 / -0.24 | -0.41 / -0.08 / -0.08 | -0.23 / -0.02 / -0.02 | +0.08 / +0.09 / +0.09 | +0.23 / +0.16 / +0.16 | +0.09 / +0.09 / +0.09 |
| obs_o_snap512_mean | -0.34 / -0.23 / -0.25 | -0.39 / -0.24 / -0.24 | -0.66 / -0.17 / -0.17 | -0.57 / -0.04 / -0.04 | -0.19 / +0.03 / +0.03 | -0.10 / +0.13 / +0.13 | +0.23 / +0.18 / +0.18 |
| obs_o_vq_def_mean | +0.10 / +0.14 / +0.14 | +0.03 / +0.07 / +0.06 | -0.54 / -0.19 / -0.19 | -0.59 / -0.10 / -0.10 | -0.47 / -0.24 / -0.25 | -0.45 / -0.13 / -0.13 | -0.03 / -0.25 / -0.25 |
| obs_o_hyb_def_mean | -0.13 / -0.06 / -0.07 | -0.17 / -0.09 / -0.10 | -0.59 / -0.15 / -0.15 | -0.52 / -0.02 / -0.02 | -0.35 / -0.05 / -0.04 | -0.27 / +0.07 / +0.07 | +0.18 / +0.08 / +0.08 |
| obs_o_hyb_t2w_mean | -0.06 / +0.03 / +0.02 | -0.10 / -0.02 / -0.03 | -0.60 / -0.13 / -0.13 | -0.49 / +0.01 / +0.01 | -0.42 / -0.18 / -0.17 | -0.30 / -0.01 / -0.01 | -0.04 / -0.22 / -0.22 |
| obs_d_snap4096_hyb_def | -0.28 / -0.15 / -0.16 | -0.22 / -0.11 / -0.11 | +0.32 / +0.06 / +0.05 | +0.39 / -0.11 / -0.11 | +0.48 / +0.13 / +0.13 | +0.49 / +0.03 / +0.03 | -0.04 / +0.01 / +0.01 |
| obs_d_snap4096_vq_def | -0.26 / -0.19 / -0.20 | -0.19 / -0.12 / -0.10 | +0.36 / +0.15 / +0.15 | +0.49 / +0.04 / +0.04 | +0.46 / +0.26 / +0.26 | +0.49 / +0.13 / +0.12 | +0.05 / +0.24 / +0.24 |
| P | +0.21 / +0.13 / +0.14 | +0.23 / +0.10 / +0.08 | +0.26 / -0.01 / -0.02 | +0.26 / +0.03 / +0.03 | -0.02 / -0.05 / -0.05 | -0.02 / +0.00 / +0.00 | -0.11 / -0.06 / -0.05 |

cells: pooled Spearman / within-task (rank, pooled) / mean per-task Spearman

## Correlations, ctx 32768, oracle features

| feature | tau4_snapkv4096 | alpha1_snapkv4096 | tau4_hyb512pL2 | tau4_vqpL2+r128 | logratio4_hyb_vs_snap4096 | logratio4_vq_vs_snap4096 | logratio4_hybt2w_vs_snap4096 |
|---|---|---|---|---|---|---|---|
| orc_cov4096_mean | +0.89 / +0.72 / +0.76 | +0.88 / +0.70 / +0.68 | +0.34 / +0.21 / +0.20 | +0.16 / +0.16 / +0.17 | -0.63 / -0.50 / -0.50 | -0.74 / -0.56 / -0.56 | -0.78 / -0.78 / -0.78 |
| orc_cov512_mean | +0.72 / +0.57 / +0.60 | +0.73 / +0.57 / +0.55 | +0.32 / +0.24 / +0.23 | +0.16 / +0.22 / +0.23 | -0.44 / -0.32 / -0.32 | -0.54 / -0.39 / -0.39 | -0.64 / -0.65 / -0.65 |
| orc_far4096_mean | +0.89 / +0.69 / +0.73 | +0.89 / +0.69 / +0.67 | +0.38 / +0.22 / +0.21 | +0.20 / +0.18 / +0.18 | -0.60 / -0.47 / -0.47 | -0.71 / -0.54 / -0.54 | -0.78 / -0.76 / -0.76 |
| orc_cov4096_q10 | +0.91 / +0.74 / +0.78 | +0.91 / +0.72 / +0.70 | +0.42 / +0.24 / +0.23 | +0.24 / +0.20 / +0.20 | -0.60 / -0.51 / -0.51 | -0.70 / -0.56 / -0.56 | -0.77 / -0.81 / -0.81 |
| orc_ent_mean | -0.39 / -0.28 / -0.29 | -0.39 / -0.33 / -0.32 | -0.11 / -0.21 / -0.21 | -0.03 / -0.24 / -0.24 | +0.23 / +0.04 / +0.04 | +0.26 / +0.11 / +0.11 | +0.15 / +0.23 / +0.23 |
| orc_logn90_mean | -0.41 / -0.41 / -0.44 | -0.39 / -0.45 / -0.43 | +0.01 / -0.19 / -0.19 | +0.13 / -0.21 / -0.22 | +0.36 / +0.20 / +0.20 | +0.39 / +0.26 / +0.26 | +0.21 / +0.37 / +0.37 |
| orc_mfar128_mean | -0.02 / -0.21 / -0.22 | +0.02 / -0.17 / -0.16 | -0.13 / -0.08 / -0.08 | +0.07 / -0.12 / -0.13 | -0.02 / +0.28 / +0.28 | +0.05 / +0.13 / +0.13 | +0.34 / +0.43 / +0.43 |
| orc_o_snap4096_mean | -0.91 / -0.74 / -0.78 | -0.89 / -0.72 / -0.71 | -0.30 / -0.18 / -0.17 | -0.10 / -0.16 / -0.16 | +0.70 / +0.56 / +0.55 | +0.79 / +0.57 / +0.57 | +0.83 / +0.83 / +0.83 |
| orc_o_snap512_mean | -0.85 / -0.65 / -0.68 | -0.84 / -0.62 / -0.60 | -0.32 / -0.26 / -0.26 | -0.10 / -0.22 / -0.23 | +0.64 / +0.40 / +0.40 | +0.74 / +0.45 / +0.45 | +0.75 / +0.78 / +0.78 |
| orc_o_vq_def_mean | +0.56 / +0.05 / +0.05 | +0.60 / +0.10 / +0.11 | +0.34 / -0.23 / -0.23 | +0.24 / -0.35 / -0.35 | -0.24 / -0.03 / -0.03 | -0.33 / -0.20 / -0.20 | -0.04 / +0.13 / +0.13 |
| orc_o_hyb_def_mean | -0.68 / -0.54 / -0.57 | -0.66 / -0.49 / -0.47 | -0.38 / -0.36 / -0.35 | -0.12 / -0.31 / -0.31 | +0.44 / +0.30 / +0.30 | +0.56 / +0.32 / +0.32 | +0.68 / +0.70 / +0.70 |
| orc_o_hyb_t2w_mean | -0.59 / -0.51 / -0.54 | -0.53 / -0.50 / -0.48 | -0.13 / -0.34 / -0.33 | +0.16 / -0.31 / -0.32 | +0.48 / +0.26 / +0.26 | +0.61 / +0.28 / +0.28 | +0.63 / +0.77 / +0.77 |
| orc_d_snap4096_hyb_def | -0.78 / -0.60 / -0.62 | -0.78 / -0.58 / -0.56 | -0.06 / -0.01 / -0.01 | -0.03 / +0.01 / +0.01 | +0.77 / +0.58 / +0.57 | +0.77 / +0.66 / +0.66 | +0.77 / +0.76 / +0.76 |
| orc_d_snap4096_vq_def | -0.90 / -0.68 / -0.71 | -0.91 / -0.68 / -0.67 | -0.40 / -0.11 / -0.11 | -0.24 / -0.06 / -0.06 | +0.60 / +0.50 / +0.50 | +0.70 / +0.60 / +0.60 | +0.69 / +0.75 / +0.75 |

cells: pooled Spearman / within-task (rank, pooled) / mean per-task Spearman

### ctx 32768: strongest prefill features by |mean per-task Spearman|

| target | feature | pooled | within (ranks) | mean task rho | per task |
|---|---|---|---|---|---|
| tau4_snapkv4096 | obs_logn90_mean | -0.41 | -0.28 | -0.30 | {'gov_report': -0.451, 'lcc': -0.141, 'multi_news': 0.058, 'qmsum': 0.214, 'repobench-p': -0.374, 'syn_cont': -0.731, 'syn_quote': -0.312} |
| tau4_snapkv4096 | obs_cov4096_mean | +0.34 | +0.25 | +0.27 | {'gov_report': 0.306, 'lcc': 0.225, 'multi_news': -0.203, 'qmsum': -0.119, 'repobench-p': 0.467, 'syn_cont': 0.728, 'syn_quote': 0.218} |
| tau4_snapkv4096 | obs_far4096_mean | +0.36 | +0.24 | +0.26 | {'gov_report': 0.368, 'lcc': 0.26, 'multi_news': -0.203, 'qmsum': -0.119, 'repobench-p': 0.385, 'syn_cont': 0.728, 'syn_quote': 0.19} |
| alpha1_snapkv4096 | obs_logn90_mean | -0.44 | -0.31 | -0.30 | {'gov_report': -0.462, 'lcc': -0.245, 'multi_news': 0.058, 'qmsum': 0.071, 'repobench-p': -0.394, 'syn_cont': -0.7, 'syn_kv': 0.0, 'syn_quote': -0.348} |
| alpha1_snapkv4096 | obs_cov4096_mean | +0.36 | +0.26 | +0.25 | {'gov_report': 0.334, 'lcc': 0.298, 'multi_news': -0.203, 'qmsum': -0.048, 'repobench-p': 0.432, 'syn_cont': 0.556, 'syn_kv': 0.065, 'syn_quote': 0.255} |
| alpha1_snapkv4096 | obs_o_snap4096_mean | -0.48 | -0.24 | -0.24 | {'gov_report': -0.211, 'lcc': -0.235, 'multi_news': -0.029, 'qmsum': -0.476, 'repobench-p': -0.334, 'syn_cont': -0.342, 'syn_kv': -0.13, 'syn_quote': -0.215} |
| tau4_hyb512pL2 | obs_o_vq_def_mean | -0.54 | -0.19 | -0.19 | {'gov_report': 0.056, 'lcc': -0.099, 'multi_news': -0.577, 'qmsum': -0.476, 'repobench-p': -0.081, 'syn_cont': -0.071, 'syn_kv': -0.367, 'syn_quote': -0.197} |
| tau4_hyb512pL2 | obs_o_snap512_mean | -0.66 | -0.17 | -0.17 | {'gov_report': -0.345, 'lcc': -0.212, 'multi_news': 0.213, 'qmsum': -0.286, 'repobench-p': -0.319, 'syn_cont': -0.368, 'syn_kv': 0.247, 'syn_quote': -0.141} |
| tau4_hyb512pL2 | obs_mfar128_mean | -0.61 | -0.16 | -0.16 | {'gov_report': 0.136, 'lcc': -0.216, 'multi_news': -0.395, 'qmsum': -0.619, 'repobench-p': -0.609, 'syn_cont': 0.339, 'syn_kv': 0.06, 'syn_quote': -0.199} |
| tau4_vqpL2+r128 | obs_cov4096_min | -0.30 | +0.11 | +0.11 | {'gov_report': -0.365, 'lcc': 0.251, 'multi_news': 0.698, 'qmsum': 0.738, 'repobench-p': -0.025, 'syn_cont': 0.429, 'syn_kv': -0.198, 'syn_quote': 0.045} |
| tau4_vqpL2+r128 | obs_d_snap4096_hyb_def | +0.39 | -0.11 | -0.11 | {'gov_report': 0.23, 'lcc': -0.152, 'multi_news': -0.273, 'qmsum': -0.476, 'repobench-p': -0.53, 'syn_cont': 0.45, 'syn_kv': -0.201, 'syn_quote': -0.12} |
| tau4_vqpL2+r128 | obs_d_snap4096_hyb_t2w | +0.19 | -0.10 | -0.10 | {'gov_report': 0.163, 'lcc': 0.392, 'multi_news': -0.091, 'qmsum': -0.429, 'repobench-p': -0.046, 'syn_cont': 0.137, 'syn_kv': -0.576, 'syn_quote': -0.233} |
| logratio4_hyb_vs_snap4096 | obs_d_snap4096_vq_def | +0.46 | +0.26 | +0.26 | {'gov_report': 0.501, 'lcc': 0.084, 'multi_news': -0.086, 'qmsum': 0.81, 'repobench-p': 0.067, 'syn_cont': 0.168, 'syn_kv': 0.311, 'syn_quote': 0.258} |
| logratio4_hyb_vs_snap4096 | obs_o_vq_def_mean | -0.47 | -0.24 | -0.25 | {'gov_report': -0.59, 'lcc': -0.098, 'multi_news': 0.257, 'qmsum': -0.905, 'repobench-p': 0.133, 'syn_cont': -0.095, 'syn_kv': -0.367, 'syn_quote': -0.244} |
| logratio4_hyb_vs_snap4096 | obs_o_vq_def_q90 | -0.59 | -0.20 | -0.20 | {'gov_report': -0.684, 'lcc': -0.105, 'multi_news': 0.086, 'qmsum': -0.762, 'repobench-p': 0.284, 'syn_cont': 0.161, 'syn_kv': -0.291, 'syn_quote': -0.221} |
| logratio4_vq_vs_snap4096 | obs_o_vq_def_q90 | -0.59 | -0.17 | -0.17 | {'gov_report': -0.579, 'lcc': 0.217, 'multi_news': 0.029, 'qmsum': -0.19, 'repobench-p': -0.028, 'syn_cont': -0.021, 'syn_kv': -0.254, 'syn_quote': -0.234} |
| logratio4_vq_vs_snap4096 | obs_o_snap4096_mean | +0.23 | +0.16 | +0.16 | {'gov_report': 0.069, 'lcc': 0.448, 'multi_news': 0.029, 'qmsum': 0.214, 'repobench-p': 0.75, 'syn_cont': -0.042, 'syn_kv': -0.304, 'syn_quote': 0.138} |
| logratio4_vq_vs_snap4096 | obs_logn90_mean | +0.05 | +0.15 | +0.15 | {'gov_report': 0.102, 'lcc': 0.566, 'multi_news': -0.058, 'qmsum': -0.238, 'repobench-p': 0.701, 'syn_cont': -0.232, 'syn_kv': -0.403, 'syn_quote': 0.282} |
| logratio4_hybt2w_vs_snap4096 | obs_o_vq_def_q90 | -0.04 | -0.29 | -0.29 | {'gov_report': -0.546, 'syn_quote': -0.2} |
| logratio4_hybt2w_vs_snap4096 | obs_o_vq_def_mean | -0.03 | -0.25 | -0.25 | {'gov_report': -0.413, 'syn_quote': -0.188} |
| logratio4_hybt2w_vs_snap4096 | obs_logn90_mean | +0.27 | +0.25 | +0.25 | {'gov_report': 0.136, 'syn_quote': 0.285} |

## Does the prefill (obs-window) feature track the same feature at the generated positions?

| ctx | obs -> oracle | pooled Spearman | within-task (ranks) | mean task rho |
|---|---|---|---|---|
| 16384 | obs_cov4096_mean->orc_cov4096_mean | +0.25 | +0.30 | +0.30 |
| 16384 | obs_cov512_mean->orc_cov512_mean | +0.38 | +0.34 | +0.34 |
| 16384 | obs_o_snap4096_mean->orc_o_snap4096_mean | +0.31 | +0.24 | +0.24 |
| 16384 | obs_o_vq_def_mean->orc_o_vq_def_mean | -0.15 | +0.39 | +0.39 |
| 16384 | obs_o_hyb_def_mean->orc_o_hyb_def_mean | +0.18 | +0.25 | +0.25 |
| 16384 | obs_d_snap4096_hyb_def->orc_d_snap4096_hyb_def | +0.27 | +0.24 | +0.24 |
| 32768 | obs_cov4096_mean->orc_cov4096_mean | +0.43 | +0.41 | +0.41 |
| 32768 | obs_cov512_mean->orc_cov512_mean | +0.40 | +0.30 | +0.30 |
| 32768 | obs_o_snap4096_mean->orc_o_snap4096_mean | +0.48 | +0.39 | +0.39 |
| 32768 | obs_o_vq_def_mean->orc_o_vq_def_mean | -0.05 | +0.32 | +0.32 |
| 32768 | obs_o_hyb_def_mean->orc_o_hyb_def_mean | +0.15 | +0.26 | +0.26 |
| 32768 | obs_d_snap4096_hyb_def->orc_d_snap4096_hyb_def | +0.51 | +0.33 | +0.33 |

## Per-sequence draft selection (speedup = tau * T_AR / T_round, measured B16 TP4 step model, all-active)

- step model ctx 16384 (llama8b_magicvq_v6e2e_hotpotqa_q_ctx16384_B16_gen256.json): T_AR 17.70 ms, t_draft/step snapkv2048 10.11, vqpL2+r128 10.76, hyb512pL2 11.09, vqpL2+r128@t2w 10.77, hyb512pL2@t2w 11.10, snapkv4096 11.37 ms, t_verify = 18.76 + 1.06*gamma ms
- step model ctx 32768 (llama8b_magicvq_v6e2e_syn_quote_ctx32768_B16_gen256.json): T_AR 26.41 ms, t_draft/step snapkv4096 11.37, vqpL2+r128 11.87, hyb512pL2 12.20, vqpL2+r128@t2w 11.86, hyb512pL2@t2w 12.21 ms, t_verify = 28.23 + 0.87*gamma ms

### 16384|def (n = 215, gamma per draft {'snapkv4096': 3, 'hyb512pL2': 3, 'vqpL2+r128': 2})

- pooled mean speedup per draft: {'snapkv4096': 1.1429, 'hyb512pL2': 1.165, 'vqpL2+r128': 1.1585}; best single = hyb512pL2 1.165
- oracle best-of-3 per sequence: 1.191 (**+2.2%** vs best single); task oracle (best draft per task): 1.181 (+1.4%), task best = {'gov_report': 'vqpL2+r128', 'hotpotqa_q': 'hyb512pL2', 'lcc': 'snapkv4096', 'multi_news': 'vqpL2+r128', 'qmsum': 'vqpL2+r128', 'repobench-p': 'snapkv4096', 'syn_cont': 'snapkv4096', 'syn_kv': 'hyb512pL2', 'syn_quote': 'hyb512pL2'}
- upper bound, per-sequence oracle over (draft, gamma): 1.223 (+5.0%)
- binary ['snapkv4096', 'hyb512pL2']: best single 1.165, oracle 1.181 (+1.4%), share of sequences where the VQ-side draft wins 0.57

| predictor (leave-one-task-out) | S | gain vs best single | share of oracle gain | accuracy |
|---|---|---|---|---|
| thr:obs_cov4096_mean | 1.154 | -1.0% | -72% | 0.54 |
| thr:obs_far4096_mean | 1.153 | -1.0% | -73% | 0.54 |
| thr:obs_covho4096_mean | 1.154 | -0.9% | -69% | 0.53 |
| thr:obs_d_snap4096_hyb_def | 1.171 | +0.5% | 39% | 0.68 |
| thr:obs_o_snap4096_mean | 1.163 | -0.2% | -13% | 0.56 |
| thr:obs_logn90_mean | 1.167 | +0.1% | 11% | 0.60 |
| lr:cov4096+o_hyb+ent | 1.159 | -0.5% | -40% | 0.57 |
| lr:o_snap4096+o_hyb+far4096 | 1.159 | -0.5% | -36% | 0.56 |
| ORACLE-FEAT thr:orc_d_snap4096_hyb_def | 1.172 | +0.6% | 43% | 0.68 |
| ORACLE-FEAT lr:orc cov4096+o_hyb+o_snap | 1.169 | +0.4% | 26% | 0.64 |

| task | n | mean speedup per draft | oracle3 | share VQ-side better |
|---|---|---|---|---|
| gov_report | 36 | {'snapkv4096': 1.014, 'hyb512pL2': 1.06, 'vqpL2+r128': 1.092} | 1.097 | 0.861 |
| hotpotqa_q | 36 | {'snapkv4096': 1.129, 'hyb512pL2': 1.2, 'vqpL2+r128': 1.174} | 1.216 | 0.583 |
| lcc | 18 | {'snapkv4096': 1.22, 'hyb512pL2': 1.198, 'vqpL2+r128': 1.166} | 1.225 | 0.333 |
| multi_news | 18 | {'snapkv4096': 1.073, 'hyb512pL2': 1.08, 'vqpL2+r128': 1.094} | 1.116 | 0.444 |
| qmsum | 18 | {'snapkv4096': 1.063, 'hyb512pL2': 1.04, 'vqpL2+r128': 1.073} | 1.099 | 0.222 |
| repobench-p | 17 | {'snapkv4096': 1.188, 'hyb512pL2': 1.181, 'vqpL2+r128': 1.164} | 1.209 | 0.412 |
| syn_cont | 18 | {'snapkv4096': 1.248, 'hyb512pL2': 1.198, 'vqpL2+r128': 1.207} | 1.249 | 0.0 |
| syn_kv | 18 | {'snapkv4096': 1.253, 'hyb512pL2': 1.263, 'vqpL2+r128': 1.21} | 1.266 | 0.667 |
| syn_quote | 36 | {'snapkv4096': 1.193, 'hyb512pL2': 1.25, 'vqpL2+r128': 1.228} | 1.252 | 0.917 |

### 16384|t2w (n = 108, gamma per draft {'snapkv4096': 2, 'hyb512pL2@t2w': 5, 'vqpL2+r128@t2w': 5})

- pooled mean speedup per draft: {'snapkv4096': 1.12, 'hyb512pL2@t2w': 1.2429, 'vqpL2+r128@t2w': 1.2418}; best single = hyb512pL2@t2w 1.243
- oracle best-of-3 per sequence: 1.260 (**+1.4%** vs best single); task oracle (best draft per task): 1.251 (+0.7%), task best = {'gov_report': 'hyb512pL2@t2w', 'hotpotqa_q': 'hyb512pL2@t2w', 'syn_quote': 'vqpL2+r128@t2w'}
- upper bound, per-sequence oracle over (draft, gamma): 1.284 (+3.3%)
- binary ['snapkv4096', 'hyb512pL2@t2w']: best single 1.243, oracle 1.244 (+0.1%), share of sequences where the VQ-side draft wins 0.95

| predictor (leave-one-task-out) | S | gain vs best single | share of oracle gain | accuracy |
|---|---|---|---|---|
| thr:obs_cov4096_mean | 1.231 | -1.0% | -1093% | 0.88 |
| thr:obs_far4096_mean | 1.221 | -1.7% | -1943% | 0.82 |
| thr:obs_covho4096_mean | 1.202 | -3.3% | -3694% | 0.66 |
| thr:obs_d_snap4096_hyb_t2w | 1.236 | -0.6% | -672% | 0.89 |
| thr:obs_o_snap4096_mean | 1.243 | +0.0% | 0% | 0.95 |
| thr:obs_logn90_mean | 1.239 | -0.3% | -388% | 0.92 |
| lr:cov4096+o_hyb+ent | 1.243 | +0.0% | 0% | 0.95 |
| lr:o_snap4096+o_hyb+far4096 | 1.243 | +0.0% | 0% | 0.95 |
| ORACLE-FEAT thr:orc_d_snap4096_hyb_t2w | 1.239 | -0.3% | -377% | 0.94 |
| ORACLE-FEAT lr:orc cov4096+o_hyb+o_snap | 1.243 | +0.0% | 0% | 0.95 |

| task | n | mean speedup per draft | oracle3 | share VQ-side better |
|---|---|---|---|---|
| gov_report | 36 | {'snapkv4096': 1.055, 'hyb512pL2@t2w': 1.14, 'vqpL2+r128@t2w': 1.128} | 1.153 | 0.889 |
| hotpotqa_q | 36 | {'snapkv4096': 1.133, 'hyb512pL2@t2w': 1.277, 'vqpL2+r128@t2w': 1.261} | 1.29 | 0.972 |
| syn_quote | 36 | {'snapkv4096': 1.172, 'hyb512pL2@t2w': 1.312, 'vqpL2+r128@t2w': 1.336} | 1.337 | 1.0 |

### 32768|def (n = 111, gamma per draft {'snapkv4096': 4, 'hyb512pL2': 5, 'vqpL2+r128': 4})

- pooled mean speedup per draft: {'snapkv4096': 1.4816, 'hyb512pL2': 1.4604, 'vqpL2+r128': 1.4315}; best single = snapkv4096 1.482
- oracle best-of-3 per sequence: 1.556 (**+5.0%** vs best single); task oracle (best draft per task): 1.537 (+3.7%), task best = {'gov_report': 'vqpL2+r128', 'lcc': 'snapkv4096', 'multi_news': 'snapkv4096', 'qmsum': 'snapkv4096', 'repobench-p': 'snapkv4096', 'syn_cont': 'snapkv4096', 'syn_kv': 'snapkv4096', 'syn_quote': 'hyb512pL2'}
- upper bound, per-sequence oracle over (draft, gamma): 1.609 (+8.6%)
- binary ['snapkv4096', 'hyb512pL2']: best single 1.482, oracle 1.549 (+4.6%), share of sequences where the VQ-side draft wins 0.30

| predictor (leave-one-task-out) | S | gain vs best single | share of oracle gain | accuracy |
|---|---|---|---|---|
| thr:obs_cov4096_mean | 1.425 | -3.8% | -83% | 0.39 |
| thr:obs_far4096_mean | 1.426 | -3.8% | -82% | 0.40 |
| thr:obs_covho4096_mean | 1.474 | -0.5% | -11% | 0.59 |
| thr:obs_d_snap4096_hyb_def | 1.475 | -0.4% | -9% | 0.64 |
| thr:obs_o_snap4096_mean | 1.430 | -3.5% | -76% | 0.42 |
| thr:obs_logn90_mean | 1.411 | -4.8% | -105% | 0.23 |
| lr:cov4096+o_hyb+ent | 1.469 | -0.8% | -18% | 0.59 |
| lr:o_snap4096+o_hyb+far4096 | 1.465 | -1.1% | -24% | 0.52 |
| ORACLE-FEAT thr:orc_d_snap4096_hyb_def | 1.515 | +2.3% | 50% | 0.76 |
| ORACLE-FEAT lr:orc cov4096+o_hyb+o_snap | 1.481 | -0.1% | -1% | 0.68 |

| task | n | mean speedup per draft | oracle3 | share VQ-side better |
|---|---|---|---|---|
| gov_report | 13 | {'snapkv4096': 1.206, 'hyb512pL2': 1.197, 'vqpL2+r128': 1.238} | 1.257 | 0.385 |
| lcc | 12 | {'snapkv4096': 1.496, 'hyb512pL2': 1.476, 'vqpL2+r128': 1.428} | 1.544 | 0.333 |
| multi_news | 6 | {'snapkv4096': 1.374, 'hyb512pL2': 1.233, 'vqpL2+r128': 1.239} | 1.385 | 0.167 |
| qmsum | 8 | {'snapkv4096': 1.375, 'hyb512pL2': 1.24, 'vqpL2+r128': 1.243} | 1.377 | 0.125 |
| repobench-p | 12 | {'snapkv4096': 1.632, 'hyb512pL2': 1.499, 'vqpL2+r128': 1.374} | 1.632 | 0.0 |
| syn_cont | 12 | {'snapkv4096': 1.682, 'hyb512pL2': 1.385, 'vqpL2+r128': 1.458} | 1.682 | 0.0 |
| syn_kv | 12 | {'snapkv4096': 1.709, 'hyb512pL2': 1.656, 'vqpL2+r128': 1.519} | 1.709 | 0.0 |
| syn_quote | 36 | {'snapkv4096': 1.425, 'hyb512pL2': 1.585, 'vqpL2+r128': 1.558} | 1.616 | 0.611 |

### 32768|t2w (n = 49, gamma per draft {'snapkv4096': 3, 'hyb512pL2@t2w': 6, 'vqpL2+r128@t2w': 6})

- pooled mean speedup per draft: {'snapkv4096': 1.3747, 'hyb512pL2@t2w': 1.6163, 'vqpL2+r128@t2w': 1.6234}; best single = vqpL2+r128@t2w 1.623
- oracle best-of-3 per sequence: 1.636 (**+0.8%** vs best single); task oracle (best draft per task): 1.625 (+0.1%), task best = {'gov_report': 'hyb512pL2@t2w', 'syn_quote': 'vqpL2+r128@t2w'}
- upper bound, per-sequence oracle over (draft, gamma): 1.653 (+1.8%)
- binary ['snapkv4096', 'vqpL2+r128@t2w']: best single 1.623, oracle 1.623 (+0.0%), share of sequences where the VQ-side draft wins 1.00

| predictor (leave-one-task-out) | S | gain vs best single | share of oracle gain | accuracy |
|---|---|---|---|---|
| thr:obs_cov4096_mean | 1.608 | -1.0% | n/a (oracle gain <0.1%) | 0.90 |
| thr:obs_far4096_mean | 1.608 | -1.0% | n/a (oracle gain <0.1%) | 0.90 |
| thr:obs_covho4096_mean | 1.623 | +0.0% | n/a (oracle gain <0.1%) | 1.00 |
| thr:obs_d_snap4096_hyb_t2w | 1.602 | -1.3% | n/a (oracle gain <0.1%) | 0.88 |
| thr:obs_o_snap4096_mean | 1.614 | -0.6% | n/a (oracle gain <0.1%) | 0.96 |
| thr:obs_logn90_mean | 1.620 | -0.2% | n/a (oracle gain <0.1%) | 0.98 |
| lr:cov4096+o_hyb+ent | 1.623 | +0.0% | n/a (oracle gain <0.1%) | 1.00 |
| lr:o_snap4096+o_hyb+far4096 | 1.623 | +0.0% | n/a (oracle gain <0.1%) | 1.00 |
| ORACLE-FEAT thr:orc_d_snap4096_hyb_t2w | 1.542 | -5.0% | n/a (oracle gain <0.1%) | 0.55 |
| ORACLE-FEAT lr:orc cov4096+o_hyb+o_snap | 1.623 | +0.0% | n/a (oracle gain <0.1%) | 1.00 |

| task | n | mean speedup per draft | oracle3 | share VQ-side better |
|---|---|---|---|---|
| gov_report | 13 | {'snapkv4096': 1.245, 'hyb512pL2@t2w': 1.408, 'vqpL2+r128@t2w': 1.404} | 1.421 | 1.0 |
| syn_quote | 36 | {'snapkv4096': 1.421, 'hyb512pL2@t2w': 1.691, 'vqpL2+r128@t2w': 1.703} | 1.714 | 1.0 |

## Per-head heterogeneity (obs window; layer 0 excluded; quantiles across the 31x8 (layer, kv-head) per sequence, median over sequences)

| ctx | stat | q10 | q50 | q90 | q90/q10 | head-rank stability same task | diff task | frac heads cov<0.9 | share of total err in worst 10% heads |
|---|---|---|---|---|---|---|---|---|---|
| 16384 | cov4096 | 0.944 | 0.978 | 0.993 | 1.05 | 0.92 | 0.75 | 0.02 | - |
| 16384 | cov512 | 0.808 | 0.914 | 0.970 | 1.20 | 0.94 | 0.79 | 0.41 | - |
| 16384 | e_vq_def | 0.042 | 0.146 | 0.328 | 7.27 | 0.95 | 0.86 | - | 0.248 |
| 16384 | o_vq_def | 0.008 | 0.047 | 0.143 | 15.18 | 0.95 | 0.85 | - | 0.299 |
| 16384 | o_hyb_def | 0.004 | 0.013 | 0.034 | 7.88 | 0.92 | 0.75 | - | 0.256 |
| 16384 | orc_cov4096 | 0.851 | 0.955 | 0.989 | 1.16 | 0.95 | 0.78 | 0.225 | - |
| 16384 | orc_o_vq_def | 0.007 | 0.036 | 0.179 | 24.97 | 0.97 | 0.89 | - | 0.341 |
| 32768 | cov4096 | 0.919 | 0.968 | 0.990 | 1.08 | 0.95 | 0.77 | 0.07 | - |
| 32768 | cov512 | 0.806 | 0.918 | 0.971 | 1.20 | 0.95 | 0.79 | 0.439 | - |
| 32768 | e_vq_def | 0.049 | 0.138 | 0.340 | 5.93 | 0.93 | 0.82 | - | 0.242 |
| 32768 | o_vq_def | 0.009 | 0.039 | 0.141 | 12.46 | 0.94 | 0.84 | - | 0.29 |
| 32768 | o_hyb_def | 0.004 | 0.014 | 0.041 | 8.98 | 0.93 | 0.75 | - | 0.257 |
| 32768 | orc_cov4096 | 0.798 | 0.939 | 0.986 | 1.23 | 0.95 | 0.77 | 0.291 | - |
| 32768 | orc_o_vq_def | 0.010 | 0.046 | 0.187 | 18.32 | 0.96 | 0.86 | - | 0.325 |
- per-sequence table moved to /data/lsh/results_accept/accept_pred/accept_pred_v1_per_seq.json (3.8 MB, not in git)
