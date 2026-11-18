# E7 magicvq v7: more measured e2e points with the v6 t2w codebook (TP=4 RTX 4090, GPUs 4-7, bf16 Llama-3.1-8B-Instruct, gen 256, EOS ignored)

- e2e_real.py --accounting both --cbx t2w=cb_d8K256x2_post_cn_t2w.pt; static and all-active reps interleaved, warm-up excluded, 3 timed reps each, median. Speedup vs the AR of the same process.
- all-active (headline, MagicDec accounting): no row frozen, stop when the longest row has >= 256 tokens. static: every row exactly 256 tokens.
- StreamingLLM-512 (new draft kind 'stream512' in e2e_real.py): 16 sinks + last 496 prompt tokens + the growing generated window (MagicDec prefill_draft semantics, same cache handling as our SnapKV draft).
- LongBench samples: the v4 (+v4b) tau samples in order (passed lb_tau's min_gen 128 filter), first B. hotpotqa_q 32k prompts rebuilt after the sorted(own) determinism fix (identical under PYTHONHASHSEED 1 and 2), so they differ from the v4/v6 32k prompts.
- ratio = best VQ/hybrid (t2w) speedup / best SnapKV speedup over ALL measured SnapKV budgets / gammas (and / best StreamingLLM). v6 computed = best trained-2-bit / best SnapKV from results/E7/llama8b_magicvq_v6_tables.md (replay tau x step times).

## Summary per point

| point | P min-max | AR tok/s | all-active: SnapKV / StreamingLLM / VQ t2w | **aa ratio VQ/SnapKV** | aa VQ/Stream | static: SnapKV / StreamingLLM / VQ t2w | static ratio VQ/SnapKV | v6 computed VQ/SnapKV | SnapKV gammas measured (g5/6?) | co-tenant |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa_q ctx 16384 B=16 (gamma 5/6 for t2w) | 15062-16375 | 905.3 / 905.8 / 905.3 | 1.197 (SnapKV-2048 g4) / - / 1.302 (SnapKV512+RQ2 hybrid g6) | **1.088** | - | 0.936 (SnapKV-2048 g2) / - / 1.218 (SnapKV512+RQ2 hybrid g6) | 1.301 | 1.158 (hyb512pt2wL2 g6) | 2048:g2, 2048:g4, 2048:g5, 2048:g6, 4096:g6 (yes) | no |
| pg19 ctx 32768 B=16 (gamma 5/6 for t2w) | 32768-32768 | 592.4 / 583.2 / 591.6 | 1.619 (SnapKV-4096 g6) / - / 1.623 (SnapKV512+RQ2 hybrid g6) | **1.002** | - | 1.474 (SnapKV-4096 g5) / - / 1.407 (SnapKV512+RQ2 hybrid g5) | 0.955 | 0.964 (hyb512pt2wL2 g6) | 4096:g3, 4096:g4, 4096:g5, 4096:g6 (yes) | no |
| gov_report ctx 32768 B=8 | 25249-32751 | 523.7 | 1.056 (SnapKV-4096 g3) / 0.924 (StreamingLLM-512 g3) / 1.144 (RQ2+r128 g3) | **1.083** | 1.239 | 0.936 (SnapKV-4096 g3) / 0.824 (StreamingLLM-512 g3) / 1.101 (RQ2+r128 g3) | 1.177 | 1.062 (hyb512pt2wL2 g3) | 2048:g3, 2048:g4, 4096:g3, 4096:g4 (NO) | no |
| gov_report ctx 16384 B=16 | 15016-16375 | 899.6 | 1.021 (SnapKV-4096 g3) / 0.889 (StreamingLLM-512 g3) / 1.185 (SnapKV512+RQ2 hybrid g6) | **1.161** | 1.333 | 0.872 (SnapKV-4096 g3) / 0.777 (StreamingLLM-512 g3) / 1.098 (SnapKV512+RQ2 hybrid g3) | 1.260 | 1.099 (hyb512pt2wL2 g3) | 2048:g3, 2048:g4, 4096:g3, 4096:g4 (NO) | no |
| qmsum ctx 16384 B=16 | 15316-16375 | 896.9 | 1.104 (SnapKV-2048 g3) / 0.944 (StreamingLLM-512 g3) / 1.199 (RQ2+r128 g6) | **1.085** | 1.270 | 1.004 (SnapKV-4096 g3) / 0.828 (StreamingLLM-512 g3) / 1.089 (SnapKV512+RQ2 hybrid g4) | 1.084 | - | 2048:g3, 2048:g4, 4096:g3, 4096:g4 (NO) | no |
| repobench-p ctx 16384 B=16 | 15750-16375 | 892.8 | 1.229 (SnapKV-2048 g3) / 1.161 (StreamingLLM-512 g3) / 1.279 (SnapKV512+RQ2 hybrid g6) | **1.041** | 1.101 | 0.948 (SnapKV-4096 g3) / 0.846 (StreamingLLM-512 g3) / 1.201 (SnapKV512+RQ2 hybrid g3) | 1.267 | - | 2048:g3, 2048:g4, 4096:g3, 4096:g4 (NO) | no |

## hotpotqa_q ctx 16384 B=16 (gamma 5/6 for t2w)

processes: v7g6: AR 905.3 tok/s (step 17.67 ms), co-tenant no, run_v7g6_hotpotqa_q_ctx16384_B16.json (PARTIAL: process aborted after the listed configs (hotpotqa: vqpL2+r128@t2w:6 static outputs differed between reps; pg19: OOM at vqpL2+r128@t2w:6); end co-check not recorded, per-config co_before/co_after are); v7g6v: AR 905.8 tok/s (step 17.66 ms), co-tenant no, run_v7g6v_hotpotqa_q_ctx16384_B16.json; v7g6s: AR 905.3 tok/s (step 17.67 ms), co-tenant no, run_v7g6s_hotpotqa_q_ctx16384_B16.json
prompt lengths P: [15515, 16245, 15600, 15566, 15062, 15091, 16265, 15765, 16035, 15593, 15851, 16375, 16372, 15740, 15489, 16348]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-2048 | v7g6 | 2 | 0.936 | **1.170** (1.170-1.170) | 2.70 / 2.73 | 41.3 / 41.2 | 5/16 | no | no |
| SnapKV-2048 | v7g6 | 4 | 0.645 | **1.197** (1.196-1.197) | 4.05 / 4.30 | 63.7 / 63.6 | 7/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6 | 6 | 1.218 | **1.302** (1.302-1.303) | 6.69 / 6.68 | 90.6 / 90.6 | 7/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6 | 5 | 1.190 | **1.280** (1.280-1.281) | 5.74 / 5.74 | 79.2 / 79.2 | 8/16 | no | no |
| SnapKV-2048 | v7g6v | 4 | 0.645 | **1.196** (1.196-1.197) | 4.05 / 4.30 | 63.7 / 63.5 | 7/16 | no | no |
| RQ2+r128 [t2w] | v7g6v | 6 | 1.180 | **1.288** (1.287-1.289) | 6.48 / 6.49 | 89.1 / 89.0 | 7/16 | no | YES: static rows vs rep0 [2, 2, 2], aa rows [], reps 1-3 same divergence pattern: True |
| RQ2+r128 [t2w] | v7g6v | 5 | 1.135 | **1.265** (1.264-1.265) | 5.59 / 5.59 | 78.1 / 78.0 | 8/16 | no | no |
| SnapKV-2048 | v7g6s | 5 | 0.587 | **1.188** (1.187-1.188) | 4.60 / 5.01 | 74.8 / 74.5 | 7/16 | no | no |
| SnapKV-2048 | v7g6s | 6 | 0.537 | **1.152** (1.152-1.153) | 5.07 / 5.54 | 85.1 / 84.9 | 9/16 | no | YES: static rows vs rep0 [2, 2, 2], aa rows [], reps 1-3 same divergence pattern: True |
| SnapKV-4096 | v7g6s | 6 | 0.603 | **1.139** (1.138-1.139) | 5.57 / 5.94 | 92.7 / 92.2 | 7/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6s | 6 | 1.216 | **1.297** (1.295-1.297) | 6.66 / 6.64 | 90.7 / 90.5 | 7/16 | no | no |

## pg19 ctx 32768 B=16 (gamma 5/6 for t2w)

processes: v7g6: AR 592.4 tok/s (step 27.01 ms), co-tenant no, run_v7g6_pg19_ctx32768_B16.json (PARTIAL: process aborted after the listed configs (hotpotqa: vqpL2+r128@t2w:6 static outputs differed between reps; pg19: OOM at vqpL2+r128@t2w:6); end co-check not recorded, per-config co_before/co_after are); v7g6v: AR 583.2 tok/s (step 27.43 ms), co-tenant no, run_v7g6v_pg19_ctx32768_B16.json; v7g6s: AR 591.6 tok/s (step 27.05 ms), co-tenant no, run_v7g6s_pg19_ctx32768_B16.json
prompt lengths P: [32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v7g6 | 3 | 1.293 | **1.545** (1.545-1.545) | 3.72 / 3.73 | 65.2 / 65.1 | 16/16 | no | no |
| SnapKV-4096 | v7g6 | 4 | 1.432 | **1.582** (1.581-1.582) | 4.54 / 4.55 | 77.9 / 77.7 | 1/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6 | 6 | 1.405 | **1.623** (1.623-1.623) | 6.41 / 6.42 | 107.0 / 106.9 | 1/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6 | 5 | 1.407 | **1.595** (1.595-1.595) | 5.57 / 5.58 | 94.5 / 94.4 | 1/16 | no | no |
| SnapKV-4096 | v7g6v | 4 | 1.452 | **1.605** (1.605-1.605) | 4.54 / 4.55 | 78.0 / 77.8 | 1/16 | no | no |
| RQ2+r128 [t2w] | v7g6v | 6 | 1.220 | **1.491** (1.490-1.492) | 5.82 / 5.79 | 106.6 / 106.5 | 1/16 | no | no |
| RQ2+r128 [t2w] | v7g6v | 5 | 1.243 | **1.504** (1.504-1.505) | 5.18 / 5.16 | 94.2 / 94.1 | 1/16 | no | no |
| SnapKV-4096 | v7g6s | 5 | 1.474 | **1.615** (1.615-1.616) | 5.40 / 5.40 | 90.4 / 90.3 | 1/16 | no | no |
| SnapKV-4096 | v7g6s | 6 | 1.443 | **1.619** (1.619-1.620) | 6.11 / 6.10 | 102.1 / 101.8 | 1/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7g6s | 6 | 1.401 | **1.622** (1.622-1.624) | 6.41 / 6.42 | 107.4 / 107.1 | 1/16 | no | no |

## gov_report ctx 32768 B=8

processes: v7: AR 523.7 tok/s (step 15.28 ms), co-tenant no, run_v7_gov_report_ctx32768_B8.json
prompt lengths P: [27082, 26799, 27082, 32751, 25372, 27755, 25249, 26799]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v7 | 3 | 0.936 | **1.056** (1.056-1.056) | 3.09 / 3.14 | 45.4 / 45.4 | 4/8 | no | no |
| SnapKV-4096 | v7 | 4 | 0.863 | **1.013** (1.012-1.013) | 3.52 / 3.71 | 56.0 / 55.9 | 5/8 | no | no |
| SnapKV-2048 | v7 | 3 | 0.889 | **1.014** (1.014-1.014) | 2.82 / 2.89 | 43.5 / 43.5 | 4/8 | no | no |
| SnapKV-2048 | v7 | 4 | 0.778 | **0.935** (0.935-0.935) | 3.11 / 3.27 | 53.4 / 53.4 | 5/8 | no | no |
| StreamingLLM-512 | v7 | 3 | 0.824 | **0.924** (0.923-0.924) | 2.49 / 2.54 | 42.0 / 42.0 | 5/8 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 3 | 1.079 | **1.144** (1.144-1.144) | 3.57 / 3.57 | 47.7 / 47.7 | 3/8 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 4 | 1.021 | **1.123** (1.123-1.123) | 4.32 / 4.33 | 59.0 / 59.0 | 3/8 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 6 | 0.997 | **1.106** (1.106-1.106) | 5.69 / 5.68 | 78.5 / 78.5 | 3/8 | no | no |
| RQ2+r128 [t2w] | v7 | 3 | 1.101 | **1.144** (1.144-1.144) | 3.55 / 3.54 | 47.3 / 47.3 | 3/8 | no | no |
| RQ2+r128 [t2w] | v7 | 4 | 1.013 | **1.115** (1.115-1.115) | 4.26 / 4.27 | 58.5 / 58.5 | 3/8 | no | no |
| RQ2+r128 [t2w] | v7 | 6 | 1.028 | **1.095** (1.095-1.095) | 5.59 / 5.57 | 77.7 / 77.7 | 2/8 | no | no |

## gov_report ctx 16384 B=16

processes: v7: AR 899.6 tok/s (step 17.79 ms), co-tenant no, run_v7_gov_report_ctx16384_B16.json
prompt lengths P: [15110, 16360, 16372, 16375, 15216, 16375, 15493, 16375, 16375, 16375, 15016, 16375, 16375, 16375, 15594, 16375]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v7 | 3 | 0.872 | **1.021** (1.020-1.021) | 3.18 / 3.22 | 56.2 / 56.0 | 16/16 | no | no |
| SnapKV-4096 | v7 | 4 | 0.770 | **0.975** (0.975-0.975) | 3.73 / 3.77 | 68.8 / 68.8 | 4/16 | no | no |
| SnapKV-2048 | v7 | 3 | 0.869 | **0.990** (0.989-0.990) | 2.90 / 2.91 | 52.4 / 52.4 | 16/16 | no | no |
| SnapKV-2048 | v7 | 4 | 0.735 | **0.941** (0.941-0.941) | 3.31 / 3.38 | 63.8 / 63.8 | 5/16 | no | no |
| StreamingLLM-512 | v7 | 3 | 0.777 | **0.889** (0.889-0.890) | 2.48 / 2.48 | 49.6 / 49.6 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 3 | 1.098 | **1.176** (1.176-1.177) | 3.66 / 3.66 | 55.3 / 55.3 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 4 | 1.086 | **1.146** (1.146-1.146) | 4.36 / 4.36 | 67.6 / 67.7 | 2/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 6 | 1.043 | **1.185** (1.184-1.186) | 5.72 / 6.06 | 91.0 / 91.0 | 4/16 | no | YES: static rows vs rep0 [5, 6, 6], aa rows [3, 4, 4], reps 1-3 same divergence pattern: False |
| RQ2+r128 [t2w] | v7 | 3 | 1.083 | **1.183** (1.183-1.184) | 3.62 / 3.63 | 54.6 / 54.6 | 10/16 | no | no |
| RQ2+r128 [t2w] | v7 | 4 | 1.018 | **1.166** (1.166-1.168) | 4.38 / 4.39 | 66.8 / 66.9 | 3/16 | no | no |
| RQ2+r128 [t2w] | v7 | 6 | 0.461 | **1.167** (1.158-1.195) | 5.60 / 5.87 | 89.8 / 89.5 | 0/16 | no | YES: static rows vs rep0 [16, 16, 16], aa rows [9, 9, 9], reps 1-3 same divergence pattern: False |

## qmsum ctx 16384 B=16

processes: v7: AR 896.9 tok/s (step 17.84 ms), co-tenant no, run_v7_qmsum_ctx16384_B16.json
prompt lengths P: [15687, 16375, 15316, 16375, 16162, 16375, 16068, 16068, 16057, 16375, 15370, 16162, 16375, 16172, 16068, 16082]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v7 | 3 | 1.004 | **1.098** (1.097-1.098) | 3.45 / 3.46 | 56.2 / 56.2 | 16/16 | no | no |
| SnapKV-4096 | v7 | 4 | 0.908 | **1.072** (1.072-1.073) | 4.11 / 4.14 | 68.9 / 68.8 | 2/16 | no | no |
| SnapKV-2048 | v7 | 3 | 0.956 | **1.104** (1.103-1.105) | 3.23 / 3.25 | 52.5 / 52.5 | 16/16 | no | no |
| SnapKV-2048 | v7 | 4 | 0.917 | **1.055** (1.055-1.055) | 3.75 / 3.77 | 63.8 / 63.8 | 2/16 | no | no |
| StreamingLLM-512 | v7 | 3 | 0.828 | **0.944** (0.944-0.944) | 2.63 / 2.62 | 49.7 / 49.6 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 3 | 1.071 | **1.168** (1.167-1.168) | 3.62 / 3.62 | 55.4 / 55.3 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 4 | 1.089 | **1.160** (1.160-1.161) | 4.41 / 4.40 | 67.7 / 67.7 | 2/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 6 | 1.046 | **1.193** (1.192-1.193) | 5.79 / 6.08 | 91.0 / 90.9 | 1/16 | no | YES: static rows vs rep0 [6, 6, 6], aa rows [], reps 1-3 same divergence pattern: True |
| RQ2+r128 [t2w] | v7 | 3 | 1.058 | **1.193** (1.192-1.193) | 3.64 / 3.65 | 54.6 / 54.6 | 10/16 | no | no |
| RQ2+r128 [t2w] | v7 | 4 | 1.036 | **1.171** (1.171-1.172) | 4.38 / 4.38 | 66.8 / 66.7 | 1/16 | no | no |
| RQ2+r128 [t2w] | v7 | 6 | 0.944 | **1.199** (1.198-1.199) | 5.75 / 6.01 | 89.6 / 89.4 | 1/16 | no | YES: static rows vs rep0 [2, 2, 2], aa rows [], reps 1-3 same divergence pattern: True |

## repobench-p ctx 16384 B=16

processes: v7: AR 892.8 tok/s (step 17.92 ms), co-tenant no, run_v7_repobench-p_ctx16384_B16.json
prompt lengths P: [15829, 16371, 15963, 16355, 16369, 16369, 16373, 16362, 16364, 16368, 16370, 15750, 16337, 16375, 16357, 16351]

| draft | proc | gamma | static | all-active (min-max) | tau static / aa | ms/round static / aa | aa rows identical to AR on overlap | co-tenant | outputs differ across reps |
|---|---|---|---|---|---|---|---|---|---|
| SnapKV-4096 | v7 | 3 | 0.948 | **1.196** (1.196-1.199) | 3.75 / 3.75 | 56.3 / 56.2 | 16/16 | no | no |
| SnapKV-4096 | v7 | 4 | 0.913 | **1.204** (1.203-1.204) | 4.60 / 4.62 | 68.9 / 68.8 | 10/16 | no | no |
| SnapKV-2048 | v7 | 3 | 0.919 | **1.229** (1.228-1.229) | 3.58 / 3.59 | 52.6 / 52.4 | 16/16 | no | no |
| SnapKV-2048 | v7 | 4 | 0.835 | **1.216** (1.216-1.217) | 4.29 / 4.32 | 63.9 / 63.7 | 10/16 | no | no |
| StreamingLLM-512 | v7 | 3 | 0.846 | **1.161** (1.161-1.162) | 3.23 / 3.22 | 49.7 / 49.6 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 3 | 1.201 | **1.253** (1.252-1.253) | 3.87 / 3.87 | 55.4 / 55.4 | 16/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 4 | 1.168 | **1.275** (1.275-1.276) | 4.81 / 4.81 | 67.7 / 67.6 | 10/16 | no | no |
| SnapKV512+RQ2 hybrid [t2w] | v7 | 6 | 1.120 | **1.279** (1.279-1.279) | 6.49 / 6.49 | 91.0 / 90.9 | 8/16 | no | YES: static rows vs rep0 [2, 2, 2], aa rows [2, 2, 2], reps 1-3 same divergence pattern: True |
| RQ2+r128 [t2w] | v7 | 3 | 1.048 | **1.241** (1.240-1.241) | 3.78 / 3.79 | 54.7 / 54.8 | 14/16 | no | no |
| RQ2+r128 [t2w] | v7 | 4 | 1.164 | **1.268** (1.267-1.268) | 4.74 / 4.73 | 66.8 / 66.9 | 8/16 | no | no |
| RQ2+r128 [t2w] | v7 | 6 | 1.044 | **1.272** (1.271-1.273) | 6.37 / 6.36 | 89.7 / 89.6 | 8/16 | no | no |

