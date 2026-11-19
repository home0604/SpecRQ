# E7 gamma-6 rep-mismatch: target-cache overflow in static accounting (fix + reruns)

- Cause: static mode keeps frozen rows (ngen >= gen) running draft + verify at their final length (<= P + gen + gamma) until the slowest row finishes, so they append K/V up to position P + gen + 2 gamma. Old T = ceil64(Pmax + gen + gmax + 2): at gamma 6 with P near Pmax (16k LongBench points, slack 9 < 12) the write wraps into position 0.. of the next (seq, kv-head) [or the next k/v plane / layer] of the TARGET cache (BOS / attention sinks) and stays corrupted for the rest of the process (all later configs and reps).
- Fix: T = ceil64(Pmax + gen + 2 gmax + 2) + guard (assert length.max + gamma < T after each rep) + prompt-head integrity check (first 32 positions of every head vs the post-prefill snapshot, after every config). Fixed-T reruns: same prompts / configs / codebooks, --accounting both --soft_rep_check, reps 3.
- speedup = config tok/s / AR tok/s of the same process (median of timed reps); aa = all-active (headline).
- tau static = rep-0 acc_log mean (as in the v7/v8 tables); tau aa = median over timed reps.

## gov_report 16k B16 - new process `rmfix7`

AR tok/s new 894.3; old v7 899.6; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv4096:3 | 1.021 (1.020-1.021) [v7] | **1.012 (1.012-1.013)** | 0.872 / 0.887 | 3.22 / 3.18 | 3.18 / 3.13 | no / no | 16 / 16 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.185 (1.184-1.186) [v7] | **1.140 (1.140-1.141)** | 1.043 / 0.969 | 6.06 / 5.79 | 5.72 / 5.78 | yes / no | 4 / 3 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:3 | 1.183 (1.183-1.184) [v7] | **1.176 (1.175-1.177)** | 1.083 / 1.089 | 3.63 / 3.59 | 3.62 / 3.58 | no / no | 10 / 16 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:4 | 1.166 (1.166-1.168) [v7] | **1.157 (1.156-1.157)** | 1.018 / 1.072 | 4.39 / 4.32 | 4.38 / 4.30 | no / no | 3 / 3 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:6 | 1.167 (1.158-1.195) [v7] | **1.131 (1.131-1.131)** | 0.461 / 0.984 | 5.87 / 5.66 | 5.60 / 5.60 | yes / no | 0 / 3 | 0 heads, rows [], pos [] |

## qmsum 16k B16 - new process `rmfix7`

AR tok/s new 895.7; old v7 896.9; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv2048:3 | 1.104 (1.103-1.105) [v7] | **1.092 (1.092-1.092)** | 0.956 / 0.928 | 3.25 / 3.20 | 3.23 / 3.18 | no / no | 16 / 16 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.193 (1.192-1.193) [v7] | **1.122 (1.120-1.123)** | 1.046 / 0.985 | 6.08 / 5.70 | 5.79 / 5.71 | yes / no | 1 / 3 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:3 | 1.193 (1.192-1.193) [v7] | **1.143 (1.143-1.143)** | 1.058 / 1.034 | 3.65 / 3.49 | 3.64 / 3.49 | no / no | 10 / 16 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:4 | 1.171 (1.171-1.172) [v7] | **1.124 (1.124-1.124)** | 1.036 / 1.022 | 4.38 / 4.20 | 4.38 / 4.20 | no / no | 1 / 3 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:6 | 1.199 (1.198-1.199) [v7] | **1.067 (1.067-1.067)** | 0.944 / 0.963 | 6.01 / 5.34 | 5.75 / 5.34 | yes / no | 1 / 3 | 0 heads, rows [], pos [] |

## repobench-p 16k B16 - new process `rmfix7`

AR tok/s new 890.3; old v7 892.8; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv2048:3 | 1.229 (1.228-1.229) [v7] | **1.227 (1.227-1.228)** | 0.919 / 0.921 | 3.59 / 3.58 | 3.58 / 3.57 | no / no | 16 / 16 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.279 (1.279-1.279) [v7] | **1.286 (1.286-1.287)** | 1.120 / 1.124 | 6.49 / 6.51 | 6.49 / 6.50 | yes / no | 8 / 7 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:3 | 1.241 (1.240-1.241) [v7] | **1.258 (1.257-1.259)** | 1.048 / 1.183 | 3.79 / 3.83 | 3.78 / 3.84 | no / no | 14 / 16 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:4 | 1.268 (1.267-1.268) [v7] | **1.268 (1.268-1.269)** | 1.164 / 1.166 | 4.73 / 4.73 | 4.74 / 4.73 | no / no | 8 / 7 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:6 | 1.272 (1.271-1.273) [v7] | **1.277 (1.277-1.277)** | 1.044 / 1.091 | 6.36 / 6.37 | 6.37 / 6.38 | no / no | 8 / 7 | 0 heads, rows [], pos [] |

## hotpotqa_q 16k B16 - new process `rmfix7g6`

AR tok/s new 901.8; old v7g6 905.3, v7g6s 905.3, v7g6v 905.8; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv2048:4 | 1.197 (1.196-1.197) [v7g6] | **1.162 (1.161-1.162)** | 0.645 / 0.651 | 4.30 / 4.18 | 4.05 / 3.95 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| snapkv2048:6 | 1.152 (1.152-1.153) [v7g6s] | **1.119 (1.119-1.120)** | 0.537 / 0.537 | 5.54 / 5.37 | 5.07 / 4.86 | yes / no | 9 / 7 | 0 heads, rows [], pos [] |
| snapkv4096:6 | 1.139 (1.138-1.139) [v7g6s] | **1.136 (1.136-1.137)** | 0.603 / 0.603 | 5.94 / 5.93 | 5.57 / 5.57 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.302 (1.302-1.303) [v7g6] | **1.296 (1.296-1.298)** | 1.218 / 1.190 | 6.68 / 6.65 | 6.69 / 6.66 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:5 | 1.280 (1.280-1.281) [v7g6] | **1.281 (1.281-1.281)** | 1.190 / 1.188 | 5.74 / 5.75 | 5.74 / 5.75 | no / no | 8 / 5 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:6 | 1.288 (1.287-1.289) [v7g6v] | **1.288 (1.287-1.288)** | 1.180 / 1.182 | 6.49 / 6.48 | 6.48 / 6.49 | yes / no | 7 / 5 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:5 | 1.265 (1.264-1.265) [v7g6v] | **1.275 (1.274-1.276)** | 1.135 / 1.207 | 5.59 / 5.63 | 5.59 / 5.64 | no / no | 8 / 5 | 0 heads, rows [], pos [] |

## gov_report 16k B16 - new process `rmfixa`

AR tok/s new 893.8; old v8e2ea 899.2; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv4096:3 | 1.020 (1.020-1.020) [v8e2ea] | **1.013 (1.012-1.013)** | 0.872 / 0.885 | 3.22 / 3.18 | 3.18 / 3.13 | no / no | 16 / 16 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.187 (1.186-1.188) [v8e2ea] | **1.138 (1.138-1.138)** | 1.043 / 0.966 | 6.06 / 5.79 | 5.72 / 5.78 | yes / no | 4 / 3 | 0 heads, rows [], pos [] |
| hyb512pL1@t1w:4 | 1.057 (1.056-1.057) [v8e2ea] | **0.971 (0.970-0.971)** | 0.831 / 0.826 | 3.86 / 3.54 | 3.81 / 3.52 | no / no | 3 / 3 | 0 heads, rows [], pos [] |
| hyb512pL1@t1w:6 | 1.253 (1.209-1.297) [v8e2ea] | **0.854 (0.854-0.854)** | 0.931 / 0.728 | 6.14 / 4.17 | 4.45 / 4.12 | yes / no | 0 / 3 | 0 heads, rows [], pos [] |

## hotpotqa_q 16k B16 - new process `rmfixa`

AR tok/s new 901.0; old v8e2ea 906.4; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv2048:4 | 1.194 (1.194-1.195) [v8e2ea] | **1.161 (1.160-1.161)** | 0.644 / 0.651 | 4.30 / 4.18 | 4.05 / 3.95 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.302 (1.301-1.302) [v8e2ea] | **1.296 (1.296-1.297)** | 1.216 / 1.189 | 6.68 / 6.65 | 6.69 / 6.66 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL1@t1w:4 | 1.189 (1.188-1.189) [v8e2ea] | **1.148 (1.146-1.148)** | 0.952 / 0.952 | 4.37 / 4.22 | 4.33 / 4.20 | no / no | 8 / 6 | 0 heads, rows [], pos [] |
| hyb512pL1@t1w:6 | 1.180 (1.179-1.181) [v8e2ea] | **1.118 (1.117-1.118)** | 0.837 / 0.838 | 5.80 / 5.50 | 5.63 / 5.39 | yes / no | 8 / 6 | 0 heads, rows [], pos [] |

## hotpotqa_q 16k B16 - new process `rmfixmse`

AR tok/s new 899.2; old v7mse 905.1; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| snapkv2048:4 | 1.194 (1.194-1.195) [v7mse] | **1.166 (1.166-1.167)** | 0.644 / 0.653 | 4.30 / 4.18 | 4.05 / 3.95 | no / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL2:6 | 1.212 (1.211-1.212) [v7mse] | **1.205 (1.205-1.206)** | 1.060 / 1.066 | 6.21 / 6.15 | 6.26 / 6.16 | yes / no | 8 / 6 | 0 heads, rows [], pos [] |
| hyb512pL2@t2mw:6 | 1.290 (1.289-1.297) [v7mse] | **1.282 (1.281-1.282)** | 1.159 / 1.194 | 6.62 / 6.54 | 6.56 / 6.55 | yes / no | 7 / 5 | 0 heads, rows [], pos [] |
| hyb512pL2@t2w:6 | 1.299 (1.298-1.299) [v7mse] | **1.303 (1.303-1.304)** | 1.216 / 1.193 | 6.67 / 6.65 | 6.66 / 6.66 | yes / no | 6 / 5 | 0 heads, rows [], pos [] |
| vqpL2+r128:6 | 1.131 (1.130-1.131) [v7mse] | **1.121 (1.121-1.121)** | 0.939 / 0.981 | 5.70 / 5.63 | 5.80 / 5.69 | yes / no | 7 / 6 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2mw:6 | 1.183 (1.183-1.183) [v7mse] | **1.226 (1.225-1.226)** | 0.975 / 1.133 | 5.97 / 6.15 | 6.05 / 6.19 | no / no | 7 / 6 | 0 heads, rows [], pos [] |
| vqpL2+r128@t2w:6 | 1.274 (1.273-1.275) [v7mse] | **1.292 (1.291-1.292)** | 1.180 / 1.187 | 6.43 / 6.48 | 6.44 / 6.49 | no / no | 6 / 5 | 0 heads, rows [], pos [] |

## gov_report 16k B16 - new process `rmlegacy` (LEGACY T, bug reproduction, reps 1)

AR tok/s new 900.4; old v8e2ea 899.2; co-tenant any (new): False

| config | aa speedup OLD (min-max) | aa speedup NEW (min-max) | static OLD / NEW | tau aa OLD / NEW | tau static OLD / NEW | rep diff OLD / NEW | aa rows = AR OLD / NEW | prompt-head changed after (NEW) |
|---|---|---|---|---|---|---|---|---|
| hyb512pL2@t2w:6 | 1.187 (1.186-1.188) [v8e2ea] | **1.187 (1.187-1.187)** | 1.043 / 1.063 | 6.06 / 6.07 | 5.72 / 5.72 | yes / yes | 4 / 4 | 1536 heads, rows [3, 4, 11, 12, 13], pos [0, 1, 2, 3] |
| hyb512pL1@t1w:6 | 1.253 (1.209-1.297) [v8e2ea] | **1.244 (1.244-1.244)** | 0.931 / 0.548 | 6.14 / 6.11 | 4.45 / 4.35 | yes / yes | 0 / 0 | 3580 heads, rows [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15], pos [0, 1, 2, 3] |
| hyb512pL1@t1w:4 | 1.057 (1.056-1.057) [v8e2ea] | **1.232 (1.232-1.232)** | 0.831 / 0.919 | 3.86 / 4.51 | 3.81 / 4.46 | no / no | 3 / 0 | 3580 heads, rows [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15], pos [0, 1, 2, 3] |

