# E7: phase-3 LUT kernel in the e2e draft path (MV_LUT_VARIANT=4090) — 4090 TP4 timing

- commit 44c4e9e; job `p3e2e_timing` (`experiments/26_magicvq/p3_e2e_run.sh timing`), GPUs 4-7, co-tenant False in all 4 runs (other user only on GPU 2).
- syn_quote 32k, B16, gen 256, reps 3, all_active; order off1, on1, off2, on2. Preset 4090: KST1 `k1_w8_v1_r1_h1` (tensor-core P.V), KST2 `k2_w8_v2_r1_h0` (HFMA2). The p3 path always uses lazy max; off = original kernel without lazy max.
- raw: `/data/lsh/results_accept/magicvq/e2e_real/run_p3{off,on}{1,2}_syn_quote_ctx32768_B16.json`, log `/data/lsh/logs/p3e2e_timing.log`.

| config | off tok/s (r1 / r2) | on tok/s (r1 / r2) | gain | AR speedup off -> on | tau off / on |
|---|---|---|---|---|---|
| AR | 606.5 | 606.5 | - | 1 | - |
| vqpL2+r128@t2w:4 (RQ2) | 973.5 / 975.4 | 1002.9 / 1001.0 | +2.9% | 1.607 -> 1.652 | 4.859 / 4.864 |
| hyb512pL2@t2w:6 | 1033.7 / 1038.1 | 1063.7 / 1063.6 | +2.7% | 1.708 -> 1.754 | 6.89-6.90 / 6.89 |
| vqpL1+r128@t1w:4 (RQ1) | 805.2 / 806.2 | 838.2 / 837.5 | +4.0% | 1.328 -> 1.382 | 3.790 / 3.794 |

- vs lazy-max-only e2e (results/E7/llama8b_magicvq_lazymax_e2e.md: 982 / 1046 / 815 tok/s) the extra gain is about +2%, +1.7%, +2.8%.
- On the 4090 the VQ kernel is ~15% of the draft round, so the kernel speed-up (1.1-1.5x) shows up as +3-4% e2e. The A100 kernel gain is larger (2.07-2.29x KST1, 1.65-1.73x KST2, octopus) and the VQ kernel is a larger share of the A100 draft step; e2e on A100 not measured yet.
