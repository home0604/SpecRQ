# E7: lazy online max in the VQ LUT draft kernel, measured e2e (commit ce9668d flag)

syn_quote 32k, B16, gen 256, TP=4 (4090 GPUs 4-7, co-tenants none at start/end of every run), all-active accounting,
3 reps per run (median). Order off1, on1, off2, on2. Raw: /data/lsh/results_accept/magicvq/e2e_real/run_lazy{off,on}{1,2}_syn_quote_ctx32768_B16.json
Script: experiments/26_magicvq/lazy_e2e_run.sh

| config | tok/s off1 / off2 | tok/s on1 / on2 | gain | draft ms/round off -> on | speedup vs AR off -> on | tau off / on |
|---|---|---|---|---|---|---|
| AR | 606.4 / 606.1 | 606.7 / 605.8 | - | - | - | - |
| snapkv4096:4 (kernel unaffected) | 891.9 / 889.8 | 893.2 / 889.7 | +0.1% | 45.3-45.5 (same) | 1.47 | 4.303 |
| vqpL2+r128@t2w:4 | 974.1 / 973.3 | 983.3 / 980.1 | +0.8% | 47.5 -> 46.9 | 1.607 -> 1.620 | 4.859 / 4.862 |
| hyb512pL2@t2w:6 | 1038.6 / 1037.3 | 1047.1 / 1044.6 | +0.8% | 72.9 -> 72.1 | 1.712 -> 1.725 | 6.900 / 6.900 |
| vqpL1+r128@t1w:4 | 806.1 / 804.8 | 815.8 / 814.0 | +1.2% | 43.2 -> 42.3 | 1.329 -> 1.345 | 3.790 / 3.788 |

- Gain is small because the VQ LUT kernel is a small part of the draft round (draft dominated by weight reads + all_reduce):
  draft round -0.6 to -0.9 ms; kernel-only 1.09-1.18x.
- Exactness: outputs identical across reps within each mode (off1 == off2, on1 == on2). Off vs on, generated tokens agree with
  AR on the overlap exactly as before (vqpL2 row 5 first divergence at 216 in both modes, a known bf16 near-tie); some rows
  differ only in generated length / tau at the 3rd decimal (fp32 summation order of l changes draft logits in the last bits).
