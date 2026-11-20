# E7 v14: Marlin INT4 draft weights, measured e2e (TP=4, 32k)

- Target: Llama-3.1-8B-Instruct bf16 weights + bf16 KV; verify = unchanged bf16 forward. Draft: compressed KV (same configs as v12/lazy-max)
  + linears (qkv / o / gate_up / down) as **Marlin FP16xINT4** (E4 GPTQ g128 checkpoint
  `/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048`), TP-sharded like the target (column-parallel qkv / gate_up,
  row-parallel o / down) by slicing the packed Marlin tensors (bit-exact vs the dense dequant, below). Norms / embedding / lm_head /
  attention kernels / all-reduces unchanged (bf16); the bf16 hidden state is cast to fp16 per linear and back.
  Context KV = the target's; in-round draft K/V come from the INT4 qkv (v12 semantics).
- Code: `experiments/26_magicvq/e2e_w4.py` (new), hook `--draft_w4 DIR` in `e2e_real.py` (rec `draft_w4`), checks
  `e2e_w4_test.py`, runs `v14_e2e_run.sh` (B16) / `v14_e2e_run2.sh` (TAGP / BSZ / XENV), tables `v14_summarize.py`.
- Hardware: 4x RTX 4090 GPUs 4-7 (shark), NCCL_P2P_DISABLE=1, CUDA graphs + tune_plan, all-active accounting, gen 256, 3 timed reps per
  process (median), order off1, on1, off2, on2 (off = bf16 draft weights, on = INT4). nvidia-smi before every process: no process on GPUs
  4-7 (others only on GPUs 0-2); in-run co-tenant record `cotenant_any = False` for every run.
- Raw: `/data/lsh/results_accept/magicvq/e2e_real/run_v14w4{,b15}{off,on}{1,2}_{syn_quote,pg19}_ctx32768_B{16,15}.json`,
  checks `v14_w4test.json`, summaries `v14w4_summary.json`, `v14w4b15_summary.json`.
- c = (draft ms/round / gamma) / AR ms/step (same process). "other" = accept + RQ encode + host per round.

## Memory (24 GB 4090)

- INT4 shards: 0.84 GiB/rank (alloc after weights 3.74 -> 4.61 GiB); device free after the B16 target cache: 3.39 -> 2.52 GiB.
- **B16 x 32k: AR and snapkv4096 fit (peak 20.73 GiB), vqpL2+r128 OOMs** at the draft CUDA-graph capture (`capture_end`: `RuntimeError:
  CUDA error: out of memory`; its RQ code planes are +2 GiB), also with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (run v14w4x, stopped after on1). hyb512 at B16 was not reached (run aborted at vqpL2); its bf16 peak equals vqpL2 (21.91 GiB).
- **B15 x 32k fits every config** (INT4 peak 21.63-21.89 GiB). The comparison below is therefore at B15 (both weight types); B16 only for snapkv.

## Correctness

- Shard slicing: kernel dequant of every per-rank Marlin shard (layers 0, 1, 15, 30, 31; qkv / o / gu / down; 4 ranks) == the same slice
  of the dense dequant `w_fp16.safetensors` (the v12 '@w4' source) **bit for bit** (0 differing elements).
- Draft logits, teacher-forced first 512 tokens of 3 syn_quote prompts (Marlin rows in chunks of 16 = decode kernel path), full vocab:

| prompt | Marlin vs dense-bf16 (v12) max / mean abs | last 8 pos max / mean | top-1 agree | INT4 (either) vs target bf16 mean abs | top-1 agree vs target |
|---|---|---|---|---|---|
| 0 | 3.63 / 0.044 | 0.38 / 0.035 | 0.988 | 0.541 (Marlin) / 0.542 (dense) | 0.836 / 0.840 |
| 1 | 5.22 / 0.047 | 0.84 / 0.038 | 0.994 | 0.566 (Marlin) / 0.566 (dense) | 0.863 / 0.865 |
| 2 | 3.58 / 0.038 | 0.20 / 0.027 | 0.992 | 0.518 (Marlin) / 0.517 (dense) | 0.828 / 0.830 |

  Marlin vs dense differ by fp16 activations + fp16 weights (dense v12 cast the fp16 dequant to bf16), ~12x smaller than INT4 vs bf16.
- Flag off = previous code path: v14w4off1 (B16 syn_quote) reproduces run_lazyoff1 bit for bit (AR out, all-active out and acc_log of all
  3 drafts). hyb512 varies between processes in both arms (v14w4off2 hyb differs from lazyoff1/off1; B15 off1 vs off2 tau 6.843 / 6.831;
  pg19 also INT4 on1 vs on2) - within a process all reps agree (`rep_out_differs` absent). Code path for the flag-off draft is unchanged,
  so this looks pre-existing (hyb compression / build), not caused by v14 (not investigated).
- Greedy exactness: every emitted token is the bf16 verify argmax; prompt-head check 0 changed in every config. Rows diverging from AR
  (bf16 M=1 vs M=gamma+1 near-ties, known) are the same class with bf16 and INT4 drafts: syn_quote B15 2-3 of 15 rows (first div 167,
  207, and 216 / 235 / 240) in both; pg19 13-15 of 15 rows in both arms (as before: v8e2eb pg19 15/16 rows, AR-overlap match ~0.24).

## syn_quote 32k, B15, gen 256, all-active (median of 3 reps per run)

runs: v14w4b15off1 (co-tenant False), v14w4b15on1 (co-tenant False), v14w4b15off2 (co-tenant False), v14w4b15on2 (co-tenant False)

| config | weights | tok/s r1 / r2 | speedup vs AR | draft ms/round | draft ms/step | verify ms/round | other ms/round | c | tau | peak GiB | rows != AR (first div) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | (off runs) | 592.1 / 592.7 | 1 | - | - | - | - | - | - | - | - |
| AR | (on runs) | 592.2 / 593.0 | 1 | - | - | - | - | - | - | - | - |
| snapkv4096:4 | bf16 | 852.2 / 850.5 | 1.439 / 1.435 | 43.67 / 43.76 | 10.92 / 10.94 | 31.24 / 31.31 | 0.06 / 0.07 | 0.431 / 0.432 | 4.258 / 4.258 | 18.87 / 18.87 | 2 [167, 207] / 2 [167, 207] |
| snapkv4096:4 | INT4 | 964.4 / 965.2 | 1.628 / 1.628 | 35.30 / 35.27 | 8.82 / 8.82 | 31.23 / 31.21 | 0.06 / 0.06 | 0.348 / 0.349 | 4.281 / 4.281 | 19.73 / 19.73 | 2 [167, 207] / 2 [167, 207] |
| vqpL2+r128@t2w:4 | bf16 | 941.1 / 938.5 | 1.589 / 1.583 | 45.84 / 46.00 | 11.46 / 11.50 | 31.23 / 31.29 | 0.45 / 0.45 | 0.452 / 0.454 | 4.863 / 4.863 | 20.76 / 20.76 | 3 [216, 167, 207] / 3 [216, 167, 207] |
| vqpL2+r128@t2w:4 | INT4 | 1064.1 / 1063.6 | 1.797 / 1.794 | 37.06 / 37.12 | 9.26 / 9.28 | 31.26 / 31.20 | 0.44 / 0.44 | 0.366 / 0.367 | 4.876 / 4.876 | 21.63 / 21.63 | 3 [235, 167, 207] / 3 [235, 167, 207] |
| hyb512pL2@t2w:6 | bf16 | 986.7 / 985.6 | 1.667 / 1.663 | 70.75 / 70.73 | 11.79 / 11.79 | 33.22 / 33.19 | 0.06 / 0.06 | 0.465 / 0.466 | 6.843 / 6.831 | 20.79 / 20.79 | 3 [167, 240, 207] / 3 [167, 240, 207] |
| hyb512pL2@t2w:6 | INT4 | 1124.2 / 1124.1 | 1.898 / 1.896 | 57.38 / 57.38 | 9.56 / 9.56 | 33.21 / 33.25 | 0.06 / 0.06 | 0.378 / 0.378 | 6.795 / 6.795 | 21.66 / 21.66 | 3 [167, 240, 207] / 3 [167, 240, 207] |

| config | tok/s bf16 -> INT4 | gain | draft ms/round bf16 -> INT4 | saved ms/step | tau bf16 -> INT4 | outputs same across the 2 processes per arm |
|---|---|---|---|---|---|---|
| snapkv4096:4 | 851.4 -> 964.8 | +13.3% | 43.71 -> 35.28 | 2.11 | 4.258 -> 4.281 | True |
| vqpL2+r128@t2w:4 | 939.8 -> 1063.8 | +13.2% | 45.92 -> 37.09 | 2.21 | 4.863 -> 4.876 | True |
| hyb512pL2@t2w:6 | 986.2 -> 1124.2 | +14.0% | 70.74 -> 57.38 | 2.23 | 6.837 -> 6.795 | False |

## pg19 32k, B15, gen 256, all-active (median of 3 reps per run)

runs: v14w4b15off1 (co-tenant False), v14w4b15on1 (co-tenant False), v14w4b15off2 (co-tenant False), v14w4b15on2 (co-tenant False)

| config | weights | tok/s r1 / r2 | speedup vs AR | draft ms/round | draft ms/step | verify ms/round | other ms/round | c | tau | peak GiB | rows != AR (first div) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | (off runs) | 579.8 / 580.3 | 1 | - | - | - | - | - | - | - | - |
| AR | (on runs) | 580.5 / 579.7 | 1 | - | - | - | - | - | - | - | - |
| snapkv4096:4 | bf16 | 912.1 / 912.8 | 1.573 / 1.573 | 43.66 / 43.67 | 10.91 / 10.92 | 31.17 / 31.11 | 0.06 / 0.06 | 0.422 / 0.422 | 4.554 / 4.554 | 19.07 / 19.07 | 14 [44, 2, 12, 44, 42, 29, 13, 20, 8, 28, 111, 48, 22, 5] / 14 [44, 2, 12, 44, 42, 29, 13, 20, 8, 28, 111, 48, 22, 5] |
| snapkv4096:4 | INT4 | 943.3 / 941.8 | 1.625 / 1.625 | 35.13 / 35.18 | 8.78 / 8.79 | 31.07 / 31.14 | 0.06 / 0.06 | 0.340 / 0.340 | 4.167 / 4.167 | 19.94 / 19.94 | 15 [44, 2, 12, 25, 42, 20, 13, 23, 20, 8, 28, 111, 48, 25, 5] / 15 [44, 2, 12, 25, 42, 20, 13, 23, 20, 8, 28, 111, 48, 25, 5] |
| vqpL2+r128@t2w:4 | bf16 | 820.8 / 821.7 | 1.416 / 1.416 | 46.36 / 46.35 | 11.59 / 11.59 | 31.18 / 31.12 | 0.42 / 0.42 | 0.448 / 0.448 | 4.267 / 4.267 | 20.99 / 20.99 | 15 [44, 2, 12, 25, 42, 114, 13, 23, 20, 8, 28, 52, 48, 25, 5] / 15 [44, 2, 12, 25, 42, 114, 13, 23, 20, 8, 28, 52, 48, 25, 5] |
| vqpL2+r128@t2w:4 | INT4 | 905.7 / 903.2 | 1.560 / 1.558 | 37.23 / 37.38 | 9.31 / 9.35 | 31.09 / 31.15 | 0.42 / 0.42 | 0.360 / 0.361 | 4.150 / 4.150 | 21.86 / 21.86 | 15 [44, 2, 12, 25, 42, 164, 13, 23, 20, 8, 28, 132, 48, 117, 5] / 15 [44, 2, 12, 25, 42, 164, 13, 23, 20, 8, 28, 132, 48, 117, 5] |
| hyb512pL2@t2w:6 | bf16 | 905.3 / 891.0 | 1.561 / 1.535 | 70.76 / 70.84 | 11.79 / 11.81 | 33.11 / 33.10 | 0.06 / 0.06 | 0.456 / 0.457 | 6.272 / 6.177 | 21.01 / 21.02 | 14 [44, 2, 12, 42, 164, 13, 23, 20, 8, 28, 126, 48, 13, 5] / 14 [44, 2, 12, 42, 164, 13, 23, 20, 8, 28, 111, 48, 13, 5] |
| hyb512pL2@t2w:6 | INT4 | 942.2 / 924.3 | 1.623 / 1.595 | 57.46 / 57.50 | 9.58 / 9.58 | 33.13 / 33.10 | 0.06 / 0.06 | 0.371 / 0.370 | 5.692 / 5.586 | 21.89 / 21.88 | 13 [44, 2, 12, 42, 184, 13, 23, 20, 8, 28, 77, 48, 5] / 14 [44, 2, 12, 42, 164, 13, 23, 20, 8, 28, 77, 48, 22, 5] |

| config | tok/s bf16 -> INT4 | gain | draft ms/round bf16 -> INT4 | saved ms/step | tau bf16 -> INT4 | outputs same across the 2 processes per arm |
|---|---|---|---|---|---|---|
| snapkv4096:4 | 912.4 -> 942.5 | +3.3% | 43.67 -> 35.15 | 2.13 | 4.554 -> 4.167 | True |
| vqpL2+r128@t2w:4 | 821.3 -> 904.5 | +10.1% | 46.35 -> 37.31 | 2.26 | 4.267 -> 4.150 | True |
| hyb512pL2@t2w:6 | 898.1 -> 933.2 | +3.9% | 70.80 -> 57.48 | 2.22 | 6.224 -> 5.639 | False |


## B16 syn_quote (INT4 only snapkv fits; vqpL2/hyb INT4 rows missing = OOM)

| config | weights | tok/s r1 / r2 | speedup vs AR | draft ms/round | draft ms/step | verify ms/round | other ms/round | c | tau | peak GiB | rows != AR (first div) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AR | (off runs) | 606.1 / 605.9 | 1 | - | - | - | - | - | - | - | - |
| AR | (on runs) | 606.0 / 606.6 | 1 | - | - | - | - | - | - | - | - |
| snapkv4096:4 | bf16 | 891.9 / 889.9 | 1.472 / 1.469 | 45.30 / 45.45 | 11.32 / 11.36 | 31.84 / 31.84 | 0.06 / 0.06 | 0.429 / 0.430 | 4.303 / 4.303 | 19.87 / 19.87 | 0 [] / 0 [] |
| snapkv4096:4 | INT4 | 1004.0 / 1003.6 | 1.657 / 1.654 | 36.99 / 36.96 | 9.25 / 9.24 | 31.70 / 31.77 | 0.06 / 0.06 | 0.350 / 0.350 | 4.315 / 4.315 | 20.73 / 20.73 | 0 [] / 0 [] |
| vqpL2+r128@t2w:4 | bf16 | 974.4 / 972.7 | 1.608 / 1.605 | 47.47 / 47.59 | 11.87 / 11.90 | 31.85 / 31.87 | 0.48 / 0.48 | 0.450 / 0.451 | 4.859 / 4.859 | 21.89 / 21.89 | 1 [216] / 1 [216] |
| hyb512pL2@t2w:6 | bf16 | 1037.8 / 1035.8 | 1.712 / 1.709 | 72.88 / 73.13 | 12.15 / 12.19 | 33.45 / 33.28 | 0.06 / 0.06 | 0.460 / 0.462 | 6.900 / 6.892 | 21.91 / 21.92 | 1 [216] / 1 [216] |

| config | tok/s bf16 -> INT4 | gain | draft ms/round bf16 -> INT4 | saved ms/step | tau bf16 -> INT4 | outputs same across the 2 processes per arm |
|---|---|---|---|---|---|---|
| snapkv4096:4 | 890.9 -> 1003.8 | +12.7% | 45.37 -> 36.98 | 2.10 | 4.303 -> 4.315 | True |


## INT4 tau retention: e2e vs v12 (dense-dequant tau eval, 36 samples)

| task | draft | gamma | e2e B15 tau bf16 -> INT4 (ratio) | v12 retention |
|---|---|---|---|---|
| syn_quote | snapkv4096 | 4 | 4.258 -> 4.281 (1.005) | 1.003 |
| syn_quote | vqpL2+r128 t2w | 4 | 4.863 -> 4.876 (1.003) | 0.999 |
| syn_quote | hyb512pL2 t2w | 6 | 6.837 -> 6.795 (0.994) | 0.994 |
| pg19 | snapkv4096 | 4 | 4.554 -> 4.167 (0.915) | 0.946 |
| pg19 | vqpL2+r128 t2w | 4 | 4.267 -> 4.150 (0.973) | 0.947 |
| pg19 | hyb512pL2 t2w | 6 | 6.224 -> 5.639 (0.906) | 0.905 |

e2e tau is on the first 15 prompts and (pg19) on trajectories that diverge from AR early and differ between drafts, so the pg19 ratios
are noisy; syn_quote matches v12 within 0.4%.

## Draft step breakdown (per rank, ms per draft step)

| part | bf16 | INT4 (Marlin) | source |
|---|---|---|---|
| e2e draft step (syn_quote B15, vqpL2) | 11.46-11.50 | 9.26-9.28 | e2e draft_s / rounds / gamma |
| e2e draft step (snapkv4096) | 10.92-10.94 | 8.82 | same |
| attention, all layers (vqpL2 / snapkv / hyb) | 3.25 / 2.60 / 3.50 | same | draft tune_plan attn_ms (graph) |
| 32 x (qkv + o + gate_up + down) at M=16 | 4.20 | 2.22 (incl. bf16<->fp16 casts) | e2e_w4_test micro (graph) |
|   qkv / o / gate_up / down | 0.72 / 0.39 / 2.10 / 1.09 | 0.58 / 0.38 / 0.79 / 0.57 | same |
| 66 all-reduces [16, 4096] bf16 | 2.53 | same | same |
| rest (embed, norms, act, lm_head + argmax) | ~1.5 | ~1.3 | difference |

Measured saving 2.1-2.3 ms/step (modeled assumption was 2.47). gate_up dominates the saving; o_proj (K=1024 per rank) gains nothing.
