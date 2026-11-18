# E7: official MagicDec on our 4x RTX 4090 (TP=4, GPUs 4-7) vs our SnapKV self-spec (e2e_real.py)

Model: Llama-3.1-8B-Instruct bf16 (rev 0e9e39f) for both. PG-19 prompts from the official loader (Data/pg19, first 50 docs, skip 8000 tokens, 32672-token chunks, BOS first). Paper (arXiv 2408.11049, Llama-3.1-8B base, PG-19, prefill 32k, 8xH100): SnapKV gamma3 Omega 3.41-3.42 at B16 (speedup 1.17-1.18), gamma4 Omega 4.03-4.10 at B32; StreamingLLM-512 gamma3 Omega 2.82-3.09.

## 1. Official MagicDec (unmodified code @09cd671, torch 2.5.1 + flashinfer 0.1.6, --compile)

Exactly as the scripts report: 10 batches, batches 0-4 are warm-up, numbers = cumulative over batches 5-9. Prefill excluded (timer starts after encode). SD generation stops when the longest sequence in the batch has >= 80 new tokens or any sequence emits EOS; AR generates max_len - prefix_len = 96 tokens (stops on EOS). Omega = generated tokens / (verify steps x B). Speedup = SD tok/s / AR tok/s at the same B (paper's definition). Spread = min-max over the 5 measured batches (different prompts).

| B | draft | gamma | Omega (batch min-max) | SD tok/s (batch min-max) | AR tok/s | speedup | ms/round draft / verify / accept+host | AR step ms | gen/seq | co-tenant |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | SnapKV-4097 | 3 | 3.63 (3.57-3.69) | 649.9 (637.5-663.8) | 501.2 | **1.297** | 24.6 / 19.5 / 0.64 | 15.96 | 74.1 | False |
| 8 | SnapKV-4097 | 4 | 4.41 (4.21-4.54) | 650.9 (622.6-664.5) | 501.2 | **1.299** | 32.7 / 20.8 / 0.64 | 15.96 | 75.0 | False |
| 8 | StreamingLLM-513 | 3 | 3.01 (2.98-3.05) | 567.9 (559.8-572.5) | 501.2 | **1.133** | 22.3 / 19.4 / 0.82 | 15.96 | 72.3 | False |
| 8 | AR | - | - | - | 501.2 | 1.000 | - | 15.96 | 95 | False |
| 16 | SnapKV-513 | 3 | 3.27 (3.15-3.39) | 952.9 (917.7-985.8) | 617.0 | **1.544** | 24.3 / 30.0 / 0.62 | 25.93 | 69.4 | False |
| 16 | StreamingLLM-513 | 3 | 2.98 (2.84-3.12) | 820.9 (774.8-854.3) | 617.0 | **1.330** | 27.2 / 30.1 / 0.86 | 25.93 | 65.7 | False |
| 16 | AR | - | - | - | 617.0 | 1.000 | - | 25.93 | 95 | False |

## 2. Ours (e2e_real.py, unchanged) on the same PG-19 rows

B16 = official rows 80-95 (official B16 batch 5), B8 = rows 40-47 (official B8 batch 5). Fixed gen tokens/seq, EOS ignored, static batch (a finished sequence waits). Decode-only tok/s, median of 3 reps; speedup vs our AR in the same run. tau = mean accepted+1 per active (seq, round) (= Omega definition); tau_eff = gen/rounds.

| B | gen | draft | gamma | tau | tau_eff | decode tok/s | speedup | speedup if every row stayed active (tau x T_AR / T_round, derived) | ms/round draft / verify / accept / host | AR step ms | official batch-5 Omega / tok/s (same rows) | co-tenant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 256 | snapkv4096 | 3 | 3.69 | 3.37 | 826.8 | **1.393** | 1.528 | 34.0 / 31.2 / 0.04 / 0.02 | 26.96 | - | False |
| 16 | 256 | snapkv4096 | 4 | 4.48 | 4.13 | 848.5 | **1.430** | 1.550 | 45.3 / 32.5 / 0.04 / 0.02 | 26.96 | - | False |
| 16 | 256 | snapkv2048 | 3 | 3.62 | 3.32 | 864.8 | **1.457** | 1.587 | 30.3 / 31.1 / 0.04 / 0.02 | 26.96 | - | False |
| 16 | 256 | snapkv2048 | 4 | 4.41 | 4.06 | 892.5 | **1.504** | 1.633 | 40.4 / 32.4 / 0.04 / 0.02 | 26.96 | - | False |
| 16 | 256 | AR | - | - | - | 593.4 | 1.000 | - | - | 26.96 | - | False |
| 8 | 256 | snapkv4096 | 3 | 3.71 | 3.46 | 590.1 | **1.235** | 1.327 | 26.8 / 20.0 / 0.04 / 0.02 | 16.75 | 3.64 / 654.5 | False |
| 8 | 256 | snapkv4096 | 4 | 4.45 | 3.88 | 539.7 | **1.130** | 1.296 | 35.8 / 21.6 / 0.04 / 0.02 | 16.75 | 4.38 / 648.2 | False |
| 8 | 256 | AR | - | - | - | 477.7 | 1.000 | - | - | 16.75 | - | False |
| 16 | 256 | snapkv512 | 3 | 3.45 | 3.05 | 830.9 | **1.400** | 1.587 | 27.5 / 31.1 / 0.04 / 0.02 | 26.96 | 3.39 / 985.8 | False |
| 16 | 256 | snapkv512 | 4 | 4.06 | 3.51 | 811.2 | **1.367** | 1.581 | 36.7 / 32.4 / 0.04 / 0.02 | 26.96 | - | False |
| 16 | 256 | AR | - | - | - | 593.4 | 1.000 | - | - | 26.96 | - | False |
| 16 | 80 | snapkv4096 | 3 | 3.64 | 3.08 | 757.4 | **1.269** | 1.501 | 33.9 / 31.1 / 0.04 / 0.03 | 26.81 | - | False |
| 16 | 80 | snapkv4096 | 4 | 4.43 | 4.00 | 825.8 | **1.384** | 1.533 | 45.1 / 32.3 / 0.04 / 0.02 | 26.81 | - | False |
| 16 | 80 | snapkv2048 | 3 | 3.56 | 2.76 | 720.4 | **1.207** | 1.559 | 30.1 / 31.1 / 0.04 / 0.02 | 26.81 | - | False |
| 16 | 80 | snapkv2048 | 4 | 4.31 | 3.64 | 801.5 | **1.343** | 1.591 | 40.2 / 32.3 / 0.04 / 0.03 | 26.81 | - | False |
| 16 | 80 | AR | - | - | - | 596.9 | 1.000 | - | - | 26.81 | - | False |
| 8 | 80 | snapkv4096 | 3 | 3.68 | 3.33 | 571.3 | **1.187** | 1.309 | 26.7 / 20.0 / 0.04 / 0.02 | 16.62 | 3.64 / 654.5 | False |
| 8 | 80 | snapkv4096 | 4 | 4.46 | 4.00 | 559.4 | **1.162** | 1.296 | 35.5 / 21.6 / 0.04 / 0.03 | 16.62 | 4.38 / 648.2 | False |
| 8 | 80 | AR | - | - | - | 481.4 | 1.000 | - | - | 16.62 | - | False |
| 16 | 80 | snapkv512 | 3 | 3.33 | 2.67 | 731.3 | **1.224** | 1.529 | 27.3 / 31.0 / 0.04 / 0.02 | 26.79 | 3.39 / 985.8 | False |
| 16 | 80 | snapkv512 | 4 | 4.03 | 3.20 | 745.1 | **1.248** | 1.570 | 36.4 / 32.3 / 0.04 / 0.02 | 26.79 | - | False |
| 16 | 80 | AR | - | - | - | 597.2 | 1.000 | - | - | 26.79 | - | False |

## 3. Failed / not run official points (memory, 4x24 GB)

- B16 SnapKV-4097 (official default budget) gamma 3 and 4: CUDA OOM in the first prefill (model.py _norm), 23.6 GiB in use per GPU (/data/lsh/logs/magicdec/snap4097_g{3,4}_b16.out).
- B16 SnapKV-2049 gamma 3 / 4: OOM in prefill (gen_draft_kv softmax) after 2 / 1 batches (snap2049_g{3,4}_b16.out). B16 SnapKV-513 gamma 3 ran with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator setting only, code unchanged).
- B32: not attempted; the target KV alone is 32 layers x 2 x 2 kv-heads x 128 x 2 B x 32 x 32768 = 34 GiB per GPU.
- B16 StreamingLLM-513 gamma 4: stopped to save GPU time (each official B16 run takes 15-30 min: eager 128-token chunked prefill, StreamingLLM re-prefills the whole prompt for the draft cache).

## 4. Reading

- Same rows, same model: official Omega and our tau agree within ~2% (B8 SnapKV-4096: 3.64 vs 3.68 at gamma 3, 4.38 vs 4.46 at gamma 4; B16 SnapKV-512 gamma 3: 3.39 vs 3.33 (gen 80) / 3.45 (gen 256)). Our SnapKV draft is not weaker than the official one in acceptance.
- Step times: official kernels (flashinfer + torch.compile max-autotune) are slightly faster: AR step 15.96 vs 16.62 ms (B8), 25.93 vs 26.8 ms (B16); SnapKV draft step 8.2 vs 8.9 ms (B8, 4096), 8.1 vs 9.1 ms (B16, 512); verify 19.5 vs 20.0 ms (B8 g3), 30.0 vs 31.0 ms (B16 g3).
- The decode speedup gap (official 1.30 / 1.54 vs our 1.19 / 1.22 at B8 SnapKV-4096 g3 / B16 SnapKV-512 g3, gen 80) is the batch accounting, not the draft: the official loop keeps every sequence generating until the longest one has >= 80 tokens (and stops the whole batch at the first EOS) and counts all tokens, so throughput ~ Omega x T_AR / T_round; ours gives every sequence exactly gen tokens and lets finished rows idle (tau_eff < tau). Recomputing ours with the official accounting from our measured tau / step times gives 1.31 / 1.53, matching the official 1.30 / 1.54.
- Official-vs-paper: StreamingLLM-513 g3 Omega 3.01 (B8) / 2.98 (B16) is inside the paper range 2.82-3.09; SnapKV g3 Omega 3.63 (budget 4097, B8) / 3.27 (budget 513, B16) vs paper 3.41-3.42 (base model, budget not stated in the table we quote).
