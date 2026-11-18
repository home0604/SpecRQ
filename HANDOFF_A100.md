# Handoff: continue E7 (MagicDec-VQ) on an A100 40GB server

Written 2026-11-18 on the original server ("shark", 8× RTX 4090 24GB). Goal on the new server: measure contexts **≥ 32k**
(64k, 128k), which do not fit on 4× 4090.

## Source of truth

- Notion project root: https://app.notion.com/p/SpecRQ-40ed62f8ef5b80c3a7c7e51b9190c7a3
  - 03 experiment plan (E0–E7) is the live experiment list.
- Notion **05 progress log** (all E7 rounds v1–v5 in Korean, with numbers): https://app.notion.com/p/410d62f8ef5b81cdb2b2eeb468033168
- Results in this repo: `results/E7/` (tables `*_tables.md`, records `*.json`, plots `*.png`).
- Code: `experiments/26_magicvq/` (E7) and `experiments/27_magicdec_repro/` (official MagicDec comparison, in progress).
- `CLAUDE.md` holds the general project notes. It was written for the old nested-weight-VQ direction and the 4090 server.

## What E7 is

- **Target:** original bf16 Llama-3.1-8B-Instruct with bf16 KV.
- **Draft:** the same weights with a compressed KV cache (self-speculation, MagicDec style).
- **Comparison:** eviction drafts (StreamingLLM, SnapKV = MagicDec's drafts) vs quantized-KV drafts (plain RQ post-RoPE, KIVI, GSRQ, CommVQ), each optionally with a bf16 recent window.
- **Hybrid draft:** SnapKV-selected tokens kept in bf16, all other tokens in RQ 2-bit (`hyb512pL2`).
- **Headline metric (user decision 2026-11-18): decode-only speedup vs AR**, measured. Prefill and compression cost go only in a secondary column.
- **Batch accounting (decided 2026-11-18, after the MagicDec repro): report both, headline = all-active.**
  - *all-active* (MagicDec official accounting, close to continuous batching): speedup = τ·T_AR/T_round. Our recomputation matches official MagicDec within 1%.
  - *static* (what `e2e_real.py` measures): fixed gen per sequence, finished rows idle (τ_eff < τ). Eviction drafts lose more here (τ_eff/τ 0.58–0.91 for SnapKV vs 0.88–0.96 for VQ/hybrid), so static inflates the VQ advantage.
  - v5 best VQ ÷ best SnapKV, static → all-active: syn_quote 32k B16 1.26 → 1.12, hotpotqa_q 16k 1.21 → 1.05, syn_quote B8 1.07 → 1.05, PG-19 0.92 → 0.94 (`results/E7/llama8b_magicvq_v5_accounting.md`, `experiments/26_magicvq/v5_accounting.py`).
  - **Measured (v5aa):** `e2e_real.py --accounting both` measures static and all-active in one run. VQ ÷ SnapKV all-active measured: syn_quote B16 1.108, gen512 1.065, B8 1.049, hotpotqa_q 16k **1.012** (derived was 1.049), PG-19 0.931 (`results/E7/llama8b_magicvq_v5aa_tables.md`). Headline = all-active **measured**. Use `--accounting both` on the A100.
  - all-active stops when the longest row reaches gen, so slow rows contribute only their early tokens (where acceptance may be higher), as in the official MagicDec loop.
  - Known issue: `csrc/vqattn_lut_rag.cu:95` float atomicAdd in shared memory makes VQ/hybrid draft logits non-deterministic → per-row lengths differ across reps (tokens identical on overlap; throughput spread 0.1–1%). Not fixed yet.
  - Draft-max reuse (unified max for verify): estimate is good (gap p99 0.5–2.2) but the saving bound is 0.3–0.5% of verify on TP4/4090 (`results/E7/llama8b_magicvq_maxreuse_tables.md`). Dropped.

## Results so far (4× RTX 4090, TP=4, PCIe, NCCL_P2P_DISABLE=1)

| round | what | key result |
|---|---|---|
| v1–v2 | τ vs KV bytes, PG-19 + question set | SnapKV leads on PG-19. On questions, VQ + window has higher τ at equal bytes. |
| v3 | step timing in the KV-dominated regime, TP=4 | SnapKV fastest at every (B, ctx). B·ctx ≤ ~512k tokens fits. |
| v4/v4b | eviction-unfavourable tasks (LongBench + synthetic), 36 samples, hybrid c measured | RQ2 drafts beat SnapKV only when the output copies far context: syn_quote 32k hybrid 1.117× [1.081, 1.152] of SnapKV, hotpotqa_q 16k 1.085×. |
| **v5** | **fully measured e2e**: real weights, prefill, compression, ragged static-batch decode | Decode speedup vs AR. syn_quote 32k B16: SnapKV 1.200×, RQ2+r128 1.399×, hybrid **1.510×**. hotpotqa_q 16k: SnapKV 0.937×, hybrid 1.134×. PG-19 32k: SnapKV **1.434×** > hybrid 1.320×. |
| v6 | draft-specific RQ codebooks (layer-local attention-output loss, `rqd_*.py`; codebooks `pt1,pt1w,pt2,pt2w` in the artifact tar) | Trained 2-bit raises τ on every task (hyb ×1.03–1.16). Computed vs SnapKV, B16: syn_quote 32k 1.192×, hotpotqa_q 32k 1.140×, gov_report 32k 1.118× (was 1.01×), PG-19 32k 0.964× (was 0.895×). Trained 1-bit +18–89% τ but still < old 2-bit. **Not yet e2e-measured: run e2e_real.py with `--cbfile .../cb_d8K256x2_post_cn_t2w.pt` first on the A100.** |
| MagicDec official (done, `24476a1`) | `experiments/27_magicdec_repro`, `results/E7/llama8b_magicdec_official_tables.md` | Same PG-19 rows on 4×4090: our SnapKV τ within ~2% of official Ω. Official speedup B8 SnapKV-4097 γ3 1.30×, B16 SnapKV-513 γ3 1.54× vs ours 1.19×/1.22× (gen 80); the gap is batch accounting (official keeps every row generating), ours recomputed that way = 1.31×/1.53×. Official kernels 4–12% faster. Official B16 SnapKV-4097@32k OOMs on 24 GB. Env `/data/lsh/envs/magicdec25` (torch 2.5.1, flashinfer 0.1.6). |

Comparison with the MagicDec paper (PG-19, 32k, Llama-3.1-8B):

| | SnapKV γ3 Ω | StreamingLLM-512 γ3 Ω | SnapKV B16×32k speedup |
|---|---|---|---|
| MagicDec paper | 3.41 | 2.82–3.09 | 1.17× (8×H100) |
| ours | 3.60 | 3.39 | 1.43× (measured, 4×4090) |

- Our τ is 5–10% higher than the paper's Ω.
- The paper's large speedups (2.0–2.5×) are at B64–128 and 64k–100k context. Its Table 3(c) uses **8×A100**, so the new server allows a direct comparison.

## Story / framing for the coursework talk (user decision 2026-11-18)

- Theory: `experiments/28_sd_theory/breakeven.py` → `results/E7/sd_theory_speedup_vs_c_alpha.png`, `sd_theory_tables.md`. S = max_γ τ/(γc+v), break-even α ≈ c, ceiling 1/c.
- **Framing:** a self-spec draft reads the full weights + all_reduce, so c has a floor (~0.43 at B16×32k TP4). The gain has to come from acceptance α. Eviction drafts lose α on hard / far-context tasks, VQ-KV drafts keep it → VQ wins there. Lowering draft weights too is out of scope (not a fair MagicDec comparison).
- **Caveat to measure:** the 2-bit VQ draft reads ~1/8 of the target KV, which grows with ctx, while SnapKV's budget is fixed. Per rank at B16 TP4: 32k ≈ 2 GB for both, 64k VQ ≈ 4 GB, 128k ≈ 8 GB. So c_VQ − c_SnapKV widens at 64k/128k, and VQ must win more α. The A100 runs test this trade-off (hybrid / 1-bit matter here).
- Always show the loss cases too (PG-19: SnapKV wins) so the task-dependent boundary is visible.
- **Planned figure (no GPU needed):** per-task points for SnapKV vs VQ/hybrid on the c–α theory map (measured c from step timing, α from τ), arrows SnapKV→VQ, from v4b/v5/v6 data.

## Setup on the new server

```bash
git clone -b e7-a100 https://github.com/home0604/SpecRQ.git && cd SpecRQ
scp shark:/data/lsh/handoff/specrq_e7_artifacts.tar* .      # 297 MB + sha256 (codebooks, prompts, PG-19, LongBench, raw tau/timing)
bash tools/setup_server.sh <DATA_ROOT> <MODELS_ROOT> ./specrq_e7_artifacts.tar
source .specrq_env; export TORCH_CUDA_ARCH_LIST=8.0          # A100 = sm_80 (code defaults to 8.9 via setdefault)
# if conda is not at /hai/anaconda3: export SPECRQ_CONDA_SH=<conda>/etc/profile.d/conda.sh
```

- **Paths.** The code hardcodes `/data/lsh`, `/data/models` and `/hai/home/lsh/SpecRQ`. `setup_server.sh` symlinks those paths to your roots if it can. If it can't, it rewrites the literals in the code; keep that change local and don't commit it.
- **Environment.** `env/nestvq.yml` and `env/nestvq_pip_freeze.txt` pin torch 2.4.1+cu121, triton 3.0.0 and transformers 4.46.3.
  - matplotlib is not in nestvq. The plots on shark used a base python (`/hai/anaconda3/bin/python3`).
- **Model.** `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`. The repo is gated: your own HF access is needed, and no workarounds are allowed.
- **Optional data.** pile_val (`/data/lsh/datasets/pile_val/val.jsonl`, AWQ's `mit-han-lab/pile-val-backup`) is needed only to re-calibrate or re-train codebooks. CommVQ artifacts and GSRQ codebooks are not in the tar (both were dropped: no fast kernel).
- **CUDA extensions.**
  - The kvq LUT kernels (`specrq/kvq/csrc/vqattn_lut.cu`, `experiments/26_magicvq/csrc/vqattn_lut_rag.cu`) use bf16 WMMA and should build for sm_80. They have only been tested on sm_89.
  - Build with the system gcc that nvcc 12.1 supports. On shark this needed gcc-11, see CLAUDE.md Gotchas.
  - Run `python experiments/26_magicvq/e2e_kern_test.py` first; it must print `ALL OK`.
- **NCCL.** On the 4090s, P2P hung, so all TP runs used `NCCL_P2P_DISABLE=1`. On A100 with NVLink, try P2P on first: it lowers the per-step all_reduce floor (~2–3 ms/step on 4090 PCIe). Record which one you used.

## A100 measurement plan (≥ 32k) — server has **2× A100 40GB** (user, 2026-11-18)

Memory budget:
- bf16 KV of Llama-3.1-8B is 128 KB per token (32 layers × 8 kv-heads × 128 × K,V × 2 B).
- **TP=2** (only option): about 8 GB weights per rank, roughly 26–28 GB left per rank for KV at 64 KB/token/rank, so **B·ctx ≈ 0.40–0.45M tokens** in total. That is about the same as shark's 4×4090 (~0.5M). The gain is longer ctx at small B, not more total tokens.
- Feasible points (target KV only): 32k B8–12, 64k B4–6, 128k B2–3. VQ/hybrid drafts need the extra draft cache below, so the largest B is a bit lower.
- Per-step weight time per rank is about 2× that of TP4. At small B the regime is less KV-dominated, so expect smaller speedups than at B16×32k on shark. Report T_AR, the KV share, and c alongside the speedup.
- `e2e_real.py` / `mgpu_bench.py` shard by `WORLD` (`HQA//WORLD`, `HKVA//WORLD`), so TP=2 should work (4 kv-heads per rank). This has never been run: smoke-test first and check the AR output against the 1-GPU reference.
- The RQ2 or hybrid draft cache adds about 1/8 of the target KV plus the window.
- Check `fit` before running. Report the largest B that fits all drafts and compare all drafts at that same B.

Steps:

1. **Kernel sanity:** `e2e_kern_test.py`, then re-tune split plans. `e2e_real.py` tunes by CUDA-graph timing.
2. **Fully measured e2e** (`experiments/26_magicvq/e2e_real.py`, decode-only headline):
   - TP=2, ctx ∈ {32k, 64k, 128k} × B ∈ {largest that fits all drafts; roughly 32k B8, 64k B4, 128k B2}. Also run 32k B8 to overlap with shark's v5 points (B8×32k syn_quote).
   - Drafts: AR, SnapKV {2048, 4096}, StreamingLLM 512, RQ2+r128 (`vqpL2+r128`), hybrid `hyb512pL2`. **Also both RQ2 drafts with the v6-trained codebook** (`--cbfile <DATA>/results_accept/magicvq/cb/cb_d8K256x2_post_cn_t2w.pt`). v6 raised τ on every task, but this has not been e2e-measured yet, so do it first. Add gov_report 32k: v6 turned it into a VQ win (computed).
   - Tasks:
     - `syn_quote`: synthetic far-context quoting. Scales to any ctx; haystack = PG-19 books.
     - PG-19: books long enough for 64k/128k exist in pg19-test.
     - `hotpotqa_q`: 32k+ uses distractor passages.
       - Bug: the 32k prompts are not reproducible across processes because of set order in `lb_tasks.lb_prompts`. Fix with `sorted(own)`; this changes the v4 32k prompts.
     - gov_report: only 13 samples reach ≥ 24k tokens.
   - Same AR kernels and graphs. Record the co-tenant check before and after each config.
3. **τ at 64k/128k** (`tau_eval.py`, `lb_tau.py`): one 40GB GPU should fit 64k (16 GB weights + 8 GB KV + harness), run the two GPUs as two parallel τ jobs. 128k (16 GB KV) probably needs both GPUs or KV offload in the τ harness. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (v6 32k τ OOMed without it on 24 GB).
4. **Official MagicDec** (`experiments/27_magicdec_repro`): the same-hardware comparison is planned on shark's 4×4090 once GPUs 4–7 are free (env `/data/lsh/envs/magicdec` there). Its Table 3(c) uses 8×A100, which 2×A100 cannot match. On the A100, optionally run the official code at TP=2 with small B as a second same-hardware check.
5. **Draft-specific codebook training (v6 done, layer-local):** if more τ is needed (e.g. for 1-bit), the planned next objective is an end-to-end loss on the target's greedy token (cross-entropy or margin on top-1), not full KL. Greedy acceptance depends only on top-1 agreement.

## Working rules carried over (from the user)

- **Language:** reply to the user in Korean.
- **Memory:** the Claude memory of the original server does not travel with git; the rules it held are copied into this section. Re-save the important ones in the new server's memory.
- **Roles:** the main session is supervisor. Executors implement and run, and they don't commit. The supervisor verifies, commits specific paths only (never `git add -a`), updates Notion 05, and reports.
- **Commit trailer:** end commit messages with the attribution lines your harness specifies.
- **Timing:** speed is measured only on GPUs with no other process. Record the check in each JSON.
  - Exception granted for v5 only: continue if a co-tenant appears mid-run, but flag the record.
  - Training and τ jobs may share GPUs.
- **GPU 0:** on shark the user excluded GPU 0. Ask about the new server's GPU policy and other users.
- **Storage:** heavy outputs and temp go to DATA_ROOT, never to a NAS home.
- **Gated repos:** no workarounds. Don't use other users' data directories.
- **Scope:**
  - No 3-bit targets. This rule is for the nested weight direction; E7 targets stay bf16.
  - Don't build the coursework slides until the user asks. E7 is the planned coursework presentation, so keep plots presentation-ready.
- **Notion:** add a section to the 05 page per round, with numbers and commit hashes.
