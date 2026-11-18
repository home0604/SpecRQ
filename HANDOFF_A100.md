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

## Results so far (4× RTX 4090, TP=4, PCIe, NCCL_P2P_DISABLE=1)

| round | what | key result |
|---|---|---|
| v1–v2 | τ vs KV bytes, PG-19 + question set | SnapKV leads on PG-19. On questions, VQ + window has higher τ at equal bytes. |
| v3 | step timing in the KV-dominated regime, TP=4 | SnapKV fastest at every (B, ctx). B·ctx ≤ ~512k tokens fits. |
| v4/v4b | eviction-unfavourable tasks (LongBench + synthetic), 36 samples, hybrid c measured | RQ2 drafts beat SnapKV only when the output copies far context: syn_quote 32k hybrid 1.117× [1.081, 1.152] of SnapKV, hotpotqa_q 16k 1.085×. |
| **v5** | **fully measured e2e**: real weights, prefill, compression, ragged static-batch decode | Decode speedup vs AR. syn_quote 32k B16: SnapKV 1.200×, RQ2+r128 1.399×, hybrid **1.510×**. hotpotqa_q 16k: SnapKV 0.937×, hybrid 1.134×. PG-19 32k: SnapKV **1.434×** > hybrid 1.320×. |
| v6 (in progress) | draft-specific RQ codebooks (layer-local attention-output loss, `rqd_*.py`) | Trained codebooks `pt1, pt1w, pt2, pt2w` exist; τ eval is running. The 32k eval OOMed on a 24GB card. |
| MagicDec official (in progress) | `experiments/27_magicdec_repro` | Env at `/data/lsh/envs/magicdec` on shark; not finished. |

Comparison with the MagicDec paper (PG-19, 32k, Llama-3.1-8B):

| | SnapKV γ3 Ω | StreamingLLM-512 γ3 Ω | SnapKV B16×32k speedup |
|---|---|---|---|
| MagicDec paper | 3.41 | 2.82–3.09 | 1.17× (8×H100) |
| ours | 3.60 | 3.39 | 1.43× (measured, 4×4090) |

- Our τ is 5–10% higher than the paper's Ω.
- The paper's large speedups (2.0–2.5×) are at B64–128 and 64k–100k context. Its Table 3(c) uses **8×A100**, so the new server allows a direct comparison.

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

## A100 40GB measurement plan (≥ 32k)

Memory budget:
- bf16 KV of Llama-3.1-8B is 128 KB per token (32 layers × 8 kv-heads × 128 × K,V × 2 B).
- TP=4 on 40GB: about 4 GB weights per rank, roughly 30 GB left for KV, so B·ctx ≈ 0.9–1.0M tokens (shark: ~0.5M).
- TP=8: about 2M tokens. TP must divide 8 kv-heads, so TP ≤ 8.
- The RQ2 or hybrid draft cache adds about 1/8 of the target KV plus the window.
- Check `fit` before running. Report the largest B that fits all drafts and compare all drafts at that same B.

Steps:

1. **Kernel sanity:** `e2e_kern_test.py`, then re-tune split plans. `e2e_real.py` tunes by CUDA-graph timing.
2. **Fully measured e2e** (`experiments/26_magicvq/e2e_real.py`, decode-only headline):
   - ctx ∈ {32k, 64k, 128k} × B ∈ {as large as fits; e.g. TP4: 32k B16/B32, 64k B8/B16, 128k B4/B8}.
   - Drafts: AR, SnapKV {2048, 4096}, StreamingLLM 512, RQ2+r128 (`vqpL2+r128`), hybrid `hyb512pL2`. Add trained codebooks (`pt2w` etc.) if v6 shows a τ gain.
   - Tasks:
     - `syn_quote`: synthetic far-context quoting. Scales to any ctx; haystack = PG-19 books.
     - PG-19: books long enough for 64k/128k exist in pg19-test.
     - `hotpotqa_q`: 32k+ uses distractor passages.
       - Bug: the 32k prompts are not reproducible across processes because of set order in `lb_tasks.lb_prompts`. Fix with `sorted(own)`; this changes the v4 32k prompts.
     - gov_report: only 13 samples reach ≥ 24k tokens.
   - Same AR kernels and graphs. Record the co-tenant check before and after each config.
3. **τ at 64k/128k** (`tau_eval.py`, `lb_tau.py`): one 40GB GPU fits 64k. 128k probably needs 2 GPUs or KV offload in the τ harness.
4. **Official MagicDec** (`experiments/27_magicdec_repro`): reproduce its Table 3(c) setting (8×A100, PG-19, StreamingLLM-512, B32–64, 32k–64k) and compare with our implementation on the same hardware.
5. **Draft-specific codebook training (v6):** if layer-local training doesn't raise τ enough, the planned next objective is an end-to-end loss on the target's greedy token (cross-entropy or margin on top-1), not full KL. Greedy acceptance depends only on top-1 agreement.

## Working rules carried over (from the user)

- **Language:** reply to the user in Korean.
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
