# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project

**SpecRQ — nested weight-VQ self-speculative decoding (direction since 2026-11-05).**
Goal: raise real generation throughput of a weight-VQ model. The **target** uses all `L` additive
codebooks; the **draft** uses the first `k` (prefix), optionally with a draft-only low-precision
(`q`=8/4 bit) copy of those codebooks. Verify on RTX 4090 first; KV compression and serving come later.

- **Source of truth: Notion** https://app.notion.com/p/SpecRQ-40ed62f8ef5b80c3a7c7e51b9190c7a3
  (01 research questions/criteria, 02 weight design & training, **03 experiment plan = the only live
  experiment list (E0–E7)**, 04 KV/serving extension, 90 references + old plan).
- Start config **C1**: d=8, K=256, L=5 (5 bit/weight target), draft k=3 → 2, q=16 → 8/4. C2: d=8, K=4096, L=2.
- Models: Qwen2.5-0.5B-Instruct (pipeline check) → Llama-3.1-8B-Instruct (main). Training base: Drop-by-Drop-style
  prefix-aware additive-codebook (AQ) learning, not pure RQ.
- Success = measured end-to-end confirmed tokens/s vs. **optimized AR of the same VQ target**, under a target-quality
  constraint. Acceptance alone or modeled speed-up is not success. Speedup ≈ τ·T_AR / (T_draft_round + T_verify + T_overhead).
- `research.md` and the KV notes below describe the **previous KV-cache GSRQ direction** (kept for KV stage 3).

## Structure

```
SpecRQ/
├── CLAUDE.md            # this file
├── research.md          # research plan, hypotheses, experiment stages, status
├── VQLLM/               # git clone of https://github.com/eaglesung/VQLLM (codebook training; upstream code)
│   ├── vqllm/quantizer/quantize.py   # GSRQ quantizer: k-means init + EM gain/shape refinement
│   ├── vqllm/models/                 # torchtune model builders with VQ attention (_modules.py)
│   ├── recipes/run_rqllm_experiment.sh   # train (fast/ema/bp) + lm-eval entry point
│   └── recipes/config/*.yaml
├── specdec/             # self-speculative decoding harness (acceptance + speed-up)
│   ├── model.py         # own Llama-3 forward over Meta checkpoints (Meta key layout = codebook layout)
│   ├── kvstore.py       # draft/target caches + KV reps: FP16Rep (sanity), IntRep, RQRep (VQLLM codebook, prefix m)
│   ├── decoding.py      # AR baseline + speculative (greedy / rejection sampling), phase timers
│   ├── costmodel.py     # memory-traffic modeled speed-up vs (ctx, batch)
│   └── run.py           # CLI: python -m specdec.run --rep {fp16,int,rq} ...
├── specrq/              # (old) scalar formats for codeword experiments
├── results/             # JSON + tables (results/format_rq, results/specdec = old KV direction)
└── (weight-VQ, planned)
    ├── experiments/e0_kernel/    # T(M) of FP16 / INT4 Marlin / VQ C1,C2 linear kernels (Notion E0)
    ├── experiments/e1_qprec/     # q=16/8/4 kernel cost (E1)
    ├── experiments/e2_qwen_c1/   # Qwen C1 target AR + nested draft + exact greedy verify (E2)
    └── /data/lsh/ckpts/weightvq/ # quantized weight checkpoints (heavy, local disk)
```
The specdec/ and kvstore code above is from the KV direction; `decoding.py` loop/timers may be reused.
Git: repo initialized 2026-11-05 (`VQLLM/` is its own upstream clone and is git-ignored).

Keep upstream changes inside `VQLLM/` minimal and deliberate; put new code in `specrq/` and `experiments/`.

## Workflow (supervisor / executor)

- The user talks only to the main session (supervisor). It scopes tasks, verifies executor claims itself, commits,
  and updates Notion 03 at stage ends or decision changes. Executors follow `.claude/agents/executor.md`
  (no commits, `[V]/[I]` reports, ask before irreversible actions).
- Long jobs run in tmux: `tools/run_tmux.sh <name> <gpus|none> <env> -- <cmd...>` → session `sr-<name>`,
  log/meta/runner in `/data/lsh/logs/<name>.{log,meta.json,sh}`. `tools/jobs.sh` lists jobs and GPU usage.
- Result records: `tools/results.py::save_result` → `results/<E#>/<run>.json` with commit, versions, model revision.
- Weight-VQ env: `nestvq` (separate from `VQLLM_CUDA`). Code: `specrq/dbd/` (vendored Drop-by-Drop/AQLM mirror,
  original at `/data/lsh/src/DropbyDrop2026` — keep it).

## Environment

```bash
source /hai/anaconda3/etc/profile.d/conda.sh && conda activate VQLLM_CUDA
export HF_DATASETS_OFFLINE=1      # wikitext-2 is in the local HF cache
# WANDB_DIR=/data/lsh/wandb is set automatically (conda env config vars)
```

**Storage rule:** `/hai` is the NAS (`165.132.146.29:/volume1/hai`) and `/data` is the local disk. Code and small results stay
under `/hai/home/lsh/SpecRQ`. Heavy outputs (codebook ckpts, wandb, scratch) go to **`/data/lsh/`**.
`/data/home` is not writable for us.

- Env `/hai/home/lsh/.conda/envs/VQLLM_CUDA`: python 3.11, torch 2.4.1+cu121, torchtune (pinned commit),
  lm_eval 0.4.2, cuml 25.6, scikit-learn 1.3.2, transformers 5.3.0, datasets 4.8.4.
- `vqllm` is an editable install pointing at `SpecRQ/VQLLM` (`pip install -e . --no-deps`).
- **Do not run `VQLLM/setup_env.sh`**. It installs Miniconda, writes `/workspace/.tmux.conf`, and contains hardcoded HF/W&B tokens.
- GPUs: 8× RTX 4090 (24 GB). GPU 2/3 are often used by other jobs, so check `nvidia-smi` first.

## Paths

| What | Path | Notes |
|---|---|---|
| Base models | `/data/models/{llama2_7b, llama3_3b, llama3_8b, mistral_7b}` | local disk (`/data`, 23 TB free) |
| Weight-VQ models | `/data/models/qwen2.5_0.5b_instruct` (rev 7ae5576), `/data/models/llama3.1_8b_instruct` (rev 0e9e39f, 4 safetensors shards, 15 GB) | `REVISION.txt` in each dir; downloaded 2026-11-05 |
| Llama-3-8B Meta weights | `/data/models/llama3_8b/consolidated.00.pth`, `tokenizer.model` | `original/` holds only `config.json`; HF safetensors shards absent |
| Llama-3.2-3B | `/data/models/llama3_3b` (HF safetensors + `original/` Meta) | 28 layers, 8 KV heads; **no builder in the current VQLLM clone** |
| Llama-2-7B / Mistral-7B | HF `.bin` + safetensors | MHA (Llama-2) is the planned primary model |
| Codebook output | `/data/lsh/ckpts/{dataset}/{config}/` | `CKPT_ROOT_DIR` in `run_rqllm_experiment.sh` + eval yaml `output_dir` (local edits in the clone) |
| W&B logs | `/data/lsh/wandb` | via `WANDB_DIR`; use `WANDB_MODE=offline` for smoke tests |
| HF cache | `~/.cache/huggingface` (NAS, ~87 GB, mostly `hub/`) | left in place |
| Model dir used by recipes | `BASE_CKPT_DIR=/data/models`, `MODEL_CKPT_DIR=$BASE_CKPT_DIR/$MODEL_NAME` | script overrides the yaml defaults (`/data/models/llama3_8b/original/...`) |

Previously trained codebooks (Llama-3.2-3B, `D∈{16,32,128}`) were deleted, so codebooks must be retrained. Target model: **Llama-3-8B**.
Checkpoint dir naming: `{both|key|value}_quant_Key_K[S]C[C]D[D]_Value_..._SubVec{n}_Stages{n}/meta_model.pt`
(Meta models) or `hf_model00{1,2}.pt` (HF models). The file contains full model weights plus the quantizer state.

## Spec-decode harness notes (KV-cache direction, before 2026-11-05)

- **Main setting (since 2026-10-23): quantized target at ≈4 BPA** (RQ `--target full --target-stages <4 bit>`; scalar
  baselines: nested `--target full` and separate copies `--target sep4`). Reason: at long context the KV cache is
  quantized anyway, so an FP16 target is not the practical reference; lower targets (e.g. 3 bit) are also in scope.
  Always report target fidelity next to acceptance: `python -m specdec.ppl --ref` gives PPL, KL(exact‖target) and
  top-1 agreement. `--target fp16` (exact target, draft error only) is kept as a diagnostic.
  Weights: separate RTN copy `--draft-wbits {2,3,4}` keeps the target bf16; nested `n{2b}` quantizes the target.
- **Greedy only** (`--temperature 0`, the default). Rejection sampling exists in `decoding.py` but is unverified and not used.
- Batch 1 only for now. Both caches hold dequantized bf16 KV, so **measured wall-clock cannot show KV savings**.
  Use the modeled speed-up (`costmodel.py`) together with the measured tokens/iteration. A real speed-up needs an
  index-reading attention kernel (E6).
- Greedy spec output can diverge from AR output even when draft == target. Single-token and (γ+1)-token forwards
  give slightly different bf16 logits, which flips near-tie argmaxes. Report `first_divergence`, not just `greedy_match`.
- Llama-3-8B max context is 8192, and draft weights equal target weights. At batch 1 the weight traffic dominates,
  so KV-only drafting cannot win there.

## Codebook format (VQLLM KV checkpoint, old direction)

- Keys: `layers.{l}.attention.quantizer.{key|value}.{stage}.codebooks.{cb}.weight` → `(C, D)` unit-norm **shape**;
  `...{stage}.gains` → `(NSV, C)` fp16 **gain**. Effective codeword = `gain * shape` (GSRQ).
- KV is flattened per token to `H_kv*128` dims and split into `NSV = H_kv*128/D` sub-vectors; with `SubVec1`
  each sub-vector has its own codebook. Index rate = `log2(C)/D` bit/element per stage.
- Quantization point: **pre-RoPE** k/v projection output (`_modules.py`, before `pos_embeddings`).

## Gotchas

- **Key layout:** torchtune/Meta `wk` and HF `k_proj` differ by the RoPE row permutation
  (`hf = meta.view(H, hd/2, 2, dim).transpose(1, 2)`). Codebooks are trained on Meta-layout keys, so
  HF keys must be mapped with `x.view(..., H, 2, hd/2).transpose(-1, -2)` before encoding and mapped back after.
  Skipping this inflated full-stage key error from 1.3% to 44% in an earlier harness. Values are unaffected.
- With nearest-codeword search on `||r - g·shape||²`, the result is identical to VQLLM's `argmax 2g(r·shape) - g²`.
- Scale search for wide-range formats (posit, fp8) must span the format's dynamic range. Their dense region sits far
  below `max|v|`, and a clip search limited to [1/16, 1] unfairly penalizes them.
- Draft/target metrics: report the encoding cost (prefill and per-token) separately. Never claim system gains
  that exclude nearest-neighbor search time.
