# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RQ-SpecDecode: Residual Quantized KV Cache with Oracle Speculative Decoding for Long-Context LLM Inference.

Combines three techniques:
1. **RQ KV Cache**: Learned codebook-based KV compression (from VQLLM) — draft uses fewer stages, target uses all
2. **Oracle Attention**: Unified global max (φ) eliminates online softmax correction (~20% attention compute saving)
3. **Speculative Decoding**: Draft (coarse RQ) generates tokens, target (full RQ) verifies

## Project Structure

```
spec/
├── src/                            # Core implementations
│   ├── speculative_decoding/       # QuantSpec speculative decoding pipeline
│   │   ├── cli.py                  # CLI entry point
│   │   ├── decoding.py             # Core spec decode + autoregressive algorithms
│   │   ├── benchmark.py            # Benchmark orchestration
│   │   ├── models.py               # Model loading (FP16 target, NF4 draft)
│   │   └── dataset.py              # Dataset loading (PG19, WikiText, C4, GSM8K)
│   ├── kv_cache/                   # KV cache implementations
│   │   ├── hierarchical.py         # QuantSpec INT4 hierarchical KV cache
│   │   └── rq_cache.py             # RQ codebook-based KV cache (draft/target stages)
│   └── kernels/                    # CUDA kernel packages
│       └── splitk_kernels -> ...   # symlink to flash/splitk_kernels
│
├── experiments/                    # Analysis experiments
│   ├── max_exponent/               # Max exponent approximation (δK, δQ error bounds)
│   │   ├── run.py                  # Main experiment
│   │   └── plot.py                 # Visualization
│   ├── rq_stages/                  # RQ stage-wise reconstruction error
│   │   └── run.py                  # Stage vs acceptance rate analysis
│   └── softmax/                    # Safe softmax experiments
│       └── run.py
│
├── benchmarks/                     # Performance benchmarks
│   ├── attention_kernels/          # Isolated attention kernel benchmarks
│   │   ├── bench_cuda_splitk.py    # CUDA single-kernel split-K (18.8% verification)
│   │   ├── bench_triton_splitk.py  # Triton 2-kernel split-K
│   │   ├── bench_attention.py      # Flash attention variants
│   │   ├── bench_triton.py         # Triton kernel microbenchmarks
│   │   └── bench_triton_4bit.py    # 4-bit Triton kernels
│   └── e2e_decode/                 # End-to-end model decode benchmarks
│       ├── bench_e2e.py            # Full model decode throughput
│       └── bench_single_layer.py   # Single layer decode (long context OK)
│
├── flash/                          # Attention kernel development
│   ├── splitk_kernels/             # Installable CUDA kernel package
│   │   ├── csrc/kernels.cu         # All CUDA kernels
│   │   ├── __init__.py             # Python API
│   │   └── setup.py               # pip install -e .
│   └── KIVI/                       # Vendored KIVI reference
│
├── VQLLM/                          # RQ codebook training (separate repo)
│
├── papers/                         # Reference papers
│   ├── QuantSpec.pdf
│   └── flashdecoding++.pdf
│
├── results/                        # Benchmark results & plots
│
├── CLAUDE.md                       # This file
└── .gitignore
```

## Key Commands

```bash
# ── Speculative Decoding ──
python src/speculative_decoding/cli.py --model /data/models/llama2_7b \
    --dataset pg19 --context-length 4096 --max-new-tokens 90 --K 5 --temperature 0

# ── Attention Kernel Benchmark (CUDA) ──
pip install -e flash/splitk_kernels  # First time only
python benchmarks/attention_kernels/bench_cuda_splitk.py --batch_size 1 8 --seq_lens 1024 4096

# ── E2E Decode Benchmark ──
python benchmarks/e2e_decode/bench_e2e.py --model /data/models/llama2_7b \
    --batch-size 8 --context-length 4096 --attn-backend sdpa splitk_oracle oracle_async

# ── Single Layer (long context, no OOM) ──
python benchmarks/e2e_decode/bench_single_layer.py --model /data/models/llama2_7b \
    --batch-size 1 --context-lengths 4096 32768 131072

# ── RQ Stage Analysis ──
python experiments/rq_stages/run.py --model /data/models/llama2_7b \
    --codebook /data/home/lsh2/ckpts/wikitext/both_quant_Key_K[12]C[256]D[256].../meta_model.pt

# ── Max Exponent Experiment ──
python experiments/max_exponent/run.py --model /data/models/llama2_7b --dataset wikitext
```

## CUDA Kernels (splitk_kernels package)

```python
import splitk_kernels as sk

sk.attention(Q, K, V, sm_scale, num_splits=8)              # Standard split-K
sk.attention_oracle(Q, K, V, global_max, sm_scale, ns)     # Single-CTA oracle
sk.attention_localmax_async(Q, K, V, sm_scale, ns)         # Multi-CTA (FlashDecoding)
sk.attention_oracle_async(Q, K, V, global_max, sm_scale, ns)  # Multi-CTA oracle (best)
```

## Key Findings

- FlashDecoding++ partial softmax sync overhead: **18.8%** (reproduced on A100)
- Oracle attention E2E improvement: **~5% at ctx=4K**, **~14% at ctx=128K**
- Multi-CTA (oracle_async) matches SDPA at batch≥8, within 5-7% at batch=1
- Decode attention is **memory-bound** (AI≈1, ridge≈200): KV cache compression > compute optimization

## Dependencies

PyTorch, transformers, bitsandbytes, datasets, triton. CUDA GPU required (A100 tested).
splitk_kernels: `cd flash/splitk_kernels && pip install -e .`
