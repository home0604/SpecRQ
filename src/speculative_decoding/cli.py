"""
QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache

논문 재현 구현 (arXiv:2502.10424)

핵심 구성요소:
  1. Draft model: bitsandbytes 4-bit 양자화 (동일 아키텍처)
  2. Target model: FP16 원본 모델
  3. Hierarchical INT4 KV cache: draft는 upper INT4만, target은 INT8 (upper+lower) 복원
  4. Double full-precision buffer: 최근 2G 토큰은 FP16 유지 → acceptance rate 향상
  5. KV cache 기반 incremental decoding

사용법:
  python speculative_decoding.py --model togethercomputer/Llama-2-7B-32K-Instruct \\
      --max-new-tokens 90 --K 5 --temperature 0 --context-length 4096 \\
      --dataset pg19
"""

import argparse

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark import run_benchmark, run_sanity_check


def main():
    parser = argparse.ArgumentParser(
        description="QuantSpec: Self-Speculative Decoding Benchmark")
    parser.add_argument("--model", type=str,
                        default="togethercomputer/Llama-2-7B-32K-Instruct",
                        help="Target model (HF name)")
    parser.add_argument("--dataset", type=str, default="pg19",
                        choices=["pg19", "wikitext", "c4", "gsm8k", "default"],
                        help="Dataset for benchmarking")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="Context length for prompts")
    parser.add_argument("--max-new-tokens", type=int, default=90,
                        help="Number of tokens to generate (논문: 90)")
    parser.add_argument("--K", type=int, default=5,
                        help="Draft speculation length (γ)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0=greedy)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of test samples (논문: 10)")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Device to use")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="모델 다운로드/캐시 디렉터리. 지정 시 여기서 로드·저장하고 재실행 시 재다운로드 안 함.")
    parser.add_argument("--no-kv-cache", action="store_true",
                        help="Disable KV cache (slow baseline)")
    parser.add_argument("--hierarchical-kv", action="store_true",
                        help="Use HierarchicalKVCache (draft=upper 4bit, target=8bit)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 2 samples and short context")
    parser.add_argument("--sanity-check", action="store_true",
                        help="draft=target, greedy: acceptance 100%% 아니면 speculative loop 버그로 종료")
    args = parser.parse_args()

    if args.sanity_check:
        ok = run_sanity_check(
            model_name=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
            max_new_tokens=30,
            K=args.K,
        )
        raise SystemExit(0 if ok else 1)

    if args.quick:
        args.num_samples = 2
        args.context_length = min(args.context_length, 512)
        args.max_new_tokens = 30

    run_benchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        K=args.K,
        temperature=args.temperature,
        num_samples=args.num_samples,
        device=args.device,
        cache_dir=args.cache_dir,
        use_hierarchical_kv=args.hierarchical_kv,
    )


if __name__ == "__main__":
    main()
