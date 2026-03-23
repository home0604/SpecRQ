"""Benchmark and sanity check for QuantSpec."""

import torch

from models import load_models
from decoding import speculative_decode, autoregressive_decode
from dataset import load_dataset_prompts


def run_benchmark(
    model_name: str,
    dataset_name: str = "pg19",
    context_length: int = 4096,
    max_new_tokens: int = 90,
    K: int = 5,
    temperature: float = 0.0,
    num_samples: int = 10,
    device: str = "cuda:1",
    cache_dir: str | None = None,
    use_hierarchical_kv: bool = False,
):
    """
    QuantSpec 논문 Table 3 재현을 위한 벤치마크.

    논문 설정:
      - Llama-2-7B-32K-Instruct
      - PG19, Multi-LexSum 데이터셋
      - context_length: 4k, 8k, 32k
      - output: 90 tokens
      - group_size: 128, residual_length: 256
      - batch_size: 1
    """
    print("=" * 70)
    print(f"QuantSpec Benchmark")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Context length: {context_length}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  K (speculation length): {K}")
    print(f"  Temperature: {temperature}")
    print(f"  Num samples: {num_samples}")
    print(f"  Device: {device}")
    if cache_dir:
        print(f"  Cache dir: {cache_dir}")
    if use_hierarchical_kv:
        print(f"  Hierarchical KV cache: ON (draft=upper 4bit, target=8bit)")
    print("=" * 70)

    # Load models (cache_dir 지정 시 해당 경로에 다운로드/캐시 후 재사용)
    tokenizer, draft_model, target_model = load_models(
        model_name, device=device, cache_dir=cache_dir
    )

    # 모델 최대 시퀀스 길이 초과 방지 (해당 경고/인덱싱 에러 방지)
    model_max_length = getattr(
        target_model.config, "model_max_length", None
    ) or getattr(target_model.config, "max_position_embeddings", None)
    if model_max_length is None:
        model_max_length = getattr(tokenizer, "model_max_length", 32768)
    if model_max_length > 1000000:  # tokenizer 기본값인 1e9 등
        model_max_length = 32768
    if context_length > model_max_length:
        print(f"  [Cap] context_length {context_length} → model max {model_max_length}")
        context_length = model_max_length

    # Load dataset (model_max_length 넘겨서 long text encode 시 경고/초과 방지)
    prompts = load_dataset_prompts(
        dataset_name, tokenizer, context_length,
        num_samples=num_samples, max_new_tokens=max_new_tokens,
        model_max_length=model_max_length,
    )

    if not prompts:
        print("ERROR: No valid prompts loaded!")
        return

    # Warmup
    print("\nWarming up...")
    warmup_len = min(512, prompts[0].shape[1], model_max_length)
    warmup_ids = prompts[0][:, :warmup_len].to(device)
    _ = speculative_decode(tokenizer, draft_model, target_model,
                           warmup_ids, max_new_tokens=5, K=K, temperature=temperature,
                           use_hierarchical_kv=use_hierarchical_kv)
    _ = autoregressive_decode(tokenizer, target_model, warmup_ids,
                              max_new_tokens=5, temperature=temperature)
    torch.cuda.synchronize()

    # Run benchmark
    spec_results = []
    ar_results = []

    for idx, prompt_ids in enumerate(prompts):
        # 모델 max 길이 초과 시 잘라서 인덱싱 에러 방지
        if prompt_ids.shape[1] > model_max_length:
            prompt_ids = prompt_ids[:, :model_max_length].clone()
            print(f"\n--- Sample {idx + 1}/{len(prompts)} (seq_len={prompt_ids.shape[1]} truncated from longer) ---")
        else:
            print(f"\n--- Sample {idx + 1}/{len(prompts)} (seq_len={prompt_ids.shape[1]}) ---")
        prompt_ids = prompt_ids.to(device)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Speculative decoding
        spec = speculative_decode(
            tokenizer, draft_model, target_model,
            prompt_ids, max_new_tokens=max_new_tokens,
            K=K, temperature=temperature,
            use_hierarchical_kv=use_hierarchical_kv,
        )
        torch.cuda.synchronize()
        spec_results.append(spec)
        print(f"  [Spec] tokens={spec['tokens_generated']}, "
              f"accept={spec['acceptance_rate']:.2%}, "
              f"speed={spec['tokens_per_second']:.1f} tok/s, "
              f"time={spec['elapsed_seconds']:.3f}s")

        # Clear CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Autoregressive baseline
        ar = autoregressive_decode(
            tokenizer, target_model, prompt_ids,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        torch.cuda.synchronize()
        ar_results.append(ar)
        print(f"  [AR]   tokens={ar['tokens_generated']}, "
              f"speed={ar['tokens_per_second']:.1f} tok/s, "
              f"time={ar['elapsed_seconds']:.3f}s")

        speedup = ar["elapsed_seconds"] / max(spec["elapsed_seconds"], 1e-9)
        print(f"  Speedup: {speedup:.2f}x")

        # Greedy correctness check
        if temperature <= 0:
            spec_text = tokenizer.decode(spec["output_ids"][0, prompt_ids.shape[1]:],
                                         skip_special_tokens=True)
            ar_text = tokenizer.decode(ar["output_ids"][0, prompt_ids.shape[1]:],
                                       skip_special_tokens=True)
            match = (spec_text == ar_text)
            if not match:
                print(f"  [WARN] Greedy outputs differ!")
                print(f"    Spec: {spec_text[:100]}...")
                print(f"    AR:   {ar_text[:100]}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_accept = sum(s["acceptance_rate"] for s in spec_results) / len(spec_results)
    avg_spec_speed = sum(s["tokens_per_second"] for s in spec_results) / len(spec_results)
    avg_ar_speed = sum(a["tokens_per_second"] for a in ar_results) / len(ar_results)
    avg_spec_time = sum(s["elapsed_seconds"] for s in spec_results) / len(spec_results)
    avg_ar_time = sum(a["elapsed_seconds"] for a in ar_results) / len(ar_results)
    avg_speedup = avg_ar_time / max(avg_spec_time, 1e-9)

    print(f"  Dataset: {dataset_name}, Context: {context_length}")
    print(f"  Avg Acceptance Rate: {avg_accept:.2%}")
    print(f"  Avg Spec Speed: {avg_spec_speed:.1f} tok/s")
    print(f"  Avg AR Speed:   {avg_ar_speed:.1f} tok/s")
    print(f"  Avg Speedup:    {avg_speedup:.2f}x")
    print()
    print(f"  논문 참고 (QuantSpec, Llama-2-7B-32K-Instruct):")
    print(f"    PG19  4k: accept=92.46%, speedup=1.35x")
    print(f"    PG19  8k: accept=89.88%, speedup=1.44x")
    print(f"    Multi-LexSum 8k:  accept=91.23%, speedup=1.61x")
    print(f"    Multi-LexSum 32k: accept=91.16%, speedup=2.08x")

    return {
        "avg_acceptance_rate": avg_accept,
        "avg_speedup": avg_speedup,
        "avg_spec_speed": avg_spec_speed,
        "avg_ar_speed": avg_ar_speed,
        "spec_results": spec_results,
        "ar_results": ar_results,
    }


def run_sanity_check(
    model_name: str,
    device: str = "cuda:0",
    cache_dir: str | None = None,
    max_new_tokens: int = 20,
    K: int = 5,
) -> bool:
    """
    draft_model = target_model, temperature=0 으로 1회 실행.
    acceptance가 100%가 아니면 speculative loop 버그 → False 반환.
    """
    print("=" * 70)
    print("SANITY CHECK: draft=target, temperature=0 → acceptance must be 100%")
    print("=" * 70)

    tokenizer, draft_model, target_model = load_models(
        model_name, device=device, cache_dir=cache_dir
    )
    # draft = target 로 동일 모델 사용 (speculative loop 버그 검증용)
    draft_model = target_model
    print("  (sanity check: draft_model = target_model 사용)")

    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    spec = speculative_decode(
        tokenizer, draft_model, target_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        K=K,
        temperature=0.0,
    )

    rate = spec["acceptance_rate"]
    ok = rate >= 1.0 - 1e-6  # 100% (부동소수 오차 허용)

    print(f"  Acceptance rate: {rate:.2%}")
    if ok:
        print("  [OK] Sanity check passed (speculative loop consistent).")
    else:
        print("  [FAIL] Acceptance < 100% with draft=target, temp=0 → bug in speculative loop.")

    return ok
