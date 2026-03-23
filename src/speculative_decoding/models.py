"""Model loading for QuantSpec.

Draft / Target 모델 설정 요약
────────────────────────────────────────────────────────────────────────────
  Target model (검증용, 정확도 우선)
    • 아키텍처: draft와 동일 (동일 model_name)
    • 가중치: FP16 (torch.float16)
    • KV cache: 현재 decoding.py에서는 HF DynamicCache (전체 FP, 양자화 없음)
    • 로드: AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)
    • device_map / use_safetensors / trust_remote_code / cache_dir: 아래와 동일

  Draft model (추론 속도/메모리 절감용)
    • 아키텍처: target과 동일 (동일 model_name)
    • 가중치: 4-bit NF4 (bitsandbytes)
      - load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    • KV cache: 현재 decoding.py에서는 target과 동일한 shared_past (전체 FP).
      논문의 "8bit 양자화 후 draft는 upper 4bit만 사용"은 kv_cache.HierarchicalKVCache에
      구현되어 있으나, 실제 speculative_decode 경로에는 연결되어 있지 않음.
────────────────────────────────────────────────────────────────────────────
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except Exception:
    BitsAndBytesConfig = None
    _BNB_AVAILABLE = False


def load_models(
    model_name: str,
    device: str = "cuda",
    draft_use_bnb_4bit: bool = True,
    cache_dir: str | None = None,
):
    """
    Load target (fp16) and draft (bnb 4-bit) models.
    QuantSpec: 동일 아키텍처, draft만 4-bit weight quantization.

    cache_dir: 지정 시 여기에 다운로드하고 이후 같은 경로에서 로드. None이면 HF 기본 캐시(~/.cache/huggingface/hub).
    """
    # 로컬 경로면 절대 경로로 변환 (상대 경로는 HF repo ID로 오인되어 에러 남)
    if "/" in model_name or os.sep in model_name:
        model_name = os.path.abspath(model_name)

    load_kw = {"trust_remote_code": True}
    if cache_dir is not None:
        load_kw["cache_dir"] = cache_dir

    print(f"Loading tokenizer for {model_name}" + (f" (cache: {cache_dir})" if cache_dir else "") + "...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kw)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Target model: fp16. use_safetensors=False: 로컬에 .safetensors 없고 .bin만 있으면 PyTorch 가중치 로드
    print("Loading target model (fp16)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=device,
        use_safetensors=False,
        **load_kw,
    ).eval()

    # Draft model: bitsandbytes 4-bit (NF4)
    if draft_use_bnb_4bit and _BNB_AVAILABLE:
        print("Loading draft model (bnb 4-bit NF4)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        draft_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            use_safetensors=False,
            **load_kw,
        ).eval()
    else:
        raise RuntimeError("bitsandbytes 4-bit required for QuantSpec draft model")

    # Report sizes
    target_bytes = sum(p.numel() * p.element_size() for p in target_model.parameters())
    draft_bytes = sum(p.numel() * p.element_size() for p in draft_model.parameters())
    draft_buf = sum(b.numel() * b.element_size() for b in draft_model.buffers())
    print(f"  Target model memory: {target_bytes / 1e9:.2f} GB")
    print(f"  Draft model memory:  {(draft_bytes + draft_buf) / 1e9:.2f} GB")

    return tokenizer, draft_model, target_model
