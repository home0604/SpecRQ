"""Dataset loading for QuantSpec benchmarking."""

import torch


def load_dataset_prompts(dataset_name: str, tokenizer, context_length: int,
                         num_samples: int = 10, max_new_tokens: int = 90,
                         model_max_length: int | None = None):
    """
    논문 설정에 맞게 데이터셋 로드.
    context_length 길이의 prompt를 생성하고, max_new_tokens만큼 생성.
    model_max_length: 모델 최대 시퀀스 길이. 넘으면 encode 시 truncation하여 경고/에러 방지.
    """
    from datasets import load_dataset

    # tokenizer.encode 시 길이 제한 (경고 "Token indices sequence length is longer than..." 방지)
    encode_max = None
    if model_max_length is not None and model_max_length > 0:
        encode_max = min(model_max_length, context_length + max_new_tokens + 256)

    def encode(text, return_tensors=None):
        kw = dict(add_special_tokens=False)
        if encode_max is not None:
            kw["max_length"] = encode_max
            kw["truncation"] = True
        out = tokenizer.encode(text, **kw) if return_tensors is None else tokenizer.encode(text, return_tensors=return_tensors, **kw)
        return out

    prompts = []

    if dataset_name == "pg19":
        print(f"Loading PG19 dataset (context_length={context_length})...")
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        count = 0
        for example in ds:
            text = example["text"]
            tokens = encode(text)
            if len(tokens) >= context_length + max_new_tokens:
                prompt_tokens = tokens[:context_length]
                prompts.append(torch.tensor([prompt_tokens]))
                count += 1
                if count >= num_samples:
                    break
        print(f"  Loaded {len(prompts)} samples from PG19")

    elif dataset_name == "wikitext":
        print(f"Loading WikiText-2 dataset (context_length={context_length})...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        # Concatenate all text
        all_text = "\n".join([t for t in ds["text"] if t.strip()])
        tokens = encode(all_text)
        # Split into chunks
        for i in range(0, len(tokens) - context_length - max_new_tokens, context_length):
            prompt_tokens = tokens[i:i + context_length]
            prompts.append(torch.tensor([prompt_tokens]))
            if len(prompts) >= num_samples:
                break
        print(f"  Loaded {len(prompts)} samples from WikiText-2")

    elif dataset_name == "c4":
        print(f"Loading C4 dataset (context_length={context_length})...")
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        count = 0
        for example in ds:
            text = example["text"]
            tokens = encode(text)
            if len(tokens) >= context_length + max_new_tokens:
                prompt_tokens = tokens[:context_length]
                prompts.append(torch.tensor([prompt_tokens]))
                count += 1
                if count >= num_samples:
                    break
        print(f"  Loaded {len(prompts)} samples from C4")

    elif dataset_name == "gsm8k":
        print(f"Loading GSM8K dataset...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        count = 0
        for example in ds:
            question = example["question"]
            prompt_text = f"Question: {question}\nAnswer: Let's think step by step.\n"
            tokens = encode(prompt_text)
            if len(tokens) > 0:
                prompts.append(torch.tensor([tokens]))
                count += 1
                if count >= num_samples:
                    break
        print(f"  Loaded {len(prompts)} samples from GSM8K")

    else:
        # 기본: 짧은 프롬프트
        print(f"Using default prompt (short context)...")
        text = "The future of artificial intelligence is"
        tokens = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
        prompts.append(tokens)

    return prompts
