"""Speculative decoding and autoregressive decoding implementations."""

import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from kv_cache import HierarchicalKVCache


def _past_to_tuple(past_key_values):
    """HF cache 또는 tuple → ( (k0,v0), (k1,v1), ... ) 형태."""
    if past_key_values is None:
        return None
    out = []
    for layer in past_key_values:
        k, v = _cache_layer_kv(layer)
        out.append((k, v))
    return tuple(out)


def _tuple_to_dynamic_cache(kv_tuple):
    """( (k,v), ... ) per layer → DynamicCache (HF 모델이 받는 형태)."""
    if kv_tuple is None or len(kv_tuple) == 0:
        return None
    return DynamicCache(ddp_cache_data=kv_tuple)


def _extract_cache_tail(past_key_values, n: int):
    """캐시에서 마지막 n개 위치의 KV만 추출 → append_tokens용 ( (k,v), ... ) per layer."""
    if past_key_values is None or n <= 0:
        return None
    out = []
    for layer in past_key_values:
        k, v = _cache_layer_kv(layer)
        # (B, H, S, D) → last n
        if k.shape[2] < n:
            out.append((k, v))
        else:
            out.append((k[:, :, -n:, :].contiguous(), v[:, :, -n:, :].contiguous()))
    return tuple(out)


def _cache_layer_kv(layer):
    """past_key_values 레이어에서 (key, value) 추출. HF 버전별 (k,v) 또는 (k,v,...) / Cache 객체 대응."""
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]
    if hasattr(layer, "key_states") and hasattr(layer, "value_states"):
        k = layer.key_states if isinstance(layer.key_states, torch.Tensor) else layer.key_states[0]
        v = layer.value_states if isinstance(layer.value_states, torch.Tensor) else layer.value_states[0]
        return k, v
    return layer[0], layer[1]


def _trim_cache(past, trim):
    """Trim `trim` tokens from the end of a KV cache, returning a DynamicCache."""
    new_cache = DynamicCache()
    for layer_idx, layer in enumerate(past):
        k, v = _cache_layer_kv(layer)
        new_cache.update(k[:, :, :-trim, :], v[:, :, :-trim, :], layer_idx=layer_idx)
    return new_cache


def _get_cache_len(cache):
    """DynamicCache의 현재 시퀀스 길이를 반환."""
    if hasattr(cache, 'get_seq_length'):
        try:
            return cache.get_seq_length(0)
        except TypeError:
            return cache.get_seq_length()
    return None


def _crop_cache(cache, new_len):
    """DynamicCache를 new_len으로 crop."""
    if hasattr(cache, 'crop'):
        cache.crop(new_len)
    else:
        cur_len = _get_cache_len(cache)
        if cur_len is not None and cur_len > new_len:
            trim = cur_len - new_len
            # Fallback: 새 cache 생성
            return _trim_cache(cache, trim)
    return cache


@torch.no_grad()
def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def speculative_decode(
    tokenizer, draft_model, target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 90,
    K: int = 5,
    temperature: float = 1.0,
    use_kv_cache: bool = True,
    use_hierarchical_kv: bool = False,
) -> dict:
    """
    QuantSpec Speculative Decoding with Shared KV Cache.

    논문의 핵심 설계를 반영:
      - Single prefill (target만) → 공유 DynamicCache
      - Draft는 공유 cache를 임시 확장하며 K개 토큰 생성 후 crop
      - Target verify가 공유 cache에 올바른 KV를 기록
      - k=0은 target logits 재사용 (guaranteed accept) → draft forward 절약
      - All-K-accepted 시 extra target forward 불필요
    """
    generated = input_ids.clone()
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_target_calls = 0
    total_draft_calls = 0
    tokens_generated = 0

    start_time = time.perf_counter()

    if use_kv_cache:
        # ── Single prefill: target model only ──
        target_out = target_model(input_ids, use_cache=True)
        shared_past = target_out.past_key_values
        # last_target_logits: P(next | prefix) — 다음 위치 예측
        last_target_logits = target_out.logits[:, -1:, :]
        total_target_calls += 1

        if use_hierarchical_kv:
            hcache = HierarchicalKVCache(group_size=128)
            hcache.prefill(_past_to_tuple(shared_past))
            shared_past = None  # 이후 hcache만 사용

        while tokens_generated < max_new_tokens:
            cache_len_before = hcache.total_length() if use_hierarchical_kv else _get_cache_len(shared_past)

            # ── Step 1: Draft generates K tokens ──
            # k=0: target logits 재사용 (guaranteed accept, draft forward 불필요)
            # k=1..K-1: draft model forward (shared cache 임시 확장)
            draft_probs_list = []
            draft_tokens_list = []

            if use_hierarchical_kv:
                draft_past = _tuple_to_dynamic_cache(hcache.get_kv_for_draft())
            else:
                draft_past = shared_past

            for k_idx in range(K):
                if tokens_generated + len(draft_tokens_list) >= max_new_tokens:
                    break

                if k_idx == 0:
                    # Target logits 재사용 → 이 토큰은 target과 동일 → guaranteed accept
                    draft_logits = last_target_logits[:, 0, :]
                else:
                    d_out = draft_model(
                        draft_tokens_list[-1].unsqueeze(-1),
                        past_key_values=draft_past,
                        use_cache=True,
                    )
                    draft_logits = d_out.logits[:, -1, :]
                    total_draft_calls += 1
                    if use_hierarchical_kv:
                        draft_past = d_out.past_key_values

                if temperature > 0:
                    probs = F.softmax(draft_logits / temperature, dim=-1)
                else:
                    probs = F.one_hot(draft_logits.argmax(-1), draft_logits.shape[-1]).float()

                next_token = sample_from_logits(draft_logits, temperature)
                draft_probs_list.append(probs)
                draft_tokens_list.append(next_token)

            num_draft = len(draft_tokens_list)
            if num_draft == 0:
                break

            total_draft_tokens += num_draft
            draft_tokens = torch.stack(draft_tokens_list, dim=-1)  # (B, num_draft)

            # ── Crop shared cache back (draft의 임시 확장 제거). Hierarchical는 draft_past만 쓰고 hcache는 그대로. ──
            if not use_hierarchical_kv and cache_len_before is not None:
                _crop_cache(shared_past, cache_len_before)

            # ── Step 2: Target model verifies all K tokens in one pass ──
            verify_past = (_tuple_to_dynamic_cache(hcache.get_kv_for_target()) if use_hierarchical_kv else shared_past)
            target_out_verify = target_model(
                draft_tokens,
                past_key_values=verify_past,
                use_cache=True,
            )
            total_target_calls += 1
            target_logits = target_out_verify.logits  # (B, num_draft, vocab)
            if use_hierarchical_kv:
                hcache.append_tokens(_extract_cache_tail(target_out_verify.past_key_values, num_draft))
            else:
                shared_past = target_out_verify.past_key_values

            # ── Step 3: Accept/reject ──
            # i=0: last_target_logits verifies draft_tokens[:, 0] (guaranteed accept)
            # i>=1: target_logits[:, i-1, :] verifies draft_tokens[:, i]
            accepted = 0
            for i in range(num_draft):
                if i == 0:
                    target_logits_i = last_target_logits[:, 0, :]
                else:
                    target_logits_i = target_logits[:, i - 1, :]

                if temperature > 0:
                    target_probs_i = F.softmax(target_logits_i / temperature, dim=-1)
                else:
                    target_probs_i = F.one_hot(target_logits_i.argmax(-1), target_logits_i.shape[-1]).float()

                draft_token_i = draft_tokens[:, i]
                draft_probs_i = draft_probs_list[i]

                if temperature <= 0:
                    # Greedy: accept if target agrees
                    if target_logits_i.argmax(-1)[0].item() == draft_token_i[0].item():
                        accepted += 1
                        generated = torch.cat([generated, draft_token_i.unsqueeze(-1)], dim=-1)
                        tokens_generated += 1
                        if draft_token_i[0].item() == tokenizer.eos_token_id:
                            break
                    else:
                        resampled = target_logits_i.argmax(-1, keepdim=True)
                        generated = torch.cat([generated, resampled], dim=-1)
                        tokens_generated += 1
                        break
                else:
                    p_target = target_probs_i[0, draft_token_i[0]].item()
                    p_draft = draft_probs_i[0, draft_token_i[0]].item()
                    accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
                    if torch.rand(1).item() < accept_prob:
                        accepted += 1
                        generated = torch.cat([generated, draft_token_i.unsqueeze(-1)], dim=-1)
                        tokens_generated += 1
                        if draft_token_i[0].item() == tokenizer.eos_token_id:
                            break
                    else:
                        adjusted = torch.clamp(target_probs_i - draft_probs_i, min=0)
                        adj_sum = adjusted.sum()
                        adjusted = adjusted / adj_sum if adj_sum > 0 else target_probs_i
                        resampled = torch.multinomial(adjusted, num_samples=1)
                        generated = torch.cat([generated, resampled], dim=-1)
                        tokens_generated += 1
                        break

            total_accepted_tokens += accepted

            # ── Step 4: Cache update & next iteration 준비 ──
            if accepted == num_draft:
                # All accepted → bonus token은 다음 iteration의 k=0이 됨
                # shared_past / hcache는 이미 target verify에서 K tokens의 올바른 KV를 가짐
                last_target_logits = target_logits[:, -1:, :]
                # Extra target forward 불필요!
            else:
                # Partial accept → cache를 accepted 지점까지 crop
                if use_hierarchical_kv:
                    hcache.truncate_to(cache_len_before + accepted)
                    last_past = _tuple_to_dynamic_cache(hcache.get_kv_for_target())
                else:
                    trim_count = num_draft - accepted
                    if trim_count > 0:
                        cur_len = _get_cache_len(shared_past)
                        if cur_len is not None:
                            _crop_cache(shared_past, cur_len - trim_count)
                    last_past = shared_past

                # Resampled/rejected token을 target에 통과시켜 올바른 KV + logits 확보
                last_token = generated[:, -1:]
                t_out = target_model(last_token, past_key_values=last_past, use_cache=True)
                last_target_logits = t_out.logits[:, -1:, :]
                total_target_calls += 1
                if use_hierarchical_kv:
                    hcache.append_tokens(_extract_cache_tail(t_out.past_key_values, 1))
                else:
                    shared_past = t_out.past_key_values

            if generated[0, -1].item() == tokenizer.eos_token_id:
                break
    else:
        # No KV cache version (slow baseline)
        while tokens_generated < max_new_tokens:
            seq_len_at_step_start = generated.shape[1]
            draft_ids = generated.clone()
            draft_probs_list = []

            for _ in range(K):
                if tokens_generated + len(draft_probs_list) >= max_new_tokens:
                    break
                outputs = draft_model(draft_ids)
                total_draft_calls += 1
                next_logits = outputs.logits[:, -1, :]
                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                else:
                    probs = F.one_hot(next_logits.argmax(-1), next_logits.shape[-1]).float()
                next_token = sample_from_logits(next_logits, temperature)
                draft_probs_list.append(probs)
                draft_ids = torch.cat([draft_ids, next_token.unsqueeze(-1)], dim=-1)

            num_draft = len(draft_probs_list)
            if num_draft == 0:
                break
            total_draft_tokens += num_draft
            draft_tokens = draft_ids[:, seq_len_at_step_start:]

            target_outputs = target_model(draft_ids)
            total_target_calls += 1
            target_logits = target_outputs.logits

            accepted = 0
            for i in range(num_draft):
                target_pos = seq_len_at_step_start - 1 + i
                target_logits_i = target_logits[:, target_pos, :]
                if temperature > 0:
                    target_probs_i = F.softmax(target_logits_i / temperature, dim=-1)
                else:
                    target_probs_i = F.one_hot(target_logits_i.argmax(-1), target_logits_i.shape[-1]).float()

                draft_token_i = draft_tokens[:, i]
                draft_probs_i = draft_probs_list[i]

                if temperature <= 0:
                    if target_logits_i.argmax(-1)[0].item() == draft_token_i[0].item():
                        accepted += 1
                        generated = torch.cat([generated, draft_token_i.unsqueeze(-1)], dim=-1)
                        tokens_generated += 1
                    else:
                        resampled = target_logits_i.argmax(-1, keepdim=True)
                        generated = torch.cat([generated, resampled], dim=-1)
                        tokens_generated += 1
                        break
                else:
                    p_target = target_probs_i[0, draft_token_i[0]].item()
                    p_draft = draft_probs_i[0, draft_token_i[0]].item()
                    accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
                    if torch.rand(1).item() < accept_prob:
                        accepted += 1
                        generated = torch.cat([generated, draft_token_i.unsqueeze(-1)], dim=-1)
                        tokens_generated += 1
                    else:
                        adjusted = torch.clamp(target_probs_i - draft_probs_i, min=0)
                        adj_sum = adjusted.sum()
                        adjusted = adjusted / adj_sum if adj_sum > 0 else target_probs_i
                        resampled = torch.multinomial(adjusted, num_samples=1)
                        generated = torch.cat([generated, resampled], dim=-1)
                        tokens_generated += 1
                        break

            total_accepted_tokens += accepted
            if accepted == num_draft:
                bonus_logits = target_logits[:, -1, :]
                bonus_token = sample_from_logits(bonus_logits, temperature)
                generated = torch.cat([generated, bonus_token.unsqueeze(-1)], dim=-1)
                tokens_generated += 1

            if generated[0, -1].item() == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "output_ids": generated,
        "tokens_generated": tokens_generated,
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "acceptance_rate": total_accepted_tokens / max(total_draft_tokens, 1),
        "total_target_calls": total_target_calls,
        "total_draft_calls": total_draft_calls,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_generated / max(elapsed, 1e-9),
    }


@torch.no_grad()
def autoregressive_decode(
    tokenizer, model, input_ids: torch.Tensor,
    max_new_tokens: int = 90, temperature: float = 1.0,
    use_kv_cache: bool = True,
) -> dict:
    """Standard autoregressive decoding baseline with KV cache."""
    generated = input_ids.clone()
    start_time = time.perf_counter()
    tokens_generated = 0

    if use_kv_cache:
        # Prefill
        outputs = model(input_ids, use_cache=True)
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        next_token = sample_from_logits(next_logits, temperature)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        tokens_generated += 1

        while tokens_generated < max_new_tokens:
            if next_token[0].item() == tokenizer.eos_token_id:
                break
            outputs = model(next_token.unsqueeze(-1), past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            next_token = sample_from_logits(next_logits, temperature)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            tokens_generated += 1
    else:
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_logits = outputs.logits[:, -1, :]
            next_token = sample_from_logits(next_logits, temperature)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            tokens_generated += 1
            if next_token[0].item() == tokenizer.eos_token_id:
                break

    elapsed = time.perf_counter() - start_time
    return {
        "output_ids": generated,
        "tokens_generated": tokens_generated,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_generated / max(elapsed, 1e-9),
    }
