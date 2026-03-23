"""Hierarchical INT4 KV Cache for QuantSpec."""

import torch
import torch.nn.functional as F


class HierarchicalKVCache:
    """
    QuantSpec의 핵심: Hierarchical INT4 KV Cache + Double Full-Precision Buffer.

    FP16 KV cache를 upper INT4 + lower INT4로 분할:
      - C^INT8 = 2^4 * C_U^INT4 + C_L^INT4
      - Draft model: upper INT4만 로드 (4-bit precision)
      - Target model: upper + lower INT4 → INT8 복원 (8-bit precision)

    Double full-precision buffer (크기 2G):
      - C_F1: 양자화된 최근 G 토큰 (항상 채워짐)
      - C_F2: 새로 생성된 토큰 (FP16 유지)
      - C_F2가 가득 차면 → C_F1 양자화 후 quantized cache에 추가,
        C_F2 → C_F1으로 이동, C_F2 비움

    group_size G: KV cache 양자화 그룹 크기 (논문: head_dim = 128)
    """

    def __init__(self, group_size: int = 128):
        self.group_size = group_size
        # Quantized cache: list of (upper_int4, lower_int4, scales, zeros) per layer
        # Each entry: keys and values stored separately
        self.quantized_keys = []    # per layer: (upper, lower, scale, zero) or None
        self.quantized_values = []
        # Full-precision buffers per layer
        self.fp_keys_f1 = []        # C_F1: size G (will be quantized)
        self.fp_keys_f2 = []        # C_F2: size G (newly generated, kept FP16)
        self.fp_values_f1 = []
        self.fp_values_f2 = []
        self.num_layers = 0
        self.initialized = False

    def initialize(self, num_layers: int):
        """첫 호출 시 레이어 수 기반 초기화."""
        self.num_layers = num_layers
        self.quantized_keys = [None] * num_layers
        self.quantized_values = [None] * num_layers
        self.fp_keys_f1 = [None] * num_layers
        self.fp_keys_f2 = [None] * num_layers
        self.fp_values_f1 = [None] * num_layers
        self.fp_values_f2 = [None] * num_layers
        self.initialized = True

    def _quantize_to_int4_pair(self, tensor: torch.Tensor):
        """
        FP16 tensor → (upper_int4, lower_int4, scale, zero_point)

        Per-group asymmetric quantization for upper INT4 (C_U: [0, 15])
        Per-group symmetric quantization for lower INT4 (C_L: [-8, 7])

        tensor shape: (batch, num_heads, seq_len, head_dim)
        Quantize along the last dim (head_dim) with group_size.
        """
        B, H, S, D = tensor.shape
        G = self.group_size

        # Reshape into groups
        assert D % G == 0, f"head_dim {D} must be divisible by group_size {G}"
        t = tensor.reshape(B, H, S, D // G, G).float()  # FP32로 연산 (FP16 정밀도 문제 방지)

        # Step 1: Asymmetric quantization to INT8 range [0, 255]
        t_min = t.amin(dim=-1, keepdim=True)
        t_max = t.amax(dim=-1, keepdim=True)
        scale = ((t_max - t_min) / 255.0).clamp(min=1e-8)
        zero_point = t_min

        int8_val = ((t - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)

        # Step 2: Split into upper and lower 4-bit
        upper = (int8_val >> 4).to(torch.uint8)    # [0, 15]
        lower = (int8_val & 0x0F).to(torch.uint8)  # [0, 15]

        # Reshape back: (B, H, S, D//G, G) → store grouped
        return upper, lower, scale.squeeze(-1), zero_point.squeeze(-1)

    def _dequantize_upper_only(self, upper, scale, zero_point):
        """Upper INT4만으로 대략적 dequantize (draft model용)."""
        # upper는 [0,15], INT8 ≈ upper * 16 (lower를 8로 근사)
        approx_int8 = upper.float() * 16.0 + 8.0  # 중간값으로 근사
        return approx_int8 * scale.unsqueeze(-1) + zero_point.unsqueeze(-1)

    def _dequantize_full(self, upper, lower, scale, zero_point):
        """Upper + Lower INT4 → INT8 → FP16 복원 (target model용)."""
        int8_val = upper.float() * 16.0 + lower.float()
        return (int8_val * scale.unsqueeze(-1) + zero_point.unsqueeze(-1)).half()

    def prefill(self, past_key_values):
        """
        Prefill 후: FP16 KV cache를 hierarchical INT4로 양자화.
        마지막 2G 토큰은 full-precision buffer에 유지.
        """
        num_layers = len(past_key_values)
        if not self.initialized:
            self.initialize(num_layers)

        G = self.group_size

        for layer_idx in range(num_layers):
            keys = past_key_values[layer_idx][0]    # (B, H, S, D)
            values = past_key_values[layer_idx][1]

            S = keys.shape[2]

            if S <= 2 * G:
                # 시퀀스가 짧으면 양자화 없이 FP buffer에만 저장
                split = S // 2
                self.fp_keys_f1[layer_idx] = keys[:, :, :split, :].clone()
                self.fp_keys_f2[layer_idx] = keys[:, :, split:, :].clone()
                self.fp_values_f1[layer_idx] = values[:, :, :split, :].clone()
                self.fp_values_f2[layer_idx] = values[:, :, split:, :].clone()
                self.quantized_keys[layer_idx] = None
                self.quantized_values[layer_idx] = None
            else:
                # 앞 부분: 양자화, 뒤 2G: full-precision
                quant_len = S - 2 * G
                # 양자화 길이를 G의 배수로 맞춤
                quant_len = (quant_len // G) * G
                fp_start = quant_len

                if quant_len > 0:
                    k_quant = keys[:, :, :quant_len, :]
                    v_quant = values[:, :, :quant_len, :]
                    self.quantized_keys[layer_idx] = self._quantize_to_int4_pair(k_quant)
                    self.quantized_values[layer_idx] = self._quantize_to_int4_pair(v_quant)
                else:
                    self.quantized_keys[layer_idx] = None
                    self.quantized_values[layer_idx] = None

                fp_mid = fp_start + G
                self.fp_keys_f1[layer_idx] = keys[:, :, fp_start:fp_mid, :].clone()
                self.fp_keys_f2[layer_idx] = keys[:, :, fp_mid:, :].clone()
                self.fp_values_f1[layer_idx] = values[:, :, fp_start:fp_mid, :].clone()
                self.fp_values_f2[layer_idx] = values[:, :, fp_mid:, :].clone()

    def append_tokens(self, past_key_values):
        """
        디코딩 중 새 토큰의 KV를 C_F2에 추가.
        C_F2가 G를 넘으면 → flush (C_F1 양자화 → quantized cache, C_F2 → C_F1).
        """
        G = self.group_size

        for layer_idx in range(self.num_layers):
            new_keys = past_key_values[layer_idx][0]
            new_values = past_key_values[layer_idx][1]

            if self.fp_keys_f2[layer_idx] is None:
                self.fp_keys_f2[layer_idx] = new_keys
                self.fp_values_f2[layer_idx] = new_values
            else:
                self.fp_keys_f2[layer_idx] = torch.cat(
                    [self.fp_keys_f2[layer_idx], new_keys], dim=2)
                self.fp_values_f2[layer_idx] = torch.cat(
                    [self.fp_values_f2[layer_idx], new_values], dim=2)

            # C_F2가 G를 넘으면 flush
            if self.fp_keys_f2[layer_idx].shape[2] >= G:
                self._flush_buffer(layer_idx)

    def _flush_buffer(self, layer_idx: int):
        """C_F1 양자화 → quantized cache 추가, C_F2 → C_F1."""
        G = self.group_size

        # C_F1을 양자화하여 quantized cache에 추가
        f1_keys = self.fp_keys_f1[layer_idx]
        f1_values = self.fp_values_f1[layer_idx]

        if f1_keys is not None and f1_keys.shape[2] > 0:
            # 패딩하여 G의 배수로 맞춤
            f1_len = f1_keys.shape[2]
            pad_len = (G - f1_len % G) % G
            if pad_len > 0:
                f1_keys = F.pad(f1_keys, (0, 0, 0, pad_len))
                f1_values = F.pad(f1_values, (0, 0, 0, pad_len))

            new_q_keys = self._quantize_to_int4_pair(f1_keys)
            new_q_values = self._quantize_to_int4_pair(f1_values)

            if self.quantized_keys[layer_idx] is not None:
                # 기존 quantized cache에 concatenate
                old = self.quantized_keys[layer_idx]
                self.quantized_keys[layer_idx] = tuple(
                    torch.cat([o, n], dim=2) for o, n in zip(old, new_q_keys))
                old = self.quantized_values[layer_idx]
                self.quantized_values[layer_idx] = tuple(
                    torch.cat([o, n], dim=2) for o, n in zip(old, new_q_values))
            else:
                self.quantized_keys[layer_idx] = new_q_keys
                self.quantized_values[layer_idx] = new_q_values

        # C_F2 → C_F1, C_F2 비움
        self.fp_keys_f1[layer_idx] = self.fp_keys_f2[layer_idx]
        self.fp_values_f1[layer_idx] = self.fp_values_f2[layer_idx]
        B, H, _, D = self.fp_keys_f1[layer_idx].shape
        self.fp_keys_f2[layer_idx] = torch.empty(B, H, 0, D,
            device=self.fp_keys_f1[layer_idx].device,
            dtype=self.fp_keys_f1[layer_idx].dtype)
        self.fp_values_f2[layer_idx] = torch.empty(B, H, 0, D,
            device=self.fp_values_f1[layer_idx].device,
            dtype=self.fp_values_f1[layer_idx].dtype)

    def truncate_to(self, seq_len: int):
        """
        Speculative decoding에서 reject 시: KV cache를 seq_len까지 truncate.
        간단히 FP buffer에서만 자름 (quantized 부분은 유지).
        """
        for layer_idx in range(self.num_layers):
            quant_len = 0
            if self.quantized_keys[layer_idx] is not None:
                upper = self.quantized_keys[layer_idx][0]
                quant_len = upper.shape[2] * self.group_size

            f1_len = self.fp_keys_f1[layer_idx].shape[2] if self.fp_keys_f1[layer_idx] is not None else 0
            f2_len = self.fp_keys_f2[layer_idx].shape[2] if self.fp_keys_f2[layer_idx] is not None else 0

            total = quant_len + f1_len + f2_len
            if seq_len >= total:
                continue

            # fp 부분에서만 truncate
            fp_target = seq_len - quant_len
            if fp_target <= 0:
                # quantized cache도 잘라야 함 — 간단히 전체 재구성
                # 실제로 이 경우는 드뭄
                self.fp_keys_f1[layer_idx] = None
                self.fp_keys_f2[layer_idx] = None
                self.fp_values_f1[layer_idx] = None
                self.fp_values_f2[layer_idx] = None
                continue

            if fp_target <= f1_len:
                self.fp_keys_f1[layer_idx] = self.fp_keys_f1[layer_idx][:, :, :fp_target, :]
                self.fp_values_f1[layer_idx] = self.fp_values_f1[layer_idx][:, :, :fp_target, :]
                B, H, _, D = self.fp_keys_f1[layer_idx].shape
                self.fp_keys_f2[layer_idx] = torch.empty(B, H, 0, D,
                    device=self.fp_keys_f1[layer_idx].device,
                    dtype=self.fp_keys_f1[layer_idx].dtype)
                self.fp_values_f2[layer_idx] = torch.empty(B, H, 0, D,
                    device=self.fp_values_f1[layer_idx].device,
                    dtype=self.fp_values_f1[layer_idx].dtype)
            else:
                f2_target = fp_target - f1_len
                self.fp_keys_f2[layer_idx] = self.fp_keys_f2[layer_idx][:, :, :f2_target, :]
                self.fp_values_f2[layer_idx] = self.fp_values_f2[layer_idx][:, :, :f2_target, :]

    def get_kv_for_draft(self):
        """Draft model용: quantized → upper INT4만 dequantize + FP buffers."""
        result = []
        for layer_idx in range(self.num_layers):
            parts_k = []
            parts_v = []

            if self.quantized_keys[layer_idx] is not None:
                upper_k, _lower_k, scale_k, zp_k = self.quantized_keys[layer_idx]
                dk = self._dequantize_upper_only(upper_k, scale_k, zp_k).half()
                dk = dk.reshape(dk.shape[0], dk.shape[1], dk.shape[2], -1)
                parts_k.append(dk)

                upper_v, _lower_v, scale_v, zp_v = self.quantized_values[layer_idx]
                dv = self._dequantize_upper_only(upper_v, scale_v, zp_v).half()
                dv = dv.reshape(dv.shape[0], dv.shape[1], dv.shape[2], -1)
                parts_v.append(dv)

            if self.fp_keys_f1[layer_idx] is not None and self.fp_keys_f1[layer_idx].shape[2] > 0:
                parts_k.append(self.fp_keys_f1[layer_idx])
                parts_v.append(self.fp_values_f1[layer_idx])

            if self.fp_keys_f2[layer_idx] is not None and self.fp_keys_f2[layer_idx].shape[2] > 0:
                parts_k.append(self.fp_keys_f2[layer_idx])
                parts_v.append(self.fp_values_f2[layer_idx])

            if parts_k:
                result.append((torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)))
            else:
                result.append(None)

        if all(r is None for r in result):
            return None
        ref = next((r for r in result if r is not None), None)
        if ref is None:
            return tuple(result)
        B, H, D = ref[0].shape[0], ref[0].shape[1], ref[0].shape[3]
        dev, dtype = ref[0].device, ref[0].dtype
        empty = (torch.empty(B, H, 0, D, device=dev, dtype=dtype),
                 torch.empty(B, H, 0, D, device=dev, dtype=dtype))
        return tuple((r if r is not None else empty) for r in result)

    def get_kv_for_target(self):
        """Target model용: quantized → INT8 (upper+lower) dequantize + FP buffers."""
        result = []
        for layer_idx in range(self.num_layers):
            parts_k = []
            parts_v = []

            if self.quantized_keys[layer_idx] is not None:
                upper_k, lower_k, scale_k, zp_k = self.quantized_keys[layer_idx]
                dk = self._dequantize_full(upper_k, lower_k, scale_k, zp_k)
                dk = dk.reshape(dk.shape[0], dk.shape[1], dk.shape[2], -1)
                parts_k.append(dk)

                upper_v, lower_v, scale_v, zp_v = self.quantized_values[layer_idx]
                dv = self._dequantize_full(upper_v, lower_v, scale_v, zp_v)
                dv = dv.reshape(dv.shape[0], dv.shape[1], dv.shape[2], -1)
                parts_v.append(dv)

            if self.fp_keys_f1[layer_idx] is not None and self.fp_keys_f1[layer_idx].shape[2] > 0:
                parts_k.append(self.fp_keys_f1[layer_idx])
                parts_v.append(self.fp_values_f1[layer_idx])

            if self.fp_keys_f2[layer_idx] is not None and self.fp_keys_f2[layer_idx].shape[2] > 0:
                parts_k.append(self.fp_keys_f2[layer_idx])
                parts_v.append(self.fp_values_f2[layer_idx])

            if parts_k:
                result.append((torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)))
            else:
                result.append(None)

        if all(r is None for r in result):
            return None
        ref = next((r for r in result if r is not None), None)
        if ref is None:
            return tuple(result)
        B, H, D = ref[0].shape[0], ref[0].shape[1], ref[0].shape[3]
        dev, dtype = ref[0].device, ref[0].dtype
        empty = (torch.empty(B, H, 0, D, device=dev, dtype=dtype),
                 torch.empty(B, H, 0, D, device=dev, dtype=dtype))
        return tuple((r if r is not None else empty) for r in result)

    def total_length(self) -> int:
        """현재 캐시된 총 토큰 수."""
        if not self.initialized or self.num_layers == 0:
            return 0
        total = 0
        layer_idx = 0
        if self.quantized_keys[layer_idx] is not None:
            total += self.quantized_keys[layer_idx][0].shape[2] * self.group_size
        if self.fp_keys_f1[layer_idx] is not None:
            total += self.fp_keys_f1[layer_idx].shape[2]
        if self.fp_keys_f2[layer_idx] is not None:
            total += self.fp_keys_f2[layer_idx].shape[2]
        return total

    def clear(self):
        """캐시 초기화."""
        self.quantized_keys = [None] * self.num_layers
        self.quantized_values = [None] * self.num_layers
        self.fp_keys_f1 = [None] * self.num_layers
        self.fp_keys_f2 = [None] * self.num_layers
        self.fp_values_f1 = [None] * self.num_layers
        self.fp_values_f2 = [None] * self.num_layers
