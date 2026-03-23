"""
RQ KV Cache for Speculative Decoding
=====================================
Residual Quantization (RQ) based KV cache that supports partial decoding
for draft/target model split in speculative decoding.

Draft model: decode first N stages (coarse approximation, fast)
Target model: decode all stages (accurate, slower)

Checkpoint format (from VQLLM):
  layers.{l}.attention.quantizer.{key|value}.{stage}.codebooks.{cb_idx}.weight: (C, D)
  layers.{l}.attention.quantizer.{key|value}.{stage}.gains: (num_pq_codebooks, C)

Checkpoint structure (K=12, C=256, D=256, PQ=4):
  - KV 전체를 flatten: (B, H_kv=8, Lk, D_head=128) → (B, Lk, 1024)
  - PQ split: (B, Lk, 4 subvectors, 256 dim each)
  - 각 subvector를 codebook[256, 256]에서 lookup
  - 12 residual stages 반복
"""

import re
import math
import torch
import torch.nn.functional as F


class RQKVCache:
    """
    RQ-based KV cache with hierarchical decode support.

    PQ 구조: KV의 (H_kv × D_head) = 1024차원을 num_pq_codebooks개의 subvector로 나눔.
    각 subvector는 codebook_dim차원이며, codebook에서 nearest entry를 찾아 index 저장.
    Residual quantization으로 여러 stage를 쌓아 정밀도 향상.

    Draft: 처음 N stages만 decode → 빠르지만 부정확
    Target: 모든 stages decode → 느리지만 정확
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.num_layers = 0
        self.num_stages_key = 0
        self.num_stages_value = 0
        self.num_pq_codebooks = 0  # PQ subvectors per stage
        self.codebook_size = 0     # C: entries per codebook
        self.codebook_dim = 0      # D: dimension per entry
        self.num_kv_heads = 0
        self.head_dim = 0

        # Codebooks: [layer][stage][pq_idx] → (C, D) tensor
        self.key_codebooks = []
        self.value_codebooks = []
        self.key_gains = []     # [layer][stage][pq_idx] → (C,)
        self.value_gains = []

        # Encoded indices: [layer] → (B, Lk, num_stages, num_pq_codebooks) int16
        self.key_indices = []
        self.value_indices = []

        self.initialized = False

    def load_codebooks(self, checkpoint_path, num_kv_heads=8, head_dim=128):
        """VQLLM 체크포인트에서 codebook 로드."""
        print(f"Loading RQ codebooks from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Parse structure from checkpoint keys
        key_stages = set()
        value_stages = set()
        pq_codebooks = set()
        layers = set()

        for k in ckpt.keys():
            m = re.match(
                r"layers\.(\d+)\.attention\.quantizer\.(key|value)\.(\d+)\.codebooks\.(\d+)\.", k)
            if m:
                layers.add(int(m.group(1)))
                if m.group(2) == "key":
                    key_stages.add(int(m.group(3)))
                else:
                    value_stages.add(int(m.group(3)))
                pq_codebooks.add(int(m.group(4)))

        self.num_layers = max(layers) + 1
        self.num_stages_key = len(key_stages)
        self.num_stages_value = len(value_stages)
        self.num_pq_codebooks = max(pq_codebooks) + 1

        sample_key = f"layers.0.attention.quantizer.key.0.codebooks.0.weight"
        self.codebook_size = ckpt[sample_key].shape[0]
        self.codebook_dim = ckpt[sample_key].shape[1]

        total_kv_dim = num_kv_heads * head_dim
        expected_pq = total_kv_dim // self.codebook_dim

        print(f"  Layers: {self.num_layers}")
        print(f"  Key stages: {self.num_stages_key}, Value stages: {self.num_stages_value}")
        print(f"  PQ codebooks: {self.num_pq_codebooks} (expected from dim: {expected_pq})")
        print(f"  Codebook: {self.codebook_size} entries × {self.codebook_dim} dim")
        print(f"  Codebook dtype: {ckpt[sample_key].dtype}")
        print(f"  KV structure: {num_kv_heads} heads × {head_dim} dim = {total_kv_dim}")

        # Load codebooks and gains
        for l in range(self.num_layers):
            layer_key_cbs = []
            layer_val_cbs = []
            layer_key_gains = []
            layer_val_gains = []

            for s in range(self.num_stages_key):
                stage_cbs = []
                stage_gains = []
                gains_key = f"layers.{l}.attention.quantizer.key.{s}.gains"
                gains_tensor = ckpt.get(gains_key)
                if gains_tensor is not None:
                    gains_tensor = gains_tensor.float().to(self.device)

                for cb in range(self.num_pq_codebooks):
                    cb_key = f"layers.{l}.attention.quantizer.key.{s}.codebooks.{cb}.weight"
                    cb_tensor = ckpt[cb_key].float().to(self.device)
                    stage_cbs.append(cb_tensor)
                    if gains_tensor is not None:
                        stage_gains.append(gains_tensor[cb])
                    else:
                        stage_gains.append(torch.ones(self.codebook_size, device=self.device))

                layer_key_cbs.append(stage_cbs)
                layer_key_gains.append(stage_gains)

            for s in range(self.num_stages_value):
                stage_cbs = []
                stage_gains = []
                gains_key = f"layers.{l}.attention.quantizer.value.{s}.gains"
                gains_tensor = ckpt.get(gains_key)
                if gains_tensor is not None:
                    gains_tensor = gains_tensor.float().to(self.device)

                for cb in range(self.num_pq_codebooks):
                    cb_key = f"layers.{l}.attention.quantizer.value.{s}.codebooks.{cb}.weight"
                    cb_tensor = ckpt[cb_key].float().to(self.device)
                    stage_cbs.append(cb_tensor)
                    if gains_tensor is not None:
                        stage_gains.append(gains_tensor[cb])
                    else:
                        stage_gains.append(torch.ones(self.codebook_size, device=self.device))

                layer_val_cbs.append(stage_cbs)
                layer_val_gains.append(stage_gains)

            self.key_codebooks.append(layer_key_cbs)
            self.value_codebooks.append(layer_val_cbs)
            self.key_gains.append(layer_key_gains)
            self.value_gains.append(layer_val_gains)

        self.initialized = True
        total_cb_params = (self.num_layers *
                           (self.num_stages_key + self.num_stages_value) *
                           self.num_pq_codebooks *
                           self.codebook_size * self.codebook_dim)
        print(f"  Total codebook params: {total_cb_params / 1e6:.1f}M")

        bpa_key = self.num_stages_key * math.log2(self.codebook_size) / self.codebook_dim
        bpa_val = self.num_stages_value * math.log2(self.codebook_size) / self.codebook_dim
        print(f"  BPA: key={bpa_key:.4f}, value={bpa_val:.4f}")

    def _flatten_kv(self, kv_tensor):
        """(B, H_kv, Lk, D_head) → (B, Lk, H_kv*D_head) → (B, Lk, num_pq, codebook_dim)"""
        B, H_kv, Lk, D_head = kv_tensor.shape
        # (B, Lk, H_kv * D_head)
        flat = kv_tensor.permute(0, 2, 1, 3).reshape(B, Lk, H_kv * D_head)
        # (B, Lk, num_pq, codebook_dim)
        return flat.view(B, Lk, self.num_pq_codebooks, self.codebook_dim)

    def _unflatten_kv(self, flat, B, H_kv, Lk, D_head):
        """(B, Lk, num_pq, codebook_dim) → (B, H_kv, Lk, D_head)"""
        # (B, Lk, H_kv * D_head)
        flat = flat.view(B, Lk, H_kv * D_head)
        # (B, Lk, H_kv, D_head) → (B, H_kv, Lk, D_head)
        return flat.view(B, Lk, H_kv, D_head).permute(0, 2, 1, 3)

    def encode(self, kv_tensor, layer_idx, kv_type="key"):
        """
        FP16 KV tensor → RQ indices.

        Args:
            kv_tensor: (B, H_kv, Lk, D_head)
        Returns:
            indices: (B, Lk, num_stages, num_pq_codebooks) int16
        """
        B, H_kv, Lk, D_head = kv_tensor.shape
        num_stages = self.num_stages_key if kv_type == "key" else self.num_stages_value
        codebooks = self.key_codebooks[layer_idx] if kv_type == "key" else self.value_codebooks[layer_idx]
        gains = self.key_gains[layer_idx] if kv_type == "key" else self.value_gains[layer_idx]

        # (B, Lk, num_pq, codebook_dim)
        x = self._flatten_kv(kv_tensor).float()
        residual = x.clone()

        all_indices = []

        for s in range(num_stages):
            stage_indices = []
            for pq in range(self.num_pq_codebooks):
                cb = codebooks[s][pq]   # (C, D)
                g = gains[s][pq]        # (C,)

                # Subvector: (B, Lk, D)
                sub = residual[:, :, pq, :]

                # Gain-shape nearest: maximize g_k*(sub·cb_k) - 0.5*g_k^2
                dots = torch.matmul(sub, cb.T)  # (B, Lk, C)
                scores = g.unsqueeze(0).unsqueeze(0) * dots
                scores -= 0.5 * (g ** 2).unsqueeze(0).unsqueeze(0)

                idx = scores.argmax(dim=-1)  # (B, Lk)
                stage_indices.append(idx)

                # Update residual
                selected_cb = cb[idx]              # (B, Lk, D)
                selected_g = g[idx]                # (B, Lk)
                reconstructed = selected_g.unsqueeze(-1) * selected_cb
                residual[:, :, pq, :] -= reconstructed

            # (B, Lk, num_pq)
            stage_idx = torch.stack(stage_indices, dim=-1)
            all_indices.append(stage_idx)

        # (B, Lk, num_stages, num_pq)
        indices = torch.stack(all_indices, dim=-2).to(torch.int16)
        return indices

    def decode(self, indices, layer_idx, kv_type="key", num_stages=None):
        """
        RQ indices → reconstructed KV tensor.

        Args:
            indices: (B, Lk, total_stages, num_pq) int16
        Returns:
            (B, H_kv, Lk, D_head) float16
        """
        max_stages = self.num_stages_key if kv_type == "key" else self.num_stages_value
        if num_stages is None:
            num_stages = max_stages
        num_stages = min(num_stages, max_stages, indices.shape[-2])

        codebooks = self.key_codebooks[layer_idx] if kv_type == "key" else self.value_codebooks[layer_idx]
        gains = self.key_gains[layer_idx] if kv_type == "key" else self.value_gains[layer_idx]

        B, Lk = indices.shape[0], indices.shape[1]

        # Accumulate: (B, Lk, num_pq, codebook_dim)
        result = torch.zeros(B, Lk, self.num_pq_codebooks, self.codebook_dim,
                             device=self.device, dtype=torch.float32)

        for s in range(num_stages):
            for pq in range(self.num_pq_codebooks):
                cb = codebooks[s][pq]
                g = gains[s][pq]

                idx = indices[:, :, s, pq].long()
                selected_cb = cb[idx]           # (B, Lk, D)
                selected_g = g[idx]             # (B, Lk)
                result[:, :, pq, :] += selected_g.unsqueeze(-1) * selected_cb

        # Unflatten: (B, Lk, num_pq, D) → (B, H_kv, Lk, D_head)
        return self._unflatten_kv(result, B, self.num_kv_heads, Lk, self.head_dim).half()

    def encode_and_store(self, key, value, layer_idx):
        """Encode K,V and store indices."""
        key_idx = self.encode(key, layer_idx, "key")
        value_idx = self.encode(value, layer_idx, "value")

        while len(self.key_indices) <= layer_idx:
            self.key_indices.append(None)
            self.value_indices.append(None)

        self.key_indices[layer_idx] = key_idx
        self.value_indices[layer_idx] = value_idx

    def get_kv_for_draft(self, layer_idx, num_stages=2):
        """Draft: 처음 N stages만 decode."""
        key = self.decode(self.key_indices[layer_idx], layer_idx, "key", num_stages)
        value = self.decode(self.value_indices[layer_idx], layer_idx, "value", num_stages)
        return key, value

    def get_kv_for_target(self, layer_idx, num_stages=None):
        """Target: 모든 stages decode."""
        key = self.decode(self.key_indices[layer_idx], layer_idx, "key", num_stages)
        value = self.decode(self.value_indices[layer_idx], layer_idx, "value", num_stages)
        return key, value

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self.key_indices) and self.key_indices[layer_idx] is not None:
            return self.key_indices[layer_idx].shape[1]
        return 0

    def reconstruction_error(self, key_orig, value_orig, layer_idx, num_stages=None):
        """Stage별 재구성 오차 측정."""
        key_recon = self.decode(self.key_indices[layer_idx], layer_idx, "key", num_stages)
        value_recon = self.decode(self.value_indices[layer_idx], layer_idx, "value", num_stages)

        key_err = (key_orig.float() - key_recon.float()).norm() / key_orig.float().norm()
        val_err = (value_orig.float() - value_recon.float()).norm() / value_orig.float().norm()
        return key_err.item(), val_err.item()

    def summary(self):
        """Cache 상태 요약."""
        seq_len = self.get_seq_length()
        bpa_key = self.num_stages_key * math.log2(self.codebook_size) / self.codebook_dim
        bpa_val = self.num_stages_value * math.log2(self.codebook_size) / self.codebook_dim

        print(f"RQKVCache Summary:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Stages: key={self.num_stages_key}, value={self.num_stages_value}")
        print(f"  Codebook: {self.codebook_size}C × {self.codebook_dim}D × {self.num_pq_codebooks} PQ")
        print(f"  KV heads: {self.num_kv_heads}, head_dim: {self.head_dim}")
        print(f"  BPA: key={bpa_key:.4f}, value={bpa_val:.4f}")
        print(f"  Seq length: {seq_len}")
        if seq_len > 0 and self.key_indices:
            B = self.key_indices[0].shape[0]
            idx_bytes = sum(
                self.key_indices[l].numel() * 2 + self.value_indices[l].numel() * 2
                for l in range(self.num_layers)
                if l < len(self.key_indices) and self.key_indices[l] is not None
            )
            fp16_bytes = (self.num_layers * B * self.num_kv_heads * seq_len *
                          self.head_dim * 2 * 2)
            print(f"  Index memory: {idx_bytes / 1e6:.1f} MB")
            print(f"  FP16 equivalent: {fp16_bytes / 1e6:.1f} MB")
            print(f"  Compression: {fp16_bytes / max(idx_bytes, 1):.1f}x")
