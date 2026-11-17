"""E7 magicvq: CommVQ (Li et al., ICML 2025; official code UMass-Embodied-AGI/CommVQ @ d1f0377, cloned read-only to
/data/lsh/src/CommVQ) as a DRAFT KV quantizer for the original bf16 Llama-3.1-8B-Instruct.

Uses the official KeyCompressor.encode/decode and ValueCompressor.encode/decode methods unchanged (commvq/
compress_evaluation.py, imported from the clone; einops from /data/lsh/envs/commvq_pylib). Only the artifact loading
is done here: key codebooks from senfu/Llama-3.1-8B-Instruct-CommVQ-{1,2}bit-codebook (torch.load weights_only=True;
same tensor assembly as KeyCompressor.__init__), value-compressor weights (`model.layers.{l}.self_attn.value_vq_model.*`)
range-fetched from the safetensors of senfu/Llama-3.1-8B-Instruct-CommVQ-{1,2}bit (commvq_fetch.py). The base weights of those repos are NOT used (the
target stays /data/models/llama3.1_8b_instruct; see check_base_weights()).
K: pre-RoPE, HF -> RoPE-pair layout, per-token L2 prescale over all kv heads, residual VQ (R steps, 4096 codewords of
128 dims = one kv head, uint16 codes). V: learned MLP encoder -> binary codes (gumbel_softmax hard, as in the official
evaluation code; mode 'argmax' replaces the Gumbel sample by the argmax, deterministic) -> linear decoder `embedding`,
renormalized to the per-token prescale. Interface = kvtau KVQuant (quant / quant_pos / keep_first / sizes / d).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/data/lsh/envs/commvq_pylib")
sys.path.insert(0, "/data/lsh/src/CommVQ")
from commvq.compress_evaluation import KeyCompressor, ValueCompressor  # noqa: E402

EXT = "/data/lsh/models_ext"


def load_key_compressor(bits, layer, H=8, n_q_heads=32):
    d = f"{EXT}/Llama-3.1-8B-Instruct-CommVQ-{bits}bit-codebook"
    kc = KeyCompressor.__new__(KeyCompressor)
    torch.nn.Module.__init__(kc)
    kc.feat_dim, kc.num_key_value_heads, kc.layer_idx = H * 128, H, layer
    ng = len(glob.glob(f"{d}/{0:03d}_*.pt"))
    cbs, cts = [], []
    for i in range(ng):
        steps = torch.load(f"{d}/{layer:03d}_{i}.pt", map_location="cpu", weights_only=True)["steps"]
        cbs.append(torch.cat([x["clustering_centers"].unsqueeze(0) for x in steps]).unsqueeze(1))
        theta = torch.cat([x["theta"].transpose(0, 1).unsqueeze(0) for x in steps])
        cts.append(theta)
    kc.codebook = torch.cat(cbs, dim=1).half()  # [R, H, 4096, 128]; host copy in fp16 (cast to fp32 on GPU)
    kc.theta = torch.stack(cts)  # compact commutative parameters (for byte accounting)
    kc.num_key_value_groups = ng
    return kc


def load_value_compressor(bits, layer):
    from safetensors.torch import load_file
    d = f"{EXT}/Llama-3.1-8B-Instruct-CommVQ-{bits}bit"
    global _VCACHE
    if bits not in _VCACHE:
        _VCACHE[bits] = load_file(f"{d}/value_vq.safetensors")  # range-fetched by commvq_fetch.py
    pre = f"model.layers.{layer}.self_attn.value_vq_model."
    sd = {k[len(pre):]: v.float() for k, v in _VCACHE[bits].items() if k.startswith(pre)}
    vc = ValueCompressor(quant_bits=bits, feat_dim=1024)
    missing, unexpected = vc.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return vc.eval()


_VCACHE = {}
_KCACHE = {}
_VMODS = {}
_VGPU = {}


class CVQuant:
    def __init__(self, bits, mode="gumbel", dev="cuda", nl=32, seed=0):
        self.bits, self.mode, self.dev = bits, mode, dev
        if bits not in _KCACHE:  # fp32 key codebooks stay in host RAM (~5.9 / 11 GB); moved per layer on use
            _KCACHE[bits] = [load_key_compressor(bits, l) for l in range(nl)]
        self.kc = _KCACHE[bits]
        self._on_gpu = None
        if bits not in _VMODS:  # shared by all CVQuant objects of this rate; host copy fp16, fp32 on GPU per layer
            _VMODS[bits] = [load_value_compressor(bits, l).bfloat16() for l in range(nl)]
        self.vc = _VMODS[bits]
        _VCACHE.pop(bits, None)
        self.sizes, self.d, self.S = [2 ** bits], 1, 1  # bits_of() override below
        self.keep_first = 0
        self.gen = torch.Generator(device=dev).manual_seed(seed)

    def bits_k(self):
        R = self.kc[0].codebook.shape[0]
        return R * 12 / 128 + 16 / 1024  # uint16 index of 4096 codewords per 128 dims per step + fp16 prescale

    def bits_v(self):
        return self.vc[0].embedding.shape[0] / 1024 + 16 / 1024  # binary code bits per token / 1024 + prescale

    def cb_bytes(self):
        """fp16 bytes read per step: compact commutative key parameters (theta) + value decoder `embedding`."""
        n = sum(k.theta.numel() for k in self.kc) + sum(v.embedding.numel() for v in self.vc)
        return 2 * n

    def quant_pos(self, l, kv, x, levels, p0):
        out = self.quant(l, kv, x, levels)
        n = min(self.keep_first - p0, x.shape[1])
        if n > 0:
            for lv in out:
                out[lv][:, :n] = x[:, :n]
        return out

    @torch.no_grad()
    def quant(self, l, kv, x, levels, chunk=1024):
        """x [B, T, 8, 128] pre-RoPE HF layout -> {level: recon}; single level (any requested level maps to it)."""
        B, T, H, D = x.shape
        xf = x.float().reshape(B, T, H * D)
        y = torch.empty_like(xf)
        for i in range(0, T, chunk):
            xc = xf[:, i:i + chunk].contiguous()
            if kv == 0:
                kc = self.kc[l]
                if self._on_gpu is not kc:
                    if self._on_gpu is not None:
                        self._on_gpu.codebook = self._on_gpu.codebook_cpu
                    kc.codebook_cpu = getattr(kc, "codebook_cpu", kc.codebook)
                    kc.codebook = kc.codebook_cpu.to(self.dev, non_blocking=True).float()
                    self._on_gpu = kc
                code, pre = kc.encode(xc.clone())
                r = kc.decode(code, pre)  # [B, n, H*128] in RoPE-pair layout
                r = r.view(B, -1, H, 64, 2).transpose(-1, -2).reshape(B, -1, H * D)  # back to HF layout
            else:
                vc = self.vc[l]
                if _VGPU.get(self.bits) is not vc:
                    if _VGPU.get(self.bits) is not None:
                        _VGPU[self.bits].to("cpu").bfloat16()
                    vc.to(self.dev).float()
                    vc.CONST1, vc.CONST2 = vc.CONST1.to(self.dev), vc.CONST2.to(self.dev)
                    _VGPU[self.bits] = vc
                torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen, device=self.dev)))
                if self.mode == "argmax":
                    vc.quantize = _argmax_quantize.__get__(vc)
                try:
                    code, pre = vc.encode(xc)
                    r = vc.decode(code, pre)
                finally:
                    if "quantize" in vc.__dict__:
                        del vc.quantize
            y[:, i:i + chunk] = r
        rec = y.view(B, T, H, D).to(x.dtype)
        return {lv: rec for lv in levels}


def _argmax_quantize(self, x):
    B, S, E = x.shape
    x = x.reshape(B, S, self.n_e, self.code_range)
    x = torch.nn.functional.one_hot(x.argmax(-1), self.code_range).to(x.dtype)
    x[..., :2] += x[..., 2:3]
    return x[..., :2].flatten(2)


@torch.no_grad()
def check_base_weights(bits, ref="/data/models/llama3.1_8b_instruct"):
    """max |diff| between a sample of the CommVQ repo's base Llama tensors (range-fetched) and the original."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    d = f"{EXT}/Llama-3.1-8B-Instruct-CommVQ-{bits}bit"
    smp = load_file(f"{d}/base_sample.safetensors")
    rm = json.load(open(f"{ref}/model.safetensors.index.json"))["weight_map"]
    res = {}
    for k, ta in smp.items():
        with safe_open(f"{ref}/{rm[k]}", framework="pt") as b:
            tb = b.get_tensor(k)
        res[k] = {"max_abs_diff": (ta.float() - tb.float()).abs().max().item(), "dtype": str(ta.dtype),
                  "ref_dtype": str(tb.dtype), "shape": list(ta.shape)}
    return res
