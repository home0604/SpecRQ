"""E7 magicvq: fetch only the needed tensors of senfu/Llama-3.1-8B-Instruct-CommVQ-{1,2}bit by HTTP range requests
(safetensors header -> byte offsets), instead of the full 16-17 GB repos:
  * all `model.layers.{l}.self_attn.value_vq_model.*` tensors (the CommVQ value encoder / decoder), and
  * a sample of base-model tensors (embed, lm_head, layers 0/15/31 projections and norms) to compare with the original
    /data/models/llama3.1_8b_instruct weights (are the CommVQ base weights modified?).
Writes /data/lsh/models_ext/Llama-3.1-8B-Instruct-CommVQ-{b}bit/value_vq.safetensors and base_sample.safetensors.
Usage: python experiments/26_magicvq/commvq_fetch.py --bits 1
"""
import argparse
import json
import os
import struct
import urllib.request

import torch
from safetensors.torch import save_file

DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
BASE_SAMPLE = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"] + [
    f"model.layers.{l}.{n}" for l in (0, 15, 31) for n in (
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight")]


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    assert len(data) == b - a, (len(data), b - a)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    a = ap.parse_args()
    repo = f"senfu/Llama-3.1-8B-Instruct-CommVQ-{a.bits}bit"
    out_dir = f"/data/lsh/models_ext/Llama-3.1-8B-Instruct-CommVQ-{a.bits}bit"
    os.makedirs(out_dir, exist_ok=True)
    wm = json.load(open(f"{out_dir}/model.safetensors.index.json"))["weight_map"]
    want = {"value_vq": [k for k in wm if "value_vq_model" in k], "base_sample": [k for k in BASE_SAMPLE if k in wm]}
    headers = {}
    for f in sorted(set(wm.values())):
        url = f"https://huggingface.co/{repo}/resolve/main/{f}"
        n = struct.unpack("<Q", rng(url, 0, 8))[0]
        headers[f] = (url, 8 + n, json.loads(rng(url, 8, 8 + n)))
    for name, keys in want.items():
        tensors = {}
        for k in keys:
            url, base, hd = headers[wm[k]]
            info = hd[k]
            s, e = info["data_offsets"]
            buf = bytearray(rng(url, base + s, base + e))
            tensors[k] = torch.frombuffer(buf, dtype=DT[info["dtype"]]).reshape(info["shape"]).clone()
        save_file(tensors, f"{out_dir}/{name}.safetensors")
        print(name, len(tensors), sum(t.numel() for t in tensors.values()), "params", flush=True)


if __name__ == "__main__":
    main()
