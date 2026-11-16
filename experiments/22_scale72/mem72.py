"""E5 scale72: GPU memory footprint of the nested d8 K256 Qwen2.5-72B ckpt (from config shapes; no measurement).
Offload regime: k=2 prefix (2 bit/weight) resident on one RTX 4090 as draft, stages 3-4 (another 2 bit/weight) on CPU.
Usage: python experiments/22_scale72/mem72.py [--model /data/models/qwen2.5_72b_instruct] [--save]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

GiB = 2 ** 30
GPU_GIB = 24564 / 1024  # RTX 4090 total as reported by nvidia-smi (24564 MiB)


def footprint(cfg, k, d=8, nbits=8, L=4, ctx=(2048, 4096, 8192, 16384, 32768), runtime_gib=1.0):
    H, I, nl = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    hd = H // cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"] * hd
    V = cfg["vocab_size"]
    lin = [(H, H), (KV, H), (KV, H), (H, H), (I, H), (I, H), (H, I)]  # (out, in)
    n_lin = sum(o * i for o, i in lin) * nl
    b = {}
    b["codes_k"] = n_lin * k * nbits / d / 8  # packed: one nbits code per d weights per stage
    b["codebooks_k_bf16"] = len(lin) * nl * k * 2 ** nbits * d * 2
    b["scales_bf16"] = sum(o for o, _ in lin) * nl * 2
    b["hadamard_signs_bf16"] = sum(i for _, i in lin) * nl * 2  # could be bit-packed; tiny either way
    b["bias_qkv_norms_bf16"] = (H + 2 * KV + 2 * H) * nl * 2 + H * 2
    b["embed_bf16"] = V * H * 2
    b["lm_head_bf16"] = V * H * 2
    kv_per_tok = 2 * KV * nl * 2  # K and V, bf16, all layers
    # prefill transient at the largest chunk we would run (one 4k chunk): hidden + MLP gate/up + attn scores via SDPA
    rows = []
    base_res = b["codes_k"] + b["codebooks_k_bf16"] + b["scales_bf16"] + b["hadamard_signs_bf16"] + b["bias_qkv_norms_bf16"]
    variants = {
        "all_on_gpu": base_res + b["embed_bf16"] + b["lm_head_bf16"],
        "embed_on_cpu": base_res + b["lm_head_bf16"],
        "embed_on_cpu+lm_head_int8": base_res + b["lm_head_bf16"] / 2,
        "embed_on_cpu+lm_head_int4": base_res + b["lm_head_bf16"] / 4,
    }
    for vname, w in variants.items():
        for c in ctx:
            prefill_chunk = min(c, 4096)
            act = prefill_chunk * (H * 2 * 3 + 2 * I * 2)  # residual/normed/attn-out + gate,up (bf16)
            tot = w + kv_per_tok * c + act + runtime_gib * GiB
            rows.append({"variant": vname, "k": k, "ctx": c, "weights_gib": w / GiB, "kv_bf16_gib": kv_per_tok * c / GiB,
                         "prefill_act_gib": act / GiB, "runtime_gib": runtime_gib, "total_gib": tot / GiB,
                         "fits_4090": tot / GiB <= GPU_GIB})
    cpu_back = n_lin * (L - k) * nbits / d / 8
    return {"bytes": {kk: v for kk, v in b.items()}, "gib": {kk: v / GiB for kk, v in b.items()},
            "kv_bytes_per_token": kv_per_tok, "quantized_linear_params": n_lin, "rows": rows,
            "cpu_back_stages_codes_gib": cpu_back / GiB, "gpu_total_gib": GPU_GIB}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_72b_instruct")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    out = {f"k{k}": footprint(cfg, k) for k in (1, 2, 4)}
    print(f"quantized linear params: {out['k2']['quantized_linear_params'] / 1e9:.2f} G; 4090 total {GPU_GIB:.2f} GiB")
    for k in ("k2", "k4"):
        print(f"{k}: " + ", ".join(f"{n}={v:.3f}" for n, v in out[k]["gib"].items()) + " GiB")
    print(f"{'variant':28s} {'ctx':>6s} {'weights':>8s} {'kv':>6s} {'act':>5s} {'total':>7s} fits")
    for k in ("k2",):
        for r in out[k]["rows"]:
            print(f"{r['variant']:28s} {r['ctx']:6d} {r['weights_gib']:8.2f} {r['kv_bf16_gib']:6.2f} "
                  f"{r['prefill_act_gib']:5.2f} {r['total_gib']:7.2f} {r['fits_4090']}")
    print(f"k2 back stages (3-4) codes on CPU: {out['k2']['cpu_back_stages_codes_gib']:.2f} GiB; "
          f"k4 all-GPU weights {out['k4']['rows'][0]['weights_gib']:.2f} GiB")
    if a.save:
        from tools.results import save_result
        save_result("E5", "qwen72b_mem_fit", config={
            "what": "GPU memory footprint of nested d8 K256 Qwen2.5-72B, k-stage prefix resident (computed from config "
                    "shapes, not measured)", "assumptions": {
                "codes": "packed 8-bit code per 8 weights per stage (2 bit/weight for k=2)",
                "kv": "bf16 K,V for all 80 layers (8 KV heads x 128)",
                "prefill_act": "one prefill chunk of min(ctx,4096) tokens: 3 hidden bf16 buffers + gate/up bf16",
                "runtime_gib": "1.0 GiB for CUDA context + allocator slack + kernel workspace (assumed)"},
            "script": "experiments/22_scale72/mem72.py"}, metrics=out, model_dir=a.model)


if __name__ == "__main__":
    main()
