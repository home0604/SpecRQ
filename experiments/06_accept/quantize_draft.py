"""[specrq] Phase 5D(i): rounding-only draft codebooks at precision q (shared codes, draft = first k_d codebooks).

For every QuantizedLinear the first k_d target codebooks C[j] (K x g values, stored bf16 in the DbD ckpt) are copied
and rounded to the format below with a scale chosen by an MSE grid search; draft per-row scales stay fp16.
Formats: bf16 (q=16, exact copy; the DbD ckpt stores codebooks in bf16, not fp16 as the paper's size model says), int8 / int4 (symmetric, levels -2^(q-1)+1 .. 2^(q-1)-1), fp8_e4m3 (torch float8_e4m3fn,
max 448, subnormals included),
16-level LUT formats (value = per-group fp16 scale x level; experiments/11_qcmp/run_fmt4.sh): nf4 / nf3 (QLoRA NormalFloat),
fp4_e2m1 (+-{0,.5,1,1.5,2,3,4,6}), posit4_es0 (+-{0,1/4,1/2,3/4,1,3/2,2,4}), posit4_es1 (+-{0,1/16,1/4,1/2,1,2,4,16}),
int3 (+-3). LUT scale grid: max|v| -> 2^-6 .. 2^0.5 x largest level (spans the formats' dynamic range).
Scale granularity: "codebook" (one fp16 scale per codebook) or "dim" (one per codebook and vector coordinate, g scales).
Scale search (CLAUDE.md gotcha): INT grids search the clip ratio in [0.25, 1] x max|v| / qmax. FP8 has a wide dynamic
range whose dense region sits far below its max, so its scale grid spans 2^-12 .. 2^4 x (max|v| / 448), i.e. it may
map max|v| anywhere from far below 448 to beyond it (clipped), not only max|v| -> 448.
Output: draft_codebooks.pt (src/draft.py format, values already dequantized to float) + meta json with bit accounting.

Usage: python quantize_draft.py --model DIR --ckpt DIR --fmt int8 --gran codebook --out FILE.pt [--kd 3]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch
from src.aq import QuantizedLinear
from src.draft import attach_draft_codebooks, draft_state_dict
from src.modelutils import get_model

from src.draft import QBITS, best_scale, round_fmt  # shared with STE fake-quant in training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fmt", required=True, choices=list(QBITS))
    ap.add_argument("--gran", default="codebook", choices=["codebook", "dim"])
    ap.add_argument("--kd", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_draft", default=None, help="round an existing draft copy (e.g. TV-trained draft_codebooks.pt) "
                    "instead of the target codebooks; its draft scales are kept as stored")
    a = ap.parse_args()
    model = get_model(a.model, a.ckpt, "auto")
    attach_draft_codebooks(model, a.kd)
    if a.init_draft:
        from src.draft import load_draft_state_dict
        n = load_draft_state_dict(model, torch.load(a.init_draft))
        print("rounding draft copies from", a.init_draft, n, "linears")
    stats = {"rel_mse": [], "kurt": [], "bits_codebook": 0, "bits_scales": 0, "n_linears": 0, "n_params_target": 0}
    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, QuantizedLinear):
                continue
            cb = m.draft_codebooks.data.float()  # [k, K, og, ig]
            ref = cb.clone()
            z = cb.flatten(1) - cb.flatten(1).mean(1, keepdim=True)  # per draft plane value distribution
            stats["kurt"] += ((z ** 4).mean(1) / (z ** 2).mean(1) ** 2).tolist()  # Pearson kurtosis (normal = 3)
            if a.fmt != "bf16":
                dims = (1, 2) if a.gran == "dim" else (1, 2, 3)  # dim: per (codebook, coordinate); codebook: per codebook
                s = best_scale(cb, a.fmt, dims)
                cb = round_fmt(cb, s, a.fmt)
                stats["bits_scales"] += s.numel() * 16
            # bf16: exact copy of the stored target codebooks (DbD ckpt stores codebooks in the model dtype, bf16)
            m.draft_codebooks.data.copy_(cb)
            stats["rel_mse"].append(((cb - ref) ** 2).sum().item() / (ref ** 2).sum().item())
            stats["bits_codebook"] += cb.numel() * QBITS[a.fmt]
            stats["n_linears"] += 1
            stats["n_params_target"] += m.out_features * m.in_features
    sd = draft_state_dict(model)
    torch.save(sd, a.out)
    rm = stats.pop("rel_mse")
    ku = torch.tensor(stats.pop("kurt"))
    stats["plane_kurtosis"] = {"mean": ku.mean().item(), "median": ku.median().item(), "p90": ku.quantile(0.9).item(),
                               "max": ku.max().item(), "n_planes": ku.numel(),
                               "median_by_codebook": ku.view(-1, a.kd).median(0).values.tolist()}
    meta = {"args": vars(a), "q": QBITS[a.fmt], **stats,
            "codebook_rel_mse_mean": sum(rm) / len(rm), "codebook_rel_mse_max": max(rm),
            "draft_codebook_overhead_bits_per_weight": (stats["bits_codebook"] + stats["bits_scales"]) / stats["n_params_target"]}
    json.dump(meta, open(a.out.replace(".pt", ".meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
