"""E7 v14 checks for e2e_w4 (TP=4, torchrun --nproc_per_node 4):
 1. shard exactness: Marlin per-rank shards (packed-tensor slicing) dequantized with the kernel (identity rows, as
    experiments/13_int4_marlin/eval_quality.dequant_marlin) == the same slice of the dense dequant checkpoint
    (w_fp16.safetensors, = v12 '@w4' source) bit for bit, for --check_layers;
 2. draft logits: teacher-forced forward over the first --T tokens of a few prompts, three weight sets with identical
    bf16 attention / norms / embedding / lm_head: target bf16, 'dense' = dense INT4 dequant cast to bf16 (v12 tau
    semantics), 'marlin' = Marlin INT4 (e2e_w4.MLin, rows in chunks of 16 = the decode M<=16 kernel path). Reports
    max / mean |logit diff| and top-1 agreement (marlin vs dense, marlin vs target, dense vs target);
 3. per-step linear / comm micro-timing at M = B rows (CUDA graph, median of reps): all 32 layers' 4 linears in bf16
    (target kernels) vs Marlin, and the 66 all-reduces of a decode step ([B, 4096] bf16).
Writes JSON to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_real as E  # noqa: E402
import e2e_w4  # noqa: E402

D = E.D


def dense_shards(path, r, world):
    from safetensors import safe_open
    f = safe_open(path, "pt", device="cpu")
    hq, hkv, inter = E.HQA // world, E.HKVA // world, E.INTER // world
    sl = lambda n: f.get_slice(n)  # noqa: E731
    out = []
    for l in range(E.NL):
        p = f"model.layers.{l}."
        q = sl(p + "self_attn.q_proj")[r * hq * D:(r + 1) * hq * D]
        k = sl(p + "self_attn.k_proj")[r * hkv * D:(r + 1) * hkv * D]
        v = sl(p + "self_attn.v_proj")[r * hkv * D:(r + 1) * hkv * D]
        g = sl(p + "mlp.gate_proj")[r * inter:(r + 1) * inter]
        u = sl(p + "mlp.up_proj")[r * inter:(r + 1) * inter]
        out.append({"qkv": torch.cat([q, k, v]).cuda(), "o": sl(p + "self_attn.o_proj")[:, r * hq * D:(r + 1) * hq * D].cuda(),
                    "gu": torch.cat([g, u]).cuda(), "down": sl(p + "mlp.down_proj")[:, r * inter:(r + 1) * inter].cuda()})
    return out  # fp16 [N, K]


def dequant(lin):
    """kernel dequant of one MLin shard -> fp16 [N, K] (identity rows, M = 16 per call)."""
    W = torch.empty(lin.K, lin.N, dtype=torch.float16, device="cuda")
    eye = torch.eye(16, dtype=torch.float16, device="cuda")
    x = torch.zeros(16, lin.K, dtype=torch.float16, device="cuda")
    mar = sys.modules["marlin"]
    for i in range(0, lin.K, 16):
        x.zero_()
        x[:, i:i + 16] = eye
        mar.mul(x, lin.B, W[i:i + 16], lin.s, lin.ws)
    return W.t().contiguous()


def full_forward(m, ids, lin):
    """teacher-forced forward of one sequence; lin(l, name, x) -> bf16; returns full-vocab logits [T, V] (fp32)."""
    T = ids.shape[0]
    h = E.embed(m, ids)
    ang = torch.arange(T, device="cuda").float()[:, None] * m.inv[None]
    cos, sin = torch.cat([ang.cos()] * 2, -1)[:, None], torch.cat([ang.sin()] * 2, -1)[:, None]
    rope = lambda x: x * cos + torch.cat([-x[..., D // 2:], x[..., :D // 2]], -1) * sin  # noqa: E731
    hq, hkv = m.hq, m.hkv
    for l, L in enumerate(m.layers):
        qkv = lin(l, "qkv", E.rms(h, L["n1"]))
        q = rope(qkv[:, :hq * D].view(T, hq, D).float()).to(torch.bfloat16)
        k = rope(qkv[:, hq * D:(hq + hkv) * D].view(T, hkv, D).float()).to(torch.bfloat16)
        v = qkv[:, (hq + hkv) * D:].view(T, hkv, D)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1).repeat_interleave(m.G, 0)[None],
                                           v.transpose(0, 1).repeat_interleave(m.G, 0)[None], is_causal=True)
        o = o[0].transpose(0, 1).reshape(T, hq * D)
        h = h + E.allreduce(lin(l, "o", o))
        gu = lin(l, "gu", E.rms(h, L["n2"]))
        h = h + E.allreduce(lin(l, "down", F.silu(gu[:, :m.inter]) * gu[:, m.inter:]))
    lg = (E.rms(h, m.norm) @ m.lm_head.t()).float()
    parts = [torch.empty_like(lg) for _ in range(E.WORLD)]
    dist.all_gather(parts, lg)
    return torch.cat(parts, -1)


def graph_time(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        dist.barrier()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    del g
    t = torch.tensor([statistics.median(ts), min(ts), max(ts)], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return {"median_ms": t[0].item(), "min_ms": t[1].item(), "max_ms": t[2].item(), "reps": reps}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w4", default="/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048")
    ap.add_argument("--dense", default="/data/lsh/ckpts/weightvq/llama31_8b/int4_marlin_gptq_g128_pile128x2048_dense/w_fp16.safetensors")
    ap.add_argument("--check_layers", type=int, nargs="*", default=[0, 1, 15, 30, 31])
    ap.add_argument("--task", default="syn_quote")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--nprompts", type=int, default=3)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dist.init_process_group("nccl")
    E.RANK, E.WORLD = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(E.RANK)
    rec = {"args": vars(a), "world": E.WORLD, "nvsmi_start": E.nvsmi_apps()}
    m = E.Shard()
    w4 = e2e_w4.W4(a.w4, E.RANK, E.WORLD, E.NL, D, E.HQA, E.HKVA, E.INTER)
    dn = dense_shards(a.dense, E.RANK, E.WORLD)
    rec["w4"] = w4.info()
    # ---- 1. shard exactness
    ex = {}
    for l in a.check_layers:
        for n in ("qkv", "o", "gu", "down"):
            W = dequant(w4.layers[l][n])
            ok = torch.tensor([int(torch.equal(W, dn[l][n])), int((W != dn[l][n]).sum())], device="cuda")
            dist.all_reduce(ok, op=dist.ReduceOp.SUM)
            ex[f"{l}.{n}"] = {"ranks_equal": int(ok[0]), "n_diff_total": int(ok[1])}
    rec["shard_exact"] = ex
    rec["shard_exact_all"] = all(v["ranks_equal"] == E.WORLD for v in ex.values())
    E.log("shard exact:", rec["shard_exact_all"], ex)
    # ---- 2. logits
    dnb = [{k: v.to(torch.bfloat16) for k, v in L.items()} for L in dn]

    def lin_t(l, n, x):
        return x @ m.layers[l][n].t()

    def lin_d(l, n, x):
        return x @ dnb[l][n].t()

    def lin_m(l, n, x):
        return torch.cat([w4.layers[l][n](x[i:i + 16]) for i in range(0, x.shape[0], 16)])
    prompts = json.load(open(f"{E.OUT}/prompts_{a.task}_ctx{a.ctx}.json"))[:a.nprompts]
    lg_rec = []
    for pi, p in enumerate(prompts):
        ids = torch.tensor(p["ids"][:a.T], device="cuda")
        Lt, Ld, Lm = (full_forward(m, ids, f) for f in (lin_t, lin_d, lin_m))
        r = {"prompt": pi, "T": a.T}
        for nm, x, y in (("marlin_vs_dense", Lm, Ld), ("marlin_vs_target", Lm, Lt), ("dense_vs_target", Ld, Lt)):
            d = (x - y).abs()
            r[nm] = {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
                     "max_abs_last8": d[-8:].max().item(), "mean_abs_last8": d[-8:].mean().item(),
                     "top1_agree": (x.argmax(-1) == y.argmax(-1)).float().mean().item()}
        r["logit_absmean_target"] = Lt.abs().mean().item()
        lg_rec.append(r)
        E.log("logits", json.dumps(r))
        del Lt, Ld, Lm
    rec["logits"] = lg_rec
    del dnb, dn
    torch.cuda.empty_cache()
    # ---- 3. micro-timing (one decode step at M = B rows)
    B = a.B
    hid = torch.randn(B, E.HID, device="cuda").to(torch.bfloat16) * 0.5
    xo = torch.randn(B, m.hq * D, device="cuda").to(torch.bfloat16) * 0.5
    xd = torch.randn(B, m.inter, device="cuda").to(torch.bfloat16) * 0.5

    def lins(f):
        def run():
            for l in range(E.NL):
                f(l, "qkv", hid)
                f(l, "o", xo)
                f(l, "gu", hid)
                f(l, "down", xd)
        return run

    def lin_m1(l, n, x):
        return w4.layers[l][n](x)
    ar_buf = torch.randn(B, E.HID, device="cuda").to(torch.bfloat16)

    def comm():
        for _ in range(2 * E.NL + 2):
            dist.all_reduce(ar_buf)
    tm = {"linears_bf16": graph_time(lins(lin_t)), "linears_marlin": graph_time(lins(lin_m1)),
          "allreduce_66": graph_time(comm)}
    per = {}
    for n, x in (("qkv", hid), ("o", xo), ("gu", hid), ("down", xd)):
        per[n] = {"bf16": graph_time(lambda: [lin_t(l, n, x) for l in range(E.NL)]),
                  "marlin": graph_time(lambda: [lin_m1(l, n, x) for l in range(E.NL)])}
    tm["per_linear_32layers"] = per
    rec["timing"] = tm
    rec["nvsmi_end"] = E.nvsmi_apps()
    E.log("timing", json.dumps(tm))
    if E.RANK == 0:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(rec, open(a.out, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
