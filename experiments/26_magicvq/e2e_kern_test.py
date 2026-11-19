"""Unit checks of e2e_kern.py (ragged bf16 / VQ partials + combine, KV append, RQ encode) against fp32 torch
references. Usage: CUDA_VISIBLE_DEVICES=<free gpu> python experiments/26_magicvq/e2e_kern_test.py"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_kern as EK  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
B, Hkv, G, D, T = 3, 2, 4, 128, 700
Hq = Hkv * G
inv = (1.0 / 500000.0 ** (torch.arange(0, D, 2, device=dev).float() / D))


def rope(x, pos):  # x [..., D] fp32, pos broadcast
    ang = pos[..., None].float() * inv.repeat(2)
    rot = torch.cat((-x[..., D // 2:], x[..., :D // 2]), -1)
    return x * ang.cos() + rot * ang.sin()


def ref_attn(q_rot, keys, vals):  # q [G, D] fp32, keys [n, D] -> [G, D]
    s = q_rot @ keys.T / math.sqrt(D)
    return torch.softmax(s, -1) @ vals


def check_bf(M, use_idx):
    kvc = torch.randn(2, B, Hkv, T, D, device=dev).to(torch.bfloat16)
    qkv = torch.randn(B * M, (Hq + 2 * Hkv) * D, device=dev).to(torch.bfloat16)
    qpos = torch.tensor([400, 520, 610], device=dev, dtype=torch.int32)[:B]
    lo = torch.tensor([300, 0, 590], device=dev, dtype=torch.int32)[:B]
    hi = qpos + M
    idx = nidx = None
    if use_idx:
        n = 40
        idx = torch.stack([torch.stack([torch.randperm(250, device=dev)[:n].sort().values for _ in range(Hkv)])
                           for _ in range(B)]).int()
        nidx = torch.tensor([40, 17, 0, 33, 25, 40], device=dev, dtype=torch.int32)[:B * Hkv]
    EK.append_kv(qkv, kvc, inv, qpos, M, Hq)
    spt = 7
    ws = EK.workspace(B, Hkv, G, M, spt)
    out = torch.empty(B * M, Hq * D, device=dev, dtype=torch.bfloat16)
    nv = (hi - lo).max().item() + (40 if use_idx else 0)
    sp, ch = 5, -(-nv // 5)
    ch = -(-ch // 64) * 64
    EK.bf_partial(qkv, kvc[0], kvc[1], inv, ws, spt, 2, sp, ch, lo, hi, qpos, M, G, idx, nidx)
    # fill the unused slots 0, 1 with -inf partials
    OP, MP, LP, BM = ws
    for bh in range(B * Hkv):
        for s_ in (0, 1):
            MP[(bh * spt + s_) * BM:(bh * spt + s_ + 1) * BM] = float("-inf")
            LP[(bh * spt + s_) * BM:(bh * spt + s_ + 1) * BM] = 0
    EK.combine(ws, spt, out, B, Hkv, G, M)
    err = 0
    q = qkv[:, :Hq * D].float().view(B, M, Hkv, G, D)
    kk, vv = kvc[0].float(), kvc[1].float()
    # check appended K (RoPE) and V
    for b in range(B):
        for m in range(M):
            p = qpos[b].item() + m
            kn = qkv[b * M + m, Hq * D:(Hq + Hkv) * D].float().view(Hkv, D)
            assert (rope(kn, torch.tensor(p, device=dev)) - kk[b, :, p]).abs().max() < 0.05
    for b in range(B):
        for h in range(Hkv):
            for m in range(M):
                p = qpos[b].item() + m
                pos = list(range(lo[b].item(), p + 1))
                if use_idx:
                    pos = idx[b, h, :nidx[b * Hkv + h]].tolist() + pos
                pos = torch.tensor(pos, device=dev)
                qr = rope(q[b, m, h], torch.tensor(p, device=dev)).bfloat16().float()
                r = ref_attn(qr, kk[b, h, pos], vv[b, h, pos])
                o = out.view(B, M, Hkv, G, D)[b, m, h].float()
                err = max(err, (o - r).abs().max().item())
    print(f"bf_partial M={M} idx={use_idx}: max abs err {err:.4f}")
    assert err < 0.03


def check_vq_and_encode(S=2):
    """S = 1: one-stage (1-bit) codes / LUT kernel template KST = 1 (v8)."""
    print(f"-- VQ stages S = {S}")
    kvc = torch.randn(2, B, Hkv, T, D, device=dev).to(torch.bfloat16)
    cb = torch.randn(2, Hkv, S, 256, 8, device=dev) * torch.tensor([1.0, 0.4], device=dev)[None, None, :S, None, None]
    mu = torch.randn(2, Hkv, D, device=dev) * 0.3
    sd = torch.rand(2, Hkv, D, device=dev) + 0.5
    codes = torch.zeros(2, S, B, Hkv, T, 16, dtype=torch.uint8, device=dev)
    # range-mode encode of positions [5, 5 + n_b)
    lo = torch.tensor([5, 5, 5], device=dev, dtype=torch.int32)[:B]
    cnt = torch.tensor([600, 600, 300, 300, 650, 650], device=dev, dtype=torch.int32)[:B * Hkv]
    EK.rq_encode(kvc, codes, cb, mu, sd, cnt, 650, lo=lo)
    # torch reference encode (greedy residual, fp32)
    nbad = 0
    tot = 0
    for kv in range(2):
        for b in range(B):
            for h in range(Hkv):
                n = cnt[b * Hkv + h].item()
                x = ((kvc[kv, b, h, 5:5 + n].float() - mu[kv, h]) / sd[kv, h]).view(n, 16, 8)
                for s in range(S):
                    c = cb[kv, h, s]
                    dist = (c * c).sum(-1)[None, None] - 2 * torch.einsum("nkd,cd->nkc", x, c)
                    code = dist.argmin(-1)
                    nbad += (code.to(torch.uint8) != codes[kv, s, b, h, 5:5 + n]).sum().item()
                    tot += code.numel()
                    x = x - c[code]
    print(f"rq_encode (range): code mismatches {nbad} / {tot}")
    assert nbad <= tot * 1e-4
    # compact-mode encode
    src = torch.stack([torch.stack([torch.randperm(T, device=dev)[:200].sort().values for _ in range(Hkv)])
                       for _ in range(B)]).int()
    cnt2 = torch.full((B * Hkv,), 200, device=dev, dtype=torch.int32)
    codes2 = torch.zeros(2, S, B, Hkv, 256, 16, dtype=torch.uint8, device=dev)
    EK.rq_encode(kvc, codes2, cb, mu, sd, cnt2, 200, src=src)
    ok = (codes2[:, :, :, :, :200] == codes[:, :, torch.arange(B)[:, None, None], torch.arange(Hkv)[None, :, None],
                                               src.long()].permute(0, 1, 2, 3, 4, 5)).float()
    # positions >= 5 and inside each range were encoded above; compare where src is in range
    inr = torch.zeros(B, Hkv, 200, dtype=torch.bool, device=dev)
    for b in range(B):
        for h in range(Hkv):
            n = cnt[b * Hkv + h].item()
            inr[b, h] = (src[b, h] >= 5) & (src[b, h] < 5 + n)
    frac = ok[:, :, inr].mean().item()
    print(f"rq_encode (compact) agreement with range encode: {frac:.5f}")
    assert frac > 0.9999
    # VQ partial (LUT) over [LO, HI) per (b, h) + bf16 sinks, vs dequantized reference
    M = 1
    qkv = torch.randn(B, (Hq + 2 * Hkv) * D, device=dev).to(torch.bfloat16)
    qpos = torch.tensor([660, 640, 690], device=dev, dtype=torch.int32)[:B]
    vlo = torch.tensor([16, 16, 20, 16, 16, 16], device=dev, dtype=torch.int32)[:B * Hkv]
    vhi = torch.tensor([500, 500, 300, 290, 640, 600], device=dev, dtype=torch.int32)[:B * Hkv]
    blo = torch.tensor([500, 300, 640], device=dev, dtype=torch.int32)[:B]  # bf16 range [blo, qpos]
    sink = torch.arange(16, device=dev, dtype=torch.int32)[None, None].expand(B, Hkv, 16).contiguous()
    nsink = torch.full((B * Hkv,), 16, device=dev, dtype=torch.int32)
    EK.append_kv(qkv, kvc, inv, qpos, 1, Hq)
    vq_sp, vq_ch = 3, 256
    bf_sp, bf_ch = 2, 256  # covers 16 sinks + the longest bf16 range (b=1: 341)
    spt = bf_sp + 16 * vq_sp
    ws = EK.workspace(B, Hkv, G, M, spt)
    out = torch.empty(B, Hq * D, device=dev, dtype=torch.bfloat16)
    cbb = cb.to(torch.bfloat16)
    EK.bf_partial(qkv, kvc[0], kvc[1], inv, ws, spt, 0, bf_sp, bf_ch, blo, qpos + 1, qpos, 1, G, sink, nsink)
    EK.lut_partial(qkv, [codes[0, s] for s in range(S)], [codes[1, s] for s in range(S)],
                   [cbb[0, :, s] for s in range(S)], [cbb[1, :, s] for s in range(S)], inv, ws, spt, bf_sp, vq_sp,
                   vq_ch, vlo, vhi, qpos, sd[0].contiguous(), mu[0].contiguous(), sd[1].contiguous(),
                   mu[1].contiguous(), B, Hkv, S)
    EK.combine(ws, spt, out, B, Hkv, G, 1)
    err = 0
    q = qkv[:, :Hq * D].float().view(B, Hkv, G, D)
    for b in range(B):
        for h in range(Hkv):
            a, z = vlo[b * Hkv + h].item(), vhi[b * Hkv + h].item()
            deq = []
            for kv in range(2):
                c = codes[kv, :, b, h, a:z].long()  # [S, n, 16]
                r = sum(cbb[kv, h, s].float()[c[s]] for s in range(S)).reshape(z - a, D)
                deq.append(r * sd[kv, h] + mu[kv, h])
            p = qpos[b].item()
            bpos = torch.cat([torch.arange(16, device=dev), torch.arange(blo[b].item(), p + 1, device=dev)])
            K = torch.cat([kvc[0, b, h, bpos].float(), deq[0]])
            V = torch.cat([kvc[1, b, h, bpos].float(), deq[1]])
            qr = rope(q[b, h], torch.tensor(p, device=dev))
            r = ref_attn(qr, K, V)
            o = out.view(B, Hkv, G, D)[b, h].float()
            err = max(err, (o - r).abs().max().item())
    print(f"lut_partial + bf16 sinks/window + combine: max abs err {err:.4f}")
    assert err < 0.05


if __name__ == "__main__":
    for M in (1, 4, 7):
        check_bf(M, False)
    check_bf(1, True)
    check_bf(3, True)
    check_vq_and_encode(2)
    check_vq_and_encode(1)
    print("ALL OK")
