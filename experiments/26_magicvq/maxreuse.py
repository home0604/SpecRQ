"""E7 phase-2 probe: can the draft's softmax max be reused as a fixed ("unified") max in the target verify attention?

Analysis only (no new kernel). Reuses the e2e_real.py model / drafts / partial kernels on TP=4.
  --mode stats : real prompts + prefill, eager spec loop (static, greedy) for `gen` tokens per row; per layer, every
                 round: m_target[b, qhead, row j] = max_k q_t.k/sqrt(d) over the exact bf16 cache (from the verify
                 kernel's per-split maxima), m_hat_qd = max of the DRAFT step j over its own cache with its own query
                 (rows j < gamma), m_hat_qt = max over the draft cache with the VERIFY query q_t (rows 0..gamma, incl.
                 the bonus row). Scores in natural units (q.k/sqrt(d)). Gaps gathered over ranks -> stats.
  --mode prof  : B x ctx caches (prompt lengths of the real prompts, cache contents random: kernel time does not
                 depend on values), one verify forward (gamma+1 rows) eager under torch.profiler (graphs off) ->
                 per-kernel CUDA time; plus the CUDA-graph verify step time; plus a microbench of the bf16 partial
                 kernel with the online rescale removed (fixed max, same memory traffic) = in-kernel rescale cost.
Usage: torchrun --nproc_per_node 4 experiments/26_magicvq/maxreuse.py --mode stats --task pg19 --ctx 32768 --B 4 \
         --gen 64 --configs vqpL2+r128:4 hyb512pL2:4 snapkv4096:4 --tag mr1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import e2e_real as ER  # noqa: E402
import e2e_kern as EK  # noqa: E402

LN2 = 0.6931471805599453
OUTD = f"{ER.OUTROOT}/maxreuse"


def row_max(attn, M):
    """[B, hq, M] max score (natural units) of the last Attn call, from its per-split partial maxima."""
    OP, MP, LP, BM = attn.ws
    B, hkv, G = attn.st.B, attn.m.hkv, attn.m.G
    mp = MP.view(B, hkv, attn.spt, BM).amax(2)[..., :G * M]  # rows r = m*G + g
    return (mp.view(B, hkv, M, G).permute(0, 1, 3, 2).reshape(B, hkv * G, M) * LN2).clone()


class Rec:
    def __init__(self, attn, M, sink, pre=None):
        self.a, self.M, self.sink, self.pre = attn, M, sink, pre

    def __call__(self, qkv, l):
        self.a.ws[1].fill_(float("-inf"))
        o = self.a(qkv, l)
        self.sink(l, row_max(self.a, self.M))
        if self.pre is not None:
            self.pre(qkv, l)
        return o


def stats_mode(a, m, st, prompts, qlast, cb):
    res = {}
    for c in a.configs:
        name, gamma = c.split(":")
        gamma = int(gamma)
        G1 = gamma + 1
        d = ER.Draft(name, m, st, cb)
        d.build(st.tc, qlast)
        dplan, _ = ER.tune_plan(m, st, 1, d)
        vplan, _ = ER.tune_plan(m, st, G1, None)
        dsteps = [ER.Attn(m, st, 1, j, d, dplan) for j in range(gamma)]
        dq = [ER.Attn(m, st, 1, j, d, dplan) for j in range(G1)]  # draft cache, verify query (rows 0..gamma)
        va = ER.Attn(m, st, G1, 0, None, vplan)
        B = st.B
        jj = torch.arange(G1, device="cuda")
        enc_cnt = torch.full((B * m.hkv,), G1, dtype=torch.int32, device="cuda")
        st.reset()
        gaps_qd, gaps_qt, mt_all, layers_qd, layers_qt, rowj_qd, rowj_qt = [], [], [], [], [], [], []
        mhat_qd = {}
        rounds = 0
        while True:
            act = (st.ngen < st.gen)
            dtok = torch.zeros(B, G1, dtype=torch.int64, device="cuda")
            dtok[:, 0] = st.cur
            for j in range(gamma):
                dsteps[j].prep(st.length)
                sink = (lambda j_: (lambda l, x: mhat_qd.__setitem__((l, j_), x[..., 0])))(j)
                dtok[:, j + 1] = ER.forward(m, dtok[:, j], Rec(dsteps[j], 1, sink))
            for x in dq:
                x.prep(st.length)
            va.prep(st.length)
            cur = {}

            def vsink(l, x):
                cur["mt"] = x  # [B, hq, G1]

            def after(qkv, l):
                mt = cur["mt"]
                q3 = qkv.view(B, G1, -1)
                mh = []
                for j in range(G1):
                    x = dq[j]
                    x.ws[1].fill_(float("-inf"))
                    x(q3[:, j].contiguous(), l)
                    mh.append(row_max(x, 1)[..., 0])
                mh = torch.stack(mh, -1)  # [B, hq, G1]
                sel = act[:, None, None].expand_as(mt)
                gaps_qt.append((mt - mh)[sel])
                layers_qt.append(torch.full_like(mt, l)[sel])
                rowj_qt.append(jj[None, None].expand_as(mt)[sel].float())
                mt_all.append(mt[sel])
                mq = torch.stack([mhat_qd[(l, j)] for j in range(gamma)], -1)
                s2 = act[:, None, None].expand_as(mq)
                gaps_qd.append((mt[..., :gamma] - mq)[s2])
                layers_qd.append(torch.full_like(mq, l)[s2])
                rowj_qd.append(jj[None, None, :gamma].expand_as(mq)[s2].float())
            targ = ER.forward(m, dtok.reshape(-1), Rec(va, G1, vsink, after)).view(B, G1)
            match = (dtok[:, 1:] == targ[:, :gamma]).to(torch.int64)
            acc = match.cumprod(1).sum(1)
            inc = torch.where(act, acc + 1, torch.zeros_like(acc))
            old = st.length.clone()
            st.length.add_(inc.to(torch.int32))
            st.ngen.add_(inc)
            st.cur.copy_(torch.where(act, targ.gather(1, acc[:, None])[:, 0], st.cur))
            if d.kind == "rq":
                for l in range(ER.NL):
                    EK.rq_encode(st.tc[l], d.codes[l], d.cb.f32[l], d.cb.mu[l], d.cb.sd[l], enc_cnt, G1, lo=old)
            rounds += 1
            if bool((st.ngen >= st.gen).all()) or rounds > st.gen:
                break
        cat = lambda xs: torch.cat(xs).float().cpu()  # noqa: E731
        loc = {k: cat(v) for k, v in dict(qd=gaps_qd, qt=gaps_qt, mt=mt_all, lqd=layers_qd, lqt=layers_qt,
                                           jqd=rowj_qd, jqt=rowj_qt).items()}
        allr = [None] * ER.WORLD
        dist.all_gather_object(allr, loc)
        if ER.RANK == 0:
            g = {k: torch.cat([x[k] for x in allr]) for k in loc}
            res[c] = summarize(g, gamma, rounds)
            ER.log(c, json.dumps({k: res[c][k] for k in ("m_hat_qd", "m_hat_qt")}))
        d.free()
        del d
        torch.cuda.empty_cache()
    return res


def dstat(x):
    if x.numel() == 0:
        return None
    q = torch.quantile(x[torch.randperm(x.numel())[:2_000_000]], torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
    return {"n": x.numel(), "mean": x.mean().item(), "std": x.std().item(), "min": x.min().item(),
            "p0.1": q[0].item(), "p1": q[1].item(), "p50": q[2].item(), "p99": q[3].item(), "p99.9": q[4].item(),
            "max": x.max().item(), "frac_gt20": (x > 20).float().mean().item(),
            "frac_lt_m20": (x < -20).float().mean().item(), "frac_abs_gt5": (x.abs() > 5).float().mean().item(),
            "frac_abs_gt1": (x.abs() > 1).float().mean().item()}


def summarize(g, gamma, rounds):
    out = {"rounds": rounds, "m_target": dstat(g["mt"]), "m_hat_qd": dstat(g["qd"]), "m_hat_qt": dstat(g["qt"])}
    out["m_hat_qt_bonus_row"] = dstat(g["qt"][g["jqt"] == gamma])
    out["m_hat_qt_rows_lt_gamma"] = dstat(g["qt"][g["jqt"] < gamma])
    out["per_layer"] = {}
    for l in range(ER.NL):
        a, b = g["qd"][g["lqd"] == l], g["qt"][g["lqt"] == l]
        out["per_layer"][l] = {k: {"mean": x.mean().item(), "p99": torch.quantile(x[:2_000_000], 0.99).item(),
                                   "max": x.max().item(), "min": x.min().item()} for k, x in (("qd", a), ("qt", b))}
    return out


# ----------------------------------------------------------------------------------------------- profile mode
@triton.jit
def _bf_partial_fixed(QKV, KC, VC, INVF, OP, MP, LP, LO, HI, QPOS, chunk, qk_scale, sq_r, s_b, s_h, s_t, SPT, OFF,
                      HKV: tl.constexpr, G: tl.constexpr, M: tl.constexpr, BM: tl.constexpr, D: tl.constexpr,
                      BN: tl.constexpr):
    """_bf_partial (no IDX) with a FIXED max m = 0 (no running max / alpha rescale): timing probe only."""
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    b = bh // HKV
    h = bh % HKV
    lo = tl.load(LO + b)
    hi = tl.load(HI + b)
    qpos = tl.load(QPOS + b)
    nv = hi - lo
    rows = tl.arange(0, BM)
    rm = rows // G
    rg = rows % G
    rvalid = rows < G * M
    dd = tl.arange(0, D)
    qrow = QKV + (b * M + rm)[:, None] * sq_r + (h * G + rg)[:, None] * D
    q = tl.load(qrow + dd[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    qp = tl.load(qrow + ((dd + D // 2) % D)[None, :], mask=rvalid[:, None], other=0.0).to(tl.float32)
    f = tl.load(INVF + (dd % (D // 2)))
    ang = (qpos + rm).to(tl.float32)[:, None] * f[None, :]
    sgn = tl.where(dd < D // 2, -1.0, 1.0)
    q = ((q * tl.cos(ang) + sgn[None, :] * qp * tl.sin(ang)) * qk_scale).to(tl.bfloat16)
    lim = qpos + rm
    start = sp * chunk
    end = tl.minimum(start + chunk, nv)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    kb = KC + b * s_b + h * s_h
    vb = VC + b * s_b + h * s_h
    for t0 in range(start, end, BN):
        v = t0 + tl.arange(0, BN)
        vm = v < end
        pos = lo + v
        k = tl.load(kb + pos[:, None] * s_t + dd[None, :], mask=vm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))
        s = tl.where((pos[None, :] <= lim[:, None]) & vm[None, :], s, float("-inf"))
        p = tl.exp2(s)
        l_i = l_i + tl.sum(p, 1)
        vv = tl.load(vb + pos[:, None] * s_t + dd[None, :], mask=vm[:, None], other=0.0)
        acc = acc + tl.dot(p.to(tl.bfloat16), vv)
    o = (bh * SPT + OFF + sp) * BM + rows
    tl.store(OP + o[:, None] * D + dd[None, :], acc)
    tl.store(MP + o, tl.zeros([BM], dtype=tl.float32))
    tl.store(LP + o, l_i)


def gtime(fn, reps=20):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    del g
    return statistics.median(ts), min(ts), max(ts)


def prof_mode(a, m, st):
    from torch.profiler import profile, ProfilerActivity
    res = {}
    B = st.B
    torch.manual_seed(0)
    for gamma in a.gammas:
        G1 = gamma + 1
        vplan, vtune = ER.tune_plan(m, st, G1, None)
        va = ER.Attn(m, st, G1, 0, None, vplan)
        st.reset()
        tok = torch.randint(0, 1000, (B * G1,), device="cuda")

        def vstep():
            va.prep(st.length)
            return ER.forward(m, tok, va)
        for _ in range(3):
            vstep()
        torch.cuda.synchronize()
        dist.barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(a.prof_iters):
                vstep()
            torch.cuda.synchronize()
        kt = {}
        for e in p.key_averages():
            t = e.self_device_time_total
            if t > 0:
                kt[e.key] = {"us_per_step": t / a.prof_iters, "calls_per_step": e.count / a.prof_iters}
        tot = sum(v["us_per_step"] for v in kt.values())
        cls = {"bf_partial": 0.0, "combine": 0.0, "append": 0.0, "nccl": 0.0, "gemm": 0.0, "other": 0.0}
        for k, v in kt.items():
            kl = k.lower()
            c = ("bf_partial" if "_bf_partial" in k else "combine" if "_combine" in k else "append" if "_append" in k
                 else "nccl" if "nccl" in kl else "gemm" if any(s in kl for s in ("gemm", "cutlass", "sm80_xmma", "sm90",
                                                                                     "ampere", "cublas", "splitk"))
                 else "other")
            cls[c] += v["us_per_step"]
        # graph time of the verify step (as in e2e_real) and of its attention only
        ver_ms = gtime(vstep)
        qkv = torch.randn(B * G1, (m.hq + 2 * m.hkv) * ER.D, device="cuda").to(torch.bfloat16) * 0.1
        attn_ms = gtime(lambda: [va(qkv, l) for l in range(ER.NL)])
        (bs, bch), _ = vplan
        OP, MP, LP, BM = va.ws
        tcl = st.tc
        part_ms = gtime(lambda: [EK.bf_partial(qkv, tcl[l][0], tcl[l][1], m.inv, va.ws, va.spt, 0, bs, bch, va.blo,
                                               va.bhi, va.qpos, G1, m.G) for l in range(ER.NL)])
        comb_ms = gtime(lambda: [EK.combine(va.ws, va.spt, va.out, B, m.hkv, m.G, G1) for _ in range(ER.NL)])

        def fixed():
            for l in range(ER.NL):
                kc, vc = tcl[l][0], tcl[l][1]
                _bf_partial_fixed[(B * m.hkv, bs)](qkv, kc, vc, m.inv, OP, MP, LP, va.blo, va.bhi, va.qpos, bch,
                                                   ER.D ** -0.5 * EK.LOG2E, qkv.stride(0), kc.stride(0), kc.stride(1),
                                                   kc.stride(2), va.spt, 0, HKV=m.hkv, G=m.G, M=G1, BM=BM, D=ER.D,
                                                   BN=64, num_warps=4, num_stages=1)
        fixed_ms = gtime(fixed)
        r = {"gamma": gamma, "M": G1, "verify_plan": vplan, "profile_us_per_step": kt, "profile_total_us": tot,
             "profile_class_us": cls, "profile_class_share": {k: v / tot for k, v in cls.items()},
             "graph_verify_ms": ver_ms, "graph_attn_all_layers_ms": attn_ms, "graph_bf_partial_ms": part_ms,
             "graph_combine_ms": comb_ms, "graph_bf_partial_fixed_max_ms": fixed_ms,
             "upper_bound_saving_ms": comb_ms[0] + max(0.0, part_ms[0] - fixed_ms[0]),
             "upper_bound_saving_share_of_verify": (comb_ms[0] + max(0.0, part_ms[0] - fixed_ms[0])) / ver_ms[0]}
        res[f"g{gamma}"] = r
        ER.log(f"g{gamma}: verify {ver_ms[0]:.2f} ms, attn {attn_ms[0]:.2f}, partial {part_ms[0]:.2f}, "
               f"fixed-max partial {fixed_ms[0]:.2f}, combine {comb_ms[0]:.3f}; profile share {r['profile_class_share']}")
        del va
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stats", "prof"], required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--gammas", nargs="*", type=int, default=[3, 4])
    ap.add_argument("--prof-iters", type=int, default=5)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    ER.RANK, ER.WORLD = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(ER.RANK)
    lst = [None] * ER.WORLD
    dist.all_gather_object(lst, os.getpid())
    own = set(lst)
    rec = {"args": vars(a), "co_start": ER.cocheck(own)}
    prompts = json.load(open(f"{ER.OUT}/prompts_{a.task}_ctx{a.ctx}.json"))[a.skip:a.skip + a.B]
    m = ER.Shard()
    gmax = max([int(c.split(":")[1]) for c in a.configs] + a.gammas + [1])
    P_list = [p["P"] for p in prompts]
    T = -(-(max(P_list) + a.gen + gmax + 2) // 64) * 64
    st = ER.State(m, a.B, T, a.gen, gmax)
    g0 = torch.zeros(a.B, dtype=torch.int64, device="cuda")
    if a.mode == "stats":
        qlast = [[None] * a.B for _ in range(ER.NL)]
        for b, p in enumerate(prompts):
            g0[b] = ER.prefill(m, st.tc, b, torch.tensor(p["ids"], device="cuda"), qlast)
            torch.cuda.empty_cache()
        st.set_prompts(P_list, g0)
        rec["results"] = stats_mode(a, m, st, prompts, qlast, ER.CB(m))
    else:
        st.tc.normal_(0, 1)  # random cache contents (timing only)
        st.set_prompts(P_list, g0)
        rec["results"] = prof_mode(a, m, st)
    rec["co_end"] = ER.cocheck(own)
    rec["cotenant_any"] = rec["co_start"]["cotenant"] or rec["co_end"]["cotenant"]
    if ER.RANK == 0:
        os.makedirs(OUTD, exist_ok=True)
        json.dump(rec, open(f"{OUTD}/{a.mode}_{a.tag}_{a.task}_ctx{a.ctx}_B{a.B}.json", "w"), indent=1)
    ER.log("done; co-tenant", rec["cotenant_any"])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
