"""E5 tree-drafting offline data: per-position, per-depth rank of the target greedy token in the draft logits on the REAL
Qwen2.5-32B 2-bit nested checkpoint (d8 K256 L2 noFT), draft = prefix k1 (A = target codebook plane 1, B = TV-trained
draft codebooks), shared-KV condition as in SpecDecoder(draft_kv="shared"):
  committed context [0, n) = TARGET (k2) KV, the in-round tokens = the draft's own k1 KV.
For a round starting at committed position n (cur = token at n), the draft runs ONE forward M=4 over the target greedy
continuation [cur, g1, g2, g3] at positions n..n+3 (each layer writes its KV before attending, causal mask), i.e. the
draft path is conditioned on the correct tokens of the earlier depths ("conditional on the correct path"). Row j gives
the depth-(j+1) draft logits; we store rank = #(draft logits > logit of g_{j+1}) over the full vocab and over the FR-Spec
V_d=32768 subset (huge if g_{j+1} is not in the subset). The work cache is restored from the target cache after every
position, so every position sees exactly target KV for the context.
[I] numerics: one M=4 forward instead of 4 sequential M=1 steps (same math, bf16 split differences); the target KV of
the context comes from a teacher-forced prefill instead of the verify forwards of a real decode.

Sources:
  paths  : 8 chat + 8 wikitext2 64-token prompts of skip_tau.py, target AR k2 greedy path (EOS stop, <=128 tokens);
           continuation = the AR path itself. Every round start c=1..N-1 is a sample (depth j valid if c-1+j <= N-1).
  corpus : n_windows x 512-token windows of wikitext2 test / c4 val; starts t = 64, 72, ..., <= 504; continuation
           g1 = teacher-forced target argmax at t, g2..g4 = target k2 M=1 greedy steps (work cache, restored).
Output: torch .pt with ranks per (draft, source) + meta.
Usage: python experiments/14_q32b_upper/tree_ranks.py --out /data/lsh/results_accept/.../tree/ranks.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "14_q32b_upper"))
sys.path.insert(0, str(REPO / "specrq" / "dbd"))

import torch  # noqa: E402

from specrq.infer.decode import ARDecoder  # noqa: E402
from specrq.infer.model import NestModel, load_draft_file  # noqa: E402
from skip_tau import INSTRUCT_PROMPTS  # noqa: E402

BIG = 1 << 30
DEPTH = 4


def ranks_of(logits, tgt, sub, in_sub):
    """logits [M, V] (bf16), tgt [M] long -> (rank_full [M], rank_vd [M]) int64."""
    lf = logits.float()
    tv = lf.gather(1, tgt[:, None])
    rf = (lf > tv).sum(1)
    rv = (lf[:, sub] > tv).sum(1)
    rv = torch.where(in_sub[tgt], rv, torch.full_like(rv, BIG))
    return rf, rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_32b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/scale/qwen32b_d8K256L2_noft")
    ap.add_argument("--draft_b", default="/data/lsh/results_accept/scale/qwen32b_d8K256L2_noft/kd1/train_B_bf16/"
                                         "draft_codebooks.pt")
    ap.add_argument("--vd", type=int, default=32768)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--n_windows", type=int, default=8)
    ap.add_argument("--win_stride", type=int, default=8)
    ap.add_argument("--max_prompts", type=int, default=16, help="smoke: fewer prompts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    from transformers import AutoTokenizer, GenerationConfig
    from src.datautils import get_loaders
    from specrq.infer.frspec import coverage, default_freq_dir, load_subset

    m = NestModel(a.model, a.ckpt)
    print("[tree_ranks] draft B linears", m.load_draft(load_draft_file(a.draft_b)), flush=True)
    V = m.lm_head.shape[0]
    fd = default_freq_dir(m.cfg)
    sub = load_subset(fd, a.vd, None).to(m.device)
    in_sub = torch.zeros(V, dtype=torch.bool, device=m.device)
    in_sub[sub] = True
    tok = AutoTokenizer.from_pretrained(a.model)
    e = GenerationConfig.from_pretrained(a.model).eos_token_id
    eos = sorted(set(e if isinstance(e, list) else [e]))
    prompts = []
    for p in INSTRUCT_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        prompts.append(("instruct", tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]))
    wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=512, eval_mode=True)
    for i in range(8):
        prompts.append(("wikitext2", wt[0, (100 + i) * 512:(100 + i) * 512 + 64]))
    if a.max_prompts < 16:
        h = a.max_prompts // 2
        prompts = prompts[:h] + prompts[8:8 + a.max_prompts - h]
    prompts = [(k, p.to(m.device)) for k, p in prompts]
    T = (max(p.numel() for _, p in prompts) + a.gen_tokens + DEPTH + 32 + 15) // 16 * 16
    drafts = {"A": False, "B": True}

    ar = ARDecoder(m, T, k=m.L, eos=eos)
    ar.generate(prompts[0][1], 8)  # warm-up
    paths = [ar.generate(p, a.gen_tokens)[0] for _, p in prompts]
    del ar
    torch.cuda.empty_cache()
    print("[tree_ranks] path lens", [len(x) for x in paths], flush=True)

    out = {"meta": {"args": vars(a), "eos": eos, "V": V, "vd_coverage_pileval": coverage(fd, sub.cpu()),
                    "kinds": [k for k, _ in prompts], "path_lens": [len(x) for x in paths], "depth": DEPTH,
                    "BIG": BIG}, "paths": {}, "corpus": {}}
    tc0, wc = m.new_cache(T), m.new_cache(T)
    with torch.no_grad():
        # ---------------- generated paths
        for dn in drafts:
            out["paths"][dn] = []
        for pi, ((kind, p), x) in enumerate(zip(prompts, paths)):
            P, N = p.numel(), len(x)
            seq = torch.cat([p, torch.tensor(x, device=m.device)])
            L0 = seq.numel()
            m.prefill(seq, tc0, m.L)  # target KV of prompt + path at [0, L0)
            wc.k.copy_(tc0.k); wc.v.copy_(tc0.v)
            recs = {dn: {"rank_full": torch.full((N, DEPTH), BIG, dtype=torch.long),
                         "rank_vd": torch.full((N, DEPTH), BIG, dtype=torch.long)} for dn in drafts}
            for c in range(1, N):  # round start: committed x0..x_{c-1}, cur = x_{c-1} at position n
                n = P + c - 1
                nd = min(DEPTH, N - c)  # valid depths
                inp = seq[n:n + DEPTH] if n + DEPTH <= L0 else torch.cat(
                    [seq[n:], seq[-1:].repeat(n + DEPTH - L0)])
                tgt = seq[n + 1:n + 1 + nd]
                pos = torch.tensor(n, device=m.device)
                for dn, ud in drafts.items():
                    lg = m.forward(inp, pos, wc, 1, ud)[:nd]
                    rf, rv = ranks_of(lg, tgt, sub, in_sub)
                    recs[dn]["rank_full"][c, :nd] = rf.cpu()
                    recs[dn]["rank_vd"][c, :nd] = rv.cpu()
                    hi = min(n + DEPTH, wc.k.shape[2])
                    wc.k[:, :, n:hi].copy_(tc0.k[:, :, n:hi]); wc.v[:, :, n:hi].copy_(tc0.v[:, :, n:hi])
            for dn in drafts:
                out["paths"][dn].append({"kind": kind, "N": N, **recs[dn]})
            r0 = {dn: (recs[dn]["rank_full"][1:, 0] == 0).float().mean().item() for dn in drafts}
            print(f"[paths {pi} {kind} N={N}] depth1 top1 {r0} ({time.time() - t_start:.0f}s)", flush=True)
        torch.save(out, a.out)
        del tc0, wc
        torch.cuda.empty_cache()

        # ---------------- corpus windows
        Tw = 512 + 16
        tc0, wc = m.new_cache(Tw), m.new_cache(Tw)
        for name in ("wikitext2", "c4"):
            if not a.n_windows:
                break
            enc = get_loaders(name, seed=0, model_path=a.model, seqlen=512, eval_mode=True)
            out["corpus"][name] = {dn: {"rank_full": [], "rank_vd": []} for dn in drafts}
            for wi in range(min(a.n_windows, enc.numel() // 512)):
                ids = enc[0, wi * 512:(wi + 1) * 512].to(m.device)
                pos0 = torch.zeros((), dtype=torch.long, device=m.device)
                lt = m.forward(ids, pos0, tc0, m.L)  # teacher-forced target logits, target KV [0, 512)
                g1all = lt.argmax(-1)
                del lt
                wc.k.copy_(tc0.k); wc.v.copy_(tc0.v)
                for t in range(64, 512 - DEPTH + 1, a.win_stride):
                    g = [g1all[t:t + 1]]
                    for j in range(1, DEPTH):  # target greedy continuation on the work cache
                        lg = m.forward(g[-1], torch.tensor(t + j, device=m.device), wc, m.L, last_only=True)
                        g.append(lg.argmax(-1))
                    g = torch.cat(g)
                    inp = torch.cat([ids[t:t + 1], g[:-1]])
                    for dn, ud in drafts.items():
                        lg = m.forward(inp, torch.tensor(t, device=m.device), wc, 1, ud)
                        rf, rv = ranks_of(lg, g, sub, in_sub)
                        out["corpus"][name][dn]["rank_full"].append(rf.cpu())
                        out["corpus"][name][dn]["rank_vd"].append(rv.cpu())
                    wc.k[:, :, t:t + DEPTH + 1].copy_(tc0.k[:, :, t:t + DEPTH + 1])
                    wc.v[:, :, t:t + DEPTH + 1].copy_(tc0.v[:, :, t:t + DEPTH + 1])
                print(f"[corpus {name} {wi}] ({time.time() - t_start:.0f}s)", flush=True)
            for dn in drafts:
                for key in ("rank_full", "rank_vd"):
                    out["corpus"][name][dn][key] = torch.stack(out["corpus"][name][dn][key])
            torch.save(out, a.out)
    out["meta"]["wall_s"] = time.time() - t_start
    out["meta"]["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.save(out, a.out)
    print("[tree_ranks] done", out["meta"]["wall_s"], flush=True)


if __name__ == "__main__":
    main()
