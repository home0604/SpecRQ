"""E7 magicvq v6 step 1: collect bf16 Llama-3.1-8B-Instruct activations for draft-specific RQ KV codebook training.

Per training sequence (<= 16384 + gen tokens) and layer: pre-RoPE K, V of every position (bf16) and pre-RoPE Q of
`nq` sampled query positions (bf16) + the attention output of those queries (exact, fp32 attention) is NOT stored
(recomputed in training at the chosen RoPE offset). Data (disjoint from every v1-v5 evaluation sample):
  pg19  : PG-19 test books with 16.4k <= tokens(text[:600k]) < 40k. tau_eval/e2e (load_books: >= 41984 tokens) and the
          synthetic haystacks (lb_tasks.load_haystack_books: >= 40000 tokens) only ever select books with >= 40000
          tokens, so these books are never used by any evaluation. BOS + first T-1 tokens (raw continuation).
          Queries: uniform over [T/4, T).
  pile  : pile-val windows of T tokens read from the heldout end of the shuffled document order
          (datautils.get_pile_val(heldout=True)); the 'p' codebooks were calibrated on the other end (seed 0).
          Queries: uniform over [T/4, T).
  wiki2 : LongBench 2wikimqa (never used in v4/v4b/v5 eval), hotpotqa_q-style instruction (answer + verbatim
          supporting quotes), Llama-3.1 chat template, LongBench middle truncation to T - gen; the bf16 target greedily
          generates `gen` tokens (stopped at EOS). Queries: all generated positions + uniform prompt positions [T/4, P).
Output: /data/lsh/ckpts/magicvq_v6/acts/{s<idx>_L<l>.pt, meta.json, rope.pt}
Usage: python experiments/26_magicvq/rqd_collect.py --n_pg19 11 --n_pile 4 --n_wiki 4
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tau_eval as TE  # noqa: E402
from mv_common import MODEL, PG19  # noqa: E402

OUT = "/data/lsh/ckpts/magicvq_v6/acts"
WIKI_T = ("Answer the question based on the given passages. First state the answer in one sentence. Then quote, word "
          "for word, every sentence from the passages that supports the answer, one quote per line, each prefixed with "
          "its passage number.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the "
          "given passages. First state the answer in one sentence, then give the verbatim supporting quotes with "
          "their passage numbers.\n\nQuestion: {input}\nAnswer:")


def pg19_train_books(tok, n, T):
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19}/*.parquet")))
    out = []
    for i, text in enumerate(df["text"]):
        ids = tok(text[:600000], add_special_tokens=False).input_ids
        n4 = len(tok(text[:400000], add_special_tokens=False).input_ids)
        if T - 1 <= len(ids) < 40000 and n4 < 40000:
            out.append((f"pg19_row{i}", [tok.bos_token_id] + ids[: T - 1]))
        if len(out) >= n:
            break
    return out


def pile_windows(tok, n, T):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "specrq", "dbd"))
    from src.datautils import get_pile_val
    ws = get_pile_val(n, T, tok, heldout=True)
    return [(f"pile_heldout{i}", w[0].tolist()) for i, w in enumerate(ws)]


def wiki_prompts(tok, n, T, gen):
    from lb_tasks import _finish, _lb_rows
    rows = []
    for f in ("2wikimqa", "2wikimqa_e"):
        rows += [json.loads(l) for l in open(f"/data/lsh/datasets/longbench/data/{f}.jsonl")]
    rng = random.Random(6)
    rng.shuffle(rows)
    out = []
    for r in rows:
        L = len(tok(r["context"], add_special_tokens=False).input_ids)
        if L < 0.5 * (T - gen):
            continue
        TEMPL_KEY = "hotpotqa_q"  # _finish only uses the task name for RAW detection (chat template here)
        ids = _finish(tok, TEMPL_KEY, WIKI_T.format(context=r["context"], input=r["input"]), T - gen)
        out.append((f"2wikimqa_{r['_id']}", ids))
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def forward_store(f, ids, qpos, cs, sidx):
    """layer-wise exact forward (SDPA, bf16) of ids [T]; saves K, V [T, Hkv, D] and Q[qpos] [nq, H, D] pre-RoPE."""
    T = ids.shape[0]
    cos, sin = cs
    h = f.mm.embed_tokens(ids)
    for l, layer in enumerate(f.mm.layers):
        q, k, v = f.qkv(layer, h)
        torch.save({"K": k.cpu(), "V": v.cpu(), "Q": q[qpos].cpu(), "qpos": qpos.cpu()}, f"{OUT}/s{sidx}_L{l}.pt")
        qr = TE.rp(q.float(), cos[0, :T], sin[0, :T]).to(torch.bfloat16)
        kr = TE.rp(k.float(), cos[0, :T], sin[0, :T]).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(qr.transpose(0, 1)[None], kr.repeat_interleave(f.G, 1).transpose(0, 1)[None],
                                           v.repeat_interleave(f.G, 1).transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
        h = f.post(layer, h, o.reshape(T, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--nq", type=int, default=1024)
    ap.add_argument("--n_pg19", type=int, default=11)
    ap.add_argument("--n_pile", type=int, default=4)
    ap.add_argument("--n_wiki", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    seqs = [("pg19",) + x for x in pg19_train_books(tok, a.n_pg19, a.T)]
    seqs += [("pile",) + x for x in pile_windows(tok, a.n_pile, a.T)]
    seqs += [("wiki2",) + x for x in wiki_prompts(tok, a.n_wiki, a.T, a.gen)]
    print("sequences", [(s[0], s[1], len(s[2])) for s in seqs], f"{time.time() - t0:.0f}s", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    f = TE.M(model)
    maxT = a.T + a.gen + 8
    cs = f.cos_sin(torch.arange(40000, device="cuda"))
    torch.save({"cos": cs[0][0].cpu(), "sin": cs[1][0].cpu()}, f"{OUT}/rope.pt")
    Kst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    Vst = [torch.empty(maxT, f.Hkv, f.D, device="cuda", dtype=torch.bfloat16) for _ in range(f.nl)]
    g = torch.Generator().manual_seed(0)
    meta = []
    for si, (src, name, ids) in enumerate(seqs):
        ids = torch.tensor(ids, device="cuda")
        P = ids.shape[0]
        text = None
        if src == "wiki2":
            path = TE.prefill_and_generate(f, ids, a.gen, Kst, Vst, cs)
            text = tok.decode(path, skip_special_tokens=True)
            ids = torch.cat([ids, torch.tensor(path, device="cuda")])
            T = ids.shape[0]
            gq = torch.arange(P, T)
            nr = max(0, a.nq - gq.numel())
            rq = torch.randint(T // 4, P, (nr,), generator=g)
            qpos = torch.cat([gq, rq]).sort().values
        else:
            T = P
            qpos = torch.randint(T // 4, T, (a.nq,), generator=g).unique()
        forward_store(f, ids, qpos.cuda(), cs, si)
        meta.append({"idx": si, "src": src, "name": name, "T": T, "P": P, "nq": int(qpos.numel()),
                     "gen_text": text[:300] if text else None})
        print(si, src, name, T, P, int(qpos.numel()), f"{time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    json.dump({"args": vars(a), "seqs": meta, "wall_s": time.time() - t0, "model": MODEL}, open(f"{OUT}/meta.json", "w"),
              indent=1)
    print("done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
