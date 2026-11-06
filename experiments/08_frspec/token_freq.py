"""FR-Spec token frequency table from pile-val (calibration corpus, disjoint from the wikitext2 / c4 eval sets).

Counts unigram token frequencies of every document in the local pile-val-backup file (same file as
specrq/dbd/src/datautils.py::get_pile_val; all 214,670 docs = calibration front + heldout back of the shuffled order).
Each document is encoded on its own with add_special_tokens=False, so BOS/EOS/chat tokens are NOT counted from text;
the generation stop tokens (generation_config eos_token_id list) are recorded as `forced` and always put into the
draft vocab by specrq/infer/frspec.py::load_subset.

Differences from FR-Spec fr/fr.py (github thunlp/FR-Spec): there the corpus is the first 1,000,001 streamed
SlimPajama-627B train docs, tokenizer.encode(text) with default special tokens (BOS counted once per doc), and the eos
fix-up appends tokenizer.encode(eos_token) (for Llama-3 = [BOS, EOS]) when EOS is not in the top r.

Outputs (out_dir): counts.pt (int64 [V_model]), order.pt (int64 [V_model], ids by count desc, ties by id asc),
meta.json (docs, tokens, forced ids, coverage of top-V_d, overlap check vs eval sets).
Usage: python experiments/08_frspec/token_freq.py --model DIR --out DIR [--workers 24] [--overlap-check 100]
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import torch

PILE = os.environ.get("SPECRQ_PILE_VAL_JSONL", "/data/lsh/datasets/pile_val/val.jsonl")
# cached arrow of the c4 eval loader (allenai/c4 en/c4-validation.00000-of-00008 @ 607bd4c8); offline hash lookup fails
C4_ARROW = ("/data/lsh/hf_cache/datasets/allenai___c4/default-4cb1202b07c0cd0b/0.0.0/"
            "607bd4c8450a42878aa9ddc051a65a055450ef87/c4-validation.arrow")
_TOK = None
_V = None


def _init(model, V):
    global _TOK, _V
    from transformers import AutoTokenizer
    _TOK, _V = AutoTokenizer.from_pretrained(model), V


def _count(texts):
    ids = _TOK(texts, add_special_tokens=False).input_ids
    flat = np.fromiter((t for x in ids for t in x), dtype=np.int64)
    return np.bincount(flat, minlength=_V), int(flat.size)


def _detok_wikitext(s):
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "),
                 (" : ", ": "), (" 's", "'s"), ("( ", "("), (" )", ")")):
        s = s.replace(a, b)
    return s


def overlap_check(texts, n, seed=0):
    """Exact-substring probe: n 64-char snippets from wikitext2 test paragraphs (de-tokenized) and n from the c4
    validation docs that the eval loader uses (first 1100 of shard 0), searched in the joined pile-val text."""
    from datasets import load_dataset
    big = "\n".join(texts)
    rng = random.Random(seed)
    wt = [p.strip() for p in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(p.strip()) > 300]
    from datasets import Dataset
    c4 = [d["text"] for d in Dataset.from_file(C4_ARROW).select(range(1100)) if len(d["text"]) > 300]
    out = {}
    for name, docs, f in (("wikitext2_test", wt, _detok_wikitext), ("c4_val_eval", c4, lambda s: s)):
        hits, probes = [], rng.sample(docs, min(n, len(docs)))
        for d in probes:
            d = f(d)
            i = len(d) // 2
            snip = d[i:i + 64]
            if big.find(snip) >= 0:
                hits.append(snip)
        out[name] = {"probes": len(probes), "hits": len(hits), "hit_examples": hits[:5]}
        print("[overlap]", name, out[name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--overlap-check", type=int, default=0, help="snippets per eval set (0 = skip)")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    cfg = AutoConfig.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    V = cfg.vocab_size
    try:
        eos = GenerationConfig.from_pretrained(a.model).eos_token_id
    except Exception:
        eos = cfg.eos_token_id
    eos = sorted(set(eos if isinstance(eos, list) else [eos]))
    t0 = time.time()
    with open(PILE) as f:
        texts = [json.loads(line)["text"] for line in f]
    if a.max_docs:
        texts = texts[:a.max_docs]
    chunks = [texts[i:i + a.chunk] for i in range(0, len(texts), a.chunk)]
    counts, n_tok = np.zeros(V, dtype=np.int64), 0
    with Pool(a.workers, initializer=_init, initargs=(a.model, V)) as pool:
        for j, (c, n) in enumerate(pool.imap(_count, chunks)):
            counts += c[:V] if c.size >= V else np.pad(c, (0, V - c.size))
            n_tok += n
            if j % 100 == 0:
                print(f"[freq] {j}/{len(chunks)} chunks, {n_tok} tokens, {time.time() - t0:.0f}s", flush=True)
    counts_t = torch.from_numpy(counts)
    # stable order: count desc, id asc
    order = torch.from_numpy(np.lexsort((np.arange(V), -counts))).long()
    os.makedirs(a.out, exist_ok=True)
    torch.save(counts_t, os.path.join(a.out, "counts.pt"))
    torch.save(order, os.path.join(a.out, "order.pt"))
    tot = counts.sum()
    cs = np.cumsum(counts[order.numpy()])
    cov = {str(r): float(cs[r - 1] / tot) for r in (4096, 8192, 16384, 32768, 65536) if r <= V}
    meta = {"model": a.model, "tokenizer": type(tok).__name__, "len_tokenizer": len(tok), "vocab_size_model": V,
            "corpus": PILE, "corpus_source": "HF mit-han-lab/pile-val-backup @ 2f5e46ae, val.jsonl.zst decompressed",
            "n_docs": len(texts), "n_tokens": int(n_tok), "add_special_tokens": False,
            "doc_join": "each document encoded separately (no join string)",
            "forced": eos, "forced_tokens": [tok.convert_ids_to_tokens(i) for i in eos],
            "forced_counts": [int(counts[i]) for i in eos],
            "n_nonzero_ids": int((counts > 0).sum()), "coverage_topVd": cov,
            "top20": [[int(i), tok.convert_ids_to_tokens(int(i)), int(counts[i])] for i in order[:20]],
            "elapsed_s": time.time() - t0}
    print("[meta]", json.dumps({k: meta[k] for k in ("n_docs", "n_tokens", "forced", "coverage_topVd")}), flush=True)
    if a.overlap_check:
        meta["overlap_check"] = overlap_check(texts, a.overlap_check)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
