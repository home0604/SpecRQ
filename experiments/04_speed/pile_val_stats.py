"""Token count of the local pile-val file and overlap of the specrq pile_val loader for 1024 x 4096, plus a loader smoke.
Usage: python pile_val_stats.py <model_dir> [<model_dir> ...]"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from transformers import AutoTokenizer
from src.datautils import PILE_VAL_JSONL, get_loaders
texts = [json.loads(l)["text"] for l in open(PILE_VAL_JSONL)]
res = {"file": PILE_VAL_JSONL, "n_docs": len(texts), "n_chars": sum(map(len, texts))}
for p in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(p, use_fast=True)
    t = time.time(); n = 0; lens = []
    for i in range(0, len(texts), 2000):
        enc = tok(texts[i:i + 2000], add_special_tokens=False).input_ids
        lens += [len(e) for e in enc]
    n = sum(lens)
    need = 1024 * 4096
    res[p] = {"tokens": n, "requested_1024x4096": need, "ratio_requested_over_available": need / n,
              "docs_ge_4096_tokens": sum(l >= 4096 for l in lens), "median_doc_tokens": sorted(lens)[len(lens) // 2],
              "tokenize_s": time.time() - t}
    print(p, json.dumps(res[p]), flush=True)
    t = time.time()
    data = get_loaders("pile_val", nsamples=1024, seed=0, seqlen=4096, model_path=p, use_fast_tokenizer=True)
    flat = [tuple(x[0, :16].tolist()) for x in data]
    res[p]["loader"] = {"n": len(data), "shape": list(data[0].shape), "distinct_window_prefixes": len(set(flat)),
                        "first_tokens": data[0][0, :8].tolist(), "loader_s": time.time() - t}
    print(p, json.dumps(res[p]["loader"]), flush=True)
json.dump(res, open("/data/lsh/logs/pile_val_stats.json", "w"), indent=1)
