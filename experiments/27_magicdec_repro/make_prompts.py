"""E7 MagicDec repro: dump the exact PG-19 prompt rows that the official MagicDec benchmarks feed (DataLoader order,
shuffle=False) so our e2e_real.py can run on the identical prompts.

Uses the official Data/data_converter.convert_pg19_dataset (first 50 docs of Data/pg19/*.json in os.listdir order,
drop first 8000 tokens, split into seq_len chunks, BOS at position 0, dataset repeated 20x).
Output: /data/lsh/results_accept/magicvq/e2e_real/prompts_pg19md_ctx<ctx>.json  (rows [0, n))
Run in env /data/lsh/envs/magicdec (transformers 4.36.2 tokenizer = what the official run uses), CPU only:
  cd /data/lsh/src/MagicDec && python /hai/home/lsh/SpecRQ/experiments/27_magicdec_repro/make_prompts.py --prefix_len 32672
"""
import argparse
import json
import os
import sys

MD = "/data/lsh/src/MagicDec"
sys.path.insert(0, os.path.dirname(MD))
os.chdir(MD)
from MagicDec.Data.data_converter import convert_pg19_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prefix_len", type=int, default=32672)
ap.add_argument("--ctx_label", type=int, default=32768)
ap.add_argument("--n", type=int, default=160)
ap.add_argument("--model_name", default="/data/models/llama3.1_8b_instruct")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_name)
ds = convert_pg19_dataset(tokenizer=tok, seq_len=a.prefix_len)
data = ds.tensors[0]
nuniq = data.shape[0] // 20
print("rows", data.shape, "unique chunks", nuniq, "listdir", os.listdir("Data/pg19/"))
recs = [{"id": f"pg19md_{i}", "P": a.prefix_len, "ids": data[i].tolist(), "tau_rec": {}, "uniq_row": i % nuniq}
        for i in range(min(a.n, data.shape[0]))]
out = f"/data/lsh/results_accept/magicvq/e2e_real/prompts_pg19md_ctx{a.ctx_label}.json"
json.dump(recs, open(out, "w"))
print("wrote", out, len(recs))
