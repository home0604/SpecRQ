"""Download Phase-1 models (safetensors + configs + tokenizer) to /data/models. HF_HOME must point to /data/lsh/hf_cache."""
import os, sys
from huggingface_hub import snapshot_download
assert os.environ.get("HF_HOME") == "/data/lsh/hf_cache", os.environ.get("HF_HOME")
JOBS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "/data/models/qwen2.5_0.5b_instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("meta-llama/Llama-3.1-8B-Instruct", "/data/models/llama3.1_8b_instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
]
for repo, out, rev in JOBS:
    try:
        p = snapshot_download(repo, revision=rev, local_dir=out,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "*.md", "tokenizer*"],
                              ignore_patterns=["original/*"])
        with open(os.path.join(out, "REVISION.txt"), "w") as f:
            f.write(f"repo={repo}\nrevision={rev}\n")
        print("OK", repo, p, flush=True)
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:500], flush=True)
