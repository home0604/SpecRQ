"""E4 scale72: experiments/18_offload micro-benchmarks re-targeted to Qwen2.5-72B shapes (no change to 18_offload).
  cpu  : bench_cpu.py sweep over 80 layers x 7 linears x 2 residual planes (stages 3-4) of 72B, M = 1..5 (+8, 16)
  pcie : bench_gpu.py per-layer split-stage round trips with 72B stage sizes (qkv 8192->10240, o 8192->8192,
         gate+up 8192->59136, down 29568->8192)
Usage: python experiments/22_scale72/bench_offload72.py --target cpu|pcie [bench args...] --out X.json
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "18_offload"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

Q72 = {"q": (8192, 8192), "k": (8192, 1024), "v": (8192, 1024), "o": (8192, 8192),
       "gate": (8192, 29568), "up": (8192, 29568), "down": (29568, 8192)}
STAGES72 = {"qkv": (8192, 10240), "o": (8192, 8192), "gateup": (8192, 59136), "down": (29568, 8192)}

if __name__ == "__main__":
    i = sys.argv.index("--target")
    target = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    if target == "cpu":
        import bench_cpu as B
        B.QWEN32B = Q72  # (in, out) per linear; the sweep reads this module global
        B.N_LAYERS = 80
        B.main()
    else:
        import bench_gpu as G
        G.STAGES = STAGES72
        G.main()
