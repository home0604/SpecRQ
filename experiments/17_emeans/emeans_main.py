"""specrq/dbd/main.py with e-means codebook initialization (no edits to specrq/dbd).

Replaces src.aq.fit_kmeans (the per-stage residual clustering inside init_aq_kmeans) by
faiss.contrib.e_means.QuantizedEmeans(d, K, quantizer=None, nepochs=$SPECRQ_EMEANS_EPOCHS (default 5), seed 0,
assignment_backend="native") on the GPU in fp32; CUDA-graph replay is disabled (torch 2.4 lacks torch.cuda.MemPool,
the eager path runs the same update rule). Codes are then the exact nearest centroid (find_nearest_cluster), as in
the default path. Everything else (AQ Adam + beam search, eval) is unchanged.
Needs faiss >= 1.16.0 on PYTHONPATH (isolated: /data/lsh/temp/pylib_faiss116).
Usage: cd specrq/dbd && PYTHONPATH=/data/lsh/temp/pylib_faiss116 python ../../experiments/17_emeans/emeans_main.py <main.py args>
"""
import os
import runpy
import sys

DBD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.aq as saq  # noqa: E402
from init_compare import make_emeans_fit  # noqa: E402

NEPOCHS = float(os.environ.get("SPECRQ_EMEANS_EPOCHS", "5"))
SEED = int(os.environ.get("SPECRQ_EMEANS_SEED", "0"))

if __name__ == "__main__":
    import faiss
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    saq.fit_kmeans = make_emeans_fit(NEPOCHS, SEED)
    print(f"[specrq-emeans] faiss {faiss.__version__} ({faiss.__file__}); e-means init nepochs={NEPOCHS} seed={SEED}")
    sys.argv = [os.path.join(DBD, "main.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(DBD, "main.py"), run_name="__main__")
