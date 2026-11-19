"""E7 magicvq v10: run lb_tau.py / tau_eval.py with extra 'p'-format codebook tags registered at runtime (no edit of
mv_common.py). Tags (env V10_TAGS, space separated, e.g. "e2ekl2") map to
/data/lsh/results_accept/magicvq/cb/cb_d8K256x2_post_cn_<tag>.pt, draft names vqp<tag>L<k>+r128 / hyb512p<tag>L<k>.
Usage: V10_TAGS="e2ekl2" python experiments/26_magicvq/v10_tau.py lb_tau.py --ctx 32768 ... (args as the script)
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mv_common  # noqa: E402

for t in os.environ.get("V10_TAGS", "").split():
    p = os.path.join(mv_common.OUTROOT, "cb", f"cb_d8K256x2_post_cn_{t}.pt")
    assert os.path.exists(p), p
    mv_common.CB["p" + t] = p
    mv_common.POST.add("p" + t)
script = os.path.join(HERE, sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
