"""E6 kvtau: table + result records from /data/lsh/results_accept/kvtau/<wmode>/{ppl,tau}_*.json.
Usage: python experiments/25_kvtau/summarize.py [--save]   (--save writes results/E6/llama8b_kvtau_<wmode>.json)"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = "/data/lsh/results_accept/kvtau"
KVBITS = {"a": {1: 1, 2: 2}, "b": {1: 0.5, 2: 1, 3: 2}, "c": {1: 0.5, 2: 1, 3: 1.5, 4: 2}}
WBITS = {"k2": 2, "k1B": "1 (draft-B cb)", "bf16": 16, "same_W": None}


def pooled(res, dname, sets=("pg19", "wt2", "chat")):
    out = {}
    for g in (1, 2, 4):
        T = it = 0
        for s in sets:
            if s not in res:
                continue
            for r in res[s]["per_prompt"]:
                t, i, _ = r["drafts"][dname][f"g{g}"]
                T += t; it += i
        out[f"tau_g{g}"] = T / it if it else None
    for k in ("tv_all", "top1_all", "tv_path", "top1_path"):
        xs = [r["drafts"][dname][k] for s in sets if s in res for r in res[s]["per_prompt"]]
        out[k] = sum(xs) / len(xs)
    out["per_set_tau_g1_g2_g4"] = {s: [res[s]["agg"][dname][f"taug{g}"] for g in (1, 2, 4)] for s in sets if s in res}
    out["target_tf_vs_gen"] = {s: res[s]["target_tf_vs_gen"] for s in sets if s in res}
    out["n_prompts"] = sum(len(res[s]["per_prompt"]) for s in sets if s in res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    for w in ("bf16", "w2"):
        d = os.path.join(ROOT, w)
        tgt_w = "k2" if w == "w2" else "bf16"
        ppl = {}
        for fn in sorted(glob.glob(f"{d}/ppl_{a.tag}*.json")):
            r = json.load(open(fn))
            for corpus in ("wikitext2", "c4_new"):
                for k, v in r[corpus].items():
                    if isinstance(v, dict):
                        ppl.setdefault(k, {}).setdefault(corpus, v)  # first file wins (ppl_v1_cn_keep1: 40 windows)
        rows = []
        print(f"\n### weights {w}  (PPL @4096 wt2 / c4_new, KL(bf16KV||cfg), top1 vs bf16KV)")
        for k, v in ppl.items():
            print(f"  {k:45s} " + "  ".join(f"{c}: {v[c]['ppl']:.2f} KL {v[c]['kl_bf16kv_to_cfg']:.3f} top1 {v[c]['top1_vs_bf16kv']:.3f}"
                                             for c in ("wikitext2", "c4_new") if c in v))
        taus = {}
        for fn in sorted(glob.glob(f"{d}/tau_{a.tag}_ctx*_*.json")):
            m = re.match(rf".*/tau_{a.tag}_ctx(\d+)_(.*)\.json", fn)
            ctx, kv = int(m.group(1)), m.group(2)
            r = json.load(open(fn))
            for dname, _, lv in r["config"]["drafts"]:
                taus.setdefault((kv, dname), {})[ctx] = pooled(r, dname)
        print(f"  {'KV format (target)':32s} {'draft':24s} {'KVb':>4s} {'Wb':>5s} {'PPLt wt2/c4':>14s} "
              f"{'TV/top1 long':>13s} {'tau long g1/g2/g4':>20s} {'tau 512 g1/g2/g4':>20s}")
        for (kv, dname), byctx in taus.items():
            dw, dl = dname.split("|")
            f = kv[0] if kv != "bf16kv" else None
            lv = re.match(r"kv(\d+)", dl)
            kvb = 16 if kv == "bf16kv" else KVBITS[f][int(lv.group(1))]
            wb = WBITS[dw] if dw != "same_W" else WBITS[tgt_w]
            pk = f"{kv}|L{max(KVBITS[f])}" if f else "bf16kv"
            pt = ppl.get(pk, {})
            pstr = "/".join(f"{pt[c]['ppl']:.2f}" for c in ("wikitext2", "c4_new") if c in pt)
            L, S = byctx.get(6144), byctx.get(512)
            tvs = "%.3f/%.3f" % (L["tv_all"], L["top1_all"]) if L else "-"
            fmt = lambda x: "/".join(f"{x[f'tau_g{g}']:.2f}" for g in (1, 2, 4)) if x else "-"
            print(f"  {kv:32s} {dname:24s} {kvb!s:>4s} {wb!s:>5s} {pstr:>14s} "
                  f"{tvs:>13s} {fmt(L):>20s} {fmt(S):>20s}")
            rows.append({"weights": w, "kv_target": kv, "draft": dname, "draft_kv_bits": kvb, "draft_w_bits": wb,
                         "ppl_target": pt, "long_ctx6144": L, "short_ctx512": S})
        if a.save and rows:
            from tools.results import save_result
            p = save_result("E6", f"llama8b_kvtau_{w}", config={
                "model": "Llama-3.1-8B-Instruct", "weights": w,
                "target_weights": "nested 2-bit d8K256L2_N_noft_n192 k=2" if w == "w2" else "bf16",
                "kv_formats": {"a": "d8 K256x2 (1+1 bit), draft L1 = 1 bit", "b": "d8 K16,K16,K256 (0.5+0.5+1), drafts L1 0.5 b / L2 1 b",
                               "c": "d8 K16x4 (0.5 each), drafts L1 0.5 b / L2 1 b"},
                "kv_quant": "pre-RoPE K and V, per (layer, K/V, kv-head, stage) codebook shared over the 16 sub-vectors, "
                            "greedy residual encoding (exact fp32), RoPE after dequant; meta (RoPE-pair) layout; "
                            "codebooks: k-means per stage on residuals (greedy RQ) on pile_val 16x4096 KV activations of the "
                            "same weights (24576 train tokens); 'refine' = 8 rounds prefix-weighted codebook update; "
                            "'_cn' = static per-channel standardization (calib mean/std, foldable) ; keep1 = token 0 (sink) bf16",
                "tau": "greedy, teacher-forced on the target's own greedy path (incremental decode with quantized cache, "
                       "128 tokens, stop at EOS); acceptance vs the target's teacher-forced argmax (fp32-attention pass); "
                       "drafts with exact shared-KV round semantics (step m reads target KV prefix before the round and its "
                       "own prefix-encoded KV inside the round); sets pg19 / wt2 / chat x 8 prompts at ctx 6144 and 512",
                "ppl": "teacher-forced @4096, 40 windows wikitext2 test + 40 windows c4_new (AQLM c4 validation concat)",
                "quality_only": True},
                metrics={"rows": rows, "ppl": ppl}, model_dir="/data/models/llama3.1_8b_instruct",
                notes="quality only (no timing). raw: " + d)
            print("saved", p)


if __name__ == "__main__":
    main()
