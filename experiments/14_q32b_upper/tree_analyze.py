"""E5 offline tree-drafting analysis for the Qwen2.5-32B 2-bit nested checkpoint, draft k1 (A / B), shared KV.
Inputs: tree_ranks.py output (per round start, per depth 1..4: rank of the target greedy token in the draft logits, draft
fed the correct path) and tree_timing.py output (synthetic-model forward M-curves, T_AR).

Tree model (greedy verification of a token tree): a node = rank tuple (r1..rj) = "child r_j (0-based draft rank) of
the node (r1..r_{j-1})". For a round start s, the accepted length = the deepest tree node whose tuple equals the
target-token ranks (R_s1..R_sj); confirmed tokens = accepted + 1 (verify bonus). This uses the EMPIRICAL joint rank
vector of every position, so no independence assumption (Sequoia's DP assumes per-depth independence; here the
tau-optimal tree of n nodes and depth <= d is exactly the top-n tuples by empirical match frequency, which is prefix-
closed because a child never matches more often than its parent).
  tau_pos : 1 + mean accepted over all round starts with all depths valid (position-uniform estimate)
  tau_sim : offline decode simulation along each target greedy path (start c=1, c += a+1, EOS/length end handled by the
            valid-depth mask) -> confirmed decode tokens / rounds, summed per kind (= skip_tau decode tau definition)
  CV      : tree built on half the prompts (even / odd index, both kinds), tau_sim on the other half, averaged;
            "corpus-built": tree built on the wt2+c4 window samples, evaluated on the paths.
Cost per round [I, modeled from measured forward times]: draft = sum over levels j=1..d of T_k1(M_j), M_j = number of
depth-(j-1) nodes with children (root: 1); verify = T_k2(n+1); overhead = measured chain-loop overhead (E4 record,
~0.045 ms) [I: tree bookkeeping on the host is not measured]. Tree attention mask cost is not modeled [I].
expected speedup = tau_sim * T_AR / round_ms. Not a measured end-to-end speed-up.
Usage: python experiments/14_q32b_upper/tree_analyze.py --ranks R.pt --timing T.json [--save RUN]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
E4 = REPO / "results/E4/q32b_L2_synth_prefix_layerskip.json"
RMAX = 16
NMAX = 31  # verify M = n + 1 <= 32 (measured range)


def interp(curve, M):
    """curve: dict M -> ms (measured). Linear interpolation; linear extrapolation from the last two points beyond."""
    xs = sorted(curve)
    if M in curve:
        return curve[M], False
    if M > xs[-1]:
        x0, x1 = xs[-2], xs[-1]
        return curve[x1] + (curve[x1] - curve[x0]) / (x1 - x0) * (M - x1), True
    return float(np.interp(M, xs, [curve[x] for x in xs])), False


def node_counts(R, dmax=4):
    """R [S, 4] ranks -> Counter of rank tuples (all ranks < RMAX) with depth <= dmax."""
    c = Counter()
    for row in R.tolist():
        for j in range(dmax):
            if row[j] >= RMAX:
                break
            c[tuple(row[:j + 1])] += 1
    return c


def best_tree(counts, n, dmax):
    cand = sorted((k for k in counts if len(k) <= dmax), key=lambda k: (-counts[k], len(k), k))
    tree = set(cand[:n])
    for k in tree:  # prefix closure (holds by construction; checked)
        assert len(k) == 1 or k[:-1] in tree, k
    return tree


def full_tree(w, d):
    out, frontier = set(), [()]
    for _ in range(d):
        frontier = [p + (r,) for p in frontier for r in range(w)]
        out.update(frontier)
    return out


def accepted(row, tree):
    a = 0
    for j in range(len(row)):
        if tuple(row[:j + 1]) in tree:
            a = j + 1
        else:
            break
    return a


def tau_pos(R, tree):
    full = R[(R < (1 << 30)).all(1)] if len(R) else R
    return 1 + float(np.mean([accepted(r, tree) for r in full.tolist()])) if len(full) else float("nan")


def tau_sim(paths, tree, kinds=None):
    """paths: list of dict(kind, N, R [N, 4]) -> {kind: tau, 'all': tau}."""
    agg = {}
    for p in paths:
        if kinds is not None and p["kind"] not in kinds:
            continue
        N, R = p["N"], p["R"]
        c, rounds = 1, 0
        while c < N:
            a = accepted(R[c].tolist(), tree)
            c += a + 1
            rounds += 1
        tok = N - 1
        for key in (p["kind"], "all"):
            t = agg.setdefault(key, [0, 0])
            t[0] += tok
            t[1] += rounds
    return {k: v[0] / v[1] for k, v in agg.items() if v[1]}


def level_ms(tree):
    """Number of nodes per draft forward level: M_j = #depth-(j-1) nodes with children."""
    d = max((len(k) for k in tree), default=0)
    Ms = []
    for j in range(1, d + 1):
        parents = {k[:-1] for k in tree if len(k) == j}
        Ms.append(len(parents))
    return Ms


def cost(tree, Td, Tv, ovh):
    Ms = level_ms(tree)
    dr = [interp(Td, M) for M in Ms]
    v = interp(Tv, len(tree) + 1)
    return {"draft_ms": sum(x for x, _ in dr), "verify_ms": v[0], "overhead_ms": ovh,
            "round_ms": sum(x for x, _ in dr) + v[0] + ovh, "level_M": Ms, "n": len(tree), "depth": len(Ms),
            "extrapolated": v[1] or any(e for _, e in dr)}


def load(ranks_path, key):
    d = torch.load(ranks_path, weights_only=False)
    out = {}
    big = d["meta"]["BIG"]

    def fix(x):  # rank_vd: BIG also marks "target not in the draft vocab" -> BIG-1 (a miss), BIG = invalid depth only
        r, f = x[key].numpy().copy(), x["rank_full"].numpy()
        r[(f < big) & (r >= big)] = big - 1
        return r
    for dn, plist in d["paths"].items():
        out[dn] = [{"kind": p["kind"], "N": p["N"], "R": fix(p)} for p in plist]
    corp = {}
    for name, dd in d["corpus"].items():
        for dn, x in dd.items():
            corp.setdefault(dn, {})[name] = fix(x)
    return d, out, corp


def path_samples(plist, idx=None):
    rows = [p["R"][1:] for i, p in enumerate(plist) if idx is None or i in idx]
    return np.concatenate(rows) if rows else np.zeros((0, 4), np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    tm = json.load(open(a.timing))
    T_ar = tm["T_ar_ms"]
    Tv = {r["M"]: r["us"]["median"] / 1e3 for r in tm["verify"]}
    Tdraft = {}
    for r in tm["draft"]:
        Tdraft.setdefault(r["vd"], {})[r["M"]] = r["us"]["median"] / 1e3
    e4 = json.load(open(E4))["metrics"]["summary"]
    ovh = float(np.median([g["overhead_ms"] for r in e4 if r["skip"] == "none" for g in r["gamma"].values()]))
    V = None
    results = {"T_ar_ms": T_ar, "verify_ms": Tv, "draft_ms": {str(k): v for k, v in Tdraft.items()},
               "overhead_ms_chain_e4": ovh, "configs": []}
    for vocab, key in (("vd32768", "rank_vd"), ("full", "rank_full")):
        d, paths, corp = load(a.ranks, key)
        V = d["meta"]["V"]
        Td = Tdraft[32768 if vocab == "vd32768" else V]
        for dn, plist in paths.items():
            cfg = {"draft": dn, "vocab": vocab}
            n_p = len(plist)
            R_all = path_samples(plist)
            rec = {}
            for kind in ("instruct", "wikitext2"):
                idx = [i for i, p in enumerate(plist) if p["kind"] == kind]
                Rk = path_samples(plist, idx)
                Vk = Rk < (1 << 30)
                rec[kind] = {f"depth{j + 1}": {"n": int(Vk[:, j].sum()), **{
                    f"top{k}": float((Rk[Vk[:, j], j] < k).mean()) for k in range(1, 9)}} for j in range(4)}
                # chain conditional: depth j top-1 given depths < j top-1
                ch = {}
                for j in range(4):
                    ok = Vk[:, j] & (Rk[:, :j] == 0).all(1)
                    ch[f"depth{j + 1}"] = float((Rk[ok, j] == 0).mean()) if ok.any() else None
                rec[kind]["chain_top1_given_prev_top1"] = ch
            for name, Rc in corp.get(dn, {}).items():
                rec[f"corpus_{name}"] = {f"depth{j + 1}": {"n": int(len(Rc)), **{f"top{k}": float((Rc[:, j] < k).mean())
                                                                    for k in range(1, 9)}} for j in range(4)}
            cfg["recall"] = rec
            counts = node_counts(R_all)
            halves = [[i for i in range(n_p) if i % 2 == h] for h in (0, 1)]
            hcounts = [node_counts(path_samples(plist, h)) for h in halves]
            ccounts = node_counts(np.concatenate(list(corp[dn].values()))) if dn in corp else None
            rows = []
            for dmax in range(1, 5):
                for n in range(1, NMAX + 1):
                    tree = best_tree(counts, n, dmax)
                    if len(tree) < n:
                        break
                    c = cost(tree, Td, Tv, ovh)
                    ts = tau_sim(plist, tree)
                    cv = []
                    for h in (0, 1):
                        tr = best_tree(hcounts[h], n, dmax)
                        cv.append(tau_sim([p for i, p in enumerate(plist) if i in halves[1 - h]], tr))
                    cv_all = {k: float(np.mean([x[k] for x in cv if k in x])) for k in ("instruct", "wikitext2", "all")}
                    row = {"n": n, "dmax": dmax, **c, "tau_pos": tau_pos(R_all, tree), "tau_sim": ts, "tau_cv": cv_all,
                           "tree": sorted(tree, key=lambda k: (len(k), k))}
                    if ccounts is not None:
                        trc = best_tree(ccounts, n, dmax)
                        row["tau_corpus_built"] = tau_sim(plist, trc)
                        row["corpus_tau_pos"] = {nm: tau_pos(Rc, tree) for nm, Rc in corp[dn].items()}
                    for k in ("instruct", "wikitext2", "all"):
                        row[f"speedup_{k}"] = ts[k] * T_ar / c["round_ms"]
                        row[f"speedup_cv_{k}"] = cv_all[k] * T_ar / c["round_ms"]
                    row["tau_breakeven"] = c["round_ms"] / T_ar
                    rows.append(row)
            fixed = []
            for w in (1, 2, 4):
                for dd in range(1, 5):
                    tree = full_tree(w, dd)
                    c = cost(tree, Td, Tv, ovh)
                    ts = tau_sim(plist, tree)
                    fr = {"w": w, "d": dd, **c, "tau_pos": tau_pos(R_all, tree), "tau_sim": ts,
                          "tau_breakeven": c["round_ms"] / T_ar,
                          **{f"speedup_{k}": ts[k] * T_ar / c["round_ms"] for k in ts}}
                    if dn in corp:
                        fr["corpus_tau_pos"] = {nm: tau_pos(Rc, tree) for nm, Rc in corp[dn].items()}
                    fixed.append(fr)
            cfg["opt_trees"] = rows
            cfg["fixed_trees"] = fixed
            best = max(rows, key=lambda r: r["speedup_all"])
            best_cv = max(rows, key=lambda r: r["speedup_cv_all"])
            # draft-free ceiling: tau of the best n-node tree with zero draft cost
            ceil = max(rows, key=lambda r: r["tau_sim"]["all"] * T_ar / (r["verify_ms"] + ovh))
            cfg["best"] = {k: best[k] for k in best if k != "tree"} | {"tree": best["tree"]}
            cfg["best_cv"] = {k: best_cv[k] for k in best_cv if k != "tree"}
            cfg["ceiling_zero_draft_cost"] = {"n": ceil["n"], "dmax": ceil["dmax"], "tau_sim": ceil["tau_sim"],
                                              "speedup_all": ceil["tau_sim"]["all"] * T_ar /
                                                             (ceil["verify_ms"] + ovh)}
            cfg["max_tau_sim_n31_d4"] = max(r["tau_sim"]["all"] for r in rows)
            results["configs"].append(cfg)
            print(f"\n=== draft {dn} {vocab} === T_AR {T_ar:.2f} ms")
            for kind, rr in rec.items():
                print(f"  recall {kind:9s} " + " | ".join(
                    f"d{j + 1}: " + " ".join(f"{rr[f'depth{j + 1}'][f'top{k}']:.2f}" for k in (1, 2, 4, 8))
                    for j in range(4)))
            print("  fixed trees (w,d): n tau_sim(chat/wt2/all) round_ms speedup_all")
            for f in fixed:
                print(f"   w{f['w']} d{f['d']} n={f['n']:3d} tau {f['tau_sim']['instruct']:.2f}/"
                      f"{f['tau_sim']['wikitext2']:.2f}/{f['tau_sim']['all']:.2f} round {f['round_ms']:.2f}"
                      f"{'*' if f['extrapolated'] else ''} speedup {f['speedup_all']:.3f}")
            print("  opt trees (best per actual depth over n, dmax):")
            cfg["best_per_depth"] = {}
            for dmax in range(1, 5):
                rr = [r for r in rows if r["depth"] == dmax]
                if rr:
                    b = max(rr, key=lambda r: r["speedup_all"])
                    cfg["best_per_depth"][dmax] = {k: b[k] for k in b}
                    cfg["best_per_depth"][dmax]["max_tau_sim_all"] = max(r["tau_sim"]["all"] for r in rr)
                    print(f"   dmax{dmax}: n={b['n']} levels {b['level_M']} tau_sim {b['tau_sim']['all']:.2f} "
                          f"(cv {b['tau_cv']['all']:.2f}) round {b['round_ms']:.2f} speedup {b['speedup_all']:.3f} "
                          f"(cv {b['speedup_cv_all']:.3f}); max tau {cfg['best_per_depth'][dmax]['max_tau_sim_all']:.2f}")
            print(f"  BEST {best['n']} d{best['depth']} speedup {best['speedup_all']:.3f}; ceiling(zero draft) "
                  f"{cfg['ceiling_zero_draft_cost']['speedup_all']:.3f}")
    if a.save:
        from tools.results import save_result
        config = {
            "setting": "REAL Qwen2.5-32B-Instruct 2-bit nested ckpt d8 K256 L2 noFT; draft = prefix k1 (A = target "
                       "codebook plane 1, B = TV-trained draft codebooks); shared KV (context = target k2 KV, in-round "
                       "tokens = draft k1 KV); greedy tree verification, OFFLINE",
            "ckpt": "/data/lsh/ckpts/weightvq/scale/qwen32b_d8K256L2_noft",
            "draft_B": "/data/lsh/results_accept/scale/qwen32b_d8K256L2_noft/kd1/train_B_bf16/draft_codebooks.pt",
            "ranks_file": a.ranks, "timing_file": a.timing,
            "prompts": "8 chat + 8 wikitext2 64-token prompts (skip_tau.py), target AR k2 greedy path <=128 tok, EOS",
            "corpus": "8 x 512 windows each wikitext2 test / c4 val, round starts every 8 tokens from 64",
            "tree_model": "node = draft-rank tuple; accepted = deepest node matching the target-token rank tuple; "
                          "tau-optimal n-node tree = top-n tuples by empirical frequency (prefix-closed)",
            "timing": "synthetic-model forward M-curves (tree_timing.py, CUDA graph, warm-up excluded, median of reps); "
                      "causal mask, tree-mask cost not measured [I]; overhead = E4 chain loop overhead [I]",
            "expected_speedup": "[I] tau_sim x T_AR / modeled round_ms; NOT a measured end-to-end speed-up",
            "scripts": ["experiments/14_q32b_upper/tree_ranks.py", "experiments/14_q32b_upper/tree_timing.py",
                        "experiments/14_q32b_upper/tree_analyze.py"],
            "rmax_candidates": RMAX, "nmax": NMAX}
        rk = torch.load(a.ranks, weights_only=False)["meta"]
        metrics = {**results, "ranks_meta": rk, "timing_raw": tm}
        print(save_result("E5", a.save, config, metrics, model_dir="/data/models/qwen2.5_32b_instruct"))


if __name__ == "__main__":
    main()
