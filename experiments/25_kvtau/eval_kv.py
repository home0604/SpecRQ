"""E6 kvtau step 2: target quality of a 2-bit nested VQ KV cache and greedy tau of KV-prefix drafts (quality only).

Parts
  ppl : teacher-forced PPL @ seqlen (wikitext2 test, c4_new = AQLM c4 validation concat) of the target with bf16 KV
        and with each VQ KV format (all stages), plus the "prefix-KV-only" model (every token reads `level` stages);
        KL(bf16-KV target || VQ-KV target) and top-1 agreement, same weights.
  tau : prompt sets at long (ctx_long) and short (ctx_short) prefill: pg19 (raw book prefix), wt2 (wikitext2 test
        stream window), chat (Llama-3.1 chat template, long pg19 excerpt + a question). For each VQ KV format the
        TARGET (weights k2 / bf16 + VQ KV all stages, incremental decode with a quantized cache) generates a greedy
        path of gen tokens (stop at EOS). Drafts are then evaluated teacher-forced on that path with exact
        shared-KV round semantics (Fwd.draft_steps): step m of a round reads the target's KV prefix (level stages)
        for tokens before the round and its own KV (level stages) for the tokens it drafted in this round.
        tau(gamma) = tokens/iteration of the greedy round walk with step-dependent matches (accepted + 1).
        Also TV / top-1 of the step-1 draft vs the target (teacher-forced, all positions and path positions).
Usage: python experiments/25_kvtau/eval_kv.py --wmode w2 --parts ppl tau --formats a_d8K256x2 ... --variants greedy
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvtau_common import FORMATS, MODEL, OUTROOT, Fwd, KVQuant, load_model, set_weights  # noqa: E402

PG19 = "/data/lsh/datasets/pg19-test/data"
C4_VAL_ARROW = "/data/lsh/hf_cache/datasets/allenai___c4/default-*/0.0.0/607bd4c8450a42878aa9ddc051a65a055450ef87/c4-validation.arrow"
QUESTIONS = [
    "Summarize what happens in this passage in five sentences.",
    "Who are the main characters in this text, and how would you describe each of them?",
    "Describe the setting of this passage and the mood it creates.",
    "What is the central conflict in this passage? Explain briefly.",
    "List the most important events in this passage in order.",
    "Write a short review of this passage as if for a literary magazine.",
    "Explain what the narrator thinks about the other characters.",
    "What questions does this passage leave unanswered? List them.",
]


# ---------------------------------------------------------------- prompts
def build_prompts(tok, ctx, n):
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19}/*.parquet")))
    bos = [tok.bos_token_id]
    books = []
    for text in df["text"]:
        ids = tok(text[: 200000], add_special_tokens=False).input_ids
        if len(ids) >= 3 * 8192:
            books.append(ids)
        if len(books) >= 2 * n:
            break
    sets = {"pg19": [], "wt2": [], "chat": []}
    for b in books[:n]:  # raw continuation of the book start
        sets["pg19"].append(torch.tensor(bos + b[: ctx - 1]))
    from src.datautils import get_loaders
    wt = get_loaders("wikitext2", seed=0, model_path=MODEL, seqlen=2048, eval_mode=True)[0]
    stride = (wt.numel() - ctx) // n
    for i in range(n):
        sets["wt2"].append(torch.cat([torch.tensor(bos), wt[i * stride + 1: i * stride + ctx]]))
    for i, b in enumerate(books[n: 2 * n]):  # long-document chat: excerpt from the middle of another book + question
        q = QUESTIONS[i % len(QUESTIONS)]
        head = tok.apply_chat_template([{"role": "user", "content": "Read the following text.\n\n<text>\n"}],
                                       tokenize=False, add_generation_prompt=False)
        lo, hi = 50, ctx  # binary search the excerpt length so the whole prompt is <= ctx tokens
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            doc = tok.decode(b[8192: 8192 + mid])
            msg = [{"role": "user", "content": f"Read the following text.\n\n<text>\n{doc}\n</text>\n\n{q}"}]
            ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
            if len(ids) <= ctx:
                best, lo = ids, mid + 1
            else:
                hi = mid - 1
        sets["chat"].append(torch.tensor(best))
    return sets


# ---------------------------------------------------------------- metrics
def tv_top1(f, ht, hd, chunk=1024):
    """hidden [T, hid] target / draft -> per-position TV and top-1 agreement (fp32 logits, chunked)."""
    tv, t1 = [], []
    for i in range(0, ht.shape[0], chunk):
        lt, ld = f.logits(ht[i:i + chunk]), f.logits(hd[i:i + chunk])
        tv.append(0.5 * (lt.softmax(-1) - ld.softmax(-1)).abs().sum(-1))
        t1.append((lt.argmax(-1) == ld.argmax(-1)).float())
    return torch.cat(tv), torch.cat(t1)


def walk(matches, gamma, T):
    """matches[m-1][t]: step-m draft argmax == path token t. Greedy round walk -> tokens / iteration."""
    i, iters, acc = 0, 0, 0
    while i < T:
        a = 0
        while a < gamma and i + a < T and matches[a][i + a]:
            a += 1
        acc += a; iters += 1; i += a + 1
    return T, iters, acc


@torch.no_grad()
def generate(f, ids, kvq, level, gen, eos):
    cache = {"k": [], "v": [], "pos": 0}
    h = f.forward(ids, kvq, level, cache=cache, return_hidden=True)
    nxt = f.logits(h[:, -1]).argmax(-1)
    out = [nxt.item()]
    for _ in range(gen - 1):
        if out[-1] in eos:
            break
        h = f.forward(nxt[:, None], kvq, level, cache=cache, return_hidden=True)
        nxt = f.logits(h[:, -1]).argmax(-1)
        out.append(nxt.item())
    del cache
    return out


@torch.no_grad()
def run_tau(model, f, tok, kvq, S, tgt_w, drafts, sets, gen, G, eos):
    """drafts: list of (name, weights, level) ; level None = bf16 KV (only when kvq is None)."""
    res = {}
    for sname, prompts in sets.items():
        per = []
        for pi, p in enumerate(prompts):
            t0 = time.time()
            ids = p[None].cuda()
            P = ids.shape[1]
            set_weights(model, tgt_w)
            path = generate(f, ids, kvq, S, gen, eos)
            full = torch.cat([ids, torch.tensor([path], device=ids.device)], 1)
            T = len(path)
            levels = sorted(set(d[2] for d in drafts), key=lambda x: -1 if x is None else x)
            collect = {lv: [] for lv in levels}
            ht = f.forward(full, kvq, S, collect=collect, return_hidden=True, fp32attn=True)[0]
            # acceptance reference = the target's teacher-forced (verify-mode) argmax at each path position; the
            # incremental-decode path only supplies the context (VQ code flips between 1-row and many-row k/v
            # projections make the two differ more often than with bf16 KV; see target_tf_vs_gen)
            tgt = f.logits(ht[P - 1:P - 1 + T]).argmax(-1).cpu()
            consist = (tgt == torch.tensor(path)).float().mean().item()
            rec = {"set": sname, "idx": pi, "P": P, "T": T, "target_tf_vs_gen": consist, "drafts": {}}
            for dname, w, lv in drafts:
                set_weights(model, w)
                hd = f.draft_steps(full, collect[lv], kvq, lv, G)
                set_weights(model, tgt_w)
                dpred = [f.logits(hd[m, P - 1:P - 1 + T]).argmax(-1).cpu() for m in range(G)]
                matches = [(x == tgt).tolist() for x in dpred]
                tv, t1 = tv_top1(f, ht[:-1], hd[0, :-1])
                d = {"match_rate_step": [sum(x) / T for x in matches],
                     "match_rate_step1_vs_genpath": (dpred[0] == torch.tensor(path)).float().mean().item(),
                     "tv_all": tv.mean().item(), "top1_all": t1.mean().item(),
                     "tv_path": tv[P - 1:P - 1 + T].mean().item(), "top1_path": t1[P - 1:P - 1 + T].mean().item()}
                for g in (1, 2, 4):
                    if g <= G:
                        d[f"g{g}"] = walk(matches, g, T)
                        d[f"g{g}_indep"] = walk([matches[0]] * g, g, T)  # step-1 semantics for all steps (approx.)
                rec["drafts"][dname] = d
                del hd
            per.append(rec)
            del collect, ht
            torch.cuda.empty_cache()
            print(sname, pi, P, T, f"{time.time() - t0:.0f}s",
                  {k: (round(v["match_rate_step"][0], 3), round(v["g1"][0] / v["g1"][1], 3)) for k, v in rec["drafts"].items()},
                  flush=True)
        agg = {}
        for dname, _, _ in drafts:
            a = {}
            for g in (1, 2, 4):
                for sfx in ("", "_indep"):
                    k = f"g{g}{sfx}"
                    if k in per[0]["drafts"][dname]:
                        toks = sum(r["drafts"][dname][k][0] for r in per)
                        its = sum(r["drafts"][dname][k][1] for r in per)
                        a[f"tau{k}"] = toks / its
            for k in ("tv_all", "top1_all", "tv_path", "top1_path"):
                a[k] = sum(r["drafts"][dname][k] for r in per) / len(per)
            a["match_rate_step"] = [sum(r["drafts"][dname]["match_rate_step"][m] for r in per) / len(per) for m in range(G)]
            agg[dname] = a
        res[sname] = {"agg": agg, "per_prompt": per,
                      "target_tf_vs_gen": sum(r["target_tf_vs_gen"] for r in per) / len(per),
                      "mean_P": sum(r["P"] for r in per) / len(per), "mean_T": sum(r["T"] for r in per) / len(per)}
        print(sname, json.dumps(agg), flush=True)
    return res


@torch.no_grad()
def run_ppl(model, f, tgt_w, kvqs, seqlen, nwin, partial_path=None, corpora=("wikitext2", "c4_new")):
    from src.datautils import get_loaders
    out = {}
    for corpus in corpora:
        if corpus == "c4_new":
            # = AQLM get_c4_new eval (first 1100 docs of c4 validation shard 0 joined by " "), read from the local
            # arrow cache directly (load_dataset in offline mode misses the cached config hash)
            from datasets import Dataset
            from transformers import AutoTokenizer
            vd = Dataset.from_file(glob.glob(C4_VAL_ARROW)[0])
            tk = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
            enc = tk(" ".join(vd[:1100]["text"]), return_tensors="pt").input_ids[:, :256 * seqlen]
        else:
            enc = get_loaders(corpus, seed=0, model_path=MODEL, seqlen=seqlen, eval_mode=True)
        n = min(nwin, enc.numel() // seqlen)
        acc = {}
        for i in range(n):
            ids = enc[:, i * seqlen:(i + 1) * seqlen].cuda()
            y = ids[0, 1:]
            set_weights(model, tgt_w)
            hr = f.forward(ids, None, None, return_hidden=True)[0, :-1]
            cfgs = [("bf16kv", None, None)]
            for name, (kvq, S, dlev) in kvqs.items():
                cfgs += [(f"{name}|L{S}", kvq, S)] + [(f"{name}|L{lv}", kvq, lv) for lv in dlev]
            for cname, kvq, lv in cfgs:
                h = hr if kvq is None else f.forward(ids, kvq, lv, return_hidden=True)[0, :-1]
                a = acc.setdefault(cname, {"nll": 0.0, "kl": 0.0, "top1": 0.0, "n": 0})
                for j in range(0, h.shape[0], 1024):
                    lr, lt = f.logits(hr[j:j + 1024]), f.logits(h[j:j + 1024])
                    lpr, lpt = lr.log_softmax(-1), lt.log_softmax(-1)
                    a["nll"] += F.nll_loss(lpt, y[j:j + 1024], reduction="sum").item()
                    a["kl"] += (lpr.exp() * (lpr - lpt)).sum().item()
                    a["top1"] += (lr.argmax(-1) == lt.argmax(-1)).sum().item()
                    a["n"] += lt.shape[0]
            if i % 10 == 0:
                print(corpus, i, {k: round(float(torch.exp(torch.tensor(v["nll"] / v["n"]))), 3) for k, v in acc.items()}, flush=True)
        out[corpus] = {k: {"ppl": float(torch.exp(torch.tensor(v["nll"] / v["n"]))), "kl_bf16kv_to_cfg": v["kl"] / v["n"],
                           "top1_vs_bf16kv": v["top1"] / v["n"], "n_tokens": v["n"]} for k, v in acc.items()}
        out[corpus]["n_windows"] = n
        print(corpus, json.dumps(out[corpus]), flush=True)
        if partial_path:
            json.dump(out, open(partial_path, "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wmode", choices=["bf16", "w2"], required=True)
    ap.add_argument("--parts", nargs="+", default=["ppl", "tau"])
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--variants", nargs="+", default=["greedy", "refine"])
    ap.add_argument("--ppl_seqlen", type=int, default=4096)
    ap.add_argument("--ppl_windows", type=int, default=40)
    ap.add_argument("--ctxs", type=int, nargs="+", default=[6144, 512])
    ap.add_argument("--n_prompts", type=int, default=8)
    ap.add_argument("--sets", nargs="+", default=["pg19", "wt2", "chat"])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--cb_suffix", default="", help="codebook file suffix, e.g. _cn (calib_kv.py --chnorm --suffix _cn)")
    ap.add_argument("--keep_first", type=int, default=0, help="first n positions stay bf16 (attention sink)")
    ap.add_argument("--no_bf16kv_tau", action="store_true", help="skip the bf16-KV reference tau run")
    a = ap.parse_args()
    t00 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = {128001, 128008, 128009}  # Llama-3.1-Instruct generation_config eos ids
    model = load_model(a.wmode, draft_b=(a.wmode == "w2"))
    if a.wmode == "w2":
        from kvtau_common import densify
        print("dense k2 copies for", densify(model), "linears", flush=True)
    f = Fwd(model)
    tgt_w = "k2" if a.wmode == "w2" else "bf16"
    od = os.path.join(OUTROOT, a.wmode); os.makedirs(od, exist_ok=True)
    kvqs = {}
    for fmt in a.formats:
        for var in a.variants:
            kq = KVQuant(os.path.join(od, f"cb_{fmt}{a.cb_suffix}_{var}.pt"))
            kq.keep_first = a.keep_first
            kvqs[f"{fmt}{a.cb_suffix}_{var}_keep{a.keep_first}"] = (kq, kq.S, FORMATS[fmt][2])
    if "ppl" in a.parts:
        pp = os.path.join(od, f"ppl_{a.tag}{a.cb_suffix}_keep{a.keep_first}.json")
        r = run_ppl(model, f, tgt_w, kvqs, a.ppl_seqlen, a.ppl_windows, partial_path=pp + ".partial")
        json.dump(r, open(pp, "w"), indent=1)
    if "tau" in a.parts:
        for ctx in a.ctxs:
            sets = {k: v for k, v in build_prompts(tok, ctx, a.n_prompts).items() if k in a.sets}
            # reference: bf16-KV target, weight-only draft (w2 only) / sanity draft == target
            runs = ([] if a.no_bf16kv_tau else [("bf16kv", None, None)]) + [(name, kq, S) for name, (kq, S, _) in kvqs.items()]
            for name, kq, S in runs:
                fn = os.path.join(od, f"tau_{a.tag}_ctx{ctx}_{name}.json")
                if os.path.exists(fn):
                    print("exists", fn); continue
                if kq is None:
                    drafts = [("same_W|bf16kv", tgt_w, None)] + ([("k1B|bf16kv", "k1B", None)] if a.wmode == "w2" else [])
                else:
                    dlev = kvqs[name][2]
                    drafts = [(f"same_W|kv{lv}", tgt_w, lv) for lv in dlev]
                    if name.startswith("a_") and "greedy" in name:
                        drafts.append((f"same_W|kv{S}(sanity)", tgt_w, S))
                    if a.wmode == "w2":
                        drafts += [(f"k1B|kv{lv}", "k1B", lv) for lv in dlev] + [(f"k1B|kv{S}", "k1B", S)]
                print(f"== tau ctx {ctx} {name} drafts {[d[0] for d in drafts]}", flush=True)
                r = run_tau(model, f, tok, kq, S, tgt_w, drafts, sets, a.gen, a.G, eos)
                r["config"] = {"wmode": a.wmode, "kv": name, "ctx": ctx, "gen": a.gen, "G": a.G, "drafts": drafts}
                json.dump(r, open(fn, "w"), indent=1, default=str)
    print(f"done {time.time() - t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
