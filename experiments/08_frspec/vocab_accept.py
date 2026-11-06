"""FR-Spec draft vocab: greedy draft/target agreement vs V_d on a nested-VQ checkpoint (NestModel forward).

Teacher-forced on wikitext2 test / c4 val windows (eval sets; the frequency table is from pile-val):
  agree(kd, V_d)   = [argmax over the V_d subset of draft(kd) logits == argmax over the full vocab of target logits]
  oov_target(V_d)  = target argmax not in the V_d subset (an upper bound on what FR-Spec can lose per token)
  coverage(V_d)    = gold next token in the subset
Greedy speculative simulation on target greedy paths (ARDecoder, k_t = L, 128 tokens; instruct chat prompts +
wikitext2 prompts as in experiments/06_accept/draft_target_eval.py): per (kd, V_d) the subset-argmax draft match on the
path -> tokens/iteration for gamma in {1, 2, 4}.
Subset logits are taken as columns of the full draft logits here (same lm_head rows; the runner uses the gathered
[V_d, H] buffer). Usage: python experiments/08_frspec/vocab_accept.py --ckpt DIR --run NAME [--kds 3 2 1]  (-> results/E4/NAME.json)
"""
import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from specrq.infer.decode import ARDecoder  # noqa: E402
from specrq.infer.frspec import coverage, default_freq_dir, load_subset  # noqa: E402
from specrq.infer.model import NestModel  # noqa: E402

INSTRUCT_PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean at night.",
    "What are three common causes of high blood pressure?",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Give step-by-step instructions to make a cup of green tea.",
    "Why is the sky blue? Answer for a ten-year-old.",
    "Translate to French: The meeting has been moved to Thursday afternoon.",
    "List five tips for writing clear technical documentation.",
]


def accepted_length_sim(match, gamma):
    i, iters, T = 0, 0, len(match)
    while i < T:
        a = 0
        while a < gamma and i + a < T and match[i + a]:
            a += 1
        iters += 1
        i += a + 1
    return T, iters


@torch.no_grad()
def logits_full(m, ids, k):
    cache = m.new_cache(ids.numel() + 16)
    pos = torch.zeros((), dtype=torch.long, device=m.device)
    return m.forward(ids, pos, cache, k).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_0.5b_instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--freq-dir", default=None)
    ap.add_argument("--kds", type=int, nargs="+", default=[3, 2, 1])
    ap.add_argument("--vds", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 0])
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--n_windows", type=int, default=40)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--n_corpus_prompts", type=int, default=8)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, os.path.join(REPO, "specrq", "dbd"))
    from src.datautils import get_loaders
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = NestModel(a.model, a.ckpt)
    V = m.lm_head.shape[0]
    kt = m.L
    freq_dir = a.freq_dir or default_freq_dir(m.cfg)
    subs = {vd: (load_subset(freq_dir, vd, V).to(m.device) if vd else None) for vd in a.vds}
    res = {"model": a.model, "ckpt": a.ckpt, "freq_dir": freq_dir, "kt": kt, "kds": a.kds, "vds": a.vds, "V": V,
           "seqlen": a.seqlen, "pileval_coverage": {str(vd or V): (coverage(freq_dir, s.cpu()) if vd else 1.0)
                                                    for vd, s in subs.items()}}
    inset = {vd: (torch.zeros(V, dtype=torch.bool, device=m.device).index_fill_(0, s, True) if vd else None)
             for vd, s in subs.items()}

    def sub_argmax(l, vd):
        return l.argmax(-1) if vd == 0 else subs[vd][l[:, subs[vd]].argmax(-1)]

    for name in ("wikitext2", "c4"):
        enc = get_loaders(name, seed=0, model_path=a.model, seqlen=a.seqlen, eval_mode=True)
        n = min(a.n_windows, enc.numel() // a.seqlen)
        agree = {(kd, vd): 0 for kd in a.kds for vd in a.vds}
        agree_kt = {vd: 0 for vd in a.vds}
        oov_t = {vd: 0 for vd in a.vds}
        gold = {vd: 0 for vd in a.vds}
        ntok = 0
        for i in range(n):
            ids = enc[0, i * a.seqlen:(i + 1) * a.seqlen].to(m.device)
            lt = logits_full(m, ids, kt)[:-1]
            ta = lt.argmax(-1)
            y = ids[1:]
            ntok += y.numel()
            for vd in a.vds:
                agree_kt[vd] += int((sub_argmax(lt, vd) == ta).sum())
                if vd:
                    oov_t[vd] += int((~inset[vd][ta]).sum())
                    gold[vd] += int(inset[vd][y].sum())
            for kd in a.kds:
                ld = logits_full(m, ids, kd)[:-1]
                for vd in a.vds:
                    agree[(kd, vd)] += int((sub_argmax(ld, vd) == ta).sum())
                del ld
            del lt
        r = {"n_windows": n, "n_tokens": ntok,
             "agree": {f"kd{kd}": {str(vd or V): agree[(kd, vd)] / ntok for vd in a.vds} for kd in a.kds},
             "agree_kt_subset": {str(vd or V): agree_kt[vd] / ntok for vd in a.vds},
             "target_argmax_oov": {str(vd): oov_t[vd] / ntok for vd in a.vds if vd},
             "gold_in_subset": {str(vd): gold[vd] / ntok for vd in a.vds if vd}}
        res[name] = r
        print(name, json.dumps(r), flush=True)

    # greedy target paths
    prompts = []
    for p in INSTRUCT_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) \
            if tok.chat_template else p
        prompts.append(("instruct", tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]))
    wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=a.seqlen, eval_mode=True)
    for i in range(a.n_corpus_prompts):
        prompts.append(("wikitext2", wt[0, (100 + i) * a.seqlen:(100 + i) * a.seqlen + 64]))
    T = max(p.numel() for _, p in prompts) + a.gen_tokens + 16
    ar = ARDecoder(m, T, k=kt)
    sims = {(kind, kd, vd, g): [0, 0] for kind in ("instruct", "wikitext2") for kd in a.kds for vd in a.vds
            for g in (1, 2, 4)}
    match_n = {(kind, kd, vd): [0, 0] for kind in ("instruct", "wikitext2") for kd in a.kds for vd in a.vds}
    oov_path = {(kind, vd): [0, 0] for kind in ("instruct", "wikitext2") for vd in a.vds if vd}
    oov_ids = {(kind, vd): {} for kind in ("instruct", "wikitext2") for vd in a.vds if vd}
    for kind, pid in prompts:
        pid = pid.to(m.device)
        path, _ = ar.generate(pid, a.gen_tokens)
        path = torch.tensor(path, device=m.device)
        full = torch.cat([pid, path])
        P = pid.numel()
        for vd in a.vds:
            if vd:
                o = oov_path[(kind, vd)]
                o[0] += int((~inset[vd][path]).sum())
                o[1] += path.numel()
                for t in path[~inset[vd][path]].tolist():
                    oov_ids[(kind, vd)][t] = oov_ids[(kind, vd)].get(t, 0) + 1
        for kd in a.kds:
            ld = logits_full(m, full, kd)[P - 1:-1]
            for vd in a.vds:
                match = (sub_argmax(ld, vd) == path).tolist()
                mn = match_n[(kind, kd, vd)]
                mn[0] += sum(match)
                mn[1] += len(match)
                for g in (1, 2, 4):
                    t, it = accepted_length_sim(match, g)
                    sims[(kind, kd, vd, g)][0] += t
                    sims[(kind, kd, vd, g)][1] += it
    gs = {}
    for kind in ("instruct", "wikitext2"):
        d = {"target_path_oov": {str(vd): oov_path[(kind, vd)][0] / oov_path[(kind, vd)][1] for vd in a.vds if vd}}
        d["target_path_oov_top_ids"] = {str(vd): [[t, tok.convert_ids_to_tokens(t), c] for t, c in
                                                  sorted(oov_ids[(kind, vd)].items(), key=lambda x: -x[1])[:20]]
                                         for vd in a.vds if vd}
        for kd in a.kds:
            d[f"kd{kd}"] = {str(vd or V): {"match_rate": match_n[(kind, kd, vd)][0] / match_n[(kind, kd, vd)][1],
                                           **{f"gamma{g}_tokens_per_iter": sims[(kind, kd, vd, g)][0] /
                                              sims[(kind, kd, vd, g)][1] for g in (1, 2, 4)}} for vd in a.vds}
        gs[kind] = d
    res["greedy_sim"] = gs
    res["note"] = ("teacher-forced draft on the target greedy path; path from NestModel ARDecoder (CUDA graph, M=1), "
                   "draft logits from one M=len forward (bf16 near-tie flips possible)")
    res["wall_s"] = time.time() - t0
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    print(json.dumps(gs), flush=True)
    from tools.results import save_result
    cfg = {k: res.pop(k) for k in ("model", "ckpt", "freq_dir", "kt", "kds", "vds", "V", "seqlen")}
    cfg.update({"setting": "FR-Spec draft vocab: greedy draft(kd, V_d) vs target(kt) agreement, NestModel, bf16",
                "n_windows": a.n_windows, "gen_tokens": a.gen_tokens, "n_corpus_prompts": a.n_corpus_prompts,
                "instruct_prompts": INSTRUCT_PROMPTS, "chat_forced_draft_vocab": True})
    print("saved", save_result("E4", a.run, config=cfg, metrics=res, model_dir=a.model))


if __name__ == "__main__":
    main()
