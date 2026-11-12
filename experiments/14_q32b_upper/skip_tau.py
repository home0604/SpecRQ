"""E5/E4 follow-up: acceptance (tau) of "1-bit prefix (k1) + draft decoder-layer skip" on the REAL Qwen2.5-32B 2-bit
nested checkpoint (d8 K256 L2 noFT), measured on our runner (NestModel + nestkern kernels), so the draft semantics are
exactly those of the timing run (results/E4/q32b_L2_synth_prefix_layerskip.json, q32b_upper.py --mode skip).

Draft A = target codebook (first plane), B = draft-only TV-trained codebooks+scales (--draft_b, train_draft_codebooks.py).
Skip: SkipView (q32b_upper.py): draft forwards (k < L) run only the kept layers; skipped layers pass the residual
through and neither compute nor write KV. Verify (k2) runs all layers.
Prompts / protocol of experiments/06_accept/draft_target_eval.py: 8 chat prompts (chat template) + 8 wikitext2-test
cuts of 64 tokens (windows 100..107 of 512), target greedy path of up to 128 tokens (generation_config EOS stop).

Per (draft, skip, vd):
  sim     : teacher-forced (draft runs the whole prompt+path with its OWN k1/skip KV, = existing protocol) argmax vs the
            target greedy path -> greedy simulation tokens/iter for gamma 1/2/4 (accepted_length_sim, path incl. the
            prefill token). vd: argmax over the FR-Spec subset columns of the full-vocab draft logits.
  decode  : real SpecDecoder rounds (draft_kv="shared" as in the timing run: the draft attends to target k2 KV for the
            committed context, its own k1 KV only for the in-round tokens; with skip only the kept layers) ->
            tau = confirmed decode tokens / rounds (prefill token excluded), output vs target AR path (exactness).
  corpus  : teacher-forced TV(p_T, p_D), top-1 agreement on n_windows x 512 tokens of wikitext2 test and c4 val.
Optional --own_check: decode with draft_kv="own" for the no-skip configs (bridge between sim and shared decode).
Timings printed here are single-pass and informational only (not the Notion-03 protocol).
Usage: python experiments/14_q32b_upper/skip_tau.py --skips none back16 ... --out JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "14_q32b_upper"))
sys.path.insert(0, str(REPO / "specrq" / "dbd"))

import torch  # noqa: E402

from specrq.infer.decode import ARDecoder, SpecDecoder  # noqa: E402
from specrq.infer.model import NestModel, load_draft_file  # noqa: E402
from q32b_upper import SkipView, skip_patterns  # noqa: E402

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
    """Same as draft_target_eval.accepted_length_sim."""
    i, iters, acc_total, T = 0, 0, 0, len(match)
    while i < T:
        a = 0
        while a < gamma and i + a < T and match[i + a]:
            a += 1
        acc_total += a
        iters += 1
        i += a + 1
    return {"tokens": T, "iters": iters, "acc": acc_total}


@torch.no_grad()
def full_logits(view, ids, k, use_draft, T):
    """Teacher-forced forward over ids [M] (fresh cache, own KV); returns bf16 logits [M, V]."""
    cache = view.new_cache(T)
    pos = torch.zeros((), dtype=torch.long, device=ids.device)
    out = view.forward(ids, pos, cache, k, use_draft)
    del cache
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/qwen2.5_32b_instruct")
    ap.add_argument("--ckpt", default="/data/lsh/ckpts/weightvq/scale/qwen32b_d8K256L2_noft")
    ap.add_argument("--draft_b", default="/data/lsh/results_accept/scale/qwen32b_d8K256L2_noft/kd1/train_B_bf16/"
                                         "draft_codebooks.pt")
    ap.add_argument("--extra_drafts", nargs="*", default=[], help="NAME=PATH draft files (e.g. skip-aware retrained B)")
    ap.add_argument("--drafts", nargs="+", default=["A", "B"])
    ap.add_argument("--skips", nargs="+", default=["none"])
    ap.add_argument("--vds", type=int, nargs="+", default=[32768, 0], help="draft vocab for decode (0 = full)")
    ap.add_argument("--gammas", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--n_windows", type=int, default=8, help="512-token windows per corpus for TV/top-1 (0 = off)")
    ap.add_argument("--own_check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_start = time.time()
    from transformers import AutoTokenizer, GenerationConfig
    from src.datautils import get_loaders
    from specrq.infer.frspec import coverage, default_freq_dir, load_subset

    m = NestModel(a.model, a.ckpt)
    drafts = {"A": None, "B": a.draft_b}
    for s in a.extra_drafts:
        n, p = s.split("=", 1)
        drafts[n] = p
    # one draft codebook set per k on the model: extra drafts are swapped in when used
    n_b = m.load_draft(load_draft_file(a.draft_b)) if "B" in a.drafts else 0
    print("[skip_tau] draft B linears", n_b, flush=True)
    V = m.lm_head.shape[0]
    fd = default_freq_dir(m.cfg)
    sub = {}
    for vd in a.vds:
        if vd:
            ids = load_subset(fd, vd, None)
            m.set_draft_vocab(ids)
            sub[vd] = ids.to(m.device)
    tok = AutoTokenizer.from_pretrained(a.model)
    e = GenerationConfig.from_pretrained(a.model).eos_token_id
    eos = sorted(set(e if isinstance(e, list) else [e]))
    prompts = []
    for p in INSTRUCT_PROMPTS:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        prompts.append(("instruct", tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]))
    wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=512, eval_mode=True)
    for i in range(8):
        prompts.append(("wikitext2", wt[0, (100 + i) * 512:(100 + i) * 512 + 64]))
    prompts = [(k, p.to(m.device)) for k, p in prompts]
    T = (max(p.numel() for _, p in prompts) + a.gen_tokens + max(a.gammas) + 32 + 15) // 16 * 16

    # ---- target greedy paths (AR k2, EOS stop) ----
    ar = ARDecoder(m, T, k=m.L, eos=eos)
    ar.generate(prompts[0][1], 8)  # warm-up
    paths = [ar.generate(p, a.gen_tokens)[0] for _, p in prompts]
    del ar
    torch.cuda.empty_cache()
    print("[skip_tau] path lens", [len(x) for x in paths], flush=True)
    full = [torch.cat([p, torch.tensor(x, device=m.device)]) for (_, p), x in zip(prompts, paths)]
    tgt_pred, consist = [], []
    for (_, p), x, f in zip(prompts, paths, full):
        lt = full_logits(m, f, m.L, False, T)
        P = p.numel()
        pr = lt[P - 1:-1].argmax(-1)
        tgt_pred.append(pr)
        consist.append((pr == torch.tensor(x, device=m.device)).float().mean().item())
    print("[skip_tau] target tf-vs-generated consistency", sum(consist) / len(consist), flush=True)

    # ---- corpus windows: target logits once (bf16, as produced) ----
    wins = []
    if a.n_windows:
        for name in ("wikitext2", "c4"):
            enc = get_loaders(name, seed=0, model_path=a.model, seqlen=512, eval_mode=True)
            for i in range(min(a.n_windows, enc.numel() // 512)):
                ids = enc[0, i * 512:(i + 1) * 512].to(m.device)
                wins.append((name, ids, full_logits(m, ids, m.L, False, 512)[:-1]))
    print(f"[skip_tau] setup done {time.time() - t_start:.0f}s", flush=True)

    pats = skip_patterns(m.n_layers, a.skips)
    res = {"args": vars(a), "eos": eos, "T": T, "path_lens": [len(x) for x in paths],
           "kinds": [k for k, _ in prompts], "target_tf_consistency": sum(consist) / len(consist),
           "skip_patterns": pats, "vd_coverage_pileval": {vd: coverage(fd, sub[vd].cpu()) for vd in sub},
           "freq_dir": fd, "rows": []}
    cur_extra = None
    for dn in a.drafts:
        use_draft = dn != "A"
        if dn not in ("A", "B") and cur_extra != dn:
            print("[skip_tau] draft", dn, m.load_draft(load_draft_file(drafts[dn])), "linears", flush=True)
            cur_extra = dn
        elif dn == "B" and cur_extra is not None:
            m.load_draft(load_draft_file(a.draft_b))
            cur_extra = None
        for sn, sk in pats.items():
            sv = SkipView(m, sk)
            t0 = time.time()
            row = {"draft": dn, "skip": sn, "n_skip": len(sk), "sim": {}, "corpus": {}, "decode": {}}
            # teacher-forced on target paths (own KV)
            dl = [full_logits(sv, f, 1, use_draft, T)[p.numel() - 1:-1] for (_, p), f in zip(prompts, full)]
            for vd in [0] + [v for v in a.vds if v]:
                agg = {}
                for kind in ("instruct", "wikitext2"):
                    st = {g: [] for g in a.gammas}
                    mr, tp1 = [], []
                    for (kk, _), x, d, tpr in zip(prompts, paths, dl, tgt_pred):
                        if kk != kind:
                            continue
                        pred = d.argmax(-1) if not vd else sub[vd][d[:, sub[vd]].argmax(-1)]
                        match = (pred == torch.tensor(x, device=m.device)).tolist()
                        mr.append(sum(match) / len(match))
                        for g in a.gammas:
                            st[g].append(accepted_length_sim(match, g))
                    agg[kind] = {"match_rate": sum(mr) / len(mr)}
                    for g in a.gammas:
                        agg[kind][f"gamma{g}"] = sum(s["tokens"] for s in st[g]) / sum(s["iters"] for s in st[g])
                row["sim"][vd or V] = agg
            del dl
            # corpus TV / top-1 (full vocab) and top-1 within the vd subset
            for name in ("wikitext2", "c4"):
                tv, t1, t1v = [], [], {v: [] for v in sub}
                for nm, ids, lt in wins:
                    if nm != name:
                        continue
                    ld = full_logits(sv, ids, 1, use_draft, 512)[:-1].float()
                    ltf = lt.float()
                    tv.append((0.5 * (ltf.softmax(-1) - ld.softmax(-1)).abs().sum(-1)).cpu())
                    ta = ltf.argmax(-1)
                    t1.append((ta == ld.argmax(-1)).float().cpu())
                    for v, s in sub.items():
                        t1v[v].append((ta == s[ld[:, s].argmax(-1)]).float().cpu())
                    del ld, ltf
                if tv:
                    row["corpus"][name] = {"tv": torch.cat(tv).mean().item(), "top1": torch.cat(t1).mean().item(),
                                           **{f"top1_vd{v}": torch.cat(x).mean().item() for v, x in t1v.items()}}
            # real decode (shared KV, as in the timing run)
            kv_modes = ["shared"] + (["own"] if a.own_check and not sk else [])
            for kv in kv_modes:
                for vd in a.vds:
                    for g in a.gammas:
                        sd = SpecDecoder(sv, T, g, 1, None, use_draft, vd=vd or None, eos=eos, draft_kv=kv)
                        agg = {kind: {"tok": 0, "rounds": 0, "draft_s": 0.0, "ver_s": 0.0, "dec_s": 0.0}
                               for kind in ("instruct", "wikitext2")}
                        exact, firstdiv = 0, []
                        for (kind, p), x in zip(prompts, paths):
                            out, s = sd.generate(p, a.gen_tokens)
                            q = agg[kind]
                            q["tok"] += s["n_decode_tokens"]; q["rounds"] += s["rounds"]
                            q["draft_s"] += s["t_draft_s"]; q["ver_s"] += s["t_verify_s"]; q["dec_s"] += s["t_decode_s"]
                            exact += out == x
                            firstdiv.append(next((i for i, (u, w) in enumerate(zip(out, x)) if u != w),
                                                 min(len(out), len(x))) if out != x else None)
                        del sd
                        torch.cuda.empty_cache()
                        r = {}
                        for kind, q in agg.items():
                            r[kind] = {"tau": q["tok"] / q["rounds"], "tokens": q["tok"], "rounds": q["rounds"],
                                       "round_ms_info": 1e3 * q["dec_s"] / q["rounds"],
                                       "draft_ms_info": 1e3 * q["draft_s"] / q["rounds"]}
                        r["exact_prompts"] = exact
                        r["first_divergence"] = firstdiv
                        row["decode"][f"{kv}_vd{vd or V}_g{g}"] = r
                        print(f"[{dn} {sn} {kv} vd{vd or V} g{g}] tau chat {r['instruct']['tau']:.3f} "
                              f"wt2 {r['wikitext2']['tau']:.3f} exact {exact}/{len(prompts)}", flush=True)
            row["wall_s"] = time.time() - t0
            print(f"[{dn} {sn}] sim {json.dumps(row['sim'])} corpus {json.dumps(row['corpus'])} "
                  f"({row['wall_s']:.0f}s)", flush=True)
            res["rows"].append(row)
            json.dump(res, open(a.out, "w"), indent=1)
    res["wall_s"] = time.time() - t_start
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    json.dump(res, open(a.out, "w"), indent=1)
    print("[skip_tau] done", res["wall_s"], flush=True)


if __name__ == "__main__":
    main()
