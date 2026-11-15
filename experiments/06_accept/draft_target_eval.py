"""Draft (prefix k_d) vs target (prefix k_t) agreement for one nested DbD checkpoint, plus target fidelity vs FP16.

Token-level metrics (teacher-forced on held-out text; draft and target see the same prefix, as in speculative decoding
where the prefix is always target-accepted):
  TV(p_T, p_D) = 0.5 * sum |p_T - p_D|  -> speculative-sampling acceptance per token is 1 - TV
  top-1 agreement argmax p_T == argmax p_D (greedy acceptance per token)
  KL(p_T || p_D), KL(p_D || p_T)
Target fidelity vs the FP16 model: PPL, KL(p_FP16 || p_T), top-1 agreement.
Greedy speculative simulation: the target (k_t) generates a greedy path from each prompt; the draft's teacher-forced
argmax on that path gives match_t; for gamma in {1,2,4} we walk the path: accept consecutive matches (<= gamma), then
the target supplies one token. tokens/iteration = accepted + 1 (exact for greedy if logits are batch-invariant, which
bf16 is not exactly; see CLAUDE.md note on near-tie flips).

Usage: python draft_target_eval.py --model DIR --ckpt DIR [--kd 3] [--kt 5] [--out JSON] [--n_windows 40] ...
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.aq import QuantizedLinear
from src.datautils import get_loaders
from src.modelutils import get_model

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


def shrink_codes_(model):
    """[specrq] store codes as uint8 when every stage has <= 256 entries (codes are cast to int32 per forward by
    QuantizedWeight.get_codes, so outputs are unchanged); needed to fit Llama-8B C1 (int32 codes ~17 GB) on one GPU."""
    import torch.nn as nn
    n = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            qw = m.quantized_weight
            if qw.codes is not None and qw.codes.dtype != torch.uint8 and int(qw.codes.max()) < 256 and int(qw.codes.min()) >= 0:
                qw.codes = nn.Parameter(qw.codes.data.to(torch.uint8), requires_grad=False)
                n += 1
    return n


def set_k(model, k, use_draft=True):
    """[specrq] use_draft=False hides the draft-only copies (draft_k -> None), so a target call never takes the draft
    path even when kd == kt (target-side codebook-format probe); for kd < kt the target path is unchanged."""
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m.active_num_codebooks = k
            if hasattr(m, "draft_k") or hasattr(m, "_draft_k_saved"):
                if not hasattr(m, "_draft_k_saved"):
                    m._draft_k_saved = m.draft_k
                m.draft_k = m._draft_k_saved if use_draft else None


@torch.no_grad()
def logits_at(model, ids, k=None, use_draft=True):
    if k is not None:
        set_k(model, k, use_draft)
    return model(ids).logits.float()[0]  # [T, V]


def token_metrics(lt, ld):
    pt, pd = lt.softmax(-1), ld.softmax(-1)
    lpt, lpd = lt.log_softmax(-1), ld.log_softmax(-1)
    tv = 0.5 * (pt - pd).abs().sum(-1)
    return {
        "tv": tv, "top1": (lt.argmax(-1) == ld.argmax(-1)).float(),
        "kl_t_d": (pt * (lpt - lpd)).sum(-1), "kl_d_t": (pd * (lpd - lpt)).sum(-1),
    }


def accepted_length_sim(match, gamma):
    """match[t] = draft argmax == target token at position t of the target greedy path."""
    i, iters, acc_total, T = 0, 0, 0, len(match)
    while i < T:
        a = 0
        while a < gamma and i + a < T and match[i + a]:
            a += 1
        acc_total += a
        iters += 1
        i += a + 1  # accepted tokens + one target token
    return {"tokens_per_iter": T / iters if iters else 0.0, "mean_accepted": acc_total / iters if iters else 0.0,
            "iters": iters, "tokens": T}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kd", type=int, default=3)
    ap.add_argument("--kt", type=int, default=5)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--n_windows", type=int, default=40, help="windows per corpus (wikitext2 test, c4 val)")
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--n_corpus_prompts", type=int, default=8, help="extra greedy prompts cut from wikitext2 (64 tok)")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--draft", default=None, help="draft_codebooks.pt from train_draft_codebooks.py (draft-only copies)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--codes_uint8", action="store_true", help="store 8-bit codes as uint8 (memory; outputs unchanged)")
    ap.add_argument("--ref_device", default="cuda:0", help="device of the FP16 reference model")
    ap.add_argument("--ref_max_memory", default=None,
                    help="[specrq] dispatch the FP16 reference with accelerate device_map='auto' under this budget, e.g. "
                         "'1:21GiB,cpu:150GiB' (large models; CPU part is offloaded per forward); default: whole model on "
                         "--ref_device (unchanged)")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    q = get_model(a.model, a.ckpt, "auto")
    if a.codes_uint8:
        print("codes -> uint8 for", shrink_codes_(q), "linears")
    q = q.to(dev).eval()
    if a.draft:
        from src.draft import load_draft_state_dict
        print("loaded draft copies for", load_draft_state_dict(q, torch.load(a.draft)), "linears")
    if a.no_fp16:
        ref = None
    elif a.ref_max_memory:
        mm = {(k if k == "cpu" else int(k)): v for k, v in (x.split(":") for x in a.ref_max_memory.split(","))}
        ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=q.dtype, device_map="auto", max_memory=mm).eval()
        a.ref_device = str(ref.get_input_embeddings().weight.device)
        print("ref dispatched:", sorted(set(map(str, ref.hf_device_map.values()))), "input device", a.ref_device)
    else:
        ref = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=q.dtype).to(a.ref_device).eval()
    res = {"model": a.model, "ckpt": a.ckpt, "draft": a.draft, "kd": a.kd, "kt": a.kt, "seqlen": a.seqlen}

    # ---- teacher-forced corpus metrics ----
    for name in ["wikitext2", "c4"]:
        enc = get_loaders(name, seed=0, model_path=a.model, seqlen=a.seqlen, eval_mode=True)
        n = min(a.n_windows, enc.numel() // a.seqlen)
        acc = {k: [] for k in ["tv", "top1", "kl_t_d", "kl_d_t", "nll_t", "nll_d", "nll_ref", "kl_ref_t", "top1_ref_t"]}
        for i in range(n):
            ids = enc[:, i * a.seqlen:(i + 1) * a.seqlen].to(dev)
            lt, ld = logits_at(q, ids, a.kt, use_draft=False), logits_at(q, ids, a.kd)
            m = token_metrics(lt[:-1], ld[:-1])
            for k in ["tv", "top1", "kl_t_d", "kl_d_t"]:
                acc[k].append(m[k].cpu())
            y = ids[0, 1:]
            acc["nll_t"].append(F.cross_entropy(lt[:-1], y, reduction="none").cpu())
            acc["nll_d"].append(F.cross_entropy(ld[:-1], y, reduction="none").cpu())
            if ref is not None:
                lr = ref(ids.to(a.ref_device)).logits.float()[0].to(dev)
                acc["nll_ref"].append(F.cross_entropy(lr[:-1], y, reduction="none").cpu())
                pr, lpr, lpt = lr[:-1].softmax(-1), lr[:-1].log_softmax(-1), lt[:-1].log_softmax(-1)
                acc["kl_ref_t"].append((pr * (lpr - lpt)).sum(-1).cpu())
                acc["top1_ref_t"].append((lr[:-1].argmax(-1) == lt[:-1].argmax(-1)).float().cpu())
        cat = {k: torch.cat(v) for k, v in acc.items() if v}
        r = {"n_windows": n, "n_tokens": int(cat["tv"].numel()),
             "tv_mean": cat["tv"].mean().item(), "accept_prob_1mTV": 1 - cat["tv"].mean().item(),
             "top1_agree": cat["top1"].mean().item(), "kl_t_d": cat["kl_t_d"].mean().item(), "kl_d_t": cat["kl_d_t"].mean().item(),
             "ppl_target": torch.exp(cat["nll_t"].mean()).item(), "ppl_draft": torch.exp(cat["nll_d"].mean()).item()}
        if ref is not None:
            r.update({"ppl_fp16": torch.exp(cat["nll_ref"].mean()).item(), "kl_fp16_target": cat["kl_ref_t"].mean().item(),
                      "top1_fp16_target": cat["top1_ref_t"].mean().item()})
        res[name] = r
        print(name, json.dumps(r), flush=True)

    # ---- greedy speculative simulation on target greedy paths ----
    prompts = []
    for p in INSTRUCT_PROMPTS:
        msgs = [{"role": "user", "content": p}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if tok.chat_template else p
        prompts.append(("instruct", tok(s, return_tensors="pt", add_special_tokens=False).input_ids))
    wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=a.seqlen, eval_mode=True)
    for i in range(a.n_corpus_prompts):
        prompts.append(("wikitext2", wt[:, (100 + i) * a.seqlen:(100 + i) * a.seqlen + 64]))
    sims = {g: {"instruct": [], "wikitext2": []} for g in (1, 2, 4)}
    match_rates, consist, per_prompt = {"instruct": [], "wikitext2": []}, [], []
    for kind, pid in prompts:
        pid = pid.to(dev)
        set_k(q, a.kt, use_draft=False)
        with torch.no_grad():
            # pure greedy: override the model's generation_config (Qwen2.5-Instruct sets repetition_penalty=1.05,
            # top_k/top_p/temperature), otherwise the "greedy" path is not the target argmax path
            gen = q.generate(pid, attention_mask=torch.ones_like(pid), max_new_tokens=a.gen_tokens, do_sample=False,
                             repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        full = gen[:, :]
        P = pid.shape[1]
        path = full[0, P:]
        lt, ld = logits_at(q, full, a.kt, use_draft=False), logits_at(q, full, a.kd)
        pred_t = lt[P - 1:-1].argmax(-1)
        pred_d = ld[P - 1:-1].argmax(-1)
        consist.append((pred_t == path).float().mean().item())  # teacher-forced target argmax vs generated path
        match = (pred_d == path).tolist()
        match_rates[kind].append(sum(match) / len(match))
        per_prompt.append({"kind": kind, "len": len(match), "match_rate": sum(match) / len(match),
                           "first_mismatch": next((i for i, x in enumerate(match) if not x), len(match))})
        for g in (1, 2, 4):
            sims[g][kind].append(accepted_length_sim(match, g))
    res["greedy_sim"] = {}
    for kind in ("instruct", "wikitext2"):
        d = {"draft_match_rate_on_target_path": sum(match_rates[kind]) / len(match_rates[kind])}
        for g in (1, 2, 4):
            xs = sims[g][kind]
            d[f"gamma{g}"] = {"tokens_per_iter": sum(x["tokens"] for x in xs) / sum(x["iters"] for x in xs),
                              "mean_accepted": sum(x["mean_accepted"] * x["iters"] for x in xs) / sum(x["iters"] for x in xs)}
        res["greedy_sim"][kind] = d
    res["greedy_sim"]["target_teacher_forced_vs_generated_consistency"] = sum(consist) / len(consist)
    res["greedy_sim"]["per_prompt"] = per_prompt
    res["n_prompts"] = len(prompts); res["gen_tokens"] = a.gen_tokens; res["wall_s"] = time.time() - t0
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps(res["greedy_sim"]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
