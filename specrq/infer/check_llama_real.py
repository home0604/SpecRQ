"""Real-checkpoint correctness (Llama-3.1-8B C1): NestModel (C1-specialized kernels) vs the DbD reference forward
(specrq/dbd get_model, QuantizedLinear with Hadamard rotation, bf16) on chat + wikitext2 prompts.

Reference on --ref-device (codes stored as uint8, as experiments/06_accept), runner on cuda:0.
Per prompt:
  ref greedy path   : HF generate (do_sample=False, sampling knobs cleared), --gen tokens
  teacher-forced    : logits on prompt + ref path at the generated positions:
                      ref (one forward) vs runner prefill path (M>16 chunks), runner decode path (M=1 steps over the
                      ref path, the AR kernels: gemv + fused glue), runner verify path (M=5 blocks, mma)
  greedy tokens     : runner AR (CUDA graph, M=1) greedy path vs ref greedy path: first_divergence and, at the first
                      divergence, the top-2 margins of ref and runner logits (near-tie in bf16 ulps)
Optionally with a draft set (--draft-sd, --kd): runner draft logits (use_draft) vs ref draft logits (DbD draft path).
Usage: CUDA_VISIBLE_DEVICES=1,2 python -m specrq.infer.check_llama_real --ckpt DIR --ref-device cuda:1 [--save RUN]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from specrq.infer.decode import ARDecoder, first_divergence
from specrq.infer.model import DBD, NestModel
from specrq.infer.run_decode import INSTRUCT_PROMPTS

REPO = Path(__file__).resolve().parents[2]


def metrics(a, ref):
    a, ref = a.float(), ref.float()
    lp, lq = torch.log_softmax(ref, -1), torch.log_softmax(a, -1)
    return {"rel": ((a - ref).norm() / ref.norm()).item(), "maxabs": (a - ref).abs().max().item(),
            "top1": (a.argmax(-1) == ref.argmax(-1)).float().mean().item(),
            "kl": (lp.exp() * (lp - lq)).sum(-1).mean().item()}


def margin(logits, a, b):
    """l[a] - l[b] and the bf16 ulp of max(|l[a]|, |l[b]|)."""
    la, lb = float(logits[a]), float(logits[b])
    ulp = torch.finfo(torch.bfloat16).eps * max(abs(la), abs(lb))
    return la - lb, ulp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/llama3.1_8b_instruct")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref-device", default="cuda:1")
    ap.add_argument("--n-chat", type=int, default=4)
    ap.add_argument("--n-wt2", type=int, default=4)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--draft-sd", default=None)
    ap.add_argument("--kd", type=int, default=None)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(DBD))
    from src.aq import QuantizedLinear
    from src.datautils import get_loaders
    from src.modelutils import get_model
    from transformers import AutoTokenizer
    import torch.nn as nn

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = []
    for p in INSTRUCT_PROMPTS[: a.n_chat]:
        s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        prompts.append(("chat", tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]))
    wt = get_loaders("wikitext2", seed=0, model_path=a.model, seqlen=512, eval_mode=True)
    for i in range(a.n_wt2):
        prompts.append(("wikitext2", wt[0, (100 + i) * 512:(100 + i) * 512 + 64]))

    # ---- reference (DbD) ----
    q = get_model(a.model, a.ckpt, "auto")
    for mod in q.modules():
        if isinstance(mod, QuantizedLinear):
            qw = mod.quantized_weight
            if qw.codes is not None and qw.codes.dtype != torch.uint8 and int(qw.codes.max()) < 256:
                qw.codes = nn.Parameter(qw.codes.data.to(torch.uint8), requires_grad=False)
    q = q.to(a.ref_device).eval()
    if a.draft_sd:
        from src.draft import load_draft_state_dict
        print("ref draft copies:", load_draft_state_dict(q, torch.load(a.draft_sd, map_location="cpu")), flush=True)

    def set_k(k):
        for mod in q.modules():
            if isinstance(mod, QuantizedLinear):
                mod.active_num_codebooks = k

    ref = []
    with torch.no_grad():
        for kind, pid in prompts:
            set_k(a.k)
            x = pid[None].to(a.ref_device)
            gen = q.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.gen, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=None)[0, pid.numel():].cpu()
            full = torch.cat([pid, gen])[None].to(a.ref_device)
            lt = q(full).logits[0, pid.numel() - 1:-1].float().cpu()  # predicts gen[0..]
            ld = None
            if a.kd:
                set_k(a.kd)
                ld = q(full).logits[0, pid.numel() - 1:-1].float().cpu()
            ref.append((kind, pid, gen, lt, ld))
            print(f"[ref] {kind} prompt len {pid.numel()} -> {tok.decode(gen[:16])!r}", flush=True)
    del q
    torch.cuda.empty_cache()

    # ---- runner ----
    m = NestModel(a.model, a.ckpt)
    if a.draft_sd:
        print("runner draft sets:", m.load_draft(torch.load(a.draft_sd, map_location="cpu", weights_only=False)))
    rows = []
    with torch.no_grad():
        for kind, pid, gen, lt, ld in ref:
            P, G = pid.numel(), gen.numel()
            full = torch.cat([pid, gen]).cuda()
            pos0 = torch.zeros((), dtype=torch.long, device="cuda")
            T = P + G + 16
            c = m.new_cache(T)
            pre = m.forward(full, pos0, c, a.k)[P - 1:P + G - 1].float().cpu()
            # decode path: logits for position t come from feeding token t (prefill covered [0, P))
            c = m.new_cache(T)
            m.forward(full[:P - 1], pos0, c, a.k, last_only=True)
            dec = torch.stack([m.forward(full[t:t + 1], torch.tensor(t, device="cuda"), c, a.k)[0].float().cpu()
                               for t in range(P - 1, P + G - 1)])
            c = m.new_cache(T)
            m.forward(full[:P - 1], pos0, c, a.k, last_only=True)
            ver = torch.cat([m.forward(full[t:min(t + 5, P + G - 1)], torch.tensor(t, device="cuda"), c, a.k).float().cpu()
                             for t in range(P - 1, P + G - 1, 5)])
            r = {"kind": kind, "prompt_len": P, "gen": G, "prefill_vs_ref": metrics(pre, lt),
                 "decode_M1_vs_ref": metrics(dec, lt), "verify_M5_vs_ref": metrics(ver, lt)}
            if a.kd:
                c = m.new_cache(T)
                m.forward(full[:P - 1], pos0, c, a.kd, use_draft=True, last_only=True)
                dd = torch.stack([m.forward(full[t:t + 1], torch.tensor(t, device="cuda"), c, a.kd, use_draft=True)[0]
                                  .float().cpu() for t in range(P - 1, P + G - 1)])
                r["draft_decode_M1_vs_ref_draft"] = metrics(dd, ld)
            # runner AR greedy path (CUDA graph) vs ref greedy path
            ar = ARDecoder(m, T, k=a.k)
            toks = ar.generate(pid.cuda(), G)[0]
            del ar
            i = first_divergence(toks, gen.tolist())
            r["greedy_first_divergence"] = i
            if i is not None and i < G:
                rm, ru = margin(lt[i], int(gen[i]), toks[i])
                r["divergence"] = {"index": i, "ref_tok": int(gen[i]), "runner_tok": toks[i],
                                   "ref_margin": rm, "ref_ulp": ru, "near_tie_ref_ulps": rm / ru if ru else None,
                                   "teacher_forced_runner_top1_matches_ref": bool(dec[i].argmax() == gen[i])}
            rows.append(r)
            print(f"[cmp] {kind} P={P}: prefill {r['prefill_vs_ref']}\n        decode {r['decode_M1_vs_ref']}\n"
                  f"        verify {r['verify_M5_vs_ref']}" + (f"\n        draft {r['draft_decode_M1_vs_ref_draft']}"
                                                              if a.kd else "") +
                  f"\n        greedy first_divergence={i} {r.get('divergence', '')}", flush=True)
    if a.save:
        sys.path.insert(0, str(REPO))
        from tools.results import save_result
        print("saved", save_result("E2", a.save, config={"setting": "real Llama-3.1-8B C1 ckpt: NestModel (C1-specialized) "
                                                          "vs DbD reference (bf16, uint8 codes)", "ckpt": a.ckpt,
                                                          "draft_sd": a.draft_sd, "kd": a.kd, "k": a.k, "gen": a.gen,
                                                          "prompts": f"{a.n_chat} chat + {a.n_wt2} wikitext2 (64 tok)"},
                                   metrics={"rows": rows}, model_dir=a.model))


if __name__ == "__main__":
    main()
