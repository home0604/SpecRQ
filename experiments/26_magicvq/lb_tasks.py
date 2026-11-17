"""E7 magicvq v4: long-context workloads where eviction may lose information (LongBench + local synthetic tasks).

Every builder returns a list of candidate prompts (token-id lists, BOS included, len <= ctx) for one (task, ctx);
the tau driver (lb_tau.py) keeps the first n whose target greedy output has >= min_gen tokens.

LongBench (THUDM/LongBench data.zip in /data/lsh/datasets/longbench/data, base + _e files merged by _id):
  gov_report, qmsum, multi_news : LongBench prompt templates, Llama-3.1 chat template.
  lcc, repobench-p              : LongBench templates, raw completion (no chat template, as LongBench pred.py does
                                  for these tasks); the target simply continues the code for `gen` tokens.
  hotpotqa_q                    : hotpotqa passages + question, the instruction asks for the answer AND verbatim
                                  supporting quotes (long output). ctx 32k: hotpotqa samples are <= 16.3k tokens, so
                                  passages of other hotpotqa samples are added as distractors (shuffled, renumbered).
  Middle truncation as LongBench: the filled template is cut to (ctx - chat overhead) tokens keeping the first and the
  last half. Samples are taken longest-first among those with >= `min_frac` * ctx tokens (fixed seed shuffle within).
Synthetic (haystack = PG-19 test books 40.., not used by the v1-v3 PG-19 / chat sets):
  syn_quote : quote verbatim the paragraph that begins with a given 10-word prefix (paragraph at depth 5-60 %).
  syn_kv    : 16 "The access code of <name> is <code>." lines scattered through the haystack; list all 16 in a given
              order.
  syn_cont  : 6 ledgers (40 records each) scattered at depths 5-60 %; the assistant turn is prefilled with the title
              and first 3 records of one ledger and the instruction asks to continue it verbatim.
"""
from __future__ import annotations

import glob
import json
import random
import re

LB = "/data/lsh/datasets/longbench/data"
PG19 = "/data/lsh/datasets/pg19-test/data"

TEMPL = {
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n"
                  "{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in "
             "one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting "
             "transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n"
                  "Now, write a one-page summary of all the news.\n\nSummary:",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
    "hotpotqa_q": "Answer the question based on the given passages. First state the answer in one sentence. Then quote, "
                  "word for word, every sentence from the passages that supports the answer, one quote per line, each "
                  "prefixed with its passage number.\n\nThe following are given passages.\n{context}\n\nAnswer the "
                  "question based on the given passages. First state the answer in one sentence, then give the verbatim "
                  "supporting quotes with their passage numbers.\n\nQuestion: {input}\nAnswer:",
}
RAW = {"lcc", "repobench-p"}
LB_TASKS = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p", "hotpotqa_q"]
SYN_TASKS = ["syn_quote", "syn_kv", "syn_cont"]


def _lb_rows(task):
    base = "hotpotqa" if task == "hotpotqa_q" else task
    rows = {}
    for f in (base, base + "_e"):
        try:
            for line in open(f"{LB}/{f}.jsonl"):
                r = json.loads(line)
                rows[r["_id"]] = r
        except FileNotFoundError:
            pass
    return [rows[k] for k in sorted(rows)]


def _chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def _finish(tok, task, prompt, ctx):
    """LongBench middle truncation of the filled template to fit ctx tokens incl. BOS / chat template."""
    over = len(_chat(tok, "")) if task not in RAW else 1
    ids = tok(prompt, add_special_tokens=False).input_ids
    lim = ctx - over - 8
    if len(ids) > lim:
        h = lim // 2
        prompt = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(ids[-h:], skip_special_tokens=True)
    out = [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids if task in RAW else _chat(tok, prompt)
    while len(out) > ctx:  # decode / re-encode drift
        ids = tok(prompt, add_special_tokens=False).input_ids
        h = (len(ids) - (len(out) - ctx) - 4) // 2
        prompt = tok.decode(ids[:h], skip_special_tokens=True) + tok.decode(ids[-h:], skip_special_tokens=True)
        out = [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids if task in RAW else _chat(tok, prompt)
    return out


def _passages(ctx_text):
    parts = re.split(r"Passage \d+:\n", ctx_text)
    return [p.strip() for p in parts if p.strip()]


def lb_prompts(tok, task, ctx, n_cand, min_frac=0.75, seed=0):
    rows = _lb_rows(task)
    rng = random.Random(seed)
    if task == "hotpotqa_q" and ctx > 17000:
        pool = [p for r in rows for p in _passages(r["context"])]
        cands = []
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order[:n_cand]:
            r = rows[i]
            own = set(_passages(r["context"]))
            ps = list(own)
            tot = len(tok(r["context"], add_special_tokens=False).input_ids)
            while tot < ctx * 1.05:
                p = pool[rng.randrange(len(pool))]
                if p in own:
                    continue
                ps.append(p)
                tot += len(tok(p, add_special_tokens=False).input_ids) + 6
            rng.shuffle(ps)
            c = "\n\n".join(f"Passage {k + 1}:\n{p}" for k, p in enumerate(ps))
            cands.append({"id": r["_id"], "ids": _finish(tok, task, TEMPL[task].format(context=c, input=r["input"]), ctx),
                          "orig_tokens": tot, "distractors": len(ps) - len(own)})
        return cands
    lens = []
    for r in rows:
        L = len(tok(r["context"] + r["input"], add_special_tokens=False).input_ids)
        if L >= min_frac * ctx:
            lens.append((L, r))
    rng.shuffle(lens)
    cands = []
    for L, r in lens[:n_cand]:
        cands.append({"id": r["_id"], "ids": _finish(tok, task, TEMPL[task].format(context=r["context"], input=r["input"]), ctx),
                      "orig_tokens": L})
    return cands


# ---------------------------------------------------------------- synthetic
FIRST = ["Margaret", "Thomas", "Eleanor", "Arthur", "Beatrice", "Walter", "Clara", "Henry", "Agnes", "Frederick", "Lydia",
         "Edmund", "Harriet", "Oswald", "Matilda", "Rupert", "Cecily", "Ambrose", "Florence", "Gideon", "Winifred",
         "Lionel", "Rosalind", "Percival", "Imogen", "Bartholomew", "Philippa", "Cornelius", "Dorothea", "Silas"]
LAST = ["Holloway", "Pemberton", "Ashcroft", "Whitlock", "Farrington", "Blackwood", "Thornbury", "Kingsley", "Marlowe",
        "Prescott", "Davenport", "Hargreaves", "Fairfax", "Wainwright", "Crowther", "Lockhart", "Radcliffe", "Sterling",
        "Huxley", "Carrington", "Ellsworth", "Montague", "Rutherford", "Beckett", "Tremaine", "Godwin"]
CITY = ["Lisbon", "Kraków", "Valparaíso", "Tromsø", "Marrakesh", "Hobart", "Quebec", "Tbilisi", "Cusco", "Galway", "Nagasaki",
        "Zanzibar", "Bergen", "Trieste", "Odessa", "Halifax", "Perth", "Seville", "Riga", "Salzburg", "Kyoto", "Asmara"]
ITEM = ["copper kettles", "wool blankets", "brass lanterns", "linen sheets", "oak barrels", "glass bottles", "iron nails",
        "silk ribbons", "tin buttons", "leather boots", "clay pipes", "hemp rope", "candle wax", "salt cod", "tea chests",
        "cotton thread", "pewter mugs", "whale oil", "coffee sacks", "indigo dye"]
ADJ = ["amber", "silent", "crimson", "northern", "hollow", "golden", "velvet", "frozen", "hidden", "copper", "ancient",
       "restless", "silver", "broken", "distant", "emerald", "pale", "iron", "wandering", "quiet"]
NOUN = ["falcon", "lantern", "harbor", "orchard", "compass", "meadow", "anchor", "violin", "glacier", "monastery",
        "lighthouse", "canyon", "sparrow", "observatory", "vineyard", "bridge", "telescope", "garden", "chapel", "river"]


def load_haystack_books(tok, n=40, start=40, min_tokens=40000, max_chars=400000):
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{PG19}/*.parquet")))
    out, k = [], 0
    for text in df["text"]:
        t = text[:max_chars]
        if len(tok(t, add_special_tokens=False).input_ids) >= min_tokens:
            if k >= start:
                out.append(t)
            k += 1
        if len(out) >= n:
            break
    return out


def _paras(text):
    """PG-19 paragraphs (blank-line separated, inner line breaks joined)."""
    ps = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in ps if p]


def _hay(tok, paras, budget):
    """leading paragraphs of a book up to `budget` tokens -> list of (para, ntok)."""
    out, tot = [], 0
    for p in paras:
        n = len(tok(p, add_special_tokens=False).input_ids) + 2
        if tot + n > budget:
            break
        out.append(p); tot += n
    return out, tot


def _code(rng):
    return f"{rng.choice(ADJ).upper()}-{rng.randint(1000, 9999)}-{rng.choice(NOUN).upper()}-{rng.randint(10, 99)}"


def syn_prompts(tok, task, ctx, n_cand, books, seed=0):
    rng = random.Random(1000 + seed + ctx)
    cands = []
    head = {"syn_quote": "Read the following book excerpt.", "syn_kv": "Read the following book excerpt.",
            "syn_cont": "Read the following document."}[task]
    for bi in range(n_cand):
        paras = _paras(books[bi % len(books)])
        rng.shuffle(paras) if bi >= len(books) else None
        if task == "syn_quote":
            ins_q = ("\n\nQuote verbatim the paragraph of the excerpt that begins with \"{p}\". Reproduce the whole "
                     "paragraph exactly, word for word, and output nothing else.")
            budget = ctx - 200
            hay, _ = _hay(tok, paras, budget)
            lo, hi = int(0.05 * len(hay)), int(0.60 * len(hay))
            ok = [i for i in range(lo, hi) if 150 <= len(tok(hay[i], add_special_tokens=False).input_ids) <= 600
                  and len(hay[i].split()) > 20]
            ok = [i for i in ok if sum(p.startswith(" ".join(hay[i].split()[:10])) for p in hay) == 1]
            if not ok:
                continue
            i = rng.choice(ok)
            pre = " ".join(hay[i].split()[:10])
            text = f"{head}\n\n<excerpt>\n" + "\n\n".join(hay) + "\n</excerpt>" + ins_q.format(p=pre)
            ids = _fit(tok, text, hay, ctx, lambda h: f"{head}\n\n<excerpt>\n" + "\n\n".join(h) + "\n</excerpt>"
                       + ins_q.format(p=pre), keep=i)
            cands.append({"id": f"{task}_{bi}", "ids": ids, "depth": i / len(hay), "target_par_tokens":
                          len(tok(hay[i], add_special_tokens=False).input_ids), "ref": hay[i]})
        elif task == "syn_kv":
            K = 16
            names = []
            while len(names) < K:
                nm = f"the {rng.choice(ADJ)} {rng.choice(NOUN)}"
                if nm not in names:
                    names.append(nm)
            codes = [_code(rng) for _ in range(K)]
            hay, _ = _hay(tok, paras, ctx - 600)
            pos = sorted(rng.sample(range(1, max(K + 2, int(0.9 * len(hay)))), K))
            body = list(hay)
            for p, nm, c in sorted(zip(pos, names, codes), reverse=True):
                body.insert(min(p, len(body)), f"The access code of {nm} is {c}.")
            order = list(range(K))
            rng.shuffle(order)
            ask = ("\n\nThe excerpt contains sentences of the form \"The access code of X is Y.\" List the access code of "
                   "each of the following, in this order, one per line as \"X: Y\":\n"
                   + "\n".join(names[o] for o in order))
            ids = _fit(tok, None, body, ctx, lambda h: f"{head}\n\n<excerpt>\n" + "\n\n".join(h) + "\n</excerpt>" + ask,
                       keep_pred=lambda s: s.startswith("The access code of "))
            cands.append({"id": f"{task}_{bi}", "ids": ids, "needle_depths": [p / len(body) for p in pos],
                          "ref": [f"{names[o]}: {codes[o]}" for o in order]})
        else:  # syn_cont
            L, NR = 6, 40
            ledgers = []
            for li in range(L):
                title = f"Ledger of the {rng.choice(ADJ).capitalize()} {rng.choice(NOUN).capitalize()} Company"
                recs = [f"{k + 1:03d} | {rng.choice(FIRST)} {rng.choice(LAST)} | {rng.choice(CITY)} | "
                        f"{rng.randint(2, 480)} {rng.choice(ITEM)} | {rng.randint(10, 9999)}.{rng.randint(0, 99):02d}"
                        for k in range(NR)]
                ledgers.append((title, recs))
            blocks = [f"{t}\n" + "\n".join(r) + "\nEnd of ledger." for t, r in ledgers]
            ltok = sum(len(tok(b, add_special_tokens=False).input_ids) + 2 for b in blocks)
            hay, _ = _hay(tok, paras, ctx - ltok - 450)
            pos = sorted(rng.sample(range(int(0.05 * len(hay)) + 1, int(0.6 * len(hay))), L))
            body = list(hay)
            for p, b in sorted(zip(pos, blocks), reverse=True):
                body.insert(p, b)
            tgt = rng.randrange(L)
            title, recs = ledgers[tgt]
            ask = (f"\n\nThe document contains a section titled \"{title}\". Reproduce that ledger exactly as it "
                   "appears in the document, record by record, continuing from where the beginning below stops.")
            pre = f"{title}\n" + "\n".join(recs[:3]) + "\n"
            def mk(h):
                return f"{head}\n\n<document>\n" + "\n\n".join(h) + "\n</document>" + ask
            ids = _fit(tok, None, body, ctx - len(tok(pre, add_special_tokens=False).input_ids) - 2, mk,
                       keep_pred=lambda s: s.startswith("Ledger of the "))
            ids = ids + tok(pre, add_special_tokens=False).input_ids
            cands.append({"id": f"{task}_{bi}", "ids": ids, "ledger_depth": pos[tgt] / len(body), "ref": recs[3:]})
    return cands


def _fit(tok, _text, hay, ctx, mk, keep=None, keep_pred=None):
    """drop haystack paragraphs from the end (never `keep` / predicate matches) until the chat prompt fits ctx."""
    h = list(hay)
    prot = [False] * len(h)
    if keep is not None:
        prot[keep] = True
    if keep_pred is not None:
        prot = [p or keep_pred(s) for p, s in zip(prot, h)]
    while True:
        ids = _chat(tok, mk(h))
        if len(ids) <= ctx:
            return ids
        over = len(ids) - ctx
        j = len(h) - 1
        while prot[j]:
            j -= 1
        # drop a few unprotected paragraphs near the end (keeps the protected items' relative depth roughly fixed)
        del h[j]; del prot[j]
        if over > 2000:
            for _ in range(over // 400):
                j = len(h) - 1
                while prot[j]:
                    j -= 1
                del h[j]; del prot[j]
