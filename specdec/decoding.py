"""
Autoregressive baseline and self-speculative decoding (batch 1).

Speculative loop, with n = committed KV length and `cur` = last committed token (KV not yet cached):
  draft  : γ single-token draft passes from `cur`, sampling d_1..d_γ from q (draft cache, m stages)
  verify : one target pass on [cur, d_1..d_γ] at positions n..n+γ -> p_0..p_γ (target cache);
           this also re-writes the draft cache for those positions from the target's K/V
  accept : greedy  — accept while argmax p_{j} == d_{j+1}
           sample  — accept d_{j+1} w.p. min(1, p/q); on reject resample from norm(max(p-q, 0))
           all accepted -> bonus token from p_γ
  commit : n += a + 1, cur = correction / bonus token
"""

import time

import torch

from specdec.kvstore import KVStore, SnapKVRep


def _probs(logits, temperature):
    if temperature == 0:
        return None
    return torch.softmax(logits / temperature, -1)


def _prefill(model, rep, prompt, kv):
    obs = (rep.window, rep.observe) if isinstance(rep, SnapKVRep) else None
    return model.prefill(prompt.view(1, -1).to(model.device), kv, observer_last=obs)[0, -1]


def _pick(logits, temperature, gen):
    if temperature == 0:
        return int(logits.argmax(-1))
    return int(torch.multinomial(_probs(logits, temperature), 1, generator=gen))


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _Ctx:
            def __enter__(self):
                torch.cuda.synchronize()
                self.s = time.perf_counter()

            def __exit__(self, *a):
                torch.cuda.synchronize()
                timer.t[key] = timer.t.get(key, 0.0) + time.perf_counter() - self.s
        return _Ctx()


@torch.no_grad()
def autoregressive(model, rep, prompt, n_new, temperature=0.0, seed=0):
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    kv = KVStore(model, rep, 1, prompt.numel() + n_new + 1)
    tm = Timer()
    with tm("prefill"):
        logits = _prefill(model, rep, prompt, kv)
    out = [_pick(logits, temperature, gen)]
    n = prompt.numel()
    with tm("decode"):
        while len(out) < n_new:
            logits = model.forward(torch.tensor([[out[-1]]], device=dev), n, kv, "target")[0, -1]
            n += 1
            kv.commit(n)
            out.append(_pick(logits, temperature, gen))
    return out, {"time": tm.t, "decode_tok_per_s": (n_new - 1) / tm.t["decode"]}


@torch.no_grad()
def speculative(model, rep, prompt, n_new, gamma=4, temperature=0.0, seed=0):
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    kv = KVStore(model, rep, 1, prompt.numel() + n_new + gamma + 2)
    tm = Timer()
    with tm("prefill"):
        logits = _prefill(model, rep, prompt, kv)
    out = [_pick(logits, temperature, gen)]
    n, cur = prompt.numel(), out[0]
    accepted, iters = [], 0
    with tm("decode"):
        while len(out) < n_new:
            # ---- draft
            drafts, qs = [], []
            inp = cur
            with tm("draft"):
                for i in range(gamma):
                    lg = model.forward(torch.tensor([[inp]], device=dev), n + i, kv, "draft")[0, -1]
                    qs.append(_probs(lg, temperature))
                    inp = _pick(lg, temperature, gen)
                    drafts.append(inp)
            # ---- verify
            with tm("verify"):
                seq = torch.tensor([[cur] + drafts], device=dev)
                plog = model.forward(seq, n, kv, "target")[0]          # (γ+1, V)
            # ---- accept
            a, nxt = 0, None
            if temperature == 0:
                tgt = plog.argmax(-1).tolist()
                while a < gamma and tgt[a] == drafts[a]:
                    a += 1
                nxt = tgt[a]
            else:
                p = torch.softmax(plog / temperature, -1)
                while a < gamma:
                    d = drafts[a]
                    r = torch.rand((), device=dev, generator=gen)
                    if r < (p[a, d] / qs[a][d]).clamp(max=1.0):
                        a += 1
                        continue
                    resid = (p[a] - qs[a]).clamp(min=0)
                    nxt = int(torch.multinomial(resid / resid.sum(), 1, generator=gen))
                    break
                if nxt is None:
                    nxt = int(torch.multinomial(p[gamma], 1, generator=gen))
            out += drafts[:a] + [nxt]
            accepted.append(a)
            n += a + 1
            kv.commit(n)
            cur = nxt
            iters += 1
    out = out[:n_new]
    acc = torch.tensor(accepted, dtype=torch.float)
    return out, {
        "time": tm.t, "iters": iters, "gamma": gamma,
        "accept_rate": acc.sum().item() / (gamma * iters),          # accepted drafts / proposed
        "mean_accepted": acc.mean().item(),                          # per iteration, ≤ γ
        "tokens_per_iter": acc.mean().item() + 1,
        "accept_hist": torch.bincount(acc.long(), minlength=gamma + 1).tolist(),
        "decode_tok_per_s": (len(out) - 1) / tm.t["decode"],
    }
