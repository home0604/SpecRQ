"""Greedy AR baseline and greedy self-speculative decoding on NestModel, with CUDA-graph-captured steps.

Target = prefix k_t (default all L codebooks), draft = prefix k_d (target codebooks, or a separate draft codebook set
installed with NestModel.load_draft when use_draft=True). Separate static KV caches for draft and target.

AR   : one graph = one decode step (forward M=1, argmax written on device into the next-input buffer and the output
       buffer, pos += 1 on device). Replayed n times with no host sync in between (optimized baseline).
Spec : per round, with n = committed KV length (target cache valid for [0, n)), cur = last committed token:
       draft  graph : gamma M=1 draft steps from cur at positions n..n+gamma-1 (argmax chained on device)
                      variant "catch-up" (after a fully accepted round the draft cache lacks d_gamma): first step is
                      M=2 on [d_gamma_prev, cur] at positions n-1, n
       verify graph : target forward M=gamma+1 on [cur, d_1..d_gamma] at positions n..n+gamma, argmax per row
       host         : a = matched prefix length; commit d_1..d_a + target token v_a; n += a + 1
       Rollback is implicit: stale KV beyond the committed length is masked (position mask) and later overwritten.
Timing (Notion 03): decode wall = perf_counter over the decode loop (sync at end); draft / verify = CUDA-event time of
the draft / verify graph replays; overhead = wall - draft - verify (host acceptance, buffer updates, launch gaps).
Prefill is timed separately (eager forward, chunked mma).
FR-Spec (vd = V_d): every draft step uses the partial-vocab lm_head (NestModel.set_draft_vocab) and maps its argmax
back to token ids on device inside the graph; verify always uses the full lm_head (exact greedy). ARDecoder(vd=...)
exists only to time a draft-style step (its output is the subset-restricted greedy path, not the target's).
EOS stop (eos = iterable of stop ids, e.g. generation_config eos; None = off = previous behaviour): a request ends at
the first confirmed stop token (kept in the output) or at n_new confirmed tokens (first token from prefill included).
  AR  : graph replays are pipelined with a 1-step-delayed host check (step j is enqueued before the token of step j-1
        is inspected via a pinned async copy + event), so the GPU never idles; at most one wasted step after EOS,
        and it is inside the timed wall.
  Spec: the accepted tokens d_1..d_a, v_a of a round are cut at the first stop token; output is also cut at n_new.
  tokens_per_s = confirmed decode tokens (after the cut, excluding the prefill token) / decode wall time.
"""
from __future__ import annotations

import time

import torch


def _sync():
    torch.cuda.synchronize()


class _Graph:
    def __init__(self, fn):
        fn()  # warm-up (lazy extension loads / workspaces) outside capture
        _sync()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            fn()
        _sync()

    def replay(self):
        self.g.replay()


class ARDecoder:
    def __init__(self, model, T_max, k=None, use_draft=False, n_max=4096, vd=None, eos=None):
        self.m, self.k, self.use_draft, self.vd = model, k, use_draft, vd
        self.eos = frozenset(int(e) for e in eos) if eos else None
        self.hbuf = torch.zeros(n_max, dtype=torch.long, pin_memory=True) if self.eos else None
        self.hev = [torch.cuda.Event() for _ in range(n_max)] if self.eos else None
        self.cache = model.new_cache(T_max)
        dev = model.device
        self.tok = torch.zeros(1, dtype=torch.long, device=dev)
        self.pos = torch.zeros((), dtype=torch.long, device=dev)
        self.out = torch.zeros(n_max, dtype=torch.long, device=dev)
        self.idx = torch.zeros((), dtype=torch.long, device=dev)
        self.graph = _Graph(self._step)

    def _step(self):
        logits = self.m.forward(self.tok, self.pos, self.cache, self.k, self.use_draft, last_only=True, vd=self.vd)
        nxt = self.m.to_token_ids(logits.argmax(-1), self.vd)
        self.out.index_copy_(0, self.idx.view(1), nxt)
        self.tok.copy_(nxt)
        self.pos.add_(1)
        self.idx.add_(1)

    @torch.no_grad()
    def generate(self, prompt, n_new):
        m = self.m
        _sync()
        t0 = time.perf_counter()
        logits = m.prefill(prompt, self.cache, self.k, self.use_draft, vd=self.vd)
        first = m.to_token_ids(logits.argmax(-1), self.vd)
        _sync()
        t_prefill = time.perf_counter() - t0
        self.tok.copy_(first)
        self.pos.fill_(prompt.numel())
        self.idx.zero_()
        _sync()
        if self.eos is None:
            t1 = time.perf_counter()
            for _ in range(n_new - 1):
                self.graph.replay()
            _sync()
            t_dec = time.perf_counter() - t1
            toks = [int(first)] + self.out[: n_new - 1].tolist()
            return toks, {"t_prefill_s": t_prefill, "t_decode_s": t_dec, "n_decode_tokens": n_new - 1,
                          "tokens_per_s": (n_new - 1) / t_dec}
        n_dec = n_new - 1  # confirmed decode tokens
        first_i = int(first)
        t1 = time.perf_counter()
        if first_i in self.eos:
            n_dec = 0
        else:
            hb, hev = self.hbuf, self.hev
            for j in range(n_new - 1):
                self.graph.replay()
                hb[j:j + 1].copy_(self.tok, non_blocking=True)
                hev[j].record()
                if j >= 1:
                    hev[j - 1].synchronize()
                    if int(hb[j - 1]) in self.eos:
                        n_dec = j  # tokens 0..j-1
                        break
            # no break: the last token (index n_new - 2) is the n_new-th confirmed token either way
        _sync()
        t_dec = time.perf_counter() - t1
        toks = [first_i] + self.out[:n_dec].tolist()
        return toks, {"t_prefill_s": t_prefill, "t_decode_s": t_dec, "n_decode_tokens": n_dec,
                      "tokens_per_s": n_dec / t_dec if t_dec > 0 and n_dec else 0.0,
                      "eos_hit": toks[-1] in self.eos}


class SpecDecoder:
    def __init__(self, model, T_max, gamma, k_draft, k_target=None, use_draft=False, vd=None, eos=None):
        self.m, self.g, self.kd, self.kt, self.use_draft, self.vd = model, gamma, k_draft, k_target, use_draft, vd
        self.eos = frozenset(int(e) for e in eos) if eos else None
        self.tc, self.dc = model.new_cache(T_max), model.new_cache(T_max)
        dev = model.device
        self.n = torch.zeros((), dtype=torch.long, device=dev)       # committed length (pos of cur)
        self.buf = torch.zeros(gamma + 1, dtype=torch.long, device=dev)  # [cur, d_1..d_gamma]
        self.cbuf = torch.zeros(2, dtype=torch.long, device=dev)    # [d_gamma_prev, cur] for catch-up
        self.vout = torch.zeros(gamma + 1, dtype=torch.long, device=dev)
        self.n.fill_(1)  # warm-up/capture with valid positions (catch-up step writes position n-1)
        self.gd = _Graph(lambda: self._draft(False))
        self.gdc = _Graph(lambda: self._draft(True))
        self.gv = _Graph(self._verify)

    def _draft(self, catch_up):
        m = self.m
        for j in range(self.g):
            if j == 0 and catch_up:
                logits = m.forward(self.cbuf, self.n - 1, self.dc, self.kd, self.use_draft, last_only=True, vd=self.vd)
            else:
                logits = m.forward(self.buf[j:j + 1], self.n + j, self.dc, self.kd, self.use_draft, last_only=True,
                                   vd=self.vd)
            self.buf[j + 1:j + 2].copy_(m.to_token_ids(logits.argmax(-1), self.vd))

    def _verify(self):
        logits = self.m.forward(self.buf, self.n, self.tc, self.kt, False)
        self.vout.copy_(logits.argmax(-1))

    @torch.no_grad()
    def generate(self, prompt, n_new):
        m, g = self.m, self.g
        _sync()
        t0 = time.perf_counter()
        logits = m.prefill(prompt, self.tc, self.kt, False)
        m.prefill(prompt, self.dc, self.kd, self.use_draft)
        first = int(logits.argmax(-1))
        _sync()
        t_prefill = time.perf_counter() - t0
        out = [first]
        n, cur, catch_up, prev_last = prompt.numel(), first, False, 0
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(n_new + 1)]
        accepted, rounds = [], 0
        stop = self.eos is not None and first in self.eos
        _sync()
        t1 = time.perf_counter()
        while len(out) < n_new and not stop:
            self.n.fill_(n)
            self.buf[0].fill_(cur)
            if catch_up:
                self.cbuf[0].fill_(prev_last)
                self.cbuf[1].fill_(cur)
            e0, e1, e2 = ev[rounds]
            e0.record()
            (self.gdc if catch_up else self.gd).replay()
            e1.record()
            self.gv.replay()
            e2.record()
            d = self.buf[1:].tolist()  # syncs
            v = self.vout.tolist()
            a = 0
            while a < g and d[a] == v[a]:
                a += 1
            new = d[:a] + [v[a]]
            if self.eos is not None:
                cut = next((i for i, t in enumerate(new) if t in self.eos), None)
                room = n_new - len(out)
                if cut is not None and cut < room:
                    new, stop = new[:cut + 1], True
                elif len(new) > room:
                    new = new[:room]
            out.extend(new)
            accepted.append(a)
            catch_up, prev_last = a == g, d[g - 1]
            n += a + 1
            cur = v[a]
            rounds += 1
        _sync()
        t_dec = time.perf_counter() - t1
        t_draft = sum(ev[i][0].elapsed_time(ev[i][1]) for i in range(rounds)) / 1e3
        t_ver = sum(ev[i][1].elapsed_time(ev[i][2]) for i in range(rounds)) / 1e3
        n_tok = len(out) - 1  # tokens confirmed during decode (the first comes from prefill)
        rounds = max(rounds, 1)
        extra = {} if self.eos is None else {"eos_hit": out[-1] in self.eos}
        return out[:n_new], {**extra,
            "t_prefill_s": t_prefill, "t_decode_s": t_dec, "t_draft_s": t_draft, "t_verify_s": t_ver,
            "t_overhead_s": t_dec - t_draft - t_ver, "rounds": rounds, "n_decode_tokens": n_tok,
            "tokens_per_s": n_tok / t_dec, "tau": n_tok / rounds, "accept_rate": sum(accepted) / (g * rounds),
            "t_draft_round_ms": 1e3 * t_draft / rounds, "t_verify_ms": 1e3 * t_ver / rounds,
            "t_overhead_ms": 1e3 * (t_dec - t_draft - t_ver) / rounds}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))
