"""E7 MagicDec repro: parse the stdout of one OFFICIAL MagicDec benchmark run (run_official.sh) into a result record.

Official scripts (tests/baseline_benchmark.py, tests/{SnapKV,StreamingLLM}/selfspec_benchmark.py) run 10 batches
(DataLoader, shuffle=False); counters are reset after batches 0-4 (warm-up), so the line printed after batch i>=5 is
cumulative over batches 5..i and the last one is what the script reports. We keep the official cumulative numbers
verbatim and also difference them into per-batch values (spread over the 5 measured batches).
  SD: Omega = 'avg generate len per sentence' = generated tokens / (verify steps * B)  (prefill excluded, timer starts
      after encode); tok/s = generated tokens / wall of the decode loop (draft + verify + accept/rollback + host).
  AR: tok/s = B * steps / wall (prefill excluded).
Usage: python parse_official.py <tag> [--save]   (reads /data/lsh/logs/magicdec/<tag>.{out,co.json})
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OD = Path("/data/lsh/logs/magicdec")
NUM = r"(?:tensor\()?([0-9.eE+-]+)"
SD1 = re.compile(r"total time :([0-9.]+)s, time per iter :([0-9.]+)s, decoding step: " + NUM + r".*large model step: (\d+)")
SD2 = re.compile(r"target time :([0-9.]+)s, draft time :([0-9.]+)s, verify loop : ([0-9.eE+-]+), avg generate len per sentence: " + NUM)
AR = re.compile(r"^Tokens per second :([0-9.]+)")
FIN = re.compile(r"Final tokens per second :" + NUM)


def arg(args, k, cast=str, default=None):
    t = args.split()
    return cast(t[t.index(k) + 1]) if k in t else default


def parse(tag):
    co = json.load(open(OD / f"{tag}.co.json"))
    lines = (OD / f"{tag}.out").read_text(errors="replace").splitlines()
    args = co["args"]
    B = arg(args, "--B", int)
    fin = [float(m.group(1)) for m in map(FIN.search, lines) if m]
    is_sd = "selfspec" in co["script"]
    rec = {"tag": tag, "script": co["script"], "args": args, "rc": co["rc"], "wall_s": co["wall_s"],
           "final_tok_per_s": fin[-1] if fin else None, "B": B,
           "prefix_len": arg(args, "--prefix_len", int), "max_len": arg(args, "--max_len", int),
           "gamma": arg(args, "--gamma", int), "draft_budget": arg(args, "--draft_budget", int),
           "cotenant_before": co["before"]["cotenant"], "cotenant_after": co["after"]["cotenant"],
           "co_before": co["before"], "co_after": co["after"]}
    if is_sd:
        a = [m.groups() for m in map(SD1.search, lines) if m]
        b = [m.groups() for m in map(SD2.search, lines) if m]
        assert len(a) == len(b) == 10, (len(a), len(b))
        cum = [dict(total_s=float(x[0]), gen=float(x[2]), steps=int(x[3]), target_s=float(y[0]) * int(x[3]),
                    draft_s=float(y[1]) * int(x[3]), loop_s=float(y[2]) * int(x[3]), omega=float(y[3]))
               for x, y in zip(a, b)]
        per = [cum[5]] + [{k: cum[i][k] - cum[i - 1][k] for k in ("total_s", "gen", "steps", "target_s", "draft_s",
                                                                    "loop_s")} for i in range(6, 10)]
        for p in per:
            p["omega"] = p["gen"] / p["steps"] / B
            p["tok_per_s"] = p["gen"] / p["total_s"]
            p["round_ms"] = 1e3 * p["total_s"] / p["steps"]
        c = cum[9]
        rec.update(omega=c["omega"], round_ms=1e3 * c["total_s"] / c["steps"], verify_ms=1e3 * c["target_s"] / c["steps"],
                   draft_ms=1e3 * c["draft_s"] / c["steps"], accept_loop_ms=1e3 * c["loop_s"] / c["steps"],
                   gen_tokens=c["gen"], verify_steps=c["steps"], gen_per_seq=c["gen"] / B / 5,
                   tok_per_s_cum=c["gen"] / c["total_s"], warmup_batches=[dict(zip(("total_s", "gen", "steps"), (float(x[0]), float(x[2]), int(x[3])))) | {"omega": float(y[3])} for x, y in zip(a[:5], b[:5])],
                   per_batch=per,
                   tok_per_s_batch=dict(median=statistics.median(p["tok_per_s"] for p in per),
                                        min=min(p["tok_per_s"] for p in per), max=max(p["tok_per_s"] for p in per)),
                   omega_batch=dict(median=statistics.median(p["omega"] for p in per),
                                    min=min(p["omega"] for p in per), max=max(p["omega"] for p in per)))
    else:
        t = [float(m.group(1)) for m in map(AR.search, lines) if m]
        assert len(t) == 10, len(t)
        rec.update(tok_per_s_cum_lines=t, step_ms=1e3 * B / fin[-1],
                   gen_per_seq=rec["max_len"] - rec["prefix_len"] - 1)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    r = parse(a.tag)
    print(json.dumps({k: v for k, v in r.items() if k not in ("co_before", "co_after", "per_batch", "warmup_batches")}, indent=1))
    if a.save:
        from tools.results import save_result
        kind = "AR" if r["gamma"] is None else ("SnapKV" if "SnapKV" in r["script"] else "StreamingLLM")
        cfg = {"model": "Llama-3.1-8B-Instruct (bf16, /data/lsh/ckpts/magicdec/Llama-3.1-8B-Instruct/model.pth)",
               "setting": f"official MagicDec self-spec {kind}, PG-19 (official loader), TP=4 RTX 4090 GPUs 4-7",
               "code": "github.com/Infini-AI-Lab/MagicDec @ 09cd671c01bc87f7570d3243c46200a96f0fc0b4 (unmodified)",
               "env": "/data/lsh/envs/magicdec: torch 2.4.1+cu121, flashinfer 0.1.6+cu121torch2.4, transformers 4.36.2",
               "kind": kind, "B": r["B"], "T": r["prefix_len"], "gamma": r["gamma"], "draft_budget": r["draft_budget"],
               "args": r["args"], "script": r["script"]}
        print(save_result("E7", f"llama8b_magicdec_official_{a.tag}", cfg, r,
                          model_dir="/data/models/llama3.1_8b_instruct", notes=a.notes))
