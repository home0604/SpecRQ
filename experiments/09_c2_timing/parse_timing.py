"""Per-layer / per-stage timing and losses from a DbD main.py log.

Works on timestamp-prefixed logs (run_c2_timing.sh: '<epoch_s> <line>') and on plain logs (then only the
in-process numbers are available: aq_epoch_start t=, layer_time, out_loss, out_val_loss).
Stages per layer (timestamped logs only):
  pre   : layer header -> first 'Quantizing module' (calc outs before quantization + XTX accumulation)
  init  : 'Quantizing module' -> aq_epoch_start 0 (k-means init), summed over modules
  aq    : aq_epoch_start 0 -> next module / 'PREPARING TO FINETUNE' (Adam steps + beam search), summed over modules
  ft    : 'PREPARING TO FINETUNE' -> 'FINISHED FINETUNING'
  post  : 'FINISHED FINETUNING' -> stats dict (save + calc outs after quantization, train + val)
Per module: n_epochs (aq_epoch_start count; the last epoch is the early-stop check) and mean epoch time from t=.
Usage: python parse_timing.py LOG [--json OUT]
"""
import ast, json, re, sys

TS = re.compile(r"^(\d{9,}\.\d+) ")


def parse(path):
    text = open(path, errors="replace").read().replace("\r", "\n")
    layers, cur, mod, last_ts = [], None, None, None
    for raw in text.split("\n"):
        m = TS.match(raw)
        ts = float(m.group(1)) if m else None
        line = raw[m.end():] if m else raw
        if ts is not None:
            last_ts = ts
        t = ts if ts is not None else last_ts
        if "---------------- Layer" in line:
            cur = {"layer": int(line.split("Layer")[1].split()[0]), "t_start": t, "modules": []}
            layers.append(cur)
            mod = None
        elif cur is None:
            continue
        elif line.startswith("Quantizing module"):
            mod = {"name": line.split()[2], "t_q": t, "epoch_t": []}
            cur["modules"].append(mod)
        elif "[specrq] aq_epoch_start" in line and mod is not None:
            e = int(re.search(r"epoch=(\d+)", line).group(1))
            mod["epoch_t"].append(float(re.search(r"t=([0-9.]+)s", line).group(1)))
            if e == 0:
                mod["t_e0"] = t
        elif line.startswith("hadamard ") and mod is not None:
            mod["t_end"] = t
        elif line.startswith("PREPARING TO FINETUNE"):
            cur["t_ft0"] = t
        elif line.startswith("FINISHED FINETUNING"):
            cur["t_ft1"] = t
        elif line.startswith("{") and "layer_time" in line:
            d = ast.literal_eval(line.strip())
            cur.update({k: d[k] for k in ("layer_time", "out_loss", "out_val_loss") if k in d})
            cur["t_end"] = t
    out = []
    for L in layers:
        r = {"layer": L["layer"], "layer_time_s": L.get("layer_time"), "out_loss": L.get("out_loss"),
             "out_val_loss": L.get("out_val_loss"), "modules": []}
        for M in L["modules"]:
            et = M["epoch_t"]
            r["modules"].append({"name": M["name"], "n_epoch_starts": len(et),
                                 "aq_s_inproc": et[-1] if et else None,
                                 "epoch_s_mean": (et[-1] / (len(et) - 1)) if len(et) > 1 else None})
        if L["t_start"] is not None and L["modules"] and L["modules"][0].get("t_q") is not None and TS:
            mods = L["modules"]
            try:
                r["pre_s"] = mods[0]["t_q"] - L["t_start"]
                r["init_s"] = sum(M["t_e0"] - M["t_q"] for M in mods)
                ends = [mods[i + 1]["t_q"] for i in range(len(mods) - 1)] + [L.get("t_ft0")]
                r["aq_s"] = sum(e - M["t_e0"] for M, e in zip(mods, ends))
                r["ft_s"] = L["t_ft1"] - L["t_ft0"]
                r["post_s"] = L["t_end"] - L["t_ft1"]
            except (KeyError, TypeError):
                pass
        out.append(r)
    return out


if __name__ == "__main__":
    res = parse(sys.argv[1])
    print(json.dumps(res, indent=1))
    if "--json" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
