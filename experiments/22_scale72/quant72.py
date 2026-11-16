"""E5 scale72: layer-streaming AQ quantization driver for models whose bf16 weights do not fit in host RAM
(Qwen2.5-72B-Instruct: 145 GB bf16 vs 150 GiB shared cgroup).

Same recipe and same checkpoint format as specrq/dbd/main.py (as used for qwen32b_d8K256L2_noft), but
  * decoder layers are built one at a time from the safetensors shards (model skeleton on the meta device; only
    embed_tokens / rotary / norm / lm_head are materialised),
  * train activations (inps / outs, nsamples x seqlen x hidden bf16, 34 GB each at 72B) live in two flat files on
    /data/lsh and are streamed per sample with explicit pread/pwrite (no mmap -> host RSS stays small),
  * a resumable state file records which file holds the current layer inputs.
AQ itself is DbD's AQEngine.quantize unchanged (multi-GPU = --devices cuda:0 cuda:1 shards output groups).
The val split (val_size samples) is kept in RAM and also saved in the state dir for resume.

Usage (all main.py flags are accepted; the parser is main.py's own, reached via main.main()):
  python experiments/22_scale72/quant72.py MODEL wikitext2 --nsamples 512 --val_size 32 --model_seqlen 4096 \
     --num_codebooks 2 --nbits_per_codebook 8 --in_group_size 8 --out_group_size 1 --codebook_weights 0.5 0.5 \
     --hadamard --accum_dtype float32 --relative_mse_tolerance 0.01 --max_epochs 1000 --finetune_max_epochs 0 \
     --offload_activations --devices cuda:0 [cuda:1] --resume --save CKPT [--quantize_num_layers N]
  env ACT_DIR=/data/lsh/temp/act72/<tag> (activation files + state), default derived from --save.
Writes CKPT/{i}.pth (torch.save of the quantized Qwen2DecoderLayer, as main.py), CKPT/args.pt and
CKPT/not_quantized_weights.pt after the last layer, and CKPT/layer_stats.jsonl (time, out_loss, per-module time).
"""
import json
import os
import sys
import threading
import time
from queue import Queue

HERE = os.path.dirname(os.path.abspath(__file__))
DBD = os.path.abspath(os.path.join(HERE, "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)
os.chdir(DBD)  # main.py relies on relative paths (data caches) like run_scale.sh's `cd specrq/dbd`

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

import main as M  # noqa: E402
from aq_engine import AQEngine  # noqa: E402
from src.aq import QuantizedLinear  # noqa: E402
from src.datautils import get_loaders  # noqa: E402
from src.hadamard import layer_seed  # noqa: E402
from src.modelutils import find_sublayers, suspend_nn_inits  # noqa: E402


# --------------------------------------------------------------------------------------------- weights from shards
class ShardReader:
    def __init__(self, model_path):
        ip = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(ip):
            idx = json.load(open(ip))["weight_map"]
        else:  # single-shard model (e.g. the 0.5B used for the pipeline smoke test)
            with safe_open(os.path.join(model_path, "model.safetensors"), framework="pt") as sf:
                idx = {k: "model.safetensors" for k in sf.keys()}
        self.path, self.wmap = model_path, idx

    def get(self, names, device="cpu"):
        by_file = {}
        for n in names:
            by_file.setdefault(self.wmap[n], []).append(n)
        out = {}
        for f, ns in by_file.items():
            with safe_open(os.path.join(self.path, f), framework="pt", device=str(device)) as sf:
                for n in ns:
                    out[n] = sf.get_tensor(n)
        return out

    def layer_state(self, i, device="cpu"):
        pre = f"model.layers.{i}."
        sd = self.get([n for n in self.wmap if n.startswith(pre)], device)
        return {k[len(pre):]: v for k, v in sd.items()}


def build_layer(model, reader, i, device):
    """Real Qwen2DecoderLayer i on `device` (bf16), weights from the shards; buffers (rotary inv_freq) as in HF init."""
    cls = type(model.model.layers[i])
    with suspend_nn_inits(), torch.device("cpu"):
        layer = cls(model.config, layer_idx=i)
    layer = layer.to(dtype=model.config.torch_dtype)
    # keep inv_freq in fp32 like a freshly loaded HF model (from_pretrained keeps non-persistent buffers fp32)
    for mod in layer.modules():
        if hasattr(mod, "inv_freq") and hasattr(mod, "original_inv_freq"):
            mod.inv_freq = mod.original_inv_freq.clone()
    missing, unexpected = layer.load_state_dict(reader.layer_state(i), strict=False)
    assert not unexpected and all("rotary" in m for m in missing), (missing, unexpected)
    return layer.to(device).eval()


def skeleton(model_path, reader):
    cfg = AutoConfig.from_pretrained(model_path)
    with init_empty_weights(include_buffers=False), suspend_nn_inits():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=cfg.torch_dtype)
    model.config.torch_dtype = cfg.torch_dtype
    names = ["model.embed_tokens.weight", "model.norm.weight"] + (["lm_head.weight"] if "lm_head.weight" in reader.wmap else [])
    nq = reader.get(names)
    model.model.embed_tokens.weight = nn.Parameter(nq["model.embed_tokens.weight"], requires_grad=False)
    model.model.norm.weight = nn.Parameter(nq["model.norm.weight"], requires_grad=False)
    if "lm_head.weight" in nq:
        model.lm_head.weight = nn.Parameter(nq["lm_head.weight"], requires_grad=False)
    else:  # tied (0.5B)
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval(), nq


# --------------------------------------------------------------------------------------------- disk activation store
class DiskActs:
    """n x (seqlen, hidden) bf16 rows in one flat file; explicit pread/pwrite, no mmap."""

    def __init__(self, path, n, seqlen, hidden, dtype=torch.bfloat16, create=False):
        self.path, self.n, self.shape, self.dtype = path, n, (seqlen, hidden), dtype
        self.row_bytes = seqlen * hidden * torch.finfo(dtype).bits // 8
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        self.fd = os.open(path, flags, 0o644)
        if create:
            os.ftruncate(self.fd, self.row_bytes * n)

    def __len__(self):
        return self.n

    def read_into(self, j, buf):
        mv = memoryview(buf.view(torch.uint8).numpy()).cast("B")
        got = os.preadv(self.fd, [mv], j * self.row_bytes)
        assert got == self.row_bytes, (got, self.row_bytes)
        return buf

    def write_from(self, j, buf):
        mv = memoryview(buf.contiguous().view(torch.uint8).numpy()).cast("B")
        put = os.pwritev(self.fd, [mv], j * self.row_bytes)
        assert put == self.row_bytes
        # drop written pages from the page cache (keeps cgroup memory low; data is re-read once per layer pass)
        os.posix_fadvise(self.fd, j * self.row_bytes, self.row_bytes, os.POSIX_FADV_DONTNEED)


def stream(acts_in, layer, device, fargs, acts_out=None, ref=None, n_io=3):
    """For each row j: y = layer(x_j); if ref: relative mse vs ref row j; if acts_out: write y.
    Reader/writer threads overlap disk I/O with the GPU forward. ref and acts_out may be the same store
    (the ref row is read before the write of the same row)."""
    n = len(acts_in)
    shape, dt = acts_in.shape, acts_in.dtype
    pool, opool = Queue(), Queue()
    for _ in range(2 * n_io + 2):
        pool.put(torch.empty(shape, dtype=dt, pin_memory=True))
    for _ in range(n_io + 2 if acts_out is not None else 0):
        opool.put(torch.empty(shape, dtype=dt, pin_memory=True))
    q_in, q_out, losses = Queue(maxsize=n_io), Queue(maxsize=n_io), []

    def reader():
        for j in range(n):
            x = acts_in.read_into(j, pool.get())
            r = ref.read_into(j, pool.get()) if ref is not None else None
            q_in.put((j, x, r))
        q_in.put(None)

    def writer():
        while True:
            it = q_out.get()
            if it is None:
                return
            j, y_cpu, ev = it
            ev.synchronize()
            acts_out.write_from(j, y_cpu)
            opool.put(y_cpu)

    tr = threading.Thread(target=reader, daemon=True)
    tr.start()
    tw = threading.Thread(target=writer, daemon=True) if acts_out is not None else None
    if tw:
        tw.start()
    while True:
        it = q_in.get()
        if it is None:
            break
        j, x, r = it
        y = layer(x.to(device).unsqueeze(0), **fargs)[0].view(shape)
        if r is not None:
            yf = y.float()
            losses.append(((yf - r.to(device).float()).square().mean() / yf.square().mean().clamp(min=1e-6)).item())
        pool.put(x)
        if r is not None:
            pool.put(r)
        if acts_out is not None:
            y_cpu = opool.get()
            y_cpu.copy_(y, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            q_out.put((j, y_cpu, ev))
    if tw:
        q_out.put(None)
        tw.join()
    tr.join()
    return losses


class _XTXWrap(nn.Module):
    def __init__(self, lin, eng):
        super().__init__()
        self.wrapped_layer, self.eng = lin, eng

    def forward(self, x, *a, **k):
        self.eng.add_batch(x)
        return self.wrapped_layer(x, *a, **k)


# --------------------------------------------------------------------------------------------- quantization loop
@torch.no_grad()
def catch_inputs(model, reader, data, store, dev):
    """Layer-0 inputs + forward kwargs via DbD's Catcher idea, with only embed/rotary/layer0 materialised."""
    emb = model.model.embed_tokens.to(dev)
    model.model.rotary_emb.to(dev)
    real0 = build_layer(model, reader, 0, dev)
    cache = {}

    class Exit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, inp, **kw):
            cache["x"] = inp
            cache["kw"] = {k: kw.get(k) for k in ("attention_mask", "position_ids")}
            raise Exit()

    layers = model.model.layers
    saved0 = layers[0]
    layers[0] = Catcher(real0)
    buf = torch.empty(store.shape, dtype=store.dtype, pin_memory=True)
    for j, ids in enumerate(data):
        if isinstance(ids, (list, tuple)):
            ids = ids[0]
        ids = ids.to(dev)
        try:
            model(ids, attention_mask=torch.ones_like(ids))
        except Exit:
            pass
        buf.copy_(cache["x"].view(store.shape))
        store.write_from(j, buf)
    layers[0] = saved0
    emb.to("cpu")
    del real0
    torch.cuda.empty_cache()
    return cache["kw"]


@torch.no_grad()  # as main.quantize_aq; AQEngine.quantize re-enables grad itself
def quantize72(model_unused, args):
    t_all = time.time()
    dev = args.devices[0]
    reader = ShardReader(args.model_path)
    model, nq = skeleton(args.model_path, reader)
    cfg = model.config
    L, H, S = cfg.num_hidden_layers, cfg.hidden_size, args.model_seqlen
    nql = min(args.quantize_num_layers if args.quantize_num_layers is not None else L, L)
    act_dir = os.environ.get("ACT_DIR") or os.path.join("/data/lsh/temp/act72", os.path.basename(args.save.rstrip("/")))
    os.makedirs(act_dir, exist_ok=True)
    os.makedirs(args.save, exist_ok=True)
    state_p = os.path.join(act_dir, "state.json")
    stats_p = os.path.join(args.save, "layer_stats.jsonl")
    n_train = args.nsamples - args.val_size

    if os.path.exists(state_p):
        st = json.load(open(state_p))
        assert st["n_train"] == n_train and st["seqlen"] == S and st["hidden"] == H
        A = DiskActs(os.path.join(act_dir, st["inps"]), n_train, S, H)
        B = DiskActs(os.path.join(act_dir, st["outs"]), n_train, S, H)
        val = torch.load(os.path.join(act_dir, "val.pt"))
        val_inps, fargs = val["inps"], val["fargs"]
        start = st["next_layer"]
        print(f"[q72] resume: next_layer={start} inps={st['inps']}", flush=True)
    else:
        print("Loading data ...", flush=True)
        data = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model_path=args.model_path,
                           seqlen=S, use_fast_tokenizer=args.use_fast_tokenizer, trust_remote_code=args.trust_remote_code)
        all_ids = torch.randperm(len(data))
        train_data = [data[i] for i in all_ids[args.val_size:]]
        val_data = [data[i] for i in all_ids[:args.val_size]]
        A = DiskActs(os.path.join(act_dir, "acts_a.bin"), n_train, S, H, create=True)
        B = DiskActs(os.path.join(act_dir, "acts_b.bin"), n_train, S, H, create=True)
        t0 = time.time()
        fargs = catch_inputs(model, reader, train_data, A, dev)
        # val inputs in RAM (as main.py: pinned CPU tensor)
        vstore = DiskActs(os.path.join(act_dir, "val_tmp.bin"), len(val_data), S, H, create=True)
        catch_inputs(model, reader, val_data, vstore, dev)
        val_inps = torch.empty((len(val_data), S, H), dtype=torch.bfloat16, pin_memory=True)
        for j in range(len(val_data)):
            vstore.read_into(j, val_inps[j])
        os.remove(vstore.path)
        fargs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in fargs.items()}
        print(f"[q72] caught inputs train={n_train} val={len(val_data)} in {time.time() - t0:.0f}s "
              f"fargs={ {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else v) for k, v in fargs.items()} }", flush=True)
        start = 0
        st = {"n_train": n_train, "seqlen": S, "hidden": H, "inps": "acts_a.bin", "outs": "acts_b.bin", "next_layer": 0}
        torch.save({"inps": val_inps, "fargs": fargs}, os.path.join(act_dir, "val.pt"))
        json.dump(st, open(state_p, "w"))
    fd = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in fargs.items()}
    val_outs = torch.zeros_like(val_inps, pin_memory=True)

    for i in range(start, nql):
        print(f"\n---------------- Layer {i} of {L} ----------------", flush=True)
        t0 = time.time()
        stats = {"layer": i, "mod_time_s": {}}
        layer = build_layer(model, reader, i, dev)
        pth = os.path.join(args.save, f"{i}.pth")
        loaded = args.resume and os.path.exists(pth)
        if loaded:
            print(f"Loading layer {i} from {pth}", flush=True)
            layer = torch.load(pth, map_location=dev, weights_only=False)
        else:
            # val reference outputs of the fp layer (main.py: update_outs before quantization)
            M.update_outs(layer, val_inps, val_outs, compute_mse=False, **fd)
            names = list(find_sublayers(layer).keys())
            engines = {n: AQEngine(find_sublayers(layer)[n]) for n in names}
            wrapped = {e.layer: e for e in engines.values()}
            for mod in list(layer.modules()):
                for cn, ch in list(mod.named_children()):
                    if ch in wrapped:
                        setattr(mod, cn, _XTXWrap(ch, wrapped[ch]))
            t1 = time.time()
            stream(A, layer, dev, fd, acts_out=B)  # fp outputs (targets for out_loss) + XTX
            stats["calc_outs_before_s"] = time.time() - t1
            for mod in list(layer.modules()):
                for cn, ch in list(mod.named_children()):
                    if isinstance(ch, _XTXWrap):
                        setattr(mod, cn, ch.wrapped_layer)
            for n, eng in engines.items():
                print(f"Quantizing module {n} of layer {i}", flush=True)
                t1 = time.time()
                if args.hadamard:
                    eng.hadamard_seed = layer_seed(args.seed, i, n)
                qw = eng.quantize(args=args, verbose=True)
                new = QuantizedLinear(qw, eng.layer.bias)
                if eng.rotation_signs is not None:
                    new.set_rotation(eng.rotation_signs, eng.hadamard_block_size)
                found = False
                for mod in layer.modules():
                    for cn, ch in mod.named_children():
                        if ch is eng.layer:
                            setattr(mod, cn, new)
                            found = True
                assert found, n
                stats["mod_time_s"][n] = time.time() - t1
                print(f"[q72] module {n} done in {stats['mod_time_s'][n]:.1f}s", flush=True)
            del engines, wrapped, eng
            # main.py: layer.to(float32) -> finetune_groupwise (no-op at finetune_max_epochs=0) -> layer.to(orig dtype)
            layer = layer.to(dtype=torch.float32).to(dtype=cfg.torch_dtype)
            for p in layer.parameters():
                p.grad = None
            torch.save(layer, pth + ".tmp")
            os.replace(pth + ".tmp", pth)
        t1 = time.time()
        losses = stream(A, layer, dev, fd, acts_out=B, ref=None if loaded else B)
        stats["calc_outs_after_s"] = time.time() - t1
        if losses:
            stats["out_loss"] = sum(losses) / len(losses)
        vl = M.update_outs(layer, val_inps, val_outs, compute_mse=not loaded, **fd)
        if vl:
            stats["out_val_loss"] = sum(vl) / len(vl)
        del layer
        torch.cuda.empty_cache()
        A, B = B, A
        val_inps, val_outs = val_outs, val_inps
        st.update(inps=os.path.basename(A.path), outs=os.path.basename(B.path), next_layer=i + 1)
        torch.save({"inps": val_inps, "fargs": fargs}, os.path.join(act_dir, "val.pt.tmp"))
        os.replace(os.path.join(act_dir, "val.pt.tmp"), os.path.join(act_dir, "val.pt"))
        json.dump(st, open(state_p + ".tmp", "w"))
        os.replace(state_p + ".tmp", state_p)
        stats["layer_time"] = time.time() - t0
        stats["loaded"] = loaded
        stats["max_cuda_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
        print(stats, flush=True)
        print(f"{{'out_loss': {stats.get('out_loss')}, 'layer_time': {stats['layer_time']}, 'Step': {i}}}", flush=True)
        if not loaded:
            with open(stats_p, "a") as f:
                f.write(json.dumps(stats) + "\n")

    if nql >= L:
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        torch.save(nq, os.path.join(args.save, "not_quantized_weights.pt"))
        print("[q72] saved args.pt and not_quantized_weights.pt", flush=True)
    print(f"quantization time: {time.time() - t_all:.1f}", flush=True)


if __name__ == "__main__":
    # reuse main.py's argument parser and global setup; replace model loading and the quantization loop
    M.get_model = lambda *a, **k: nn.Module()
    M.quantize_model = quantize72
    import argparse

    _parse = argparse.ArgumentParser.parse_args

    def parse_and_force_partial(self, *a, **k):
        ns = _parse(self, *a, **k)
        # main() evaluates PPL after quantize_model unless quantize_num_layers is set; the skeleton cannot be evaluated
        if ns.quantize_num_layers is None:
            ns.quantize_num_layers = 10 ** 6
        return ns

    argparse.ArgumentParser.parse_args = parse_and_force_partial
    M.main()
