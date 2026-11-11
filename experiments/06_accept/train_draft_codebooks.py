"""[specrq] Phase 5B: train draft-only codebook/scale copies so that p_D^(k_d) matches the frozen target p_T^(k_t).

Common tool for all q / d / K configurations: one fixed objective (TV = 1 - speculative acceptance; forward KL only
as a recorded fallback if TV is unstable), one fixed static dataset (pile_val_heldout: Pile-val documents disjoint from
the pile_val calibration windows and from the wikitext2/c4 evaluation sets), same step count.
Teacher = same checkpoint with all k_t codebooks (frozen). Student = shared first-k_d codes (frozen) with draft-only
codebooks/scales (trainable, init = target values or an externally supplied draft, e.g. a rounded q=8/4 copy).
Memory (Qwen-0.5B): one model on GPU, two forwards per step (teacher no_grad, student with grad).

Usage: python train_draft_codebooks.py --model DIR --ckpt DIR --out DIR [--init_draft PT] [--steps 200]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "specrq", "dbd"))
import torch
from src.datautils import get_loaders
from src.draft import attach_draft_codebooks, divergence, draft_parameters, draft_state_dict
from src.modelutils import get_model
from src.aq import QuantizedLinear


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


def set_k(model, k):
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            m.active_num_codebooks = k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kd", type=int, default=3)
    ap.add_argument("--kt", type=int, default=5)
    ap.add_argument("--div", default="tv", choices=["tv", "fkl"], help="tv (default); fkl only as a recorded fallback")
    ap.add_argument("--init_draft", default=None, help="draft_codebooks.pt to start from (e.g. rounded q=8/4 copy)")
    ap.add_argument("--data_name", default="pile_val_heldout")
    ap.add_argument("--qat_fmt", default=None, choices=[None, "int8", "int4", "fp8_e4m3"],
                    help="D(ii): keep draft codebooks q-bit during training via STE fake-quant; rounded at save")
    ap.add_argument("--qat_gran", default="codebook", choices=["codebook", "dim"])
    ap.add_argument("--qat_refresh", type=int, default=10, help="re-search fake-quant scales every N steps")
    ap.add_argument("--nsamples", type=int, default=256)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_train_scales", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codes_uint8", action="store_true", help="store 8-bit codes as uint8 (memory; outputs unchanged)")
    ap.add_argument("--micro_batch", type=int, default=0, help="split each batch into micro-batches of this size "
                    "(gradient accumulation, same update); 0 = no split")
    ap.add_argument("--checkpoint_linears", action="store_true",
                    help="recompute each QuantizedLinear forward in backward (QuantizedLinear.use_checkpoint) so dequantized "
                         "weights are not kept for all layers (Llama-8B OOM otherwise); gradients unchanged")
    ap.add_argument("--checkpoint_layers", action="store_true",
                    help="[specrq] recompute each decoder layer in backward (torch.utils.checkpoint, non-reentrant) so "
                         "only layer inputs are kept (Qwen-32B); gradients unchanged")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:0")
    model = get_model(a.model, a.ckpt, "auto")
    if a.codes_uint8:
        print("codes -> uint8 for", shrink_codes_(model), "linears")
    model = model.to(dev)
    if a.checkpoint_linears:
        for m in model.modules():
            if isinstance(m, QuantizedLinear):
                m.use_checkpoint = True
    if a.checkpoint_layers:
        import functools
        from torch.utils.checkpoint import checkpoint
        for layer in model.model.layers:
            f = layer.forward
            layer.forward = functools.partial(
                lambda f, *x, **kw: checkpoint(f, *x, use_reentrant=False, **kw) if torch.is_grad_enabled() else f(*x, **kw), f)
        model.config.use_cache = False  # a KV cache would be appended again on recompute
        print("checkpointing", len(model.model.layers), "decoder layers")
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = attach_draft_codebooks(model, a.kd, train_scales=not a.no_train_scales)
    if a.init_draft:
        from src.draft import load_draft_state_dict
        n_loaded = load_draft_state_dict(model, torch.load(a.init_draft), trainable=True, train_scales=not a.no_train_scales)
        print("initialized draft copies from", a.init_draft, n_loaded, "linears")
    if a.qat_fmt:
        from src.draft import attach_fakequant, bake_fakequant, refresh_fakequant
        print("STE fake-quant on", attach_fakequant(model, a.qat_fmt, a.qat_gran), "linears:", a.qat_fmt, a.qat_gran)
    params = list(draft_parameters(model))
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.95))
    # data shuffle seed fixed to 0 (= calibration seed) so pile_val_heldout stays disjoint from the pile_val calibration
    data = get_loaders(a.data_name, nsamples=a.nsamples, seed=0, seqlen=a.seqlen, model_path=a.model)
    data = torch.cat(list(data), 0)  # [N, T]
    log, t0 = [], time.time()
    model.eval()  # no dropout; grads still flow into draft params
    for step in range(a.steps):
        if a.qat_fmt and step % a.qat_refresh == 0:
            refresh_fakequant(model)
        idx = torch.randint(0, data.shape[0], (a.batch,))
        ids = data[idx].to(dev)
        if a.micro_batch and a.micro_batch < a.batch:
            # [specrq] same sampled batch, split into equal micro-batches; grads accumulated with weight 1/n_micro so the
            # update equals the full-batch mean-loss gradient (memory: Llama-8B 128k-vocab logits in fp32)
            assert a.batch % a.micro_batch == 0
            n_micro = a.batch // a.micro_batch
            opt.zero_grad(set_to_none=True)
            loss_sum = tv_sum = top1_sum = 0.0
            for mb in ids.split(a.micro_batch):
                set_k(model, a.kt)
                with torch.no_grad():
                    lt = model(mb).logits[:, :-1]
                set_k(model, a.kd)
                ld = model(mb).logits[:, :-1]
                loss = divergence(lt.reshape(-1, lt.shape[-1]), ld.reshape(-1, ld.shape[-1]), a.div).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}; rerun with --div fkl and record it")
                (loss / n_micro).backward()
                with torch.no_grad():
                    tv_sum += divergence(lt.reshape(-1, lt.shape[-1]), ld.detach().reshape(-1, ld.shape[-1]), "tv").mean().item()
                    top1_sum += (lt.argmax(-1) == ld.argmax(-1)).float().mean().item()
                loss_sum += loss.item()
                del lt, ld, loss
            opt.step()
            log.append({"step": step, "loss": loss_sum / n_micro, "tv": tv_sum / n_micro, "top1": top1_sum / n_micro,
                        "t": time.time() - t0})
            if step % 10 == 0 or step == a.steps - 1:
                print(json.dumps(log[-1]), flush=True)
            continue
        set_k(model, a.kt)
        with torch.no_grad():
            lt = model(ids).logits[:, :-1]
        set_k(model, a.kd)
        ld = model(ids).logits[:, :-1]
        start = 0
        loss = divergence(lt[:, start:].reshape(-1, lt.shape[-1]), ld[:, start:].reshape(-1, ld.shape[-1]), a.div).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}; rerun with --div fkl and record it")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            tv = divergence(lt[:, start:].reshape(-1, lt.shape[-1]), ld[:, start:].reshape(-1, ld.shape[-1]).detach(), "tv").mean().item()
            top1 = (lt[:, start:].argmax(-1) == ld[:, start:].argmax(-1)).float().mean().item()
        log.append({"step": step, "loss": loss.item(), "tv": tv, "top1": top1, "t": time.time() - t0})
        if step % 10 == 0 or step == a.steps - 1:
            print(json.dumps(log[-1]), flush=True)
    if a.qat_fmt:
        refresh_fakequant(model)
        bake_fakequant(model)  # stored values = q-bit grid values (dequantized to float)
    torch.save(draft_state_dict(model), os.path.join(a.out, "draft_codebooks.pt"))
    json.dump({"args": vars(a), "n_trainable": n_train, "log": log,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
