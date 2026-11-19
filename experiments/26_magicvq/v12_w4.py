"""E7 magicvq v12: tau of self-spec drafts whose WEIGHTS come from another source (QuantSpec-style INT4) in addition to
the compressed KV. Target unchanged (bf16 weights, bf16 KV). No edit of mv_common.py / lb_tau.py / tau_eval.py.

Draft names get an optional weight-source suffix '@<src>' (e.g. snapkv4096@w4, vqpt2wL2+r128@w4, exact@w4); without a
suffix the draft uses the target's bf16 weights (bit-identical to the existing runs). A weight source is a safetensors
file of dense weights keyed by the HF linear names (model.layers.<l>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj})
- generic: any later quantized checkpoint (e.g. a weight-VQ nested prefix) can be dumped to the same layout. Norms,
embeddings and lm_head stay the target's. Only the draft's own forward uses the source: the context KV the draft reads is
still the target's (bf16-weight) KV in the draft's KV format; the draft's in-round K/V come from the source weights.

Mechanism: tau_eval.tf_pass is rebuilt from its own source with exactly two line edits (draft qkv / post use
_draft_layer(l, layer, spec)), so the bf16 path is the same code. _draft_layer returns the target layer, or one shared
GPU "shadow" decoder layer loaded with layer l's source weights (cast to bf16; ~0.44 GB) and the target's norms;
the source file is mmapped (CPU, page cache), one H2D copy per (layer, sample) and source.

Usage:
  prep (once, GPU): python experiments/26_magicvq/v12_w4.py --prep_int4 MARLIN_CKPT OUT.safetensors
  run : V12_WSRC="w4=OUT.safetensors" V12_TAGS="e2enkl_w10" python experiments/26_magicvq/v12_w4.py lb_tau.py <args>
        (or tau_eval.py <args>); draft names may carry @w4. A source path ':target' feeds the target's own weights
        through the shadow layer (hook check: such drafts must equal the plain ones bit for bit).
"""
import copy
import inspect
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
LIN = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
       "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}


def prep_int4(marlin_ckpt, out):
    """Dequantize the Marlin INT4 checkpoint (E4) to dense fp16 with the same kernel-based routine as
    experiments/13_int4_marlin/eval_quality.py (int4dense): W = (q - 8) * s, bitwise deterministic."""
    sys.path.insert(0, os.path.join(REPO, "experiments", "13_int4_marlin"))
    from eval_quality import dequant_marlin
    from safetensors.torch import save_file
    pk = torch.load(marlin_ckpt, map_location="cpu", mmap=True)
    names = sorted({k[:-2] for k in pk if k.endswith(".B")})
    assert len(names) == 7 * 32, len(names)
    sd = {}
    for n in names:
        sd[n] = dequant_marlin(pk[n + ".B"], pk[n + ".s"], "cuda").cpu()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(sd, out, metadata={"src": marlin_ckpt, "format": "dense fp16 [N, K] = (q - 8) * s (Marlin INT4 g128)"})
    print("wrote", out, len(sd), sum(t.numel() for t in sd.values()) / 1e9, "G params", flush=True)


class WeightSource:
    def __init__(self, path):
        from safetensors import safe_open
        self.path = path
        self.f = safe_open(path, framework="pt", device="cpu")
        self.keys = set(self.f.keys())

    def get(self, name):
        return self.f.get_tensor(name)


SOURCES = {}
_SHADOW = {}  # src -> [shadow layer, loaded layer index]


def _draft_layer(l, layer, spec):
    src = spec.get("wsrc")
    if src is None:
        return layer
    if src not in _SHADOW:
        _SHADOW[src] = [copy.deepcopy(layer), None]
    sh = _SHADOW[src]
    if sh[1] != l:
        lw = sh[0]
        with torch.no_grad():
            lw.input_layernorm.weight.copy_(layer.input_layernorm.weight)
            lw.post_attention_layernorm.weight.copy_(layer.post_attention_layernorm.weight)
            for lin, sub in LIN.items():
                dst = getattr(getattr(lw, sub), lin).weight
                if SOURCES[src] == "target":  # hook check: the target's own weights through the shadow layer
                    w = getattr(getattr(layer, sub), lin).weight
                else:
                    w = SOURCES[src].get(f"model.layers.{l}.{sub}.{lin}")
                assert w.shape == dst.shape, (l, lin, w.shape, dst.shape)
                dst.copy_(w.to(dst.device).to(dst.dtype))
                b = getattr(getattr(layer, sub), lin).bias
                assert b is None, "bias not handled"
        sh[1] = l
    return sh[0]


def patch():
    import mv_common
    for t in os.environ.get("V12_TAGS", "").split():  # extra 'p'-format codebook tags (as v10_tau.py)
        p = os.path.join(mv_common.OUTROOT, "cb", f"cb_d8K256x2_post_cn_{t}.pt")
        assert os.path.exists(p), p
        mv_common.CB["p" + t] = p
        mv_common.POST.add("p" + t)
    for item in os.environ.get("V12_WSRC", "").split(","):
        if item:
            k, _, p = item.partition("=")
            SOURCES[k] = "target" if p == ":target" else WeightSource(p)
    base = mv_common.draft_specs

    def draft_specs(names):
        out = {}
        for n in names:
            b, at, src = n.partition("@")
            s = dict(base([b])[b])
            if at:
                assert src in SOURCES, (n, list(SOURCES))
                s["wsrc"] = src
            out[n] = s
        return out
    mv_common.draft_specs = draft_specs  # before lb_tau / tau_eval do 'from mv_common import draft_specs'
    import tau_eval as TE
    src = inspect.getsource(TE.tf_pass)
    edits = [("              qs, ks, vs = f.qkv(layer, h.view(G * N, -1))\n",
              "              lw = _draft_layer(l, layer, s)\n              qs, ks, vs = f.qkv(lw, h.view(G * N, -1))\n"),
             ("              hd[n] = f.post(layer, h.reshape(G * N, -1), out.view(G * N, -1)).view(G, N, -1)\n",
              "              hd[n] = f.post(lw, h.reshape(G * N, -1), out.view(G * N, -1)).view(G, N, -1)\n")]
    for a, b in edits:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    TE.__dict__["_draft_layer"] = _draft_layer
    exec(compile(src, TE.__file__ + ":tf_pass(v12)", "exec"), TE.__dict__)  # redefines TE.tf_pass in TE's globals
    return TE


def main():
    if sys.argv[1] == "--prep_int4":
        return prep_int4(sys.argv[2], sys.argv[3])
    TE = patch()
    script = sys.argv[1]
    sys.argv = [os.path.join(HERE, script)] + sys.argv[2:]
    if script == "tau_eval.py":
        TE.main()
    elif script == "lb_tau.py":
        import lb_tau
        assert lb_tau.TE is TE
        lb_tau.main()
    else:
        raise ValueError(script)


if __name__ == "__main__":
    main()
