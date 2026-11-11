"""Residual quantization (RQ) baseline on top of specrq/dbd/main.py, without editing specrq/dbd.

Run exactly like specrq/dbd/main.py (same CLI, same cwd = specrq/dbd); this wrapper replaces AQEngine.quantize:

  SPECRQ_RQ=act  (calibration-aware RQ): stages are trained one at a time, s = 1..L. Stage s
    * init: residual k-means of the normalized weight residual W/scale - sum_{j<s} C_j[codes_j]  (= the AQ init step
      for one codebook, same k-means settings: --init_max_iter, --init_max_points_per_centroid)
    * then AQ epochs restricted to stage s: Adam (fresh per stage; lr, betas (0, .95), amsgrad as AQ) on codebook s
      only (other codebooks' gradients masked to 0) and beam search over codes of stage s only (the beam search is
      run on the residual reference W - scale * prefix_{s-1}, which gives the same objective as the full prefix-s
      error), loss = tr(dW_s XX^T dW_s^T) of the prefix s (= codebook_weights one-hot at s), same
      --steps_per_epoch / --max_epochs / --relative_mse_tolerance early stop, applied per stage.
    * per-row scales (the AQ format has one shared scale per output row) are trained in stage 1 only and frozen
      afterwards, so earlier stages are never changed once trained (no joint refinement).
  SPECRQ_RQ=wmse (weight-MSE RQ): only the AQ residual k-means init of all stages on the normalized weight (no
    activation statistics, no Adam, no beam search) = sequential k-means on residuals.
  unset: unchanged AQ (T / N), i.e. plain main.py.
Output format, eval and checkpoints are unchanged (QuantizedWeight with L codebooks).

SPECRQ_OFFLOAD_FILE_DIR=<dir> (any mode, use with --offload_activations): the large pinned host buffers of main.py
  (layer inputs/outputs, >= 256 MiB) are allocated as file-backed shared mmaps in <dir> (torch.from_file) instead of
  pinned RAM. Same dtype and values; only the storage moves (pageable page cache, reclaimable; H2D copies are not
  pinned). Files are deleted at exit.
Usage: cd specrq/dbd && SPECRQ_RQ=act python ../../experiments/16_rq/rq_main.py <main.py args>
"""
import os
import random
import runpy
import sys
import time

DBD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "specrq", "dbd"))
sys.path.insert(0, DBD)

import torch  # noqa: E402

import aq_engine  # noqa: E402
from src.aq import QuantizedWeight, init_aq_kmeans  # noqa: E402
from src.beam_search_xtx import beam_search_optimal_codes  # noqa: E402
from src.hadamard import make_signs, pick_block_size, rotate_weight, rotate_xtx  # noqa: E402

MODE = os.environ.get("SPECRQ_RQ", "")
OFFLOAD_DIR = os.environ.get("SPECRQ_OFFLOAD_FILE_DIR", "")


def _install_file_offload(root):
    import atexit, itertools, shutil, tempfile
    d = tempfile.mkdtemp(prefix=f"offload_{os.getpid()}_", dir=root)
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    cnt = itertools.count()
    orig_zeros, orig_zeros_like = torch.zeros, torch.zeros_like
    min_bytes = 256 * 2**20
    filebacked = set()  # data_ptrs of file-backed buffers (main.py allocates outs = zeros_like(inps, pin_memory=inps.is_pinned()))

    def _file_tensor(shape, dtype):
        numel = 1
        for x in shape:
            numel *= int(x)
        path = os.path.join(d, f"buf{next(cnt)}.bin")
        t = torch.from_file(path, shared=True, size=numel, dtype=dtype).view(*shape)
        t.zero_()
        filebacked.add(t.data_ptr())
        print(f"[specrq-offload] file-backed buffer {tuple(shape)} {dtype} -> {path}", flush=True)
        return t

    def zeros(*size, dtype=None, device=None, pin_memory=False, **kw):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else size
        dt = dtype or torch.get_default_dtype()
        n = 1
        for x in shape:
            n *= int(x)
        if pin_memory and (device is None or str(device) == "cpu") and not kw and n * torch.empty((), dtype=dt).element_size() >= min_bytes:
            return _file_tensor(shape, dt)
        return orig_zeros(*size, dtype=dtype, device=device, pin_memory=pin_memory, **kw)

    def zeros_like(t, *a, pin_memory=False, **kw):
        if (pin_memory or t.data_ptr() in filebacked) and not a and not kw and t.device.type == "cpu" \
                and t.numel() * t.element_size() >= min_bytes:
            return _file_tensor(tuple(t.shape), t.dtype)
        return orig_zeros_like(t, *a, pin_memory=pin_memory, **kw)

    torch.zeros, torch.zeros_like = zeros, zeros_like
    print(f"[specrq-offload] large pinned host buffers -> file-backed mmaps in {d}", flush=True)
_orig_quantize = aq_engine.AQEngine.quantize


def _setup(self, args):
    """the part of AQEngine.quantize before the training loop (Hadamard + QuantizedWeight with residual k-means init)"""
    if self.hadamard_seed is not None:
        block_size = pick_block_size(self.columns)
        if block_size > 1:
            signs = make_signs(self.columns, self.hadamard_seed, device=self.device)
            self.rotation_signs, self.hadamard_block_size = signs, block_size
            self.rotated_weight = rotate_weight(self.layer.weight.detach().float(), signs, block_size)
            self.XTX.copy_(rotate_xtx(self.XTX, signs, block_size))
    assert len(args.devices) == 1, "RQ wrapper: single device only"
    assert getattr(args, "codebook_bits", None) is None, "RQ wrapper: uniform codebook size only"
    self.quantized_weight = QuantizedWeight(
        reference_weight=self._reference_weight().to(device=self.device, dtype=torch.float32),
        out_group_size=args.out_group_size, in_group_size=args.in_group_size, num_codebooks=args.num_codebooks,
        nbits_per_codebook=args.nbits_per_codebook, codebook_value_nbits=args.codebook_value_nbits,
        codebook_value_num_groups=args.codebook_value_num_groups, scale_nbits=args.scale_nbits,
        max_iter=args.init_max_iter, max_points_per_centroid=args.init_max_points_per_centroid,
        devices=args.devices, verbose=True,
    )


@torch.no_grad()
def _reinit_stage(self, args, s):
    """k-means init of codebook s on the normalized residual of the trained prefix (same k-means as the AQ init)"""
    qw = self.quantized_weight
    ref = self._reference_weight().to(device=self.device, dtype=torch.float32)
    prefix = qw(num_codebooks=s)  # [out, in], includes scales
    og, ig = qw.out_group_size, qw.in_group_size
    scales = qw.get_scales()  # [num_out_groups, 1 or num_in_groups, 1, 1]
    res = (ref - prefix).reshape(qw.out_features // og, og, qw.in_features // ig, ig).swapaxes(1, 2) / scales
    res = res.swapaxes(1, 2).reshape_as(ref)
    codes_s, codebook_s = init_aq_kmeans(res, num_codebooks=1, out_group_size=og, in_group_size=ig,
                                         codebook_size=qw.codebook_size, max_iter=args.init_max_iter,
                                         max_points_per_centroid=args.init_max_points_per_centroid,
                                         devices=args.devices, verbose=False)
    qw.codebooks.data[s].copy_(codebook_s[0])
    qw.codes.data[..., s] = codes_s[..., 0].to(qw.codes.dtype)


@torch.no_grad()
def _beam_stage(self, args, s, seed):
    qw = self.quantized_weight
    dtype = qw.codebooks.dtype
    ref = self._reference_weight().to(dtype)
    residual_ref = ref - (qw(num_codebooks=s) if s > 0 else torch.zeros_like(ref))
    new = beam_search_optimal_codes(
        XTX=self.XTX.to(dtype), reference_weight=residual_ref, codebooks=qw.get_codebooks()[s:s + 1],
        prev_codes=qw.get_codes()[..., s:s + 1].contiguous(), scales=qw.get_scales(), beam_size=args.beam_size,
        dim_rng=random.Random(seed), verbose=True)
    qw.codes.data[..., s] = new[..., 0].to(qw.codes.dtype)


@torch.enable_grad()
def rq_quantize(self, *, args, verbose: bool = True):
    if MODE not in ("act", "wmse"):
        return _orig_quantize(self, args=args, verbose=verbose)
    _setup(self, args)
    qw = self.quantized_weight
    L = args.num_codebooks
    if MODE == "wmse":
        print(f"[specrq-rq] wmse: residual k-means init only (L={L})")
        return qw
    for s in range(L):
        t0 = time.perf_counter()
        if s > 0:
            _reinit_stage(self, args, s)
        self.codebook_weights = tuple(1.0 if j == s else 0.0 for j in range(L))
        params = [qw.codebooks]
        mask = torch.zeros_like(qw.codebooks)
        mask[s] = 1
        hook = qw.codebooks.register_hook(lambda g, m=mask: g * m)
        if s == 0 and qw.scales is not None and qw.scales.requires_grad:
            params.append(qw.scales)
        elif qw.scales is not None:
            qw.scales.requires_grad_(False)
        opt = torch.optim.Adam(params, lr=args.lr, betas=(0.0, 0.95), amsgrad=True)
        prev_best, n_ep = float("inf"), 0
        for epoch in range(args.max_epochs):
            n_ep = epoch + 1
            stop = False
            for step in range(args.steps_per_epoch):
                loss = self._compute_mse()
                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / prev_best > (1.0 - args.relative_mse_tolerance):
                        stop = True
                        break
                    prev_best = min(prev_best, loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                    print(f"rq_stage={s}\tepoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")
            if stop:
                break
            _beam_stage(self, args, s, random.getrandbits(256))
        hook.remove()
        print(f"[specrq-rq] stage {s} done: epochs={n_ep} loss={prev_best:.10f} t={time.perf_counter() - t0:.1f}s")
    if qw.scales is not None:
        qw.scales.requires_grad_(True)  # restore the flag (finetune, if any, would train them as in AQ)
    return qw


aq_engine.AQEngine.quantize = rq_quantize

if __name__ == "__main__":
    print(f"[specrq-rq] mode={MODE or 'off (plain AQ)'}")
    if OFFLOAD_DIR:
        _install_file_offload(OFFLOAD_DIR)
    sys.argv = [os.path.join(DBD, "main.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(DBD, "main.py"), run_name="__main__")
