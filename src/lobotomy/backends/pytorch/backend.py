"""PyTorch backend: NVIDIA GPUs through CUDA, or the CPU when there's no GPU.

It mirrors the MLX backend method for method (``backends/mlx/backend.py``), so
everything above it (engine, edits, server, UI) works unchanged. Picks CUDA
when available and the CPU otherwise; ``LOBOTOMY_DEVICE`` (or ``device=``)
overrides that, e.g. ``mps`` on a Mac or ``cuda:1``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ...config import ModelConfig
from ...edits import SliceOp, is_gain_norm
from ..base import Backend, RecordSpec, Recording
from .model import AttnCache, DeltaCache, Qwen35, Recorder

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Tensors kept in float32 regardless of the compute dtype.
_KEEP_F32 = ("A_log", "dt_bias")


def default_device() -> str:
    if os.environ.get("LOBOTOMY_DEVICE"):
        return os.environ["LOBOTOMY_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_dtype(device: torch.device) -> str:
    """bfloat16 on GPUs that do it natively (NVIDIA Ampere and later), float32
    elsewhere: older GPUs emulate bfloat16 slowly, float16 can overflow in
    this model, and CPUs run float32 fastest."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def _apply_ops(base: torch.Tensor, ops: list[SliceOp]) -> torch.Tensor:
    cur = base.float()
    for op in ops:
        if op.axis is None:
            mask = None
        else:
            n = cur.shape[op.axis]
            m = torch.zeros(n, dtype=torch.float32, device=cur.device)
            for a, b in op.indices:
                m[a:b] = 1.0
            shape = [1] * cur.ndim
            shape[op.axis] = n
            mask = m.reshape(shape)

        if op.op == "scale":
            cur = cur * op.value if mask is None else cur * (1.0 + mask * (op.value - 1.0))
        elif op.op == "noise":
            # seeded on the CPU, so a recipe's noise is the same on every device
            gen = torch.Generator().manual_seed(int(op.seed))
            noise = torch.randn(cur.shape, generator=gen, dtype=torch.float32).to(cur.device)
            if mask is None:
                rms = cur.pow(2).mean().sqrt()
                cur = cur + noise * (op.value * rms)
            else:
                count = mask.sum() * (cur.numel() / mask.numel())
                rms = ((cur.pow(2) * mask).sum() / count).sqrt()
                cur = cur + noise * mask * (op.value * rms)
        else:
            raise ValueError(f"unknown op {op.op!r}")
    return cur


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy() if t.dtype == torch.bfloat16 else t.detach().cpu().numpy()


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, dtype: str | None = None, device: str | None = None):
        self.device = torch.device(device or default_device())
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch can't see a CUDA GPU. Install the CUDA build of PyTorch "
                "(the installer does: uv pip install torch --torch-backend=auto), or run on the CPU "
                "with --backend torch (slower)."
            )
        self.dtype_name = dtype or default_dtype(self.device)
        if self.dtype_name not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        self.dtype = _DTYPES[self.dtype_name]
        self.model: Qwen35 | None = None
        self._pristine: dict[str, torch.Tensor] = {}

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"CUDA · {torch.cuda.get_device_name(self.device)}"
        return self.device.type.upper()

    # --- loading ---------------------------------------------------------------

    def _to_device(self, name: str, arr) -> torch.Tensor:
        t = arr if isinstance(arr, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(arr))
        if name.endswith(_KEEP_F32):
            return t.to(device=self.device, dtype=torch.float32)
        return t.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def load(self, weights: dict[str, np.ndarray], config: ModelConfig) -> None:
        self.config = config
        w = {}
        for name in list(weights):
            w[name] = self._to_device(name, weights.pop(name))  # free numpy memory as we go
        self.model = Qwen35(config, w)
        self._pristine.clear()

    def save_cache(self, path) -> None:
        """Save the pristine weights as safetensors (the MLX backend reads the same file)."""
        from safetensors.torch import save_file

        w = dict(self.model.w)
        w.update(self._pristine)  # never cache edited values
        save_file({k: v.detach().contiguous().cpu() for k, v in w.items()}, str(path))

    @torch.no_grad()
    def load_cache(self, path, config: ModelConfig) -> None:
        from safetensors.torch import load_file

        self.config = config
        w = {k: self._to_device(k, v) for k, v in load_file(str(path)).items()}
        self.model = Qwen35(config, w)
        self._pristine.clear()

    # --- inference -------------------------------------------------------------

    def new_cache(self):
        return self.model.new_cache()

    def cache_length(self, cache) -> int:
        return cache[0].offset

    def snapshot_cache(self, cache):
        snap = []
        for c in cache:
            if isinstance(c, AttnCache):
                if c.keys is None:
                    snap.append(("attn", None, None, 0))
                else:
                    snap.append(("attn", c.keys[..., : c.offset, :].clone(), c.values[..., : c.offset, :].clone(), c.offset))
            else:
                snap.append(("delta", c.conv, c.ssm, c.offset))  # replaced, never mutated in place
        return snap

    def restore_cache(self, snap):
        cache = []
        for kind, a, b, off in snap:
            if kind == "attn":
                c = AttnCache()
                # copies: the cache grows in place, the snapshot must stay reusable
                c.keys = None if a is None else a.clone()
                c.values = None if b is None else b.clone()
                c.offset = off
            else:
                c = DeltaCache()
                c.conv, c.ssm, c.offset = a, b, off
            cache.append(c)
        return cache

    def _ids(self, token_ids) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def forward(self, token_ids, cache, record: RecordSpec | None = None):
        offset = self.cache_length(cache)
        T = len(token_ids)
        rec = None
        if record is not None:
            rec = Recorder(self.config.n_layers, start=max(0, record.start - offset), attention=record.attention)
        h = self.model.hidden(self._ids(token_ids), cache, rec)
        logits = self.model.logits(h[:, -1, :]).float()[0]
        recording = None
        if rec is not None and rec.start < T:
            recording = self._to_recording(rec, offset, T)
        return logits.cpu().numpy(), recording

    def _to_recording(self, rec: Recorder, offset: int, T: int) -> Recording:
        H = self.config.max_heads
        s = rec.start

        def pad(a):
            if a.shape[-1] == H:
                return a
            return torch.nn.functional.pad(a, (0, H - a.shape[-1]), value=float("nan"))

        # float16 tops out at 65504; a boosted neuron can exceed it, so clip
        # rather than let an inf reach the JSON API.
        f16 = lambda a: a.float().clamp(-65000.0, 65000.0).half()  # noqa: E731
        scalars = torch.stack(
            [torch.stack([a[0] for a in grp], dim=1) for grp in (rec.mixer_out, rec.mlp_out, rec.resid)]
        ).cpu().numpy()  # (3, Tr, L)
        return Recording(
            positions=np.arange(offset + s, offset + T),
            mlp=f16(torch.stack([a[0] for a in rec.mlp], dim=1)).cpu().numpy(),
            heads=torch.stack([pad(a[0]) for a in rec.heads], dim=1).float().cpu().numpy(),
            mixer_out=scalars[0],
            mlp_out=scalars[1],
            resid=scalars[2],
            resid_vec=f16(torch.stack([a[0] for a in rec.resid_vec], dim=1)).cpu().numpy(),
            attention={i: a.cpu().numpy() for i, a in rec.attn.items()},
        )

    @torch.no_grad()
    def unembed_topk(self, vectors, k=5, apply_final_norm=True):
        v = torch.as_tensor(np.asarray(vectors, dtype=np.float32), device=self.device).to(self.dtype)
        if apply_final_norm:
            v = self.model._rms(v, "model.norm.weight")
        lp = self.model.logits(v).float().log_softmax(-1)
        vals, idx = lp.topk(k, dim=-1)
        return idx.cpu().numpy(), vals.exp().cpu().numpy()

    @torch.no_grad()
    def lens_probs(self, vectors, targets, chunk: int = 128):
        v_all = np.asarray(vectors, dtype=np.float32)
        t_all = np.asarray(targets, dtype=np.int64)
        out = np.empty(len(t_all), np.float32)
        for a in range(0, len(t_all), chunk):
            v = torch.as_tensor(v_all[a : a + chunk], device=self.device).to(self.dtype)
            lg = self.model.logits(self.model._rms(v, "model.norm.weight")).float()
            tgt = torch.as_tensor(t_all[a : a + chunk], device=self.device)[:, None]
            lp = lg.gather(-1, tgt)[:, 0] - lg.logsumexp(-1)
            out[a : a + chunk] = lp.exp().cpu().numpy()
        return out

    @torch.no_grad()
    def logits_for(self, token_ids, cache=None):
        cache = cache if cache is not None else self.new_cache()
        h = self.model.hidden(self._ids(token_ids), cache, None)
        return self.model.logits(h[0]).float().cpu().numpy()

    @torch.no_grad()
    def logprobs_for(self, token_ids, positions: slice, cache=None) -> torch.Tensor:
        """Log-probabilities (on device) at ``positions``, for metrics."""
        cache = cache if cache is not None else self.new_cache()
        h = self.model.hidden(self._ids(token_ids), cache, None)[0, positions]
        return self.model.logits(h).float().log_softmax(-1)

    @torch.no_grad()
    def damage_report(self, token_ids, start, healthy):
        pos = slice(start - 1, len(token_ids) - 1)
        target = torch.as_tensor(token_ids[start:], device=self.device)[:, None]
        lp_cur = self.logprobs_for(token_ids, pos)
        with healthy:
            lp_h = self.logprobs_for(token_ids, pos)
        kl = (lp_h.exp() * (lp_h - lp_cur)).sum(-1)
        out = {
            "kl": kl,
            "logp_current": lp_cur.gather(-1, target)[:, 0],
            "logp_healthy": lp_h.gather(-1, target)[:, 0],
            "top_current": lp_cur.argmax(-1),
            "top_healthy": lp_h.argmax(-1),
        }
        return {k: v.cpu().numpy() for k, v in out.items()}

    def attribute(self, token_ids, target_pos):
        """Gradient of log p(token_ids[target_pos] | prefix) with respect to a
        multiplicative mask on every MLP neuron (L, N) and head (L, H_max), at 1."""
        cfg = self.config
        m = self.model
        ids = self._ids(token_ids[:target_pos])
        target = int(token_ids[target_pos])
        m_mlp = torch.ones((cfg.n_layers, cfg.intermediate_size), dtype=torch.float32, device=self.device, requires_grad=True)
        m_heads = torch.ones((cfg.n_layers, cfg.max_heads), dtype=torch.float32, device=self.device, requires_grad=True)
        with torch.enable_grad():
            m._masks = {"mlp": m_mlp, "heads": m_heads}
            try:
                h = m.hidden(ids, m.new_cache(), None)
            finally:
                m._masks = None
            lg = m.logits(h[0, -1]).float()
            logp = lg[target] - lg.logsumexp(-1)
            g_mlp, g_heads = torch.autograd.grad(logp, (m_mlp, m_heads))
        return float(logp.item()), g_mlp.cpu().numpy(), g_heads.cpu().numpy()

    @torch.no_grad()
    def score_variants(self, token_ids, start, variants, current_ops):
        pos = slice(start - 1, len(token_ids) - 1)
        base = self.logprobs_for(token_ids, pos)
        base_top = base.argmax(-1)
        p_base = base.exp()
        out = []
        for v in variants:
            try:
                for t, ops in v.items():
                    self.set_tensor_ops(t, ops)
                lp = self.logprobs_for(token_ids, pos)
                kl = (p_base * (base - lp)).sum(-1)
                agree = (lp.argmax(-1) == base_top).float().mean()
                out.append({"kl": kl.cpu().numpy(), "mean_kl": float(kl.mean().item()), "top1_agree": float(agree.item())})
            finally:
                for t in v:
                    self.set_tensor_ops(t, current_ops.get(t, []))
        return out

    # --- weights -----------------------------------------------------------------

    def tensor_names(self):
        return sorted(self.model.w)

    def tensor_shape(self, tensor):
        return tuple(self.model.w[tensor].shape)

    def read_tensor(self, tensor):
        return self.model.w[tensor].detach().float().cpu().numpy()

    def read_pristine(self, tensor):
        return self._pristine.get(tensor, self.model.w[tensor]).detach().float().cpu().numpy()

    @torch.no_grad()
    def neuron_wiring(self, layer, neuron, k):
        P = lambda name: self._pristine.get(name, self.model.w[name])  # noqa: E731
        d = P(f"model.layers.{layer}.mlp.down_proj.weight")[:, neuron].float()
        scores = []
        for later in range(layer + 1, self.config.n_layers):
            p = f"model.layers.{later}."
            gain = P(p + "post_attention_layernorm.weight").float() + 1.0  # stored as w, gain 1 + w
            up = P(p + "mlp.up_proj.weight")
            scores.append((up @ (gain * d).to(up.dtype)).float())
        if not scores:
            return []
        S = torch.stack(scores).cpu().numpy()
        out = []
        for j, s in enumerate(S):
            kk = min(k, s.shape[0])
            idx = np.argpartition(-np.abs(s), kk - 1)[:kk]
            idx = idx[np.argsort(-np.abs(s[idx]), kind="stable")]
            out.append((layer + 1 + j, idx, s[idx]))
        return out

    @torch.no_grad()
    def set_tensor_ops(self, tensor: str, ops: list[SliceOp]) -> None:
        w = self.model.w
        if tensor not in w:
            raise KeyError(f"no tensor named {tensor!r}")
        self.model.invalidate(tensor)
        if not ops:
            if tensor in self._pristine:
                w[tensor] = self._pristine.pop(tensor)
            return
        base = self._pristine.setdefault(tensor, w[tensor])
        gain = is_gain_norm(tensor)
        cur = _apply_ops(base.float() + 1.0 if gain else base, ops)
        if gain:
            cur = cur - 1.0
        w[tensor] = cur.to(base.dtype)

    def edited_tensors(self) -> list[str]:
        return sorted(self._pristine)
