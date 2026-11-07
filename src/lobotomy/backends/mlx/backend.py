"""MLX backend (Apple Silicon)."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from ...config import ModelConfig
from ...edits import SliceOp, is_gain_norm
from ..base import Backend, RecordSpec, Recording
from .model import AttnCache, DeltaCache, Qwen35, Recorder

_DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}

# Tensors kept in float32 regardless of the compute dtype.
_KEEP_F32 = ("A_log", "dt_bias")


def _apply_ops(base: mx.array, ops: list[SliceOp]) -> mx.array:
    cur = base.astype(mx.float32)
    for op in ops:
        if op.axis is None:
            mask = None
        else:
            n = cur.shape[op.axis]
            m = np.zeros(n, np.float32)
            for a, b in op.indices:
                m[a:b] = 1.0
            shape = [1] * cur.ndim
            shape[op.axis] = n
            mask = mx.array(m).reshape(shape)

        if op.op == "scale":
            if mask is None:
                cur = cur * op.value
            else:
                cur = cur * (1.0 + mask * (op.value - 1.0))
        elif op.op == "noise":
            if mask is None:
                rms = mx.sqrt(mx.mean(mx.square(cur)))
                cur = cur + mx.random.normal(cur.shape, key=mx.random.key(op.seed)) * (op.value * rms)
            else:
                count = mx.sum(mask) * (cur.size / mask.size)
                rms = mx.sqrt(mx.sum(mx.square(cur) * mask) / count)
                noise = mx.random.normal(cur.shape, key=mx.random.key(op.seed))
                cur = cur + noise * mask * (op.value * rms)
        else:
            raise ValueError(f"unknown op {op.op!r}")
    return cur


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, dtype: str = "bfloat16"):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        self.dtype = _DTYPES[dtype]
        self.model: Qwen35 | None = None
        self._pristine: dict[str, mx.array] = {}

    # --- loading ---------------------------------------------------------------

    def load(self, weights: dict[str, np.ndarray], config: ModelConfig) -> None:
        self.config = config
        w = {}
        for name in list(weights):
            arr = weights.pop(name)  # free numpy memory as we go
            if arr.ndim >= 2:
                w[name] = mx.array(arr).astype(self.dtype)
            elif name.endswith(_KEEP_F32):
                w[name] = mx.array(arr.astype(np.float32))
            else:
                w[name] = mx.array(arr.astype(np.float32)).astype(self.dtype)
            mx.eval(w[name])
        self.model = Qwen35(config, w)
        self._pristine.clear()

    def save_cache(self, path) -> None:
        """Save the pristine weights, so the next start skips dequantizing the GGUF."""
        w = dict(self.model.w)
        w.update(self._pristine)  # never cache edited values
        mx.save_safetensors(str(path), w)

    def load_cache(self, path, config: ModelConfig) -> None:
        self.config = config
        w = mx.load(str(path))
        mx.eval(list(w.values()))
        self.model = Qwen35(config, w)
        self._pristine.clear()

    # --- inference -------------------------------------------------------------

    def new_cache(self):
        return self.model.new_cache()

    def cache_length(self, cache) -> int:
        return cache[0].offset

    def snapshot_cache(self, cache):
        """An independent copy of the cache (KV for attention, state for DeltaNet)."""
        snap = []
        for c in cache:
            if isinstance(c, AttnCache):
                if c.keys is None:
                    snap.append(("attn", None, None, 0))
                else:
                    k = mx.contiguous(c.keys[..., : c.offset, :] * 1)
                    v = mx.contiguous(c.values[..., : c.offset, :] * 1)
                    snap.append(("attn", k, v, c.offset))
            else:
                snap.append(("delta", c.conv, c.ssm, c.offset))  # replaced, never mutated in place
        mx.eval([x for s in snap for x in s[1:3] if x is not None])
        return snap

    def restore_cache(self, snap):
        cache = []
        for kind, a, b, off in snap:
            c = AttnCache() if kind == "attn" else DeltaCache()
            if kind == "attn":
                c.keys, c.values, c.offset = a, b, off
            else:
                c.conv, c.ssm, c.offset = a, b, off
            cache.append(c)
        return cache

    def forward(self, token_ids, cache, record: RecordSpec | None = None):
        offset = self.cache_length(cache)
        ids = mx.array([token_ids], dtype=mx.uint32)
        T = len(token_ids)
        rec = None
        if record is not None:
            rec = Recorder(self.config.n_layers, start=max(0, record.start - offset), attention=record.attention)
        h = self.model.hidden(ids, cache, rec)
        logits = self.model.logits(h[:, -1, :]).astype(mx.float32)[0]
        to_eval = [logits]
        for c in cache:
            to_eval.extend(c.state())
        if rec is not None:
            to_eval.extend(rec.arrays())
        mx.eval(to_eval)
        recording = None
        if rec is not None and rec.start < T:
            recording = self._to_recording(rec, offset, T)
        return np.array(logits, copy=False), recording

    def _to_recording(self, rec: Recorder, offset: int, T: int) -> Recording:
        cfg = self.config
        s = rec.start
        H = cfg.max_heads
        # Stack on the device first: a few large copies beat ~150 small ones.
        pad = lambda a: a if a.shape[-1] == H else mx.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, H - a.shape[-1])], constant_values=mx.nan)
        # float16 tops out at 65504; a boosted neuron can exceed it, so clip
        # rather than let an inf reach the JSON API.
        f16 = lambda a: mx.clip(a.astype(mx.float32), -65000.0, 65000.0).astype(mx.float16)
        dev = {
            "mlp": f16(mx.stack([a[0] for a in rec.mlp], axis=1)),  # (Tr, L, N)
            "heads": mx.stack([pad(a[0]) for a in rec.heads], axis=1),
            "scalars": mx.stack(
                [mx.stack([a[0] for a in grp], axis=1) for grp in (rec.mixer_out, rec.mlp_out, rec.resid)]
            ),  # (3, Tr, L)
            "resid_vec": f16(mx.stack([a[0] for a in rec.resid_vec], axis=1)),
        }
        mx.eval(list(dev.values()))
        scalars = np.array(dev["scalars"])
        return Recording(
            positions=np.arange(offset + s, offset + T),
            mlp=np.array(dev["mlp"]),
            heads=np.array(dev["heads"]),
            mixer_out=scalars[0],
            mlp_out=scalars[1],
            resid=scalars[2],
            resid_vec=np.array(dev["resid_vec"]),
            attention={i: np.array(a) for i, a in rec.attn.items()},
        )

    def unembed_topk(self, vectors, k=5, apply_final_norm=True):
        v = mx.array(np.asarray(vectors, dtype=np.float32)).astype(self.dtype)
        if apply_final_norm:
            v = self.model._rms(v, "model.norm.weight")
        lg = self.model.logits(v).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        idx = mx.argpartition(-lg, k - 1, axis=-1)[:, :k]
        vals = mx.take_along_axis(lp, idx, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx = mx.take_along_axis(idx, order, axis=-1)
        vals = mx.take_along_axis(vals, order, axis=-1)
        mx.eval(idx, vals)
        return np.array(idx), np.exp(np.array(vals))

    def lens_probs(self, vectors, targets, chunk: int = 128):
        """Logit lens for many residual vectors at once: the probability each
        vector (through the final norm and unembedding) gives its target token.
        vectors: (n, hidden); targets: (n,). Chunked so the (n, vocab) logits
        never all live at once."""
        v_all = np.asarray(vectors, dtype=np.float32)
        t_all = np.asarray(targets, dtype=np.int64)
        out = np.empty(len(t_all), np.float32)
        for a in range(0, len(t_all), chunk):
            v = mx.array(v_all[a : a + chunk]).astype(self.dtype)
            lg = self.model.logits(self.model._rms(v, "model.norm.weight")).astype(mx.float32)
            tgt = mx.array(t_all[a : a + chunk])[:, None]
            lp = mx.take_along_axis(lg, tgt, axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1)
            mx.eval(lp)
            out[a : a + chunk] = np.exp(np.array(lp))
        return out

    def damage_report(self, token_ids, start, healthy):
        pos = slice(start - 1, len(token_ids) - 1)
        target = mx.array(token_ids[start:])[:, None]
        lp_cur = self.logprobs_for(token_ids, pos)
        mx.eval(lp_cur)
        with healthy:
            lp_h = self.logprobs_for(token_ids, pos)
            mx.eval(lp_h)
        kl = mx.sum(mx.exp(lp_h) * (lp_h - lp_cur), axis=-1)
        out = {
            "kl": kl,
            "logp_current": mx.take_along_axis(lp_cur, target, axis=-1)[:, 0],
            "logp_healthy": mx.take_along_axis(lp_h, target, axis=-1)[:, 0],
            "top_current": mx.argmax(lp_cur, axis=-1),
            "top_healthy": mx.argmax(lp_h, axis=-1),
        }
        mx.eval(list(out.values()))
        return {k: np.array(v) for k, v in out.items()}

    def logits_for(self, token_ids, cache=None):
        cache = cache if cache is not None else self.new_cache()
        ids = mx.array([token_ids], dtype=mx.uint32)
        h = self.model.hidden(ids, cache, None)
        out = self.model.logits(h[0]).astype(mx.float32)
        mx.eval(out)
        return np.array(out)

    def logprobs_for(self, token_ids, positions: slice, cache=None) -> mx.array:
        """Log-probabilities (on device) at ``positions``, for metrics."""
        cache = cache if cache is not None else self.new_cache()
        ids = mx.array([token_ids], dtype=mx.uint32)
        h = self.model.hidden(ids, cache, None)[0, positions]
        lg = self.model.logits(h).astype(mx.float32)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def attribute(self, token_ids, target_pos):
        """Gradient of log p(token_ids[target_pos] | prefix) with respect to a
        multiplicative mask on every MLP neuron (L, N) and head (L, H_max), at 1.

        A mask scales the unit everywhere, like a weight edit does, so -grad is
        the first-order estimate of how much log p would drop if it were cut.
        """
        cfg = self.config
        m = self.model
        ids = mx.array([token_ids[:target_pos]], dtype=mx.uint32)
        target = int(token_ids[target_pos])

        def logp(m_mlp, m_heads):
            m._masks = {"mlp": m_mlp, "heads": m_heads}
            try:
                h = m.hidden(ids, m.new_cache(), None)
            finally:
                m._masks = None
            lg = m.logits(h[0, -1]).astype(mx.float32)
            return lg[target] - mx.logsumexp(lg)

        ones_mlp = mx.ones((cfg.n_layers, cfg.intermediate_size), mx.float32)
        ones_heads = mx.ones((cfg.n_layers, cfg.max_heads), mx.float32)
        value, (g_mlp, g_heads) = mx.value_and_grad(logp, argnums=(0, 1))(ones_mlp, ones_heads)
        mx.eval(value, g_mlp, g_heads)
        return float(value.item()), np.array(g_mlp), np.array(g_heads)

    def score_variants(self, token_ids, start, variants, current_ops):
        """For each variant (a {tensor: ops} override), the KL from the current
        model on the tokens ``start..`` of ``token_ids``. Weights are restored to
        ``current_ops`` after each variant."""
        pos = slice(start - 1, len(token_ids) - 1)
        base = self.logprobs_for(token_ids, pos)
        base_top = mx.argmax(base, axis=-1)
        p_base = mx.exp(base)
        mx.eval(base, base_top, p_base)
        out = []
        for v in variants:
            try:
                for t, ops in v.items():
                    self.set_tensor_ops(t, ops)
                lp = self.logprobs_for(token_ids, pos)
                kl = mx.sum(p_base * (base - lp), axis=-1)
                agree = mx.mean((mx.argmax(lp, axis=-1) == base_top).astype(mx.float32))
                mx.eval(kl, agree)
                out.append({"kl": np.array(kl), "mean_kl": float(kl.mean().item()), "top1_agree": float(agree.item())})
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
        return np.array(self.model.w[tensor].astype(mx.float32))

    def neuron_wiring(self, layer, neuron, k):
        P = lambda name: self._pristine.get(name, self.model.w[name])  # noqa: E731
        d = P(f"model.layers.{layer}.mlp.down_proj.weight")[:, neuron].astype(mx.float32)
        scores = []
        for later in range(layer + 1, self.config.n_layers):
            p = f"model.layers.{later}."
            gain = P(p + "post_attention_layernorm.weight").astype(mx.float32) + 1.0  # stored as w, gain 1 + w
            up = P(p + "mlp.up_proj.weight")
            scores.append((up @ (gain * d).astype(up.dtype)).astype(mx.float32))
        if not scores:
            return []
        S = np.array(mx.stack(scores))
        out = []
        for j, s in enumerate(S):
            kk = min(k, s.shape[0])
            idx = np.argpartition(-np.abs(s), kk - 1)[:kk]
            idx = idx[np.argsort(-np.abs(s[idx]), kind="stable")]
            out.append((layer + 1 + j, idx, s[idx]))
        return out

    def read_pristine(self, tensor):
        return np.array(self._pristine.get(tensor, self.model.w[tensor]).astype(mx.float32))

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
        cur = _apply_ops(base + 1.0 if gain else base, ops)
        if gain:
            cur = cur - 1.0
        w[tensor] = cur.astype(base.dtype)
        mx.eval(w[tensor])

    def edited_tensors(self) -> list[str]:
        return sorted(self._pristine)
