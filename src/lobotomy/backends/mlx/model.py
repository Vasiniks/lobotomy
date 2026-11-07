"""Qwen3.5 text model in MLX, written for inspection and live editing.

The math follows mlx-lm's ``qwen3_5.py`` / ``qwen3_next.py`` (MIT, Apple) and
uses its Metal gated-delta kernel. The differences:

- Weights live in a flat dict keyed by Hugging Face tensor names, so an edit is
  a dict assignment and the next forward pass uses it.
- RMSNorm weights use the Hugging Face convention (gain = 1 + w).
- Every layer can record its internals (see ``Recorder``) for a chosen range of
  positions.
- Attention probabilities are computed explicitly for recorded positions; the
  output itself still uses the fused attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk

from ...config import ModelConfig


class AttnCache:
    """Pre-allocated KV cache for one full-attention layer."""

    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k: mx.array, v: mx.array):
        B, H, T, D = k.shape
        prev = self.offset
        if self.keys is None or prev + T > self.keys.shape[2]:
            n = ((prev + T + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, n, D), k.dtype)
            new_v = mx.zeros((B, H, n, D), v.dtype)
            if self.keys is not None and prev > 0:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., prev : prev + T, :] = k
        self.values[..., prev : prev + T, :] = v
        self.offset += T
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def state(self):
        return [] if self.keys is None else [self.keys, self.values]


class DeltaCache:
    """Conv window and recurrent state for one Gated DeltaNet layer."""

    def __init__(self):
        self.conv = None
        self.ssm = None
        self.offset = 0

    def state(self):
        return [a for a in (self.conv, self.ssm) if a is not None]


@dataclass
class Recorder:
    """Collects internals for positions >= ``start`` (relative to this forward call)."""

    n_layers: int
    start: int = 0
    attention: bool = True
    mlp: list = field(default_factory=list)
    heads: list = field(default_factory=list)
    mixer_out: list = field(default_factory=list)
    mlp_out: list = field(default_factory=list)
    resid: list = field(default_factory=list)
    resid_vec: list = field(default_factory=list)
    attn: dict = field(default_factory=dict)

    def __post_init__(self):
        L = self.n_layers
        self.mlp = [None] * L
        self.heads = [None] * L
        self.mixer_out = [None] * L
        self.mlp_out = [None] * L
        self.resid = [None] * L
        self.resid_vec = [None] * L

    def arrays(self) -> list[mx.array]:
        out = []
        for group in (self.mlp, self.heads, self.mixer_out, self.mlp_out, self.resid, self.resid_vec):
            out.extend(a for a in group if a is not None)
        out.extend(self.attn.values())
        return out


def _norm(x: mx.array) -> mx.array:
    return mx.sqrt(mx.sum(mx.square(x.astype(mx.float32)), axis=-1))


def head_gram(w_out: mx.array, n_heads: int) -> mx.array:
    """Per-head Gram matrices G_h = W_hᵀ W_h, (H, D, D) float32.

    W_h is head h's slice of the output projection (hidden, D). A head's
    contribution norm is then ||W_h o_h|| = sqrt(o_hᵀ G_h o_h), which avoids
    touching the full projection on every decode step.
    """
    hidden = w_out.shape[0]
    D = w_out.shape[1] // n_heads
    w = w_out.astype(mx.float32).reshape(hidden, n_heads, D).transpose(1, 0, 2)  # (H, hidden, D)
    return w.swapaxes(1, 2) @ w


def _head_contrib_norms(out_heads: mx.array, gram: mx.array) -> mx.array:
    """L2 norm of each head's contribution to the residual stream.

    out_heads: (B, T, H, D); gram: (H, D, D). Returns (B, T, H) float32.
    """
    o = out_heads.astype(mx.float32)[..., None, :]  # (B, T, H, 1, D)
    q = mx.sum((o @ gram) * o, axis=(-2, -1))
    return mx.sqrt(mx.maximum(q, 0.0))


class Qwen35:
    def __init__(self, config: ModelConfig, weights: dict[str, mx.array]):
        self.cfg = config
        self.w = weights
        self._gram: dict[str, mx.array] = {}
        # Set during attribution: multiplicative masks on MLP neurons (L, N) and
        # heads (L, H_max), and a differentiable path (no fused kernels, no cache).
        self._masks: dict[str, mx.array] | None = None

    def invalidate(self, tensor: str) -> None:
        """Call after changing a weight, so cached derivatives are rebuilt."""
        self._gram.pop(tensor, None)

    def gram(self, tensor: str, n_heads: int) -> mx.array:
        g = self._gram.get(tensor)
        if g is None:
            g = head_gram(self.w[tensor], n_heads)
            mx.eval(g)
            self._gram[tensor] = g
        return g

    # --- helpers -------------------------------------------------------------

    def _lin(self, x: mx.array, name: str) -> mx.array:
        return x @ self.w[name].T

    def _rms(self, x: mx.array, name: str) -> mx.array:
        # Hugging Face convention: gain = 1 + w
        return mx.fast.rms_norm(x, 1.0 + self.w[name], self.cfg.rms_norm_eps)

    def new_cache(self):
        return [AttnCache() if self.cfg.is_full_attention(i) else DeltaCache() for i in range(self.cfg.n_layers)]

    # --- blocks --------------------------------------------------------------

    def _attention(self, i: int, x: mx.array, cache: AttnCache, rec: Recorder | None) -> mx.array:
        cfg = self.cfg
        p = f"model.layers.{i}.self_attn."
        B, T, _ = x.shape
        H, Hkv, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

        qg = self._lin(x, p + "q_proj.weight").reshape(B, T, H, 2 * D)
        q, gate = mx.split(qg, 2, axis=-1)
        k = self._lin(x, p + "k_proj.weight").reshape(B, T, Hkv, D)
        v = self._lin(x, p + "v_proj.weight").reshape(B, T, Hkv, D)

        q = mx.fast.rms_norm(q, 1.0 + self.w[p + "q_norm.weight"], cfg.rms_norm_eps).transpose(0, 2, 1, 3)
        k = mx.fast.rms_norm(k, 1.0 + self.w[p + "k_norm.weight"], cfg.rms_norm_eps).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        diff = self._masks is not None
        off = 0 if diff else cache.offset
        q = mx.fast.rope(q, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=off)
        k = mx.fast.rope(k, cfg.rope_dim, traditional=False, base=cfg.rope_theta, scale=1.0, offset=off)
        keys, values = (k, v) if diff else cache.update_and_fetch(k, v)

        scale = D**-0.5
        if diff:
            # plain ops so autodiff works
            rep = H // Hkv
            sc = (q.reshape(B, Hkv, rep, T, D) * scale) @ keys[:, :, None].swapaxes(-1, -2)
            causal = mx.arange(T)[None, :] <= mx.arange(T)[:, None]
            sc = mx.where(causal, sc.astype(mx.float32), -mx.inf)
            out = (mx.softmax(sc, axis=-1).astype(v.dtype) @ values[:, :, None]).reshape(B, H, T, D)
        else:
            mask = "causal" if T > 1 else None
            out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3)  # (B, T, H, D)
        out = out * mx.sigmoid(gate)
        if diff:
            out = out * self._masks["heads"][i, :H][None, None, :, None].astype(out.dtype)

        if rec is not None and rec.attention and rec.start < T:
            s = rec.start
            K = keys.shape[2]
            rep = H // Hkv
            qs = q[:, :, s:, :].reshape(B, Hkv, rep, T - s, D)
            scores = (qs * scale) @ keys[:, :, None].swapaxes(-1, -2)  # (B, Hkv, rep, Tr, K)
            scores = scores.astype(mx.float32)
            qpos = off + s + mx.arange(T - s)
            allowed = mx.arange(K)[None, :] <= qpos[:, None]
            scores = mx.where(allowed, scores, -mx.inf)
            probs = mx.softmax(scores, axis=-1).reshape(B, H, T - s, K)
            rec.attn[i] = probs[0].transpose(1, 0, 2).astype(mx.float16)  # (Tr, H, K)

        y = self._lin(out.reshape(B, T, H * D), p + "o_proj.weight")
        if rec is not None and rec.start < T:
            rec.heads[i] = _head_contrib_norms(out[:, rec.start :], self.gram(p + "o_proj.weight", H))
        return y

    def _deltanet(self, i: int, x: mx.array, cache: DeltaCache, rec: Recorder | None) -> mx.array:
        cfg = self.cfg
        p = f"model.layers.{i}.linear_attn."
        B, T, _ = x.shape
        Hk, Hv = cfg.linear_n_k_heads, cfg.linear_n_v_heads
        Dk, Dv = cfg.linear_k_head_dim, cfg.linear_v_head_dim
        key_dim = Hk * Dk
        K = cfg.linear_conv_kernel

        qkv = self._lin(x, p + "in_proj_qkv.weight")
        z = self._lin(x, p + "in_proj_z.weight").reshape(B, T, Hv, Dv)
        b = self._lin(x, p + "in_proj_b.weight")
        a = self._lin(x, p + "in_proj_a.weight")

        if self._masks is not None or cache.conv is None:
            conv_state = mx.zeros((B, K - 1, qkv.shape[-1]), qkv.dtype)
        else:
            conv_state = cache.conv
        conv_in = mx.concatenate([conv_state, qkv], axis=1)
        if self._masks is None:
            cache.conv = mx.contiguous(conv_in[:, -(K - 1) :, :])
        # conv weight is stored (C, 1, K); mx.conv1d wants (C_out, K, C_in/groups)
        w_conv = self.w[p + "conv1d.weight"].swapaxes(1, 2)
        conv_out = nn.silu(mx.conv1d(conv_in, w_conv, groups=qkv.shape[-1]))

        q, k, v = mx.split(conv_out, [key_dim, 2 * key_dim], axis=-1)
        q = q.reshape(B, T, Hk, Dk)
        k = k.reshape(B, T, Hk, Dk)
        v = v.reshape(B, T, Hv, Dv)
        q, k = normalize_qk(q, k, inv_scale=Dk**-0.5, eps=1e-6)

        diff = self._masks is not None
        out, state = gated_delta_update(
            q, k, v, a, b, self.w[p + "A_log"], self.w[p + "dt_bias"], None if diff else cache.ssm, None, use_kernel=not diff
        )
        if not diff:
            cache.ssm = state
            cache.offset += T

        # gated RMSNorm (plain gain, not 1 + w)
        normed = mx.fast.rms_norm(out, self.w[p + "norm.weight"], cfg.rms_norm_eps)
        out = (nn.silu(z.astype(mx.float32)) * normed.astype(mx.float32)).astype(x.dtype)
        if diff:
            out = out * self._masks["heads"][i, :Hv][None, None, :, None].astype(out.dtype)

        y = self._lin(out.reshape(B, T, Hv * Dv), p + "out_proj.weight")
        if rec is not None and rec.start < T:
            rec.heads[i] = _head_contrib_norms(out[:, rec.start :], self.gram(p + "out_proj.weight", Hv))
        return y

    def _mlp(self, i: int, x: mx.array, rec: Recorder | None) -> mx.array:
        p = f"model.layers.{i}.mlp."
        act = swiglu(self._lin(x, p + "gate_proj.weight"), self._lin(x, p + "up_proj.weight"))
        if self._masks is not None:
            act = act * self._masks["mlp"][i].astype(act.dtype)
        if rec is not None and rec.start < x.shape[1]:
            rec.mlp[i] = act[:, rec.start :]
        return self._lin(act, p + "down_proj.weight")

    # --- model ---------------------------------------------------------------

    def hidden(self, ids: mx.array, cache: list, rec: Recorder | None = None) -> mx.array:
        """Final normed hidden states (B, T, hidden)."""
        h = self.w["model.embed_tokens.weight"][ids]
        s = rec.start if rec is not None else 0
        for i in range(self.cfg.n_layers):
            p = f"model.layers.{i}."
            x = self._rms(h, p + "input_layernorm.weight")
            if self.cfg.is_full_attention(i):
                r = self._attention(i, x, cache[i], rec)
            else:
                r = self._deltanet(i, x, cache[i], rec)
            h = h + r
            m = self._mlp(i, self._rms(h, p + "post_attention_layernorm.weight"), rec)
            h = h + m
            if rec is not None and s < h.shape[1]:
                rec.mixer_out[i] = _norm(r[:, s:])
                rec.mlp_out[i] = _norm(m[:, s:])
                rec.resid[i] = _norm(h[:, s:])
                rec.resid_vec[i] = h[:, s:].astype(mx.float16)
        return self._rms(h, "model.norm.weight")

    def logits(self, h: mx.array) -> mx.array:
        head = self.w.get("lm_head.weight", self.w["model.embed_tokens.weight"])
        return h @ head.T
