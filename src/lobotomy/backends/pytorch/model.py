"""Qwen3.5 text model in PyTorch, for NVIDIA GPUs (CUDA) and the CPU.

A port of ``backends/mlx/model.py``: the same math, the same recording and the
same attribution masks, so both backends show the same model. Gated DeltaNet
runs as the plain recurrence, one step per token, like mlx-lm's reference ops
path; there are no custom kernels, so it runs anywhere PyTorch does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ...config import ModelConfig


class AttnCache:
    """Pre-allocated KV cache for one full-attention layer."""

    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, k: torch.Tensor, v: torch.Tensor):
        B, H, T, D = k.shape
        prev = self.offset
        if self.keys is None or prev + T > self.keys.shape[2]:
            n = ((prev + T + self.step - 1) // self.step) * self.step
            new_k = k.new_zeros((B, H, n, D))
            new_v = v.new_zeros((B, H, n, D))
            if self.keys is not None and prev > 0:
                new_k[..., :prev, :] = self.keys[..., :prev, :]
                new_v[..., :prev, :] = self.values[..., :prev, :]
            self.keys, self.values = new_k, new_v
        self.keys[..., prev : prev + T, :] = k
        self.values[..., prev : prev + T, :] = v
        self.offset += T
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class DeltaCache:
    """Conv window and recurrent state for one Gated DeltaNet layer. Both are
    replaced on every step, never changed in place, so a snapshot can share them."""

    def __init__(self):
        self.conv = None
        self.ssm = None
        self.offset = 0


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


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm computed in float32, returned in the input dtype (like mx.fast.rms_norm)."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if w is not None:
        y = y * w.float()
    return y.to(x.dtype)


def rope(x: torch.Tensor, dims: int, base: float, offset: int) -> torch.Tensor:
    """Rotary embedding on the first ``dims`` features of each head, in halves
    (mx.fast.rope with traditional=False). x: (B, H, T, D)."""
    T = x.shape[2]
    half = dims // 2
    inv = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) * 2.0 / dims)
    pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2, rest = xf[..., :half], xf[..., half:dims], xf[..., dims:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], dim=-1)
    return out.to(x.dtype)


def normalize_qk(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6):
    """L2-normalize q and k, and fold the head_dim**-0.5 readout scale into q
    (mlx-lm's ``normalize_qk``; the eps is added to the sum of squares)."""
    D = q.shape[-1]
    qf, kf = q.float(), k.float()
    qn = qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) + eps) * D**-0.5
    kn = kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) + eps)
    return qn.to(q.dtype), kn.to(k.dtype)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return x.float().pow(2).sum(-1).sqrt()


def head_gram(w_out: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Per-head Gram matrices G_h = W_hᵀ W_h, (H, D, D) float32, so a head's
    contribution norm ||W_h o_h|| is sqrt(o_hᵀ G_h o_h)."""
    hidden = w_out.shape[0]
    D = w_out.shape[1] // n_heads
    w = w_out.float().reshape(hidden, n_heads, D).permute(1, 0, 2)  # (H, hidden, D)
    return w.transpose(1, 2) @ w


def _head_contrib_norms(out_heads: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
    """(B, T, H, D) head outputs -> (B, T, H) norms of their residual contributions."""
    o = out_heads.float()
    q = torch.einsum("bthd,hde,bthe->bth", o, gram, o)
    return q.clamp_min(0.0).sqrt()


class Qwen35:
    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor]):
        self.cfg = config
        self.w = weights
        self._gram: dict[str, torch.Tensor] = {}
        # Set during attribution: multiplicative masks on MLP neurons (L, N) and
        # heads (L, H_max), and a differentiable path (no cache).
        self._masks: dict[str, torch.Tensor] | None = None

    def invalidate(self, tensor: str) -> None:
        self._gram.pop(tensor, None)

    def gram(self, tensor: str, n_heads: int) -> torch.Tensor:
        g = self._gram.get(tensor)
        if g is None:
            g = head_gram(self.w[tensor], n_heads)
            self._gram[tensor] = g
        return g

    # --- helpers -------------------------------------------------------------

    def _lin(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x, self.w[name])

    def _rms(self, x: torch.Tensor, name: str) -> torch.Tensor:
        # Hugging Face convention: gain = 1 + w
        return rms_norm(x, 1.0 + self.w[name].float(), self.cfg.rms_norm_eps)

    def new_cache(self):
        return [AttnCache() if self.cfg.is_full_attention(i) else DeltaCache() for i in range(self.cfg.n_layers)]

    # --- blocks --------------------------------------------------------------

    def _attention(self, i: int, x: torch.Tensor, cache: AttnCache, rec: Recorder | None) -> torch.Tensor:
        cfg = self.cfg
        p = f"model.layers.{i}.self_attn."
        B, T, _ = x.shape
        H, Hkv, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

        qg = self._lin(x, p + "q_proj.weight").reshape(B, T, H, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        k = self._lin(x, p + "k_proj.weight").reshape(B, T, Hkv, D)
        v = self._lin(x, p + "v_proj.weight").reshape(B, T, Hkv, D)

        q = rms_norm(q, 1.0 + self.w[p + "q_norm.weight"].float(), cfg.rms_norm_eps).transpose(1, 2)
        k = rms_norm(k, 1.0 + self.w[p + "k_norm.weight"].float(), cfg.rms_norm_eps).transpose(1, 2)
        v = v.transpose(1, 2)

        diff = self._masks is not None
        off = 0 if diff else cache.offset
        q = rope(q, cfg.rope_dim, cfg.rope_theta, off)
        k = rope(k, cfg.rope_dim, cfg.rope_theta, off)
        keys, values = (k, v) if diff else cache.update_and_fetch(k, v)
        K = keys.shape[2]
        rep = H // Hkv
        kk = keys.repeat_interleave(rep, dim=1)  # query head h reads kv head h // rep
        vv = values.repeat_interleave(rep, dim=1)

        scale = D**-0.5
        if T == 1:
            mask = None
        else:  # query t (absolute off + t) sees keys 0 .. off + t
            mask = torch.arange(K, device=x.device)[None, :] <= (off + torch.arange(T, device=x.device))[:, None]
        out = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=scale)
        out = out.transpose(1, 2)  # (B, T, H, D)
        out = out * torch.sigmoid(gate)
        if diff:
            out = out * self._masks["heads"][i, :H][None, None, :, None].to(out.dtype)

        if rec is not None and rec.attention and rec.start < T:
            s = rec.start
            scores = (q[:, :, s:, :].float() * scale) @ kk.float().transpose(-1, -2)  # (B, H, Tr, K)
            qpos = off + s + torch.arange(T - s, device=x.device)
            allowed = torch.arange(K, device=x.device)[None, :] <= qpos[:, None]
            probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
            rec.attn[i] = probs[0].transpose(0, 1).half()  # (Tr, H, K)

        y = self._lin(out.reshape(B, T, H * D), p + "o_proj.weight")
        if rec is not None and rec.start < T:
            rec.heads[i] = _head_contrib_norms(out[:, rec.start :], self.gram(p + "o_proj.weight", H))
        return y

    def _deltanet(self, i: int, x: torch.Tensor, cache: DeltaCache, rec: Recorder | None) -> torch.Tensor:
        cfg = self.cfg
        p = f"model.layers.{i}.linear_attn."
        B, T, _ = x.shape
        Hk, Hv = cfg.linear_n_k_heads, cfg.linear_n_v_heads
        Dk, Dv = cfg.linear_k_head_dim, cfg.linear_v_head_dim
        key_dim = Hk * Dk
        Kc = cfg.linear_conv_kernel
        diff = self._masks is not None

        qkv = self._lin(x, p + "in_proj_qkv.weight")
        z = self._lin(x, p + "in_proj_z.weight").reshape(B, T, Hv, Dv)
        b = self._lin(x, p + "in_proj_b.weight")
        a = self._lin(x, p + "in_proj_a.weight")

        C = qkv.shape[-1]
        conv_state = qkv.new_zeros((B, Kc - 1, C)) if diff or cache.conv is None else cache.conv
        conv_in = torch.cat([conv_state, qkv], dim=1)
        if not diff:
            cache.conv = conv_in[:, -(Kc - 1) :, :].clone()
        # depthwise causal conv, written as K shifted multiply-adds: torch's
        # grouped conv1d runs one slow kernel per channel on the CPU (6,144 per
        # layer, ~3 s per token); this is the same sum. Weight stored (C, 1, K).
        w_conv = self.w[p + "conv1d.weight"][:, 0, :]  # (C, K)
        conv = conv_in[:, 0:T, :] * w_conv[:, 0]
        for j in range(1, Kc):
            conv = conv + conv_in[:, j : j + T, :] * w_conv[:, j]
        conv_out = F.silu(conv)

        q, k, v = conv_out.split([key_dim, key_dim, Hv * Dv], dim=-1)
        q = q.reshape(B, T, Hk, Dk)
        k = k.reshape(B, T, Hk, Dk)
        v = v.reshape(B, T, Hv, Dv)
        q, k = normalize_qk(q, k)
        if Hv // Hk > 1:
            q = q.repeat_interleave(Hv // Hk, dim=2)
            k = k.repeat_interleave(Hv // Hk, dim=2)

        beta = torch.sigmoid(b).float()  # (B, T, Hv)
        g = torch.exp(-torch.exp(self.w[p + "A_log"].float()) * F.softplus(a.float() + self.w[p + "dt_bias"].float()))
        state = None if diff else cache.ssm
        if state is None:
            state = torch.zeros((B, Hv, Dv, Dk), dtype=torch.float32, device=x.device)

        qf, kf, vf = q.float(), k.float(), v.float()
        ys = []
        for t in range(T):  # the gated delta rule, one token at a time
            kt = kf[:, t]  # (B, Hv, Dk)
            state = state * g[:, t, :, None, None]
            kv_mem = (state @ kt[..., None])[..., 0]  # (B, Hv, Dv)
            delta = (vf[:, t] - kv_mem) * beta[:, t, :, None]
            state = state + delta[..., None] * kt[:, :, None, :]
            ys.append((state @ qf[:, t, :, :, None])[..., 0].to(x.dtype))
        out = torch.stack(ys, dim=1)  # (B, T, Hv, Dv)
        if not diff:
            cache.ssm = state
            cache.offset += T

        # gated RMSNorm (plain gain, not 1 + w)
        normed = rms_norm(out, self.w[p + "norm.weight"], cfg.rms_norm_eps)
        out = (F.silu(z.float()) * normed.float()).to(x.dtype)
        if diff:
            out = out * self._masks["heads"][i, :Hv][None, None, :, None].to(out.dtype)

        y = self._lin(out.reshape(B, T, Hv * Dv), p + "out_proj.weight")
        if rec is not None and rec.start < T:
            rec.heads[i] = _head_contrib_norms(out[:, rec.start :], self.gram(p + "out_proj.weight", Hv))
        return y

    def _mlp(self, i: int, x: torch.Tensor, rec: Recorder | None) -> torch.Tensor:
        p = f"model.layers.{i}.mlp."
        act = F.silu(self._lin(x, p + "gate_proj.weight")) * self._lin(x, p + "up_proj.weight")
        if self._masks is not None:
            act = act * self._masks["mlp"][i].to(act.dtype)
        if rec is not None and rec.start < x.shape[1]:
            rec.mlp[i] = act[:, rec.start :]
        return self._lin(act, p + "down_proj.weight")

    # --- model ---------------------------------------------------------------

    def hidden(self, ids: torch.Tensor, cache: list, rec: Recorder | None = None) -> torch.Tensor:
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
                rec.resid_vec[i] = h[:, s:]
        return self._rms(h, "model.norm.weight")

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        head = self.w.get("lm_head.weight", self.w["model.embed_tokens.weight"])
        return F.linear(h, head)
