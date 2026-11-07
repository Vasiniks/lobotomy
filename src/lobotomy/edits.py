"""Edits: what the user breaks, and how that maps onto weights.

An :class:`Edit` names a part of the model (a neuron, a head, a layer, a token,
or a raw weight slice) and an operation (scale it, or add noise). It resolves to
:class:`SliceOp` s, plain descriptions of weight changes that any backend can
apply. Backends recompute an edited tensor from its pristine copy plus every
active op on it, so undoing any edit, in any order, is exact.

How each target maps onto weights (W has PyTorch layout ``(out, in)``):

- **neuron** (layer L, MLP neuron n): rows ``n`` of ``mlp.up_proj``.
  The neuron's activation ``silu(gate_n·x) * (up_n·x)`` is linear in ``up_n``,
  so scaling the row by ``s`` scales the activation, and its contribution to the
  output, by exactly ``s``. Recorded activations show the edit.
- **head** (layer L, head h): the head's columns of the output projection,
  ``self_attn.o_proj`` (attention layers, 256 wide) or ``linear_attn.out_proj``
  (DeltaNet layers, 128 wide). This scales the head's contribution to the
  residual stream by exactly ``s``. (A full-attention head's sigmoid output gate
  can only reach 0.5, so gating it is not a true off switch.)
- **layer** (layer L, part): ``mixer`` scales the whole attention/DeltaNet
  output projection, ``mlp`` scales ``mlp.down_proj``, ``all`` scales both.
  With ``s = 0`` the layer passes the residual stream through untouched.
- **token** (vocab id t): row ``t`` of the embedding. The embedding is tied to
  the output head, so this changes both how the model reads the token and how
  likely it is to say it.
- **weight** (any tensor name, optional rows/cols): a raw slice, for power use.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .config import ModelConfig

Op = Literal["scale", "noise"]
TargetKind = Literal["neuron", "head", "layer", "token", "weight"]


@dataclass(frozen=True)
class SliceOp:
    """Change a slice of one weight tensor.

    ``axis`` is 0 for rows, 1 for columns, None for the whole tensor.
    ``indices`` lists rows/columns as half-open ``(start, stop)`` ranges.
    ``op`` is "scale" (multiply by ``value``) or "noise" (add Gaussian noise with
    standard deviation ``value`` times the slice's RMS, from ``seed``).
    """

    tensor: str
    axis: int | None
    indices: tuple[tuple[int, int], ...]
    op: Op
    value: float
    seed: int = 0


@dataclass(frozen=True)
class Target:
    kind: TargetKind
    layer: int | None = None
    index: int | None = None  # neuron, head or token id
    part: str = "all"  # for layer targets: "all" | "mixer" | "mlp"
    tensor: str | None = None  # for weight targets
    rows: tuple[int, int] | None = None
    cols: tuple[int, int] | None = None

    def label(self) -> str:
        if self.kind == "neuron":
            return f"L{self.layer} neuron {self.index}"
        if self.kind == "head":
            return f"L{self.layer} head {self.index}"
        if self.kind == "layer":
            return f"L{self.layer}" + ("" if self.part == "all" else f" {self.part}")
        if self.kind == "token":
            return f"token {self.index}"
        sl = []
        if self.rows:
            sl.append(f"rows {self.rows[0]}:{self.rows[1]}")
        if self.cols:
            sl.append(f"cols {self.cols[0]}:{self.cols[1]}")
        return f"{self.tensor}" + (f" [{', '.join(sl)}]" if sl else "")


@dataclass(frozen=True)
class Edit:
    id: int
    target: Target
    op: Op = "scale"
    value: float = 0.0  # scale factor (0 = off), or relative noise std
    seed: int = 0
    note: str = ""

    def label(self) -> str:
        if self.op == "scale":
            if self.value == 0:
                verb = "off"
            elif self.value < 1:
                verb = f"×{self.value:g} (down)"
            else:
                verb = f"×{self.value:g} (boost)" if self.value > 1 else "×1"
        else:
            verb = f"noise {self.value:g}"
        return f"{self.target.label()}: {verb}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label()
        return d


GAIN_NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.norm.weight",
)


def is_gain_norm(tensor: str) -> bool:
    """True for RMSNorm weights stored as ``w`` with gain ``1 + w``.

    Ops on these tensors act on the gain, so "scale by 0" really silences the
    norm's output instead of resetting it to a gain of 1.
    """
    return tensor.endswith(GAIN_NORMS)


def _layer_prefix(layer: int) -> str:
    return f"model.layers.{layer}."


def _check_layer(cfg: ModelConfig, layer: int | None) -> int:
    if layer is None or not 0 <= layer < cfg.n_layers:
        raise ValueError(f"layer must be in 0..{cfg.n_layers - 1}, got {layer}")
    return layer


def mixer_out_proj(cfg: ModelConfig, layer: int) -> tuple[str, int]:
    """(tensor name, head width) of a layer's attention/DeltaNet output projection."""
    p = _layer_prefix(layer)
    if cfg.is_full_attention(layer):
        return p + "self_attn.o_proj.weight", cfg.head_dim
    return p + "linear_attn.out_proj.weight", cfg.linear_v_head_dim


def resolve(edit: Edit, cfg: ModelConfig) -> list[SliceOp]:
    t = edit.target

    def op(tensor, axis, indices):
        return SliceOp(tensor, axis, tuple(indices), edit.op, float(edit.value), edit.seed)

    if t.kind == "neuron":
        layer = _check_layer(cfg, t.layer)
        n = t.index
        if n is None or not 0 <= n < cfg.intermediate_size:
            raise ValueError(f"neuron must be in 0..{cfg.intermediate_size - 1}, got {n}")
        return [op(_layer_prefix(layer) + "mlp.up_proj.weight", 0, [(n, n + 1)])]

    if t.kind == "head":
        layer = _check_layer(cfg, t.layer)
        n_heads = cfg.n_layer_heads(layer)
        h = t.index
        if h is None or not 0 <= h < n_heads:
            raise ValueError(f"layer {layer} has heads 0..{n_heads - 1}, got {h}")
        tensor, width = mixer_out_proj(cfg, layer)
        return [op(tensor, 1, [(h * width, (h + 1) * width)])]

    if t.kind == "layer":
        layer = _check_layer(cfg, t.layer)
        if t.part not in ("all", "mixer", "mlp"):
            raise ValueError(f"layer part must be all, mixer or mlp, got {t.part!r}")
        ops = []
        if t.part in ("all", "mixer"):
            ops.append(op(mixer_out_proj(cfg, layer)[0], None, []))
        if t.part in ("all", "mlp"):
            ops.append(op(_layer_prefix(layer) + "mlp.down_proj.weight", None, []))
        return ops

    if t.kind == "token":
        tid = t.index
        if tid is None or not 0 <= tid < cfg.vocab_size:
            raise ValueError(f"token id must be in 0..{cfg.vocab_size - 1}, got {tid}")
        return [op("model.embed_tokens.weight", 0, [(tid, tid + 1)])]

    if t.kind == "weight":
        if not t.tensor:
            raise ValueError("weight target needs a tensor name")
        if t.rows and t.cols:
            raise ValueError("a weight edit takes rows or cols, not both")
        if t.rows:
            return [op(t.tensor, 0, [t.rows])]
        if t.cols:
            return [op(t.tensor, 1, [t.cols])]
        return [op(t.tensor, None, [])]

    raise ValueError(f"unknown target kind {t.kind!r}")


@dataclass
class EditStack:
    """The active edits, in the order they were made."""

    config: ModelConfig
    edits: list[Edit] = field(default_factory=list)
    _next_id: int = 1

    def add(self, target: Target, op: Op = "scale", value: float = 0.0, seed: int | None = None, note: str = "") -> Edit:
        e = Edit(id=self._next_id, target=target, op=op, value=value, seed=self._next_id if seed is None else seed, note=note)
        resolve(e, self.config)  # validate before accepting
        self._next_id += 1
        self.edits.append(e)
        return e

    def get(self, edit_id: int) -> Edit:
        for e in self.edits:
            if e.id == edit_id:
                return e
        raise KeyError(f"no edit with id {edit_id}")

    def update(self, edit_id: int, value: float | None = None, op: Op | None = None) -> Edit:
        """Change an edit's value (e.g. a slider moving) without changing its place or seed."""
        for i, e in enumerate(self.edits):
            if e.id == edit_id:
                new = Edit(
                    id=e.id,
                    target=e.target,
                    op=op or e.op,
                    value=e.value if value is None else float(value),
                    seed=e.seed,
                    note=e.note,
                )
                resolve(new, self.config)
                self.edits[i] = new
                return new
        raise KeyError(f"no edit with id {edit_id}")

    def remove(self, edit_id: int) -> Edit:
        for i, e in enumerate(self.edits):
            if e.id == edit_id:
                return self.edits.pop(i)
        raise KeyError(f"no edit with id {edit_id}")

    def clear(self) -> list[Edit]:
        old, self.edits = self.edits, []
        return old

    def ops_by_tensor(self) -> dict[str, list[SliceOp]]:
        out: dict[str, list[SliceOp]] = {}
        for e in self.edits:
            for o in resolve(e, self.config):
                out.setdefault(o.tensor, []).append(o)
        return out

    def tensors_of(self, edits: list[Edit]) -> set[str]:
        return {o.tensor for e in edits for o in resolve(e, self.config)}
