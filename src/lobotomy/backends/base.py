"""The contract every backend implements.

A backend owns the model's weights on its device and runs forward passes. It
does not know about chats, sampling or the edit list. The engine handles those
and talks to the backend only through this interface.

To add a backend (CUDA, for example):

1. Subclass :class:`Backend`.
2. ``load()`` receives float weights with Hugging Face names and layout (see
   ``gguf_loader``) plus the :class:`ModelConfig`.
3. ``forward()`` runs the tokens through the model, updates the cache, and
   returns float32 logits for the last position plus a :class:`Recording` when
   asked.
4. ``set_tensor_ops()`` makes a tensor equal to its pristine value with the
   given :class:`SliceOp` list applied in order. An empty list restores it. Keep
   pristine copies of the tensors you edit, so undo is exact.

``tests/test_backend_contract.py`` runs the same checks against any backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ModelConfig
from ..edits import SliceOp


@dataclass
class RecordSpec:
    """What to record during a forward pass.

    Positions before ``start`` (absolute token position) are dropped, so long
    chats don't record the whole history every turn.
    """

    start: int = 0
    attention: bool = True  # attention probabilities for full-attention layers


@dataclass
class Recording:
    """Per-position internals, aligned with the input tokens.

    Shapes use T = recorded positions, L = layers, N = MLP neurons, H = max heads.

    - ``positions``: (T,) absolute token positions.
    - ``mlp``: (T, L, N) float16. MLP neuron activations ``silu(gate)*up``.
      Signed: positive and negative both count as firing.
    - ``heads``: (T, L, H) float32. L2 norm of each head's contribution to the
      residual stream. NaN where a layer has fewer heads than H.
    - ``mixer_out``: (T, L) float32. Norm of the attention/DeltaNet block output.
    - ``mlp_out``: (T, L) float32. Norm of the MLP block output.
    - ``resid``: (T, L) float32. Norm of the residual stream after each layer.
    - ``resid_vec``: (T, L, hidden) float16. The residual stream itself after each
      layer, for the logit lens.
    - ``attention``: {layer: (T, n_heads, K) float16}, attention probabilities
      of each recorded query over all K keys so far (full-attention layers only).
    """

    positions: np.ndarray
    mlp: np.ndarray
    heads: np.ndarray
    mixer_out: np.ndarray
    mlp_out: np.ndarray
    resid: np.ndarray
    resid_vec: np.ndarray
    attention: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        return int(self.positions.shape[0])

    @staticmethod
    def concat(parts: list["Recording"]) -> "Recording":
        parts = [p for p in parts if p is not None and p.n_positions > 0]
        if not parts:
            raise ValueError("nothing to concatenate")
        if len(parts) == 1:
            return parts[0]
        attn: dict[int, np.ndarray] = {}
        layers = parts[0].attention.keys()
        for layer in layers:
            k_max = max(p.attention[layer].shape[-1] for p in parts)
            padded = [
                np.pad(p.attention[layer], ((0, 0), (0, 0), (0, k_max - p.attention[layer].shape[-1])))
                for p in parts
            ]
            attn[layer] = np.concatenate(padded, axis=0)
        return Recording(
            positions=np.concatenate([p.positions for p in parts]),
            mlp=np.concatenate([p.mlp for p in parts]),
            heads=np.concatenate([p.heads for p in parts]),
            mixer_out=np.concatenate([p.mixer_out for p in parts]),
            mlp_out=np.concatenate([p.mlp_out for p in parts]),
            resid=np.concatenate([p.resid for p in parts]),
            resid_vec=np.concatenate([p.resid_vec for p in parts]),
            attention=attn,
        )


class Backend(ABC):
    name: str = "base"
    config: ModelConfig

    @abstractmethod
    def load(self, weights: dict[str, np.ndarray], config: ModelConfig) -> None:
        """Take ownership of the weights (HF names, (out, in) layout)."""

    @abstractmethod
    def new_cache(self) -> Any:
        """A fresh, empty cache (KV cache for attention, state for DeltaNet)."""

    @abstractmethod
    def cache_length(self, cache: Any) -> int:
        """How many tokens the cache has seen."""

    @abstractmethod
    def forward(
        self, token_ids: list[int], cache: Any, record: RecordSpec | None = None
    ) -> tuple[np.ndarray, Recording | None]:
        """Run tokens through the model, appending them to ``cache``.

        Returns float32 logits ``(vocab,)`` for the last position, and a
        Recording when ``record`` is given (None otherwise).
        """

    @abstractmethod
    def set_tensor_ops(self, tensor: str, ops: list[SliceOp]) -> None:
        """Set ``tensor`` to pristine-with-``ops``. ``ops == []`` restores it."""

    @abstractmethod
    def tensor_names(self) -> list[str]: ...

    @abstractmethod
    def tensor_shape(self, tensor: str) -> tuple[int, ...]: ...

    @abstractmethod
    def read_tensor(self, tensor: str) -> np.ndarray:
        """Current (possibly edited) value as float32 numpy, for visualization."""

    def read_pristine(self, tensor: str) -> np.ndarray:
        """The unedited value. Backends that keep pristine copies should override."""
        return self.read_tensor(tensor)

    # --- optional extras (the UI hides features a backend doesn't support) -------

    def snapshot_cache(self, cache: Any) -> Any:
        """An independent copy of ``cache`` (later forwards must not change it).
        Needed for "keep memory" mode; without it the engine always re-reads."""
        raise NotImplementedError

    def restore_cache(self, snapshot: Any) -> Any:
        """A new cache equal to a snapshot (the snapshot stays reusable)."""
        raise NotImplementedError

    def logits_for(self, token_ids: list[int], cache: Any | None = None) -> np.ndarray:
        """Logits at every position, (T, vocab). Used by tests."""
        raise NotImplementedError

    def unembed_topk(self, vectors: np.ndarray, k: int = 5, apply_final_norm: bool = True):
        """Project residual-stream vectors (n, hidden) to the vocabulary.

        Returns (ids, probs), each (n, k). This is the logit lens.
        """
        raise NotImplementedError

    def attribute(self, token_ids: list[int], target_pos: int) -> tuple[float, np.ndarray, np.ndarray]:
        """Attribution: ``(logp, grad_mlp (L, N), grad_heads (L, H_max))`` where the
        grads are d log p(token_ids[target_pos] | prefix) / d mask, for
        multiplicative masks (all 1) on every MLP neuron and head."""
        raise NotImplementedError

    def score_variants(self, token_ids, start, variants, current_ops) -> list[dict]:
        """Ablation scan. For each variant ``{tensor: [SliceOp, ...]}`` (complete op
        lists for those tensors), measure the KL from the current model on tokens
        ``start..``. Restore every touched tensor to ``current_ops`` afterwards.
        Returns ``[{"kl": (n,) array, "mean_kl": float, "top1_agree": float}]``.
        """
        raise NotImplementedError

    def damage_report(self, token_ids: list[int], start: int, healthy) -> dict[str, np.ndarray]:
        """Compare current weights with the healthy model on a fixed token sequence.

        Positions ``start-1 .. T-2`` predict tokens ``start .. T-1``. ``healthy`` is
        a context manager that temporarily restores pristine weights. Returns
        numpy arrays, one entry per predicted token: ``kl`` (healthy || current),
        ``logp_current`` and ``logp_healthy`` of the actual token, and
        ``top_current`` / ``top_healthy`` (each model's most likely token).
        """
        raise NotImplementedError
