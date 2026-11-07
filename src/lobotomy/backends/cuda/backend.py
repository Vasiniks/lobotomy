"""CUDA backend (PyTorch). Not implemented yet.

This is the slot for an NVIDIA backend. The engine, edits, sampling, server and
UI are shared, so a backend only has to implement ``Backend`` (see
``backends/base.py``). The MLX backend (``backends/mlx``) is the reference.

Suggested route:

- ``load()``: build ``transformers.Qwen3_5ForCausalLM`` from a config matching
  ``ModelConfig`` and ``load_state_dict`` the weights. The loader already
  produces Hugging Face names and conventions. Or port ``backends/mlx/model.py``
  to torch, which keeps recording simple.
- Recording: forward hooks on ``mlp.act_fn`` (or the MLP itself) and on the
  attention / linear-attention output projections. Use eager attention to get
  probabilities, for recorded positions only.
- ``set_tensor_ops()``: keep a pristine copy of each tensor on first edit, then
  rebuild it as pristine-plus-ops. Ops on RMSNorm weights act on the gain
  ``1 + w``; see ``edits.is_gain_norm``.
- DeltaNet: transformers falls back to pure torch if flash-linear-attention /
  causal-conv1d are missing; installing them is much faster on CUDA.

``tests/test_backend_contract.py`` checks any backend against the same rules.
"""

from __future__ import annotations

from ..base import Backend

_MSG = "The CUDA backend isn't implemented yet; see lobotomy/backends/cuda/backend.py."


class CUDABackend(Backend):
    name = "cuda"

    def __init__(self, **kwargs):
        raise NotImplementedError(_MSG)

    def load(self, weights, config):
        raise NotImplementedError(_MSG)

    def new_cache(self):
        raise NotImplementedError(_MSG)

    def cache_length(self, cache):
        raise NotImplementedError(_MSG)

    def forward(self, token_ids, cache, record=None):
        raise NotImplementedError(_MSG)

    def set_tensor_ops(self, tensor, ops):
        raise NotImplementedError(_MSG)

    def tensor_names(self):
        raise NotImplementedError(_MSG)

    def tensor_shape(self, tensor):
        raise NotImplementedError(_MSG)

    def read_tensor(self, tensor):
        raise NotImplementedError(_MSG)
