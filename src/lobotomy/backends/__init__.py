"""Backends run the model. See ``base.Backend`` for the contract.

- ``mlx``: Apple Silicon (Metal). The default on Apple Silicon Macs.
- ``torch``: PyTorch on an NVIDIA GPU (CUDA) or the CPU. The default anywhere
  else. ``cuda`` is the same backend, required to find a GPU.
"""

from __future__ import annotations

from .base import Backend, RecordSpec, Recording

__all__ = ["Backend", "RecordSpec", "Recording", "get_backend", "available_backends", "default_backend"]


def get_backend(name: str, **kwargs) -> Backend:
    if name == "mlx":
        from .mlx import MLXBackend

        kwargs.pop("device", None)
        return MLXBackend(**kwargs)
    if name in ("torch", "cuda"):
        from .pytorch import TorchBackend

        if name == "cuda":
            kwargs["device"] = kwargs.get("device") or "cuda"
        return TorchBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (choose from {available_backends()})")


def available_backends() -> list[str]:
    return ["mlx", "torch", "cuda"]


def default_backend() -> str:
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"
