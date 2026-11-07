"""Backends run the model. See ``base.Backend`` for the contract."""

from __future__ import annotations

from .base import Backend, RecordSpec, Recording

__all__ = ["Backend", "RecordSpec", "Recording", "get_backend", "available_backends"]


def get_backend(name: str, **kwargs) -> Backend:
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend(**kwargs)
    if name == "cuda":
        from .cuda import CUDABackend

        return CUDABackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (choose from {available_backends()})")


def available_backends() -> list[str]:
    return ["mlx", "cuda"]


def default_backend() -> str:
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "cuda"
