"""Summaries of a Recording for display: which layers, neurons and heads fired.

"Fired" means large magnitude. MLP activations ``silu(gate) * up`` are signed and
both signs move the output, so everything here ranks by absolute value.

``rows`` selects recorded positions (rows of the Recording). ``None`` means all.
"""

from __future__ import annotations

import numpy as np

from .backends.base import Recording


def _rows(rec: Recording, rows) -> np.ndarray:
    if rows is None:
        return np.arange(rec.n_positions)
    return np.asarray(rows, dtype=np.int64)


def layer_activity(rec: Recording, rows=None) -> list[dict]:
    """Per layer: how busy it was over the selected positions."""
    r = _rows(rec, rows)
    mlp = np.abs(rec.mlp[r].astype(np.float32))  # (T, L, N)
    out = []
    for i in range(rec.mlp.shape[1]):
        a = mlp[:, i, :]
        heads = rec.heads[r, i, :]
        heads = heads[:, ~np.isnan(heads[0])] if len(r) else heads
        out.append(
            {
                "layer": i,
                "neuron_mean": float(a.mean()) if a.size else 0.0,
                "neuron_peak": float(a.max()) if a.size else 0.0,
                "mixer_out": float(rec.mixer_out[r, i].mean()) if len(r) else 0.0,
                "mlp_out": float(rec.mlp_out[r, i].mean()) if len(r) else 0.0,
                "resid": float(rec.resid[r, i].mean()) if len(r) else 0.0,
                "heads": heads.mean(axis=0).tolist() if len(r) else [],
            }
        )
    return out


def neuron_grid(rec: Recording, layer: int, rows=None, reduce: str = "mean") -> np.ndarray:
    """(N,) activity of every neuron in a layer: mean or max of |activation|."""
    a = np.abs(rec.mlp[_rows(rec, rows), layer, :].astype(np.float32))
    if a.shape[0] == 0:
        return np.zeros(rec.mlp.shape[-1], np.float32)
    return a.max(axis=0) if reduce == "max" else a.mean(axis=0)


def signed_neuron_values(rec: Recording, layer: int, rows=None) -> np.ndarray:
    """(N,) mean signed activation, to tell positive firing from negative."""
    a = rec.mlp[_rows(rec, rows), layer, :].astype(np.float32)
    return a.mean(axis=0) if a.shape[0] else np.zeros(rec.mlp.shape[-1], np.float32)


def top_neurons(rec: Recording, layer: int, k: int = 20, rows=None, reduce: str = "mean") -> list[dict]:
    grid = neuron_grid(rec, layer, rows, reduce)
    signed = signed_neuron_values(rec, layer, rows)
    k = min(k, grid.size)
    idx = np.argpartition(-grid, k - 1)[:k]
    idx = idx[np.argsort(-grid[idx])]
    return [{"neuron": int(n), "activity": float(grid[n]), "mean_signed": float(signed[n])} for n in idx]


def neuron_trace(rec: Recording, layer: int, neuron: int, rows=None) -> np.ndarray:
    """(T,) signed activation of one neuron across positions."""
    return rec.mlp[_rows(rec, rows), layer, neuron].astype(np.float32)


def top_neurons_at(rec: Recording, row: int, k: int = 8) -> list[dict]:
    """For one position: each layer's most active neurons."""
    out = []
    for layer in range(rec.mlp.shape[1]):
        a = rec.mlp[row, layer, :].astype(np.float32)
        idx = np.argpartition(-np.abs(a), k - 1)[:k]
        idx = idx[np.argsort(-np.abs(a[idx]))]
        out.append({"layer": layer, "neurons": [{"neuron": int(n), "value": float(a[n])} for n in idx]})
    return out


def sparkline(values, width: int | None = None) -> str:
    """Unicode bar chart for terminals."""
    bars = " ▁▂▃▄▅▆▇█"
    v = np.asarray(values, dtype=np.float64)
    if width and v.size > width:
        v = np.array([c.mean() for c in np.array_split(v, width)])
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return bars[4] * v.size
    return "".join(bars[int(round((x - lo) / (hi - lo) * (len(bars) - 1)))] for x in v)
