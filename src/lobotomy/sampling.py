"""Token sampling on CPU (numpy), shared by every backend.

Sampling happens outside the backend so the same seed and the same logits give
the same token on any backend. "Ask again" reuses the seed, so a changed answer
comes from the edits, not from luck.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Turning thinking on raises a lower reply limit to this, so the model can finish
# its thought before answering (512 cut magi-1.2-2B off mid-thought).
THINK_MAX_TOKENS = 2048


@dataclass
class SamplingParams:
    temperature: float = 0.7  # 0 = greedy
    top_k: int = 20
    top_p: float = 0.8
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 512
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _top_candidates(lg: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest logits, sorted descending.

    Pre-filters to logits within 16 of the max (anything lower has probability
    below e^-16 relative to the top token), which is ~10x faster than a full
    argpartition over a 248k vocabulary.
    """
    n = lg.size
    if k >= n:
        return np.argsort(-lg)
    near = np.flatnonzero(lg > lg.max() - 16.0)
    if near.size < k:
        cand = np.argpartition(-lg, k - 1)[:k]
    elif near.size > k:
        cand = near[np.argpartition(-lg[near], k - 1)[:k]]
    else:
        cand = near
    return cand[np.argsort(-lg[cand])]


def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m, dtype=np.float32).sum(dtype=np.float64)))


class Sampler:
    n_alternatives = 5

    def __init__(self, params: SamplingParams):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.seen: dict[int, int] = {}

    def __call__(self, logits: np.ndarray) -> tuple[int, float, list[tuple[int, float]]]:
        """Pick a token.

        Returns (token_id, its probability under the final sampling distribution,
        the top alternatives as (token_id, probability at temperature 1)).
        """
        p = self.p
        lg = np.asarray(logits, dtype=np.float32)
        if self.seen and (p.presence_penalty or p.repetition_penalty != 1.0):
            lg = lg.copy()
            ids = np.fromiter(self.seen.keys(), dtype=np.int64)
            if p.repetition_penalty != 1.0:
                v = lg[ids]
                lg[ids] = np.where(v > 0, v / p.repetition_penalty, v * p.repetition_penalty)
            if p.presence_penalty:
                lg[ids] -= p.presence_penalty

        n = lg.size
        k = p.top_k if 0 < p.top_k < n else n
        kk = min(n, max(k, self.n_alternatives))
        cand = _top_candidates(lg, kk)
        lse = _logsumexp(lg)
        alts = [(int(i), float(np.exp(lg[i] - lse))) for i in cand[: self.n_alternatives]]

        if p.temperature <= 0:
            tok = int(cand[0])
            self._saw(tok)
            return tok, 1.0, alts

        cand = cand[:k]
        cl = lg[cand].astype(np.float64) / p.temperature
        probs = np.exp(cl - cl.max())
        probs /= probs.sum()
        if p.min_p > 0:
            keep = probs >= p.min_p * probs[0]
            cand, probs = cand[keep], probs[keep] / probs[keep].sum()
        if 0 < p.top_p < 1:
            cut = int(np.searchsorted(np.cumsum(probs), p.top_p) + 1)
            cand, probs = cand[:cut], probs[:cut] / probs[:cut].sum()
        i = int(self.rng.choice(len(cand), p=probs))
        tok = int(cand[i])
        self._saw(tok)
        return tok, float(probs[i]), alts

    def _saw(self, tok: int) -> None:
        self.seen[tok] = self.seen.get(tok, 0) + 1


def top_tokens(logits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """The k most likely tokens and their probabilities (temperature 1)."""
    lg = np.asarray(logits, dtype=np.float32)
    idx = _top_candidates(lg, k)
    lse = _logsumexp(lg)
    return [(int(i), float(np.exp(lg[i] - lse))) for i in idx]
