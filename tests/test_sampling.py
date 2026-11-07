import numpy as np

from lobotomy.sampling import Sampler, SamplingParams, top_tokens


def logits(n=1000, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * 4).astype(np.float32)


def test_greedy_is_argmax():
    lg = logits()
    tok, p, alts = Sampler(SamplingParams(temperature=0))(lg)
    assert tok == int(np.argmax(lg)) and p == 1.0
    assert alts[0][0] == tok


def test_same_seed_same_tokens():
    lg = logits()
    s1, s2 = Sampler(SamplingParams(seed=7)), Sampler(SamplingParams(seed=7))
    assert [s1(lg)[0] for _ in range(20)] == [s2(lg)[0] for _ in range(20)]


def test_top_k_restricts_choices():
    lg = logits()
    allowed = set(np.argsort(-lg)[:3].tolist())
    s = Sampler(SamplingParams(temperature=5.0, top_k=3, top_p=1.0, seed=1))
    assert all(s(lg)[0] in allowed for _ in range(200))


def test_alternatives_are_probabilities_in_order():
    lg = logits()
    _, _, alts = Sampler(SamplingParams())(lg)
    probs = [p for _, p in alts]
    assert probs == sorted(probs, reverse=True)
    full = np.exp(lg - lg.max())
    full /= full.sum()
    assert abs(probs[0] - full.max()) < 1e-5


def test_top_tokens_handles_flat_and_peaked():
    flat = np.zeros(500, np.float32)
    assert len(top_tokens(flat, 5)) == 5
    peaked = np.full(500, -100.0, np.float32)
    peaked[[3, 9]] = [10.0, 9.0]
    assert [t for t, _ in top_tokens(peaked, 2)] == [3, 9]


def test_presence_penalty_discourages_repeats():
    lg = np.zeros(10, np.float32)
    lg[0] = 1.0
    s = Sampler(SamplingParams(temperature=0, presence_penalty=5.0))
    assert s(lg)[0] == 0
    assert s(lg)[0] != 0
