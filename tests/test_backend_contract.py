"""Rules every backend must follow. Runs against each backend in
``LOBOTOMY_TEST_BACKENDS`` (default: mlx on Apple Silicon, cuda elsewhere).
"""

import numpy as np
import pytest

from lobotomy.backends.base import RecordSpec
from lobotomy.edits import SliceOp, Target

from conftest import needs_model

pytestmark = needs_model


def test_incremental_decode_matches_full_forward(engine, prompt_ids):
    be = engine.backend
    extra = engine.tokenizer.encode(" Red, orange and yellow.")
    full = be.logits_for(prompt_ids + extra)[-1]
    cache = be.new_cache()
    logits, _ = be.forward(prompt_ids, cache)
    for t in extra:
        logits, _ = be.forward([t], cache)
    assert be.cache_length(cache) == len(prompt_ids) + len(extra)
    assert int(np.argmax(logits)) == int(np.argmax(full))
    p = np.exp(full - full.max()); p /= p.sum()
    q = np.exp(logits - logits.max()); q /= q.sum()
    assert np.abs(p - q).sum() < 0.05  # bf16 drift between batched and stepwise paths


def test_recording_shapes(engine, prompt_ids):
    cfg = engine.config
    _, rec = engine.backend.forward(prompt_ids, engine.backend.new_cache(), RecordSpec(start=len(prompt_ids) - 5))
    T = 5
    assert rec.positions.tolist() == list(range(len(prompt_ids) - 5, len(prompt_ids)))
    assert rec.mlp.shape == (T, cfg.n_layers, cfg.intermediate_size)
    assert rec.heads.shape == (T, cfg.n_layers, cfg.max_heads)
    assert np.isnan(rec.heads[:, 3, 8:]).all() and not np.isnan(rec.heads[:, 3, :8]).any()
    assert not np.isnan(rec.heads[:, 0, :]).any()
    for arr in (rec.mixer_out, rec.mlp_out, rec.resid):
        assert arr.shape == (T, cfg.n_layers)
    assert rec.resid_vec.shape == (T, cfg.n_layers, cfg.hidden_size)
    assert sorted(rec.attention) == cfg.attention_layers
    probs = rec.attention[3].astype(np.float32)
    assert probs.shape == (T, cfg.n_heads, len(prompt_ids))
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=2e-2)
    # causal: the first recorded query can't see the last key
    assert probs[0, :, -1].max() == 0


def test_restore_is_bitwise_exact(engine):
    be = engine.backend
    name = "model.layers.2.mlp.down_proj.weight"
    before = be.read_tensor(name)
    be.set_tensor_ops(name, [SliceOp(name, 1, ((0, 100),), "noise", 1.0, seed=3)])
    assert not np.array_equal(be.read_tensor(name), before)
    be.set_tensor_ops(name, [])
    assert np.array_equal(be.read_tensor(name), before)


def test_scale_ops_match_reference(engine):
    be = engine.backend
    name = "model.layers.4.linear_attn.out_proj.weight"
    before = be.read_tensor(name)
    ops = [SliceOp(name, 1, ((128, 256),), "scale", 0.0), SliceOp(name, 0, ((0, 10),), "scale", 2.0)]
    be.set_tensor_ops(name, ops)
    after = be.read_tensor(name)
    ref = before.copy()
    ref[:, 128:256] = 0
    ref[0:10, :] *= 2
    np.testing.assert_allclose(after, ref, rtol=1e-2, atol=1e-6)
    be.set_tensor_ops(name, [])


def test_noise_is_seeded_and_scaled(engine):
    be = engine.backend
    name = "model.layers.5.mlp.up_proj.weight"
    before = be.read_tensor(name)
    op = SliceOp(name, 0, ((0, 512),), "noise", 0.5, seed=11)
    be.set_tensor_ops(name, [op])
    a = be.read_tensor(name)
    be.set_tensor_ops(name, [op])
    b = be.read_tensor(name)
    be.set_tensor_ops(name, [])
    assert np.array_equal(a, b)
    assert np.array_equal(a[512:], before[512:])
    ratio = np.std(a[:512] - before[:512]) / np.sqrt(np.mean(before[:512] ** 2))
    assert 0.4 < ratio < 0.6


def test_gain_norm_scale_zero_silences(engine):
    be = engine.backend
    name = "model.layers.1.post_attention_layernorm.weight"
    be.set_tensor_ops(name, [SliceOp(name, None, (), "scale", 0.0)])
    np.testing.assert_allclose(be.read_tensor(name), -1.0, atol=1e-6)  # gain 1 + w == 0
    be.set_tensor_ops(name, [])


def test_neuron_off_zeroes_its_activation(fresh, prompt_ids):
    rec_spec = RecordSpec(start=0)
    _, healthy = fresh.backend.forward(prompt_ids, fresh.backend.new_cache(), rec_spec)
    n = int(np.argmax(np.abs(healthy.mlp[:, 6, :]).mean(0)))
    fresh.add_edit(Target("neuron", layer=6, index=n), value=0.0)
    _, broken = fresh.backend.forward(prompt_ids, fresh.backend.new_cache(), rec_spec)
    assert np.all(broken.mlp[:, 6, n] == 0)
    fresh.update_edit(fresh.edits[0].id, value=0.5)
    _, half = fresh.backend.forward(prompt_ids, fresh.backend.new_cache(), rec_spec)
    np.testing.assert_allclose(half.mlp[:, 6, n], healthy.mlp[:, 6, n] * 0.5, rtol=0.05, atol=1e-3)


def test_head_off_zeroes_its_contribution(fresh, prompt_ids):
    fresh.add_edit(Target("head", layer=3, index=5), value=0.0)
    fresh.add_edit(Target("head", layer=8, index=2), value=0.0)
    _, rec = fresh.backend.forward(prompt_ids, fresh.backend.new_cache(), RecordSpec(start=0))
    assert np.all(rec.heads[:, 3, 5] < 1e-3)
    assert np.all(rec.heads[:, 8, 2] < 1e-3)
    assert rec.heads[:, 3, 4].mean() > 1e-2


def test_layer_off_passes_residual_through(fresh, prompt_ids):
    fresh.add_edit(Target("layer", layer=10), value=0.0)
    _, rec = fresh.backend.forward(prompt_ids, fresh.backend.new_cache(), RecordSpec(start=0))
    assert np.all(rec.mixer_out[:, 10] < 1e-3) and np.all(rec.mlp_out[:, 10] < 1e-3)
    np.testing.assert_allclose(rec.resid_vec[:, 10], rec.resid_vec[:, 9], atol=1e-2)
