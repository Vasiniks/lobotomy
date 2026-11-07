"""The MLX model against mlx-lm's own Qwen3.5 implementation.

Needs the official checkpoint in models/qwen3.5-0.8b-hf (see README) and mlx-lm.
"""

import numpy as np
import pytest

from conftest import HF_DIR, needs_model

pytestmark = [
    needs_model,
    pytest.mark.skipif(not list(HF_DIR.glob("*.safetensors")), reason="official checkpoint not downloaded"),
]


@pytest.fixture(scope="module")
def ref(tokenizer):
    mx = pytest.importorskip("mlx.core")
    mlx_lm = pytest.importorskip("mlx_lm")
    model, _ = mlx_lm.load(str(HF_DIR))
    text = tokenizer.render_chat([{"role": "user", "content": "Explain why the sky is blue in two sentences."}])
    ids = tokenizer.encode(text)
    logits = model(mx.array([ids]))[0].astype(mx.float32)
    mx.eval(logits)
    return ids, logits


def _kl(mx, p, q):
    lp = p - mx.logsumexp(p, -1, keepdims=True)
    lq = q - mx.logsumexp(q, -1, keepdims=True)
    return mx.sum(mx.exp(lp) * (lp - lq), -1)


def test_forward_pass_matches_mlx_lm_on_official_weights(ref, config):
    mx = pytest.importorskip("mlx.core")
    from lobotomy.backends.mlx.model import Qwen35

    ids, ref_logits = ref
    raw = {}
    for f in HF_DIR.glob("*.safetensors"):
        raw.update(mx.load(str(f)))
    w = {k.replace("model.language_model.", "model."): v for k, v in raw.items() if k.startswith("model.language_model.")}
    w = {k: (v.astype(mx.float32) if k.endswith(("A_log", "dt_bias")) else v) for k, v in w.items()}
    m = Qwen35(config, w)
    ours = m.logits(m.hidden(mx.array([ids]), m.new_cache()))[0].astype(mx.float32)
    kl = _kl(mx, ref_logits, ours)
    assert kl.mean().item() < 2e-3
    assert np.array_equal(np.array(mx.argmax(ours, -1)), np.array(mx.argmax(ref_logits, -1)))


def test_q4_gguf_is_close_to_official(ref, engine):
    mx = pytest.importorskip("mlx.core")
    if engine.backend.name != "mlx":
        pytest.skip("mlx only")
    ids, ref_logits = ref
    ours = mx.array(engine.backend.logits_for(ids))
    kl = _kl(mx, ref_logits, ours)
    assert kl.mean().item() < 0.06  # typical Q4_K_M loss is ~0.02-0.03
