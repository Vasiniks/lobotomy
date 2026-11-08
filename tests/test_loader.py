import numpy as np
import pytest

from conftest import needs_model

# hidden size -> MLP neurons per layer, for the Qwen3.5 sizes lobotomy supports
SIZES = {1024: 3584, 2048: 6144}  # 0.8B, 2B


@needs_model
def test_config(config):
    assert config.n_layers == 24
    assert config.hidden_size in SIZES and config.intermediate_size == SIZES[config.hidden_size]
    assert config.attention_layers == [3, 7, 11, 15, 19, 23]
    assert config.n_heads == 8 and config.n_kv_heads == 2 and config.head_dim == 256
    assert config.linear_n_v_heads == 16 and config.linear_v_head_dim == 128
    assert config.rope_dim == 64
    assert config.tie_word_embeddings
    assert 248046 in config.eos_token_ids  # <|im_end|>


@needs_model
def test_names_shapes_and_conventions(gguf_model, config):
    shapes = {}
    for name, arr in gguf_model.iter_weights(dtype=None):
        if ".layers." in name and not name.startswith(("model.layers.0.", "model.layers.3.")):
            continue
        shapes[name] = arr.shape
        if name.endswith("linear_attn.A_log"):
            assert np.all(np.isfinite(arr))
        if name.endswith("input_layernorm.weight"):
            # HF convention: gain = 1 + w, so w hovers near 0, not near 1
            assert abs(float(arr.mean())) < 1.0
    H, N = config.hidden_size, config.intermediate_size
    attn = config.n_heads * config.head_dim  # 2048
    qkv = 2 * config.linear_n_k_heads * config.linear_k_head_dim + config.linear_n_v_heads * config.linear_v_head_dim
    assert shapes["model.embed_tokens.weight"] == (248320, H)
    assert shapes["model.layers.3.self_attn.q_proj.weight"] == (2 * attn, H)  # queries and their output gate
    assert shapes["model.layers.3.self_attn.o_proj.weight"] == (H, attn)
    assert shapes["model.layers.0.linear_attn.in_proj_qkv.weight"] == (qkv, H)  # 6144 rows
    assert shapes["model.layers.0.linear_attn.conv1d.weight"] == (qkv, 1, 4)
    assert shapes["model.layers.0.linear_attn.out_proj.weight"] == (H, 2048)
    assert shapes["model.layers.0.mlp.down_proj.weight"] == (H, N)


@needs_model
def test_matches_official_checkpoint(gguf_model, official):
    mx = pytest.importorskip("mlx.core")
    ref = {}
    for f in official.glob("*.safetensors"):
        ref.update(mx.load(str(f)))
    ref = {k.replace("model.language_model.", "model."): v for k, v in ref.items() if k.startswith("model.language_model.")}
    ours = gguf_model.load_weights(dtype=np.float32)
    assert set(ours) == set(ref)
    for k, a in ours.items():
        b = np.array(ref[k].astype(mx.float32))
        assert a.shape == b.shape, k
        if a.size > 1:
            corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            assert corr > 0.99, (k, corr)
