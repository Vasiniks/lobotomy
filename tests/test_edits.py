"""Edit resolution and the edit stack. No model needed."""

import pytest

from lobotomy.config import ModelConfig
from lobotomy.edits import Edit, EditStack, Target, is_gain_norm, resolve

CFG = ModelConfig(
    n_layers=24,
    hidden_size=1024,
    intermediate_size=3584,
    vocab_size=248320,
    rms_norm_eps=1e-6,
    n_heads=8,
    n_kv_heads=2,
    head_dim=256,
    rope_theta=1e7,
    rope_dim=64,
    linear_n_k_heads=16,
    linear_n_v_heads=16,
    linear_k_head_dim=128,
    linear_v_head_dim=128,
    linear_conv_kernel=4,
    full_attention_interval=4,
)


def ops(target, **kw):
    return resolve(Edit(id=1, target=target, **kw), CFG)


def test_layer_types():
    assert CFG.attention_layers == [3, 7, 11, 15, 19, 23]
    assert CFG.n_layer_heads(3) == 8
    assert CFG.n_layer_heads(0) == 16


def test_neuron_maps_to_up_proj_row():
    (o,) = ops(Target("neuron", layer=5, index=812), value=0.0)
    assert o.tensor == "model.layers.5.mlp.up_proj.weight"
    assert o.axis == 0 and o.indices == ((812, 813),)
    assert o.op == "scale" and o.value == 0.0


def test_attention_head_maps_to_o_proj_columns():
    (o,) = ops(Target("head", layer=3, index=2))
    assert o.tensor == "model.layers.3.self_attn.o_proj.weight"
    assert o.axis == 1 and o.indices == ((512, 768),)


def test_deltanet_head_maps_to_out_proj_columns():
    (o,) = ops(Target("head", layer=4, index=15))
    assert o.tensor == "model.layers.4.linear_attn.out_proj.weight"
    assert o.indices == ((15 * 128, 16 * 128),)


def test_layer_parts():
    assert {o.tensor for o in ops(Target("layer", layer=7))} == {
        "model.layers.7.self_attn.o_proj.weight",
        "model.layers.7.mlp.down_proj.weight",
    }
    assert [o.tensor for o in ops(Target("layer", layer=0, part="mixer"))] == ["model.layers.0.linear_attn.out_proj.weight"]
    assert [o.tensor for o in ops(Target("layer", layer=0, part="mlp"))] == ["model.layers.0.mlp.down_proj.weight"]


def test_token_maps_to_embedding_row():
    (o,) = ops(Target("token", index=42))
    assert o.tensor == "model.embed_tokens.weight" and o.indices == ((42, 43),)


@pytest.mark.parametrize(
    "target",
    [
        Target("neuron", layer=24, index=0),
        Target("neuron", layer=0, index=3584),
        Target("head", layer=3, index=8),
        Target("head", layer=0, index=16),
        Target("layer", layer=0, part="nope"),
        Target("token", index=248320),
        Target("weight", tensor="x", rows=(0, 1), cols=(0, 1)),
    ],
)
def test_invalid_targets_raise(target):
    with pytest.raises(ValueError):
        ops(target)


def test_stack_add_update_remove_keeps_order_and_seed():
    s = EditStack(CFG)
    a = s.add(Target("neuron", layer=1, index=1))
    b = s.add(Target("neuron", layer=1, index=2), op="noise", value=0.3)
    s.update(a.id, value=0.5)
    assert [e.id for e in s.edits] == [a.id, b.id]
    assert s.get(a.id).value == 0.5 and s.get(a.id).seed == a.seed
    s.remove(a.id)
    assert [e.id for e in s.edits] == [b.id]
    assert s.ops_by_tensor()["model.layers.1.mlp.up_proj.weight"][0].op == "noise"


def test_invalid_add_does_not_enter_stack():
    s = EditStack(CFG)
    with pytest.raises(ValueError):
        s.add(Target("head", layer=3, index=99))
    assert s.edits == []


def test_gain_norms():
    assert is_gain_norm("model.layers.0.input_layernorm.weight")
    assert is_gain_norm("model.norm.weight")
    assert not is_gain_norm("model.layers.0.linear_attn.norm.weight")


def test_labels():
    assert Edit(1, Target("neuron", layer=2, index=9), value=0).label() == "L2 neuron 9: off"
    assert "boost" in Edit(1, Target("head", layer=3, index=1), value=2).label()
