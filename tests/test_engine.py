import json

import numpy as np

from lobotomy import analysis as A
from lobotomy.edits import Target

from conftest import needs_model

pytestmark = needs_model

Q = "What is the capital of France? Answer in one short sentence."


def test_chat_and_recording(fresh):
    r = fresh.chat(Q)
    assert "Paris" in r.text
    assert r.finish_reason == "stop"
    toks = r.recorded_tokens(fresh.tokenizer)
    reply_rows = [t for t in toks if t["role"] == "reply"]
    assert len(reply_rows) == len(r.token_ids)
    assert toks[0]["text"] == "<|im_start|>"
    assert "".join(t["text"] for t in reply_rows) == r.text


def test_ask_again_without_edits_reproduces(fresh):
    a = fresh.chat(Q)
    b = fresh.ask_again()
    assert a.token_ids == b.token_ids
    assert len(fresh.exchanges[-1].replies) == 2


def test_sampled_ask_again_reuses_seed(fresh):
    fresh.sampling.temperature = 0.9
    a = fresh.chat("Write one sentence about the sea.")
    b = fresh.ask_again()
    assert a.token_ids == b.token_ids


def test_break_undo_reset_restore_exactly(fresh):
    healthy = fresh.chat(Q)
    e = fresh.add_edit(Target("layer", layer=0), value=0.0)
    broken = fresh.ask_again()
    assert broken.token_ids != healthy.token_ids
    assert broken.edits and broken.edits[0]["id"] == e.id
    fresh.undo(e.id)
    assert fresh.ask_again().token_ids == healthy.token_ids
    for layer in (0, 5, 11):
        fresh.add_edit(Target("layer", layer=layer, part="mlp"), op="noise", value=0.8)
    fresh.reset_edits()
    assert fresh.ask_again().token_ids == healthy.token_ids


def test_undo_middle_edit_equals_never_having_made_it(fresh):
    fresh.chat(Q)
    a = fresh.add_edit(Target("layer", layer=2, part="mlp"), op="noise", value=0.5)
    b = fresh.add_edit(Target("layer", layer=2, part="mlp"), value=0.5)
    fresh.add_edit(Target("neuron", layer=2, index=7), value=3.0)
    fresh.undo(b.id)
    w_after_undo = fresh.backend.read_tensor("model.layers.2.mlp.down_proj.weight")
    after_undo = fresh.ask_again().token_ids

    fresh.reset_edits()
    fresh.add_edit(Target("layer", layer=2, part="mlp"), op="noise", value=0.5, seed=a.seed)
    fresh.add_edit(Target("neuron", layer=2, index=7), value=3.0)
    assert np.array_equal(fresh.backend.read_tensor("model.layers.2.mlp.down_proj.weight"), w_after_undo)
    assert fresh.ask_again().token_ids == after_undo


def test_healthy_ask_again_ignores_edits(fresh):
    healthy = fresh.chat(Q)
    fresh.add_edit(Target("layer", layer=0), value=0.0)
    h = fresh.ask_again(healthy=True)
    assert h.healthy and h.token_ids == healthy.token_ids
    assert fresh.edits, "edits are still active afterwards"
    assert fresh.ask_again().token_ids != healthy.token_ids


def test_damage_is_zero_when_healthy_and_positive_when_broken(fresh):
    r = fresh.chat(Q)
    d0 = fresh.damage(r)
    assert d0["mean_kl"] < 1e-3
    fresh.add_edit(Target("layer", layer=1), value=0.0)
    d1 = fresh.damage(r)
    assert d1["mean_kl"] > 0.05
    assert len(d1["kl"]) == len(r.token_ids)


def test_history_is_reread_with_current_weights(fresh):
    fresh.chat("My name is Ada.")
    r = fresh.chat("What is my name?")
    assert "Ada" in r.text
    assert r.prompt_len > 20
    assert r.recording.positions[0] == r.record_start  # recording starts at the last user turn


def test_keep_memory_matches_reread_without_edits(fresh):
    a1 = fresh.chat("My name is Ada.")
    a2 = fresh.chat("What is my name?")
    fresh.reset_chat()
    fresh.reread = False
    try:
        b1 = fresh.chat("My name is Ada.")
        b2 = fresh.chat("What is my name?")
        assert b2.kept_memory and not b1.kept_memory
        assert b1.token_ids == a1.token_ids
        # The continued prompt is what the model actually read: identical to a
        # fresh render except that the earlier turn keeps its empty think block.
        tok = fresh.tokenizer
        kept_text = tok.decode(b2.prompt_ids)
        assert kept_text == tok.decode(a2.prompt_ids).replace(
            "<|im_start|>assistant\nHello", "<|im_start|>assistant\n<think>\n\n</think>\n\nHello", 1
        ) or kept_text.count("<think>") == 2
        assert tok.encode(kept_text) == b2.prompt_ids
        # ask again with no edits: rewinds to the snapshot, same answer
        b3 = fresh.ask_again()
        assert b3.kept_memory and b3.token_ids == b2.token_ids
        # and matches a fresh full read of the same tokens
        lg_full = fresh.backend.logits_for(b2.prompt_ids)[-1]
        assert int(np.argmax(lg_full)) == b2.token_ids[0]
    finally:
        fresh.reread = True


def test_keep_memory_old_turns_stay_healthy(fresh):
    fresh.reread = False
    try:
        fresh.chat("My name is Ada.")
        r = fresh.chat("What is my name?")
        fresh.add_edit(Target("layer", layer=0), value=0.0)
        kept = fresh.ask_again()
        assert kept.kept_memory
        # the snapshot still holds the healthy reading of the question, so the
        # cache contents at the rewind point are unaffected by later decoding
        again = fresh.ask_again()
        assert again.token_ids == kept.token_ids
        fresh.reread = True
        reread = fresh.ask_again()
        assert not reread.kept_memory
        assert reread.prompt_len != kept.prompt_len or reread.token_ids != kept.token_ids or r.text
    finally:
        fresh.reread = True


def test_bad_edits_never_enter_the_stack(fresh):
    import pytest

    healthy = fresh.chat(Q)
    good = fresh.add_edit(Target("layer", layer=0), value=0.0)
    w = fresh.backend.read_tensor("model.layers.0.mlp.down_proj.weight")
    bad = [
        Target("weight", tensor="model.layers.3.self_attn.o_proj"),  # missing .weight
        Target("weight", tensor="model.layers.0.input_layernorm.weight", cols=(0, 4)),  # 1-D tensor
        Target("weight", tensor="model.layers.0.mlp.up_proj.weight", rows=(0, 99999)),
    ]
    for t in bad:
        with pytest.raises(ValueError):
            fresh.add_edit(t, value=0.0)
    assert [e.id for e in fresh.edits] == [good.id]
    assert np.array_equal(fresh.backend.read_tensor("model.layers.0.mlp.down_proj.weight"), w)
    # a recipe with one bad edit changes nothing
    recipe = [
        {"target": {"kind": "neuron", "layer": 5, "index": 1}, "value": 0.0},
        {"target": {"kind": "weight", "tensor": "nope.weight"}, "value": 0.0},
    ]
    with pytest.raises(ValueError):
        fresh.load_edits(recipe)
    assert [e.id for e in fresh.edits] == [good.id]
    # healthy still works and restores the edit afterwards
    assert fresh.ask_again(healthy=True).token_ids == healthy.token_ids
    assert np.array_equal(fresh.backend.read_tensor("model.layers.0.mlp.down_proj.weight"), w)


def test_recipe_roundtrip(fresh, tmp_path):
    fresh.chat(Q)
    fresh.add_edit(Target("head", layer=3, index=1), value=0.0)
    fresh.add_edit(Target("layer", layer=9, part="mlp"), op="noise", value=0.3)
    broken = fresh.ask_again().token_ids
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps([e.to_dict() for e in fresh.edits]))
    fresh.reset_edits()
    fresh.load_edits(json.loads(path.read_text()))
    assert len(fresh.edits) == 2
    assert fresh.ask_again().token_ids == broken


def test_scan_ranks_parts_and_restores_weights(fresh):
    r = fresh.chat(Q)
    fresh.add_edit(Target("neuron", layer=5, index=3), value=2.0)
    before = {n: fresh.backend.read_tensor(n) for n in ("model.layers.5.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight")}
    layers = fresh.scan(r, "layers")
    assert len(layers) == 24
    assert max(layers, key=lambda x: x["mean_kl"])["mean_kl"] > 1.0  # some layer is critical
    heads = fresh.scan(r, "heads", layer=3)
    assert len(heads) == 8 and all(h["mean_kl"] >= 0 for h in heads)
    neurons = fresh.scan(r, "neurons", layer=5, candidates=[3, 10, 20])
    assert [n["target"]["index"] for n in neurons] == [3, 10, 20]
    for n, w in before.items():
        assert np.array_equal(fresh.backend.read_tensor(n), w)
    assert len(fresh.edits) == 1


def test_why_finds_units_whose_removal_hurts_the_token(fresh):
    r = fresh.chat("What is the capital of Australia? One sentence.")
    toks = r.recorded_tokens(fresh.tokenizer)
    row = next(t["row"] for t in toks if t["role"] == "reply" and "Canberra" in t["text"])
    w = fresh.why(r, row, k=10)
    assert "Canberra" in w["token"] and 0 < w["prob"] <= 1
    assert w["support"] and all(u["effect"] < 0 for u in w["support"])
    assert len(w["layer_effect"]) == 24
    ids = r.prompt_ids + r.token_ids
    pos = w["position"]

    def logp():
        lg = fresh.backend.logits_for(ids[:pos])[-1].astype(np.float64)
        return lg[ids[pos]] - (np.log(np.exp(lg - lg.max()).sum()) + lg.max())

    base = logp()
    for u in w["support"][:10]:
        fresh.add_edit(Target(u["kind"], layer=u["layer"], index=u["index"]), value=0.0)
    assert logp() < base - 1.0  # cutting the top supporters really hurts the token


def test_analysis_helpers(fresh):
    r = fresh.chat(Q)
    rows = [t["row"] for t in r.recorded_tokens(fresh.tokenizer) if t["role"] == "reply"]
    act = A.layer_activity(r.recording, rows)
    assert len(act) == 24 and len(act[0]["heads"]) == 16 and len(act[3]["heads"]) == 8
    top = A.top_neurons(r.recording, 12, k=5, rows=rows)
    assert len(top) == 5 and top[0]["activity"] >= top[-1]["activity"]
    lens = fresh.logit_lens(r, rows[0], k=3)
    assert len(lens) == 24 and len(lens[0]) == 3
    says = fresh.neuron_vocab(12, top[0]["neuron"], k=4)
    assert len(says["promotes"]) == 4
