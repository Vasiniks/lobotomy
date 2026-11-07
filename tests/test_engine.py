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
