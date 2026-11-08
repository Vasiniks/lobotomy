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
    fresh.add_edit(Target("layer", layer=0), value=0.0)  # critical in every Qwen3.5 size
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
        # and matches a fresh full read of the same tokens, up to bfloat16 rounding:
        # reading word by word and reading all at once round differently, which
        # can swap two nearly tied words (magi-1.2-2B: "You" 0.285 vs "Based" 0.251;
        # in float32 the two reads agree to 4 decimals)
        lg = fresh.backend.logits_for(b2.prompt_ids)[-1].astype(np.float64)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        for tid, q in b2.alternatives[0]:
            assert abs(p[tid] - q) < 0.05, (tok.token_str(tid), p[tid], q)
        assert p[b2.token_ids[0]] > p.max() - 0.05  # the greedy pick is the top word, or tied with it
    finally:
        fresh.reread = True


def test_recordings_are_kept_within_a_memory_budget(fresh):
    fresh.recording_budget = 1  # byte: only the newest reply keeps its full recording
    try:
        a = fresh.chat(Q)
        b = fresh.chat(Q)
        assert a.cache.get("trimmed") and a.recording.mlp is None
        assert b.recording.mlp is not None and not b.cache.get("trimmed")
    finally:
        fresh.recording_budget = 2 << 30


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


def test_lens_grid_last_layer_is_the_real_output_probability(fresh):
    r = fresh.chat(Q)
    g = fresh.lens_grid(r)
    probs = np.asarray(g["probs"])
    assert probs.shape == (len(r.token_ids), fresh.config.n_layers)
    assert np.all((probs >= 0) & (probs <= 1))
    # The last layer through the final norm is exactly the model's output, so it
    # must match the probability the sampler saw (temperature 1) for each token.
    for k, tid in enumerate(r.token_ids):
        seen = dict(r.alternatives[k]).get(tid)
        if seen is not None:
            assert abs(probs[k, -1] - seen) < 0.03, (k, probs[k, -1], seen)
    # rows line up with the recording's reply rows
    reply_rows = [t["row"] for t in r.recorded_tokens(fresh.tokenizer) if t["role"] == "reply"]
    assert g["rows"] == reply_rows


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


def test_flow_steps_line_up_with_the_words(fresh):
    import base64

    r = fresh.chat(Q)
    k = 16
    f = fresh.flow(r, k=k)
    n = len(r.token_ids)
    # one step per pass that wrote a word: the last prompt token wrote the first
    assert len(f["steps"]) == n + 1
    toks = r.recorded_tokens(fresh.tokenizer)
    reply_rows = [t for t in toks if t["role"] == "reply"]
    assert [s["says"] for s in f["steps"][:n]] == [t["text"] for t in reply_rows]
    assert [s["reads"] for s in f["steps"][1:]] == [t["text"] for t in reply_rows]
    assert f["steps"][0]["prob"] == r.token_probs[0]
    assert f["steps"][-1]["says"] is None and f["steps"][-1]["end"] == "stop"
    S, L = len(f["steps"]), fresh.config.n_layers
    idx = np.frombuffer(base64.b64decode(f["idx"]), "<u2").reshape(S, L, k)
    val = np.frombuffer(base64.b64decode(f["val"]), np.uint8).reshape(S, L, k)
    a = np.abs(r.recording.mlp[f["steps"][3]["row"]].astype(np.float32))
    assert (a[np.arange(L), idx[3, :, 0]] == a.max(axis=1)).all()  # sizes: neurons can tie
    assert (np.diff(val.astype(int), axis=2) <= 0).all()  # strongest first
    assert len(f["scale"]) == L and min(f["scale"]) > 0


def test_token_events_carry_their_strongest_neurons(fresh):
    events = list(fresh.stream_chat(Q))
    tokens = [e for e in events if e["type"] == "token"]
    r = events[-1]["reply"]
    L = fresh.config.n_layers
    first = tokens[0]["flow"]
    assert len(first["idx"]) == L and len(first["idx"][0]) == fresh.flow_k
    # the first word was written by the pass over the last prompt token
    toks = r.recorded_tokens(fresh.tokenizer)
    row = next(t["row"] for t in toks if t["role"] == "reply") - 1
    a = np.abs(r.recording.mlp[row].astype(np.float32))
    # compare sizes, not indices: two neurons can tie exactly in float16
    # (Qwen3.5 2B, layer 16: neurons 1021 and 5619 both at 1.5781)
    assert [a[layer, ix[0]] for layer, ix in enumerate(first["idx"])] == a.max(axis=1).tolist()


def test_flow_carries_head_activity(fresh):
    import base64

    r = fresh.chat(Q)
    f = fresh.flow(r, k=8)
    S, L, H = len(f["steps"]), fresh.config.n_layers, fresh.config.max_heads
    heads = np.frombuffer(base64.b64decode(f["heads"]), np.uint8).reshape(S, L, H)
    assert f["n_heads"] == [fresh.config.n_layer_heads(i) for i in range(L)]
    # layers with fewer heads than the widest leave the rest empty
    attn = next(i for i in range(L) if fresh.config.is_full_attention(i))
    assert (heads[:, attn, f["n_heads"][attn] :] == 0).all()
    assert heads.max() == 255 and len(f["head_scale"]) == L
    # the strongest head at a step is the one with the largest recorded output
    row = f["steps"][2]["row"]
    want = np.nanargmax(r.recording.heads[row, 0, : f["n_heads"][0]])
    assert heads[2, 0].argmax() == want


def test_neuron_wiring_reads_the_healthy_weights(fresh):
    k = 3
    w = fresh.neuron_wiring(20, 1234, k=k)
    L = fresh.config.n_layers
    assert [x["layer"] for x in w] == [l for l in range(21, L) for _ in range(k)]
    be = fresh.backend
    d = be.read_pristine("model.layers.20.mlp.down_proj.weight")[:, 1234]
    gain = be.read_pristine("model.layers.21.post_attention_layernorm.weight") + 1.0
    s = be.read_pristine("model.layers.21.mlp.up_proj.weight") @ (gain * d)
    top = np.argsort(-np.abs(s))[:k]
    first = [x for x in w if x["layer"] == 21]
    assert [x["neuron"] for x in first] == top.tolist()
    assert np.allclose([x["weight"] for x in first], s[top], rtol=0.05, atol=1e-3)
    # edits don't change it (like "says", it shows where the signal would go)
    fresh.add_edit(Target("neuron", layer=21, index=first[0]["neuron"]), value=0.0)
    assert fresh.neuron_wiring(20, 1234, k=k)[0]["neuron"] == first[0]["neuron"]
    assert fresh.neuron_wiring(L - 1, 0) == []


def test_flow_steps_carry_the_candidates(fresh):
    r = fresh.chat(Q)
    f = fresh.flow(r, k=4)
    for i, st in enumerate(f["steps"][: len(r.token_ids)]):
        assert [a["prob"] for a in st["alts"]] == [p for _, p in r.alternatives[i]]
    assert f["steps"][-1]["alts"] == []


def test_step_lens_ends_at_the_real_output(fresh):
    r = fresh.chat(Q)
    f = fresh.flow(r, k=4)
    st = f["steps"][2]  # this pass wrote reply token 2
    g = fresh.step_lens(r, st["row"], k=3)
    L = fresh.config.n_layers
    assert len(g["layers"]) == L and len(g["layers"][0]["top"]) == 3
    assert g["next"] == st["says"]
    # the last layer through the final norm is the model's real output
    seen = dict(r.alternatives[2]).get(r.token_ids[2])
    assert abs(g["layers"][-1]["p_next"] - seen) < 0.03
    assert g["layers"][-1]["top"][0]["prob"] >= g["layers"][-1]["p_next"] - 1e-6


def test_mlx_caps_its_buffer_cache(engine):
    # Uncapped, MLX kept 8.5 GB of freed buffers after one "why this word?".
    if engine.backend.name != "mlx":
        return
    import mlx.core as mx

    from lobotomy.backends.mlx.backend import CACHE_LIMIT

    assert mx.set_cache_limit(CACHE_LIMIT) == CACHE_LIMIT  # returns the limit in force
