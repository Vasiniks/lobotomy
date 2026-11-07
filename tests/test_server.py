import json

import pytest

from conftest import needs_model

pytestmark = needs_model


@pytest.fixture
def client(fresh):
    from fastapi.testclient import TestClient

    from lobotomy.server import create_app

    with TestClient(create_app(fresh)) as c:
        yield c


def stream(client, path, body):
    events = []
    with client.stream("POST", path, json=body) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line:
                events.append(json.loads(line))
    return events


def test_state(client):
    s = client.get("/api/state").json()
    assert s["model"]["n_layers"] == 24
    assert s["model"]["layers"][3]["type"] == "attention"
    assert s["edits"] == [] and s["exchanges"] == []


def test_chat_edit_again_inspect(client):
    ev = stream(client, "/api/chat", {"text": "What is the capital of France? One sentence."})
    assert ev[0]["type"] == "start" and ev[-1]["type"] == "done"
    text = "".join(e["text"] for e in ev if e["type"] == "token")
    reply = ev[-1]["reply"]
    assert reply["text"] == text and "Paris" in text

    r = client.post("/api/edits", json={"target": {"kind": "layer", "layer": 0}, "value": 0})
    assert r.status_code == 200
    edit_id = r.json()["edit"]["id"]
    assert client.patch(f"/api/edits/{edit_id}", json={"value": 0.1}).json()["edit"]["value"] == 0.1

    ev2 = stream(client, "/api/again", {})
    broken = ev2[-1]["reply"]
    assert broken["edits"][0]["id"] == edit_id and broken["text"] != reply["text"]

    d = client.get(f"/api/replies/{reply['id']}").json()
    assert len(d["tokens"]) > 0 and len(d["activity"]["mlp_out"][0]) == 24
    reply_rows = [t["row"] for t in d["tokens"] if t["role"] == "reply"]
    assert all("alternatives" in d["tokens"][i] for i in reply_rows)

    layers = client.get(f"/api/replies/{reply['id']}/layers").json()["layers"]
    assert len(layers) == 24 and len(layers[0]["heads"]) == 16

    lay = client.get(f"/api/replies/{reply['id']}/layer/11").json()
    assert len(lay["grid"]) == 3584 and lay["n_heads"] == 8 and len(lay["top"]) == 40

    n = lay["top"][0]["neuron"]
    nr = client.get(f"/api/replies/{reply['id']}/neuron/11/{n}").json()
    assert len(nr["trace"]) == len(d["tokens"]) and nr["says"]["promotes"]

    tok = client.get(f"/api/replies/{reply['id']}/token/{reply_rows[0]}").json()
    assert len(tok["lens"]) == 24 and set(tok["attention"]) == {"3", "7", "11", "15", "19", "23"}
    assert len(tok["attention"]["3"]["heads"]) == 8 and tok["ranked"] and tok["alternatives"]

    dmg = client.get(f"/api/replies/{reply['id']}/damage").json()
    assert dmg["mean_kl"] > 0.01

    scan = client.get(f"/api/replies/{reply['id']}/scan", params={"kind": "neurons", "layer": 11, "k": 4}).json()
    assert len(scan["results"]) == 4
    assert client.get(f"/api/replies/{reply['id']}/scan", params={"kind": "heads"}).status_code == 400

    assert client.delete(f"/api/edits/{edit_id}").json()["edits"] == []
    healthy_again = stream(client, "/api/again", {})[-1]["reply"]
    assert healthy_again["text"] == reply["text"]


def test_bad_requests(client):
    assert client.post("/api/edits", json={"target": {"kind": "head", "layer": 3, "index": 99}}).status_code == 400
    assert client.post("/api/again", json={}).status_code == 400
    assert client.get("/api/replies/999").status_code == 404
    assert client.post("/api/chat", json={"text": "  "}).status_code == 400


def test_presets_load(client):
    presets = client.get("/api/presets").json()
    assert {p["id"] for p in presets} >= {"emoji-soup", "broken-record", "tipsy"}
    for p in presets:
        r = client.post("/api/edits/load", json={"edits": p["edits"]})
        assert r.status_code == 200 and len(r.json()["edits"]) == len(p["edits"])
    client.delete("/api/edits")


def test_batch_and_recipe(client):
    body = {"edits": [{"target": {"kind": "neuron", "layer": 2, "index": i}, "value": 0} for i in range(5)]}
    r = client.post("/api/edits/batch", json=body).json()
    assert len(r["added"]) == 5
    recipe = client.get("/api/edits").json()
    client.delete("/api/edits")
    assert len(client.post("/api/edits/load", json={"edits": recipe}).json()["edits"]) == 5
