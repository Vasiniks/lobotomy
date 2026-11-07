"""Local web server: a JSON API over the engine, plus the web UI.

All model work runs on one worker thread, so the backend is never used from two
threads at once. Chat replies stream as newline-delimited JSON (one event per
line), which the browser reads with ``fetch``.

Run with ``lobotomy serve`` and open http://127.0.0.1:7860.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import analysis as A
from .edits import Target
from .engine import Engine, Reply
from .sampling import SamplingParams

WEB_DIR = Path(__file__).resolve().parent / "web"


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd)) if np.isfinite(x) else None


def _rl(arr, nd: int = 4) -> list:
    return [_r(v, nd) for v in np.asarray(arr, dtype=np.float64).ravel()]


def _finite(obj):
    """Replace NaN/inf (which JSON can't carry) with None, recursively. A badly
    broken model can produce them."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.generic):
        return _finite(obj.item())
    return obj


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_finite(content), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class Api:
    def __init__(self, engine: Engine):
        self.eng = engine
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lobotomy-model")
        self.loop: asyncio.AbstractEventLoop | None = None

    async def run(self, fn, *args, **kw):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.worker, lambda: fn(*args, **kw))

    # --- serializers ------------------------------------------------------------

    def reply_summary(self, r: Reply) -> dict:
        return {
            "id": r.id,
            "exchange_id": r.exchange_id,
            "text": r.text,
            "healthy": r.healthy,
            "edits": r.edits,
            "finish_reason": r.finish_reason,
            "kept_memory": r.kept_memory,
            "stats": r.stats,
            "sampling": r.sampling,
            "created": r.created,
            "n_tokens": len(r.token_ids),
        }

    def reply_detail(self, r: Reply) -> dict:
        tok = self.eng.tokenizer
        d = self.reply_summary(r)
        toks = r.recorded_tokens(tok)
        n_prompt = sum(1 for t in toks if t["role"] == "prompt")
        for t in toks:
            if t["role"] == "reply":
                i = t["position"] - r.prompt_len
                t["prob"] = _r(r.token_probs[i])
                t["alternatives"] = [{"text": tok.token_str(a), "prob": _r(p)} for a, p in r.alternatives[i]]
        d["tokens"] = toks
        d["n_prompt_rows"] = n_prompt
        rec = r.recording
        if rec is not None:
            # (rows, layers) activity matrices for heat strips
            d["activity"] = {
                "mlp_out": [_rl(row, 3) for row in rec.mlp_out],
                "mixer_out": [_rl(row, 3) for row in rec.mixer_out],
                "neuron_mean": [_rl(row, 4) for row in neuron_mean(r)],
            }
        return d

    def state(self) -> dict:
        e = self.eng
        return {
            "model": e.info(),
            "busy": e.busy,
            "edits": [x.to_dict() for x in e.edits],
            "exchanges": [
                {"id": ex.id, "user": ex.user, "replies": [self.reply_summary(r) for r in ex.replies]}
                for ex in e.exchanges
            ],
            "settings": self.settings(),
        }

    def settings(self) -> dict:
        e = self.eng
        return {
            **e.sampling.to_dict(),
            "enable_thinking": e.enable_thinking,
            "system_prompt": e.system_prompt,
            "reread": e.reread,
        }

    def full(self, reply_id: int) -> Reply:
        """A reply that still has its full recording (older ones are trimmed)."""
        r = self.find(reply_id)
        if r.recording is None:
            raise HTTPException(404, "no recording")
        if r.cache.get("trimmed"):
            raise HTTPException(
                410,
                f"this answer's full recording was dropped to save memory (only the {self.eng.keep_recordings} newest keep it)",
            )
        return r

    def find(self, reply_id: int) -> Reply:
        try:
            return self.eng.find_reply(reply_id)
        except KeyError:
            raise HTTPException(404, f"no reply {reply_id}")

    @staticmethod
    def rows_for(r: Reply, rows: str | None, tokenizer) -> list[int] | None:
        """rows: 'reply' (default), 'all', or comma-separated row indices.
        None means every recorded row; an empty reply selects no rows."""
        if rows == "all" or r.recording is None:
            return None
        n = r.recording.n_positions
        if rows in (None, "", "reply"):
            return [i for i in range(n) if int(r.recording.positions[i]) >= r.prompt_len]
        try:
            sel = [int(x) for x in rows.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, f"bad rows {rows!r}")
        if any(not 0 <= i < n for i in sel):
            raise HTTPException(400, f"rows must be in 0..{n - 1}")
        return sel

    # --- streaming --------------------------------------------------------------

    def stream(self, make_events) -> StreamingResponse:
        """Run an engine event generator on the worker thread; stream NDJSON."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        done = object()

        def work():
            try:
                for ev in make_events():
                    if ev["type"] == "done":
                        ev = {"type": "done", "reply": self.reply_summary(ev["reply"])}
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as exc:  # report, don't kill the worker
                loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "message": str(exc)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, done)

        self.worker.submit(work)

        async def body():
            while True:
                ev = await q.get()
                if ev is done:
                    return
                yield json.dumps(_finite(ev)) + "\n"

        return StreamingResponse(body(), media_type="application/x-ndjson")


def neuron_mean(r: Reply) -> np.ndarray:
    """(rows, layers) mean |activation|, cached on the reply (survives trimming)."""
    if "neuron_mean" not in r.cache:
        r.cache["neuron_mean"] = np.abs(r.recording.mlp.astype(np.float32)).mean(-1)
    return r.cache["neuron_mean"]


def _sampling(eng: Engine, overrides: dict | None) -> SamplingParams | None:
    if not overrides:
        return None
    base = eng.sampling.to_dict()
    base.update({k: v for k, v in overrides.items() if k in base and v is not None})
    return SamplingParams(**base)


def create_app(engine: Engine) -> FastAPI:
    api = Api(engine)
    app = FastAPI(
        title="lobotomy", docs_url="/api/docs", openapi_url="/api/openapi.json", default_response_class=SafeJSONResponse
    )
    eng = engine

    # --- state & settings --------------------------------------------------------

    @app.get("/api/state")
    def get_state():
        return api.state()

    @app.get("/api/settings")
    def get_settings():
        return api.settings()

    @app.put("/api/settings")
    def put_settings(body: dict = Body(...)):
        s = eng.sampling.to_dict()
        for k in s:
            if k in body and body[k] is not None:
                s[k] = type(s[k])(body[k])
        eng.sampling = SamplingParams(**s)
        if "enable_thinking" in body:
            eng.enable_thinking = bool(body["enable_thinking"])
        if "system_prompt" in body:
            eng.system_prompt = body["system_prompt"] or None
        if "reread" in body:
            eng.reread = bool(body["reread"])
        return api.settings()

    # --- chat ----------------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(body: dict = Body(...)):
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "empty message")
        if eng.busy:
            raise HTTPException(409, "the model is busy")
        sampling = _sampling(eng, body.get("sampling"))
        return api.stream(lambda: eng.stream_chat(text, sampling))

    @app.post("/api/again")
    async def again(body: dict = Body(default={})):
        if not eng.exchanges:
            raise HTTPException(400, "nothing to ask again yet")
        if eng.busy:
            raise HTTPException(409, "the model is busy")
        sampling = _sampling(eng, body.get("sampling"))
        healthy = bool(body.get("healthy", False))
        return api.stream(lambda: eng.stream_again(sampling, healthy=healthy))

    @app.post("/api/cancel")
    def cancel():
        eng.cancel()
        return {"ok": True}

    @app.post("/api/chat/reset")
    async def reset_chat():
        await api.run(eng.reset_chat)
        return api.state()

    # --- edits -----------------------------------------------------------------------

    @app.get("/api/edits")
    def list_edits():
        return [e.to_dict() for e in eng.edits]

    @app.post("/api/edits")
    async def add_edit(body: dict = Body(...)):
        try:
            t = dict(body["target"])
            for k in ("rows", "cols"):
                if t.get(k) is not None:
                    t[k] = tuple(t[k])
            e = await api.run(
                eng.add_edit, Target(**t), op=body.get("op", "scale"), value=float(body.get("value", 0.0)), note=body.get("note", "")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {"edit": e.to_dict(), "edits": [x.to_dict() for x in eng.edits]}

    @app.post("/api/edits/batch")
    async def add_edits(body: dict = Body(...)):
        """Add several edits at once: {"edits": [{target, op, value}, ...]}."""
        def work():
            out = []
            for d in body.get("edits", []):
                t = dict(d["target"])
                out.append(
                    eng.add_edit(Target(**t), op=d.get("op", "scale"), value=float(d.get("value", 0.0)), note=d.get("note", ""))
                )
            return out
        try:
            added = await api.run(work)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {"added": [e.to_dict() for e in added], "edits": [x.to_dict() for x in eng.edits]}

    @app.patch("/api/edits/{edit_id}")
    async def update_edit(edit_id: int, body: dict = Body(...)):
        try:
            e = await api.run(eng.update_edit, edit_id, value=body.get("value"), op=body.get("op"))
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"edit": e.to_dict(), "edits": [x.to_dict() for x in eng.edits]}

    @app.delete("/api/edits/{edit_id}")
    async def delete_edit(edit_id: int):
        try:
            await api.run(eng.undo, edit_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        return {"edits": [x.to_dict() for x in eng.edits]}

    @app.delete("/api/edits")
    async def reset_edits():
        await api.run(eng.reset_edits)
        return {"edits": []}

    @app.get("/api/presets")
    def list_presets():
        from .presets import presets

        return presets(eng.config)

    @app.post("/api/edits/load")
    async def load_edits(body: dict = Body(...)):
        try:
            await api.run(eng.load_edits, body.get("edits", []), replace_existing=body.get("replace", True))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {"edits": [x.to_dict() for x in eng.edits]}

    # --- looking inside ----------------------------------------------------------------

    @app.get("/api/replies/{reply_id}")
    def reply(reply_id: int):
        return api.reply_detail(api.find(reply_id))

    @app.get("/api/replies/{reply_id}/layers")
    def layers(reply_id: int, rows: str | None = Query(None)):
        r = api.find(reply_id)
        if r.recording is None:
            raise HTTPException(404, "no recording")
        if r.cache.get("trimmed"):
            if rows not in (None, "", "reply"):
                api.full(reply_id)  # raises 410
            sel = None
            act = [dict(a) for a in r.cache["layers:reply"]]
        else:
            sel = api.rows_for(r, rows, eng.tokenizer)
            act = A.layer_activity(r.recording, sel)
        for a in act:
            a["type"] = eng.config.layer_type(a["layer"])
            a["heads"] = _rl(a["heads"])
        return {"rows": sel, "layers": act}

    @app.get("/api/replies/{reply_id}/layer/{layer}")
    def layer(reply_id: int, layer: int, rows: str | None = Query(None), k: int = 40, reduce: str = "mean"):
        r = api.full(reply_id)
        if r.recording is None:
            raise HTTPException(404, "no recording")
        if not 0 <= layer < eng.config.n_layers:
            raise HTTPException(400, "bad layer")
        sel = api.rows_for(r, rows, eng.tokenizer)
        grid = A.neuron_grid(r.recording, layer, sel, reduce=reduce)
        signed = A.signed_neuron_values(r.recording, layer, sel)
        act = A.layer_activity(r.recording, sel)[layer]
        top = A.top_neurons(r.recording, layer, k=k, rows=sel, reduce=reduce)
        out = {
            "layer": layer,
            "type": eng.config.layer_type(layer),
            "n_heads": eng.config.n_layer_heads(layer),
            "grid": _rl(grid),
            "signed": _rl(signed),
            "n_above": int((grid > 1.0).sum()),
            "reference": _r(r.layer_refs()[layer]),
            "peak": top[0] if top else None,
            "top": top,
            "heads": _rl(act["heads"]),
            "mixer_out": _r(act["mixer_out"]),
            "mlp_out": _r(act["mlp_out"]),
            # per-row activity of this layer, for the token strip
            "per_token": _rl(np.abs(r.recording.mlp[:, layer, :].astype(np.float32)).mean(-1)),
            "head_per_token": [_rl(row) for row in r.recording.heads[:, layer, : eng.config.n_layer_heads(layer)]],
        }
        return out

    @app.get("/api/replies/{reply_id}/neuron/{layer}/{neuron}")
    async def neuron(reply_id: int, layer: int, neuron: int):
        r = api.full(reply_id)
        if r.recording is None:
            raise HTTPException(404, "no recording")
        trace = A.neuron_trace(r.recording, layer, neuron)
        says = await api.run(eng.neuron_vocab, layer, neuron, 8)
        edits = [e.to_dict() for e in eng.edits if e.target.kind == "neuron" and e.target.layer == layer and e.target.index == neuron]
        return {"layer": layer, "neuron": neuron, "trace": _rl(trace), "says": says, "edits": edits}

    @app.get("/api/replies/{reply_id}/token/{row}")
    async def token(reply_id: int, row: int, k: int = 8):
        r = api.full(reply_id)
        rec = r.recording
        if rec is None or not 0 <= row < rec.n_positions:
            raise HTTPException(404, "no such row")
        toks = r.recorded_tokens(eng.tokenizer)
        lens = await api.run(eng.logit_lens, r, row, 3)
        heads = rec.heads[row]
        all_ids = r.prompt_ids + r.token_ids
        pos0 = int(rec.positions[0])
        here = int(rec.positions[row])
        im_start = eng.tokenizer.token_id("<|im_start|>")
        header = max((i for i, t in enumerate(r.prompt_ids) if t == im_start), default=r.prompt_len)

        def where(pos: int) -> str:
            if pos == here:
                return "this word"
            if pos == 0:
                return "start of chat"
            if pos < r.record_start:
                return "earlier in the chat"
            if pos < header:
                return "in your question"
            if pos < r.prompt_len:
                return "reply header"
            return "earlier in this reply"

        def entry(i: int, w: float) -> dict:
            return {
                "position": i,
                "row": i - pos0 if i >= pos0 else None,
                "text": eng.tokenizer.token_str(all_ids[i]),
                "weight": _r(w),
                "where": where(i),
            }

        attention = {}
        for layer, probs in rec.attention.items():
            p = probs[row].astype(np.float32)  # (H, K)
            per_head = []
            for hp in p:
                top = np.argsort(-hp)[:5]
                items = [entry(int(i), float(hp[i])) for i in top if hp[i] > 0]
                per_head.append({"top": items, "rest": _r(max(0.0, 1.0 - sum(x["weight"] for x in items)))})
            mean = p.mean(0)
            top = np.argsort(-mean)[:5]
            attention[str(layer)] = {
                "heads": per_head,
                "mean": {"top": [entry(int(i), float(mean[i])) for i in top if mean[i] > 0]},
            }
        prob = alternatives = None
        if toks[row]["role"] == "reply":
            i = toks[row]["position"] - r.prompt_len
            prob = _r(r.token_probs[i])
            alternatives = [{"text": eng.tokenizer.token_str(a), "prob": _r(p)} for a, p in r.alternatives[i]]
        return {
            "row": row,
            "token": toks[row],
            "index_in_reply": toks[row]["position"] - r.prompt_len if toks[row]["role"] == "reply" else None,
            "n_reply_tokens": len(r.token_ids),
            "prob": prob,
            "alternatives": alternatives,
            "ranked": A.top_neurons_ranked(rec, row, k=12, refs=r.layer_refs()),
            "neurons": A.top_neurons_at(rec, row, k=k),
            "layer_mlp": _rl(np.abs(rec.mlp[row].astype(np.float32)).mean(-1)),
            "mlp_out": _rl(rec.mlp_out[row]),
            "mixer_out": _rl(rec.mixer_out[row]),
            "heads": [_rl(heads[i, : eng.config.n_layer_heads(i)]) for i in range(eng.config.n_layers)],
            "lens": [[{"text": t, "prob": _r(p)} for t, p in layer] for layer in lens],
            "attention": attention,
        }

    @app.get("/api/replies/{reply_id}/damage")
    async def damage(reply_id: int):
        r = api.find(reply_id)
        try:
            d = await api.run(eng.damage, r)
        except NotImplementedError:
            raise HTTPException(501, "this backend can't measure damage")
        d["kl"] = _rl(d["kl"])
        for k in ("logp_current", "logp_healthy"):
            if k in d:
                d[k] = _rl(d[k])
        return d

    @app.get("/api/replies/{reply_id}/scan")
    async def scan(
        reply_id: int,
        kind: str = "layers",
        layer: int | None = None,
        part: str = "all",
        value: float = 0.0,
        k: int = 32,
        rows: str | None = Query(None),
    ):
        """Ablation scan: the KL from the current model if each part were cut."""
        r = api.full(reply_id) if kind == "neurons" else api.find(reply_id)
        candidates = None
        if kind in ("heads", "neurons") and (layer is None or not 0 <= layer < eng.config.n_layers):
            raise HTTPException(400, "this scan needs a layer")
        if kind == "neurons":
            if r.recording is None:
                raise HTTPException(404, "no recording")
            sel = api.rows_for(r, rows, eng.tokenizer)
            candidates = [n["neuron"] for n in A.top_neurons(r.recording, layer, k=k, rows=sel, reduce="max")]
        try:
            res = await api.run(eng.scan, r, kind=kind, layer=layer, candidates=candidates, value=value, part=part)
        except NotImplementedError:
            raise HTTPException(501, "this backend can't scan")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        for x in res:
            x["mean_kl"] = _r(x["mean_kl"])
            x["top1_agree"] = _r(x["top1_agree"], 3)
        return {"kind": kind, "layer": layer, "value": value, "results": res}

    @app.get("/api/replies/{reply_id}/why/{row}")
    async def why(reply_id: int, row: int, k: int = 12):
        """Which neurons and heads pushed the model toward this token (gradient estimate)."""
        r = api.find(reply_id)
        if r.recording is None or not 0 <= row < r.recording.n_positions:
            raise HTTPException(404, "no such row")
        try:
            w = await api.run(eng.why, r, row, k)
        except NotImplementedError:
            raise HTTPException(501, "this backend can't compute attributions")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        for group in ("support", "oppose"):
            for u in w[group]:
                u["effect"] = _r(u["effect"])
                u["label"] = f"L{u['layer']} {'N' if u['kind'] == 'neuron' else 'H'}{u['index']}"
        w["logp"] = _r(w["logp"])
        w["prob"] = _r(w["prob"])
        w["layer_effect"] = _rl(w["layer_effect"])
        return w

    @app.get("/api/neuron/{layer}/{neuron}/says")
    async def neuron_says(layer: int, neuron: int, k: int = 10):
        return await api.run(eng.neuron_vocab, layer, neuron, k)

    @app.get("/api/tokenize")
    def tokenize(text: str):
        ids = eng.tokenizer.encode(text)
        return [{"id": i, "text": eng.tokenizer.token_str(i)} for i in ids]

    # --- UI ---------------------------------------------------------------------------

    if WEB_DIR.exists():

        @app.middleware("http")
        async def revalidate_assets(request, call_next):
            # Local app: always revalidate (cheap ETag check), so edits to the UI
            # never leave a stale module in the browser cache.
            response = await call_next(request)
            if request.url.path.startswith("/assets") or request.url.path == "/":
                response.headers["Cache-Control"] = "no-cache"
            return response

        app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

    return app


def serve(engine: Engine, host: str = "127.0.0.1", port: int = 7860) -> None:
    import uvicorn

    app = create_app(engine)
    print(f"lobotomy is running at http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
