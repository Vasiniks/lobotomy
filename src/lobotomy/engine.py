"""The engine: one model, one chat, a stack of edits.

Typical loop::

    eng = Engine("models/Qwen3.5-0.8B-Q4_K_M.gguf")
    r1 = eng.chat("What is the capital of France?")
    eng.add_edit(Target("layer", layer=11), value=0.0)   # knock out layer 11
    r2 = eng.ask_again()                                 # same question, same seed
    eng.undo(1)                                          # back to healthy

Each user message is an :class:`Exchange`. "Ask again" adds another
:class:`Reply` to the latest exchange, so answers to the same question can be
compared. The newest reply is the one the chat continues from.

By default every generation re-reads the whole chat with the current weights (a
fresh cache), so the model's memory of the conversation is as broken as the
model. With ``reread = False`` ("keep memory") the model keeps its memory of
earlier turns as it built it under the weights of the time, and only new tokens
use the current weights.
"""

from __future__ import annotations

import base64
import contextlib
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .analysis import top_by_layer
from .backends import Backend, RecordSpec, Recording, default_backend, get_backend
from .config import ModelConfig
from .edits import Edit, EditStack, Target, resolve
from .gguf_loader import open_gguf
from .sampling import Sampler, SamplingParams
from .tokenizer import ChatTokenizer


@dataclass
class Reply:
    id: int
    exchange_id: int
    text: str
    token_ids: list[int]
    token_texts: list[str]
    token_probs: list[float]
    alternatives: list[list[tuple[int, float]]]  # top-5 candidates before each token
    prompt_ids: list[int]
    record_start: int  # absolute position where the recording starts
    recording: Recording | None
    edits: list[dict]  # edits active when this reply was generated
    healthy: bool  # generated with every edit bypassed
    sampling: dict
    finish_reason: str  # "stop" | "length" | "cancelled"
    stats: dict
    kept_memory: bool = False  # continued from memory built under earlier weights
    created: float = field(default_factory=time.time)
    # derived data (token texts, layer ranges); a reply never changes once made
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    def recorded_tokens(self, tokenizer: ChatTokenizer) -> list[dict]:
        """Every recorded position with its token text and role (prompt or reply)."""
        if "tokens" in self.cache:
            return [dict(t) for t in self.cache["tokens"]]
        all_ids = self.prompt_ids + self.token_ids
        out = []
        if self.recording is None:
            return out
        positions = self.recording.positions.tolist()
        start = positions[0] if positions else 0
        # decode from the start of the recording (a special token, so a character
        # boundary) so characters split over several tokens come out whole
        texts = tokenizer.pieces(all_ids[start : positions[-1] + 1]) if positions else []
        for row, pos in enumerate(positions):
            tid = all_ids[pos]
            out.append(
                {
                    "row": row,
                    "position": pos,
                    "token_id": tid,
                    "text": texts[pos - start],
                    "role": "reply" if pos >= self.prompt_len else "prompt",
                }
            )
        self.cache["tokens"] = out
        return [dict(t) for t in out]

    def layer_refs(self):
        """Each layer's usual activation range over this reply's recording (cached)."""
        if "refs" not in self.cache:
            from .analysis import layer_references

            self.cache["refs"] = layer_references(self.recording)
        return self.cache["refs"]


@dataclass
class Exchange:
    id: int
    user: str
    replies: list[Reply] = field(default_factory=list)

    @property
    def active(self) -> Reply | None:
        return self.replies[-1] if self.replies else None


class Engine:
    def __init__(
        self,
        model_path: str | Path,
        backend: str | None = None,
        dtype: str = "bfloat16",
        system_prompt: str | None = None,
        on_progress: Callable[[str], None] | None = None,
        use_cache: bool = True,
    ):
        say = on_progress or (lambda msg: None)
        self.model_path = Path(model_path)
        say(f"reading {self.model_path.name}")
        g = open_gguf(self.model_path)
        self.gguf = g
        self.config: ModelConfig = g.config
        self.tokenizer = ChatTokenizer(g.tokenizer)
        name = backend or default_backend()
        self.backend = get_backend(name, dtype=dtype) if name == "mlx" else get_backend(name)
        t0 = time.time()
        self._load_weights(g, name, dtype, use_cache, say)
        self.load_seconds = time.time() - t0
        self.edit_stack = EditStack(self.config)
        self._shapes = {n: tuple(self.backend.tensor_shape(n)) for n in self.backend.tensor_names()}
        self.exchanges: list[Exchange] = []
        self.system_prompt = system_prompt
        self.sampling = SamplingParams()
        self.enable_thinking = False
        # True: every reply re-reads the whole chat with the current weights.
        # False ("keep memory"): earlier turns stay as the model read them under
        # the weights of the time; only new tokens use the current weights.
        self.reread = True
        # Full recordings (every neuron at every token) are ~100 MB per reply;
        # older replies keep only the small summaries the chat and compare views need.
        self.keep_recordings = 8
        # neurons per layer sent with every streamed token, for the live 3D view
        self.flow_k = 32
        self._mem: dict | None = None  # cache after the latest reply
        self._ask_snap: dict | None = None  # cache just before the latest prompt's last token
        self._lock = threading.RLock()
        self._ids = {"exchange": 1, "reply": 1}
        self._cancel = threading.Event()
        self.busy = False

    def _cache_path(self, dtype: str) -> Path:
        st = self.model_path.stat()
        tag = f"{self.model_path.stem}.{dtype}.{st.st_size}.{int(st.st_mtime)}"
        return self.model_path.parent / ".cache" / f"{tag}.safetensors"

    def _load_weights(self, g, name: str, dtype: str, use_cache: bool, say) -> None:
        can_cache = use_cache and hasattr(self.backend, "load_cache")
        cache = self._cache_path(dtype) if can_cache else None
        if cache is not None and cache.exists():
            say("loading cached weights")
            try:
                self.backend.load_cache(cache, self.config)
                return
            except Exception as exc:  # corrupt or stale cache: rebuild it
                say(f"cache unusable ({exc}); rebuilding")
        say("dequantizing weights (first run only; cached afterwards)" if can_cache else "dequantizing weights")
        weights = g.load_weights(dtype=np.float32)
        say(f"loading weights into {name}")
        self.backend.load(weights, self.config)
        if cache is not None:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                for old in cache.parent.glob(f"{self.model_path.stem}.{dtype}.*.safetensors"):
                    old.unlink()
                self.backend.save_cache(cache)
            except OSError as exc:
                say(f"couldn't write weight cache: {exc}")

    # --- edits -------------------------------------------------------------------

    @property
    def edits(self) -> list[Edit]:
        return list(self.edit_stack.edits)

    def _sync(self, tensors: set[str]) -> None:
        ops = self.edit_stack.ops_by_tensor()
        for t in sorted(tensors):
            if t in self._shapes:  # a rejected edit may name a tensor that doesn't exist
                self.backend.set_tensor_ops(t, ops.get(t, []))

    def _check_ops(self, edit: Edit) -> None:
        """Reject edits that name a missing tensor or fall outside its shape, before
        they enter the stack (a bad edit must never be half-applied)."""
        if edit.op not in ("scale", "noise"):
            raise ValueError(f"op must be scale or noise, got {edit.op!r}")
        if not np.isfinite(edit.value):
            raise ValueError("edit value must be a finite number")
        for o in resolve(edit, self.config):
            shape = self._shapes.get(o.tensor)
            if shape is None:
                raise ValueError(f"no tensor named {o.tensor!r}")
            if o.axis is None:
                continue
            if o.axis >= len(shape):
                raise ValueError(f"{o.tensor} has {len(shape)} dims; can't edit axis {o.axis}")
            for a, b in o.indices:
                if not 0 <= a < b <= shape[o.axis]:
                    raise ValueError(f"{o.tensor}: range {a}:{b} is outside 0:{shape[o.axis]}")

    def add_edit(
        self, target: Target | dict, op: str = "scale", value: float = 0.0, note: str = "", seed: int | None = None
    ) -> Edit:
        if isinstance(target, dict):
            target = Target(**target)
        with self._lock:
            e = self.edit_stack.add(target, op=op, value=value, seed=seed, note=note)
            try:
                self._check_ops(e)
                self._sync(self.edit_stack.tensors_of([e]))
            except Exception:
                self.edit_stack.remove(e.id)
                self._sync(self.edit_stack.tensors_of([e]))
                raise
            return e

    def update_edit(self, edit_id: int, value: float | None = None, op: str | None = None) -> Edit:
        with self._lock:
            old = self.edit_stack.get(edit_id)
            e = self.edit_stack.update(edit_id, value=value, op=op)
            try:
                self._check_ops(e)
                self._sync(self.edit_stack.tensors_of([e]))
            except Exception:
                self.edit_stack.update(edit_id, value=old.value, op=old.op)
                self._sync(self.edit_stack.tensors_of([e]))
                raise
            return e

    def undo(self, edit_id: int) -> Edit:
        with self._lock:
            e = self.edit_stack.remove(edit_id)
            self._sync(self.edit_stack.tensors_of([e]))
            return e

    def undo_last(self) -> Edit | None:
        if not self.edit_stack.edits:
            return None
        return self.undo(self.edit_stack.edits[-1].id)

    def reset_edits(self) -> list[Edit]:
        with self._lock:
            old = self.edit_stack.clear()
            self._sync(self.edit_stack.tensors_of(old))
            return old

    def load_edits(self, edits: list[dict], replace_existing: bool = True) -> list[Edit]:
        """Apply a saved recipe (a list of ``Edit.to_dict()`` dicts). All or nothing:
        if any edit is invalid, the previous edits are restored."""
        with self._lock:
            before = list(self.edit_stack.edits)
            try:
                if replace_existing:
                    self.reset_edits()
                out = []
                for d in edits:
                    t = dict(d["target"])
                    for k in ("rows", "cols"):
                        if t.get(k) is not None:
                            t[k] = tuple(t[k])
                    out.append(
                        self.add_edit(
                            Target(**t),
                            op=d.get("op", "scale"),
                            value=d.get("value", 0.0),
                            note=d.get("note", ""),
                            seed=d.get("seed"),
                        )
                    )
                return out
            except Exception:
                touched = self.edit_stack.tensors_of(self.edit_stack.edits) | self.edit_stack.tensors_of(before)
                self.edit_stack.edits = before
                self._sync(touched)
                raise

    @contextlib.contextmanager
    def healthy(self):
        """Temporarily run the pristine model."""
        with self._lock:
            tensors = set(self.edit_stack.ops_by_tensor())
            try:
                for t in tensors:
                    self.backend.set_tensor_ops(t, [])
                yield
            finally:
                self._sync(tensors)

    # --- chat --------------------------------------------------------------------

    def _messages(self, upto: Exchange) -> list[dict]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for ex in self.exchanges:
            msgs.append({"role": "user", "content": ex.user})
            if ex is upto:
                break
            if ex.active is not None:
                msgs.append({"role": "assistant", "content": ex.active.text})
        return msgs

    def _last_user_start(self, ids: list[int]) -> int:
        """Position of the last ``<|im_start|>user`` in the prompt."""
        im_start = self.tokenizer.token_id("<|im_start|>")
        user = self.tokenizer.encode("user")
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == im_start and ids[i + 1 : i + 1 + len(user)] == user:
                return i
        return 0

    def cancel(self) -> None:
        self._cancel.set()

    def stream_chat(self, text: str, sampling: SamplingParams | None = None) -> Iterator[dict]:
        """Send a user message and stream the reply. See ``_generate`` for events."""
        ex = Exchange(id=self._next("exchange"), user=text)
        self.exchanges.append(ex)
        try:
            yield from self._generate(ex, sampling, healthy=False)
        finally:
            # a failed generation must not leave a dangling user turn
            if not ex.replies and ex in self.exchanges:
                self.exchanges.remove(ex)

    def stream_again(self, sampling: SamplingParams | None = None, healthy: bool = False) -> Iterator[dict]:
        """Answer the latest user message again with the current weights (or healthy ones)."""
        if not self.exchanges:
            raise RuntimeError("nothing to ask again yet")
        ex = self.exchanges[-1]
        prev = ex.replies[0] if ex.replies else None
        if sampling is None and prev is not None:
            # current settings, but the first answer's seed, so a different
            # answer comes from the weights rather than from sampling luck
            sampling = replace(self.sampling, seed=prev.sampling["seed"])
        yield from self._generate(ex, sampling, healthy=healthy)

    def chat(self, text: str, sampling: SamplingParams | None = None, on_token=None) -> Reply:
        return self._drain(self.stream_chat(text, sampling), on_token)

    def ask_again(self, sampling: SamplingParams | None = None, healthy: bool = False, on_token=None) -> Reply:
        return self._drain(self.stream_again(sampling, healthy), on_token)

    @staticmethod
    def _drain(events: Iterator[dict], on_token) -> Reply:
        reply = None
        for ev in events:
            if ev["type"] == "token" and on_token:
                on_token(ev["text"])
            elif ev["type"] == "done":
                reply = ev["reply"]
        return reply

    def _generate(self, ex: Exchange, sampling: SamplingParams | None, healthy: bool) -> Iterator[dict]:
        """Yields ``{"type": "start", ...}``, ``{"type": "token", ...}`` per token, then
        ``{"type": "done", "reply": Reply}``."""
        params = replace(sampling or self.sampling)
        with self._lock:
            self.busy = True
            self._cancel.clear()
            ctx = self.healthy() if healthy else contextlib.nullcontext()
            try:
                with ctx:
                    yield from self._generate_locked(ex, params, healthy)
            finally:
                self.busy = False

    def _keep_memory_start(self, ex: Exchange):
        """In keep-memory mode, the cache and prompt to continue from, or None to re-read.

        Ask again rewinds to the snapshot taken just before the question's last
        token. A new message continues from the cache left by the previous reply,
        appending only the new turn's tokens.
        """
        be = self.backend
        if ex.replies:
            s = self._ask_snap
            if s is not None and s["exchange_id"] == ex.id:
                return s["prompt_ids"], be.restore_cache(s["snap"])
            return None
        m = self._mem
        prev = self.exchanges[-2] if len(self.exchanges) >= 2 and self.exchanges[-1] is ex else None
        if m is None or prev is None or prev.active is None or m["reply_id"] != prev.active.id:
            return None
        head = "<think>\n" if self.enable_thinking else "<think>\n\n</think>\n\n"
        delta = self.tokenizer.encode(f"<|im_end|>\n<|im_start|>user\n{ex.user}<|im_end|>\n<|im_start|>assistant\n{head}")
        return m["tokens"] + delta, be.restore_cache(m["snap"])

    def _generate_locked(self, ex: Exchange, params: SamplingParams, healthy: bool) -> Iterator[dict]:
        tok = self.tokenizer
        be = self.backend
        can_snap = type(be).snapshot_cache is not Backend.snapshot_cache
        start = None
        if not self.reread and not healthy and can_snap:
            start = self._keep_memory_start(ex)
        kept = start is not None
        if kept:
            prompt_ids, cache = start
        else:
            prompt_ids = tok.encode_chat(self._messages(ex), add_generation_prompt=True, enable_thinking=self.enable_thinking)
            cache = be.new_cache()
        record_start = self._last_user_start(prompt_ids)
        spec = RecordSpec(start=record_start, attention=True)
        sampler = Sampler(params)
        eos = set(self.config.eos_token_ids)
        yield {
            "type": "start",
            "exchange_id": ex.id,
            "prompt_tokens": len(prompt_ids),
            "healthy": healthy,
            "kept_memory": kept,
        }

        # Feed the prompt up to its last token, snapshot (so "ask again" can rewind
        # here in keep-memory mode), then feed the last token.
        t0 = time.time()
        parts = []
        fed = be.cache_length(cache)
        if len(prompt_ids) - 1 > fed:
            _, rec = be.forward(prompt_ids[fed:-1], cache, spec)
            parts.append(rec)
        if can_snap:
            self._ask_snap = {"exchange_id": ex.id, "prompt_ids": prompt_ids, "snap": be.snapshot_cache(cache)}
        logits, rec = be.forward(prompt_ids[-1:], cache, spec)
        parts.append(rec)
        t_prefill = time.time() - t0
        out_ids, texts, probs, alts = [], [], [], []
        finish = "length"
        t1 = time.time()
        decoded = ""
        for _ in range(params.max_tokens):
            if self._cancel.is_set():
                finish = "cancelled"
                break
            tid, p, alt = sampler(logits)
            if tid in eos:
                finish = "stop"
                break
            out_ids.append(tid)
            probs.append(p)
            alts.append(alt)
            # Decode incrementally so multi-byte characters come out whole.
            full = tok.decode(out_ids)
            piece = full[len(decoded) :] if not full.endswith("�") else ""
            if piece:
                decoded = full
            texts.append(tok.token_str(tid))
            yield {
                "type": "token",
                "text": piece,
                "token_id": tid,
                "prob": p,
                "index": len(out_ids) - 1,
                # the pass that wrote this word: its strongest neurons per layer
                "flow": self._token_flow(parts[-1]),
            }
            logits, rec = self.backend.forward([tid], cache, spec)
            parts.append(rec)
        t_decode = time.time() - t1

        text = tok.decode(out_ids)
        recording = Recording.concat([p for p in parts if p is not None]) if any(parts) else None
        reply = Reply(
            id=self._next("reply"),
            exchange_id=ex.id,
            text=text,
            token_ids=out_ids,
            token_texts=texts,
            token_probs=probs,
            alternatives=alts,
            prompt_ids=prompt_ids,
            record_start=record_start,
            recording=recording,
            edits=[] if healthy else [e.to_dict() for e in self.edit_stack.edits],
            healthy=healthy,
            sampling=params.to_dict(),
            finish_reason=finish,
            kept_memory=kept,
            stats={
                "prompt_tokens": len(prompt_ids),
                "prefill_seconds": round(t_prefill, 4),
                "reply_tokens": len(out_ids),
                "decode_seconds": round(t_decode, 4),
                "tokens_per_second": round(len(out_ids) / t_decode, 1) if t_decode > 0 and out_ids else None,
            },
        )
        ex.replies.append(reply)
        self._trim_old_recordings()
        if can_snap:
            # every generated token was fed, so the cache holds prompt + reply
            self._mem = {"reply_id": reply.id, "tokens": prompt_ids + out_ids, "snap": be.snapshot_cache(cache)}
        yield {"type": "done", "reply": reply}

    def _token_flow(self, rec: Recording | None) -> dict | None:
        if rec is None or rec.mlp is None or not rec.n_positions:
            return None
        idx, val = top_by_layer(rec.mlp[-1], self.flow_k)
        return {"idx": idx.tolist(), "val": np.round(val, 3).tolist()}

    def _trim_old_recordings(self) -> None:
        replies = sorted((r for ex in self.exchanges for r in ex.replies), key=lambda r: r.id)
        for r in replies[: max(0, len(replies) - self.keep_recordings)]:
            rec = r.recording
            if rec is None or rec.mlp is None:
                continue
            from . import analysis as A

            r.recorded_tokens(self.tokenizer)  # cache token texts
            reply_rows = [i for i in range(rec.n_positions) if int(rec.positions[i]) >= r.prompt_len]
            r.cache["neuron_mean"] = np.abs(rec.mlp.astype(np.float32)).mean(-1)
            r.cache["layers:reply"] = A.layer_activity(rec, reply_rows)
            r.recording = replace(rec, mlp=None, resid_vec=None, attention={})
            r.cache["trimmed"] = True

    def reset_chat(self) -> None:
        with self._lock:
            self.exchanges.clear()
            self._mem = None
            self._ask_snap = None

    def find_reply(self, reply_id: int) -> Reply:
        for ex in self.exchanges:
            for r in ex.replies:
                if r.id == reply_id:
                    return r
        raise KeyError(f"no reply with id {reply_id}")

    def last_reply(self) -> Reply | None:
        for ex in reversed(self.exchanges):
            if ex.replies:
                return ex.replies[-1]
        return None

    # --- measurements ------------------------------------------------------------

    def damage(self, reply: Reply) -> dict:
        """How far the current weights are from healthy, on this reply's tokens."""
        ids = reply.prompt_ids + reply.token_ids
        if not reply.token_ids:
            return {"kl": [], "mean_kl": 0.0}
        with self._lock:
            r = self.backend.damage_report(ids, reply.prompt_len, self.healthy())
        tok = self.tokenizer
        return {
            "kl": r["kl"].tolist(),
            "mean_kl": float(np.mean(r["kl"])),
            "logp_current": r["logp_current"].tolist(),
            "logp_healthy": r["logp_healthy"].tolist(),
            "healthy_would_say": [tok.token_str(int(t)) for t in r["top_healthy"]],
            "current_would_say": [tok.token_str(int(t)) for t in r["top_current"]],
        }

    def scan(
        self,
        reply: Reply,
        kind: str = "layers",
        layer: int | None = None,
        candidates: list[int] | None = None,
        value: float = 0.0,
        part: str = "all",
    ) -> list[dict]:
        """Ablation scan: how much would this reply change if each part were cut?

        ``kind``: "layers" (every layer, ``part`` = all/mixer/mlp), "heads" (every
        head of ``layer``) or "neurons" (``candidates`` in ``layer``). Each part is
        scaled by ``value`` on top of the current edits, one at a time, and scored
        by the KL divergence from the current model over the reply's tokens.
        Returns one entry per part, in scan order: ``{target, mean_kl, top1_agree}``.
        """
        cfg = self.config
        if kind == "layers":
            targets = [Target("layer", layer=i, part=part) for i in range(cfg.n_layers)]
        elif kind == "heads":
            targets = [Target("head", layer=layer, index=h) for h in range(cfg.n_layer_heads(layer))]
        elif kind == "neurons":
            if not candidates:
                raise ValueError("neuron scan needs candidate neurons")
            targets = [Target("neuron", layer=layer, index=n) for n in candidates]
        else:
            raise ValueError(f"unknown scan kind {kind!r}")
        if not reply.token_ids:
            return []
        with self._lock:
            current = self.edit_stack.ops_by_tensor()
            variants = []
            for t in targets:
                extra = resolve(Edit(id=0, target=t, op="scale", value=value), cfg)
                v: dict[str, list] = {}
                for o in extra:
                    v.setdefault(o.tensor, list(current.get(o.tensor, []))).append(o)
                variants.append(v)
            scores = self.backend.score_variants(
                reply.prompt_ids + reply.token_ids, reply.prompt_len, variants, current
            )
        return [
            {"target": asdict(t), "label": t.label(), "mean_kl": s["mean_kl"], "top1_agree": s["top1_agree"]}
            for t, s in zip(targets, scores)
        ]

    def why(self, reply: Reply, row: int, k: int = 12) -> dict:
        """Which neurons and heads pushed the model toward the token at this
        recorded row? One backward pass; estimates, not measurements.

        ``effect`` is the estimated change in log-probability of the token if
        that unit were switched off (negative = cutting it makes the token less
        likely). ``support`` lists the units whose removal would hurt the token
        most; ``oppose`` lists the ones holding it back.
        """
        if reply.recording is None:
            raise ValueError("no recording")
        pos = int(reply.recording.positions[row])
        if pos < 1:
            raise ValueError("the first token has nothing before it")
        ids = reply.prompt_ids + reply.token_ids
        with self._lock:
            logp, g_mlp, g_heads = self.backend.attribute(ids, pos)
        cfg = self.config
        for i in range(cfg.n_layers):  # heads that don't exist
            g_heads[i, cfg.n_layer_heads(i):] = 0.0
        units = []
        flat = g_mlp.ravel()
        for idx in np.concatenate([np.argpartition(-flat, k)[:k], np.argpartition(flat, k)[:k]]):
            layer, n = divmod(int(idx), g_mlp.shape[1])
            units.append({"kind": "neuron", "layer": layer, "index": n, "effect": -float(flat[idx])})
        hf = g_heads.ravel()
        kh = min(k, hf.size - 1)
        for idx in np.concatenate([np.argpartition(-hf, kh)[:kh], np.argpartition(hf, kh)[:kh]]):
            layer, h = divmod(int(idx), g_heads.shape[1])
            if h < cfg.n_layer_heads(layer):
                units.append({"kind": "head", "layer": layer, "index": h, "effect": -float(hf[idx])})
        support = sorted((u for u in units if u["effect"] < 0), key=lambda u: u["effect"])[:k]
        oppose = sorted((u for u in units if u["effect"] > 0), key=lambda u: -u["effect"])[:k]
        layer_total = (-g_mlp.sum(axis=1) - g_heads.sum(axis=1)).tolist()
        return {
            "token": self.tokenizer.token_str(ids[pos]),
            "position": pos,
            "logp": logp,
            "prob": float(np.exp(logp)),
            "support": support,
            "oppose": oppose,
            "layer_effect": layer_total,
        }

    def flow(self, reply: Reply, k: int = 96) -> dict:
        """The reply as the 3D view replays it: one step per pass that wrote a word.

        Step i reads one token and says the next: the first step is the pass over
        the last prompt token (it wrote the reply's first word), the last is the
        pass over the reply's last word (it said "stop", or nothing if the reply
        was cut off). For every step and layer: the k strongest neurons (indices,
        strongest first) and their size as 0..255 of the layer's usual range
        over this reply (values above the range clip at 255).
        """
        rec = reply.recording
        if rec is None or rec.mlp is None:
            raise ValueError("no recording")
        toks = reply.recorded_tokens(self.tokenizer)
        reply_rows = [t["row"] for t in toks if t["role"] == "reply"]
        first = reply_rows[0] - 1 if reply_rows else len(toks) - 1
        rows = list(range(max(0, first), len(toks)))
        steps = []
        for row in rows:
            nxt = row + 1
            says_pos = toks[row]["position"] + 1 - reply.prompt_len  # index into the reply
            steps.append(
                {
                    "row": row,
                    "reads": toks[row]["text"],
                    "says": toks[nxt]["text"] if nxt < len(toks) else None,
                    "prob": reply.token_probs[says_pos] if 0 <= says_pos < len(reply.token_probs) else None,
                    "end": reply.finish_reason if nxt >= len(toks) else None,
                }
            )
        scale = np.maximum(np.asarray(reply.layer_refs(), np.float32), 1e-6)
        idx, val = top_by_layer(rec.mlp[rows], k)  # (S, L, k)
        q = np.round(np.clip(np.abs(val) / scale[None, :, None], 0, 1) * 255).astype(np.uint8)
        return {
            "steps": steps,
            "k": int(idx.shape[-1]),
            "n_layers": int(idx.shape[1]),
            "n_neurons": int(rec.mlp.shape[2]),
            "scale": scale.tolist(),
            "idx": base64.b64encode(idx.astype("<u2").tobytes()).decode(),
            "val": base64.b64encode(q.tobytes()).decode(),
        }

    def lens_grid(self, reply: Reply) -> dict:
        """When did each layer know the next word? For every reply token, the
        probability each layer's residual stream (read through the final norm and
        unembedding) gave that token, at the position that predicted it.

        Returns {"rows": [...], "probs": (K, L) list}, where rows are the recording
        rows of the reply tokens (the cells line up with the layer strips).
        """
        rec = reply.recording
        if rec is None or rec.resid_vec is None:
            raise ValueError("no recording")
        if "lens_grid" in reply.cache:
            return reply.cache["lens_grid"]
        pos_to_row = {int(p): i for i, p in enumerate(rec.positions.tolist())}
        rows, src, targets = [], [], []
        for k, tid in enumerate(reply.token_ids):
            pos = reply.prompt_len + k
            prev = pos_to_row.get(pos - 1)
            if prev is None or pos not in pos_to_row:
                continue
            rows.append(pos_to_row[pos])
            src.append(prev)
            targets.append(tid)
        L = self.config.n_layers
        if not rows:
            out = {"rows": [], "probs": []}
        else:
            vecs = rec.resid_vec[src].reshape(len(src) * L, -1)  # (K*L, hidden)
            with self._lock:
                p = self.backend.lens_probs(vecs, np.repeat(targets, L))
            out = {"rows": rows, "probs": p.reshape(len(src), L)}
        reply.cache["lens_grid"] = out
        return out

    def logit_lens(self, reply: Reply, row: int, k: int = 5) -> list[list[tuple[str, float]]]:
        """For one recorded position: the top-k next tokens implied after each layer."""
        vecs = reply.recording.resid_vec[row]  # (L, hidden)
        ids, probs = self.backend.unembed_topk(vecs, k=k)
        tok = self.tokenizer
        return [[(tok.token_str(int(i)), float(p)) for i, p in zip(ri, rp)] for ri, rp in zip(ids, probs)]

    def neuron_vocab(self, layer: int, neuron: int, k: int = 10) -> dict:
        """Tokens a neuron pushes up and down when it fires positively (its down_proj
        column read through the unembedding). Uses the healthy weights, so a
        switched-off MLP still shows what the neuron would say."""
        col = self.backend.read_pristine(f"model.layers.{layer}.mlp.down_proj.weight")[:, neuron]
        ids_up, _ = self.backend.unembed_topk(col[None, :], k=k, apply_final_norm=True)
        ids_dn, _ = self.backend.unembed_topk(-col[None, :], k=k, apply_final_norm=True)
        tok = self.tokenizer
        return {
            "promotes": [tok.token_str(int(i)) for i in ids_up[0]],
            "suppresses": [tok.token_str(int(i)) for i in ids_dn[0]],
        }

    # --- misc ----------------------------------------------------------------------

    def _next(self, kind: str) -> int:
        n = self._ids[kind]
        self._ids[kind] += 1
        return n

    def info(self) -> dict:
        d = self.config.describe()
        d.update(
            {
                "model_file": self.model_path.name,
                "backend": self.backend.name,
                "load_seconds": round(self.load_seconds, 2),
                "tensor_types": self.gguf.tensor_types,
            }
        )
        return d
