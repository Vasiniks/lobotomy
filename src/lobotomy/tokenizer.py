"""Tokenizer and chat template built from the GGUF file alone.

The vocabulary and merges come from the GGUF. The pre-tokenizer split pattern is
the one in Qwen3.5's official tokenizer.json; the GGUF records it only by name
("qwen35").
"""

from __future__ import annotations

import json
from datetime import datetime

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from .gguf_loader import TokenizerData

QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# llama.cpp token types
_NORMAL, _CONTROL, _USER_DEFINED = 1, 3, 4


class ChatTokenizer:
    def __init__(self, data: TokenizerData):
        if data.pre != "qwen35":
            raise ValueError(f"unsupported pre-tokenizer {data.pre!r}")
        vocab = {}
        added: list[AddedToken] = []
        for i, (tok, typ) in enumerate(zip(data.tokens, data.token_types)):
            if typ == _NORMAL:
                vocab[tok] = i
            elif typ in (_CONTROL, _USER_DEFINED):
                added.append(AddedToken(tok, special=(typ == _CONTROL), normalized=False))
        merges = [tuple(m.split(" ", 1)) for m in data.merges]
        tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
        tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(QWEN35_SPLIT), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
            ]
        )
        tk.decoder = decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)
        # Added tokens must keep their GGUF ids; add them in id order.
        tk.add_tokens(added)
        for t in added:
            got = tk.token_to_id(t.content)
            want = data.tokens.index(t.content)
            if got != want:
                raise RuntimeError(f"added token {t.content!r} got id {got}, expected {want}")
        self._tk = tk
        self.tokens = data.tokens
        self.eos_token_id = data.eos_token_id
        self.chat_template = data.chat_template
        self._template = _compile_template(data.chat_template) if data.chat_template else None

    @property
    def backend(self) -> Tokenizer:
        return self._tk

    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        return self._tk.decode(ids, skip_special_tokens=skip_special)

    def token_str(self, token_id: int) -> str:
        """Readable text for a single token (byte-level pieces decoded)."""
        return self._tk.decode([token_id], skip_special_tokens=False)

    def token_id(self, text: str) -> int | None:
        return self._tk.token_to_id(text)

    def render_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        if self._template is None:
            raise RuntimeError("GGUF has no chat template")
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def encode_chat(self, messages: list[dict], **kw) -> list[int]:
        return self.encode(self.render_chat(messages, **kw))


def _raise(msg):
    raise jinja2.exceptions.TemplateError(msg)


def _compile_template(source: str):
    # Mirrors transformers' chat-template environment.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
        x, indent=indent, separators=separators, sort_keys=sort_keys, ensure_ascii=ensure_ascii
    )
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env.from_string(source)
