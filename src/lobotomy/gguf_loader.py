"""Read a Qwen3.5 GGUF file into float weights with Hugging Face names and conventions.

The GGUF on disk is never modified. llama.cpp's converter changes some tensors on
the way in (see llama.cpp `conversion/qwen.py`); this module undoes those changes
so every backend receives the same tensors as the official checkpoint:

- RMSNorm weights are stored as ``w + 1``, except the DeltaNet gated norm.
  Hugging Face stores ``w`` and computes ``x * (1 + w)``.
- ``A_log`` is stored as ``ssm_a = -exp(A_log)``.
- ``dt_bias`` is stored as ``ssm_dt.bias``.
- The depthwise conv weight is squeezed from ``(C, 1, K)`` to ``(C, K)``.

Output names follow ``Qwen3_5ForCausalLM`` (text only), e.g.
``model.layers.3.self_attn.q_proj.weight``. Matrices have the PyTorch ``Linear``
layout ``(out_features, in_features)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from gguf.quants import dequantize

from .config import ModelConfig

ARCH = "qwen35"
META_VERSION = 1

_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # full attention
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # gated deltanet
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

# Norms that llama.cpp stores as (w + 1).
_SHIFTED_NORMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)

_FLOAT_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


@dataclass
class TokenizerData:
    tokens: list[str]
    token_types: list[int]
    merges: list[str]
    pre: str
    eos_token_id: int
    pad_token_id: int | None
    chat_template: str | None


@dataclass
class GGUFModel:
    path: Path
    config: ModelConfig
    tokenizer: TokenizerData
    tensor_types: dict[str, str]  # hf name -> original GGUF quant type, for display
    _reader: GGUFReader | None = None

    @property
    def reader(self) -> GGUFReader:
        if self._reader is None:  # opened lazily when metadata came from the cache
            self._reader = GGUFReader(self.path)
        return self._reader

    def iter_weights(self, dtype=np.float16) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (hf_name, array). 2-D matrices are cast to ``dtype``; vectors stay float32."""
        for t in self.reader.tensors:
            name, arr = _convert(t.name, _tensor_to_float(t))
            if name is None:
                continue
            if arr.ndim >= 2 and dtype is not None:
                arr = arr.astype(dtype)
            yield name, arr

    def to_meta(self) -> dict:
        return {
            "version": META_VERSION,
            "config": asdict(self.config),
            "tokenizer": asdict(self.tokenizer),
            "tensor_types": self.tensor_types,
        }

    def load_weights(self, dtype=np.float16, on_progress: Callable[[int, int], None] | None = None) -> dict[str, np.ndarray]:
        out = {}
        n = len(self.reader.tensors)
        for i, (name, arr) in enumerate(self.iter_weights(dtype)):
            out[name] = arr
            if on_progress:
                on_progress(i + 1, n)
        return out


def _field(reader: GGUFReader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.contents()


def _tensor_to_float(t) -> np.ndarray:
    if t.tensor_type in _FLOAT_TYPES:
        arr = np.asarray(t.data)
        if t.tensor_type == GGMLQuantizationType.BF16:
            arr = dequantize(arr, t.tensor_type)
        return arr.astype(np.float32)
    return dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)


def hf_name(gguf_name: str) -> str | None:
    """The Hugging Face name for a GGUF tensor, or None for tensors we skip (e.g. MTP)."""
    if gguf_name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gguf_name == "output_norm.weight":
        return "model.norm.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if not gguf_name.startswith("blk."):
        return None
    _, layer, rest = gguf_name.split(".", 2)
    suffix = _LAYER_MAP.get(rest)
    if suffix is None:
        return None
    return f"model.layers.{int(layer)}.{suffix}"


def _convert(gguf_name: str, arr: np.ndarray) -> tuple[str | None, np.ndarray]:
    name = hf_name(gguf_name)
    if name is None:
        return None, arr
    if name == "model.norm.weight" or name.endswith(_SHIFTED_NORMS):
        arr = arr - 1.0
    elif name.endswith("linear_attn.A_log"):
        arr = np.log(-arr)
    elif name.endswith("linear_attn.conv1d.weight"):
        arr = arr.reshape(arr.shape[0], 1, arr.shape[-1])
    return name, arr


def meta_cache_path(path: Path) -> Path:
    st = path.stat()
    return path.parent / ".cache" / f"{path.stem}.{st.st_size}.{int(st.st_mtime)}.meta.json"


def open_gguf(path: str | Path, use_cache: bool = True) -> GGUFModel:
    """Open a GGUF. Parsing its metadata takes seconds (a 248k-entry vocabulary),
    so the parsed metadata is cached as JSON beside the file in ``.cache/``."""
    path = Path(path)
    cache = meta_cache_path(path) if use_cache else None
    if cache is not None and cache.exists():
        try:
            meta = json.loads(cache.read_text())
            if meta.get("version") == META_VERSION:
                cfg = meta["config"]
                cfg["eos_token_ids"] = tuple(cfg["eos_token_ids"])
                return GGUFModel(
                    path=path,
                    config=ModelConfig(**cfg),
                    tokenizer=TokenizerData(**meta["tokenizer"]),
                    tensor_types=meta["tensor_types"],
                )
        except (OSError, ValueError, TypeError, KeyError):
            pass  # fall through and re-parse
    model = _parse_gguf(path)
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            for old in cache.parent.glob(f"{path.stem}.*.meta.json"):
                old.unlink()
            cache.write_text(json.dumps(model.to_meta()))
        except OSError:
            pass
    return model


def _parse_gguf(path: Path) -> GGUFModel:
    reader = GGUFReader(path)
    arch = _field(reader, "general.architecture")
    if arch != ARCH:
        raise ValueError(f"{path.name}: architecture {arch!r} is not supported (expected {ARCH!r})")

    def a(key, default=None):
        return _field(reader, f"{ARCH}.{key}", default)

    tokens = _field(reader, "tokenizer.ggml.tokens")
    eos = int(_field(reader, "tokenizer.ggml.eos_token_id"))
    eos_ids = {eos}
    for special in ("<|im_end|>", "<|endoftext|>"):
        if special in tokens:
            eos_ids.add(tokens.index(special))

    n_heads = int(a("attention.head_count"))
    has_output = any(t.name == "output.weight" for t in reader.tensors)
    config = ModelConfig(
        n_layers=int(a("block_count")),
        hidden_size=int(a("embedding_length")),
        intermediate_size=int(a("feed_forward_length")),
        vocab_size=len(tokens),
        rms_norm_eps=float(a("attention.layer_norm_rms_epsilon")),
        n_heads=n_heads,
        n_kv_heads=int(a("attention.head_count_kv")),
        head_dim=int(a("attention.key_length")),
        rope_theta=float(a("rope.freq_base")),
        rope_dim=int(a("rope.dimension_count")),
        linear_n_k_heads=int(a("ssm.group_count")),
        linear_n_v_heads=int(a("ssm.time_step_rank")),
        linear_k_head_dim=int(a("ssm.state_size")),
        linear_v_head_dim=int(a("ssm.inner_size")) // int(a("ssm.time_step_rank")),
        linear_conv_kernel=int(a("ssm.conv_kernel")),
        full_attention_interval=int(a("full_attention_interval", 4)),
        tie_word_embeddings=not has_output,
        eos_token_ids=tuple(sorted(eos_ids)),
        name=str(_field(reader, "general.name", "qwen3.5")),
    )

    if config.linear_n_v_heads != config.linear_n_k_heads:
        # llama.cpp's converter reorders DeltaNet value heads from grouped to tiled
        # order when there are more value heads than key heads (e.g. Qwen3.5 4B);
        # undoing that isn't implemented or verified yet.
        raise NotImplementedError(
            f"{path.name}: DeltaNet layers with {config.linear_n_v_heads} value heads and "
            f"{config.linear_n_k_heads} key heads aren't supported yet (Qwen3.5 0.8B and 2B are)."
        )

    tok = TokenizerData(
        tokens=tokens,
        token_types=list(_field(reader, "tokenizer.ggml.token_type")),
        merges=_field(reader, "tokenizer.ggml.merges"),
        pre=_field(reader, "tokenizer.ggml.pre"),
        eos_token_id=eos,
        pad_token_id=_field(reader, "tokenizer.ggml.padding_token_id"),
        chat_template=_field(reader, "tokenizer.chat_template"),
    )

    types = {n: t.tensor_type.name for t in reader.tensors if (n := hf_name(t.name))}
    return GGUFModel(path=path, config=config, tokenizer=tok, tensor_types=types, _reader=reader)
