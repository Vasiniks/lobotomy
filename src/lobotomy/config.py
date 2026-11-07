"""Model configuration for Qwen3.5 text models, read from GGUF metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    # Full attention layers
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rope_dim: int  # rotary dims per head (partial RoPE)
    # Gated DeltaNet (linear attention) layers
    linear_n_k_heads: int
    linear_n_v_heads: int
    linear_k_head_dim: int
    linear_v_head_dim: int
    linear_conv_kernel: int
    full_attention_interval: int
    tie_word_embeddings: bool = True
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    name: str = "qwen3.5"

    def is_full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_interval == 0

    def layer_type(self, layer: int) -> str:
        return "attention" if self.is_full_attention(layer) else "deltanet"

    def n_layer_heads(self, layer: int) -> int:
        return self.n_heads if self.is_full_attention(layer) else self.linear_n_v_heads

    @property
    def max_heads(self) -> int:
        return max(self.n_heads, self.linear_n_v_heads)

    @property
    def attention_layers(self) -> list[int]:
        return [i for i in range(self.n_layers) if self.is_full_attention(i)]

    def describe(self) -> dict:
        """Plain-data summary of the model's structure, for UIs."""
        return {
            "name": self.name,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "n_neurons": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "layers": [
                {
                    "index": i,
                    "type": self.layer_type(i),
                    "n_heads": self.n_layer_heads(i),
                    "n_neurons": self.intermediate_size,
                }
                for i in range(self.n_layers)
            ],
        }
