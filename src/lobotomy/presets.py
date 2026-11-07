"""Starter recipes: one-click lobotomies with distinct effects.

Found by trying them on Qwen3.5 0.8B with a few prompts. The effects are
tendencies, not guarantees; they depend on the prompt and the seed.
"""

from __future__ import annotations

from .config import ModelConfig


def _layers(cfg: ModelConfig, part: str, layers, value: float, op: str = "scale", note: str = "") -> list[dict]:
    return [{"target": {"kind": "layer", "layer": L, "part": part}, "op": op, "value": value, "note": note} for L in layers]


def presets(cfg: ModelConfig) -> list[dict]:
    if cfg.n_layers != 24:  # tuned for Qwen3.5 0.8B
        return []
    attn = cfg.attention_layers
    return [
        {
            "id": "emoji-soup",
            "name": "Emoji soup",
            "description": "Switches off all six attention blocks. With no way to look back at the chat, replies usually dissolve into emoji and stray characters.",
            "edits": _layers(cfg, "mixer", attn, 0.0, note="Preset: emoji soup."),
        },
        {
            "id": "broken-record",
            "name": "Broken record",
            "description": "Switches off the DeltaNet blocks in layers 4 to 6. The model tends to loop, repeating the question or its first line.",
            "edits": _layers(cfg, "mixer", range(4, 7), 0.0, note="Preset: broken record."),
        },
        {
            "id": "overexcited",
            "name": "Overexcited",
            "description": "Doubles the MLPs in layers 16 to 23. Replies get gushing and wordy.",
            "edits": _layers(cfg, "mlp", range(16, 24), 2.0, note="Preset: overexcited."),
        },
        {
            "id": "fever-dream",
            "name": "Fever dream",
            "description": "Triples the MLPs in the last four layers. Sentences start fine and slide into odd words and strings of digits.",
            "edits": _layers(cfg, "mlp", range(20, 24), 3.0, note="Preset: fever dream."),
        },
        {
            "id": "tipsy",
            "name": "Tipsy",
            "description": "Adds noise to layers 10 to 15. Still mostly coherent, but it wanders and changes its mind.",
            "edits": _layers(cfg, "all", range(10, 16), 0.4, op="noise", note="Preset: tipsy."),
        },
    ]
