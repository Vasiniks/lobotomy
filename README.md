# lobotomy

Break a small LLM on purpose, live, while you chat with it, and see inside it to
choose what to break.

Chat with the model, look at which neurons fired in which layer, switch neurons,
heads or whole layers off (or turn them down, boost them, add noise), then ask
again and see what changed. Every edit can be undone exactly.

The default model is **Qwen3.5 0.8B** (Unsloth's Q4_K_M GGUF). It runs locally
on Apple Silicon through MLX. A CUDA backend has a slot but isn't written yet.

> Status: the engine and the terminal interface work. The web UI is in progress.
> See [docs/design.md](docs/design.md).

## Quick start (Apple Silicon)

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e ".[mlx,dev]"
scripts/download_model.sh            # ~533 MB into models/
.venv/bin/lobotomy chat
```

```
> What is the capital of France?
The capital of France is Paris.
  (7 tokens, 54 tok/s, prompt 24)

> /layer 0
#1 L0: off
> /again
- of,,,th isinginginginging…
> /damage
mean KL from healthy: 1.520 nats
> /undo
> /again
The capital of France is Paris.
```

Type `/help` for every command: `/neuron`, `/head`, `/layer`, `/noise`, `/token`,
`/topoff`, `/again`, `/healthy`, `/damage`, `/layers`, `/top`, `/lens`, `/says`,
and saving or loading edit recipes.

## What you can break

| Target | What changes | Effect |
|---|---|---|
| neuron (layer, n) | row `n` of `mlp.up_proj` | scales that neuron's activation, and its effect, exactly |
| head (layer, h) | the head's columns of the output projection | scales that head's contribution to the residual stream |
| layer (layer, all/mixer/mlp) | output projection and/or `mlp.down_proj` | 0 makes the layer pass its input straight through |
| token (id) | that row of the embedding | tied to the output head, so it changes reading *and* saying the token |
| weight (any tensor, rows/cols) | a raw slice | anything else |

Operations: **scale** (0 = off, 0.5 = turn down, 2 = boost) and **noise**
(Gaussian, relative to the slice's RMS, seeded).

Edits are writes into the in-memory weights. The GGUF on disk is never touched.
Each edited tensor keeps a pristine copy and is rebuilt from it plus all active
edits, so undoing any edit, in any order, is exact.

Each reply re-reads the whole chat with the current weights. "Ask again" reuses
the first answer's seed, so a different answer comes from the edits, not from
sampling luck.

## Qwen3.5 0.8B, briefly

24 layers. Layers 3, 7, 11, 15, 19 and 23 use regular attention (8 heads, 2 KV
heads, head dim 256, gated output). The other 18 are **Gated DeltaNet**: linear
attention that carries a running state instead of a KV cache (16 heads of 128).
Every layer has a SwiGLU MLP with 3,584 neurons.

## Layout

```
src/lobotomy/
  gguf_loader.py    GGUF → float weights with Hugging Face names and conventions
  tokenizer.py      tokenizer and chat template, built from the GGUF alone
  edits.py          what you can break, and how it maps onto weights
  engine.py         chat, ask again, edits, damage, logit lens
  analysis.py       layer activity, top neurons, sparklines
  sampling.py       numpy sampler shared by all backends
  backends/
    base.py         the backend contract
    mlx/            Apple Silicon (done)
    cuda/           NVIDIA (stub)
  cli.py            terminal interface
tests/              pytest; test_backend_contract.py runs against any backend
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The reference tests compare against mlx-lm's own Qwen3.5 on the official
weights. They are skipped unless the official checkpoint is in
`models/qwen3.5-0.8b-hf/` (`config.json`, `tokenizer.json` and the safetensors
from `Qwen/Qwen3.5-0.8B`).

## Writing the CUDA backend

Implement `lobotomy.backends.base.Backend` in `backends/cuda/backend.py`; the
docstring there has a suggested route. Then run the contract tests:

```bash
LOBOTOMY_TEST_BACKENDS=cuda python -m pytest tests/test_backend_contract.py tests/test_engine.py
```

## Credits

The MLX model follows mlx-lm's Qwen3.5 implementation and uses its Metal
gated-delta kernel (MIT, Apple). Weights: Qwen3.5 by the Qwen team (Apache-2.0),
quantized by Unsloth.
