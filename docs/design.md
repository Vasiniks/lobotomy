# lobotomy — design

## What it is

A local tool for breaking a small LLM on purpose while chatting with it, and for
seeing inside it so you can choose what to break. The starting model is
**Qwen3.5 0.8B**, Unsloth's **Q4_K_M GGUF**.

Breaking is the main point. Visualization exists to help you pick what to break
and to show what the damage did.

## Core loop (agreed)

1. Chat with the model. It replies.
2. Look inside: which neurons fired, in which layer.
3. Click into a layer and edit it: switch neurons or heads off, turn them down,
   boost them, or wipe the whole layer.
4. Ask again, and see how the answer changed.

Edits happen **between turns**, not while the model is typing.

## Screen (proposed, to be mocked up)

These are proposals, not decisions yet.

- **Chat** on the left.
- **Layer stack** on the right: 24 strips, one per layer, lit by how hard that
  layer's neurons fired during the last reply. Each strip shows the layer type,
  since only 6 layers use regular attention (see Model facts).
- **Layer view** (click a strip):
  - a grid of the layer's 3,584 MLP neurons, glowing by activity, with the
    strongest ones listed first
  - the layer's heads: 8 in attention layers, 16 in DeltaNet layers
- **Edit** (click a neuron, a head, or the layer itself): off, scale (turn down
  or boost), or reset.
- **Token inspect**: click a word in a reply to see which neurons fired for that
  word.
- **Edit list**: every active edit, each with its own undo, plus a "back to
  healthy" reset.
- **Ask again**: reruns the last question so the old and new answers sit side by
  side.
- After an edit, the next turn **re-reads the whole chat with the edited
  weights** by default, so the damage is consistent. Optionally the model can
  keep its healthy memory of the chat and generate only new tokens with the
  edited weights.

Ideas from existing tools, to consider later: a damage meter (KL divergence from
the healthy model), a per-token logit-lens heatmap, saved and shareable edit
recipes, and choosing cuts by activation similarity.

## Architecture

- A **Python engine** with swappable backends:
  - **MLX** (Apple Silicon): the main backend, built first.
  - **CUDA** (PyTorch): an interface stub for now. The user will build it.
- A **shared GGUF reader** dequantizes the Q4 file into float weights with
  Hugging Face names and conventions, so every backend loads the same thing. The
  GGUF file on disk is never modified; it is the healthy original.
- **Edits are writes into in-memory weights.** A pristine copy stays alongside,
  so undo restores the exact original values.
- **Activations are recorded per turn:**
  - the MLP neuron activations, `silu(gate) * up` (3,584 per layer per token)
  - per-head output magnitudes
  - attention probabilities for the 6 full-attention layers
- **UI:** a local web page served by a local Python server. This comes after
  the engine.

## Model facts (Qwen3.5 0.8B)

- 24 layers, hidden size 1024, MLP size 3584, vocabulary 248,320, tied
  embeddings.
- **Full-attention layers** are 3, 7, 11, 15, 19 and 23:
  - 8 query heads and 2 KV heads, head dim 256
  - a sigmoid output gate and q/k-norm
  - partial RoPE, on 64 of the 256 dims
- **The other 18 layers are Gated DeltaNet** (linear attention with recurrent
  state): 16 heads of 128 dims, conv kernel 4.
- Multimodal upstream, but the main GGUF is text only. Vision lives in a
  separate mmproj file.
- Repo: `unsloth/Qwen3.5-0.8B-GGUF` (Apache-2.0).

## Non-goals for now

- vision input
- editing while the model is typing
- the CUDA backend (left to the user)
- training or fine-tuning
