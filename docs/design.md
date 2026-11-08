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

## Built so far (2026-11-07)

Everything under "Screen" above is built, plus the following:

- **Damage meter:** KL divergence from the healthy model on the latest reply,
  shown live in the top bar.
- **What matters here?** An ablation scan. It cuts each layer, head or top
  neuron alone and measures how much the reply would change.
- **Why this word?** Gradient attribution over all 86k neurons and 336 heads
  toward a chosen token, with "Make it stop saying it" (cut the top ten).
- **Logit lens:** at every layer, for a clicked word.
- **Neuron vocabulary:** the tokens a neuron's output pushes up or down.
- **Keep-memory mode:** the "Re-read the whole chat" checkbox.
- **Recipes:** save and load edits as JSON.
- **Terminal interface:** `lobotomy chat`.

## Decisions made while building (Claude's calls, open to change)

- **Compute dtype is bfloat16.** It sits slightly closer to the reference than
  float16 (KL 0.024 vs 0.028), and the bf16 weights are cached after the first
  load.
- **Neuron edits scale the neuron's row of `mlp.up_proj`.** The activation then
  scales exactly by the factor, so recorded activations show the edit.
- **Head edits scale the head's columns of the output projection.** The sigmoid
  output gate can't reach zero, so it isn't a real off switch.
- **"Layer off" scales the layer's output projections to 0.** The layer then
  passes the residual stream through untouched.
- **Edits on norm weights act on the gain (1 + w),** so "×0" silences a norm
  rather than resetting it to a gain of 1.
- **Ask again reuses the first answer's seed,** so differences come from the
  edits.
- **Re-read is the default.** Keep memory is opt-in.
- **Activity colours are relative to each layer's own range** (the strips use
  min to max over the reply), because absolute values differ about 10× between
  early and late layers.
- **Noise edits have no hue in the design.** They use the neutral edit colour
  with a dashed ring.
- **The theme follows the system setting** until toggled. The design opened
  dark always.

### The 3D layer stack (2026-11-07, first build; the look is still open)

The user chose stacked sheets from three shapes mocked up with real data, and
asked for a timeline, moving lines and thicker lines for harder firing. These
are the calls made while building it:

- **It replaces the strips by default,** with a 3D / Activity / Lens switch, so
  the 2D strips, the lens and the cut cost are one click away.
- **One timeline step per pass that wrote a word.** The first step is the pass
  over the last prompt token (it wrote the first word); each step "reads" one
  token and "says" the next. Layers above the playhead keep the previous word,
  dimmed, since that is the model's state mid-pass.
- **Lines go to each layer's 48 strongest neurons** (the replay carries the
  top 96 per layer and word; the live stream the top 32). Strength is the size
  of the activation against the layer's 99.9th percentile over the reply, the
  same reference the "why" ranking uses. Lines start at the line through the
  sheets, because every neuron reads from and writes to that residual stream.
- **Live while it writes, replay afterwards.** At about 50 words a second the
  live view is a light show; the timeline then swaps in the full data, paused at
  the end. Play starts from the top. 1× takes 1.5 s per word; ¼× to 8×.
- **Keys:** Space plays or pauses, ← → step one layer, Shift ← → one word.
- **A clicked neuron opens the same editor as the layer view,** and lights the
  timeline by that neuron's activity (where it fired). Edited neurons are drawn
  whole in their edit colour (blue for turned down, orange for boosted, a ring
  cut through for off), larger than the rest and undimmed, so edits show from
  anywhere in the tower. When an edit lands or changes, the neuron flashes and
  swells and a ripple in its colour spreads from it. Whole-layer edits tint the
  sheet. With reduced motion there is no flash or ripple.
- **Heads are tiles along each sheet's far edge,** height and lightness by the
  head's output size against the layer's 99th percentile over the reply, with
  their own lines from the middle. A pulled-out layer's heads are clickable
  and open the head editor.
- **Wiring is read from the healthy weights,** like "pushes up": the 4
  strongest connections into every later layer, up[m] · ((1 + g) ⊙ d).
  Solid lines boost and dashed lines suppress (sign flipped when the neuron
  fired negative at that word). The popover lists the strongest six; clicking
  one pulls out that layer and opens that neuron.
- **Word probabilities:** a *Next word* card (top left, clear of the layer
  labels; top right while a sheet is pulled out) lists the 5
  candidates the model weighed, with the model's own probabilities (before
  temperature, top-k and top-p; the readout uses the same number). Beside
  each layer, once the word has reached it: that layer's best guess (logit
  lens) and a bar for the probability it already gave the word that came.
  Fetched per step from `GET /api/replies/{id}/guess/{row}`.
- **Hovering an edit** in the rail puts a beacon on it in the tower: a line
  from the neuron, head tile or sheet up out of the top, in the edit's
  colour, with its label.
- **View controls sit on the stage, open by default:** *↔ Within a layer*
  (neuron spacing, horizontal), *↕ Between layers* (layer spacing, vertical),
  neuron size, *Fit to screen*, *Full window* and *Reset*, remembered per
  browser. The user asked for horizontal and vertical scaling and for the
  structure to fill the screen.
- **Fit to screen:** the camera keeps its three-quarter view from above and
  moves so the whole tower fills 90% of the stage, leaving room for the view
  controls on the left and the layer labels on the right. It fits on open,
  on resize and while you drag the sliders, until you orbit or zoom yourself;
  *Fit to screen* hands it back. A pulled-out sheet fills about 85% of the
  stage, and the view controls step aside for it.
- **Full window** (button or F, Esc to leave) lays the 3D view over the whole
  page, timeline included.
- **Atmosphere:** the stage darkens towards its edges (`--surface-deep`), and
  neurons that fire hard glow softly. The glow ignores depth, so the tower
  reads as lit from inside, and fades towards the back. Defaults are layers
  0.16 apart and neurons 0.044 wide, larger than the first build.
- **Attention is drawn into the timeline:** when the playhead (or a pulled-out
  layer) is on an attention layer and paused, lines run from the sheet, or from
  a clicked head, down to the words it looked at most. "Your message" collects
  the prompt and anything earlier in the chat. DeltaNet heads have no
  attention weights to draw, and their popover says so.

## Non-goals for now

- vision input
- editing while the model is typing
- the CUDA backend (left to the user)
- training or fine-tuning
