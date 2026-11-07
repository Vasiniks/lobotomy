# How lobotomy works

This page covers every number the tool shows. Each one comes with what question
it answers, what it gives you beyond "the answer changed", and a way to check it
yourself. The checks run on Qwen3.5 0.8B with greedy sampling (Settings →
temperature 0) and the question "What is the capital of Australia? Answer in two
sentences."

## The model in one picture

A word enters the model as a list of 1,024 numbers, a vector. That vector is the
**residual stream**. Each of the 24 layers reads it and adds two corrections:
one from its mixer, then one from its MLP.

- The **mixer** looks back at earlier words. It's attention in layers 3, 7, 11,
  15, 19 and 23 and DeltaNet in the other 18.
- The **MLP** has 3,584 neurons.

After the last layer, the vector is compared against all 248,320 vocabulary
entries to score every possible next word.

So every part you can break does one thing: it writes a correction into that
vector.

**Check:** switch a whole layer off ("Edit whole layer" → Off). The layer then
adds nothing, so its output vector equals its input. `tests/test_backend_contract.py::test_layer_off_passes_residual_through`
asserts exactly that.

## Neurons

A neuron's activation is `silu(gate · x) × (up · x)`. Here `x` is the residual
vector, and `gate` and `up` are two rows of 1,024 weights. The neuron adds
`activation × (its column of down_proj)` to the stream.

**Switching off or scaling a neuron** multiplies its `up` row by a factor. The
activation is linear in `up` (double `up`, double the product), so the
neuron's activation and its effect scale by exactly that factor.

**Check:** switch off a strong neuron in a layer, ask again, and open the same
layer. That neuron's cell is now dark. `test_neuron_off_zeroes_its_activation`
checks that it is exactly 0, and that ×0.5 halves it.

**"When it fires, it pushes up…"** multiplies the neuron's `down_proj` column by
the vocabulary matrix. The result lists the words its correction pushes up and
down when it fires positively. If the neuron fired negatively, the lists swap,
and the popover says so. This reading ignores everything that happens in later
layers, so treat it as a hint.

## Heads

A head's output is a 128-number vector (DeltaNet) or a 256-number vector
(attention). The output projection maps it into the stream. Scaling the head's
columns of that projection scales its contribution exactly. The number inside
each head cell is the length of that contribution, averaged over the reply.

**Check:** switch off a head and ask again. Its number drops to 0.

## Damage meter (KL divergence)

**Question:** how differently does the broken model predict the same text?

**What it buys:** a number even when the visible answer didn't change, and 0
exactly when nothing changed.

At each word of the latest reply, the healthy model gives a probability to
every possible next word, a list `p`, and the edited model gives a list `q`. KL
divergence is `Σ p · ln(p / q)`, averaged over the reply's words. It is 0 only
when `q = p`, and it grows as the edited model moves probability away from where
the healthy one put it. The unit is the nat (natural log).

**Check by hand:** with two possible words, `p = (0.9, 0.1)` and
`q = (0.5, 0.5)` give `0.9·ln 1.8 + 0.1·ln 0.2 = 0.368`. A meter around 0.4
means a shift that size, on average, per word.

**Check in the tool:** with no edits the top bar says "Healthy". Undo every edit
and the meter disappears again.

## What matters here? (cut cost)

**Question:** if I cut this one part on top of my current edits, how much would
this reply change?

**What it buys:** a ranking before you cut anything.

Each part is really switched off, one at a time. The model re-reads the reply,
and the damage (KL, as above) is measured. Nothing is estimated here.

**Check:**

1. With no edits, click *What matters here?* and note a layer's cut cost.
2. Switch that layer off and read the damage meter.

The two numbers are the same, because the meter measures the same thing after
the cut. Measured: layer 0 gives 8.3128 both ways, layer 2 gives 0.7932, and
layer 23 gives 1.8580.

## Why this word? (gradient attribution)

**Question:** which neurons and heads pushed the model toward this word?

**What it buys:** one pass over all 86,016 neurons and 336 heads, instead of
86,352 re-runs.

Imagine a dial on every neuron and head, all set to 1. The model's
log-probability of the word is a function `f` of all those dials. Calculus
gives the slope `f'` for every dial at once (one backward pass). Turning one
dial from 1 to 0 changes `f` by about `−f'`; that is the first-order estimate
`f(0) ≈ f(1) − f'(1)`. "Effect if cut" is that estimate.

It's only the straight-line part of a curved function. In practice it has the
right sign, and the real effect is often about twice as big.

**Check:** after a "Why this word?", note the top unit's effect and the
probability. Switch that unit off alone and ask again with the same seed, then
open the word view on the same word and compare the new probability with
`p × e^effect`. Measured for "Canberra":

| Unit | Estimated change in log p | Actual | p before → after |
|---|---|---|---|
| L18 H9 | −0.21 | −0.41 | 0.78 → 0.52 |
| L18 N2065 | −0.17 | −0.35 | 0.78 → 0.55 |
| L17 N2117 | −0.17 | −0.30 | 0.78 → 0.58 |

The ranking is the useful part. "Make it stop saying" cuts the top ten together.
In one run (asking for a one-sentence answer), that took "Canberra" from 0.68
to 0.05, and the model answered "Sydney".

## Logit lens

**Question:** at which layer did the model already "know" the next word?

**What it buys:** a picture of depth. Activity tells you a layer was busy; the
lens tells you whether that work was about the answer.

Take the residual stream after layer L and score it against the vocabulary as
if L were the last layer (the same final norm and comparison the model really
uses). The word view lists each layer's top guesses. Lens mode on the layer
stack shows, for every word of the reply, how much probability each layer
already gave the word that actually came next.

**Check:** the last layer's lens is the model's real output. In Lens mode, hover
a word's cell in layer 23. Its `p` matches the probability shown for that same
word among the candidates in its word view.
`test_lens_grid_last_layer_is_the_real_output_probability` checks this for every
word.

## The 3D view

**Lines to neurons.** At every layer, lines go from the line through the
sheets (the residual stream) to that layer's 48 strongest neurons for the
word. Thickness is the size of the activation against the layer's 99.9th
percentile over the reply, the same reference "Why this word?" uses.
**Check:** pull out a layer at a word and open the 2D layer view for the same
word (Open in 2D, then pick the word). The thickest lines go to the brightest
cells.

**Head tiles.** A tile's height and lightness is the size of what that head
wrote into the stream for this word (the same number as in the head cells),
against the layer's 99th percentile over the reply. **Check:** the word view
lists every head's value for the word; the tallest tile has the largest.

**Wiring.** A neuron writes its `down_proj` column `d` into the stream. A
later MLP reads the stream through its norm gain `1 + g` and its `up_proj`
rows, so the connection to neuron `m` there is `up[m] · ((1 + g) ⊙ d)`: how
far one unit of this neuron's activation moves `m`'s input. Positive means it
boosts `m`, negative means it suppresses it; if the neuron fired negative at
this word, the signs flip. It is read from the healthy weights and ignores the
mixers in between, the gate, and the norm's division by the stream's size, so
treat it like "pushes up": a map of where the signal is sent, not a measured
effect. **Check:** `test_neuron_wiring_reads_the_healthy_weights` recomputes
it with numpy.

**Where attention looked.** From the word at the playhead, each attention
head splits 100% of its attention over every earlier position. The lines go
to the words with the largest share (averaged over the heads, or one head if
you clicked it); "your message" adds up the prompt and earlier chat.
**Check:** the word view's attention section lists the same words and
percentages for that word and layer, and `test_server.py` checks every head
sums to 1.

## Quantization

The GGUF stores most weights in 4 to 6 bits per number. lobotomy expands them
to 16-bit floats in memory, which is what the edits change. The file on disk
never changes.

**Check:** `tests/test_reference.py` compares the outputs with the official
full-precision checkpoint. The mean KL is about 0.024, ordinary Q4 rounding;
switching off layer 2 is about 0.8 by comparison.

## Asking again

"Ask again" re-reads the whole chat with the current weights, then samples with
the same random seed as the first answer. With no edits it gives the identical
answer, so any difference comes from your edits. Unticking "Re-read the whole
chat" keeps the model's earlier reading of the conversation; then only the new
answer uses the edited weights.

**Check:** with no edits, ask again. The answer is identical, word for word
(`test_ask_again_without_edits_reproduces`).
