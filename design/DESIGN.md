# lobotomy design language

lobotomy is a precise instrument with a dark sense of humour. The values live in
`tokens.css`; `mockups/index.html` shows everything together.

## Principles

1. **Light shows what the model did. Colour shows what you did.** Activity is a
   near-neutral lightness ramp. The only hues anywhere are the two edit poles:
   cold for turned down or off, hot for boosted. There is no accent colour and no
   red; errors are bone text, a rule and the word "Error".
2. **Off is a cut.** The logo's cut, a straight kerf with the top slipped right,
   marks "off" on swatches, cells and heads, so off never depends on colour.
3. **An instrument, not a dashboard.** Keep it dense and aligned, print real
   numbers, and animate only in answer to an action.
4. **The humour lives in the name and a few plain lines** ("Back to healthy"),
   never in the data.

## Data colours

- **Activity**, `--act-0` to `--act-8`, shows how hard something fired
  (|activation|). Bins are relative to each layer's usual range, so the busy last
  layers don't wash everything out. Always print the raw value.
- **Edits**, `--edit-down-4` (off) to `--edit-down-1`, then `--edit-none`, then
  `--edit-up-1` to `--edit-up-4`. Steps follow log2 of the scale: ×½, ×¼, ×⅛ and
  off on one side, ×2, ×4, ×8 and ×16 on the other. The slider detents there.
- **Fill is the model, the ring is you.** An edited cell keeps its activity fill
  and gains a 2px edit-colour ring, with a 1px surface gap between them.
- Text never takes a data colour. Print values in ink beside the swatch. On an
  activity fill, use `--act-ink-N`; step 4 is for large numerals only.
- Dark is the default, because a few hot neurons glow on a dark well. In light,
  activity becomes ink density and edits darken with strength. `data-theme`
  works on any element.

Measured: text is at least 5:1. Activity lightness rises strictly, including
under protanopia, deuteranopia and tritanopia simulation. Edit marks are at least
3:1 on every surface, and cold and hot at the same step differ by at least 18 ΔE
under every simulation.

## Type

Atkinson Hyperlegible Next for the interface and the model's replies, and
Atkinson Hyperlegible Mono for exact strings: tokens (`␣Canberra`), IDs
(`L14 N2817`) and scale values. Both are SIL OFL, so the app can bundle them. They
were chosen because Il1 and 0O never collide. Replies are 17/1.62, the interface
15/1.4 and dense data 13 with tabular figures. Use sentence case, with no all-caps
labels.

## Components

- **Shell:** context on the left (chat or layer list), the inspector in the
  middle, and the edits rail on the right on every screen.
- **Layer strip:** number, type in words, one cell per token, raw peak, edit
  swatches. Six blocks of four, because every fourth layer is attention.
- **Neuron grid:** 64 × 56 cells, 8px with 1px gaps. Never draw all 86k neurons.
- **Edit popover:** Off, a log slider from off to ×16, Boost ×2, Reset and a
  readout. It applies from the next reply.
- **Buttons:** at most one bone primary per screen, always the next step
  (usually "Ask again").

Spacing is a 4px grid. Radii are 1px for cells, 3px for controls and 6px for
popovers. Hairlines divide panes; only popovers cast shadows.

## Motion

Activity re-lights with a 220ms fade, popovers take 140ms, and the cut draws
across an "off" cell in 320ms. No bounce. Reduced motion sets every duration
to 0.

## Logo

Use direction A, Severed. The cut appears once per artwork, so the lockup pairs
the mark with whole o's. Use one colour only, from the `on-dark` or `on-light`
file. `favicon.svg` draws the cut heavier for 16px.

## Do and don't

- Do put each number next to the thing it measures.
- Do name layer types in words, not colour.
- Don't use edit colours for diffs, activity changes, links or focus. Focus and
  selection are bone.
- Don't add a hue. Use weight, a rule or a word instead.
