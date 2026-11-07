// Layer view: minimap | one layer opened up | edits rail.

import { api } from "../api.js";
import * as S from "../store.js";
import { state } from "../store.js";
import { closePopover, layerTypeName, openEditPopover, renderRail, renderStrips, swatch } from "../components.js";
import { actBin, clear, editClass, editValueText, fmt, fmtCost, h, percentile, showToken } from "../util.js";

const COLS = 64;
let rowsMode = "reply"; // "reply" or a recording row (number) for one token
let costOn = false;

export function setLayerRow(row) {
  rowsMode = row === null || row === undefined ? "reply" : row;
}

export async function renderLayerView(body, layer) {
  const last = S.lastReply();
  body.className = "body is-layer";
  if (!last) {
    clear(body);
    body.append(
      h("div", { class: "loading" }, h("p", {}, "Chat with the model first; layers fill in from its reply. ", h("a", { href: "#/" }, "Back to chat"))),
    );
    return;
  }
  const detail = await S.ensureDetail(last.id);
  const rowsKey = rowsMode === "reply" ? "reply" : String(rowsMode);
  const data = await S.layerData(last.id, layer, rowsKey);
  // the route may have changed while we waited
  if (state.route.view !== "layer" || state.route.layer !== layer) return;

  const scroll = body.querySelector(".layer")?.scrollTop ?? 0;
  clear(body);
  const minimap = h("nav", { class: "minimap", "aria-label": "Layers" }, h("div", { class: "pane-head" }, h("h2", {}, "Layers")));
  const strips = h("div", { class: "strips" });
  minimap.append(h("div", { class: "stack-wrap" }, strips));
  renderStrips(strips, detail, { compact: true, selectedLayer: layer, onLayer: (L) => (location.hash = `#/layer/${L}`) });

  const section = h("section", { class: "layer", "aria-labelledby": "layer-title" });
  const rail = h("aside", { class: "rail", "aria-label": "Edits" });
  body.append(minimap, section, rail);
  renderRail(rail);
  renderLayer(section, layer, last, detail, data);
  section.scrollTop = scroll;
}

function tokenText(detail, row) {
  return detail.tokens[row]?.text ?? "";
}

function renderLayer(section, layer, reply, detail, data) {
  const m = state.model;
  const attn = data.type === "attention";
  const nN = data.grid.length;
  const peak = data.peak;
  const selectedTok = rowsMode === "reply" ? null : detail.tokens[rowsMode];
  const lastTokRow = detail.tokens.filter((t) => t.role === "reply").map((t) => t.row);

  const typeDesc = attn
    ? `Gated attention. ${nN.toLocaleString()} MLP neurons, ${data.n_heads} heads of ${256} dims (2 shared key/value heads).`
    : `Gated DeltaNet. ${nN.toLocaleString()} MLP neurons, ${data.n_heads} heads of 128 dims.`;

  const costN = costOn ? S.scanResult(reply.id, "neurons", layer) : null;
  const costH = costOn ? S.scanResult(reply.id, "heads", layer) : null;

  const layerTarget = (part) => ({ kind: "layer", layer, part });
  const wholeBtn = h(
    "button",
    {
      class: "btn",
      type: "button",
      "data-pop-anchor": "",
      onclick: (e) => openLayerPopover(e.currentTarget, layer, attn),
    },
    "Edit whole layer",
  );

  section.append(
    h(
      "header",
      { class: "layer-head" },
      h(
        "div",
        {},
        h("h1", { id: "layer-title" }, `Layer ${layer}`),
        h(
          "p",
          { class: "meta" },
          typeDesc,
          peak ? ` Peak ${fmt(peak.peak)} on “${showToken(tokenText(detail, peak.peak_row))}”.` : "",
        ),
      ),
      h(
        "div",
        { class: "layer-tools" },
        selectedTok
          ? h(
              "div",
              { class: "seg", role: "group", "aria-label": "Show activity for" },
              h("button", { type: "button", "aria-pressed": "false", onclick: () => switchRows("reply", layer) }, "Whole reply"),
              h("button", { type: "button", "aria-pressed": "true" }, `“${showToken(selectedTok.text)}”`),
            )
          : null,
        h(
          "button",
          {
            class: "btn",
            type: "button",
            "aria-pressed": String(costOn),
            title: "Switch off each head and each of the strongest neurons alone, and measure how much the reply would change",
            onclick: async (e) => {
              costOn = !costOn;
              if (costOn) {
                e.currentTarget.textContent = "Measuring…";
                await Promise.all([S.scan(reply.id, "heads", layer), S.scan(reply.id, "neurons", layer)]);
              }
              S.emit();
            },
          },
          costOn ? "Hide cut cost" : "What matters here?",
        ),
        wholeBtn,
      ),
    ),
  );

  // --- neuron grid ---
  const ref = Math.max(percentile(data.grid, 99.5), 1e-6);
  const editsHere = S.editsInLayer(layer);
  const neuronEdits = new Map();
  for (const e of editsHere) if (e.target.kind === "neuron") neuronEdits.set(e.target.index, e);
  const headEdits = new Map();
  for (const e of editsHere) if (e.target.kind === "head") headEdits.set(e.target.index, e);

  const grid = h("div", {
    class: "ngrid",
    role: "img",
    "aria-label": `Activity of all ${nN.toLocaleString()} neurons in layer ${layer}. ${data.n_above} fired above 1.0.`,
  });
  const frag = document.createDocumentFragment();
  for (let n = 0; n < nN; n++) {
    const e = neuronEdits.get(n);
    const el = document.createElement("i");
    el.className = `act-${actBin(data.grid[n], ref, 1)}${e ? ` ${editClass(e)}` : ""}`;
    el.dataset.n = n;
    frag.append(el);
  }
  grid.append(frag);
  grid.addEventListener("mousemove", (ev) => {
    const n = ev.target.dataset?.n;
    if (n !== undefined) ev.target.title = `N${n}: ${fmt(data.grid[n], 3)} (mean ${fmt(data.signed[n], 3)})`;
  });
  grid.addEventListener("click", (ev) => {
    const n = ev.target.dataset?.n;
    if (n !== undefined) openNeuron(ev.target, layer, Number(n), reply, detail, data);
  });
  grid.dataset.popAnchor = "";

  const nRows = Math.ceil(nN / COLS);
  const cell = 9;
  const gridFrame = h(
    "div",
    { class: "grid-frame" },
    h("div", { class: "grid-top", "aria-hidden": "true" }, [0, 16, 32, 48].map((c) => h("span", { style: { left: `${1 + c * cell}px` } }, `+${c}`))),
    h(
      "div",
      { class: "grid-left", "aria-hidden": "true" },
      Array.from({ length: Math.ceil(nRows / 8) }, (_, i) => h("span", { style: { top: `${1 + i * 8 * cell}px` } }, String(i * 8 * COLS))),
    ),
    grid,
  );

  // --- heads ---
  const headRef = Math.max(...data.heads.filter(Number.isFinite), 1e-6);
  const heads = h(
    "div",
    { class: `heads${data.n_heads === 8 ? " is-8" : ""}`, "aria-label": `${data.n_heads} ${attn ? "attention" : "DeltaNet"} heads` },
    data.heads.map((v, i) => {
      const e = headEdits.get(i);
      const c = costH?.[i];
      return h(
        "button",
        {
          class: "head-cell",
          type: "button",
          "data-pop-anchor": "",
          title: `Head ${i}: output size ${fmt(v, 3)}${c ? `; cut cost ${fmtCost(c.mean_kl)}` : ""}`,
          onclick: (ev) => openHead(ev.currentTarget, layer, i, attn, v, reply),
        },
        h("span", { class: `head act-${actBin(v, headRef)}${e ? ` ${editClass(e)}` : ""}` }, c ? (c.mean_kl >= 0.01 ? c.mean_kl.toFixed(2) : "0") : ""),
        h("span", { class: "head-idx num" }, String(i)),
      );
    }),
  );

  // --- strongest neurons ---
  const costByN = new Map((costN ?? []).map((c) => [c.target.index, c]));
  const top = data.top.slice(0, costOn ? 32 : 16);
  const maxPeak = Math.max(...top.map((t) => Math.abs(t.peak)), 1e-6);
  const table = h(
    "table",
    { class: "list" },
    h(
      "thead",
      {},
      h(
        "tr",
        {},
        h("th", {}, "Neuron"),
        h("th", { class: "r" }, "Peak"),
        h("th", {}, "On"),
        costN ? h("th", { class: "r" }, "Cut cost") : null,
        h("th", {}, "Edit"),
      ),
    ),
    h(
      "tbody",
      {},
      top.map((t) => {
        const e = neuronEdits.get(t.neuron);
        const c = costByN.get(t.neuron);
        const row = h(
          "tr",
          {
            "data-pop-anchor": "",
            onclick: (ev) => {
              if (ev.target.closest("button")) return;
              openNeuron(row, layer, t.neuron, reply, detail, data, t);
            },
          },
          h("td", { class: "mono" }, String(t.neuron)),
          h(
            "td",
            { class: "r num" },
            fmt(t.peak),
            h("span", { class: `bar act-${actBin(Math.abs(t.peak), maxPeak)}` }, h("b", { style: { width: `${(Math.abs(t.peak) / maxPeak) * 100}%` } })),
          ),
          h("td", { class: "mono" }, showToken(tokenText(detail, t.peak_row))),
          costN ? h("td", { class: "cost" }, c ? fmtCost(c.mean_kl) : "") : null,
          h(
            "td",
            {},
            e
              ? h("span", { class: "edit-tag" }, swatch(e), editValueText(e))
              : h(
                  "span",
                  { class: "row-actions" },
                  h(
                    "button",
                    {
                      class: "btn btn-sm",
                      type: "button",
                      onclick: () => S.setScale({ kind: "neuron", layer, index: t.neuron }, 0, neuronNote(t, detail)),
                    },
                    "Off",
                  ),
                ),
          ),
        );
        return row;
      }),
    ),
  );

  const offTop = h(
    "button",
    {
      class: "btn btn-sm",
      type: "button",
      onclick: () =>
        S.addMany(
          data.top
            .slice(0, 10)
            .filter((t) => !neuronEdits.has(t.neuron))
            .map((t) => ({ target: { kind: "neuron", layer, index: t.neuron }, value: 0, note: neuronNote(t, detail) })),
        ),
    },
    "Switch off the top 10",
  );

  section.append(
    h(
      "div",
      { class: "layer-grid" },
      h(
        "div",
        { class: "grid-area" },
        h(
          "h2",
          { class: "sub" },
          "Neurons ",
          h("span", { class: "meta" }, `${COLS} × ${nRows}, one cell each, ${rowsMode === "reply" ? "strongest value over the reply" : "value on this word"}. ${data.n_above} fired above 1.0.`),
        ),
        gridFrame,
        h("h2", { class: "sub" }, "Heads ", h("span", { class: "meta" }, costH ? "numbers are cut cost" : "output size over the reply")),
        heads,
      ),
      h(
        "div",
        { class: "top-list" },
        h(
          "h2",
          { class: "sub" },
          "Strongest neurons ",
          h("span", { class: "meta" }, `${top.length} of ${nN.toLocaleString()}.`),
        ),
        table,
        h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, offTop),
        h(
          "p",
          { class: "meta list-note" },
          "Negative values fired just as hard in the other direction. Cell brightness follows the size; the sign is in the number.",
          costN ? " Cut cost: how much the reply would change if that neuron alone were switched off (KL, nats)." : "",
        ),
      ),
    ),
  );
}

function neuronNote(t, detail) {
  return t ? `Fired ${fmt(t.peak)} on “${showToken(tokenText(detail, t.peak_row))}”.` : "";
}

function switchRows(mode, layer) {
  rowsMode = mode;
  closePopover();
  renderLayerView(document.getElementById("body"), layer);
}

function openNeuron(anchor, layer, n, reply, detail, data, top) {
  const t = top ?? data.top.find((x) => x.neuron === n);
  const info = api.neuron(reply.id, layer, n);
  const note = t ? neuronNote(t, detail) : `Average ${fmt(data.grid[n], 3)} over the reply.`;
  openEditPopover(anchor, {
    target: { kind: "neuron", layer, index: n },
    title: `Neuron ${n}`,
    sub: `layer ${layer}`,
    info: t
      ? `Peak ${fmt(t.peak)} on “${showToken(tokenText(detail, t.peak_row))}”. Above 1.0 on ${t.n_above} of the recorded tokens.`
      : `Average size ${fmt(data.grid[n], 3)} over the reply.`,
    note,
    says: info.then((x) => x.says),
    trace: null,
    sign: t ? Math.sign(t.peak) || 1 : Math.sign(data.signed[n]) || 1,
  });
  // add the trace once it arrives
  info.then((x) => {
    const pop = document.querySelector(".popover");
    if (!pop || !x.trace) return;
    const rows = detail.tokens.filter((tk) => tk.role === "reply").map((tk) => tk.row);
    import("../components.js").then(({ sparkline }) => {
      const sp = sparkline(rows.map((r) => x.trace[r] ?? 0));
      pop.querySelector(".popover-head").after(sp);
    });
  });
}

function openHead(anchor, layer, i, attn, v, reply) {
  openEditPopover(anchor, {
    target: { kind: "head", layer, index: i },
    title: `Head ${i}`,
    sub: `layer ${layer}, ${attn ? "attention" : "DeltaNet"}`,
    info: `Output size ${fmt(v, 3)} on average over the reply. Scaling a head scales everything it writes into the residual stream.`,
    note: `${attn ? "Attention" : "DeltaNet"} head.`,
  });
}

function openLayerPopover(anchor, layer, attn) {
  // Three targets: the whole layer, its mixer (attention / DeltaNet) or its MLP.
  const parts = [
    ["all", "Whole layer"],
    ["mixer", attn ? "Attention" : "DeltaNet"],
    ["mlp", "MLP"],
  ];
  let part = "all";
  const open = () =>
    openEditPopover(anchor, {
      target: { kind: "layer", layer, part },
      title: `Layer ${layer}`,
      sub: parts.find((p) => p[0] === part)[1],
      info:
        part === "all"
          ? "Scales everything this layer adds to the residual stream. Off makes the layer pass its input straight through."
          : part === "mlp"
            ? "Scales the MLP's output (all 3,584 neurons at once)."
            : `Scales the ${attn ? "attention" : "DeltaNet"} block's output.`,
      note: `Whole ${parts.find((p) => p[0] === part)[1].toLowerCase()}.`,
    });
  open();
  const pop = document.querySelector(".popover");
  const seg = h(
    "div",
    { class: "seg", role: "group", "aria-label": "Part of the layer" },
    parts.map(([p, label]) =>
      h(
        "button",
        {
          type: "button",
          "aria-pressed": String(p === part),
          onclick: () => {
            part = p;
            open();
            const again = document.querySelector(".popover");
            again.querySelector(".popover-head").after(seg);
            seg.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", String(parts[j][0] === part)));
          },
        },
        label,
      ),
    ),
  );
  pop.querySelector(".popover-head").after(seg);
}
