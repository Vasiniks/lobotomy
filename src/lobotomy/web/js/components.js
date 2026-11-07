// Pieces shared by every view: top bar, edits rail, layer strips, edit popover.


import * as S from "./store.js";
import { state } from "./store.js";
import {
  clear, editClass, editValueText, fmt, fmtScale, h, percentile, rangeBin, s, showToken, SLIDER, sliderToValue, valueToSlider,
} from "./util.js";

// --- labels ---------------------------------------------------------------------------

export function layerTypeName(layer) {
  return state.model?.layers?.[layer]?.type === "attention" ? "Attention" : "DeltaNet";
}

export function shortLabel(t) {
  switch (t.kind) {
    case "neuron":
      return `L${t.layer} N${t.index}`;
    case "head":
      return `L${t.layer} H${t.index}`;
    case "layer":
      return t.part === "all" ? `L${t.layer} whole` : t.part === "mlp" ? `L${t.layer} MLP` : `L${t.layer} ${layerTypeName(t.layer)}`;
    case "token":
      return `token ${t.index}`;
    default:
      return t.tensor?.replace("model.layers.", "L").replace(".weight", "") ?? "weight";
  }
}

export function swatch(edit, lg = false) {
  return h("i", { class: `swatch ${lg ? "swatch-lg " : ""}${editClass(edit)}`, "aria-hidden": "true" });
}

// --- top bar ----------------------------------------------------------------------------

export function renderTopbar(el, crumbs = []) {
  clear(el);
  const m = state.model;
  const quant = m?.model_file?.match(/(Q\d_[A-Z0-9_]+|IQ\d_[A-Z]+|BF16|F16)/)?.[0] ?? "";
  const n = state.edits.length;

  const damage = state.damage;
  let damageEl = null;
  if (damage && lastReplyId() === damage.replyId && !damage.healthy && n) {
    const w = Math.min(100, (damage.mean_kl / 2) * 100);
    damageEl = h(
      "div",
      { class: "damage", title: "How far the edited model's predictions are from the healthy model's, on the latest reply (mean KL divergence, nats)" },
      "Damage ",
      h("b", {}, fmt(damage.mean_kl, 3)),
      h("span", { class: "meter" }, h("i", { style: { width: `${w}%` } })),
    );
  }

  el.append(
    ...[
    h(
      "a",
      { class: "brand", href: "#/", "aria-label": "lobotomy: back to the chat" },
      h("img", { class: "on-dark", src: "/assets/img/lockup-on-dark.svg", alt: "lobotomy" }),
      h("img", { class: "on-light", src: "/assets/img/lockup-on-light.svg", alt: "lobotomy" }),
    ),
    h(
      "div",
      { class: `model${state.busy ? " is-busy" : ""}` },
      h("span", { class: "status-dot", title: state.busy ? "Generating" : "Ready" }),
      h("strong", {}, m?.name ?? "…"),
      quant && h("span", {}, quant),
      m && h("span", {}, m.backend.toUpperCase()),
    ),
    crumbs.length
      ? h(
          "nav",
          { class: "crumbs", "aria-label": "Breadcrumb" },
          crumbs.map((c, i) => [
            i ? h("span", { "aria-hidden": "true" }, "/") : null,
            c.href ? h("a", { href: c.href }, c.label) : h("span", { "aria-current": "page" }, c.label),
          ]),
        )
      : null,
    h(
      "div",
      { class: "topbar-end" },
      damageEl,
      h(
        "div",
        { class: `health${n ? "" : " is-healthy"}`, title: n ? `${n} edits active` : "No edits: the model is healthy" },
        n ? h("span", { class: "swatches" }, state.edits.slice(0, 8).map((e) => swatch(e))) : null,
        n ? `${n} edit${n === 1 ? "" : "s"}` : "Healthy",
      ),
      h("button", { class: "btn btn-quiet", type: "button", onclick: (e) => openSettings(e.currentTarget) }, "Settings"),
      h(
        "button",
        { class: "btn btn-quiet", type: "button", onclick: toggleTheme },
        document.documentElement.dataset.theme === "light" ? "Dark theme" : "Light theme",
      ),
    ),
    ].filter(Boolean),
  );
}

function lastReplyId() {
  return S.lastReply()?.id ?? null;
}

export function toggleTheme() {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("lobotomy-theme", next);
  } catch {}
  import("./util.js").then((u) => u.refreshColours());
  S.emit();
}

// --- edits rail -------------------------------------------------------------------------

export function renderRail(el) {
  clear(el);
  const edits = state.edits;
  const canAsk = state.exchanges.length && !state.busy;
  const fileInput = h("input", {
    type: "file",
    accept: "application/json,.json",
    class: "visually-hidden-file",
    onchange: async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      try {
        await S.loadRecipe(JSON.parse(await f.text()));
      } catch (err) {
        S.toast(`couldn't read that recipe (${err.message})`, true);
      }
      e.target.value = "";
    },
  });
  el.append(
    h("div", { class: "pane-head" }, h("h2", {}, "Edits"), h("span", { class: "count num" }, String(edits.length))),
    edits.length
      ? h(
          "ol",
          { class: "edit-list" },
          edits.map((e) =>
            h(
              "li",
              { class: "edit" },
              swatch(e, true),
              h(
                "a",
                {
                  class: "edit-target",
                  href: e.target.layer !== undefined && e.target.layer !== null ? `#/layer/${e.target.layer}` : "#/",
                  title: e.label,
                  style: { textDecoration: "none" },
                },
                shortLabel(e.target),
              ),
              h("span", { class: "edit-val" }, editValueText(e)),
              h("button", { class: "btn btn-quiet btn-sm", type: "button", onclick: () => S.undo(e.id) }, "Undo"),
              e.note ? h("span", { class: "edit-why" }, e.note) : null,
            ),
          ),
        )
      : h("p", { class: "rail-empty" }, "Nothing broken yet. Open a layer and switch something off."),
    h(
      "div",
      { class: "rail-foot" },
      canAsk && edits.length
        ? h("button", { class: "btn btn-primary", type: "button", onclick: () => S.askAgain(false) }, "Ask again")
        : null,
      edits.length ? h("button", { class: "btn", type: "button", onclick: () => S.backToHealthy() }, "Back to healthy") : null,
      h(
        "div",
        { style: { display: "flex", gap: "8px" } },
        h(
          "button",
          { class: "btn btn-quiet btn-sm", type: "button", disabled: !edits.length, onclick: saveRecipe },
          "Save recipe",
        ),
        h("button", { class: "btn btn-quiet btn-sm", type: "button", onclick: () => fileInput.click() }, "Load recipe"),
        fileInput,
      ),
      h(
        "p",
        { class: "meta" },
        "Edits are written into the weights in memory. The GGUF file on disk is never changed, so undo restores the exact values.",
      ),
    ),
  );
}

function saveRecipe() {
  const blob = new Blob([JSON.stringify(state.edits, null, 2)], { type: "application/json" });
  const a = h("a", { href: URL.createObjectURL(blob), download: `lobotomy-recipe-${state.edits.length}-edits.json` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- layer strips -------------------------------------------------------------------------
// One strip per layer, one cell per reply token, lit by how hard the layer's
// neurons fired (relative to that layer's own range).

export function renderStrips(root, detail, opts = {}) {
  const { compact = false, selectedLayer = null, col = null, cost = null, onLayer, rows: rowsOpt, edits = null } = opts;
  clear(root);
  root.toggleAttribute("data-compact", compact);
  const nL = state.model.n_layers;
  const rows = rowsOpt ?? (detail ? detail.tokens.filter((t) => t.role === "reply").map((t) => t.row) : []);
  const act = detail?.activity?.neuron_mean;
  const peaks = detail?.layers;
  const maxCost = cost ? Math.max(...cost.map((c) => c.mean_kl), 1e-6) : 1;

  for (let b = 0; b < nL / 4; b++) {
    const block = h("div", { class: "block" });
    for (let L = b * 4; L < b * 4 + 4 && L < nL; L++) {
      const attn = state.model.layers[L].type === "attention";
      const vals = act ? rows.map((r) => act[r][L]) : [];
      const heat = h(
        "span",
        { class: `heat${vals.length ? "" : " is-empty"}` },
        vals.map((v) => h("i", { class: `act-${rangeBin(v, vals)}`, title: compact ? null : `${fmt(v, 3)}` })),
      );
      const eds = edits ? edits.filter((e) => e.target.layer === L) : S.editsInLayer(L);
      const tag = onLayer ? "button" : "a";
      const strip = h(
        tag,
        {
          class: `strip${attn ? " is-attn" : ""}${selectedLayer === L ? " is-selected" : ""}`,
          href: onLayer ? null : `#/layer/${L}`,
          type: onLayer ? "button" : null,
          onclick: onLayer ? () => onLayer(L) : null,
          "aria-label": `Layer ${L}, ${attn ? "attention" : "DeltaNet"}${eds.length ? `, ${eds.length} edits` : ""}`,
          dataset: { layer: L },
        },
        h("span", { class: "strip-idx num" }, String(L)),
        compact ? null : h("span", { class: "strip-type" }, attn ? "Attention" : "DeltaNet"),
        heat,
        compact ? null : h("span", { class: "strip-peak num" }, peaks ? fmt(peaks[L].neuron_peak) : ""),
        h(
          "span",
          { class: "strip-edit" },
          eds.slice(0, compact ? 2 : 3).map((e) => swatch(e)),
          eds.length > (compact ? 2 : 3) ? h("span", { class: "n" }, `+${eds.length - (compact ? 2 : 3)}`) : null,
        ),
        cost && !compact
          ? h(
              "span",
              { class: "strip-cost", title: `Cutting layer ${L} alone would change this reply by ${fmt(cost[L].mean_kl, 3)} nats` },
              fmt(cost[L].mean_kl, 2),
              h("span", { class: "bar" }, h("b", { style: { width: `${(cost[L].mean_kl / maxCost) * 100}%` } })),
            )
          : null,
      );
      block.append(strip);
    }
    root.append(block);
  }
  if (col !== null && rows.length) {
    const i = rows.indexOf(col);
    if (i >= 0) {
      requestAnimationFrame(() => placeCursor(root, i, rows.length));
    }
  }
}

function placeCursor(root, i, n) {
  root.querySelector(".col-cursor")?.remove();
  const heat = root.querySelector(".heat");
  if (!heat) return;
  const hr = heat.getBoundingClientRect();
  const rr = root.getBoundingClientRect();
  const w = (hr.width - 2) / n;
  root.append(
    h("span", {
      class: "col-cursor",
      style: { left: `${hr.left - rr.left + 1 + i * w - 1}px`, width: `${Math.max(3, w + 2)}px` },
    }),
  );
}

// --- popovers -------------------------------------------------------------------------------

let openPop = null;
export function closePopover() {
  openPop?.remove();
  openPop = null;
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closePopover();
});
document.addEventListener("mousedown", (e) => {
  if (openPop && !openPop.contains(e.target) && !e.target.closest("[data-pop-anchor]")) closePopover();
});

function place(pop, anchor) {
  document.body.append(pop);
  const a = anchor.getBoundingClientRect();
  const p = pop.getBoundingClientRect();
  let left = a.right + 12;
  if (left + p.width > window.innerWidth - 12) left = a.left - p.width - 12;
  if (left < 12) left = Math.max(12, Math.min(window.innerWidth - p.width - 12, a.left));
  let top = a.top - 8;
  if (top + p.height > window.innerHeight - 12) top = window.innerHeight - p.height - 12;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(12, top)}px`;
}

// The edit popover: Off, a log slider from off to x16, Boost x2, Noise, Reset.
// Changes apply to the weights immediately; the reply changes when you ask again.
export function openEditPopover(anchor, { target, title, sub, info, note, says, trace, sign = 1 }) {
  closePopover();
  const current = () => S.editFor(target, "scale");
  const noise = () => S.editFor(target, "noise");
  const readout = h("span", { class: "readout", "aria-live": "polite" });
  const slider = h("input", {
    type: "range",
    min: SLIDER.min,
    max: SLIDER.max,
    step: SLIDER.step,
    "aria-label": "Scale",
  });
  const offBtn = h("button", { class: "btn", type: "button" }, "Off");
  const sync = () => {
    const v = current()?.value ?? 1;
    slider.value = v === 1 ? 0 : valueToSlider(v);
    readout.textContent = fmtScale(v);
    offBtn.setAttribute("aria-pressed", String(v === 0));
    const n = noise();
    noiseBtn.textContent = n ? `Noise ${Number(n.value.toPrecision(2))}` : "Add noise";
  };
  const setValue = (v) => {
    readout.textContent = fmtScale(v);
    offBtn.setAttribute("aria-pressed", String(v === 0));
    S.setScale(target, v, note).then(sync);
  };
  slider.addEventListener("input", () => {
    const v = sliderToValue(slider.value);
    slider.setAttribute("aria-valuetext", v === 0 ? "off" : `times ${v}`);
    readout.textContent = fmtScale(v);
  });
  slider.addEventListener("change", () => setValue(sliderToValue(slider.value)));
  offBtn.addEventListener("click", () => setValue(current()?.value === 0 ? 1 : 0));
  const noiseBtn = h("button", {
    class: "btn btn-quiet",
    type: "button",
    title: "Add Gaussian noise to these weights (strength relative to their size). Click again to make it stronger.",
    onclick: () => {
      const n = noise();
      S.addNoise(target, n ? Math.min(4, n.value * 2) : 0.25, note).then(sync);
    },
  });

  const traceEl = trace ? sparkline(trace) : null;
  const saysEl = h("div", { class: "says" });
  if (says) {
    // Its output column read through the unembedding. A negative activation
    // flips the push, so describe it the way it actually fired here.
    says.then((v) => {
      if (!v) return;
      const [up, down] = sign < 0 ? [v.suppresses, v.promotes] : [v.promotes, v.suppresses];
      saysEl.append(
        h("span", {}, sign < 0 ? "It fired negative here, which pushes up" : "When it fires, it pushes up"),
        h("span", { class: "toks" }, up.slice(0, 6).map((t) => h("code", {}, showToken(t)))),
        h("span", {}, "and pushes down"),
        h("span", { class: "toks" }, down.slice(0, 6).map((t) => h("code", {}, showToken(t)))),
      );
    }).catch(() => {});
  }

  const pop = h(
    "div",
    { class: "popover", role: "dialog", "aria-label": title },
    h("div", { class: "popover-head" }, h("h3", {}, title), h("span", { class: "meta" }, sub ?? "")),
    info ? h("p", { class: "meta" }, info) : null,
    traceEl,
    saysEl,
    h(
      "div",
      { class: "scale-slider" },
      slider,
      h(
        "div",
        { class: "ticks", "aria-hidden": "true" },
        ["off", "", "×¼", "", "×1", "", "×4", "", "×16"].map((t, i) => h("span", { class: i === 4 ? "mid" : null }, t)),
      ),
    ),
    h(
      "div",
      { class: "popover-actions" },
      offBtn,
      h("button", { class: "btn", type: "button", onclick: () => setValue(2) }, "Boost ×2"),
      h(
        "button",
        {
          class: "btn btn-quiet",
          type: "button",
          onclick: async () => {
            const c = current();
            const n = noise();
            if (c) await S.undo(c.id);
            if (n) await S.undo(n.id);
            sync();
          },
        },
        "Reset",
      ),
      readout,
    ),
    h("div", { class: "popover-foot" }, h("p", { class: "meta" }, "Applies from the next reply."), noiseBtn),
  );
  sync();
  place(pop, anchor);
  openPop = pop;
  slider.focus();
}

export function sparkline(values) {
  const w = 288;
  const hgt = 34;
  const n = values.length;
  const m = Math.max(...values.map((v) => Math.abs(v)), 1e-6);
  const y = (v) => hgt / 2 - (v / m) * (hgt / 2 - 2);
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, n - 1)) * w).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return s(
    "svg",
    { class: "trace", viewBox: `0 0 ${w} ${hgt}`, preserveAspectRatio: "none", role: "img", "aria-label": "Activation across the recorded tokens" },
    s("line", { x1: 0, x2: w, y1: hgt / 2, y2: hgt / 2 }),
    s("path", { d }),
  );
}

// --- settings menu ----------------------------------------------------------------------------

export function openSettings(anchor) {
  closePopover();
  const st = state.settings;
  const num = (key, label, attrs) =>
    h("label", { class: "field" }, label, h("input", { type: "number", value: st[key], "data-key": key, ...attrs }));
  const think = h("input", { type: "checkbox", checked: st.enable_thinking });
  const sys = h("textarea", { placeholder: "Optional system prompt" }, st.system_prompt ?? "");
  const form = h(
    "form",
    {
      class: "menu",
      onsubmit: async (e) => {
        e.preventDefault();
        const patch = { enable_thinking: think.checked, system_prompt: sys.value };
        for (const inp of form.querySelectorAll("input[data-key]")) patch[inp.dataset.key] = Number(inp.value);
        await S.saveSettings(patch);
        closePopover();
        S.toast("Settings saved");
      },
    },
    h("h3", {}, "Sampling"),
    num("temperature", "Temperature (0 = greedy)", { step: 0.05, min: 0, max: 2 }),
    num("top_p", "Top-p", { step: 0.05, min: 0, max: 1 }),
    num("top_k", "Top-k", { step: 1, min: 0 }),
    num("max_tokens", "Max reply tokens", { step: 16, min: 1, max: 4096 }),
    num("seed", "Seed", { step: 1, min: 0 }),
    num("presence_penalty", "Presence penalty", { step: 0.1, min: 0, max: 2 }),
    h("label", { class: "field check" }, think, "Let the model think first (<think>)"),
    sys,
    h(
      "div",
      { class: "popover-actions" },
      h("button", { class: "btn btn-primary", type: "submit" }, "Save"),
      h("button", { class: "btn btn-quiet", type: "button", onclick: () => S.newChat().then(closePopover) }, "New chat"),
    ),
    h("p", { class: "meta" }, "Ask again reuses the first answer's seed, so a different answer comes from your edits, not from luck."),
  );
  form.classList.add("popover");
  place(form, anchor);
  openPop = form;
}

export { percentile };
