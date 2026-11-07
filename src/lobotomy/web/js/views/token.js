// Token view: chat (word selected) | what happened on that word | edits rail.

import { api } from "../api.js";
import * as S from "../store.js";
import { state } from "../store.js";
import { layerTypeName, renderRail, swatch } from "../components.js";
import { renderChat } from "./chat.js";
import { setLayerRow } from "./layer.js";
import { actBin, clear, editValueText, fmt, h, showLoading, showToken } from "../util.js";

const cache = new Map();
const whyCache = new Map(); // `${reply}:${row}:${edit signature}` -> Promise
let attnLayer = null;
let attnHead = null;

function whyKey(replyId, row) {
  return `${replyId}:${row}:${S.editSignature(state.edits)}`;
}

// "Why this word?": gradient attribution over every neuron and head.
function renderWhy(replyId, d) {
  const text = d.token.text;
  const box = h("div", { class: "why" });
  const key = whyKey(replyId, d.row);
  const head = h(
    "h2",
    { class: "sub" },
    "Why this word? ",
    h("span", { class: "meta" }, "which neurons and heads pushed the model toward it, from one gradient pass. Estimates, not measurements."),
  );
  box.append(head);
  const p = whyCache.get(key);
  if (!p) {
    box.append(
      h(
        "button",
        {
          class: "btn",
          type: "button",
          disabled: state.busy || d.token.position < 1,
          onclick: (e) => {
            e.currentTarget.textContent = "Working…";
            e.currentTarget.disabled = true;
            whyCache.set(key, api.why(replyId, d.row));
            whyCache.get(key).then(() => S.emit(), (err) => {
              whyCache.delete(key);
              S.toast(err.message, true);
            });
          },
        },
        "Find what pushed it here",
      ),
    );
    return box;
  }
  const slot = h("div", {}, h("p", { class: "meta" }, "Working…"));
  box.append(slot);
  p.then((w) => {
    clear(slot);
    const max = Math.max(...w.support.map((u) => -u.effect), 1e-6);
    const unitTarget = (u) => ({ kind: u.kind, layer: u.layer, index: u.index });
    const rows = w.support.slice(0, 10).map((u) => {
      const e = S.editFor(unitTarget(u));
      const est = Math.exp(w.logp + u.effect);
      return h(
        "tr",
        {},
        h("td", { class: "mono" }, u.label),
        h("td", {}, u.kind === "neuron" ? "neuron" : `${layerTypeName(u.layer) === "Attention" ? "attention" : "DeltaNet"} head`),
        h(
          "td",
          { class: "r num", title: `If cut alone: p ≈ ${fmt(w.prob, 2)} → ${fmt(est, 2)} (first-order estimate)` },
          fmt(u.effect, 2),
          h("span", { class: "bar act-6" }, h("b", { style: { width: `${(-u.effect / max) * 100}%` } })),
        ),
        h(
          "td",
          {},
          e
            ? h("span", { class: "edit-tag" }, swatch(e), editValueText(e))
            : h(
                "span",
                { class: "row-actions", style: { opacity: 1 } },
                h(
                  "button",
                  {
                    class: "btn btn-sm",
                    type: "button",
                    onclick: () => S.setScale(unitTarget(u), 0, `Pushed toward “${showToken(text)}” (est. ${fmt(u.effect, 2)}).`),
                  },
                  "Off",
                ),
              ),
        ),
      );
    });
    const fresh = w.support.slice(0, 10).filter((u) => !S.editFor(unitTarget(u)));
    slot.append(
      h(
        "p",
        { class: "meta" },
        `The model gave “${showToken(text)}” probability ${fmt(w.prob, 2)}. Effect = estimated change in log-probability if that unit alone were switched off.`,
      ),
      h(
        "table",
        { class: "list" },
        h("thead", {}, h("tr", {}, h("th", {}, "Unit"), h("th", {}, "Kind"), h("th", { class: "r" }, "Effect if cut"), h("th", {}, "Edit"))),
        h("tbody", {}, rows),
      ),
      h(
        "div",
        { class: "why-actions" },
        fresh.length
          ? h(
              "button",
              {
                class: "btn",
                type: "button",
                onclick: () =>
                  S.addMany(
                    fresh.map((u) => ({
                      target: unitTarget(u),
                      value: 0,
                      note: `Pushed toward “${showToken(text)}” (est. ${fmt(u.effect, 2)}).`,
                    })),
                  ).then(() => S.toast("Switched off. Ask again to see if it still says it.")),
              },
              `Make it stop saying “${text.trim() || showToken(text)}”`,
            )
          : h("p", { class: "meta" }, "All ten are already edited. Ask again to see the effect."),
      ),
    );
  });
  return box;
}

// Zero the token's embedding row. The embedding is tied to the output head, so
// the model can neither read this token nor say it.
function eraseButton(d) {
  const target = { kind: "token", index: d.token.token_id };
  const e = S.editFor(target);
  if (e) return h("span", { class: "meta" }, `Token ${d.token.token_id} is erased.`);
  return h(
    "button",
    {
      class: "btn btn-quiet",
      type: "button",
      title: "Zero this token's embedding. It's shared with the output layer, so the model can neither read nor say this exact token (it may still spell the word another way).",
      onclick: () =>
        S.setScale(target, 0, `Erased token “${showToken(d.token.text)}” (id ${d.token.token_id}).`).then(() =>
          S.toast("Erased. Ask again to see what it says instead."),
        ),
    },
    `Erase token “${d.token.text.trim() || showToken(d.token.text)}”`,
  );
}

export async function renderTokenView(body, replyId, row) {
  body.className = "body";
  const key = `${replyId}:${row}`;
  showLoading(body, `token:${key}`);
  if (!cache.has(key)) cache.set(key, api.token(replyId, row));
  let data, detail;
  try {
    [data, detail] = await Promise.all([cache.get(key), S.ensureDetail(replyId)]);
  } catch (e) {
    cache.delete(key);
    clear(body);
    body.append(h("div", { class: "loading" }, h("p", {}, `Couldn't load that word (${e.message}). `, h("a", { href: "#/" }, "Back to chat"))));
    return;
  }
  if (state.route.view !== "token" || state.route.replyId !== replyId || state.route.row !== row) return;

  const prevScroll = body.querySelector(".inspect")?.scrollTop ?? 0;
  clear(body);
  body.dataset.view = `token:${key}`;
  const chat = h("section", { class: "chat", "aria-label": "Chat" });
  const inspect = h("section", { class: "inspect", "aria-labelledby": "tok-title" });
  const rail = h("aside", { class: "rail", "aria-label": "Edits" });
  body.append(chat, inspect, rail);
  renderChat(chat, { selected: { replyId, row } });
  renderRail(rail);
  renderInspect(inspect, data, detail, replyId);
  inspect.scrollTop = prevScroll;
}

function renderInspect(el, d, detail, replyId) {
  const m = state.model;
  const t = d.token;
  const text = t.text;
  const lead = text.match(/^\s+/)?.[0] ?? "";
  const rest = text.slice(lead.length);

  let meta;
  if (t.role === "reply") {
    // Raw model probabilities (temperature 1), so the word and the runner-up are comparable.
    const own = (d.alternatives || []).find((a) => a.text === text);
    const alts = (d.alternatives || []).filter((a) => a.text !== text);
    meta = [
      `Token ${d.index_in_reply + 1} of ${d.n_reply_tokens}, id ${t.token_id}. `,
      own ? `The model gave it probability ${fmt(own.prob, 2)}` : "It was outside the model's top 5 (a lucky sample)",
      alts.length ? `; the runner-up was “${showToken(alts[0].text)}” at ${fmt(alts[0].prob, 2)}.` : ".",
    ].join("");
  } else {
    meta = `A token of the prompt (position ${t.position}), id ${t.token_id}.`;
  }

  // per-layer bars on this word, relative to each layer's range over the recording
  const act = detail.activity.neuron_mean;
  const lbars = h("div", { class: "lbars" });
  for (let b = 0; b < m.n_layers / 4; b++) {
    const block = h("div", { class: "block" });
    for (let L = b * 4; L < b * 4 + 4; L++) {
      const v = d.layer_mlp[L];
      const col = act.map((r) => r[L]);
      const lo = Math.min(...col);
      const hi = Math.max(...col);
      const rel = hi - lo > 1e-12 ? (v - lo) / (hi - lo) : 0.5;
      const eds = S.editsInLayer(L);
      block.append(
        h(
          "a",
          {
            class: `lbar${m.layers[L].type === "attention" ? " is-attn" : ""}`,
            href: `#/layer/${L}`,
            onclick: () => setLayerRow(replyId, d.row),
            title: `Open layer ${L} showing this word`,
          },
          h("span", { class: "strip-idx num" }, String(L)),
          h("span", { class: "strip-type" }, layerTypeName(L)),
          h("span", { class: "lbar-track" }, h("b", { class: `act-${Math.max(1, Math.round(1 + 7 * rel))}`, style: { width: `${Math.max(2, Math.min(100, rel * 100))}%` } })),
          h("span", { class: "lbar-val num" }, fmt(v, 3)),
          h("span", { class: "strip-edit" }, eds.slice(0, 2).map((e) => swatch(e))),
        ),
      );
    }
    lbars.append(block);
  }

  // logit lens: what the model would say next if it stopped after each layer
  const lens = h(
    "div",
    { class: "lens" },
    d.lens.map((top, L) =>
      h(
        "div",
        { class: `lens-row${L === m.n_layers - 1 ? " is-final" : ""}` },
        h("span", { class: "num" }, `L${L}`),
        h(
          "span",
          {},
          h("span", { class: "mono" }, showToken(top[0].text)),
          top.slice(1).map((x) => h("span", { class: "alt" }, `  ${showToken(x.text)}`)),
        ),
        h("span", { class: "num", style: { textAlign: "right" } }, fmt(top[0].prob, 2)),
      ),
    ),
  );

  // neurons that fired hardest, against their own layer
  const ranked = d.ranked;
  const maxRel = Math.max(...ranked.map((x) => x.relative), 1e-6);
  const neurons = h(
    "table",
    { class: "list" },
    h("thead", {}, h("tr", {}, h("th", {}, "Neuron"), h("th", { class: "r" }, "Value"), h("th", { class: "r" }, "× usual"), h("th", {}, "Edit"))),
    h(
      "tbody",
      {},
      ranked.map((x) => {
        const target = { kind: "neuron", layer: x.layer, index: x.neuron };
        const e = S.editFor(target);
        return h(
          "tr",
          { onclick: (ev) => !ev.target.closest("button") && (setLayerRow(replyId, d.row), (location.hash = `#/layer/${x.layer}`)) },
          h("td", { class: "mono" }, `L${x.layer} N${x.neuron}`),
          h(
            "td",
            { class: "r num" },
            fmt(x.value),
            h("span", { class: `bar act-${actBin(x.relative, maxRel)}` }, h("b", { style: { width: `${(x.relative / maxRel) * 100}%` } })),
          ),
          h("td", { class: "r num" }, fmt(x.relative, 1)),
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
                      onclick: () => S.setScale(target, 0, `Fired ${fmt(x.value)} on “${showToken(text)}”.`),
                    },
                    "Off",
                  ),
                ),
          ),
        );
      }),
    ),
  );

  // attention: pick a layer and a head
  const attnLayers = Object.keys(d.attention).map(Number).sort((a, b) => a - b);
  if (attnLayer === null || !attnLayers.includes(attnLayer)) {
    // default: the layer whose heads wrote the most on this word
    attnLayer = attnLayers.reduce((best, L) => (Math.max(...d.heads[L]) > Math.max(...d.heads[best]) ? L : best), attnLayers[0]);
    attnHead = null;
  }
  const headVals = d.heads[attnLayer];
  if (attnHead === null) attnHead = headVals.indexOf(Math.max(...headVals));
  const headRef = Math.max(...headVals, 1e-6);
  const A = d.attention[String(attnLayer)];
  const hd = A.heads[attnHead];
  const attnEl = h(
    "div",
    { class: "attn" },
    hd.top.map((x) =>
      h(
        "div",
        { class: "attn-row" },
        h("span", {}, h("span", { class: "mono" }, showToken(x.text)), " ", h("span", { class: "where" }, x.where)),
        h("span", { class: "num r" }, fmt(x.weight, 2)),
        h("span", { class: "bar" }, h("b", { style: { width: `${x.weight * 100}%` } })),
      ),
    ),
    hd.rest > 0.005
      ? h(
          "div",
          { class: "attn-row" },
          h("span", { class: "where" }, "everything else"),
          h("span", { class: "num r" }, fmt(hd.rest, 2)),
          h("span", { class: "bar" }, h("b", { style: { width: `${hd.rest * 100}%` } })),
        )
      : null,
  );
  const rerender = () => renderTokenView(document.getElementById("body"), replyId, d.row);

  el.append(
    h(
      "header",
      { class: "token-head" },
      h("h1", { id: "tok-title", class: "token-big" }, lead ? h("span", { class: "ws", "aria-label": "leading space" }, showToken(lead)) : null, showToken(rest) || "∅"),
      h("p", { class: "meta" }, meta),
      d.alternatives?.length
        ? h(
            "div",
            { class: "alts" },
            "Candidates: ",
            d.alternatives.map((a) => h("span", {}, h("code", {}, showToken(a.text)), ` ${fmt(a.prob, 2)}`)),
          )
        : null,
      h("div", { class: "why-actions" }, eraseButton(d)),
    ),
    renderWhy(replyId, d),
    h(
      "div",
      { class: "inspect-cols" },
      h(
        "div",
        {},
        h("h2", { class: "sub" }, "Layers on this word ", h("span", { class: "meta" }, "against each layer's range over the recording; raw value on the right")),
        lbars,
        h(
          "h2",
          { class: "sub" },
          "Logit lens ",
          h("span", { class: "meta" }, "the next word each layer would pick if the model stopped there"),
        ),
        lens,
      ),
      h(
        "div",
        {},
        h("h2", { class: "sub" }, "Neurons that fired hardest ", h("span", { class: "meta" }, "against their own layer's usual range")),
        neurons,
        h(
          "h2",
          { class: "sub" },
          `Where layer ${attnLayer} looked `,
          h("span", { class: "meta" }, `attention, head ${attnHead}`),
        ),
        h(
          "div",
          { class: "seg", role: "group", "aria-label": "Attention layer" },
          attnLayers.map((L) =>
            h(
              "button",
              {
                type: "button",
                "aria-pressed": String(L === attnLayer),
                onclick: () => {
                  attnLayer = L;
                  attnHead = null;
                  rerender();
                },
              },
              `L${L}`,
            ),
          ),
        ),
        h(
          "div",
          { class: "heads is-8", "aria-label": `${headVals.length} attention heads in layer ${attnLayer}` },
          headVals.map((v, i) =>
            h(
              "button",
              {
                class: `head-cell${i === attnHead ? " is-selected" : ""}`,
                type: "button",
                title: `Head ${i}: wrote ${fmt(v, 3)} on this word`,
                onclick: () => {
                  attnHead = i;
                  rerender();
                },
              },
              h("span", { class: `head act-${actBin(v, headRef)}` }),
              h("span", { class: "head-idx num" }, String(i)),
            ),
          ),
        ),
        attnEl,
        h(
          "p",
          { class: "meta list-note" },
          "Only the six attention layers record where they looked. DeltaNet layers keep a running state instead, so they have no per-word pattern to show.",
        ),
      ),
    ),
  );
}
