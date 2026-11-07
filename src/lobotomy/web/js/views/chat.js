// Chat view: transcript | layer stack | edits rail.

import * as S from "../store.js";
import { state } from "../store.js";
import { placeRowCursor, renderRail, renderStrips, swatch } from "../components.js";
import { clear, fmt, h } from "../util.js";

let showCost = false;
// The chat pane is rebuilt on state changes; keep what the user was typing.
let draft = "";
let composerFocused = false;
let lastContentKey = "";

// Transcript (shared with the token view).
export function renderChat(section, { selected = null } = {}) {
  const prevScroll = section.scrollTop;
  clear(section);
  const transcript = h("div", { class: "transcript" });
  for (const ex of state.exchanges) {
    transcript.append(h("article", { class: "msg msg-user" }, h("div", { class: "who" }, "You"), h("p", { class: "msg-body" }, ex.user)));
    const streaming = state.streaming && (state.streaming.exchangeId === ex.id || (ex.id === null && state.streaming));
    const r = ex.replies[ex.replies.length - 1];
    if (streaming && (ex === state.exchanges[state.exchanges.length - 1])) {
      transcript.append(streamingMsg());
    } else if (r) {
      transcript.append(modelMsg(ex, r, selected));
    }
  }
  const last = S.lastReply();
  const pending =
    last && !state.busy && S.editsSince(last)
      ? h(
          "div",
          { class: "pending" },
          h(
            "p",
            {},
            h("strong", {}, state.edits.length ? `${state.edits.length} edit${state.edits.length === 1 ? "" : "s"} since this reply.` : "Edits removed since this reply."),
            " Ask again to see what they change.",
          ),
          h(
            "div",
            { class: "actions" },
            h("button", { class: "btn btn-primary", type: "button", onclick: () => S.askAgain(false) }, "Ask again"),
          ),
          h(
            "label",
            { title: "Off: the model keeps its memory of the earlier chat as it read it before your edits; only the new reply uses the edited weights." },
            h("input", {
              type: "checkbox",
              checked: state.settings?.reread !== false,
              onchange: (e) => S.saveSettings({ reread: e.target.checked }),
            }),
            " Re-read the whole chat with the edits",
          ),
        )
      : null;

  const ta = h("textarea", {
    id: "ask",
    rows: 1,
    placeholder: state.exchanges.length ? "Say something else" : "Ask the model something",
    disabled: state.busy,
    onkeydown: (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    },
    oninput: (e) => {
      draft = e.target.value;
      e.target.style.height = "auto";
      e.target.style.height = `${Math.min(200, e.target.scrollHeight)}px`;
    },
    onfocus: () => (composerFocused = true),
    onblur: () => (composerFocused = false),
  });
  ta.value = draft;
  const submit = () => {
    const text = ta.value;
    if (!text.trim()) return;
    ta.value = "";
    draft = "";
    S.send(text);
  };
  const composer = h(
    "form",
    {
      class: "composer",
      onsubmit: (e) => {
        e.preventDefault();
        submit();
      },
    },
    h("label", { class: "sr-only", for: "ask" }, "Message"),
    ta,
    state.busy
      ? h("button", { class: "btn", type: "button", onclick: () => S.cancel() }, "Stop")
      : h("button", { class: "btn", type: "submit" }, "Send"),
  );
  section.append(...[transcript, pending, composer].filter(Boolean));
  // Scroll to the bottom only when the conversation itself changed.
  const key = state.exchanges.map((e) => `${e.id}:${e.replies.length}`).join(",") + (state.streaming ? ":s" : "");
  const grew = key !== lastContentKey;
  lastContentKey = key;
  section.scrollTop = grew ? section.scrollHeight : prevScroll;
  if (!state.busy && (composerFocused || document.activeElement === document.body)) {
    ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length);
  }
}

function streamingMsg() {
  const body = h("p", { class: "msg-body", id: "stream-body" }, state.streaming.text);
  return h(
    "article",
    { class: "msg msg-model is-streaming" },
    h("div", { class: "who" }, "Model"),
    body,
    h("div", { class: "msg-meta" }, h("span", {}, state.streaming.healthy ? "healthy model" : state.edits.length ? `${state.edits.length} edits` : "healthy weights")),
  );
}

// Streamed tokens go straight into the DOM.
S.setTokenHandler((st) => {
  const el = document.getElementById("stream-body");
  if (el) {
    el.textContent = st.text;
    const pane = el.closest(".chat");
    if (pane) pane.scrollTop = pane.scrollHeight;
  }
});

function modelMsg(ex, r, selected) {
  const detail = state.details.get(r.id);
  let body;
  if (detail) {
    const toks = detail.tokens.filter((t) => t.role === "reply");
    body = h(
      "p",
      { class: "msg-body" },
      toks.map((t) =>
        h(
          "span",
          {
            class: `tok${selected && selected.replyId === r.id && selected.row === t.row ? " is-selected" : ""}`,
            dataset: { row: t.row },
            title: t.prob != null ? `p = ${fmt(t.prob, 3)}` : null,
            onclick: () => (location.hash = `#/token/${r.id}/${t.row}`),
            onmouseenter: () => r.id === S.lastReply()?.id && hoverRow(t.row),
            onmouseleave: () => hoverRow(null),
          },
          t.text,
        ),
      ),
    );
  } else {
    body = h("p", { class: "msg-body" }, r.text);
    S.ensureDetail(r.id).then(() => S.emit()).catch(() => {});
  }
  const st = r.stats || {};
  const n = ex.replies.length;
  return h(
    "article",
    { class: "msg msg-model" },
    h("div", { class: "who" }, "Model"),
    body,
    h(
      "div",
      { class: "msg-meta" },
      h("span", {}, `${st.reply_tokens ?? r.n_tokens} tokens`),
      st.decode_seconds != null ? h("span", {}, `${fmt(st.prefill_seconds + st.decode_seconds, 1)} s`) : null,
      st.tokens_per_second ? h("span", {}, `${fmt(st.tokens_per_second, 0)} tok/s`) : null,
      r.healthy
        ? h("span", {}, "healthy model (edits bypassed)")
        : r.edits.length
          ? h("span", { class: "swatches" }, r.edits.slice(0, 6).map((e) => swatch(e)), ` ${r.edits.length} edit${r.edits.length === 1 ? "" : "s"}`)
          : h("span", {}, "healthy weights"),
      r.kept_memory ? h("span", { title: "Earlier turns were kept as read under the weights of the time" }, "kept memory") : null,
      r.finish_reason === "length" ? h("span", {}, "cut off at max tokens") : null,
      r.finish_reason === "cancelled" ? h("span", {}, "stopped") : null,
      n > 1 ? h("a", { href: `#/compare/${ex.id}` }, `Compare ${n} answers`) : null,
    ),
  );
}

function hoverRow(row) {
  state.hoverRow = row;
  const strips = document.getElementById("stack-main");
  if (strips) placeRowCursor(strips, row);
}

export function renderStack(section) {
  clear(section);
  const last = S.lastReply();
  const detail = last ? state.details.get(last.id) : null;
  let cost = last && showCost ? S.scanResult(last.id, "layers") : null;
  if (cost && !cost.length) cost = null; // empty reply: nothing to measure
  // Results are dropped when edits change or a new reply arrives; re-measure.
  if (showCost && last && !cost && !state.busy && last.n_tokens > 0) {
    S.scan(last.id, "layers").catch(() => {
      showCost = false;
      S.emit();
    });
  }

  const head = h(
    "div",
    { class: "pane-head" },
    h("h2", {}, "Layers"),
    h(
      "p",
      { class: "meta" },
      "One strip per layer, one cell per token of the last reply. The brighter the cell, the harder that layer fired on that token, against its own usual range. Open a strip to edit it.",
    ),
    last
      ? h(
          "div",
          { class: "stack-tools" },
          h(
            "button",
            {
              class: "btn btn-sm",
              type: "button",
              "aria-pressed": String(showCost),
              title: "Cut each layer alone and measure how much this reply would change",
              disabled: !last.n_tokens,
              onclick: () => {
                showCost = !showCost;
                S.emit();
              },
            },
            showCost ? (cost ? "Hide cut cost" : "Measuring…") : "What matters here?",
          ),
        )
      : null,
  );

  const strips = h("div", { class: "strips", id: "stack-main" });
  const wrap = h(
    "div",
    { class: `stack-wrap${cost ? " has-cost" : ""}` },
    h(
      "div",
      { class: "axis", "aria-hidden": "true" },
      h("span", { class: "axis-layer" }, "Layer"),
      h("span", { class: "axis-track" }),
      h("span", { style: { textAlign: "right" } }, "Peak"),
      h("span", {}, "Edits"),
      cost ? h("span", { style: { textAlign: "right" } }, "Cut cost") : null,
    ),
    strips,
  );
  section.append(head);
  if (!state.model) return;
  if (!last) {
    section.append(
      h("p", { class: "stack-empty" }, "The layers light up after the first reply. Each of the 24 strips shows how hard that layer fired on every word the model wrote."),
      legend(),
    );
    return;
  }
  section.append(wrap);
  renderStrips(strips, detail, { cost, col: state.hoverRow });
  strips._detail = detail;
  if (cost) {
    section.append(
      h(
        "p",
        { class: "scan-note", style: { padding: "0 24px 12px" } },
        "Cut cost: how much this reply's word-by-word predictions would change (KL divergence, nats) if that layer alone were switched off on top of your current edits.",
      ),
    );
  }
  section.append(legend());
}

export function legend() {
  const m = state.model;
  const attn = m.layers.filter((l) => l.type === "attention").map((l) => l.index);
  return h(
    "div",
    { class: "legend" },
    h(
      "div",
      { class: "scale" },
      h("span", { class: "scale-title" }, "Activity, how hard it fired"),
      h("span", { class: "scale-bar" }, Array.from({ length: 9 }, (_, i) => h("i", { class: `act-${i}` }))),
      h("span", { class: "scale-ends" }, h("span", {}, "silent"), h("span", {}, "strongest")),
    ),
    h(
      "div",
      { class: "scale" },
      h("span", { class: "scale-title" }, "Edits, what you changed"),
      h(
        "span",
        { class: "scale-bar" },
        ["edit-down-4", "edit-down-3", "edit-down-2", "edit-down-1", "none", "edit-up-1", "edit-up-2", "edit-up-3", "edit-up-4"].map((c) => h("i", { class: c })),
      ),
      h("span", { class: "scale-ends" }, h("span", {}, "off"), h("span", {}, "×1"), h("span", {}, "×16")),
    ),
    h(
      "div",
      { class: "scale" },
      h("span", { class: "scale-title" }, "Layer types"),
      h("span", {}, `Attention: layers ${attn.join(", ")}`),
      h("span", {}, `DeltaNet: the other ${m.n_layers - attn.length}`),
    ),
  );
}

export function renderChatView(body) {
  clear(body);
  body.className = "body";
  body.dataset.view = "chat";
  const chat = h("section", { class: "chat", "aria-label": "Chat" });
  const stack = h("section", { "aria-label": "Layers" });
  const rail = h("aside", { class: "rail", "aria-label": "Edits" });
  body.append(chat, stack, rail);
  renderChat(chat);
  renderStack(stack);
  renderRail(rail);
}
