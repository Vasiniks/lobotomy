// Compare view: two answers to the same question, side by side | edits rail.

import * as S from "../store.js";
import { state } from "../store.js";
import { renderRail, renderStrips, swatch } from "../components.js";
import { clear, fmt, h } from "../util.js";

const picks = new Map(); // exchangeId -> [leftReplyId, rightReplyId]

function label(r) {
  if (r.healthy) return "Healthy model";
  if (!r.edits.length) return "Healthy";
  return `With ${r.edits.length} edit${r.edits.length === 1 ? "" : "s"}`;
}

export async function renderCompareView(body, exchangeId) {
  body.className = "body is-compare";
  const ex = state.exchanges.find((e) => e.id === exchangeId);
  clear(body);
  if (!ex || !ex.replies.length) {
    body.append(h("div", { class: "loading" }, h("p", {}, "Nothing to compare yet. ", h("a", { href: "#/" }, "Back to chat"))));
    return;
  }
  const replies = ex.replies;
  let [a, b] = picks.get(exchangeId) ?? [];
  if (!replies.find((r) => r.id === a)) a = (replies.find((r) => r.healthy || !r.edits.length) ?? replies[0]).id;
  if (!replies.find((r) => r.id === b) || b === a) b = replies[replies.length - 1].id === a && replies.length > 1 ? replies[replies.length - 2].id : replies[replies.length - 1].id;
  picks.set(exchangeId, [a, b]);
  const ra = replies.find((r) => r.id === a);
  const rb = replies.find((r) => r.id === b);
  const [da, db] = await Promise.all([S.ensureDetail(a), S.ensureDetail(b)]);
  if (state.route.view !== "compare" || state.route.exchangeId !== exchangeId) return;
  clear(body);

  const ta = da.tokens.filter((t) => t.role === "reply");
  const tb = db.tokens.filter((t) => t.role === "reply");
  let split = 0;
  while (split < ta.length && split < tb.length && ta[split].token_id === tb[split].token_id) split++;
  const identical = split === ta.length && split === tb.length;

  const answer = (r, d, toks, side) =>
    h(
      "article",
      { class: "answer" },
      h(
        "div",
        { class: "answer-head" },
        h("h2", {}, label(r)),
        !r.healthy && r.edits.length ? h("span", { class: "swatches", style: { display: "flex", gap: "3px" } }, r.edits.slice(0, 8).map((e) => swatch(e))) : null,
        h("span", { class: "meta num" }, `${toks.length} tokens`),
      ),
      picker(replies, side, exchangeId),
      h(
        "p",
        { class: "msg-body" },
        toks.map((t, i) =>
          h("span", { class: `tok${i === split && !identical ? " is-split" : ""}` }, t.text),
        ),
      ),
      (() => {
        const st = h("div", { class: "strips", "aria-label": `${label(r)}, 24 layers` });
        renderStrips(st, d, { compact: true, edits: r.healthy ? [] : r.edits });
        return st;
      })(),
    );

  let note;
  if (identical) note = h("p", { class: "split-note" }, h("strong", {}, "The answers are identical."), " These edits didn't change a single token. Try breaking something bigger.");
  else
    note = h(
      "p",
      { class: "split-note" },
      h("strong", {}, `The answers split at token ${split + 1}.`),
      split > 0 ? ` The first ${split} token${split === 1 ? "" : "s"} match.` : " They differ from the very first token.",
      " Each answer was generated with the same seed, so the difference comes from the weights.",
    );

  const section = h(
    "section",
    { class: "compare", "aria-labelledby": "cmp-title" },
    h(
      "header",
      { class: "compare-head" },
      h(
        "div",
        {},
        h("h1", { id: "cmp-title" }, ex.user),
        h("p", { class: "meta" }, `${replies.length} answers to this question. The whole chat was re-read with each answer's weights.`),
      ),
      h(
        "div",
        { style: { display: "flex", gap: "8px" } },
        !replies.some((r) => r.healthy || !r.edits.length)
          ? h("button", { class: "btn", type: "button", disabled: state.busy, onclick: () => S.askAgain(true) }, "Ask the healthy model")
          : null,
        h("a", { class: "btn", href: "#/" }, "Back to chat"),
      ),
    ),
    h("div", { class: "answers" }, answer(ra, da, ta, 0), answer(rb, db, tb, 1)),
    note,
  );
  const rail = h("aside", { class: "rail", "aria-label": "Edits" });
  body.append(section, rail);
  renderRail(rail);
}

function picker(replies, side, exchangeId) {
  if (replies.length < 3) return null;
  const cur = picks.get(exchangeId)[side];
  return h(
    "div",
    { class: "answer-pick" },
    "Answer ",
    h(
      "div",
      { class: "seg", role: "group", "aria-label": "Pick an answer" },
      replies.map((r, i) =>
        h(
          "button",
          {
            type: "button",
            "aria-pressed": String(r.id === cur),
            title: label(r),
            onclick: () => {
              const p = picks.get(exchangeId);
              p[side] = r.id;
              S.emit();
            },
          },
          String(i + 1),
        ),
      ),
    ),
  );
}
