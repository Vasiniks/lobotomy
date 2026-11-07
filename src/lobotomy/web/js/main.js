// Boot, routing and the render loop.
//
// Routes:  #/                    chat + layer stack
//          #/layer/<L>           one layer opened up
//          #/token/<reply>/<row> one word of a reply
//          #/compare/<exchange>  answers to one question side by side

import * as S from "./store.js";
import { state } from "./store.js";
import { closePopover, renderTopbar } from "./components.js";
import { refreshColours } from "./util.js";
import { renderChatView } from "./views/chat.js";
import { renderLayerView, setLayerRow } from "./views/layer.js";
import { renderTokenView } from "./views/token.js";
import { renderCompareView } from "./views/compare.js";

const topbar = document.getElementById("topbar");
const body = document.getElementById("body");

function parseRoute() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "layer" && parts[1] !== undefined) return { view: "layer", layer: Number(parts[1]) };
  if (parts[0] === "token" && parts.length === 3) return { view: "token", replyId: Number(parts[1]), row: Number(parts[2]) };
  if (parts[0] === "compare" && parts[1] !== undefined) return { view: "compare", exchangeId: Number(parts[1]) };
  return { view: "chat" };
}

function crumbs(route) {
  if (route.view === "layer") return [{ label: "Chat", href: "#/" }, { label: `Layer ${route.layer}` }];
  if (route.view === "token") return [{ label: "Chat", href: "#/" }, { label: "Word" }];
  if (route.view === "compare") return [{ label: "Chat", href: "#/" }, { label: "Compare" }];
  return [];
}

function render() {
  if (!state.ready) return;
  const route = state.route;
  renderTopbar(topbar, crumbs(route));
  document.title = route.view === "layer" ? `Layer ${route.layer} · lobotomy` : "lobotomy";
  if (route.view === "layer") renderLayerView(body, route.layer);
  else if (route.view === "token") renderTokenView(body, route.replyId, route.row);
  else if (route.view === "compare") renderCompareView(body, route.exchangeId);
  else renderChatView(body);
}

window.addEventListener("hashchange", () => {
  const prev = state.route;
  state.route = parseRoute();
  closePopover();
  // A layer opened from the chat shows the whole reply; from a word, that word.
  if (state.route.view === "layer" && prev.view !== "token" && prev.view !== "layer") setLayerRow(null);
  if (state.route.view !== prev.view) body.scrollTop = 0;
  render();
});

// When a new reply lands while looking at an old word or layer, the views
// follow the latest reply automatically (they read S.lastReply()).
S.subscribe(render);

// Keyboard: "/" focuses the message box.
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !e.target.closest("input, textarea")) {
    const ta = document.getElementById("ask");
    if (ta) {
      e.preventDefault();
      ta.focus();
    }
  }
});

window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
  let saved = null;
  try {
    saved = localStorage.getItem("lobotomy-theme");
  } catch {}
  if (!saved) {
    document.documentElement.dataset.theme = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    refreshColours();
    render();
  }
});

state.route = parseRoute();
S.load()
  .then(render)
  .catch(() => {
    body.innerHTML = "";
    const p = document.createElement("div");
    p.className = "loading";
    p.textContent = "Couldn't reach the lobotomy server. Is `lobotomy serve` running?";
    body.append(p);
  });
