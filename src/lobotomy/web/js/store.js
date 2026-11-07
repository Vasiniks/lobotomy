// App state and every action that talks to the engine.

import { api } from "./api.js";
import { debounce } from "./util.js";

const listeners = new Set();

export const state = {
  ready: false,
  model: null,
  settings: null,
  edits: [],
  exchanges: [],
  busy: false,
  streaming: null, // { exchangeId, user, text, healthy }
  details: new Map(), // replyId -> reply detail (+ .layers)
  layerData: new Map(), // `${replyId}:${layer}:${rows}` -> layer payload
  scans: new Map(), // `${replyId}:${kind}:${layer}` -> results (for the current edits)
  damage: null, // { replyId, mean_kl, ... } for the latest reply under current edits
  hoverRow: null,
  route: { view: "chat" },
  error: null,
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

let scheduled = false;
export function emit() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    for (const fn of listeners) fn(state);
  });
}

// --- toasts -----------------------------------------------------------------------

let toastTimer = null;
export function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = isError ? `Error: ${msg}` : msg;
  el.classList.toggle("is-error", isError);
  el.classList.add("is-on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("is-on"), isError ? 6000 : 2400);
}

async function guard(p) {
  try {
    return await p;
  } catch (e) {
    toast(e.message || String(e), true);
    throw e;
  }
}

// --- loading ----------------------------------------------------------------------

export async function load() {
  const s = await guard(api.state());
  state.model = s.model;
  state.settings = s.settings;
  state.edits = s.edits;
  state.exchanges = s.exchanges;
  state.busy = s.busy;
  state.ready = true;
  const last = lastReply();
  if (last) await ensureDetail(last.id);
  emit();
  refreshDamage();
}

export function lastReply() {
  for (let i = state.exchanges.length - 1; i >= 0; i--) {
    const r = state.exchanges[i].replies;
    if (r.length) return r[r.length - 1];
  }
  return null;
}

export function findReply(id) {
  for (const ex of state.exchanges) for (const r of ex.replies) if (r.id === id) return r;
  return null;
}

export async function ensureDetail(replyId) {
  if (state.details.has(replyId)) return state.details.get(replyId);
  const [detail, layers] = await Promise.all([api.reply(replyId), api.layers(replyId)]);
  detail.layers = layers.layers;
  state.details.set(replyId, detail);
  return detail;
}

export async function layerData(replyId, layer, rows = "reply") {
  const key = `${replyId}:${layer}:${rows}`;
  if (!state.layerData.has(key)) state.layerData.set(key, await guard(api.layer(replyId, layer, rows)));
  return state.layerData.get(key);
}

// --- chat ---------------------------------------------------------------------------

async function runStream(start, exchange) {
  state.busy = true;
  state.streaming = { exchangeId: exchange?.id ?? null, user: exchange?.user, text: "", healthy: false };
  emit();
  let done = null;
  try {
    done = await start((ev) => {
      if (ev.type === "start") {
        state.streaming.exchangeId = ev.exchange_id;
        state.streaming.healthy = ev.healthy;
        emit();
      } else if (ev.type === "token") {
        state.streaming.text += ev.text;
        onToken?.(state.streaming);
      }
    });
  } catch (e) {
    toast(e.message, true);
  } finally {
    state.busy = false;
    state.streaming = null;
  }
  const s = await api.state();
  state.exchanges = s.exchanges;
  state.edits = s.edits;
  if (done?.reply) {
    await ensureDetail(done.reply.id);
    state.damage = null;
    refreshDamage();
  }
  emit();
  return done?.reply ?? null;
}

// The chat view registers a fast path for streamed tokens, to avoid re-rendering.
let onToken = null;
export function setTokenHandler(fn) {
  onToken = fn;
}

export async function send(text) {
  text = text.trim();
  if (!text || state.busy) return null;
  const ex = { id: null, user: text, replies: [] };
  state.exchanges.push(ex);
  return runStream((cb) => api.chat(text, cb), ex);
}

export async function askAgain(healthy = false) {
  if (state.busy || !state.exchanges.length) return null;
  const ex = state.exchanges[state.exchanges.length - 1];
  return runStream((cb) => api.again({ healthy }, cb), ex);
}

export async function cancel() {
  await api.cancel();
}

export async function newChat() {
  await guard(api.resetChat());
  state.exchanges = [];
  state.details.clear();
  state.layerData.clear();
  state.scans.clear();
  state.damage = null;
  location.hash = "#/";
  emit();
}

// --- edits ----------------------------------------------------------------------------

export function sameTarget(a, b) {
  return (
    a.kind === b.kind &&
    (a.layer ?? null) === (b.layer ?? null) &&
    (a.index ?? null) === (b.index ?? null) &&
    (a.part ?? "all") === (b.part ?? "all") &&
    (a.tensor ?? null) === (b.tensor ?? null)
  );
}

export function editFor(target, op = "scale") {
  return state.edits.find((e) => e.op === op && sameTarget(e.target, target)) || null;
}

export function editsInLayer(layer) {
  return state.edits.filter((e) => e.target.layer === layer);
}

function editsChanged() {
  state.scans.clear();
  refreshDamage();
  emit();
}

// Set a target's scale (1 removes the edit). Serialised so slider drags stay ordered.
let chain = Promise.resolve();
export function setScale(target, value, note = "") {
  chain = chain.then(() => _setScale(target, value, note)).catch(() => {});
  return chain;
}
async function _setScale(target, value, note) {
  const existing = editFor(target, "scale");
  if (existing && value === 1) {
    state.edits = (await guard(api.removeEdit(existing.id))).edits;
  } else if (existing) {
    if (existing.value === value) return;
    state.edits = (await guard(api.updateEdit(existing.id, value))).edits;
  } else if (value !== 1) {
    state.edits = (await guard(api.addEditFull({ target, op: "scale", value, note }))).edits;
  }
  editsChanged();
}

export async function addNoise(target, std, note = "") {
  const existing = editFor(target, "noise");
  if (existing) state.edits = (await guard(api.updateEdit(existing.id, std))).edits;
  else state.edits = (await guard(api.addEditFull({ target, op: "noise", value: std, note }))).edits;
  editsChanged();
}

export async function addMany(edits) {
  state.edits = (await guard(api.addEdits(edits))).edits;
  editsChanged();
}

export async function undo(id) {
  state.edits = (await guard(api.removeEdit(id))).edits;
  editsChanged();
}

export async function backToHealthy() {
  state.edits = (await guard(api.resetEdits())).edits;
  editsChanged();
}

export async function loadRecipe(edits) {
  state.edits = (await guard(api.loadEdits(edits))).edits;
  editsChanged();
  toast(`Loaded ${state.edits.length} edits`);
}

export async function saveSettings(patch) {
  state.settings = await guard(api.settings(patch));
  emit();
}

// --- measurements -----------------------------------------------------------------------

// The damage meter: KL from the healthy model on the latest reply, under the
// current edits. Debounced so dragging a slider doesn't flood the engine.
export const refreshDamage = debounce(async () => {
  const r = lastReply();
  if (!r || state.busy) return;
  if (!state.edits.length) {
    state.damage = { replyId: r.id, mean_kl: 0, healthy: true };
    emit();
    return;
  }
  try {
    const d = await api.damage(r.id);
    state.damage = { replyId: r.id, ...d };
    emit();
  } catch {
    /* backend may not support it */
  }
}, 350);

export async function scan(replyId, kind, layer = null) {
  const key = `${replyId}:${kind}:${layer}`;
  if (state.scans.has(key)) return state.scans.get(key);
  const params = { kind };
  if (layer !== null) params.layer = layer;
  const res = await guard(api.scan(replyId, params));
  state.scans.set(key, res.results);
  emit();
  return res.results;
}

export function scanResult(replyId, kind, layer = null) {
  return state.scans.get(`${replyId}:${kind}:${layer}`) || null;
}

// edits made since a reply was generated?
export function editSignature(edits) {
  return edits.map((e) => `${e.id}:${e.op}:${e.value}`).join("|");
}
export function editsSince(reply) {
  if (!reply) return false;
  return editSignature(state.edits) !== editSignature(reply.healthy ? [] : reply.edits);
}
