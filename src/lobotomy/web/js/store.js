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
  lens: new Map(), // replyId -> { rows, probs } logit-lens grid (fixed per reply)
  damage: null, // { replyId, mean_kl, ... } for the latest reply under current edits
  hoverRow: null,
  presets: [],
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
  api.presets().then((p) => {
    state.presets = p;
    emit();
  }).catch(() => {});
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
  if (state.busy) pollWhileBusy();
}

// After a reload mid-generation there is no local stream to clear `busy`;
// poll until the server is free, then pick up the new reply.
async function pollWhileBusy() {
  while (state.busy && !state.streaming) {
    await new Promise((r) => setTimeout(r, 1000));
    try {
      const s = await api.state();
      state.busy = s.busy;
      state.exchanges = s.exchanges;
      state.edits = s.edits;
    } catch {
      return;
    }
  }
  const last = lastReply();
  if (last) await ensureDetail(last.id).catch(() => {});
  emit();
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

const pendingDetails = new Map();
export function ensureDetail(replyId) {
  if (state.details.has(replyId)) return Promise.resolve(state.details.get(replyId));
  if (!pendingDetails.has(replyId)) {
    const p = Promise.all([api.reply(replyId), api.layers(replyId)])
      .then(([detail, layers]) => {
        detail.layers = layers.layers;
        state.details.set(replyId, detail);
        return detail;
      })
      .finally(() => pendingDetails.delete(replyId));
    pendingDetails.set(replyId, p);
  }
  return pendingDetails.get(replyId);
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

export async function loadRecipe(edits, name = null) {
  state.edits = (await guard(api.loadEdits(edits))).edits;
  editsChanged();
  toast(name ? `${name}: ${state.edits.length} edits. Ask again to see it.` : `Loaded ${state.edits.length} edits`);
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

// Scans are deduplicated by key; results are dropped whenever edits change
// (editsChanged clears state.scans), and views re-request what they need.
const pendingScans = new Map();
export function scan(replyId, kind, layer = null) {
  const key = `${replyId}:${kind}:${layer}`;
  if (state.scans.has(key)) return Promise.resolve(state.scans.get(key));
  if (pendingScans.has(key)) return pendingScans.get(key);
  const params = { kind };
  if (layer !== null) params.layer = layer;
  const sig = editSignature(state.edits);
  const p = api
    .scan(replyId, params)
    .then((res) => {
      // ignore results computed under edits that have since changed
      if (editSignature(state.edits) === sig) state.scans.set(key, res.results);
      emit();
      return res.results;
    })
    .catch((e) => {
      toast(e.message, true);
      throw e;
    })
    .finally(() => pendingScans.delete(key));
  pendingScans.set(key, p);
  return p;
}

const pendingLens = new Map();
export function lens(replyId) {
  if (state.lens.has(replyId)) return Promise.resolve(state.lens.get(replyId));
  if (!pendingLens.has(replyId)) {
    pendingLens.set(
      replyId,
      api
        .lens(replyId)
        .then((g) => {
          state.lens.set(replyId, g);
          emit();
          return g;
        })
        .catch((e) => {
          toast(e.message, true);
          throw e;
        })
        .finally(() => pendingLens.delete(replyId)),
    );
  }
  return pendingLens.get(replyId);
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
