// Small helpers: DOM building, number formatting, the design's colour bins.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const svgNS = "http://www.w3.org/2000/svg";
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) el.setAttribute(k, v);
  for (const c of children.flat()) if (c) el.append(c);
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return x.toExponential(1);
  return x.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : digits);
}

// KL costs: fixed decimals, never scientific notation.
export function fmtCost(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  if (x < 0.001) return "<0.001";
  return x < 1 ? x.toFixed(3) : x.toFixed(2);
}

export function pct(x) {
  return `${Math.round(x * 100)}%`;
}

// Token text made visible: spaces as ␣, newlines as ↵.
export function showToken(t) {
  return t.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
}

// --- activity bins ------------------------------------------------------------
// The design maps |activation| to 9 lightness steps, relative to a reference
// (each view picks its own, usually a high percentile of what's on screen).

export function percentile(values, p) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return 0;
  return v[Math.min(v.length - 1, Math.floor((p / 100) * v.length))];
}

// gamma < 1 lifts weak values (good for a few bars); gamma = 1 keeps a dark
// field where only the strongest cells glow (the neuron grid).
export function actBin(v, ref, gamma = 0.5) {
  if (!Number.isFinite(v) || ref <= 0 || v <= ref * 0.002) return 0;
  const t = Math.min(1, v / ref);
  return Math.max(1, Math.min(8, Math.round(8 * t ** gamma)));
}

// Bin a value against the range of its neighbours (a layer's quietest to busiest
// token). A layer's average activity barely moves between tokens in absolute
// terms, so stretching min..max is what makes the differences visible.
export function rangeBin(v, values) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const x of values) {
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  if (!(hi - lo > 1e-12)) return 4;
  const t = (v - lo) / (hi - lo);
  return Math.max(1, Math.min(8, Math.round(1 + 7 * t)));
}

// CSS colour values for the bins, read once from tokens (they change with theme).
let actColours = null;
export function actColour(bin) {
  if (!actColours) refreshColours();
  return actColours[bin];
}
export function refreshColours() {
  const cs = getComputedStyle(document.documentElement);
  actColours = Array.from({ length: 9 }, (_, i) => cs.getPropertyValue(`--act-${i}`).trim());
}

// --- edit colours ---------------------------------------------------------------
// Steps from tokens.css: down-1 x0.5..<1, down-2 x0.25..<0.5, down-3 >0..<0.25,
// down-4 off; up-1 >1..x2, up-2 >x2..x4, up-3 >x4..x8, up-4 >x8.

export function editClass(edit) {
  if (!edit) return "";
  if (edit.op === "noise") return "edit-noise";
  const v = edit.value;
  if (v === 0) return "edit-off";
  if (v < 0.25) return "edit-down-3";
  if (v < 0.5) return "edit-down-2";
  if (v < 1) return "edit-down-1";
  if (v === 1) return "edit-none";
  if (v <= 2) return "edit-up-1";
  if (v <= 4) return "edit-up-2";
  if (v <= 8) return "edit-up-3";
  return "edit-up-4";
}

export function fmtScale(v) {
  if (v === 0) return "off";
  return `×${Number(Number(v).toPrecision(3))}`;
}

export function editValueText(edit) {
  if (edit.op === "noise") return `noise ${Number(Number(edit.value).toPrecision(3))}`;
  return fmtScale(edit.value);
}

// The scale slider is log2 from -4 to 4 in steps of 0.25 (design: layer.html).
// The leftmost notch means off; whole steps are powers of two.
export const SLIDER = { min: -4, max: 4, step: 0.25 };
export function sliderToValue(l) {
  l = Number(l);
  if (l <= SLIDER.min) return 0;
  return Number((2 ** l).toPrecision(3));
}
export function valueToSlider(v) {
  if (!(v > 0)) return SLIDER.min;
  return Math.min(SLIDER.max, Math.max(SLIDER.min + SLIDER.step, Math.round(Math.log2(v) / SLIDER.step) * SLIDER.step));
}

export function debounce(fn, ms) {
  let t = null;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

// When a view starts loading something new, replace the old view at once
// (instead of leaving it on screen under the new URL).
export function showLoading(body, viewKey) {
  if (body.dataset.view === viewKey) return;
  body.dataset.view = `${viewKey}:loading`;
  clear(body);
  body.append(h("div", { class: "loading" }, h("p", {}, "Loading…")));
}
