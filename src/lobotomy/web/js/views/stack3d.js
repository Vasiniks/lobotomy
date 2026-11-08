// The 3D layer stack ("Prisms").
//
// Every layer is a sheet of neurons (64 per row), stacked like floors on a dark
// stage, because activity is drawn as light. A word climbs the stack: where it is,
// the neurons that fired rise as columns of light, taller for harder firing, and
// arcs of light reach out to them. Behind the word the columns settle into lit
// tiles and the last words' glow fades fast, so the tower stays calm. The timeline
// plays, pauses and steps through the reply: one word is a big step, one layer a
// small step. Click a sheet (or its number) to pull it out, then a neuron to edit
// it. While the model writes, the sheets light up live, one word at a time.
//
// The sizes (layers, neurons per layer, heads) come from the model, so the 0.8B
// (64 x 56 sheets) and the 2B (64 x 96) both work.
//
// The canvas is built once and moved between renders (the app re-renders the
// whole chat view on every state change), so the WebGL context survives.

import * as THREE from "three";
import { OrbitControls } from "/assets/vendor/three/OrbitControls.js";
import { api } from "../api.js";
import * as S from "../store.js";
import { state } from "../store.js";
import { closePopover, openEditPopover } from "../components.js";
import { fmt, h, quoteTok, showToken, svgNS } from "../util.js";

const COLS = 64;
const BASE = { sp: 0.034, gap: 0.135 }; // neuron spacing and layer gap at 100%
// what the view panel changes; remembered per browser. sp spreads the neurons within a
// layer (horizontal), gap spreads the layers (vertical), size scales what's drawn.
const LAYOUT = { sp: 1, gap: 1, size: 1, open: true, embers: true };
const LAYOUT_KEY = "lobotomy-3d-layout-v3"; // v3: Prisms, sliders are multiples of the default
const RANGE = { sp: [0.6, 2.6], gap: [0.45, 4], size: [0.4, 2.5] };
const FIT = 0.9; // "Fit to screen": the tower fills 90% of the stage
const LINES = 48; // lines (and columns) per layer, at most
const LAYERS_PER_SECOND = 16; // at 1x a word takes 1.5 s to climb 24 layers
const SPEEDS = [
  [0.25, "¼"],
  [0.5, "½"],
  [1, "1"],
  [2, "2"],
  [4, "4"],
  [8, "8"],
];
const STALE = 0.22; // layers the word hasn't reached yet still hold the previous word, dim
const EMBER = [0.3, 0.1]; // afterglow: how much of the last two words' firing still glows
const WIRE_PER_LAYER = 4; // a clicked neuron's strongest connections into each later layer
const LOOK_LINES = 8; // attention: lines to the words a layer (or head) looked at most
const MAX_MARKS = 220;

function loadLayout() {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null");
    if (v && typeof v === "object") {
      const out = { ...LAYOUT, ...v };
      for (const k of Object.keys(RANGE)) if (!Number.isFinite(out[k])) out[k] = LAYOUT[k];
      return out;
    }
  } catch {}
  return { ...LAYOUT };
}
function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(V.layout));
  } catch {}
}

let V = null;

// --- public --------------------------------------------------------------------------------

// Put the 3D stack into `section` (built on first use) and bring it up to date.
export function mountStack3D(section) {
  if (!state.model) return;
  if (!V) V = build();
  section.append(V.root);
  V.tl.reveal(); // moving the node resets the track's scroll position
  if (!V.running) {
    V.running = true;
    V.last = performance.now() / 1000;
    V.renderer.setAnimationLoop(frame);
  }
  sync();
}

// Live data while the model writes (wired in once; see store.onStream).
S.onStream((ev) => {
  if (!V) return;
  if (ev.type === "start") startLive();
  else if (ev.type === "token" && ev.flow) addLiveStep(ev);
  else if (ev.type === "end") endLive();
});

// --- colours: the stage is dark in both themes, so read the dark tokens ------------------------

function readColours() {
  const probe = h("div", { "data-theme": "dark", style: { display: "none" } });
  document.body.append(probe);
  const cs = getComputedStyle(probe);
  const v = (n, fallback = "#888888") => cs.getPropertyValue(n).trim() || fallback;
  const c = {
    act: Array.from({ length: 9 }, (_, i) => v(`--act-${i}`)),
    down: [1, 2, 3, 4].map((i) => v(`--edit-down-${i}`)),
    up: [1, 2, 3, 4].map((i) => v(`--edit-up-${i}`)),
    noise: v("--ink-3"),
    well: v("--surface-well"),
    deep: v("--surface-deep"),
    ink1: v("--ink-1"),
    ink2: v("--ink-2"),
  };
  probe.remove();
  return c;
}
const colour = (css) => new THREE.Color().setStyle(css);
// raw sRGB bytes, so the shaders reproduce the design's colours exactly
function paletteTexture(list) {
  const bytes = new Uint8Array(16 * 4);
  list.forEach((css, i) => {
    const n = colour(css).getHex();
    bytes.set([(n >> 16) & 255, (n >> 8) & 255, n & 255, 255], i * 4);
  });
  const tex = new THREE.DataTexture(bytes, 16, 1, THREE.RGBAFormat);
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}
// replaceChildren, minus the parts that are null or false
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
const rawVec3 = (css, out = new THREE.Vector3()) => {
  const n = colour(css).getHex();
  return out.set(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};
// the edit palette, indexed by editCode()
function editPalette(c) {
  return [c.ink2, ...c.down, ...c.up, c.down[3], c.noise];
}
const EDIT_VARS = ["--ink-2", "--edit-down-1", "--edit-down-2", "--edit-down-3", "--edit-down-4", "--edit-up-1", "--edit-up-2", "--edit-up-3", "--edit-up-4", "--edit-down-4", "--ink-3"];
function editCode(e) {
  if (!e) return 0;
  if (e.op === "noise") return 10;
  const v = e.value;
  if (v === 0) return 9;
  if (v < 0.25) return 3;
  if (v < 0.5) return 2;
  if (v < 1) return 1;
  if (v <= 2) return 5;
  if (v <= 4) return 6;
  if (v <= 8) return 7;
  return 8;
}

// --- shaders ------------------------------------------------------------------------------------

const COMMON = /* glsl */ `
  uniform float uSp, uRowsHalf;
  uniform sampler2D uRamp, uEdits;
  vec3 cellPos(float i) { return vec3((mod(i, 64.0) - 31.5) * uSp, 0.0, (floor(i / 64.0) - uRowsHalf) * uSp); }
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  vec3 editCol(float c) { return texture2D(uEdits, vec2((c + 0.5) / 16.0, 0.5)).rgb; }
`;

// the stage's own light: well in the middle, deep at the edges, a little grain against banding
const BG_VS = `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.9999, 1.0); }`;
const BG_FS = /* glsl */ `
  uniform vec3 uBg, uDeep; varying vec2 vUv;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  void main() {
    vec2 p = (vUv - vec2(0.47, 0.5)) * vec2(1.25, 1.0);
    vec3 c = mix(uBg, uDeep, smoothstep(0.18, 0.78, length(p)));
    c += (hash(gl_FragCoord.xy) - 0.5) / 255.0;
    gl_FragColor = vec4(c, 1.0);
  }`;

// a sheet: corner marks (a full frame on attention layers), the word's line of light
// reading across it, and a wash in the edit colour when the whole layer is edited
const SHEET_VS = /* glsl */ `
  uniform vec2 uPlateMin, uPlateMax;
  uniform float uFogNear, uFogFar;
  varying vec2 vL; varying float vFog;
  void main() {
    vec2 l = mix(uPlateMin, uPlateMax, vec2(position.x + 0.5, position.z + 0.5));
    vL = l;
    vec4 mv = modelViewMatrix * vec4(l.x, 0.0, l.y, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const SHEET_FS = /* glsl */ `
  ${COMMON}
  uniform vec3 uInk;
  uniform vec2 uGridHalf;
  uniform float uDim, uCur, uHot, uAttn, uTime, uMotion, uArrived, uLE, uLEAt, uStrip, uFocus;
  varying vec2 vL; varying float vFog;
  float band(float d, float w) { return 1.0 - smoothstep(w - 0.5, w + 0.5, d); }
  void main() {
    float pw = max(max(fwidth(vL.x), fwidth(vL.y)), 1e-5);
    vec2 g0 = -uGridHalf - vec2(uSp * 0.9);
    vec2 g1 = uGridHalf + vec2(uSp * 0.9);
    float dx = min(vL.x - g0.x, g1.x - vL.x);
    float dz = min(vL.y - g0.y, g1.y - vL.y);
    float lw = 1.0 + 0.8 * uCur + 0.5 * uHot + 0.6 * step(0.5, uLE);
    float onLR = band(abs(dx) / pw, lw) * step(-pw, dz);
    float onFB = band(abs(dz) / pw, lw) * step(-pw, dx);
    float arm = uSp * (uAttn > 0.5 ? 8.0 : 4.0);
    float corner = max(onLR * step(dz, arm), onFB * step(dx, arm));
    float frame = max(onLR, onFB);
    float flashLE = uMotion * step(uLEAt, uTime) * exp(-max(0.0, uTime - uLEAt) * 2.2);
    vec3 ec = uLE > 0.5 ? editCol(uLE) : (uCur > 0.5 ? uInk : ramp(0.62));
    float I = corner * mix(0.26, 1.0, max(uFocus, uHot)) + frame * (uAttn > 0.5 ? mix(0.14, 0.5, uFocus) : 0.0) + frame * 0.3 * step(0.5, uLE);
    vec3 c = ec * I;
    float inside = step(0.0, dx) * step(0.0, dz);
    if (uLE > 0.5) c += editCol(uLE) * inside * (0.05 + 0.18 * flashLE);
    if (uHot > 0.5) c += ramp(0.5) * inside * 0.025;
    float u = (uTime - uArrived) / 0.6;
    if (uMotion > 0.5 && u > 0.0 && u < 1.0) {
      float zz = mix(g1.y, g0.y, u);
      c += ramp(1.0) * exp(-pow((vL.y - zz) / (uSp * 1.6), 2.0)) * (1.0 - u) * 0.38 * step(0.0, dx);
    }
    // the heads' strip: a dotted hairline under the comb
    float zh = -uGridHalf.y - uSp * 0.9 - uStrip * 0.42;
    c += ramp(0.5) * band(abs(vL.y - zh) / pw, 0.5) * step(abs(vL.x), uGridHalf.x) * step(0.5, fract(vL.x / (uSp * 2.0))) * 0.35;
    c *= uDim * (1.0 - 0.7 * vFog);
    gl_FragColor = vec4(c, 1.0);
  }`;

// neurons: faint points at rest, a small pool of light at the foot of a firing one
// (its column stands on it), embers where the last words fired, your edits in colour
const POINT_VS = /* glsl */ `
  ${COMMON}
  attribute float aI, act, glow, birth, ed, edAt;
  uniform float uSize, uScale, uDpr, uTime, uMotion, uFogNear, uFogFar, uFocus;
  varying float vAct, vGlow, vEd, vFlash, vFog, vPop, vDust;
  void main() {
    vec4 mv = modelViewMatrix * vec4(cellPos(aI), 1.0);
    float pitch = uSp * uScale / max(0.01, -mv.z);   // device px between neighbours
    float css = pitch / uDpr;
    float k = mix(1.0, 0.5, smoothstep(6.0, 16.0, css));
    float pop = uMotion * exp(-max(0.0, uTime - birth) * 3.2);
    float fl = uMotion * step(edAt, uTime) * exp(-max(0.0, uTime - edAt) * 2.4);
    float s;
    if (act > 0.0) s = max(pitch * (1.6 + 3.6 * act) * k * uSize, 5.0 * uDpr) * (1.0 + 0.75 * pop) * 0.42;
    else if (ed > 0.5) s = max(pitch * 1.25 * uSize, 4.0 * uDpr);
    else if (glow > 0.0) s = max(pitch * (1.0 + 1.6 * glow) * k * uSize, 2.5 * uDpr);
    else s = max(pitch * 0.5 * uSize, 1.4 * uDpr);
    gl_PointSize = s * (1.0 + 1.3 * fl);
    vAct = act; vGlow = glow; vEd = ed; vFlash = fl; vPop = pop;
    // the resting lattice: clear on the sheet in focus, a whisper elsewhere, gone when it would shimmer
    vDust = mix(0.05 * smoothstep(5.0, 10.0, css), 1.0, uFocus) * mix(0.55, 1.0, smoothstep(3.0, 8.0, css));
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const POINT_FS = /* glsl */ `
  ${COMMON}
  uniform float uDim, uDust;
  varying float vAct, vGlow, vEd, vFlash, vFog, vPop, vDust;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    vec3 c;
    if (vAct > 0.0) {
      float core = 1.0 - smoothstep(0.14, 0.3, r);
      float halo = exp(-r * r * 5.0) * (1.0 - r);
      vec3 hc = vEd > 0.5 ? mix(ramp(0.75), editCol(vEd), 0.7) : ramp(0.5 + 0.5 * vAct);
      c = (hc * halo * (0.4 + 0.7 * vAct) + ramp(1.0) * core * (0.55 + 0.45 * vAct)) * (1.0 + 0.9 * vPop) * 0.6;
    } else if (vEd > 0.5) {
      c = editCol(vEd) * (1.0 - smoothstep(0.5, 0.85, r)) * 0.95;
    } else if (vGlow > 0.0) {
      c = ramp(0.35 + 0.5 * vGlow) * exp(-r * r * 3.5) * (1.0 - r) * (0.3 + 0.6 * vGlow);
    } else {
      c = ramp(0.32) * (1.0 - smoothstep(0.45, 1.0, r)) * uDust * vDust;
    }
    c += ramp(1.0) * vFlash * exp(-r * r * 3.0) * 0.9;
    c *= uDim * (1.0 - 0.72 * vFog);
    gl_FragColor = vec4(c, 1.0);
  }`;

// columns of light: tall where the word is, settling into lit tiles behind it
const COLUMN_VS = /* glsl */ `
  ${COMMON}
  attribute float aCell, aStr, aBirth, aEd;
  uniform float uGap, uSize, uTime, uMotion, uScale, uDpr, uFogNear, uFogFar, uLift;
  varying vec3 vN; varying float vS, vFog, vPop, vY, vEd;
  void main() {
    vec3 base = cellPos(aCell);
    vec4 mvc = modelViewMatrix * vec4(base, 1.0);
    float pitchPx = uSp * uScale / max(0.01, -mvc.z) / uDpr;
    float w = uSp * mix(1.4, 0.74, smoothstep(6.0, 15.0, pitchPx)) * uSize;
    float g = uMotion > 0.5 ? clamp((uTime - aBirth) / 0.34, 0.0, 1.0) : 1.0;
    g = 1.0 - (1.0 - g) * (1.0 - g) * (1.0 - g);
    float hgt = uGap * 0.75 * aStr * g * uLift + uSp * 0.3;
    vec3 p = base + vec3(position.x * w, position.y * hgt, position.z * w);
    vN = normal; vS = aStr; vY = position.y; vEd = aEd;
    vPop = uMotion * exp(-max(0.0, uTime - aBirth) * 2.8);
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const COLUMN_FS = /* glsl */ `
  ${COMMON}
  uniform float uDim, uLift;
  varying vec3 vN; varying float vS, vFog, vPop, vY, vEd;
  void main() {
    vec3 base = vEd > 0.5 ? editCol(vEd) : ramp(0.5 + 0.5 * vS);
    vec3 c; float I;
    if (vN.y > 0.5) { c = mix(base, ramp(1.0), vEd > 0.5 ? 0.2 : 0.65); I = 0.95 + 0.45 * vS; }   // the bright cap
    else if (vN.y < -0.5) { c = base; I = 0.0; }
    else { c = base; I = (0.14 + 0.8 * pow(vY, 1.4)) * (abs(vN.x) > 0.5 ? 0.7 : 1.0) * (0.5 + 0.5 * vS); }
    I *= (1.0 + 0.8 * vPop) * mix(0.3, 1.0, uLift) * uDim * (1.0 - 0.7 * vFog);
    gl_FragColor = vec4(c * I, 1.0);
  }`;

// heads: a comb of light bars on the strip along each sheet's far edge
const HEAD_VS = /* glsl */ `
  attribute float aK, aA, aEd;
  uniform float uNH, uSp, uGap, uFogNear, uFogFar, uTime, uMotion, uArrived, uStrip;
  uniform vec2 uGridHalf;
  varying vec3 vN; varying float vA, vFog, vY, vEd;
  void main() {
    float pitch = uGridHalf.x * 2.0 / uNH;
    float g = uMotion > 0.5 ? clamp((uTime - uArrived - 0.05) / 0.3, 0.0, 1.0) : 1.0;
    g = 1.0 - (1.0 - g) * (1.0 - g);
    float a = (aEd > 8.5 && aEd < 9.5) ? 0.0 : aA;   // a head switched off lies flat
    vec3 base = vec3((aK - (uNH - 1.0) * 0.5) * pitch, 0.0, -uGridHalf.y - uSp * 0.9 - uStrip * 0.42);
    float hgt = 0.003 + uGap * 0.55 * a * g;
    vec3 p = base + vec3(position.x * pitch * 0.74, position.y * hgt, position.z * uStrip * 0.5);
    vN = normal; vA = aA; vY = position.y; vEd = aEd;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const HEAD_FS = /* glsl */ `
  ${COMMON}
  uniform float uDim;
  varying vec3 vN; varying float vA, vFog, vY, vEd;
  void main() {
    vec3 c = vEd > 0.5 ? editCol(vEd) : ramp(0.4 + 0.6 * vA);
    float I = (vEd > 0.5 ? 0.3 + 0.5 * vA : 0.025 + 0.32 * vA) * mix(0.15, 1.0, vY * vY);
    gl_FragColor = vec4(c * I * uDim * (1.0 - 0.7 * vFog), 1.0);
  }`;

// lines: arcs of light from the word's line (the spine) out to each column's cap,
// thicker for harder firing, with comets running out
const LINE_VS = /* glsl */ `
  ${COMMON}
  attribute float aCell, aStr, aBirth;
  uniform float uTime, uMotion, uArc, uEndY, uRMin, uRMax, uGrow, uFogNear, uFogFar, uGap, uSize, uFocus;
  varying float vU, vS, vFog;
  void main() {
    float g = uMotion > 0.5 ? clamp((uTime - aBirth) / uGrow, 0.0, 1.0) : 1.0;
    g = 1.0 - (1.0 - g) * (1.0 - g) * (1.0 - g);
    vec3 A = vec3(0.0);
    vec3 B = cellPos(aCell) + vec3(0.0, uEndY * uGap * aStr, 0.0);
    float len = length(B - A);
    vec3 C = (A + B) * 0.5 + vec3(0.0, uArc * len, 0.0);
    float t = position.y * g;
    vec3 P = mix(mix(A, C, t), mix(C, B, t), t);
    vec3 T = normalize(2.0 * (1.0 - t) * (C - A) + 2.0 * t * (B - C) + vec3(1e-5, 0.0, 0.0));
    vec3 side = normalize(cross(T, vec3(0.0, 1.0, 0.0)) + vec3(0.0, 0.0, 1e-5));
    vec3 up = cross(side, T);
    float r = mix(uRMin, uRMax, aStr) * mix(1.0, 0.8, t) * mix(0.8, 1.0, uSize) * mix(0.6, 1.0, uFocus);
    P += side * position.x * r + up * position.z * r;
    vU = t; vS = aStr;
    vec4 mv = modelViewMatrix * vec4(P, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const LINE_FS = /* glsl */ `
  ${COMMON}
  uniform float uDim, uTime, uMotion, uFocus;
  varying float vU, vS, vFog;
  void main() {
    vec3 c = ramp(0.5 + 0.5 * vS);
    float base = (0.08 + 0.3 * vS) * mix(0.35, 1.0, vU);
    float f = fract(vU * 1.15 - uTime * 0.75 - vS * 0.41);
    float comet = smoothstep(0.0, 0.2, f) * (1.0 - smoothstep(0.2, 0.235, f));
    float I = (base + comet * (0.55 + 0.9 * vS) * uMotion) * uDim * (1.0 - 0.72 * vFog);
    I *= mix(0.16, 1.0, uFocus) * smoothstep(0.0, 0.14, vU);   // dim off the word's layer; no pile-up at the spine
    gl_FragColor = vec4(c * I, 1.0);
  }`;

const SPINE_VS = /* glsl */ `
  uniform float uY0, uY1;
  varying float vY; varying vec3 vN, vV;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vY = (wp.y - uY0) / (uY1 - uY0);
    vN = normalize(normalMatrix * normal);
    vec4 mv = viewMatrix * wp;
    vV = normalize(-mv.xyz);
    gl_Position = projectionMatrix * mv;
  }`;
const SPINE_FS = /* glsl */ `
  ${COMMON}
  uniform float uProgress, uTime, uMotion;
  varying float vY; varying vec3 vN, vV;
  void main() {
    float rim = pow(abs(dot(normalize(vN), normalize(vV))), 1.5);
    if (vY > uProgress) { gl_FragColor = vec4(ramp(0.55) * 0.16 * rim, 1.0); return; }
    float f = fract(vY * 8.0 - uTime * 1.3);
    float band = smoothstep(0.0, 0.1, f) * (1.0 - smoothstep(0.1, 0.45, f));
    gl_FragColor = vec4(ramp(1.0) * (0.42 + 0.6 * band * uMotion) * rim, 1.0);
  }`;
// the word: a point of light with a cross flare
const BEAD_VS = `uniform float uDpr, uTime, uMotion; void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = (64.0 + 6.0 * sin(uTime * 3.0) * uMotion) * uDpr; }`;
const BEAD_FS = /* glsl */ `
  ${COMMON}
  void main() {
    vec2 q = (gl_PointCoord - 0.5) * 2.0;
    float r = length(q);
    float core = 1.0 - smoothstep(0.06, 0.13, r);
    float halo = exp(-r * r * 9.0) * 0.7;
    float rays = (exp(-abs(q.y) * 70.0) * exp(-abs(q.x) * 3.2) + exp(-abs(q.x) * 70.0) * exp(-abs(q.y) * 3.2)) * 0.8;
    gl_FragColor = vec4(ramp(1.0) * (core * 1.4 + halo + rays), 1.0);
  }`;

// marks: your edits (a square ring with a glow), the selection, hover, wiring targets,
// the hovered edit's beacon and ripples. Drawn over everything so edits show from
// anywhere in the tower; rings face you, ripples lie on the sheet.
const MARK_VS = /* glsl */ `
  attribute float aCode, aAt, aKind, aBorn, aDim;
  varying vec2 vUv; varying float vCode, vAt, vKind, vBorn, vDim;
  void main() {
    vUv = uv; vCode = aCode; vAt = aAt; vKind = aKind; vBorn = aBorn; vDim = aDim;
    if (aKind < 1.5 || aKind > 2.5) {
      vec4 c = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      c.xy += vec2(position.x, -position.z) * length(instanceMatrix[0].xyz);
      gl_Position = projectionMatrix * c;
    } else {
      gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    }
  }`;
const MARK_FS = /* glsl */ `
  ${COMMON}
  uniform float uTime, uMotion;
  uniform vec3 uInk;
  varying vec2 vUv; varying float vCode, vAt, vKind, vBorn, vDim;
  float ringAt(float d, float r0, float r1, float aa) { return smoothstep(r0 - aa, r0 + aa, d) * (1.0 - smoothstep(r1 - aa, r1 + aa, d)); }
  void main() {
    vec2 p = vUv - 0.5;
    vec3 col; float a = 0.0;
    if (vKind > 1.5 && vKind < 2.5) {                  // ripple, spreading on the sheet
      float d = max(abs(p.x), abs(p.y)); float aa = fwidth(d);
      float u = clamp((uTime - vBorn) / 0.9, 0.0, 1.0);
      a = (1.0 - smoothstep(0.004, 0.004 + aa * 1.5, abs(d - 0.475))) * (1.0 - u) * (1.0 - u);
      col = editCol(vCode);
    } else if (vKind > 2.5 && vKind < 3.5) {           // hover or a wiring target: a thin ring
      float d = max(abs(p.x), abs(p.y)); float aa = fwidth(d);
      a = ringAt(d, 0.41, 0.47, aa) * 0.85;
      col = uInk;
    } else if (vKind > 3.5) {                          // the hovered edit's beacon: a thick ring in its colour
      float d = max(abs(p.x), abs(p.y)); float aa = fwidth(d);
      a = ringAt(d, 0.36, 0.47, aa);
      col = vCode > 0.5 ? editCol(vCode) : uInk;
    } else {                                           // an edit (kind 0) or the selection (kind 1)
      float fl = uMotion * step(vAt, uTime) * exp(-max(0.0, uTime - vAt) * 2.6);
      if (vCode > 8.5 && vCode < 9.5 && vKind < 0.5 && p.y - p.x > 0.0) p.x -= 0.05;  // off: the top part slips right
      float d = max(abs(p.x), abs(p.y)); float aa = fwidth(d);
      float r0 = vKind > 0.5 ? 0.40 : 0.29, r1 = vKind > 0.5 ? 0.47 : 0.43;
      float ring = ringAt(d, r0, r1, aa);
      if (vKind < 0.5) {
        if (vCode > 8.5 && vCode < 9.5 && abs(p.y - p.x) < 0.07) ring = 0.0;          // off: the cut
        if (vCode > 9.5 && fract((abs(p.x) > abs(p.y) ? p.y : p.x) * 9.0) < 0.42) ring = 0.0;  // noise: dashed
      }
      col = vKind > 0.5 ? uInk : editCol(vCode);
      col = mix(col, uInk, fl * 0.55);
      float halo = exp(-pow((d - r1) / 0.06, 2.0)) * 0.45 * step(r1, d);
      a = max(ring, halo + fl * 0.45 * (1.0 - smoothstep(0.0, r0, d)));
    }
    if (a < 0.01) discard;
    gl_FragColor = vec4(col, a * vDim);
  }`;

// a neuron's wiring: tubes of light from it to the later neurons it feeds
const WIRE_VS = /* glsl */ `
  attribute vec3 aStart, aEnd;
  attribute float aStrength, aSign;
  varying float vY, vS, vSign;
  void main() {
    vec3 dir = aEnd - aStart;
    float len = length(dir);
    vec3 ax = dir / max(len, 1e-5);
    vec3 ref = abs(ax.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 side = normalize(cross(ax, ref));
    vec3 up2 = cross(side, ax);
    float r = mix(0.0018, 0.008, aStrength);
    vec3 bow = up2 * sin(3.14159 * position.y) * len * 0.06;
    vec3 p = aStart + ax * (position.y * len) + side * (position.x * r) + up2 * (position.z * r) + bow;
    vY = position.y; vS = aStrength; vSign = aSign;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const WIRE_FS = /* glsl */ `
  uniform vec3 uBoost, uSuppress;
  uniform float uTime, uMotion;
  varying float vY, vS, vSign;
  void main() {
    if (vSign < 0.0 && fract(vY * 22.0) < 0.42) discard;   // suppressing connections are dashed
    vec3 c = vSign > 0.0 ? uBoost : uSuppress;
    float f = fract(vY * 3.0 - uTime * 1.1);
    float band = smoothstep(0.0, 0.05, f) * (1.0 - smoothstep(0.05, 0.28, f));
    gl_FragColor = vec4(c * (0.35 + 0.35 * vS + 0.6 * band * uMotion), 1.0);
  }`;

const ADD = {
  transparent: true,
  depthWrite: false,
  depthTest: false,
  blending: THREE.CustomBlending,
  blendEquation: THREE.AddEquation,
  blendSrc: THREE.OneFactor,
  blendDst: THREE.OneFactor,
};

// --- building the scene ----------------------------------------------------------------------

function build() {
  const m = state.model;
  const L = m.n_layers;
  const N = m.n_neurons;
  const ROWS = Math.ceil(N / COLS);
  const colours = readColours();
  const motion = !matchMedia("(prefers-reduced-motion: reduce)").matches;

  // DOM: the stage is dark in both themes; the timeline follows the page
  const canvas = h("canvas", { class: "s3d-canvas", tabindex: "0", "aria-label": "The layers in 3D. Space plays, arrows step, F for full window." });
  const labelsEl = h("div", { class: "s3d-labels" });
  const tip = h("div", { class: "s3d-tip s3d-card", "aria-hidden": "true" });
  const chip = h("div", { class: "s3d-chip s3d-card" });
  const empty = h("p", { class: "s3d-empty" });
  const ioIn = h("span", { class: "s3d-io s3d-card", "aria-hidden": "true" });
  const ioOut = h("span", { class: "s3d-io s3d-card", "aria-hidden": "true" });
  const cand = h("section", { class: "s3d-cand s3d-card", "aria-label": "Next word" });
  const lookCap = h("p", { class: "s3d-lookcap s3d-card" });
  const hint = h("p", { class: "s3d-hint" }, "Drag to orbit · scroll to zoom · click a sheet or its number to pull it out");
  const beaconLabel = h("span", { class: "s3d-beacon" });
  tip.hidden = chip.hidden = ioIn.hidden = ioOut.hidden = cand.hidden = lookCap.hidden = beaconLabel.hidden = true;
  const viewEl = h("div", { class: "s3d-view" });
  const stage = h("div", { class: "s3d-stage", "data-theme": "dark" }, canvas, labelsEl, ioIn, ioOut, beaconLabel, cand, chip, tip, viewEl, lookCap, hint, empty);
  const tl = buildTimeline(L);
  const root = h("div", { class: "s3d" }, stage, tl.el);
  const lookSvg = document.createElementNS(svgNS, "svg");
  lookSvg.setAttribute("class", "s3d-look");
  lookSvg.setAttribute("aria-hidden", "true");
  const lookPaths = Array.from({ length: LOOK_LINES }, () => {
    const p = document.createElementNS(svgNS, "path");
    lookSvg.append(p);
    return p;
  });
  root.append(lookSvg);

  // renderer and camera
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(colour(colours.deep), 1);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.03, 200);
  camera.position.set(6, 4.4, 6.6);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.085;
  controls.rotateSpeed = 0.75;
  controls.zoomSpeed = 0.9;
  controls.minDistance = 0.5;
  controls.maxDistance = 40;
  const world = new THREE.Group();
  scene.add(world);

  // shared uniforms
  const uni = {
    uTime: { value: 0 },
    uMotion: { value: motion ? 1 : 0 },
    uScale: { value: 800 },
    uDpr: { value: 1 },
    uSp: { value: BASE.sp },
    uRowsHalf: { value: (ROWS - 1) / 2 },
    uGap: { value: BASE.gap },
    uSize: { value: 1 },
    uFogNear: { value: 6 },
    uFogFar: { value: 14 },
    uRamp: { value: paletteTexture(colours.act) },
    uEdits: { value: paletteTexture(editPalette(colours)) },
    uBg: { value: rawVec3(colours.well) },
    uDeep: { value: rawVec3(colours.deep) },
    uInk: { value: rawVec3(colours.ink1) },
    uGridHalf: { value: new THREE.Vector2() },
    uPlateMin: { value: new THREE.Vector2() },
    uPlateMax: { value: new THREE.Vector2() },
    uStrip: { value: 0.1 },
  };

  const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({ uniforms: uni, vertexShader: BG_VS, fragmentShader: BG_FS, depthTest: false, depthWrite: false }));
  bg.frustumCulled = false;
  bg.renderOrder = -1000;
  scene.add(bg);

  const plateGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  const boxGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
  const lineBase = new THREE.CylinderGeometry(1, 1, 1, 6, 12, true).translate(0, 0.5, 0);
  const posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3);
  const iAttr = new THREE.BufferAttribute(Float32Array.from({ length: N }, (_, i) => i), 1);
  const maxHeads = Math.max(1, ...m.layers.map((x) => x.n_heads ?? 0));
  const inst = (geo, name, n) => {
    const a = new THREE.InstancedBufferAttribute(new Float32Array(n), 1).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute(name, a);
    return a;
  };
  const instGeo = (base, withNormal) => {
    const g = new THREE.InstancedBufferGeometry();
    g.index = base.index;
    g.setAttribute("position", base.getAttribute("position"));
    if (withNormal) g.setAttribute("normal", base.getAttribute("normal"));
    g.instanceCount = 0;
    return g;
  };

  const layers = [];
  for (let l = 0; l < L; l++) {
    const g = new THREE.Group();
    world.add(g);
    const attn = m.layers[l]?.type === "attention";
    const nH = m.layers[l]?.n_heads ?? 0;
    const act = new Float32Array(N);
    const glow = new Float32Array(N);
    const birth = new Float32Array(N).fill(-100);
    const edit = new Float32Array(N);
    const editAt = new Float32Array(N).fill(-100);

    // the sheet
    const pu = { uDim: { value: 1 }, uCur: { value: 0 }, uHot: { value: 0 }, uFocus: { value: 0 }, uAttn: { value: attn ? 1 : 0 }, uArrived: { value: -100 }, uLE: { value: 0 }, uLEAt: { value: -100 } };
    const sheet = new THREE.Mesh(plateGeo, new THREE.ShaderMaterial({ uniforms: { ...uni, ...pu }, vertexShader: SHEET_VS, fragmentShader: SHEET_FS, side: THREE.DoubleSide, ...ADD }));
    sheet.frustumCulled = false;

    // the neurons
    const pgeo = new THREE.BufferGeometry();
    pgeo.setAttribute("position", posAttr);
    pgeo.setAttribute("aI", iAttr);
    const at = {
      act: new THREE.BufferAttribute(act, 1).setUsage(THREE.DynamicDrawUsage),
      glow: new THREE.BufferAttribute(glow, 1).setUsage(THREE.DynamicDrawUsage),
      birth: new THREE.BufferAttribute(birth, 1).setUsage(THREE.DynamicDrawUsage),
      ed: new THREE.BufferAttribute(edit, 1).setUsage(THREE.DynamicDrawUsage),
      edAt: new THREE.BufferAttribute(editAt, 1).setUsage(THREE.DynamicDrawUsage),
    };
    for (const [k, a] of Object.entries(at)) pgeo.setAttribute(k, a);
    const pmat = new THREE.ShaderMaterial({ uniforms: { ...uni, uDim: { value: 1 }, uDust: { value: 0.3 }, uFocus: { value: 0 } }, vertexShader: POINT_VS, fragmentShader: POINT_FS, ...ADD });
    const points = new THREE.Points(pgeo, pmat);
    points.frustumCulled = false;

    // the lines and the columns, one each per neuron that fired
    const lgeo = instGeo(lineBase, false);
    const la = { aCell: inst(lgeo, "aCell", LINES), aStr: inst(lgeo, "aStr", LINES), aBirth: inst(lgeo, "aBirth", LINES) };
    const lmat = new THREE.ShaderMaterial({
      uniforms: { ...uni, uDim: { value: 1 }, uFocus: { value: 0 }, uArc: { value: 0.34 }, uEndY: { value: 0 }, uRMin: { value: 0.0024 }, uRMax: { value: 0.0085 }, uGrow: { value: 0.38 } },
      vertexShader: LINE_VS,
      fragmentShader: LINE_FS,
      ...ADD,
    });
    const lines = new THREE.Mesh(lgeo, lmat);
    lines.frustumCulled = false;
    const cgeo = instGeo(boxGeo, true);
    const ca = { aCell: inst(cgeo, "aCell", LINES), aStr: inst(cgeo, "aStr", LINES), aBirth: inst(cgeo, "aBirth", LINES), aEd: inst(cgeo, "aEd", LINES) };
    const cmat = new THREE.ShaderMaterial({ uniforms: { ...uni, uDim: { value: 1 }, uLift: { value: 0 } }, vertexShader: COLUMN_VS, fragmentShader: COLUMN_FS, ...ADD });
    const cols = new THREE.Mesh(cgeo, cmat);
    cols.frustumCulled = false;

    // the heads
    const hgeo = instGeo(boxGeo, true);
    const ha = { aK: inst(hgeo, "aK", Math.max(1, nH)), aA: inst(hgeo, "aA", Math.max(1, nH)), aEd: inst(hgeo, "aEd", Math.max(1, nH)) };
    for (let k = 0; k < nH; k++) ha.aK.array[k] = k;
    hgeo.instanceCount = nH;
    const hmat = new THREE.ShaderMaterial({ uniforms: { ...uni, uDim: { value: 1 }, uNH: { value: Math.max(1, nH) }, uArrived: { value: -100 } }, vertexShader: HEAD_VS, fragmentShader: HEAD_FS, ...ADD });
    const heads = new THREE.Mesh(hgeo, hmat);
    heads.frustumCulled = false;
    heads.visible = nH > 0;

    g.add(sheet, lines, heads, points, cols);

    // the label: layer number, then (once the word reaches it) this layer's best guess
    // for the next word, and a bar for how sure it already was of the real one
    const lblG = h("span", { class: "g" });
    const lblBar = h("i");
    const lblLE = h("span", { class: "le", hidden: true });
    const lblK = h("span", { class: "k" }, attn ? "attention" : "");
    const label = h(
      "button",
      { type: "button", class: "s3d-lbl", "aria-label": `Pull out layer ${l}, ${attn ? "attention" : "DeltaNet"}` },
      h("span", { class: "n" }, String(l)),
      lblK,
      lblLE,
      h("span", { class: "pb" }, lblBar),
      lblG,
    );
    label.addEventListener("click", () => (V.sep === l ? putBack() : pullOut(l)));
    labelsEl.append(label);

    layers.push({
      l, g, nH, kind: attn ? "attention" : "DeltaNet",
      act, glow, birth, edit, editAt,
      lineCell: new Float32Array(LINES), lineStr: new Float32Array(LINES), lineBirth: new Float32Array(LINES).fill(-100), nLines: 0,
      headAct: new Float32Array(Math.max(1, maxHeads)), headEdit: new Float32Array(Math.max(1, maxHeads)),
      sheet, pu, at, pmat, la, lgeo, lmat, ca, cgeo, cmat, ha, hmat,
      label, lblG, lblBar, lblLE, lblK,
      home: { pos: new THREE.Vector3(), quat: new THREE.Quaternion() },
      target: { pos: new THREE.Vector3(), quat: new THREE.Quaternion() },
      dim: 1, dimTarget: 1, isCur: false, lift: 0, arrived: -100, shown: -1,
      layerEdit: null, layerEditAt: -100, editEdge: null,
    });
  }

  // the word's line through the stack, and the word itself riding it
  const spineMat = new THREE.ShaderMaterial({ uniforms: { ...uni, uY0: { value: 0 }, uY1: { value: 1 }, uProgress: { value: 0 } }, vertexShader: SPINE_VS, fragmentShader: SPINE_FS, ...ADD });
  const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, 1, 12, 1, true), spineMat);
  spine.frustumCulled = false;
  const beadGeo = new THREE.BufferGeometry();
  beadGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
  const bead = new THREE.Points(beadGeo, new THREE.ShaderMaterial({ uniforms: uni, vertexShader: BEAD_VS, fragmentShader: BEAD_FS, ...ADD }));
  bead.frustumCulled = false;
  bead.renderOrder = 150;
  world.add(spine, bead);

  // marks: edits, selection, hover, wiring targets, beacon, ripples
  const markGeo = new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2);
  const markAttr = {};
  for (const k of ["aCode", "aAt", "aKind", "aBorn", "aDim"]) {
    markAttr[k] = new THREE.InstancedBufferAttribute(new Float32Array(MAX_MARKS), 1).setUsage(THREE.DynamicDrawUsage);
    markGeo.setAttribute(k, markAttr[k]);
  }
  const marks = new THREE.InstancedMesh(
    markGeo,
    new THREE.ShaderMaterial({ uniforms: uni, vertexShader: MARK_VS, fragmentShader: MARK_FS, transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide }),
    MAX_MARKS,
  );
  marks.count = 0;
  marks.frustumCulled = false;
  marks.renderOrder = 200;
  scene.add(marks);

  // a clicked neuron's wiring into later layers
  const maxWire = WIRE_PER_LAYER * L;
  const wgeo = instGeo(lineBase, false);
  const wattr = {
    aStart: new THREE.InstancedBufferAttribute(new Float32Array(maxWire * 3), 3).setUsage(THREE.DynamicDrawUsage),
    aEnd: new THREE.InstancedBufferAttribute(new Float32Array(maxWire * 3), 3).setUsage(THREE.DynamicDrawUsage),
    aStrength: new THREE.InstancedBufferAttribute(new Float32Array(maxWire), 1).setUsage(THREE.DynamicDrawUsage),
    aSign: new THREE.InstancedBufferAttribute(new Float32Array(maxWire), 1).setUsage(THREE.DynamicDrawUsage),
  };
  for (const [k, a] of Object.entries(wattr)) wgeo.setAttribute(k, a);
  const wire = new THREE.Mesh(
    wgeo,
    new THREE.ShaderMaterial({
      uniforms: { uTime: uni.uTime, uMotion: uni.uMotion, uBoost: { value: rawVec3(colours.ink1) }, uSuppress: { value: rawVec3(colours.ink2) } },
      vertexShader: WIRE_VS,
      fragmentShader: WIRE_FS,
      ...ADD,
    }),
  );
  wire.frustumCulled = false;
  scene.add(wire);

  // the hovered edit's beacon: a line from it up out of the tower
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1, true).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ transparent: true, depthTest: false, depthWrite: false }));
  beacon.renderOrder = 210;
  beacon.frustumCulled = false;
  beacon.visible = false;
  scene.add(beacon);

  const view = {
    root, stage, canvas, labelsEl, tip, chip, empty, ioIn, ioOut, cand, lookCap, hint, beaconLabel, viewEl,
    tl, lookSvg, lookPaths,
    renderer, scene, camera, controls, world, uni, layers, spine, spineMat, bead, marks, markAttr,
    wire, wattr, beacon,
    L, N, ROWS, colours,
    M: {}, // metrics for the current layout
    data: null, // { replyId, live, steps, k, idx(s, l), val(s, l), heads(s, l) }
    loading: null,
    pos: 0, // step * L + layer, fractional while playing
    playing: false,
    speed: 1,
    sep: null, // pulled-out layer
    selected: null, // { l, n }
    selHead: null, // { l, h }
    hover: null,
    hoverNeuron: null,
    hoverHead: null,
    wiring: null, // { l, n, targets }
    ripples: [],
    editSig: null,
    editMarks: [],
    rowToStep: new Map(),
    guessCache: new Map(), // row -> per-layer lens guesses
    guessPending: new Set(),
    wordsKey: null,
    lookCache: new Map(), // row -> look data
    lookPending: new Set(),
    lookKey: null,
    lookTargets: [],
    lookShown: false,
    lensRows: null, // said row -> per-layer probability (the timeline's mini-bars)
    layout: loadLayout(),
    ride: false,
    full: false,
    userMoved: false, // the camera stops fitting itself once you orbit or zoom
    fitView: null,
    camAnim: null,
    savedView: null,
    pointer: new THREE.Vector2(),
    pointerIn: false,
    pxy: [0, 0],
    down: null,
    running: false,
    last: 0,
    liveRef: null,
    liveHeadRef: null,
    escAt: 0,
    colX: 0,
    beaconFor: null,
  };
  V = view; // the helpers below read V
  buildViewPanel();
  relayout(false);
  tl.update();
  for (const Ly of layers) {
    Ly.g.position.copy(Ly.home.pos);
    Ly.target.pos.copy(Ly.home.pos);
  }
  wirePointer();
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(recolour).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  controls.addEventListener("start", () => {
    V.camAnim = null;
  });
  setTimeout(() => hint.classList.add("is-gone"), 9000);
  return view;
}

// The stage is dark in both themes, but re-read the tokens anyway: they're the source.
function recolour() {
  const c = readColours();
  V.colours = c;
  V.uni.uRamp.value.dispose();
  V.uni.uRamp.value = paletteTexture(c.act);
  V.uni.uEdits.value.dispose();
  V.uni.uEdits.value = paletteTexture(editPalette(c));
  rawVec3(c.well, V.uni.uBg.value);
  rawVec3(c.deep, V.uni.uDeep.value);
  rawVec3(c.ink1, V.uni.uInk.value);
  V.renderer.setClearColor(colour(c.deep), 1);
  V.wire.material.uniforms.uBoost.value = rawVec3(c.ink1);
  V.wire.material.uniforms.uSuppress.value = rawVec3(c.ink2);
}

function resize() {
  const r = V.stage.getBoundingClientRect();
  if (!r.width || !r.height) return;
  V.renderer.setSize(r.width, r.height, false);
  V.camera.aspect = r.width / r.height;
  V.camera.updateProjectionMatrix();
  V.uni.uDpr.value = V.renderer.getPixelRatio();
  V.uni.uScale.value = V.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(V.camera.fov) / 2));
  V.wordsKey = null;
  if (V.sep !== null || V.ride) return;
  if (!V.userMoved) fitCamera(V.fitView != null); // the first fit jumps, later ones glide
}

// Full window: the 3D view covers the whole page; Esc or F brings it back.
function setFull(on) {
  V.full = on;
  V.root.classList.toggle("is-full", on);
  if (V.fullBtn) V.fullBtn.replaceChildren(icon(on ? "unfull" : "full"), h("span", {}, on ? "Exit full" : "Full window"));
  V.userMoved = false; // a new size: fit again
  closePopover();
  V.canvas.focus({ preventScroll: true });
}

// --- data ------------------------------------------------------------------------------------

function sync() {
  if (!V) return;
  if (state.streaming || V.data?.live) return; // the stream feeds the live view
  const last = S.lastReply();
  if (!last) {
    if (V.data) setData(null);
    showEmpty("The sheets light up after the first reply: every word climbs the layers, and the neurons that fire rise as light.");
    return;
  }
  if (V.data?.replyId === last.id || V.loading === last.id) return;
  if (V.data?.after !== undefined && last.id === V.data.after) return; // the new reply isn't in yet
  V.loading = last.id;
  showEmpty("Loading this reply's neurons…");
  api
    .flow(last.id)
    .then((f) => {
      if (S.lastReply()?.id !== last.id) return;
      setData(decodeFlow(last.id, f));
      showEmpty(null);
    })
    .catch((e) => showEmpty(`Can't show this reply in 3D: ${e.message}`))
    .finally(() => {
      if (V.loading === last.id) V.loading = null;
    });
}

function decodeFlow(replyId, f) {
  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const idx = new Uint16Array(bytes(f.idx).buffer);
  const val = bytes(f.val);
  const k = f.k;
  const L = f.n_layers;
  const hq = bytes(f.heads);
  const Hm = f.steps.length ? hq.length / (f.steps.length * L) : 0;
  return {
    replyId,
    live: false,
    steps: f.steps,
    k,
    scale: f.scale,
    headScale: f.head_scale,
    idx: (s, l) => idx.subarray((s * L + l) * k, (s * L + l + 1) * k),
    val: (s, l) => val.subarray((s * L + l) * k, (s * L + l + 1) * k),
    heads: (s, l) => hq.subarray((s * L + l) * Hm, (s * L + l) * Hm + f.n_heads[l]),
  };
}

function setData(data) {
  V.data = data;
  V.rowToStep = new Map((data?.steps ?? []).map((st, i) => [st.row, i]));
  V.lookCache.clear();
  V.lookKey = null;
  V.guessCache.clear();
  V.wordsKey = null;
  V.lensRows = null;
  V.playing = false;
  for (const Ly of V.layers) Ly.shown = -2; // force a refresh
  deselect();
  V.tl.rebuild();
  const n = data?.steps.length ?? 0;
  V.pos = n ? n * V.L - 1 : 0; // show the finished reply
  applyPos(false);
  if (data && !data.live && data.replyId != null) loadLens(data.replyId);
}

// the timeline's mini-bars: how sure each layer already was of each word (logit lens)
function loadLens(replyId) {
  const apply = (g) => {
    if (V.data?.replyId !== replyId || !g?.rows) return;
    V.lensRows = new Map(g.rows.map((r, j) => [r, g.probs[j]]));
    V.tl.bars();
  };
  const have = state.lens.get(replyId);
  if (have) return apply(have);
  api
    .lens(replyId)
    .then((g) => {
      state.lens.set(replyId, g); // the 2D Lens mode shares this cache
      apply(g);
    })
    .catch(() => {}); // a backend without the lens: the chips just have no bars
}

function showEmpty(text) {
  V.empty.textContent = text || "";
  V.empty.hidden = !text;
}

// live: one step per streamed word
function startLive() {
  const prevScale = V.data?.scale ?? null;
  const prevHeadScale = V.data?.headScale ?? null;
  const steps = [];
  const idx = [];
  const val = [];
  const hv = [];
  V.liveHeadRef = null;
  V.liveRef = prevScale ? prevScale.map(() => 0) : null;
  V.data = {
    replyId: null,
    after: V.data?.replyId ?? S.lastReply()?.id ?? null, // the reply this one follows
    live: true,
    steps,
    k: 0,
    scale: prevScale,
    headScale: prevHeadScale,
    idx: (s, l) => idx[s][l],
    val: (s, l) => val[s][l],
    heads: (s, l) => hv[s][l],
    _idx: idx,
    _val: val,
    _heads: hv,
  };
  V.rowToStep = new Map();
  V.lookKey = null;
  V.wordsKey = null;
  V.lensRows = null;
  for (const Ly of V.layers) Ly.shown = -2;
  V.playing = false;
  deselect();
  V.tl.rebuild();
  V.pos = 0;
  applyPos(false);
  showEmpty(null);
}

function addLiveStep(ev) {
  const d = V.data;
  if (!d?.live) return;
  const L = V.L;
  const fi = ev.flow.idx;
  const fv = ev.flow.val;
  const k = fi[0]?.length ?? 0;
  d.k = k;
  // scale each layer like the replay does: against a high but robust value
  // (the 4th strongest so far), so one huge neuron doesn't darken the rest
  if (!V.liveRef) V.liveRef = new Array(L).fill(0);
  const idxs = [];
  const vals = [];
  for (let l = 0; l < L; l++) {
    const ref4 = Math.abs(fv[l][Math.min(3, k - 1)] ?? 0);
    V.liveRef[l] = Math.max(V.liveRef[l], ref4, 1e-6);
    const ref = d.scale ? Math.max(d.scale[l], 1e-6) : V.liveRef[l];
    idxs.push(Uint16Array.from(fi[l]));
    vals.push(Uint8Array.from(fv[l], (x) => Math.min(255, Math.round((Math.abs(x) / ref) * 255))));
  }
  d._idx.push(idxs);
  d._val.push(vals);
  const fh = ev.flow.heads || [];
  if (!V.liveHeadRef) V.liveHeadRef = new Array(L).fill(1e-6);
  d._heads.push(
    Array.from({ length: L }, (_, l) => {
      const hs = (fh[l] || []).map((x) => Math.abs(x ?? 0));
      V.liveHeadRef[l] = Math.max(V.liveHeadRef[l], ...hs, 1e-6);
      const ref = d.headScale ? Math.max(d.headScale[l], 1e-6) : V.liveHeadRef[l];
      return Uint8Array.from(hs, (x) => Math.min(255, Math.round((x / ref) * 255)));
    }),
  );
  const prev = d.steps[d.steps.length - 1];
  d.steps.push({ row: null, reads: prev ? prev.says : null, says: ev.text, prob: ev.prob, end: null });
  V.tl.append(d.steps.length - 1);
  V.pos = d.steps.length * L - 1;
  applyPos(true);
}

function endLive() {
  // the store re-renders once the finished reply is in; sync() then swaps in its full data
  if (V.data?.live) V.data.live = false;
}

// --- the playhead ----------------------------------------------------------------------------

function nSteps() {
  return V.data?.steps.length ?? 0;
}

function cur() {
  const p = Math.max(0, Math.floor(V.pos + 1e-6));
  return { t: Math.floor(p / V.L), l: p % V.L, frac: V.pos - p };
}

function setPos(p, animate) {
  const max = Math.max(0, nSteps() * V.L - 1);
  V.pos = Math.min(max, Math.max(0, p));
  applyPos(animate);
}

// Layers up to the playhead show the current word; those above still hold the
// word before it (dim), the way the model's state really is mid-pass.
function applyPos(animate) {
  const n = nSteps();
  const { t, l } = cur();
  for (const Ly of V.layers) {
    const want = !n ? -1 : Ly.l <= l ? t : t - 1;
    Ly.dimTarget = !n || Ly.l <= l ? 1 : STALE;
    Ly.isCur = n > 0 && Ly.l === l;
    if (Ly.shown !== want) setLayer(Ly, want, animate);
  }
  V.tl.update();
}

function setLayer(Ly, s, animate) {
  Ly.shown = s;
  const now = V.uni.uTime.value;
  Ly.act.fill(0);
  Ly.glow.fill(0);
  Ly.headAct.fill(0);
  Ly.nLines = 0;
  const d = V.data;
  if (s >= 0 && d) {
    // afterglow: what fired on the last two words, fading fast
    if (V.layout.embers) {
      for (let b = EMBER.length; b >= 1; b--) {
        const p = s - b;
        if (p < 0) continue;
        const idx = d.idx(p, Ly.l);
        const val = d.val(p, Ly.l);
        for (let m = 0; m < idx.length; m++) Ly.glow[idx[m]] = Math.max(Ly.glow[idx[m]], (val[m] / 255) * EMBER[b - 1]);
      }
    }
    const idx = d.idx(s, Ly.l);
    const val = d.val(s, Ly.l);
    for (let m = 0; m < idx.length; m++) {
      Ly.act[idx[m]] = Math.max(val[m] / 255, 0.02);
      Ly.birth[idx[m]] = animate ? now : -100;
    }
    // columns and lines for the strongest, thicker for harder firing
    for (let m = 0; m < idx.length && Ly.nLines < LINES; m++) {
      const a = val[m] / 255;
      if (a < 0.08) break;
      Ly.lineCell[Ly.nLines] = idx[m];
      Ly.lineStr[Ly.nLines] = a;
      Ly.lineBirth[Ly.nLines] = animate ? now + Ly.nLines * 0.006 : -100;
      Ly.nLines++;
    }
    const hv = d.heads ? d.heads(s, Ly.l) : null;
    for (let k = 0; k < Ly.nH; k++) Ly.headAct[k] = hv ? (hv[k] ?? 0) / 255 : 0;
    Ly.arrived = animate ? now : -100;
  }
  uploadLayer(Ly);
}

function uploadLayer(Ly) {
  for (const a of Object.values(Ly.at)) a.needsUpdate = true;
  Ly.la.aCell.array.set(Ly.lineCell);
  Ly.la.aStr.array.set(Ly.lineStr);
  Ly.la.aBirth.array.set(Ly.lineBirth);
  for (const a of Object.values(Ly.la)) a.needsUpdate = true;
  Ly.lgeo.instanceCount = Ly.nLines;
  uploadColumns(Ly);
  uploadHeads(Ly);
  Ly.pu.uArrived.value = Ly.arrived;
}
function uploadColumns(Ly) {
  Ly.ca.aCell.array.set(Ly.lineCell);
  Ly.ca.aStr.array.set(Ly.lineStr);
  Ly.ca.aBirth.array.set(Ly.lineBirth);
  for (let i = 0; i < Ly.nLines; i++) Ly.ca.aEd.array[i] = Ly.edit[Ly.lineCell[i]];
  for (const a of Object.values(Ly.ca)) a.needsUpdate = true;
  Ly.cgeo.instanceCount = Ly.nLines;
}
function uploadHeads(Ly) {
  if (!Ly.nH) return;
  Ly.ha.aA.array.set(Ly.headAct.subarray(0, Ly.nH));
  Ly.ha.aEd.array.set(Ly.headEdit.subarray(0, Ly.nH));
  Ly.ha.aA.needsUpdate = Ly.ha.aEd.needsUpdate = true;
  Ly.hmat.uniforms.uArrived.value = Ly.arrived;
}

function togglePlay() {
  if (!nSteps() || V.data?.live) return;
  if (!V.playing && V.pos >= nSteps() * V.L - 1) setPos(0, true); // from the top
  V.playing = !V.playing;
  V.tl.update();
}

function step(layers) {
  if (!nSteps() || V.data?.live) return;
  V.playing = false;
  setPos(Math.round(V.pos) + layers, true);
}

// --- edits -----------------------------------------------------------------------------------

function syncEdits() {
  const sig = S.editSignature(state.edits);
  if (sig === V.editSig) return;
  const flash = V.editSig !== null; // flash what changed, but not on the first look
  V.editSig = sig;
  const before = flash ? V.layers.map((Ly) => ({ edit: Ly.edit.slice(), head: Ly.headEdit.slice(), layer: Ly.layerEdit ? editCode(Ly.layerEdit) : 0 })) : null;
  for (const Ly of V.layers) {
    Ly.edit.fill(0);
    Ly.headEdit.fill(0);
    Ly.layerEdit = null;
  }
  V.editMarks = [];
  for (const e of state.edits) {
    const t = e.target;
    const Ly = V.layers[t.layer];
    if (!Ly) continue;
    if (t.kind === "neuron" && t.index < V.N) {
      Ly.edit[t.index] = editCode(e);
    } else if (t.kind === "head" && t.index < Ly.nH) Ly.headEdit[t.index] = editCode(e);
    else if (t.kind === "layer") Ly.layerEdit = e;
  }
  for (const Ly of V.layers) for (let i = 0; i < V.N; i++) if (Ly.edit[i]) V.editMarks.push({ l: Ly.l, n: i });
  if (flash) flashChanges(before);
  for (const Ly of V.layers) {
    Ly.at.ed.needsUpdate = Ly.at.edAt.needsUpdate = true;
    uploadColumns(Ly);
    uploadHeads(Ly);
  }
  if (V.sep !== null) renderChip();
}

// An edit just changed: its neurons flash and swell, and a ripple spreads from the spot.
function flashChanges(before) {
  const now = V.uni.uTime.value;
  let n = 0;
  for (const Ly of V.layers) {
    const b = before[Ly.l];
    for (let i = 0; i < V.N; i++) {
      if (Ly.edit[i] === b.edit[i]) continue;
      Ly.editAt[i] = now;
      if (n++ < 10) spawnRipple(Ly.l, i, null, Ly.edit[i] || b.edit[i], false);
    }
    for (let k = 0; k < Ly.nH; k++) {
      if (Ly.headEdit[k] === b.head[k]) continue;
      if (n++ < 10) spawnRipple(Ly.l, -1, k, Ly.headEdit[k] || b.head[k], false);
    }
    const lc = Ly.layerEdit ? editCode(Ly.layerEdit) : 0;
    if (lc !== b.layer) {
      Ly.layerEditAt = now;
      if (n++ < 14) spawnRipple(Ly.l, -1, null, lc || b.layer, true);
    }
  }
}

function spawnRipple(l, n, head, code, big) {
  if (!V.uni.uMotion.value) return;
  V.ripples.push({ l, n, head, code, big, born: V.uni.uTime.value });
  if (V.ripples.length > 24) V.ripples.shift();
}

// --- layout: the view panel -------------------------------------------------------------------

function metrics() {
  const M = V.M;
  const sp = BASE.sp * V.layout.sp;
  const gap = BASE.gap * V.layout.gap;
  const W = COLS * sp;
  const D = V.ROWS * sp;
  const pad = sp * 1.4 + 0.02;
  const strip = sp * 4.5 + 0.04;
  Object.assign(M, { sp, gap, W, D, pad, strip, xMin: -W / 2 - pad, xMax: W / 2 + pad, zMin: -D / 2 - strip, zMax: D / 2 + pad, spinePad: 0.3 + gap * 0.6 });
  M.y0 = (-(V.L - 1) / 2) * gap - M.spinePad;
  M.y1 = ((V.L - 1) / 2) * gap + M.spinePad;
  V.uni.uSp.value = sp;
  V.uni.uGap.value = gap;
  V.uni.uSize.value = V.layout.size;
  V.uni.uGridHalf.value.set(W / 2, D / 2);
  V.uni.uPlateMin.value.set(M.xMin, M.zMin);
  V.uni.uPlateMax.value.set(M.xMax, M.zMax);
  V.uni.uStrip.value = strip;
}
const cellX = (n) => ((n % COLS) - (COLS - 1) / 2) * V.M.sp;
const cellZ = (n) => (Math.floor(n / COLS) - (V.ROWS - 1) / 2) * V.M.sp;
const headX = (Ly, k) => (k - (Ly.nH - 1) / 2) * (V.M.W / Math.max(1, Ly.nH));
const headZ = () => -V.M.D / 2 - V.M.sp * 0.9 - V.M.strip * 0.42;

// Place everything for the current layout (layer gap, neuron spacing, size).
function relayout(fit = true) {
  metrics();
  for (const Ly of V.layers) {
    Ly.home.pos.set(0, (Ly.l - (V.L - 1) / 2) * V.M.gap, 0);
    if (V.sep !== Ly.l) Ly.target.pos.copy(Ly.home.pos);
  }
  V.beaconFor = null;
  if (fit && !V.userMoved && V.sep === null && !V.ride) fitCamera(true); // keep the tower fitted while you spread it
}

const ICONS = {
  prevWord: '<path fill="currentColor" d="M2 3h2v10H2zM14 3 6 8l8 5z"/>',
  prevLayer: '<path fill="currentColor" d="M11 3 4 8l7 5z"/>',
  play: '<path fill="currentColor" d="M5 2.5l8.5 5.5L5 13.5z"/>',
  pause: '<path fill="currentColor" d="M4 3h3v10H4zM9 3h3v10H9z"/>',
  nextLayer: '<path fill="currentColor" d="M5 3l7 5-7 5z"/>',
  nextWord: '<path fill="currentColor" d="M12 3h2v10h-2zM2 3l8 5-8 5z"/>',
  within: '<g fill="currentColor"><rect x="1" y="7" width="2" height="2" rx=".5"/><rect x="5" y="7" width="2" height="2" rx=".5"/><rect x="9" y="7" width="2" height="2" rx=".5"/><rect x="13" y="7" width="2" height="2" rx=".5"/></g><path d="M1 12.5h14M1 11v3M15 11v3" stroke="currentColor" fill="none"/>',
  between: '<g fill="none" stroke="currentColor"><path d="M2 3.5h9M2 8h9M2 12.5h9"/><path d="M14 2v12M12.5 3.5 14 2l1.5 1.5M12.5 12.5 14 14l1.5-1.5"/></g>',
  size: '<g fill="currentColor"><rect x="1" y="9" width="3" height="3" rx=".5"/><rect x="7" y="4" width="8" height="8" rx="1"/></g>',
  fit: '<path d="M1.5 5V1.5H5M11 1.5h3.5V5M14.5 11v3.5H11M5 14.5H1.5V11" fill="none" stroke="currentColor" stroke-width="1.5"/><rect x="5.5" y="4.5" width="5" height="7" fill="currentColor" opacity=".55"/>',
  full: '<path d="M6 1.5H1.5V6M10 1.5h4.5V6M14.5 10v4.5H10M6 14.5H1.5V10M2 2l4 4M14 2l-4 4M14 14l-4-4M2 14l4-4" fill="none" stroke="currentColor" stroke-width="1.4"/>',
  unfull: '<path d="M1.5 6H6V1.5M14.5 6H10V1.5M10 14.5V10h4.5M1.5 10H6v4.5" fill="none" stroke="currentColor" stroke-width="1.4"/>',
  reset: '<path d="M3 8a5 5 0 1 0 1.6-3.7" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M2 1.8v4h4z" fill="currentColor"/>',
  close: '<path d="M3.5 3.5l9 9M12.5 3.5l-9 9" stroke="currentColor" stroke-width="1.5"/>',
  view: '<g fill="none" stroke="currentColor" stroke-width="1.3"><path d="M2 4h12M2 8h12M2 12h12"/></g><g fill="currentColor"><rect x="9" y="2.5" width="2.5" height="3" rx=".5"/><rect x="4" y="6.5" width="2.5" height="3" rx=".5"/><rect x="10.5" y="10.5" width="2.5" height="3" rx=".5"/></g>',
};
function icon(name) {
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("class", "s3d-ico");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = ICONS[name];
  return svg;
}

// A slider drawn as a strip of cells that light up like neurons (no browser range input).
// Arrows step, Shift or PageUp/PageDown take big steps, Home/End go to the ends, and
// Enter or a double-click returns to the default.
function slider({ id, label, iconName, min, max, step = 0.01, value, def, fmtV, title, onInput, onCommit }) {
  const out = h("output", { class: "s3d-slider-out", id: `${id}-out` });
  const cells = h("span", { class: "s3d-slider-cells" });
  for (let i = 0; i < 20; i++) {
    const c = h("span", {});
    c.style.setProperty("--i", (i / 19).toFixed(3));
    cells.append(c);
  }
  const track = h(
    "div",
    { class: "s3d-slider-track", id, role: "slider", tabindex: "0", "aria-labelledby": `${id}-label`, "aria-valuemin": min, "aria-valuemax": max, title },
    cells,
    h("span", { class: "s3d-slider-notch" }),
    h("span", { class: "s3d-slider-thumb" }),
  );
  const el = h("div", { class: "s3d-slider" }, h("div", { class: "s3d-slider-head" }, icon(iconName), h("span", { class: "s3d-slider-label", id: `${id}-label` }, label), out), track);
  let v = value;
  const toP = (x) => (x - min) / (max - min);
  el.style.setProperty("--d", toP(def).toFixed(4));
  const render = () => {
    const p = toP(v);
    el.style.setProperty("--p", p.toFixed(4));
    track.setAttribute("aria-valuenow", String(v));
    track.setAttribute("aria-valuetext", fmtV(v));
    out.textContent = fmtV(v);
    [...cells.children].forEach((c, i) => c.classList.toggle("on", i / 19 <= p + 1e-6));
  };
  const set = (x, emit) => {
    x = Math.min(max, Math.max(min, Math.round((x - min) / step) * step + min));
    x = Number(x.toFixed(4));
    if (x === v) return;
    v = x;
    render();
    if (emit) onInput(v);
  };
  const fromX = (cx) => {
    const r = track.getBoundingClientRect();
    return min + Math.min(1, Math.max(0, (cx - r.left) / r.width)) * (max - min);
  };
  let dragging = false;
  track.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    track.setPointerCapture(e.pointerId);
    track.focus({ preventScroll: true });
    set(fromX(e.clientX), true);
    e.preventDefault();
  });
  track.addEventListener("pointermove", (e) => dragging && set(fromX(e.clientX), true));
  const end = () => {
    if (!dragging) return;
    dragging = false;
    onCommit?.(v);
  };
  track.addEventListener("pointerup", end);
  track.addEventListener("pointercancel", end);
  track.addEventListener("dblclick", () => {
    set(def, true);
    onCommit?.(v);
  });
  track.addEventListener("keydown", (e) => {
    const big = (max - min) / 10;
    let nv = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") nv = v + (e.shiftKey ? big : step);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") nv = v - (e.shiftKey ? big : step);
    else if (e.key === "PageUp") nv = v + big;
    else if (e.key === "PageDown") nv = v - big;
    else if (e.key === "Home") nv = min;
    else if (e.key === "End") nv = max;
    else if (e.key === "Enter") nv = def;
    if (nv === null) return;
    e.preventDefault();
    e.stopPropagation(); // the arrows step the timeline everywhere else
    set(nv, true);
    onCommit?.(v);
  });
  render();
  return { el, track, set: (x) => set(x, false) };
}

function switchButton({ id, label, value, title, onChange }) {
  const b = h(
    "button",
    { type: "button", role: "switch", class: "s3d-switch", id, "aria-checked": String(value), title },
    h("span", {}, label),
    h("span", { class: "s3d-switch-track", "aria-hidden": "true" }, h("span", { class: "s3d-switch-knob" })),
  );
  b.addEventListener("click", () => {
    const on = b.getAttribute("aria-checked") !== "true";
    b.setAttribute("aria-checked", String(on));
    onChange(on);
  });
  return b;
}

// view controls: spread the neurons (within a layer) or the layers, neuron size, ride
// with the word, afterglow, fit, full window, reset. Open by default; they fold away.
function buildViewPanel() {
  const pct = (v) => `${Math.round(v * 100)}%`;
  const row = (key, label, iconName, title) =>
    slider({
      id: `s3d-${key}`,
      label,
      iconName,
      min: RANGE[key][0],
      max: RANGE[key][1],
      value: V.layout[key],
      def: LAYOUT[key],
      fmtV: pct,
      title,
      onInput: (v) => {
        V.layout[key] = v;
        relayout();
      },
      onCommit: saveLayout,
    });
  V.sliders = {
    sp: row("sp", "Within a layer", "within", "Horizontal: spread the neurons on each sheet apart. Double-click for the default."),
    gap: row("gap", "Between layers", "between", "Vertical: spread the layers apart, or stack them tight. Double-click for the default."),
    size: row("size", "Neuron size", "size", "How big each neuron is drawn. Double-click for the default."),
  };
  V.rideSw = switchButton({ id: "s3d-ride", label: "Ride with the word", value: false, title: "The camera rises with the word, closer in", onChange: (on) => setRide(on) });
  const emberSw = switchButton({
    id: "s3d-embers",
    label: "Afterglow",
    value: V.layout.embers,
    title: "Neurons that fired on the last two words keep a fading glow",
    onChange: (on) => {
      V.layout.embers = on;
      saveLayout();
      for (const Ly of V.layers) Ly.shown = -2;
      applyPos(false);
    },
  });
  const act = (id, label, iconName, title, fn) => h("button", { type: "button", id, title, onclick: fn }, icon(iconName), h("span", {}, label));
  V.fullBtn = act("s3d-full", "Full window", "full", "The 3D view fills the window (F)", () => setFull(!V.full));
  const panel = h(
    "section",
    { class: "s3d-viewpanel s3d-card", id: "s3d-viewpanel", "aria-label": "View controls" },
    h("div", { class: "s3d-vhead" }, h("h3", {}, "View"), h("button", { type: "button", class: "s3d-iconbtn", id: "s3d-view-hide", "aria-label": "Hide view controls", title: "Hide view controls", onclick: () => setViewOpen(false, true) }, icon("close"))),
    V.sliders.sp.el,
    V.sliders.gap.el,
    V.sliders.size.el,
    h("div", { class: "s3d-vtoggles" }, V.rideSw, emberSw),
    h(
      "div",
      { class: "s3d-vactions" },
      act("s3d-fit", "Fit", "fit", "The whole tower at 90% of the view", fitToScreen),
      V.fullBtn,
      act("s3d-reset", "Reset", "reset", "Default spacing and size, then fit", () => {
        V.layout = { ...LAYOUT, open: V.layout.open, embers: V.layout.embers };
        saveLayout();
        for (const k of Object.keys(V.sliders)) V.sliders[k].set(LAYOUT[k]);
        relayout();
        fitToScreen();
      }),
    ),
  );
  const openBtn = h("button", { type: "button", class: "s3d-viewbtn s3d-card", id: "s3d-view-show", "aria-controls": "s3d-viewpanel", "aria-expanded": "false", onclick: () => setViewOpen(true, true) }, icon("view"), "View");
  V.viewEl.replaceChildren(panel, openBtn);
  V.viewPanel = panel;
  V.viewBtn = openBtn;
  setViewOpen(V.layout.open, false);
}

function setViewOpen(on, user) {
  V.layout.open = on;
  V.viewPanel.hidden = !on;
  V.viewBtn.hidden = on;
  V.viewBtn.setAttribute("aria-expanded", String(on));
  if (!user) return;
  saveLayout();
  (on ? V.sliders.sp.track : V.viewBtn).focus({ preventScroll: true });
  if (!V.userMoved && V.sep === null && !V.ride) fitCamera(true); // the panel's room changed
}

function fitToScreen() {
  putBack();
  V.savedView = null;
  V.userMoved = false;
  if (V.ride) {
    V.ride = false;
    V.rideSw.setAttribute("aria-checked", "false");
  }
  fitCamera(true);
}

// Ride with the word: the camera comes in close and rises with the word.
function setRide(on) {
  V.ride = on;
  V.rideSw.setAttribute("aria-checked", String(on));
  if (on) {
    putBack();
    if (!V.fitView) fitCamera(false);
    const d = V.fitView.dist * 0.42;
    const tgt = new THREE.Vector3(V.fitView.target.x * 0.3, V.beadY, V.fitView.target.z * 0.3);
    const dir = V.camera.position.clone().sub(V.controls.target).normalize();
    V.camAnim = { pos: tgt.clone().addScaledVector(dir, d), target: tgt };
    if (!V.uni.uMotion.value) jumpCamera();
  } else {
    V.userMoved = false;
    fitCamera(true);
  }
}
function jumpCamera() {
  if (!V.camAnim) return;
  V.camera.position.copy(V.camAnim.pos);
  V.controls.target.copy(V.camAnim.target);
  V.camAnim = null;
  V.controls.update();
}

// Put the camera where the whole tower (sheets, heads, the word's line) fills FIT of the
// stage, keeping a three-quarter view from above, with room on the left for the view
// controls (when open) and on the right for the layers' column.
const HOME_DIR = new THREE.Vector3(1, 0.72, 1.12).normalize();
const fitBox = new THREE.Box3();
const fitCorner = new THREE.Vector3();
function fitCamera(animate) {
  const w = V.stage.clientWidth;
  const hgt = V.stage.clientHeight;
  if (!w || !hgt) return;
  const M = V.M;
  const cam = V.camera;
  const keep = { pos: cam.position.clone(), quat: cam.quaternion.clone() };
  fitBox.min.set(M.xMin, M.y0, M.zMin);
  fitBox.max.set(M.xMax, M.y1 + M.gap * 0.6, M.zMax);
  const centre = fitBox.getCenter(new THREE.Vector3());
  const viewRoom = V.layout.open && w > 700 ? (V.viewPanel.offsetWidth || 232) + 18 : 0;
  const labelRoom = Math.min(w * 0.3, (V.lblW || 160) + 30);
  let roomL = (2 * viewRoom) / w;
  let roomR = (2 * labelRoom) / w;
  const total = roomL + roomR;
  if (total > 0.9) {
    roomL *= 0.9 / total;
    roomR *= 0.9 / total;
  }
  const bounds = () => {
    cam.updateMatrixWorld();
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < 8; i++) {
      fitCorner.set(i & 1 ? fitBox.max.x : fitBox.min.x, i & 2 ? fitBox.max.y : fitBox.min.y, i & 4 ? fitBox.max.z : fitBox.min.z).project(cam);
      x0 = Math.min(x0, fitCorner.x); x1 = Math.max(x1, fitCorner.x); y0 = Math.min(y0, fitCorner.y); y1 = Math.max(y1, fitCorner.y);
    }
    return { x0, x1, y0, y1 };
  };
  const aim = (target, d) => {
    cam.position.copy(target).addScaledVector(HOME_DIR, d);
    cam.lookAt(target);
  };
  let lo = 0.3, hi = 80;
  for (let i = 0; i < 34; i++) {
    const d = (lo + hi) / 2;
    aim(centre, d);
    const b = bounds();
    if (b.x1 - b.x0 <= FIT * (2 - roomL - roomR) && b.y1 - b.y0 <= 2 * FIT) hi = d;
    else lo = d;
  }
  // pan so the box's middle lands in the middle of the space left over
  const target = centre.clone();
  const t = Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2);
  const R = new THREE.Vector3();
  const U = new THREE.Vector3();
  for (let i = 0; i < 3; i++) {
    aim(target, hi);
    const b = bounds();
    R.setFromMatrixColumn(cam.matrixWorld, 0);
    U.setFromMatrixColumn(cam.matrixWorld, 1);
    const dx = (b.x0 + b.x1) / 2 - (roomL - roomR) / 2;
    const dy = (b.y0 + b.y1) / 2;
    target.addScaledVector(R, dx * hi * t * cam.aspect).addScaledVector(U, dy * hi * t);
  }
  aim(target, hi);
  const pos = cam.position.clone();
  cam.position.copy(keep.pos);
  cam.quaternion.copy(keep.quat);
  cam.updateMatrixWorld();
  V.fitView = { pos, target, dist: hi };
  V.camAnim = { pos: pos.clone(), target: target.clone() };
  if (!animate || !V.uni.uMotion.value) jumpCamera();
}

// --- hovering an edit in the rail: where is it? -------------------------------------------------

const bp = new THREE.Vector3();
function updateBeacon(t) {
  const e = S.hover.edit;
  const tg = e?.target;
  const Ly = tg && tg.layer != null ? V.layers[tg.layer] : null;
  const ok = Ly && V.sep === null && (tg.kind === "neuron" || tg.kind === "head" || tg.kind === "layer") && (tg.kind !== "head" || tg.index < Ly.nH) && (tg.kind !== "neuron" || tg.index < V.N);
  V.beaconFor = ok ? { e, Ly } : null;
  if (!ok) {
    V.beacon.visible = false;
    V.beaconLabel.hidden = true;
    return;
  }
  const code = editCode(e);
  const css = editPalette(V.colours)[code] || V.colours.ink1;
  if (tg.kind === "neuron") bp.set(cellX(tg.index), 0, cellZ(tg.index));
  else if (tg.kind === "head") bp.set(headX(Ly, tg.index), 0.02, headZ());
  else bp.set(0, 0, 0);
  Ly.g.localToWorld(bp);
  const top = Math.max(V.M.y1 + 0.6, bp.y + 0.6);
  V.beacon.position.copy(bp);
  const r = 1.2 / cssPxPerUnit(bp);
  V.beacon.scale.set(r, top - bp.y, r);
  V.beacon.material.color.copy(colour(css));
  V.beacon.material.opacity = 0.75 + 0.25 * Math.sin(t * 6);
  V.beacon.visible = true;
  V.beaconLabel.hidden = false;
  V.beaconLabel.textContent = `${e.label ?? ""}`.trim() || "edit";
  V.beaconLabel.style.setProperty("--edit", css);
  place(V.beaconLabel, new THREE.Vector3(bp.x, top, bp.z), 36);
}

// --- word probabilities: the candidates, and each layer's guess (logit lens) ---------------------

function wantGuess(row) {
  const d = V.data;
  if (row == null || !d || d.live || d.replyId == null) return;
  if (V.guessCache.has(row) || V.guessPending.has(row)) return;
  const replyId = d.replyId;
  V.guessPending.add(row);
  api
    .guess(replyId, row)
    .then((g) => {
      if (V.data?.replyId !== replyId) return;
      V.guessCache.set(row, g);
      V.wordsKey = null;
    })
    .catch(() => {})
    .finally(() => V.guessPending.delete(row));
}

function updateWords() {
  const d = V.data;
  const n = nSteps();
  const { t, l } = cur();
  const st = n ? d.steps[t] : null;
  // fetch this step's guesses, and the next step's while playing
  if (st && !d.live) {
    wantGuess(st.row);
    if (V.playing && d.steps[t + 1]) wantGuess(d.steps[t + 1].row);
  }
  const g = st ? V.guessCache.get(st.row) : null;
  const key = `${d?.replyId}:${n}:${t}:${l}:${g ? 1 : 0}:${V.sep}`;
  if (key === V.wordsKey) return;
  V.wordsKey = key;
  for (const Ly of V.layers) {
    const reached = g && (Ly.l <= l || V.sep === Ly.l);
    const lg = reached ? g.layers[Ly.l] : null;
    const top = lg?.top?.[0];
    if (top) fill(Ly.lblG, h("span", { class: "gw" }, quoteTok(top.text)), h("span", { class: "gp" }, pct(top.prob)));
    else Ly.lblG.replaceChildren();
    Ly.lblBar.style.width = `${Math.round((lg?.p_next ?? 0) * 100)}%`;
    Ly.lblBar.parentNode.style.visibility = lg ? "" : "hidden";
    Ly.label.classList.toggle("is-right", !!(top && g.next != null && top.text === g.next));
    Ly.label.title = lg && g.next != null ? `Layer ${Ly.l} (${Ly.kind}) gave ${quoteTok(g.next)} ${pct(lg.p_next ?? 0)}` : `Layer ${Ly.l} (${Ly.kind})`;
    Ly.lblK.textContent = Ly.kind === "attention" ? (V.stage.clientWidth < 900 ? "attn" : "attention") : "";
  }
  const lblW = Math.max(120, ...V.layers.map((Ly) => (Ly.label.hidden ? 0 : Ly.label.offsetWidth)));
  const refit = Math.abs(lblW - (V.lblW || 160)) > 24;
  V.lblW = lblW;
  if (refit && !V.userMoved && V.sep === null && !V.ride) fitCamera(true); // the column got wider or narrower
  if (V.sep !== null && !V.chip.contains(document.activeElement)) renderChip();
  // the candidates card
  if (!st) {
    V.cand.hidden = true;
    return;
  }
  V.cand.hidden = false;
  const alts = st.alts?.length ? st.alts : st.says != null ? [{ text: st.says, prob: st.prob ?? 0 }] : [];
  V.cand.replaceChildren(
    h("p", { class: "s3d-cand-head" }, h("strong", {}, "Next word"), st.reads == null ? "after your message" : h("span", {}, "after ", h("span", { class: "mono" }, quoteTok(st.reads)))),
    alts.length
      ? h(
          "ol",
          { class: "s3d-cand-list" },
          alts.map((a) =>
            h(
              "li",
              { class: a.text === st.says ? "is-chosen" : null },
              h("span", { class: "s3d-cand-word" }, quoteTok(a.text)),
              h("span", { class: "s3d-cand-bar" }, h("i", { style: { width: `${Math.max(1, Math.round(a.prob * 100))}%` } })),
              h("span", { class: "num" }, pct(a.prob)),
            ),
          ),
        )
      : h("p", { class: "s3d-cand-end" }, st.end === "stop" ? "It stops here." : "Cut off here."),
  );
  V.cand.title = "The model's own probabilities for the next word. Beside each layer: its best guess so far, and a bar for how sure it already was of the word that came.";
  V.ioIn.replaceChildren(h("small", {}, "reads"), st.reads == null ? "your message" : quoteTok(st.reads));
  fill(V.ioOut, h("small", {}, "says"), sayText(st), st.says != null && modelProb(st) != null ? h("span", { class: "p" }, pct(modelProb(st))) : null);
}
const pct = (p) => (p >= 0.995 ? "100%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);
function sayText(st) {
  if (st.says == null) return st.end === "stop" ? "(stops)" : "…";
  return quoteTok(st.says);
}
// The model's own probability for the word it wrote (the candidates list),
// not the sampler's after temperature, top-k and top-p.
function modelProb(st) {
  const a = st.alts?.find((x) => x.text === st.says);
  return a ? a.prob : st.prob;
}

// --- frame loop ------------------------------------------------------------------------------

function frame() {
  if (!V.root.isConnected) {
    V.renderer.setAnimationLoop(null);
    V.running = false;
    V.playing = false;
    if (V.full) {
      // you went somewhere else (e.g. "Open in 2D"): come back to the normal layout
      V.full = false;
      V.root.classList.remove("is-full");
      V.fullBtn.replaceChildren(icon("full"), h("span", {}, "Full window"));
    }
    return;
  }
  const now = performance.now() / 1000;
  const dt = Math.min(0.05, now - V.last);
  V.last = now;
  const motion = !!V.uni.uMotion.value;
  V.uni.uTime.value += dt;
  syncEdits();

  if (V.playing) {
    const max = nSteps() * V.L - 1;
    V.pos = Math.min(max, V.pos + dt * LAYERS_PER_SECOND * V.speed);
    if (V.pos >= max) V.playing = false;
    applyPos(true);
  }

  const k = motion ? 1 - Math.exp(-dt * 7) : 1;
  const kl = motion ? 1 - Math.exp(-dt * 6) : 1;
  const kf = motion ? 1 - Math.exp(-dt * 6) : 1;
  const { l: curL, frac } = cur();
  for (const Ly of V.layers) {
    Ly.g.position.lerp(Ly.target.pos, k);
    Ly.g.quaternion.slerp(Ly.target.quat, k);
    const want = V.sep === null ? Ly.dimTarget : V.sep === Ly.l ? 1 : 0.16;
    Ly.dim += (want - Ly.dim) * k;
    // columns stand tall where the word is (or where you look) and settle behind it
    let lift;
    if (V.sep !== null) lift = V.sep === Ly.l ? 1 : 0;
    else if (!nSteps() || Ly.l > curL) lift = 0;
    else if (Ly.isCur || V.hover === Ly.l || V.selected?.l === Ly.l) lift = 1;
    else lift = Math.max(0.08, Math.exp(-(curL - Ly.l) / 2.6));
    Ly.lift += (lift - Ly.lift) * kl;
    // lines (and the resting lattice) are bright on the word's layer, a pulled-out or pointed-at one
    const focus = V.sep !== null ? (V.sep === Ly.l ? 1 : 0) : Ly.isCur || V.hover === Ly.l || V.selected?.l === Ly.l ? 1 : 0;
    Ly.lmat.uniforms.uFocus.value += (focus - Ly.lmat.uniforms.uFocus.value) * kf;
    const f = Ly.lmat.uniforms.uFocus.value;
    Ly.pu.uDim.value = Ly.pmat.uniforms.uDim.value = Ly.lmat.uniforms.uDim.value = Ly.cmat.uniforms.uDim.value = Ly.hmat.uniforms.uDim.value = Ly.dim;
    Ly.pu.uFocus.value = Ly.pmat.uniforms.uFocus.value = f;
    Ly.pu.uCur.value = Ly.isCur ? 1 : 0;
    Ly.pu.uHot.value = V.hover === Ly.l || V.sep === Ly.l || V.beaconFor?.Ly === Ly ? 1 : 0;
    Ly.pu.uLE.value = Ly.layerEdit ? editCode(Ly.layerEdit) : 0;
    Ly.pu.uLEAt.value = Ly.layerEditAt;
    Ly.pmat.uniforms.uDust.value = V.sep === Ly.l ? 0.48 : 0.3; // the pulled-out sheet shows its lattice, to aim at
    Ly.cmat.uniforms.uLift.value = Ly.lift;
    Ly.lmat.uniforms.uEndY.value = 0.75 * Ly.lift;
  }

  // the word rides the line at the playhead
  const n = nSteps();
  const M = V.M;
  V.beadY = n ? V.layers[curL].home.pos.y + (curL < V.L - 1 ? frac * M.gap : 0) + 0.002 : M.y0;
  V.spine.position.y = (M.y0 + M.y1) / 2;
  V.spine.scale.y = M.y1 - M.y0;
  V.spineMat.uniforms.uY0.value = M.y0;
  V.spineMat.uniforms.uY1.value = M.y1;
  V.spineMat.uniforms.uProgress.value = n ? (V.beadY - M.y0) / (M.y1 - M.y0) : 0;
  V.bead.position.set(0, V.beadY, 0);
  V.bead.visible = n > 0;

  if (V.ride && V.sep === null && !V.camAnim) {
    const dy = (V.beadY + M.gap * 1.5 - V.controls.target.y) * (motion ? 1 - Math.exp(-dt * 7) : 1);
    V.controls.target.y += dy;
    V.camera.position.y += dy;
  }
  if (V.camAnim) {
    V.camera.position.lerp(V.camAnim.pos, k);
    V.controls.target.lerp(V.camAnim.target, k);
    if (V.camera.position.distanceTo(V.camAnim.pos) < 0.003) V.camAnim = null;
  }
  V.controls.update();
  const dist = V.camera.position.distanceTo(V.controls.target);
  const R = (M.y1 - M.y0) / 2 + M.W * 0.6;
  V.uni.uFogNear.value = Math.max(0.1, dist - R * 0.5);
  V.uni.uFogFar.value = dist + R * 1.6;
  V.world.updateMatrixWorld();
  V.camera.updateMatrixWorld();
  updateHover();
  updateWiring();
  updateBeacon(V.uni.uTime.value);
  updateMarks();
  updateWords();
  V.renderer.render(V.scene, V.camera);
  placeLabels();
  V.tl.head();
  drawLook();
}

const tmp = new THREE.Vector3();
function project(p) {
  tmp.copy(p).project(V.camera);
  return { x: (tmp.x * 0.5 + 0.5) * V.stage.clientWidth, y: (-tmp.y * 0.5 + 0.5) * V.stage.clientHeight, ok: tmp.z < 1 && Math.abs(tmp.x) < 1.3 && Math.abs(tmp.y) < 1.3 };
}
function place(el, p, minTop = null) {
  const s = project(p);
  if (!s.ok) {
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.style.left = `${s.x}px`;
  el.style.top = `${minTop == null ? s.y : Math.max(minTop, s.y)}px`;
}
const _w = new THREE.Vector3();
function cssPxPerUnit(worldPos) {
  const d = _w.copy(worldPos).applyMatrix4(V.camera.matrixWorldInverse).z;
  return V.uni.uScale.value / V.uni.uDpr.value / Math.max(0.01, -d);
}

// The layers' column: one aligned label per layer right of the tower, with a leader back
// to its sheet. They thin out when the layers sit tight, and step aside for a pulled-out sheet.
const anchorPt = new THREE.Vector3();
function placeLabels() {
  const W = V.stage.clientWidth;
  const Hh = V.stage.clientHeight;
  const M = V.M;
  const anchors = V.layers.map((Ly) => project(anchorPt.set(M.xMax, 0, M.zMin).applyMatrix4(Ly.g.matrixWorld)));
  let maxX = -Infinity;
  anchors.forEach((p) => p.ok && (maxX = Math.max(maxX, p.x)));
  const want = Math.min(maxX + 22, W - (V.lblW || 160) - 10);
  V.colX = V.colX ? V.colX + (want - V.colX) * 0.25 : want;
  const { l: curL } = cur();
  const spacing = Math.abs(anchors[V.L - 1].y - anchors[0].y) / Math.max(1, V.L - 1);
  const every = spacing >= 13 ? 1 : spacing >= 7 ? 2 : 4;
  const sepMode = V.sep !== null;
  // always label the word's layer, the hovered one, the top one and whole-layer edits; when
  // the labels thin out, the regular ones near those step aside so nothing overlaps
  const special = new Set([V.hover, nSteps() ? curL : null, ...V.layers.filter((Ly) => Ly.layerEdit).map((Ly) => Ly.l)]);
  const nearSpecial = (l) => every > 1 && [...special].some((s) => s != null && s !== l && Math.abs(s - l) < every);
  if (!nearSpecial(V.L - 1)) special.add(V.L - 1); // the top layer too, unless a closer one needs the room
  for (const Ly of V.layers) {
    const p = anchors[Ly.l];
    const el = Ly.label;
    const show = p.ok && !sepMode && (special.has(Ly.l) || (Ly.l % every === 0 && !nearSpecial(Ly.l)));
    el.hidden = !show;
    if (!show) continue;
    el.style.transform = `translate(${Math.round(V.colX)}px, ${Math.round(p.y - 8)}px)`;
    el.style.setProperty("--lead", `${Math.max(0, V.colX - p.x - 6)}px`);
    el.classList.toggle("is-cur", Ly.l === curL && nSteps() > 0);
    el.classList.toggle("is-hot", V.hover === Ly.l);
    if (Ly.layerEdit) {
      Ly.lblLE.hidden = false;
      Ly.lblLE.style.setProperty("--edit", `var(${EDIT_VARS[editCode(Ly.layerEdit)]})`);
    } else Ly.lblLE.hidden = true;
  }
  // reads (below the stack) and says (above it)
  const st = V.data?.steps[cur().t];
  V.ioIn.hidden = V.ioOut.hidden = !st || sepMode;
  if (!st || sepMode) return;
  const a = project(anchorPt.set(0, M.y0 - 0.06, 0));
  const b = project(anchorPt.set(0, M.y1 + M.gap * 0.5, 0));
  V.ioIn.style.transform = `translate(${Math.round(a.x)}px, ${Math.round(Math.min(Hh - 16, Math.max(20, a.y + 10)))}px) translate(-50%, -50%)`;
  V.ioOut.style.transform = `translate(${Math.round(b.x)}px, ${Math.round(Math.min(Hh - 16, Math.max(16, b.y - 12)))}px) translate(-50%, -50%)`;
}

// --- marks: your edits, the selection, hover, wiring targets, ripples -------------------------

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _p = new THREE.Vector3();
function updateMarks() {
  const mk = V.marks;
  const A = V.markAttr;
  const t = V.uni.uTime.value;
  const M = V.M;
  let i = 0;
  const put = (Ly, x, z, size, code, at, kind, born, dim) => {
    if (i >= MAX_MARKS) return;
    _p.set(x, 0.004, z);
    _m.compose(_p, _q.identity(), _s.set(size, 1, size));
    _m.premultiply(Ly.g.matrixWorld);
    mk.setMatrixAt(i, _m);
    A.aCode.array[i] = code;
    A.aAt.array[i] = at;
    A.aKind.array[i] = kind;
    A.aBorn.array[i] = born;
    A.aDim.array[i] = dim;
    i++;
  };
  // a ring a little bigger than the column, at least ~17px so it reads from anywhere
  const ringSize = (Ly, x, z, scale) => {
    _p.set(x, 0, z).applyMatrix4(Ly.g.matrixWorld);
    const ppu = cssPxPerUnit(_p);
    const px = Math.min(34, Math.max(17, M.sp * ppu * 3)) * scale;
    return Math.max(px / ppu, M.sp * 2.3) * Math.sqrt(V.layout.size);
  };
  for (const em of V.editMarks) {
    const Ly = V.layers[em.l];
    const x = cellX(em.n), z = cellZ(em.n);
    const at = Ly.editAt[em.n];
    const fl = V.uni.uMotion.value && t >= at ? Math.exp(-(t - at) * 3.2) : 0;
    const dim = V.sep !== null && V.sep !== em.l ? 0.35 : Math.max(0.85, Ly.dim);
    put(Ly, x, z, ringSize(Ly, x, z, 1 + 0.9 * fl), Ly.edit[em.n], at, 0, 0, dim);
  }
  const sel = V.selected;
  if (sel) {
    const Ly = V.layers[sel.l];
    put(Ly, cellX(sel.n), cellZ(sel.n), ringSize(Ly, cellX(sel.n), cellZ(sel.n), 1.45), 0, -100, 1, 0, 1);
  }
  if (V.selHead) {
    const Ly = V.layers[V.selHead.l];
    put(Ly, headX(Ly, V.selHead.h), headZ(), (M.W / Math.max(1, Ly.nH)) * 1.25, 0, -100, 1, 0, 1);
  }
  if (V.sep !== null && V.hoverNeuron !== null && (!sel || sel.n !== V.hoverNeuron)) {
    put(V.layers[V.sep], cellX(V.hoverNeuron), cellZ(V.hoverNeuron), M.sp * 2.2, 0, -100, 3, 0, 1);
  }
  if (V.wiring) {
    for (const tg of V.wiring.targets) {
      const Ly = V.layers[tg.layer];
      if (!Ly) continue;
      put(Ly, cellX(tg.neuron), cellZ(tg.neuron), ringSize(Ly, cellX(tg.neuron), cellZ(tg.neuron), 0.7), 0, -100, 3, 0, 1);
    }
  }
  if (V.beaconFor && V.beaconFor.e.target.kind !== "layer") {
    const { e, Ly } = V.beaconFor;
    const tg = e.target;
    const x = tg.kind === "neuron" ? cellX(tg.index) : headX(Ly, tg.index);
    const z = tg.kind === "neuron" ? cellZ(tg.index) : headZ();
    put(Ly, x, z, ringSize(Ly, x, z, 1.6 + 0.3 * Math.sin(t * 6)), editCode(e), -100, 4, 0, 1);
  }
  for (let j = V.ripples.length - 1; j >= 0; j--) {
    const r = V.ripples[j];
    const u = (t - r.born) / 0.9;
    if (u >= 1) {
      V.ripples.splice(j, 1);
      continue;
    }
    const e = 1 - (1 - u) * (1 - u);
    const Ly = V.layers[r.l];
    const x = r.n >= 0 ? cellX(r.n) : r.head != null ? headX(Ly, r.head) : 0;
    const z = r.n >= 0 ? cellZ(r.n) : r.head != null ? headZ() : 0;
    const size = r.big ? M.W * (0.2 + 1.1 * e) : M.sp * (2 + 14 * e);
    put(Ly, x, z, size, r.code, 0, 2, r.born, 1);
  }
  mk.count = i;
  mk.instanceMatrix.needsUpdate = true;
  for (const a of Object.values(A)) a.needsUpdate = true;
}

// --- pointer: hover, pull a layer out, pick a neuron -----------------------------------------

const ray = new THREE.Raycaster();
const _inv = new THREE.Matrix4();
const _o = new THREE.Vector3();
const _d = new THREE.Vector3();
// Where the pointer meets the sheets: the nearest hit, in that sheet's own coordinates.
function hitSheets(onlySep) {
  ray.setFromCamera(V.pointer, V.camera);
  const M = V.M;
  let best = null;
  for (const Ly of V.layers) {
    if (onlySep && Ly.l !== V.sep) continue;
    _inv.copy(Ly.g.matrixWorld).invert();
    _o.copy(ray.ray.origin).applyMatrix4(_inv);
    _d.copy(ray.ray.direction).transformDirection(_inv);
    if (Math.abs(_d.y) < 1e-6) continue;
    const t = -_o.y / _d.y;
    if (t <= 0) continue;
    const x = _o.x + _d.x * t;
    const z = _o.z + _d.z * t;
    if (x < M.xMin || x > M.xMax || z < M.zMin || z > M.zMax) continue;
    if (!best || t < best.t) best = { l: Ly.l, x, z, t };
  }
  return best;
}
function neuronAt(x, z) {
  const c = Math.round(x / V.M.sp + (COLS - 1) / 2);
  const r = Math.round(z / V.M.sp + (V.ROWS - 1) / 2);
  if (c < 0 || c >= COLS || r < 0 || r >= V.ROWS) return null;
  const n = r * COLS + c;
  return n < V.N ? n : null;
}
function headAt(Ly, x, z) {
  if (!Ly.nH || z > -V.M.D / 2 - V.M.sp * 0.9 || z < V.M.zMin) return null;
  const k = Math.floor((x + V.M.W / 2) / (V.M.W / Ly.nH));
  return k >= 0 && k < Ly.nH ? k : null;
}

function wirePointer() {
  const c = V.canvas;
  const toNdc = (e) => {
    const r = c.getBoundingClientRect();
    V.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    V.pxy = [e.clientX - r.left, e.clientY - r.top];
  };
  c.addEventListener("pointermove", (e) => {
    toNdc(e);
    V.pointerIn = true;
    // a drag orbits or pans: from then on the camera stays where you put it
    if (e.buttons && V.down && Math.hypot(e.clientX - V.down[0], e.clientY - V.down[1]) > 5) {
      V.userMoved = true;
      V.hint.classList.add("is-gone");
    }
  });
  c.addEventListener(
    "wheel",
    () => {
      V.userMoved = true;
      V.camAnim = null;
      V.hint.classList.add("is-gone");
    },
    { passive: true },
  );
  c.addEventListener("pointerleave", () => {
    V.pointerIn = false;
  });
  c.addEventListener("pointerdown", (e) => {
    V.down = [e.clientX, e.clientY];
  });
  c.addEventListener("pointerup", (e) => {
    if (!V.down || Math.hypot(e.clientX - V.down[0], e.clientY - V.down[1]) > 5) {
      V.down = null;
      return;
    }
    V.down = null;
    toNdc(e);
    if (V.sep !== null) {
      const hit = hitSheets(true);
      if (hit) {
        const Ly = V.layers[V.sep];
        const k = headAt(Ly, hit.x, hit.z);
        if (k !== null) {
          openHead(V.sep, k, e.clientX, e.clientY);
          return;
        }
        const n = neuronAt(hit.x, hit.z);
        if (n !== null) openNeuron(V.sep, n, e.clientX, e.clientY);
        return;
      }
    }
    const hit = hitSheets(false);
    if (hit) {
      if (hit.l !== V.sep) pullOut(hit.l);
      return;
    }
    putBack();
  });
}

function updateHover() {
  V.hover = V.hoverNeuron = V.hoverHead = null;
  if (!V.pointerIn || V.down) {
    V.tip.hidden = true;
    return;
  }
  let text = null;
  if (V.sep !== null) {
    const hit = hitSheets(true);
    if (hit) {
      const Ly = V.layers[V.sep];
      const k = headAt(Ly, hit.x, hit.z);
      if (k !== null) {
        V.hoverHead = k;
        text = `L${V.sep} head ${k} · ${Ly.kind} · ${Math.round(Ly.headAct[k] * 100)}% of its usual size`;
      } else {
        const n = neuronAt(hit.x, hit.z);
        if (n !== null) {
          V.hoverNeuron = n;
          const a = Ly.act[n];
          text = `L${V.sep} N${n}${a > 0 ? ` · ${Math.round(a * 100)}% of its layer's range` : " · quiet on this word"}`;
        }
      }
    }
  }
  if (V.hoverNeuron === null && V.hoverHead === null) {
    const hit = hitSheets(false);
    if (hit && hit.l !== V.sep) {
      V.hover = hit.l;
      text = `Layer ${hit.l} · ${V.layers[hit.l].kind} · click to pull it out`;
    }
  }
  V.canvas.style.cursor = V.hover !== null || V.hoverNeuron !== null || V.hoverHead !== null ? "pointer" : "";
  V.tip.hidden = !text;
  if (text) {
    V.tip.textContent = text;
    const x = Math.min(V.pxy[0] + 14, V.stage.clientWidth - V.tip.offsetWidth - 8);
    V.tip.style.transform = `translate(${Math.max(8, x)}px, ${V.pxy[1] + 16}px)`;
  }
}

function pullOut(l) {
  if (V.sep !== null) {
    const old = V.layers[V.sep];
    old.target.pos.copy(old.home.pos);
    old.target.quat.copy(old.home.quat);
  }
  if (!V.savedView) V.savedView = { pos: V.camera.position.clone(), target: V.controls.target.clone() };
  V.sep = l;
  V.playing = false;
  V.tl.update();
  deselect();
  closePopover();
  V.camera.updateMatrixWorld();
  const R = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 0);
  const U = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 1);
  const B = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 2); // towards the viewer
  const M = V.M;
  const Ly = V.layers[l];
  const pos = Ly.home.pos.clone().multiplyScalar(0.4).addScaledVector(R, M.W * 1.25).addScaledVector(B, M.W * 0.7);
  // face the viewer like the 2D grid: neuron 0 top left, rows going down
  const quat = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(R, B, U.clone().negate()));
  Ly.target.pos.copy(pos);
  Ly.target.quat.copy(quat);
  V.stage.classList.add("is-sep");
  // close enough that the sheet (heads included) fills most of the stage, between the chip
  // at the top and the next-word card, which moves to the bottom
  const t = Math.tan(THREE.MathUtils.degToRad(V.camera.fov) / 2);
  const wpx = V.stage.clientWidth || 800;
  const hpx = V.stage.clientHeight || 600;
  const topRoom = 60;
  const bottomRoom = V.cand.hidden ? 16 : 84;
  const avail = Math.max(160, hpx - topRoom - bottomRoom);
  const sheetW = M.xMax - M.xMin;
  const sheetH = M.zMax - M.zMin;
  const d = Math.max(sheetH / (2 * t * 0.96 * (avail / hpx)), (sheetW * hpx) / (2 * t * 0.92 * Math.max(240, wpx - 24)));
  const worldPerPx = (2 * d * t) / hpx;
  const mid = pos.clone().addScaledVector(U, -(M.zMin + M.zMax) / 2); // the sheet's middle, heads included
  const look = mid.addScaledVector(U, ((topRoom - bottomRoom) / 2) * worldPerPx);
  V.camAnim = { pos: look.clone().addScaledVector(B, d), target: look };
  if (!V.uni.uMotion.value) jumpCamera();
  renderChip();
}

function renderChip() {
  const l = V.sep;
  if (l === null) return;
  const Ly = V.layers[l];
  const st = V.data?.steps[Ly.shown];
  const g = st ? V.guessCache.get(st.row) : null;
  const top = g?.layers[l]?.top?.[0];
  const le = Ly.layerEdit;
  V.chip.hidden = false;
  fill(V.chip,
    h("strong", {}, `Layer ${l}`),
    h("span", { class: "meta", title: "Click a neuron or a head to edit it" }, `${Ly.kind}${le ? " · edited" : ""}`),
    top ? h("span", { class: "meta" }, "guesses ", h("span", { class: "mono" }, `${quoteTok(top.text)} ${pct(top.prob)}`)) : null,
    h("a", { class: "btn btn-quiet", href: `#/layer/${l}` }, "Open in 2D"),
    h("button", { class: "btn btn-quiet", type: "button", onclick: () => (putBack(), V.canvas.focus({ preventScroll: true })) }, "Put back", h("kbd", {}, "Esc")),
  );
}

function putBack() {
  if (V.sep === null) return;
  const Ly = V.layers[V.sep];
  Ly.target.pos.copy(Ly.home.pos);
  Ly.target.quat.copy(Ly.home.quat);
  V.sep = null;
  deselect();
  V.chip.hidden = true;
  V.stage.classList.remove("is-sep");
  if (V.ride) setRide(true);
  else if (!V.userMoved) fitCamera(true); // the stage may have changed size meanwhile
  else if (V.savedView) V.camAnim = V.savedView;
  V.savedView = null;
}

function deselect() {
  if (!V) return;
  V.selected = null;
  V.selHead = null;
  V.wiring = null;
  V.tl.colourBy(null);
  V.lookKey = null; // recolour the timeline for attention, if it applies
}

function popAnchor(x, y) {
  const anchor = h("div", { class: "s3d-anchor", "data-pop-anchor": "" });
  anchor.style.left = `${x}px`;
  anchor.style.top = `${y}px`;
  document.body.append(anchor);
  return anchor;
}

function openHead(l, k, x, y) {
  closePopover();
  deselect();
  const Ly = V.layers[l];
  V.selHead = { l, h: k };
  const anchor = popAnchor(x, y);
  const st = V.data?.steps[Ly.shown];
  const attn = Ly.kind === "attention";
  openEditPopover(anchor, {
    target: { kind: "head", layer: l, index: k },
    title: `Head ${k}`,
    sub: `layer ${l}, ${attn ? "attention" : "DeltaNet"}`,
    info:
      `${st ? `Reading ${st.reads == null ? "your message" : quoteTok(st.reads)}, it wrote ${Math.round(Ly.headAct[k] * 100)}% of its usual size. ` : ""}` +
      "Scaling a head scales everything it writes into the residual stream." +
      (attn
        ? " The lines down to the timeline show where it looked."
        : " DeltaNet heads keep a running memory instead of looking back at single words, so there is no “where it looked” to draw."),
    note: `${attn ? "Attention" : "DeltaNet"} head.`,
    onClose: () => {
      anchor.remove();
      if (V.selHead && V.selHead.l === l && V.selHead.h === k) {
        deselect();
        V.escAt = performance.now();
      }
    },
  });
}

// jump to a neuron in another layer (from the wiring list)
function jumpTo(l, n) {
  closePopover();
  pullOut(l);
  setTimeout(
    () => {
      if (V.sep !== l) return;
      const p = project(new THREE.Vector3(cellX(n), 0, cellZ(n)).applyMatrix4(V.layers[l].g.matrixWorld));
      const r = V.canvas.getBoundingClientRect();
      openNeuron(l, n, r.left + p.x, r.top + p.y);
    },
    V.uni.uMotion.value ? 750 : 30,
  );
}

// the wiring follows the sheets while they move
const wa = new THREE.Vector3();
const wb = new THREE.Vector3();
function updateWiring() {
  const w = V.wiring;
  if (!w) {
    V.wire.geometry.instanceCount = 0;
    return;
  }
  wa.set(cellX(w.n), 0, cellZ(w.n));
  V.layers[w.l].g.localToWorld(wa);
  const A = V.wattr;
  w.targets.forEach((t, i) => {
    wb.set(cellX(t.neuron), 0, cellZ(t.neuron));
    V.layers[t.layer].g.localToWorld(wb);
    A.aStart.array.set([wa.x, wa.y, wa.z], i * 3);
    A.aEnd.array.set([wb.x, wb.y, wb.z], i * 3);
    A.aStrength.array[i] = t.strength;
    A.aSign.array[i] = t.sign;
  });
  A.aStart.needsUpdate = A.aEnd.needsUpdate = A.aStrength.needsUpdate = A.aSign.needsUpdate = true;
  V.wire.geometry.instanceCount = w.targets.length;
}

function wiringList(targets) {
  const top = [...targets].sort((a, b) => b.strength - a.strength).slice(0, 6);
  if (!top.length) return null;
  return h(
    "div",
    { class: "s3d-wiring" },
    h("span", { class: "meta" }, "Feeds most, from the weights (click to go there)"),
    h(
      "div",
      { class: "s3d-wiring-list" },
      top.map((t) =>
        h(
          "button",
          { class: "btn btn-quiet", type: "button", onclick: () => jumpTo(t.layer, t.neuron), title: `Weight ${fmt(t.weight, 3)}` },
          `L${t.layer} N${t.neuron}`,
          h("span", { class: "meta" }, t.sign > 0 ? " boosts" : " suppresses"),
        ),
      ),
    ),
  );
}

async function openNeuron(l, n, x, y) {
  const reply = S.lastReply();
  const d = V.data;
  closePopover();
  deselect();
  V.selected = { l, n };
  const anchor = popAnchor(x, y);

  const wiringP = api.wiring(l, n, WIRE_PER_LAYER).catch(() => null);
  let info = null;
  if (reply && d && !d.live && d.replyId === reply.id) info = await api.neuron(reply.id, l, n).catch(() => null);
  const wiring = await wiringP;
  if (!V.selected || V.selected.l !== l || V.selected.n !== n) {
    anchor.remove();
    return; // clicked something else meanwhile
  }
  const { t } = cur();
  const st = d?.steps[t];
  const tr = info?.trace ?? null;
  const here = tr && st?.row != null ? tr[st.row] : null;
  let peak = 0;
  let peakStep = null;
  if (tr) {
    d.steps.forEach((s, i) => {
      const v = tr[s.row] ?? 0;
      if (Math.abs(v) > Math.abs(peak)) {
        peak = v;
        peakStep = i;
      }
    });
    V.tl.colourBy(d.steps.map((s) => Math.abs(tr[s.row] ?? 0)));
  }
  // wiring: flip the sign if the neuron fired negative here, so "boosts" means what happened
  const fired = Math.sign(here || peak) || 1;
  const targets = (wiring?.targets ?? []).map((t) => ({ ...t, sign: Math.sign(t.weight * fired) || 1 }));
  const maxW = Math.max(1e-9, ...targets.map((t) => Math.abs(t.weight)));
  for (const t of targets) t.strength = Math.abs(t.weight) / maxW;
  V.wiring = targets.length ? { l, n, targets } : null;
  openEditPopover(anchor, {
    target: { kind: "neuron", layer: l, index: n },
    title: `Neuron ${n}`,
    sub: `layer ${l}`,
    info:
      here != null
        ? `${fmt(here)} while reading ${st.reads == null ? "your message" : quoteTok(st.reads)}. Strongest: ${fmt(peak)}${peakStep != null ? ` while reading ${quoteTok(d.steps[peakStep].reads ?? "")}` : ""}. The timeline below is lit where it fired.`
        : "Activity shows once the reply has finished.",
    // saved with the edit, as the reason in the edits rail
    note: peakStep != null ? `Fired ${fmt(peak)} on ${quoteTok(d.steps[peakStep].reads ?? "")}.` : null,
    says: info ? Promise.resolve(info.says) : null,
    sign: Math.sign(here || peak) || 1,
    extra: wiringList(targets),
    onClose: () => {
      anchor.remove();
      if (V.selected && V.selected.l === l && V.selected.n === n) {
        deselect();
        V.escAt = performance.now();
      }
    },
  });
}

// --- attention: where a layer (or one of its heads) looked ---------------------------------------

function lookTarget() {
  const d = V.data;
  if (!d || d.live || d.replyId == null || V.playing || V.selected) return null;
  const l = V.sep ?? cur().l;
  const Ly = V.layers[l];
  if (Ly.kind !== "attention" || Ly.shown < 0) return null;
  const st = d.steps[Ly.shown];
  if (!st || st.row == null) return null;
  const head = V.selHead && V.selHead.l === l ? V.selHead.h : null;
  return { l, row: st.row, head };
}

function lookWeights(lt, data) {
  const layer = data.layers[String(lt.l)];
  if (!layer) return null;
  const hs = lt.head != null ? [lt.head] : layer.rows.map((_, k) => k);
  const per = new Float32Array(V.data.steps.length);
  let msg = 0;
  for (const k of hs) {
    msg += (layer.before[k] ?? 0) / hs.length;
    layer.rows[k].forEach((w, r) => {
      const s = V.rowToStep.get(r);
      if (s === undefined) msg += (w ?? 0) / hs.length;
      else per[s] += (w ?? 0) / hs.length;
    });
  }
  return { per, msg };
}

const lp = new THREE.Vector3();
function drawLook() {
  const lt = lookTarget();
  if (!lt) {
    V.lookKey = null;
    if (V.lookShown) {
      V.lookShown = false;
      V.lookTargets = [];
      for (const p of V.lookPaths) p.removeAttribute("d");
      V.lookCap.hidden = true;
      if (!V.selected) V.tl.colourBy(null);
    }
    return;
  }
  const data = V.lookCache.get(lt.row);
  if (!data) {
    if (!V.lookPending.has(lt.row)) {
      const replyId = V.data.replyId;
      V.lookPending.add(lt.row);
      api
        .look(replyId, lt.row)
        .then((x) => V.data?.replyId === replyId && V.lookCache.set(lt.row, x))
        .catch(() => {})
        .finally(() => V.lookPending.delete(lt.row));
    }
    return;
  }
  const key = `${V.data.replyId}:${lt.row}:${lt.l}:${lt.head}`;
  if (key !== V.lookKey) {
    V.lookKey = key;
    V.lookShown = true;
    const w = lookWeights(lt, data);
    if (!w) return;
    const all = [{ i: -1, w: w.msg }, ...Array.from(w.per, (x, i) => ({ i, w: x }))];
    V.lookTargets = all
      .filter((x) => x.w >= 0.02)
      .sort((a, b) => b.w - a.w)
      .slice(0, LOOK_LINES);
    V.tl.colourBy(Array.from(w.per), w.msg);
    const name = (t) => (t.i < 0 ? "your message" : quoteTok(V.data.steps[t.i].reads ?? ""));
    V.lookCap.hidden = false;
    V.lookCap.replaceChildren(
      h("strong", {}, `Layer ${lt.l}${lt.head != null ? `, head ${lt.head}` : ""} looked at`),
      " ",
      V.lookTargets
        .slice(0, 4)
        .map((t, j) => `${j ? " · " : ""}${name(t)} ${Math.round(t.w * 100)}%`)
        .join(""),
    );
  }
  // geometry every frame: the camera and the timeline both move
  const Ly = V.layers[lt.l];
  if (lt.head != null) lp.set(headX(Ly, lt.head), 0.02, headZ());
  else lp.set(0, 0, 0);
  const s = project(Ly.g.localToWorld(lp));
  const rootR = V.root.getBoundingClientRect();
  const stR = V.stage.getBoundingClientRect();
  const sx = s.x + stR.left - rootR.left;
  const sy = s.y + stR.top - rootR.top;
  const trackR = V.tl.trackRect();
  V.lookPaths.forEach((p, j) => {
    const t = V.lookTargets[j];
    const r = t ? V.tl.targetRect(t.i) : null;
    if (!r) {
      p.removeAttribute("d");
      return;
    }
    let ex = r.left + r.width / 2;
    const off = ex < trackR.left || ex > trackR.right;
    ex = Math.min(trackR.right - 4, Math.max(trackR.left + 4, ex)) - rootR.left;
    const ey = r.top - rootR.top;
    const dy = ey - sy;
    p.setAttribute("d", `M${sx.toFixed(1)},${sy.toFixed(1)} C${sx.toFixed(1)},${(sy + dy * 0.55).toFixed(1)} ${ex.toFixed(1)},${(ey - dy * 0.45).toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`);
    p.setAttribute("stroke-width", (1 + 4 * t.w).toFixed(2));
    p.style.opacity = off ? "0.3" : "";
  });
}

// --- keyboard ------------------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (!V?.root.isConnected) return;
  const tag = e.target?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable) return;
  if (e.target?.closest?.(".popover")) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === " " && tag !== "BUTTON" && tag !== "A") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    if (e.target?.getAttribute?.("role") === "radio") return;
    e.preventDefault();
    const dir = e.key === "ArrowRight" ? 1 : -1;
    step(e.shiftKey ? dir * V.L : dir);
  } else if (e.key === "Escape" && performance.now() - V.escAt > 80) {
    if (V.sep !== null) putBack();
    else if (V.full) setFull(false);
  } else if ((e.key === "f" || e.key === "F") && !e.shiftKey) {
    e.preventDefault();
    setFull(!V.full);
  }
});

// --- the timeline ----------------------------------------------------------------------------------

function buildTimeline(L) {
  const btn = (name, label, fn) => h("button", { class: "s3d-tbtn", type: "button", title: label, "aria-label": label, onclick: fn }, icon(name));
  const play = h("button", { class: "s3d-tbtn s3d-play", type: "button", onclick: () => togglePlay() });
  const readout = h("div", { class: "s3d-readout" });
  // speed: a radio group, arrows move within it
  const speed = h("div", { class: "s3d-seg", role: "radiogroup", "aria-label": "Playback speed" });
  const speedBtns = SPEEDS.map(([v, label]) => {
    const b = h("button", { type: "button", role: "radio", "aria-label": `${label}× speed`, title: `${label}× speed`, "aria-checked": String(v === 1), tabindex: v === 1 ? "0" : "-1" }, label);
    b.addEventListener("click", () => pickSpeed(v));
    speed.append(b);
    return b;
  });
  function pickSpeed(v, focus) {
    if (V) V.speed = v;
    SPEEDS.forEach(([s], i) => {
      speedBtns[i].setAttribute("aria-checked", String(s === v));
      speedBtns[i].tabIndex = s === v ? 0 : -1;
      if (s === v && focus) speedBtns[i].focus();
    });
  }
  speed.addEventListener("keydown", (e) => {
    const i = SPEEDS.findIndex(([s]) => s === (V?.speed ?? 1));
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") j = Math.min(SPEEDS.length - 1, i + 1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") j = Math.max(0, i - 1);
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = SPEEDS.length - 1;
    if (j === null) return;
    e.preventDefault();
    e.stopPropagation();
    pickSpeed(SPEEDS[j][0], true);
  });
  const steps = h("div", { class: "s3d-steps" });
  const msg = h("div", { class: "s3d-msg", title: "Your message, and anything earlier in the chat" }, h("span", { class: "s3d-word" }, "your message"));
  const headEl = h("div", { class: "s3d-head", "aria-hidden": "true" });
  const track = h("div", { class: "s3d-track", tabindex: "0", role: "slider", "aria-label": "Timeline: words and layers", title: "Click or drag to move through the reply" }, msg, steps, headEl);
  const el = h(
    "div",
    { class: "s3d-tl" },
    h(
      "div",
      { class: "s3d-bar" },
      h(
        "div",
        { class: "s3d-transport", role: "group", "aria-label": "Playback" },
        btn("prevWord", "Previous word (Shift ←)", () => step(-V.L)),
        btn("prevLayer", "Previous layer (←)", () => step(-1)),
        play,
        btn("nextLayer", "Next layer (→)", () => step(1)),
        btn("nextWord", "Next word (Shift →)", () => step(V.L)),
      ),
      readout,
      h("div", { class: "s3d-speedbox" }, h("span", {}, "Speed"), speed),
    ),
    track,
  );

  let boxes = []; // [left, width] per step, cached
  let revealNext = true;
  let lastT = -1;
  let lastPlaying = null;
  let lastKey = "";
  const stepEls = () => steps.children;

  function stepEl(i) {
    const st = V.data.steps[i];
    const word = st.says == null ? (st.end === "stop" ? "end" : "…") : showToken(st.says).replace(/^␣/, "") || "␣";
    // under each word: how sure each layer already was of it (logit lens), one bar per layer
    const sky = document.createElementNS(svgNS, "svg");
    sky.setAttribute("class", "s3d-sky");
    sky.setAttribute("viewBox", `0 0 ${L * 2} 11`);
    sky.setAttribute("preserveAspectRatio", "none");
    sky.setAttribute("aria-hidden", "true");
    for (let l = 0; l < L; l++) {
      const r = document.createElementNS(svgNS, "rect");
      r.setAttribute("x", l * 2);
      r.setAttribute("y", 10);
      r.setAttribute("width", 1.4);
      r.setAttribute("height", 1);
      sky.append(r);
    }
    const said = st.says == null ? "end" : quoteTok(st.says);
    return h(
      "div",
      { class: "s3d-step", dataset: { i }, title: `${st.reads == null ? "your message" : quoteTok(st.reads)} → ${said}. The bars: how sure each layer already was of ${said} (logit lens).` },
      h("span", { class: "s3d-fill" }),
      h("span", { class: "s3d-word" }, word),
      sky,
    );
  }
  function paintBars(i) {
    const c = stepEls()[i];
    const st = V.data?.steps[i];
    if (!c || !st) return;
    const probs = st.row != null ? V.lensRows?.get(st.row + 1) : null;
    const rects = c.lastChild.children;
    for (let l = 0; l < rects.length; l++) {
      const hh = probs ? 1 + Math.round(10 * Math.sqrt(Math.max(0, probs[l] ?? 0))) : 1;
      rects[l].setAttribute("y", 11 - hh);
      rects[l].setAttribute("height", hh);
    }
  }
  function measure() {
    boxes = Array.from(stepEls(), (c) => [c.offsetLeft, c.offsetWidth]);
  }
  function rebuild() {
    steps.replaceChildren(...(V.data ? V.data.steps.map((_, i) => stepEl(i)) : []));
    lastT = -1;
    lastKey = "";
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", String(Math.max(0, nSteps() * V.L - 1)));
    requestAnimationFrame(measure);
  }
  function append(i) {
    steps.append(stepEl(i));
    const c = steps.lastElementChild;
    boxes[i] = [c.offsetLeft, c.offsetWidth];
    track.setAttribute("aria-valuemax", String(Math.max(0, nSteps() * V.L - 1)));
  }
  function bars() {
    for (let i = 0; i < stepEls().length; i++) paintBars(i);
  }
  function update() {
    const n = nSteps();
    const { t, l } = cur();
    const live = V.data?.live;
    const key = `${n}:${t}:${l}:${V.playing}:${live}`;
    if (key === lastKey) return;
    lastKey = key;
    if (V.playing !== lastPlaying) {
      lastPlaying = V.playing;
      play.replaceChildren(icon(V.playing ? "pause" : "play"));
      play.setAttribute("aria-label", V.playing ? "Pause (Space)" : "Play (Space)");
      play.title = play.getAttribute("aria-label");
    }
    play.disabled = !n || live;
    for (const b of el.querySelectorAll(".s3d-tbtn:not(.s3d-play)")) b.disabled = !n || live;
    const kids = stepEls();
    if (t !== lastT) {
      for (let i = 0; i < kids.length; i++) {
        kids[i].classList.toggle("is-done", i < t);
        kids[i].classList.toggle("is-current", i === t);
      }
      lastT = t;
    }
    const c = kids[t];
    if (c) {
      c.firstChild.style.width = `${((l + 1) / V.L) * 100}%`;
      const rects = c.lastChild.children;
      for (let j = 0; j < rects.length; j++) rects[j].classList.toggle("lit", j <= l);
    }
    if (!n) {
      readout.textContent = "";
      track.removeAttribute("aria-valuenow");
      return;
    }
    const st = V.data.steps[t];
    track.setAttribute("aria-valuenow", String(Math.round(V.pos)));
    track.setAttribute("aria-valuetext", `Word ${t + 1} of ${n}, layer ${l}`);
    readout.replaceChildren(
      h("strong", {}, live ? `Writing… word ${n}` : `Word ${t + 1} of ${n}`),
      ` · layer ${l} `,
      h("span", { class: "opt" }, `${V.layers[l].kind} `),
      "· reads ",
      h("span", { class: "mono" }, st.reads == null ? "your message" : quoteTok(st.reads)),
      " → ",
      st.says == null ? (st.end === "stop" ? "stops" : "…") : h("span", { class: "mono" }, quoteTok(st.says)),
      h("span", { class: "opt" }, st.says != null && modelProb(st) != null ? ` ${pct(modelProb(st))}` : ""),
    );
  }
  function head() {
    const n = nSteps();
    headEl.hidden = !n;
    if (!n) return;
    const { t, l, frac } = cur();
    if (!boxes[t]) measure();
    const b = boxes[t];
    if (!b) return;
    const x = b[0] + ((l + frac + 0.5) / V.L) * b[1];
    headEl.style.transform = `translateX(${x}px)`;
    // keep the playhead in view while it moves
    if (V.playing || V.data?.live || headEl._x !== x || revealNext) {
      revealNext = false;
      const left = track.scrollLeft;
      const w = track.clientWidth;
      if (x < left + 30 || x > left + w - 30) track.scrollLeft = x - w * 0.4;
    }
    headEl._x = x;
  }
  // light the steps by one neuron's activity, or by attention (null to clear)
  function colourBy(values, msgValue = null) {
    const kids = stepEls();
    const max = values ? Math.max(1e-9, ...values, msgValue ?? 0) : 1;
    const paint = (el, v) => {
      if (v == null) {
        el.style.removeProperty("--fired");
        el.style.removeProperty("--fired-ink");
        el.classList.remove("is-fired");
        return;
      }
      const bin = Math.min(8, Math.round((v / max) * 8));
      el.style.setProperty("--fired", `var(--act-${bin})`);
      el.style.setProperty("--fired-ink", `var(--act-ink-${bin})`);
      el.classList.add("is-fired");
    };
    paint(msg, values && msgValue != null ? msgValue : null);
    for (let i = 0; i < kids.length; i++) paint(kids[i], values ? values[i] : null);
  }

  // click or drag to scrub
  let scrubbing = false;
  const scrub = (e) => {
    if (!nSteps() || V.data?.live) return;
    if (!boxes.length) measure();
    const x = e.clientX - track.getBoundingClientRect().left + track.scrollLeft;
    let i = boxes.findIndex(([left, w]) => x < left + w);
    if (i < 0) i = boxes.length - 1;
    const [left, w] = boxes[i];
    const l = Math.max(0, Math.min(V.L - 1, Math.floor(((x - left) / w) * V.L)));
    V.playing = false;
    setPos(i * V.L + l, false);
  };
  track.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    scrubbing = true;
    track.setPointerCapture(e.pointerId);
    scrub(e);
  });
  track.addEventListener("pointermove", (e) => scrubbing && scrub(e));
  track.addEventListener("pointerup", () => (scrubbing = false));
  track.addEventListener("pointercancel", () => (scrubbing = false));
  track.addEventListener("keydown", (e) => {
    if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      if (!nSteps() || V.data?.live) return;
      V.playing = false;
      setPos(e.key === "Home" ? 0 : nSteps() * V.L - 1, true);
    }
  });
  new ResizeObserver(() => measure()).observe(steps);

  return {
    el,
    rebuild,
    append,
    bars,
    update,
    head,
    colourBy,
    reveal: () => (revealNext = true),
    trackRect: () => track.getBoundingClientRect(),
    targetRect: (i) => (i < 0 ? msg : stepEls()[i])?.getBoundingClientRect() ?? null,
  };
}
