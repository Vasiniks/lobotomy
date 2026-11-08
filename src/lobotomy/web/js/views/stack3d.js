// The 3D layer stack.
//
// Every layer is a sheet of neurons (64 per row), stacked like floors. A line
// carries each word up through the sheets; at every layer, lines reach out to
// the neurons that fired, thicker for harder firing. The timeline plays,
// pauses and steps through the reply: one word is a big step, one layer a
// small step. Click a sheet to pull it out, then a neuron to edit it. While
// the model writes, the sheets light up live, one word at a time.
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

const SP = 0.03; // neuron spacing the sheet geometry is built at (the View sliders rescale it)
const COLS = 64;
// what the View sliders change; remembered per browser. sp spreads the neurons
// within a layer (horizontal), gap spreads the layers (vertical).
const LAYOUT = { gap: 0.16, sp: 0.03, size: 0.044, open: true };
const LAYOUT_KEY = "lobotomy-3d-layout-v2"; // v2: bigger neurons, so older saved sizes don't stick
const FIT = 0.9; // "Fit to screen": the tower fills 90% of the stage
const SPINE_PAD = 0.45; // how far the word's line runs below the first layer and above the last
const LABEL_ROOM = 190; // px kept free right of the tower for the layer labels
function loadLayout() {
  try {
    const v = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null");
    if (v && typeof v === "object") return { ...LAYOUT, ...v };
  } catch {}
  return { ...LAYOUT };
}
function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(V.layout));
  } catch {}
}
const LINES = 48; // lines per layer, at most
const LAYERS_PER_SECOND = 16; // at 1x a word takes 1.5 s to climb 24 layers
const SPEEDS = [
  [0.25, "¼×"],
  [0.5, "½×"],
  [1, "1×"],
  [2, "2×"],
  [4, "4×"],
  [8, "8×"],
];
const STALE = 0.35; // layers the playhead hasn't reached yet still show the previous word, dimmed
const WIRE_PER_LAYER = 4; // a clicked neuron's strongest connections into each later layer
const LOOK_LINES = 8; // attention: lines to the words a layer (or head) looked at most

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

// --- building the scene ----------------------------------------------------------------------

function readColours() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n, fallback = "#888888") => cs.getPropertyValue(n).trim() || fallback;
  return {
    act: Array.from({ length: 9 }, (_, i) => v(`--act-${i}`)),
    down: [1, 2, 3, 4].map((i) => v(`--edit-down-${i}`)),
    up: [1, 2, 3, 4].map((i) => v(`--edit-up-${i}`)),
    noise: v("--ink-3"),
    well: v("--surface-well"),
    rule: v("--rule"),
    ruleStrong: v("--rule-strong"),
    ink1: v("--ink-1"),
    ink2: v("--ink-2"),
  };
}

const colour = (css) => new THREE.Color().setStyle(css);
// raw sRGB bytes, so the shaders reproduce the design's colours exactly
function paletteTexture(list, width) {
  const bytes = new Uint8Array(width * 4);
  list.forEach((css, i) => {
    const n = colour(css).getHex();
    bytes.set([(n >> 16) & 255, (n >> 8) & 255, n & 255, 255], i * 4);
  });
  const tex = new THREE.DataTexture(bytes, width, 1, THREE.RGBAFormat);
  tex.magFilter = tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}
const rawVec3 = (css) => {
  const n = colour(css).getHex();
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};

// the edit palette, indexed by editCode()
function editPalette(c) {
  return [c.ink2, ...c.down, ...c.up, c.down[3], c.noise];
}
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

const POINT_VS = /* glsl */ `
  attribute float act;    // 0..1, how hard it fired against its layer's usual range
  attribute float birth;  // when it lit up, for the pop
  attribute float edit;   // editCode
  attribute float editAt; // when you last changed its edit, for the flash
  uniform float uScale, uSize, uTime, uMotion, uFogNear, uFogFar;
  varying float vAct, vEdit, vFog, vFlash;
  void main() {
    vAct = act;
    vEdit = edit;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z); // the far side of the tower recedes
    float pop = 1.0 + 1.4 * act * exp(-max(0.0, uTime - birth) * 5.0) * uMotion;
    float breathe = 1.0 + 0.14 * act * sin(uTime * 3.5 + position.x * 41.0 + position.z * 29.0) * uMotion;
    float s = (0.6 + 1.5 * act) * pop * breathe;
    vFlash = uTime >= editAt ? exp(-(uTime - editAt) * 2.2) * uMotion : 0.0;
    if (edit > 0.5) s = max(s, 2.1) * (1.0 + 0.12 * sin(uTime * 2.4) * uMotion);
    s *= 1.0 + 2.2 * vFlash;
    gl_PointSize = max(1.0, uSize * s * uScale / -mv.z);
    // A sprite is flat at one depth but its sheet is tilted, so the sheet's own neurons
    // would slice a big disc in half. Draw it a radius closer along the line of sight
    // (same spot on screen) so it sits on top of its sheet.
    float lift = uSize * s * 0.6;
    gl_Position = projectionMatrix * vec4(mv.xyz * (1.0 - lift / length(mv.xyz)), 1.0);
  }`;
const POINT_FS = /* glsl */ `
  uniform sampler2D uRamp, uEdits;
  uniform vec3 uBg;
  uniform float uDim;
  varying float vAct, vEdit, vFog, vFlash;
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    vec3 c = ramp(0.05 + 0.95 * vAct);
    float dim = uDim;
    float fog = vFog * 0.6;
    if (vEdit > 0.5) {
      // your edits are drawn in their edit colour, whole, and stay bright when the layer is dimmed
      c = texture2D(uEdits, vec2((vEdit + 0.5) / 16.0, 0.5)).rgb;
      if (vEdit > 8.5 && vEdit < 9.5 && (r2 < 0.07 || abs(d.x + d.y) < 0.06)) discard; // off: a ring, cut through
      dim = max(dim, 0.9);
      fog *= 0.3;
    }
    if (vFlash > 0.01 && r2 > 0.17) c = mix(c, ramp(1.0), vFlash); // a bright rim as it changes
    gl_FragColor = vec4(mix(mix(uBg, c, dim), uBg, fog), 1.0);
  }`;

// a soft glow around neurons that fire hard, and around your edits
const GLOW_VS = /* glsl */ `
  attribute float act, birth, edit, editAt;
  uniform float uScale, uSize, uTime, uMotion, uFogNear, uFogFar;
  varying float vA, vEdit;
  void main() {
    float flash = uTime >= editAt ? exp(-(uTime - editAt) * 2.2) * uMotion : 0.0;
    float a = edit > 0.5 ? 0.55 + flash : smoothstep(0.4, 1.0, act);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    a *= 1.0 - 0.75 * smoothstep(uFogNear, uFogFar, -mv.z); // the back of the tower glows less
    vA = a;
    vEdit = edit;
    gl_Position = projectionMatrix * mv;
    if (a < 0.01) { gl_PointSize = 0.0; gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    float pop = 1.0 + 0.8 * exp(-max(0.0, uTime - birth) * 4.0) * uMotion;
    gl_PointSize = uSize * (3.2 + 3.0 * a) * pop * (1.0 + 1.5 * flash) * uScale / -mv.z;
  }`;
const GLOW_FS = /* glsl */ `
  uniform sampler2D uRamp, uEdits;
  uniform float uDim, uGlow;
  varying float vA, vEdit;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float f = (1.0 - r) * (1.0 - r);
    vec3 c = vEdit > 0.5 ? texture2D(uEdits, vec2((vEdit + 0.5) / 16.0, 0.5)).rgb : texture2D(uRamp, vec2(8.5 / 16.0, 0.5)).rgb;
    float dim = vEdit > 0.5 ? max(uDim, 0.9) : uDim;
    gl_FragColor = vec4(c, f * min(1.0, vA) * uGlow * dim);
  }`;

const LINE_VS = /* glsl */ `
  attribute vec3 aEnd;
  attribute float aStrength, aBirth;
  uniform float uTime, uGrow, uFogNear, uFogFar, uSize;
  varying float vY, vS, vFog;
  void main() {
    float g = clamp((uTime - aBirth) / uGrow, 0.0, 1.0);
    g = 1.0 - (1.0 - g) * (1.0 - g) * (1.0 - g);
    float len = length(aEnd);
    vec3 ax = aEnd / max(len, 1e-5);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 side = normalize(cross(ax, up));
    float r = mix(0.0012, 0.0115, aStrength);
    // stop at the neuron's rim, so the line doesn't cut into the lit disc
    float reach = max(0.0, len - uSize * (0.3 + 0.75 * aStrength));
    vec3 p = ax * (position.y * reach * g) + side * (position.x * r) + up * (position.z * r);
    vY = position.y * g;
    vS = aStrength;
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    vFog = smoothstep(uFogNear, uFogFar, -mv.z);
    gl_Position = projectionMatrix * mv;
  }`;
const LINE_FS = /* glsl */ `
  uniform sampler2D uRamp;
  uniform vec3 uBg;
  uniform float uTime, uDim, uMotion;
  varying float vY, vS, vFog;
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  void main() {
    vec3 c = ramp(0.3 + 0.62 * vS);
    // signal flowing out from the line to the neuron
    float f = fract(vY * 2.5 - uTime * 1.3);
    float band = smoothstep(0.0, 0.06, f) * (1.0 - smoothstep(0.06, 0.32, f));
    c = mix(c, ramp(1.0), band * (0.35 + 0.5 * vS) * uMotion);
    gl_FragColor = vec4(mix(mix(uBg, c, uDim), uBg, vFog * 0.6), 1.0);
  }`;

const SPINE_VS = /* glsl */ `
  uniform float uH;
  varying float vY;
  void main() {
    vY = position.y / uH + 0.5;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const SPINE_FS = /* glsl */ `
  uniform sampler2D uRamp;
  uniform vec3 uIdle;
  uniform float uProgress, uTime, uMotion;
  varying float vY;
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  void main() {
    if (vY > uProgress) { gl_FragColor = vec4(uIdle, 1.0); return; }
    float f = fract(vY * 9.0 - uTime * 1.6);
    float band = smoothstep(0.0, 0.1, f) * (1.0 - smoothstep(0.1, 0.5, f));
    gl_FragColor = vec4(mix(ramp(0.7), ramp(1.0), band * uMotion), 1.0);
  }`;

// a neuron's wiring: tubes in world space from it to the later neurons it feeds
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
    float r = mix(0.0018, 0.011, aStrength);
    vec3 bow = up2 * sin(3.14159 * position.y) * len * 0.06; // a slight arc
    vec3 p = aStart + ax * (position.y * len) + side * (position.x * r) + up2 * (position.z * r) + bow;
    vY = position.y;
    vS = aStrength;
    vSign = aSign;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const WIRE_FS = /* glsl */ `
  uniform vec3 uBoost, uSuppress;
  uniform float uTime, uMotion;
  varying float vY, vS, vSign;
  void main() {
    // suppressing connections are dashed
    if (vSign < 0.0 && fract(vY * 22.0) < 0.42) discard;
    vec3 c = vSign > 0.0 ? uBoost : uSuppress;
    float f = fract(vY * 3.0 - uTime * 1.1);
    float band = smoothstep(0.0, 0.05, f) * (1.0 - smoothstep(0.05, 0.28, f));
    gl_FragColor = vec4(c * (0.6 + 0.4 * vS + 0.5 * band * uMotion), 1.0);
  }`;

let LINE_GEO = null;
function lineGeometry() {
  return LINE_GEO || (LINE_GEO = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true).translate(0, 0.5, 0));
}

// lines from the middle of a sheet out to its units (used for the heads)
function lineMesh(count, uni) {
  const base = lineGeometry();
  const geo = new THREE.InstancedBufferGeometry();
  geo.index = base.index;
  geo.setAttribute("position", base.getAttribute("position"));
  const aEnd = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const aStrength = new THREE.InstancedBufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
  const aBirth = new THREE.InstancedBufferAttribute(new Float32Array(count).fill(-100), 1).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("aEnd", aEnd);
  geo.setAttribute("aStrength", aStrength);
  geo.setAttribute("aBirth", aBirth);
  geo.instanceCount = 0;
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...uni, uDim: { value: 1 }, uGrow: { value: 0.35 } },
    vertexShader: LINE_VS,
    fragmentShader: LINE_FS,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  return { mesh, geo, aEnd, aStrength, aBirth, mat };
}

function makeCols(c) {
  return { ramp: c.act.map(colour), edit: editPalette(c).map(colour) };
}

function build() {
  const m = state.model;
  const L = m.n_layers;
  const N = m.n_neurons;
  const ROWS = Math.ceil(N / COLS);
  const W = COLS * SP;
  const D = ROWS * SP;
  const colours = readColours();
  const motion = !matchMedia("(prefers-reduced-motion: reduce)").matches;

  // DOM
  const canvas = h("canvas", { class: "s3d-canvas", tabindex: "0", "aria-label": "The layers in 3D. Space plays, arrows step." });
  const labelsEl = h("div", { class: "s3d-labels", "aria-hidden": "true" });
  const tip = h("div", { class: "s3d-tip" });
  const chip = h("div", { class: "s3d-chip" });
  const empty = h("p", { class: "s3d-empty" });
  const ioIn = h("span", { class: "s3d-io" });
  const ioOut = h("span", { class: "s3d-io" });
  tip.hidden = chip.hidden = true;
  labelsEl.append(ioIn, ioOut);
  const stage = h("div", { class: "s3d-stage" }, canvas, labelsEl, chip, tip, empty);
  const tl = buildTimeline();
  const root = h("div", { class: "s3d" }, stage, tl.el);

  // renderer and camera
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);
  const home = { pos: new THREE.Vector3(8.0, 6.3, 6.5), target: new THREE.Vector3(0.75, 0.55, -0.75) };
  camera.position.copy(home.pos);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(home.target);
  controls.enableDamping = true;
  controls.minDistance = 1.5;
  controls.maxDistance = 40;

  // shared uniforms
  const uni = {
    uScale: { value: 1 },
    uTime: { value: 0 },
    uMotion: { value: motion ? 1 : 0 },
    uRamp: { value: paletteTexture(colours.act, 16) },
    uEdits: { value: paletteTexture(editPalette(colours), 16) },
    uBg: { value: rawVec3(colours.well) },
    uSize: { value: LAYOUT.size },
    uFogNear: { value: 8 },
    uFogFar: { value: 20 },
    uGlow: { value: glowStrength(colours) },
  };

  // the neurons' places on a sheet (row-major, like the 2D grid)
  const local = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    local[i * 3] = ((i % COLS) - (COLS - 1) / 2) * SP;
    local[i * 3 + 2] = (Math.floor(i / COLS) - (ROWS - 1) / 2) * SP;
  }
  const localAttr = new THREE.BufferAttribute(local, 3);
  const plateGeo = new THREE.PlaneGeometry(W + 0.08, D + 0.08).rotateX(-Math.PI / 2);
  const edgeGeo = new THREE.EdgesGeometry(plateGeo);
  const lineGeo = lineGeometry();
  const headGeo = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);

  const world = new THREE.Group();
  scene.add(world);
  const layers = [];
  for (let l = 0; l < L; l++) {
    const g = new THREE.Group();
    g.position.set(0, (l - (L - 1) / 2) * LAYOUT.gap, 0);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", localAttr);
    const act = new Float32Array(N);
    const birth = new Float32Array(N).fill(-100);
    const edit = new Float32Array(N);
    const editAt = new Float32Array(N).fill(-100);
    const attrs = {
      act: new THREE.BufferAttribute(act, 1).setUsage(THREE.DynamicDrawUsage),
      birth: new THREE.BufferAttribute(birth, 1).setUsage(THREE.DynamicDrawUsage),
      edit: new THREE.BufferAttribute(edit, 1).setUsage(THREE.DynamicDrawUsage),
      editAt: new THREE.BufferAttribute(editAt, 1).setUsage(THREE.DynamicDrawUsage),
    };
    for (const [k, a] of Object.entries(attrs)) geo.setAttribute(k, a);
    geo.computeBoundingSphere();
    const pmat = new THREE.ShaderMaterial({
      uniforms: { ...uni, uDim: { value: 1 } },
      vertexShader: POINT_VS,
      fragmentShader: POINT_FS,
    });
    const points = new THREE.Points(geo, pmat);
    const gmat = new THREE.ShaderMaterial({
      uniforms: { ...uni, uDim: { value: 1 } },
      vertexShader: GLOW_VS,
      fragmentShader: GLOW_FS,
      transparent: true,
      depthWrite: false,
      depthTest: false, // light, not a surface: it shows through the sheets, softly
    });
    const glow = new THREE.Points(geo, gmat);
    glow.renderOrder = 1;

    const plate = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, side: THREE.DoubleSide }));
    plate.userData.layer = l;
    const attn = m.layers[l]?.type === "attention";
    const baseEdge = colour(attn ? colours.ruleStrong : colours.rule);
    const edge = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: baseEdge.clone(), transparent: true }));
    // a whole-layer edit tints the sheet in its edit colour
    const tint = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }));
    tint.visible = false;

    const lgeo = new THREE.InstancedBufferGeometry();
    lgeo.index = lineGeo.index;
    lgeo.setAttribute("position", lineGeo.getAttribute("position"));
    const aEnd = new THREE.InstancedBufferAttribute(new Float32Array(LINES * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const aStrength = new THREE.InstancedBufferAttribute(new Float32Array(LINES), 1).setUsage(THREE.DynamicDrawUsage);
    const aBirth = new THREE.InstancedBufferAttribute(new Float32Array(LINES).fill(-100), 1).setUsage(THREE.DynamicDrawUsage);
    lgeo.setAttribute("aEnd", aEnd);
    lgeo.setAttribute("aStrength", aStrength);
    lgeo.setAttribute("aBirth", aBirth);
    lgeo.instanceCount = 0;
    const lmat = new THREE.ShaderMaterial({
      uniforms: { ...uni, uDim: { value: 1 }, uGrow: { value: 0.35 } },
      vertexShader: LINE_VS,
      fragmentShader: LINE_FS,
    });
    const lines = new THREE.Mesh(lgeo, lmat);
    lines.frustumCulled = false;

    // heads: a row of tiles along the sheet's far edge, each with its own line
    const nH = m.layers[l]?.n_heads ?? 0;
    const pitch = W / Math.max(1, nH);
    const headPos = Array.from({ length: nH }, (_, k) => new THREE.Vector3((k - (nH - 1) / 2) * pitch, 0, -(D / 2 + 0.12)));
    const heads = new THREE.InstancedMesh(headGeo, new THREE.MeshBasicMaterial({ transparent: true }), Math.max(1, nH));
    heads.count = nH;
    heads.frustumCulled = false;
    heads.userData.layer = l;
    const hl = lineMesh(Math.max(1, nH), uni);

    g.add(points, glow, plate, tint, edge, lines, heads, hl.mesh);
    world.add(g);
    // the label: layer number, then (once the word reaches it) this layer's best
    // guess for the next word, and a bar for how sure it already was of the real one
    const labelBar = h("i", { class: "s3d-lbar" });
    const labelGuess = h("span", { class: "s3d-guess" });
    const label = h("span", { class: "s3d-lbl" }, h("b", {}, attn ? `${l} attn` : `${l}`), labelBar, labelGuess);
    labelsEl.append(label);
    const homePose = { pos: g.position.clone(), quat: g.quaternion.clone() };
    layers.push({
      l,
      g,
      points,
      glow,
      gmat,
      plate,
      tint,
      edge,
      baseEdge,
      lines,
      lgeo,
      act,
      birth,
      edit,
      editAt,
      attrs,
      aEnd,
      aStrength,
      aBirth,
      pmat,
      lmat,
      label,
      labelBar,
      labelGuess,
      home: homePose,
      target: { pos: homePose.pos.clone(), quat: homePose.quat.clone() },
      dim: 1,
      dimTarget: 1,
      shown: -1,
      kind: attn ? "attention" : "DeltaNet",
      nH,
      pitch,
      headPos,
      heads,
      hl,
      headAct: new Float32Array(nH),
      headEdit: new Float32Array(nH),
    });
  }

  // the line every word travels along
  const y0 = layers[0].home.pos.y - SPINE_PAD;
  const y1 = layers[L - 1].home.pos.y + SPINE_PAD;
  const H = y1 - y0;
  const spineMat = new THREE.ShaderMaterial({
    uniforms: { ...uni, uH: { value: H }, uProgress: { value: 0 }, uIdle: { value: rawVec3(colours.ruleStrong) } },
    vertexShader: SPINE_VS,
    fragmentShader: SPINE_FS,
  });
  const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, H, 10, 1, true), spineMat);
  spine.position.y = (y0 + y1) / 2;
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.06, 20, 14), new THREE.MeshBasicMaterial({ color: colour(colours.act[8]) }));
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 20, 14),
    new THREE.MeshBasicMaterial({ color: colour(colours.act[8]), transparent: true, opacity: 0.18, depthWrite: false }),
  );
  ball.add(halo);
  world.add(spine, ball);

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.024, 0.032, 32).rotateX(-Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: colour(colours.ink1), side: THREE.DoubleSide, depthTest: false }),
  );
  ring.renderOrder = 10;

  // a clicked neuron's wiring into later layers
  const wgeo = new THREE.InstancedBufferGeometry();
  wgeo.index = lineGeo.index;
  wgeo.setAttribute("position", lineGeo.getAttribute("position"));
  const maxWire = WIRE_PER_LAYER * L;
  const wattr = {
    aStart: new THREE.InstancedBufferAttribute(new Float32Array(maxWire * 3), 3).setUsage(THREE.DynamicDrawUsage),
    aEnd: new THREE.InstancedBufferAttribute(new Float32Array(maxWire * 3), 3).setUsage(THREE.DynamicDrawUsage),
    aStrength: new THREE.InstancedBufferAttribute(new Float32Array(maxWire), 1).setUsage(THREE.DynamicDrawUsage),
    aSign: new THREE.InstancedBufferAttribute(new Float32Array(maxWire), 1).setUsage(THREE.DynamicDrawUsage),
  };
  for (const [k, a] of Object.entries(wattr)) wgeo.setAttribute(k, a);
  wgeo.instanceCount = 0;
  const wire = new THREE.Mesh(
    wgeo,
    new THREE.ShaderMaterial({
      uniforms: { uTime: uni.uTime, uMotion: uni.uMotion, uBoost: { value: rawVec3(colours.ink1) }, uSuppress: { value: rawVec3(colours.ink2) } },
      vertexShader: WIRE_VS,
      fragmentShader: WIRE_FS,
    }),
  );
  wire.frustumCulled = false;
  const marks = new THREE.InstancedMesh(new THREE.SphereGeometry(0.02, 10, 8), new THREE.MeshBasicMaterial({ color: colour(colours.ink1) }), maxWire);
  marks.count = 0;
  marks.frustumCulled = false;
  world.add(wire, marks);

  // attention: lines from a layer (or head) down to the words it looked at
  const lookSvg = document.createElementNS(svgNS, "svg");
  lookSvg.setAttribute("class", "s3d-look");
  lookSvg.setAttribute("aria-hidden", "true");
  const lookPaths = Array.from({ length: LOOK_LINES }, () => {
    const p = document.createElementNS(svgNS, "path");
    lookSvg.append(p);
    return p;
  });
  root.append(lookSvg);
  const lookCap = h("p", { class: "s3d-lookcap" });
  lookCap.hidden = true;
  stage.append(lookCap);

  // the next word: the candidates the model weighed, with their probabilities
  const cand = h("div", { class: "s3d-cand", "aria-live": "off" });
  cand.hidden = true;
  stage.append(cand);

  // view controls: spread the neurons (within a layer) or the layers, neuron size,
  // fit to screen, full window. Open by default; the View button folds them away.
  const viewPanel = h("div", { class: "s3d-viewpanel", id: "s3d-viewpanel" });
  const viewBtn = h(
    "button",
    {
      class: "btn btn-quiet btn-sm s3d-viewbtn",
      type: "button",
      "aria-controls": "s3d-viewpanel",
      onclick: () => {
        V.layout.open = viewPanel.hidden;
        saveLayout();
        showViewPanel();
        if (!V.userMoved && V.sep === null) fitCamera(true); // the panel's room changed
      },
    },
    "View",
  );
  const fullBtn = h("button", { class: "btn btn-quiet btn-sm", type: "button", title: "The 3D view fills the window (F)", onclick: () => setFull(!V.full) }, "Full window");
  stage.append(h("div", { class: "s3d-view" }, viewPanel, viewBtn));

  // ripples: where an edit just landed
  const rippleGeo = new THREE.RingGeometry(0.86, 1, 48).rotateX(-Math.PI / 2);
  const ripples = Array.from({ length: 16 }, () => {
    const r = new THREE.Mesh(rippleGeo, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide }));
    r.renderOrder = 12;
    r.userData = { born: -100, r0: 0, r1: 0 };
    return r;
  });

  // an edit you hover in the edits rail: a beacon from it up out of the tower
  const beaconMat = new THREE.MeshBasicMaterial({ color: colour(colours.ink1), depthTest: false, transparent: true });
  const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 1, 6, 1, true).translate(0, 0.5, 0), beaconMat);
  const beaconRing = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.068, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: colour(colours.ink1), side: THREE.DoubleSide, depthTest: false, transparent: true }));
  const beaconPlate = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ color: colour(colours.ink1), side: THREE.DoubleSide, transparent: true, opacity: 0.18, depthWrite: false }));
  beacon.renderOrder = beaconRing.renderOrder = 11;
  beacon.visible = beaconRing.visible = beaconPlate.visible = false;
  world.add(beacon, beaconRing);
  const beaconLabel = h("span", { class: "s3d-beacon" });
  beaconLabel.hidden = true;
  labelsEl.append(beaconLabel);

  const view = {
    root,
    stage,
    canvas,
    labelsEl,
    tip,
    chip,
    empty,
    ioIn,
    ioOut,
    tl,
    renderer,
    scene,
    camera,
    controls,
    home,
    uni,
    layers,
    plates: layers.map((x) => x.plate),
    spine,
    spineMat,
    ball,
    halo,
    ring,
    wire,
    wattr,
    marks,
    wiring: null, // { l, n, targets }
    lookSvg,
    lookPaths,
    lookCap,
    lookCache: new Map(), // row -> look data
    lookPending: new Set(),
    lookKey: null,
    lookTargets: [],
    rowToStep: new Map(),
    selHead: null, // { l, h }
    cols: makeCols(colours),
    cand,
    viewPanel,
    viewBtn,
    fullBtn,
    full: false,
    userMoved: false, // the camera stops fitting itself once you orbit or zoom
    ripples,
    beacon,
    beaconRing,
    beaconPlate,
    beaconLabel,
    beaconFor: null,
    guessCache: new Map(), // row -> per-layer lens guesses
    guessPending: new Set(),
    guessKey: null,
    candKey: null,
    layout: loadLayout(),
    spineH0: H,
    local,
    localAttr,
    ROWS,
    D,
    L,
    N,
    W,
    y0,
    y1,
    data: null, // { replyId, live, steps, k, idx(s, l), val(s, l) }
    loading: null,
    pos: 0, // step * L + layer, fractional while playing
    playing: false,
    speed: 1,
    sep: null, // pulled-out layer
    selected: null, // { l, n }
    savedView: null,
    camAnim: null,
    editSig: null,
    hover: null,
    hoverNeuron: null,
    pointer: new THREE.Vector2(),
    pointerIn: false,
    pxy: [0, 0],
    down: null,
    running: false,
    last: 0,
    liveRef: null,
    escAt: 0,
  };
  V = view; // the helpers below read V
  buildViewPanel();
  relayout();
  wirePointer();
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(recolour).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  controls.addEventListener("start", () => {
    V.camAnim = null;
  });
  return view;
}

// Glow reads as light on the dark theme and as a soft shadow on paper, so paper gets less.
function glowStrength(c) {
  const n = colour(c.well).getHex();
  const lum = (0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255)) / 255;
  return lum > 0.5 ? 0.14 : 0.36;
}

function showViewPanel() {
  V.viewPanel.hidden = !V.layout.open;
  V.viewBtn.setAttribute("aria-expanded", String(!!V.layout.open));
  V.viewBtn.textContent = V.layout.open ? "Hide view controls" : "View";
}

// Full window: the 3D view covers the whole page; Esc or F brings it back.
function setFull(on) {
  V.full = on;
  V.root.classList.toggle("is-full", on);
  V.fullBtn.textContent = on ? "Exit full window" : "Full window";
  V.userMoved = false; // a new size: fit again
  closePopover();
  V.canvas.focus({ preventScroll: true });
}

function spawnRipple(Ly, x, z, code, r0, r1) {
  if (!V.uni.uMotion.value) return;
  const r = V.ripples.find((m) => !m.parent) || V.ripples.reduce((a, b) => (a.userData.born < b.userData.born ? a : b));
  r.material.color.copy(V.cols.edit[code] || V.cols.ramp[8]);
  r.position.set(x, 0.006, z);
  r.userData = { born: V.uni.uTime.value, r0, r1 };
  Ly.g.add(r);
}

function updateRipples(t) {
  for (const r of V.ripples) {
    if (!r.parent) continue;
    const u = (t - r.userData.born) / 0.9;
    if (u >= 1) {
      r.removeFromParent();
      continue;
    }
    const e = 1 - (1 - u) * (1 - u);
    r.scale.setScalar(r.userData.r0 + (r.userData.r1 - r.userData.r0) * e);
    r.material.opacity = (1 - u) * (1 - u);
  }
}

function recolour() {
  const c = readColours();
  V.uni.uRamp.value.dispose();
  V.uni.uRamp.value = paletteTexture(c.act, 16);
  V.uni.uEdits.value.dispose();
  V.uni.uEdits.value = paletteTexture(editPalette(c), 16);
  V.uni.uBg.value = rawVec3(c.well);
  V.uni.uGlow.value = glowStrength(c);
  V.spineMat.uniforms.uIdle.value = rawVec3(c.ruleStrong);
  V.ball.material.color = colour(c.act[8]);
  V.halo.material.color = colour(c.act[8]);
  V.ring.material.color = colour(c.ink1);
  for (const Ly of V.layers) Ly.baseEdge = colour(Ly.kind === "attention" ? c.ruleStrong : c.rule);
  V.hotEdge = colour(c.ink2);
  V.editSig = null; // re-read the edit colours
  V.quietEdits = true; // ...without flashing them as if they were new
  V.cols = makeCols(c);
  V.wire.material.uniforms.uBoost.value = rawVec3(c.ink1);
  V.wire.material.uniforms.uSuppress.value = rawVec3(c.ink2);
  V.marks.material.color = colour(c.ink1);
  for (const Ly of V.layers) paintHeads(Ly);
}

function resize() {
  const r = V.stage.getBoundingClientRect();
  if (!r.width || !r.height) return;
  V.renderer.setSize(r.width, r.height, false);
  V.camera.aspect = r.width / r.height;
  V.camera.updateProjectionMatrix();
  V.uni.uScale.value = V.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(V.camera.fov) / 2));
  if (!V.userMoved && V.sep === null) fitCamera(V.fitView != null); // the first fit jumps, later ones glide
}

// --- data ------------------------------------------------------------------------------------

function sync() {
  if (!V) return;
  if (state.streaming || V.data?.live) return; // the stream feeds the live view
  const last = S.lastReply();
  if (!last) {
    if (V.data) setData(null);
    showEmpty("The sheets light up after the first reply: every word climbs the 24 layers, and the neurons that fire light up.");
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
  V.guessKey = V.candKey = null;
  V.playing = false;
  for (const Ly of V.layers) Ly.shown = -2; // force a refresh
  deselect();
  V.tl.rebuild();
  const n = data?.steps.length ?? 0;
  V.pos = n ? n * V.L - 1 : 0; // show the finished reply
  applyPos(false);
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
// word before it (dimmed), the way the model's state really is mid-pass.
function applyPos(animate) {
  const n = nSteps();
  const { t, l } = cur();
  for (const Ly of V.layers) {
    const want = !n ? -1 : Ly.l <= l ? t : t - 1;
    Ly.dimTarget = !n || Ly.l <= l ? 1 : STALE;
    if (Ly.shown !== want) setLayer(Ly, want, animate);
  }
  V.tl.update();
}

function setLayer(Ly, s, animate) {
  Ly.shown = s;
  const now = V.uni.uTime.value;
  Ly.act.fill(0);
  let n = 0;
  if (s >= 0 && V.data) {
    const idx = V.data.idx(s, Ly.l);
    const val = V.data.val(s, Ly.l);
    const k = idx.length;
    for (let m = 0; m < k; m++) {
      const a = val[m] / 255;
      Ly.act[idx[m]] = a;
      Ly.birth[idx[m]] = animate ? now : -100;
    }
    // lines to the strongest, thicker for harder firing
    for (let m = 0; m < k && n < LINES; m++) {
      const a = val[m] / 255;
      if (a < 0.08) break;
      const i = idx[m];
      Ly.aEnd.array[n * 3] = V.local[i * 3];
      Ly.aEnd.array[n * 3 + 1] = 0;
      Ly.aEnd.array[n * 3 + 2] = V.local[i * 3 + 2];
      Ly.aStrength.array[n] = a;
      Ly.aBirth.array[n] = animate ? now + n * 0.006 : -100;
      n++;
    }
  }
  Ly.lgeo.instanceCount = n;
  Ly.aEnd.needsUpdate = Ly.aStrength.needsUpdate = Ly.aBirth.needsUpdate = true;
  Ly.attrs.act.needsUpdate = Ly.attrs.birth.needsUpdate = true;
  setHeads(Ly, s, animate, now);
}

function setHeads(Ly, s, animate, now) {
  if (!Ly.nH) return;
  const vals = s >= 0 && V.data?.heads ? V.data.heads(s, Ly.l) : null;
  const hl = Ly.hl;
  let n = 0;
  for (let k = 0; k < Ly.nH; k++) {
    const a = vals ? (vals[k] ?? 0) / 255 : 0;
    Ly.headAct[k] = a;
    if (a < 0.05) continue;
    const p = Ly.headPos[k];
    hl.aEnd.array[n * 3] = p.x;
    hl.aEnd.array[n * 3 + 1] = 0;
    hl.aEnd.array[n * 3 + 2] = p.z;
    hl.aStrength.array[n] = a;
    hl.aBirth.array[n] = animate ? now + 0.05 + k * 0.01 : -100;
    n++;
  }
  hl.geo.instanceCount = n;
  hl.aEnd.needsUpdate = hl.aStrength.needsUpdate = hl.aBirth.needsUpdate = true;
  paintHeads(Ly);
}

const tmpM = new THREE.Matrix4();
const tmpQ = new THREE.Quaternion();
const tmpS = new THREE.Vector3();
// head tiles: height and lightness by how much the head wrote; your edits in edit colours
function paintHeads(Ly) {
  if (!Ly.nH || !V?.cols) return;
  for (let k = 0; k < Ly.nH; k++) {
    const a = Ly.headAct[k];
    const code = Ly.headEdit[k];
    const height = code === 9 ? 0.004 : 0.008 + 0.16 * a;
    tmpS.set(Ly.pitch * 0.78, height, 0.08);
    tmpM.compose(Ly.headPos[k], tmpQ, tmpS);
    Ly.heads.setMatrixAt(k, tmpM);
    const c = code ? V.cols.edit[code] : V.cols.ramp[Math.min(8, Math.round((0.08 + 0.92 * a) * 8))];
    Ly.heads.setColorAt(k, c);
  }
  Ly.heads.instanceMatrix.needsUpdate = true;
  if (Ly.heads.instanceColor) Ly.heads.instanceColor.needsUpdate = true;
  Ly.heads.computeBoundingSphere();
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
  // flash what changed, but not on the first look or a theme change
  const flash = V.editSig !== null && !V.quietEdits;
  V.editSig = sig;
  V.quietEdits = false;
  const before = flash ? V.layers.map((Ly) => ({ edit: Ly.edit.slice(), head: Ly.headEdit.slice(), layer: Ly.layerEdit ? editCode(Ly.layerEdit) : 0 })) : null;
  for (const Ly of V.layers) {
    Ly.edit.fill(0);
    Ly.headEdit.fill(0);
    Ly.layerEdit = null;
  }
  for (const e of state.edits) {
    const t = e.target;
    const Ly = V.layers[t.layer];
    if (!Ly) continue;
    if (t.kind === "neuron" && t.index < V.N) Ly.edit[t.index] = editCode(e);
    else if (t.kind === "head" && t.index < Ly.nH) Ly.headEdit[t.index] = editCode(e);
    else if (t.kind === "layer") Ly.layerEdit = e;
  }
  const c = readColours();
  const pal = editPalette(c);
  for (const Ly of V.layers) {
    Ly.attrs.edit.needsUpdate = true;
    Ly.editEdge = Ly.layerEdit ? colour(pal[editCode(Ly.layerEdit)]) : null;
    Ly.tint.visible = !!Ly.layerEdit;
    if (Ly.layerEdit) Ly.tint.material.color.copy(Ly.editEdge);
    paintHeads(Ly);
  }
  if (flash) flashChanges(before);
}

// An edit just changed: its neurons flash and swell, and a ripple spreads from the spot.
function flashChanges(before) {
  const now = V.uni.uTime.value;
  const sp = V.layout.sp;
  let n = 0;
  for (const Ly of V.layers) {
    const b = before[Ly.l];
    let changed = false;
    for (let i = 0; i < V.N; i++) {
      if (Ly.edit[i] === b.edit[i]) continue;
      Ly.editAt[i] = now;
      changed = true;
      if (n++ < 8) spawnRipple(Ly, V.local[i * 3], V.local[i * 3 + 2], Ly.edit[i] || b.edit[i], sp * 0.8, sp * 7);
    }
    if (changed) Ly.attrs.editAt.needsUpdate = true;
    for (let k = 0; k < Ly.nH; k++) {
      if (Ly.headEdit[k] === b.head[k]) continue;
      const p = Ly.headPos[k];
      if (n++ < 8) spawnRipple(Ly, p.x, p.z, Ly.headEdit[k] || b.head[k], Ly.pitch * 0.5, Ly.pitch * 2.5);
    }
    const lc = Ly.layerEdit ? editCode(Ly.layerEdit) : 0;
    if (lc !== b.layer && n++ < 12) spawnRipple(Ly, 0, 0, lc || b.layer, V.W * 0.15, V.W * 0.75);
  }
}

// --- layout: the View sliders -------------------------------------------------------------------

function buildViewPanel() {
  const row = (label, key, min, max, step, hint) => {
    const out = h("output", { class: "num" });
    const input = h("input", { type: "range", min, max, step, value: V.layout[key], "aria-label": label, title: hint });
    const show = () => (out.textContent = `${Math.round((V.layout[key] / LAYOUT[key]) * 100)}%`);
    input.addEventListener("input", () => {
      V.layout[key] = Number(input.value);
      show();
      relayout();
    });
    input.addEventListener("change", saveLayout);
    show();
    return h("label", { class: "s3d-vrow", title: hint }, h("span", {}, label), out, input);
  };
  V.viewPanel.replaceChildren(
    row("↔ Within a layer", "sp", 0.02, 0.09, 0.001, "Horizontal: spread the neurons on each sheet apart"),
    row("↕ Between layers", "gap", 0.06, 1, 0.01, "Vertical: spread the layers apart, or stack them tight"),
    row("Neuron size", "size", 0.015, 0.1, 0.001, "How big each neuron is drawn"),
    h(
      "div",
      { class: "s3d-vactions" },
      h("button", { class: "btn btn-quiet btn-sm", type: "button", title: "The whole tower at 90% of the view", onclick: fitToScreen }, "Fit to screen"),
      V.fullBtn,
      h(
        "button",
        {
          class: "btn btn-quiet btn-sm",
          type: "button",
          title: "Default spacing and size",
          onclick: () => {
            V.layout = { ...LAYOUT };
            saveLayout();
            buildViewPanel();
            relayout();
            fitToScreen();
          },
        },
        "Reset",
      ),
    ),
  );
  showViewPanel();
}

function fitToScreen() {
  putBack();
  V.savedView = null;
  V.userMoved = false;
  fitCamera(true);
}

// Put the camera where the whole tower (sheets, heads, the word's line) fills FIT of the
// stage, keeping the usual three-quarter view from above, with room on the right for the
// layer labels. Searches the distance, then pans so the tower sits centred in its space.
const fitBox = new THREE.Box3();
const fitCorner = new THREE.Vector3();
function fitCamera(animate) {
  const w = V.stage.clientWidth;
  const hgt = V.stage.clientHeight;
  if (!w || !hgt) return;
  const cam = V.camera;
  const keep = { pos: cam.position.clone(), quat: cam.quaternion.clone() };
  fitBox.min.set(-V.W / 2 - 0.04, V.y0 - 0.15, -(V.D / 2 + 0.22));
  fitBox.max.set(V.W / 2 + 0.04, V.y1 + 0.15, V.D / 2 + 0.04);
  const centre = fitBox.getCenter(new THREE.Vector3());
  const dir = V.home.pos.clone().sub(V.home.target).normalize();
  // NDC width kept free: the view controls on the left (when open), the labels on the right
  const panel = V.viewPanel.hidden ? 0 : V.viewPanel.offsetWidth + 20;
  let roomL = (2 * panel) / w;
  let roomR = (2 * LABEL_ROOM) / w;
  const total = roomL + roomR;
  if (total > 1) {
    // never give away more than half the width
    roomL /= total;
    roomR /= total;
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
    cam.position.copy(target).addScaledVector(dir, d);
    cam.lookAt(target);
  };
  let lo = 0.5, hi = 80;
  for (let i = 0; i < 32; i++) {
    const d = (lo + hi) / 2;
    aim(centre, d);
    const b = bounds();
    if (b.x1 - b.x0 <= FIT * (2 - roomL - roomR) && b.y1 - b.y0 <= 2 * FIT) hi = d;
    else lo = d;
  }
  // pan so the box's middle lands in the middle of the space left of the labels
  const target = centre.clone();
  const t = Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2);
  const R = new THREE.Vector3();
  const U = new THREE.Vector3();
  for (let i = 0; i < 3; i++) {
    aim(target, hi);
    const b = bounds();
    R.setFromMatrixColumn(cam.matrixWorld, 0);
    U.setFromMatrixColumn(cam.matrixWorld, 1);
    const dx = (b.x0 + b.x1) / 2 - (roomL - roomR) / 2; // where it is minus where it should be
    const dy = (b.y0 + b.y1) / 2;
    target.addScaledVector(R, dx * hi * t * cam.aspect).addScaledVector(U, dy * hi * t);
  }
  aim(target, hi);
  const pos = cam.position.clone();
  cam.position.copy(keep.pos);
  cam.quaternion.copy(keep.quat);
  cam.updateMatrixWorld();
  V.fitView = { pos, target };
  if (animate) V.camAnim = { pos: pos.clone(), target: target.clone() };
  else {
    V.camAnim = null;
    cam.position.copy(pos);
    V.controls.target.copy(target);
    V.controls.update();
  }
}

// Place everything for the current layout (layer gap, neuron spacing, size).
function relayout() {
  const { gap, sp, size } = V.layout;
  const L = V.L;
  V.uni.uSize.value = size;
  for (let i = 0; i < V.N; i++) {
    V.local[i * 3] = ((i % COLS) - (COLS - 1) / 2) * sp;
    V.local[i * 3 + 2] = (Math.floor(i / COLS) - (V.ROWS - 1) / 2) * sp;
  }
  V.localAttr.needsUpdate = true;
  V.W = COLS * sp;
  V.D = V.ROWS * sp;
  const k = sp / SP;
  ray.params.Points.threshold = sp * 0.55;
  for (const Ly of V.layers) {
    Ly.points.geometry.computeBoundingSphere();
    Ly.plate.scale.set(k, 1, k);
    Ly.tint.scale.set(k, 1, k);
    Ly.edge.scale.set(k, 1, k);
    Ly.pitch = V.W / Math.max(1, Ly.nH);
    Ly.headPos.forEach((p, j) => p.set((j - (Ly.nH - 1) / 2) * Ly.pitch, 0, -(V.D / 2 + 0.12)));
    Ly.home.pos.set(0, (Ly.l - (L - 1) / 2) * gap, 0);
    if (V.sep !== Ly.l) Ly.target.pos.copy(Ly.home.pos);
    Ly.shown = -2; // redraw the lines to the new places
  }
  V.beaconPlate.scale.set(k, 1, k);
  V.ring.scale.setScalar(k);
  V.y0 = V.layers[0].home.pos.y - SPINE_PAD;
  V.y1 = V.layers[L - 1].home.pos.y + SPINE_PAD;
  V.spine.scale.y = (V.y1 - V.y0) / V.spineH0;
  V.spine.position.y = (V.y0 + V.y1) / 2;
  if (V.selected) V.ring.position.set(V.local[V.selected.n * 3], 0.004, V.local[V.selected.n * 3 + 2]);
  if (V.selHead) V.ring.position.set(V.layers[V.selHead.l].headPos[V.selHead.h].x, 0.18, V.layers[V.selHead.l].headPos[V.selHead.h].z);
  V.beaconFor = null;
  applyPos(false);
  if (!V.userMoved && V.sep === null) fitCamera(true); // keep the tower fitted while you spread it
}

// --- hovering an edit in the rail: where is it? -------------------------------------------------

const bp = new THREE.Vector3();
function updateBeacon(t) {
  const e = S.hover.edit;
  const tg = e?.target;
  const Ly = tg && tg.layer != null ? V.layers[tg.layer] : null;
  const ok = Ly && (tg.kind === "neuron" || tg.kind === "head" || tg.kind === "layer") && (tg.kind !== "head" || tg.index < Ly.nH);
  if (!ok) {
    V.beacon.visible = V.beaconRing.visible = false;
    V.beaconPlate.removeFromParent();
    V.beaconLabel.hidden = true;
    V.beaconFor = null;
    return;
  }
  const c = V.cols.edit[editCode(e)] || V.cols.ramp[8];
  if (V.beaconFor !== e.id) {
    V.beaconFor = e.id;
    V.beacon.material.color.copy(c);
    V.beaconRing.material.color.copy(c);
    V.beaconPlate.material.color.copy(c);
    V.beaconLabel.textContent = `${e.label ?? ""}`.trim() || "edit";
    V.beaconLabel.style.setProperty("--edit", `#${c.getHexString()}`);
  }
  // the spot: the neuron, the head tile, or the middle of the sheet
  if (tg.kind === "neuron") bp.set(V.local[tg.index * 3], 0, V.local[tg.index * 3 + 2]);
  else if (tg.kind === "head") bp.copy(Ly.headPos[tg.index]).setY(0.05);
  else bp.set(0, 0, 0);
  Ly.g.localToWorld(bp);
  const top = Math.max(V.y1 + 0.6, bp.y + 0.6);
  V.beacon.position.copy(bp);
  V.beacon.scale.y = top - bp.y;
  V.beacon.visible = true;
  const pulse = 1 + 0.35 * Math.sin(t * 6);
  V.beaconRing.visible = tg.kind !== "layer";
  V.beaconRing.position.copy(bp);
  V.beaconRing.quaternion.copy(Ly.g.quaternion);
  V.beaconRing.scale.setScalar(pulse * (V.layout.sp / SP));
  if (tg.kind === "layer") {
    if (V.beaconPlate.parent !== Ly.g) Ly.g.add(V.beaconPlate);
    V.beaconPlate.material.opacity = 0.14 + 0.1 * Math.sin(t * 6);
  } else V.beaconPlate.removeFromParent();
  V.beaconLabel.hidden = false;
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
    .then((g) => V.data?.replyId === replyId && V.guessCache.set(row, g))
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
  const gkey = `${d?.replyId}:${t}:${l}:${g ? 1 : 0}:${V.sep}`;
  if (gkey !== V.guessKey) {
    V.guessKey = gkey;
    for (const Ly of V.layers) {
      const reached = g && (Ly.l <= l || V.sep === Ly.l);
      const lg = reached ? g.layers[Ly.l] : null;
      if (!lg) {
        Ly.labelGuess.textContent = "";
        Ly.labelBar.style.width = "0";
        Ly.label.classList.remove("is-right");
        continue;
      }
      const top = lg.top[0];
      const right = g.next != null && top && top.text === g.next;
      Ly.labelGuess.textContent = top ? `${quoteTok(top.text)} ${pct(top.prob)}` : "";
      Ly.labelBar.style.width = `${Math.round((lg.p_next ?? 0) * 36)}px`;
      Ly.label.classList.toggle("is-right", !!right);
      Ly.label.title = g.next != null ? `Layer ${Ly.l} gave ${quoteTok(g.next)} ${pct(lg.p_next ?? 0)}` : "";
    }
  }
  // the candidates card
  const ckey = `${d?.replyId}:${n}:${t}`;
  if (ckey === V.candKey) return;
  V.candKey = ckey;
  if (!st) {
    V.cand.hidden = true;
    return;
  }
  V.cand.hidden = false;
  const alts = st.alts?.length ? st.alts : st.says != null ? [{ text: st.says, prob: st.prob ?? 0 }] : [];
  V.cand.replaceChildren(
    h("p", { class: "s3d-cand-head" }, h("strong", {}, "Next word"), h("span", { class: "meta" }, st.reads == null ? " after your message" : ` after ${quoteTok(st.reads)}`)),
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
      : h("p", { class: "meta" }, st.end === "stop" ? "It stops here." : "Cut off here."),
  );
  V.cand.title = "The model's own probabilities for the next word. Beside each layer: its best guess so far, and a bar for how sure it already was of the word that came.";
}
const pct = (p) => (p >= 0.995 ? "100%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`);

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
      V.fullBtn.textContent = "Full window";
    }
    return;
  }
  const now = performance.now() / 1000;
  const dt = Math.min(0.05, now - V.last);
  V.last = now;
  V.uni.uTime.value += dt;
  syncEdits();

  if (V.playing) {
    const max = nSteps() * V.L - 1;
    V.pos = Math.min(max, V.pos + dt * LAYERS_PER_SECOND * V.speed);
    if (V.pos >= max) V.playing = false;
    applyPos(true);
  }

  const k = 1 - Math.exp(-dt * 7);
  const hot = V.hotEdge || (V.hotEdge = colour(readColours().ink2));
  for (const Ly of V.layers) {
    Ly.g.position.lerp(Ly.target.pos, k);
    Ly.g.quaternion.slerp(Ly.target.quat, k);
    const want = V.sep === null ? Ly.dimTarget : V.sep === Ly.l ? 1 : 0.25;
    Ly.dim += (want - Ly.dim) * k;
    Ly.pmat.uniforms.uDim.value = Ly.dim;
    Ly.gmat.uniforms.uDim.value = Ly.dim;
    Ly.tint.material.opacity = 0.16 * Math.max(0.4, Ly.dim);
    Ly.lmat.uniforms.uDim.value = Ly.dim;
    Ly.hl.mat.uniforms.uDim.value = Ly.dim;
    Ly.heads.material.opacity = Math.max(0.2, Ly.dim);
    const isHot = V.hover === Ly.l || V.sep === Ly.l;
    Ly.edge.material.color.copy(Ly.editEdge || (isHot ? hot : Ly.baseEdge));
    Ly.edge.material.opacity = Math.max(0.25, Ly.dim);
  }

  // the ball rides the line at the playhead
  const n = nSteps();
  const { l, frac } = cur();
  const y = n ? V.layers[l].home.pos.y + (l < V.L - 1 ? frac * V.layout.gap : 0) : V.y0;
  V.ball.position.set(0, y, 0);
  V.ball.visible = n > 0;
  V.spineMat.uniforms.uProgress.value = n ? (y - V.y0) / (V.y1 - V.y0) : 0;

  if (V.camAnim) {
    V.camera.position.lerp(V.camAnim.pos, k);
    V.controls.target.lerp(V.camAnim.target, k);
    if (V.camera.position.distanceTo(V.camAnim.pos) < 0.005) V.camAnim = null;
  }
  V.controls.update();
  const dist = V.camera.position.distanceTo(V.controls.target);
  V.uni.uFogNear.value = dist - 1.5;
  V.uni.uFogFar.value = dist + 7;
  updateHover();
  updateWiring();
  updateBeacon(V.uni.uTime.value);
  updateRipples(V.uni.uTime.value);
  updateWords();
  V.renderer.render(V.scene, V.camera);
  placeLabels();
  V.tl.head();
  drawLook();
}

const tmp = new THREE.Vector3();
function place(el, p, minTop = null) {
  tmp.copy(p).project(V.camera);
  if (tmp.z > 1 || Math.abs(tmp.x) > 1.2 || Math.abs(tmp.y) > 1.2) {
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.style.left = `${(tmp.x * 0.5 + 0.5) * V.stage.clientWidth}px`;
  const y = (-tmp.y * 0.5 + 0.5) * V.stage.clientHeight;
  el.style.top = `${minTop == null ? y : Math.max(minTop, y)}px`;
}
const labelAt = new THREE.Vector3();
function placeLabels() {
  labelAt.set(V.W / 2 + 0.22, 0, -(V.D / 2 + 0.12)); // past the end of the head row
  for (const Ly of V.layers) {
    place(Ly.label, labelAt.clone().applyMatrix4(Ly.g.matrixWorld));
    Ly.label.classList.toggle("is-hot", V.hover === Ly.l || V.sep === Ly.l);
    Ly.label.style.opacity = V.sep === null || V.sep === Ly.l ? "" : "0.35";
  }
  hideLabelsUnderSheet();
  const st = V.data?.steps[cur().t];
  V.ioIn.hidden = V.ioOut.hidden = !st;
  if (!st) return;
  V.ioIn.replaceChildren(h("small", {}, "reads"), st.reads == null ? "your message" : quoteTok(st.reads));
  V.ioOut.replaceChildren(h("small", {}, "says"), sayText(st));
  place(V.ioIn, new THREE.Vector3(0, V.y0 - 0.1, 0).applyMatrix4(V.scene.matrixWorld));
  place(V.ioOut, new THREE.Vector3(0, V.y1 + 0.1, 0));
}
// The labels are HTML over the canvas, so a pulled-out sheet can't hide the other layers'
// labels behind it. Hide the ones that would sit on top of it (heads included).
const corner = new THREE.Vector3();
function hideLabelsUnderSheet() {
  const sep = V.sep === null ? null : V.layers.find((Ly) => Ly.l === V.sep);
  if (!sep) {
    for (const Ly of V.layers) Ly.label.style.visibility = "";
    return;
  }
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, z] of [[-V.W / 2, -(V.D / 2 + 0.12)], [V.W / 2, -(V.D / 2 + 0.12)], [-V.W / 2, V.D / 2], [V.W / 2, V.D / 2]]) {
    corner.set(x, 0, z).applyMatrix4(sep.g.matrixWorld).project(V.camera);
    const sx = (corner.x * 0.5 + 0.5) * V.stage.clientWidth;
    const sy = (-corner.y * 0.5 + 0.5) * V.stage.clientHeight;
    x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
  }
  for (const Ly of V.layers) {
    const el = Ly.label;
    if (Ly === sep || el.style.display === "none") { el.style.visibility = ""; continue; }
    const left = parseFloat(el.style.left), top = parseFloat(el.style.top);
    const over = top > y0 - 8 && top < y1 + 8 && left < x1 && left + el.offsetWidth > x0;
    el.style.visibility = over ? "hidden" : "";
  }
}
function sayText(st) {
  if (st.says == null) return st.end === "stop" ? "(stops)" : "…";
  const p = modelProb(st);
  return `${quoteTok(st.says)}${p != null ? `  ${pct(p)}` : ""}`;
}
// The model's own probability for the word it wrote (the candidates list),
// not the sampler's after temperature, top-k and top-p.
function modelProb(st) {
  const a = st.alts?.find((x) => x.text === st.says);
  return a ? a.prob : st.prob;
}

// --- pointer: hover, pull a layer out, pick a neuron -----------------------------------------

const ray = new THREE.Raycaster();
ray.params.Points.threshold = 0.016;

function wirePointer() {
  const c = V.canvas;
  c.addEventListener("pointermove", (e) => {
    const r = c.getBoundingClientRect();
    V.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    V.pxy = [e.clientX - r.left, e.clientY - r.top];
    V.pointerIn = true;
    // a drag orbits or pans: from then on the camera stays where you put it
    if (e.buttons && V.down && Math.hypot(e.clientX - V.down[0], e.clientY - V.down[1]) > 5) V.userMoved = true;
  });
  c.addEventListener("wheel", () => (V.userMoved = true), { passive: true });
  c.addEventListener("pointerleave", () => {
    V.pointerIn = false;
  });
  c.addEventListener("pointerdown", (e) => {
    V.down = [e.clientX, e.clientY];
  });
  c.addEventListener("pointerup", (e) => {
    if (!V.down || Math.hypot(e.clientX - V.down[0], e.clientY - V.down[1]) > 5) return;
    V.down = null;
    ray.setFromCamera(V.pointer, V.camera);
    if (V.sep !== null) {
      const hd = pickHead();
      if (hd !== null) {
        openHead(V.sep, hd, e.clientX, e.clientY);
        return;
      }
      const i = pickNeuron();
      if (i !== null) {
        openNeuron(V.sep, i, e.clientX, e.clientY);
        return;
      }
    }
    const hit = ray.intersectObjects(V.plates, false)[0];
    if (hit) {
      const l = hit.object.userData.layer;
      if (l !== V.sep) pullOut(l);
      return;
    }
    putBack();
  });
}

function pickHead() {
  const Ly = V.layers[V.sep];
  if (!Ly.nH) return null;
  const hit = ray.intersectObject(Ly.heads, false)[0];
  return hit ? hit.instanceId : null;
}

function pickNeuron() {
  const hits = ray.intersectObject(V.layers[V.sep].points, false);
  if (!hits.length) return null;
  let best = hits[0];
  for (const x of hits) if (x.distanceToRay < best.distanceToRay) best = x;
  return best.index;
}

function updateHover() {
  V.hover = V.hoverNeuron = null;
  let hoverHead = null;
  if (!V.pointerIn) {
    V.tip.hidden = true;
    return;
  }
  ray.setFromCamera(V.pointer, V.camera);
  if (V.sep !== null) {
    hoverHead = pickHead();
    if (hoverHead === null) V.hoverNeuron = pickNeuron();
  }
  const hit = ray.intersectObjects(V.plates, false)[0];
  if (hit) V.hover = hit.object.userData.layer;
  V.canvas.style.cursor = V.hover !== null || V.hoverNeuron !== null || hoverHead !== null ? "pointer" : "";
  let text = null;
  if (hoverHead !== null) {
    const Ly = V.layers[V.sep];
    text = `L${V.sep} head ${hoverHead} · ${Ly.kind} · ${Math.round(Ly.headAct[hoverHead] * 100)}% of its usual size`;
  } else if (V.hoverNeuron !== null) {
    const a = V.layers[V.sep].act[V.hoverNeuron];
    text = `L${V.sep} N${V.hoverNeuron}${a ? ` · ${Math.round(a * 100)}% of its layer's range` : ""}`;
  } else if (V.hover !== null && V.hover !== V.sep) text = `Layer ${V.hover} · ${V.layers[V.hover].kind} · click to pull it out`;
  V.tip.hidden = !text;
  if (text) {
    V.tip.textContent = text;
    V.tip.style.left = `${Math.min(V.pxy[0] + 14, V.stage.clientWidth - V.tip.offsetWidth - 8)}px`;
    V.tip.style.top = `${V.pxy[1] + 16}px`;
  }
}

function pullOut(l) {
  if (V.sep !== null) {
    const old = V.layers[V.sep];
    old.target = { pos: old.home.pos.clone(), quat: old.home.quat.clone() };
  }
  if (!V.savedView) V.savedView = { pos: V.camera.position.clone(), target: V.controls.target.clone() };
  V.sep = l;
  deselect();
  V.camera.updateMatrixWorld();
  const R = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 0);
  const U = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 1);
  const B = new THREE.Vector3().setFromMatrixColumn(V.camera.matrixWorld, 2); // towards the viewer
  const Ly = V.layers[l];
  const pos = Ly.home.pos.clone().multiplyScalar(0.5).addScaledVector(R, 2.4).addScaledVector(B, 1.2);
  // face the viewer like the 2D grid: neuron 0 top left, rows going down
  const quat = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(R, B, U.clone().negate()));
  Ly.target = { pos, quat };
  V.stage.classList.add("is-sep"); // the view controls step aside for the sheet
  // close enough that the sheet (heads included) fills most of the stage, beside the next-word card
  const t = Math.tan(THREE.MathUtils.degToRad(V.camera.fov) / 2);
  const wpx = V.stage.clientWidth || 800;
  const hpx = V.stage.clientHeight || 600;
  const free = Math.max(240, wpx - (V.cand.hidden ? 0 : 280));
  const sheetW = V.W + 0.1;
  const sheetH = V.D + 0.36;
  const d = Math.max(sheetH / (2 * t * 0.84), (sheetW * hpx) / (2 * t * 0.9 * free));
  const shift = ((wpx - free) / 2) * ((2 * d * t) / hpx); // world units to slide the view right
  const look = pos.clone().addScaledVector(R, shift).addScaledVector(U, 0.1);
  V.camAnim = { pos: look.clone().addScaledVector(B, d), target: look };
  V.chip.hidden = false;
  V.chip.replaceChildren(
    h("strong", {}, `Layer ${l}`),
    h("span", { class: "meta" }, `${Ly.kind} · click a neuron to edit it`),
    h("a", { class: "btn btn-quiet", href: `#/layer/${l}` }, "Open in 2D"),
    h("button", { class: "btn btn-quiet", type: "button", onclick: putBack }, "Put back"),
  );
}

function putBack() {
  if (V.sep === null) return;
  const Ly = V.layers[V.sep];
  Ly.target = { pos: Ly.home.pos.clone(), quat: Ly.home.quat.clone() };
  V.sep = null;
  deselect();
  V.chip.hidden = true;
  V.stage.classList.remove("is-sep");
  if (!V.userMoved) fitCamera(true); // the stage may have changed size meanwhile
  else if (V.savedView) V.camAnim = V.savedView;
  V.savedView = null;
}

function deselect() {
  if (!V) return;
  V.selected = null;
  V.selHead = null;
  V.wiring = null;
  V.ring.removeFromParent();
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
  V.ring.position.set(Ly.headPos[k].x, 0.18, Ly.headPos[k].z);
  Ly.g.add(V.ring);
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
        : " DeltaNet heads keep a running memory instead of looking back at single words, so there is no \u201cwhere it looked\u201d to draw."),
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
  setTimeout(() => {
    if (V.sep !== l) return;
    const p = V.layers[l].g.localToWorld(new THREE.Vector3(V.local[n * 3], 0, V.local[n * 3 + 2])).project(V.camera);
    const r = V.canvas.getBoundingClientRect();
    openNeuron(l, n, r.left + (p.x * 0.5 + 0.5) * r.width, r.top + (-p.y * 0.5 + 0.5) * r.height);
  }, 700);
}

// the wiring follows the sheets while they move
const wa = new THREE.Vector3();
const wb = new THREE.Vector3();
function updateWiring() {
  const w = V.wiring;
  if (!w) {
    V.wire.geometry.instanceCount = 0;
    V.marks.count = 0;
    return;
  }
  const src = V.layers[w.l].g;
  wa.set(V.local[w.n * 3], 0, V.local[w.n * 3 + 2]);
  src.localToWorld(wa);
  const A = V.wattr;
  w.targets.forEach((t, i) => {
    wb.set(V.local[t.neuron * 3], 0, V.local[t.neuron * 3 + 2]);
    V.layers[t.layer].g.localToWorld(wb);
    A.aStart.array.set([wa.x, wa.y, wa.z], i * 3);
    A.aEnd.array.set([wb.x, wb.y, wb.z], i * 3);
    A.aStrength.array[i] = t.strength;
    A.aSign.array[i] = t.sign;
    tmpM.makeTranslation(wb.x, wb.y, wb.z);
    V.marks.setMatrixAt(i, tmpM);
  });
  A.aStart.needsUpdate = A.aEnd.needsUpdate = A.aStrength.needsUpdate = A.aSign.needsUpdate = true;
  V.wire.geometry.instanceCount = w.targets.length;
  V.marks.count = w.targets.length;
  V.marks.instanceMatrix.needsUpdate = true;
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
  V.ring.position.set(V.local[n * 3], 0.004, V.local[n * 3 + 2]);
  V.layers[l].g.add(V.ring);
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
    V.lookTargets = all.filter((x) => x.w >= 0.02).sort((a, b) => b.w - a.w).slice(0, LOOK_LINES);
    V.tl.colourBy(Array.from(w.per), w.msg);
    const name = (t) => (t.i < 0 ? "your message" : quoteTok(V.data.steps[t.i].reads ?? ""));
    V.lookCap.hidden = false;
    V.lookCap.replaceChildren(
      h("strong", {}, `Layer ${lt.l}${lt.head != null ? `, head ${lt.head}` : ""} looked at`),
      " ",
      V.lookTargets.slice(0, 4).map((t, j) => `${j ? " · " : ""}${name(t)} ${Math.round(t.w * 100)}%`).join(""),
    );
  }
  // geometry every frame: the camera and the timeline both move
  const Ly = V.layers[lt.l];
  if (lt.head != null) lp.copy(Ly.headPos[lt.head]).setY(0.1);
  else lp.set(0, 0, 0);
  Ly.g.localToWorld(lp).project(V.camera);
  const sx = (lp.x * 0.5 + 0.5) * V.stage.clientWidth;
  const sy = (-lp.y * 0.5 + 0.5) * V.stage.clientHeight;
  const rootR = V.root.getBoundingClientRect();
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
    p.setAttribute("d", `M${sx},${sy} C${sx},${sy + dy * 0.55} ${ex},${ey - dy * 0.45} ${ex},${ey}`);
    p.setAttribute("stroke-width", (1 + 7 * t.w).toFixed(2));
    p.style.opacity = off ? "0.3" : "";
  });
}

// --- keyboard ------------------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (!V?.root.isConnected) return;
  const tag = e.target?.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === " " && tag !== "BUTTON" && tag !== "A") {
    e.preventDefault();
    togglePlay();
  } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
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

const ICONS = {
  prevWord: '<path d="M2 3h2v10H2zM14 3 6 8l8 5z"/>',
  prevLayer: '<path d="M11 3 4 8l7 5z"/>',
  play: '<path d="M5 3l8 5-8 5z"/>',
  pause: '<path d="M4 3h3v10H4zM9 3h3v10H9z"/>',
  nextLayer: '<path d="M5 3l7 5-7 5z"/>',
  nextWord: '<path d="M12 3h2v10h-2zM2 3l8 5-8 5z"/>',
};
function icon(name) {
  const span = h("span", { class: "s3d-icon", "aria-hidden": "true" });
  span.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor">${ICONS[name]}</svg>`;
  return span;
}

function buildTimeline() {
  const btn = (name, label, fn) => h("button", { class: "btn btn-quiet s3d-tbtn", type: "button", title: label, "aria-label": label, onclick: fn }, icon(name));
  const play = h("button", { class: "btn s3d-tbtn s3d-play", type: "button", onclick: () => togglePlay() });
  const readout = h("div", { class: "s3d-readout" });
  const speed = h(
    "select",
    { class: "s3d-speed", "aria-label": "Playback speed", title: "Playback speed", onchange: (e) => (V.speed = Number(e.target.value)) },
    SPEEDS.map(([v, label]) => h("option", { value: v, selected: v === 1 }, label)),
  );
  const steps = h("div", { class: "s3d-steps" });
  const msg = h("div", { class: "s3d-msg", title: "Your message, and anything earlier in the chat" }, h("span", { class: "s3d-word" }, "your message"));
  const headEl = h("div", { class: "s3d-head", "aria-hidden": "true" });
  const track = h("div", { class: "s3d-track", tabindex: "0", title: "Click or drag to move through the reply" }, msg, steps, headEl);
  const el = h(
    "div",
    { class: "s3d-tl" },
    h(
      "div",
      { class: "s3d-bar" },
      btn("prevWord", "Previous word (Shift ←)", () => step(-V.L)),
      btn("prevLayer", "Previous layer (←)", () => step(-1)),
      play,
      btn("nextLayer", "Next layer (→)", () => step(1)),
      btn("nextWord", "Next word (Shift →)", () => step(V.L)),
      readout,
      speed,
    ),
    track,
  );

  let boxes = []; // [left, width] per step, cached
  let revealNext = true;
  let lastT = -1;
  let lastPlaying = null;
  const stepEls = () => steps.children;

  function stepEl(i) {
    const st = V.data.steps[i];
    const word = st.says == null ? (st.end === "stop" ? "end" : "…") : showToken(st.says).replace(/^␣/, "") || "␣";
    return h(
      "div",
      { class: "s3d-step", dataset: { i }, title: `${st.reads == null ? "your message" : quoteTok(st.reads)} → ${st.says == null ? "end" : quoteTok(st.says)}` },
      h("span", { class: "s3d-fill" }),
      h("span", { class: "s3d-word" }, word),
    );
  }
  function measure() {
    boxes = Array.from(stepEls(), (c) => [c.offsetLeft, c.offsetWidth]);
  }
  function rebuild() {
    steps.replaceChildren(...(V.data ? V.data.steps.map((_, i) => stepEl(i)) : []));
    lastT = -1;
    lastKey = "";
    requestAnimationFrame(measure);
  }
  function append(i) {
    steps.append(stepEl(i));
    const c = steps.lastElementChild;
    boxes[i] = [c.offsetLeft, c.offsetWidth];
  }
  let lastKey = "";
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
    if (t !== lastT) {
      const kids = stepEls();
      for (let i = 0; i < kids.length; i++) {
        kids[i].classList.toggle("is-done", i < t);
        kids[i].classList.toggle("is-current", i === t);
      }
      lastT = t;
    }
    const curEl = stepEls()[t];
    if (curEl) curEl.firstChild.style.width = `${((l + 1) / V.L) * 100}%`;
    if (!n) {
      readout.textContent = "";
      return;
    }
    const st = V.data.steps[t];
    readout.replaceChildren(
      h("strong", {}, live ? `Writing… word ${n}` : `Word ${t + 1} of ${n}`),
      ` · layer ${l} ${V.layers[l].kind} · `,
      st.reads == null ? "reads your message" : `reads ${quoteTok(st.reads)}`,
      " → ",
      st.says == null ? (st.end === "stop" ? "stops" : "…") : `says ${quoteTok(st.says)}`,
      modelProb(st) != null ? ` (${pct(modelProb(st))})` : "",
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
      if (x < left + 24 || x > left + w - 24) track.scrollLeft = x - w / 2;
    }
    headEl._x = x;
  }
  // light the steps by one neuron's activity, or by attention (null to clear)
  function colourBy(values, msgValue = null) {
    const kids = stepEls();
    const max = values ? Math.max(1e-9, ...values, msgValue ?? 0) : 1;
    const mw = msg.lastChild;
    if (values && msgValue != null) {
      const bin = Math.min(8, Math.round((msgValue / max) * 8));
      msg.style.setProperty("--fired", `var(--act-${bin})`);
      mw.style.color = `var(--act-ink-${bin})`;
    } else {
      msg.style.removeProperty("--fired");
      mw.style.color = "";
    }
    for (let i = 0; i < kids.length; i++) {
      const word = kids[i].lastChild;
      if (!values) {
        kids[i].style.removeProperty("--fired");
        kids[i].classList.remove("is-fired");
        word.style.color = "";
        continue;
      }
      const bin = Math.min(8, Math.round((values[i] / max) * 8));
      kids[i].classList.add("is-fired");
      kids[i].style.setProperty("--fired", `var(--act-${bin})`);
      word.style.color = `var(--act-ink-${bin})`;
    }
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
  new ResizeObserver(() => measure()).observe(steps);

  return {
    el,
    rebuild,
    append,
    update,
    head,
    colourBy,
    reveal: () => (revealNext = true),
    trackRect: () => track.getBoundingClientRect(),
    targetRect: (i) => (i < 0 ? msg : stepEls()[i])?.getBoundingClientRect() ?? null,
  };
}
