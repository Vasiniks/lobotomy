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
import { openEditPopover } from "../components.js";
import { fmt, h, quoteTok, showToken } from "../util.js";

const SP = 0.03; // neuron spacing on a sheet
const COLS = 64;
const GAP = 0.32; // between sheets
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
  attribute float act;   // 0..1, how hard it fired against its layer's usual range
  attribute float birth; // when it lit up, for the pop
  attribute float edit;  // editCode
  uniform float uScale, uSize, uTime, uMotion;
  varying float vAct, vEdit;
  void main() {
    vAct = act;
    vEdit = edit;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float pop = 1.0 + 1.4 * act * exp(-max(0.0, uTime - birth) * 5.0) * uMotion;
    float breathe = 1.0 + 0.14 * act * sin(uTime * 3.5 + position.x * 41.0 + position.z * 29.0) * uMotion;
    float s = (0.45 + 1.7 * act) * pop * breathe;
    if (edit > 0.5) s = max(s, 1.5);
    gl_PointSize = max(1.0, uSize * s * uScale / -mv.z);
  }`;
const POINT_FS = /* glsl */ `
  uniform sampler2D uRamp, uEdits;
  uniform vec3 uBg;
  uniform float uDim;
  varying float vAct, vEdit;
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    vec3 c = ramp(0.05 + 0.95 * vAct);
    if (vEdit > 0.5) {
      vec3 e = texture2D(uEdits, vec2((vEdit + 0.5) / 16.0, 0.5)).rgb;
      if (vEdit > 8.5 && vEdit < 9.5) {
        // off: a ring cut through, like the cut on the 2D cells
        if (r2 < 0.08 || abs(d.y) < 0.075) discard;
        c = e;
      } else if (r2 > 0.1) {
        c = e; // the ring is you, the fill is the model
      }
    }
    gl_FragColor = vec4(mix(uBg, c, uDim), 1.0);
  }`;

const LINE_VS = /* glsl */ `
  attribute vec3 aEnd;
  attribute float aStrength, aBirth;
  uniform float uTime, uGrow;
  varying float vY, vS;
  void main() {
    float g = clamp((uTime - aBirth) / uGrow, 0.0, 1.0);
    g = 1.0 - (1.0 - g) * (1.0 - g) * (1.0 - g);
    float len = length(aEnd);
    vec3 ax = aEnd / max(len, 1e-5);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 side = normalize(cross(ax, up));
    float r = mix(0.0012, 0.0115, aStrength);
    vec3 p = ax * (position.y * len * g) + side * (position.x * r) + up * (position.z * r);
    vY = position.y * g;
    vS = aStrength;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;
const LINE_FS = /* glsl */ `
  uniform sampler2D uRamp;
  uniform vec3 uBg;
  uniform float uTime, uDim, uMotion;
  varying float vY, vS;
  vec3 ramp(float a) { return texture2D(uRamp, vec2((clamp(a, 0.0, 1.0) * 8.0 + 0.5) / 16.0, 0.5)).rgb; }
  void main() {
    vec3 c = ramp(0.3 + 0.62 * vS);
    // signal flowing out from the line to the neuron
    float f = fract(vY * 2.5 - uTime * 1.3);
    float band = smoothstep(0.0, 0.06, f) * (1.0 - smoothstep(0.06, 0.32, f));
    c = mix(c, ramp(1.0), band * (0.35 + 0.5 * vS) * uMotion);
    gl_FragColor = vec4(mix(uBg, c, uDim), 1.0);
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
  const home = { pos: new THREE.Vector3(8.8, 7.0, 8.8), target: new THREE.Vector3(0, 0.15, 0) };
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
  const lineGeo = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true).translate(0, 0.5, 0);

  const world = new THREE.Group();
  scene.add(world);
  const layers = [];
  for (let l = 0; l < L; l++) {
    const g = new THREE.Group();
    g.position.set(0, (l - (L - 1) / 2) * GAP, 0);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", localAttr);
    const act = new Float32Array(N);
    const birth = new Float32Array(N).fill(-100);
    const edit = new Float32Array(N);
    const attrs = {
      act: new THREE.BufferAttribute(act, 1).setUsage(THREE.DynamicDrawUsage),
      birth: new THREE.BufferAttribute(birth, 1).setUsage(THREE.DynamicDrawUsage),
      edit: new THREE.BufferAttribute(edit, 1).setUsage(THREE.DynamicDrawUsage),
    };
    for (const [k, a] of Object.entries(attrs)) geo.setAttribute(k, a);
    geo.computeBoundingSphere();
    const pmat = new THREE.ShaderMaterial({
      uniforms: { ...uni, uSize: { value: 0.026 }, uDim: { value: 1 } },
      vertexShader: POINT_VS,
      fragmentShader: POINT_FS,
    });
    const points = new THREE.Points(geo, pmat);

    const plate = new THREE.Mesh(plateGeo, new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false, side: THREE.DoubleSide }));
    plate.userData.layer = l;
    const attn = m.layers[l]?.type === "attention";
    const baseEdge = colour(attn ? colours.ruleStrong : colours.rule);
    const edge = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: baseEdge.clone(), transparent: true }));

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

    g.add(points, plate, edge, lines);
    world.add(g);
    const label = h("span", { class: "s3d-lbl" }, attn ? `${l} attn` : `${l}`);
    labelsEl.append(label);
    const homePose = { pos: g.position.clone(), quat: g.quaternion.clone() };
    layers.push({
      l,
      g,
      points,
      plate,
      edge,
      baseEdge,
      lines,
      lgeo,
      act,
      birth,
      edit,
      attrs,
      aEnd,
      aStrength,
      aBirth,
      pmat,
      lmat,
      label,
      home: homePose,
      target: { pos: homePose.pos.clone(), quat: homePose.quat.clone() },
      dim: 1,
      dimTarget: 1,
      shown: -1,
      kind: attn ? "attention" : "DeltaNet",
    });
  }

  // the line every word travels along
  const y0 = layers[0].home.pos.y - 0.7;
  const y1 = layers[L - 1].home.pos.y + 0.7;
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
    local,
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
  wirePointer();
  new ResizeObserver(resize).observe(stage);
  new MutationObserver(recolour).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  controls.addEventListener("start", () => {
    V.camAnim = null;
  });
  return view;
}

function recolour() {
  const c = readColours();
  V.uni.uRamp.value.dispose();
  V.uni.uRamp.value = paletteTexture(c.act, 16);
  V.uni.uEdits.value.dispose();
  V.uni.uEdits.value = paletteTexture(editPalette(c), 16);
  V.uni.uBg.value = rawVec3(c.well);
  V.spineMat.uniforms.uIdle.value = rawVec3(c.ruleStrong);
  V.ball.material.color = colour(c.act[8]);
  V.halo.material.color = colour(c.act[8]);
  V.ring.material.color = colour(c.ink1);
  for (const Ly of V.layers) Ly.baseEdge = colour(Ly.kind === "attention" ? c.ruleStrong : c.rule);
  V.hotEdge = colour(c.ink2);
  V.editSig = null; // re-read the edit colours
}

function resize() {
  const r = V.stage.getBoundingClientRect();
  if (!r.width || !r.height) return;
  V.renderer.setSize(r.width, r.height, false);
  V.camera.aspect = r.width / r.height;
  V.camera.updateProjectionMatrix();
  V.uni.uScale.value = V.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(V.camera.fov) / 2));
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
  return {
    replyId,
    live: false,
    steps: f.steps,
    k,
    scale: f.scale,
    idx: (s, l) => idx.subarray((s * L + l) * k, (s * L + l + 1) * k),
    val: (s, l) => val.subarray((s * L + l) * k, (s * L + l + 1) * k),
  };
}

function setData(data) {
  V.data = data;
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
  const steps = [];
  const idx = [];
  const val = [];
  V.liveRef = prevScale ? prevScale.map(() => 0) : null;
  V.data = {
    replyId: null,
    after: V.data?.replyId ?? S.lastReply()?.id ?? null, // the reply this one follows
    live: true,
    steps,
    k: 0,
    scale: prevScale,
    idx: (s, l) => idx[s][l],
    val: (s, l) => val[s][l],
    _idx: idx,
    _val: val,
  };
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
  V.editSig = sig;
  for (const Ly of V.layers) {
    Ly.edit.fill(0);
    Ly.layerEdit = null;
  }
  for (const e of state.edits) {
    const t = e.target;
    const Ly = V.layers[t.layer];
    if (!Ly) continue;
    if (t.kind === "neuron" && t.index < V.N) Ly.edit[t.index] = editCode(e);
    else if (t.kind === "layer") Ly.layerEdit = e;
  }
  const c = readColours();
  const pal = editPalette(c);
  for (const Ly of V.layers) {
    Ly.attrs.edit.needsUpdate = true;
    Ly.editEdge = Ly.layerEdit ? colour(pal[editCode(Ly.layerEdit)]) : null;
  }
}

// --- frame loop ------------------------------------------------------------------------------

function frame() {
  if (!V.root.isConnected) {
    V.renderer.setAnimationLoop(null);
    V.running = false;
    V.playing = false;
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
    Ly.lmat.uniforms.uDim.value = Ly.dim;
    const isHot = V.hover === Ly.l || V.sep === Ly.l;
    Ly.edge.material.color.copy(Ly.editEdge || (isHot ? hot : Ly.baseEdge));
    Ly.edge.material.opacity = Math.max(0.25, Ly.dim);
  }

  // the ball rides the line at the playhead
  const n = nSteps();
  const { l, frac } = cur();
  const y = n ? V.layers[l].home.pos.y + (l < V.L - 1 ? frac * GAP : 0) : V.y0;
  V.ball.position.set(0, y, 0);
  V.ball.visible = n > 0;
  V.spineMat.uniforms.uProgress.value = n ? (y - V.y0) / (V.y1 - V.y0) : 0;

  if (V.camAnim) {
    V.camera.position.lerp(V.camAnim.pos, k);
    V.controls.target.lerp(V.camAnim.target, k);
    if (V.camera.position.distanceTo(V.camAnim.pos) < 0.005) V.camAnim = null;
  }
  V.controls.update();
  updateHover();
  V.renderer.render(V.scene, V.camera);
  placeLabels();
  V.tl.head();
}

const tmp = new THREE.Vector3();
function place(el, p) {
  tmp.copy(p).project(V.camera);
  if (tmp.z > 1 || Math.abs(tmp.x) > 1.2 || Math.abs(tmp.y) > 1.2) {
    el.style.display = "none";
    return;
  }
  el.style.display = "";
  el.style.left = `${(tmp.x * 0.5 + 0.5) * V.stage.clientWidth}px`;
  el.style.top = `${(-tmp.y * 0.5 + 0.5) * V.stage.clientHeight}px`;
}
const labelAt = new THREE.Vector3();
function placeLabels() {
  labelAt.set(V.W / 2 + 0.16, 0, 0);
  for (const Ly of V.layers) {
    place(Ly.label, labelAt.clone().applyMatrix4(Ly.g.matrixWorld));
    Ly.label.classList.toggle("is-hot", V.hover === Ly.l || V.sep === Ly.l);
    Ly.label.style.opacity = V.sep === null || V.sep === Ly.l ? "" : "0.35";
  }
  const st = V.data?.steps[cur().t];
  V.ioIn.hidden = V.ioOut.hidden = !st;
  if (!st) return;
  V.ioIn.replaceChildren(h("small", {}, "reads"), st.reads == null ? "your message" : quoteTok(st.reads));
  V.ioOut.replaceChildren(h("small", {}, "says"), sayText(st));
  place(V.ioIn, new THREE.Vector3(0, V.y0 - 0.25, 0).applyMatrix4(V.scene.matrixWorld));
  place(V.ioOut, new THREE.Vector3(0, V.y1 + 0.25, 0));
}
function sayText(st) {
  if (st.says == null) return st.end === "stop" ? "(stops)" : "…";
  return `${quoteTok(st.says)}${st.prob != null ? `  ${Math.round(st.prob * 100)}%` : ""}`;
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
  });
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

function pickNeuron() {
  const hits = ray.intersectObject(V.layers[V.sep].points, false);
  if (!hits.length) return null;
  let best = hits[0];
  for (const x of hits) if (x.distanceToRay < best.distanceToRay) best = x;
  return best.index;
}

function updateHover() {
  V.hover = V.hoverNeuron = null;
  if (!V.pointerIn) {
    V.tip.hidden = true;
    return;
  }
  ray.setFromCamera(V.pointer, V.camera);
  if (V.sep !== null) V.hoverNeuron = pickNeuron();
  const hit = ray.intersectObjects(V.plates, false)[0];
  if (hit) V.hover = hit.object.userData.layer;
  V.canvas.style.cursor = V.hover !== null || V.hoverNeuron !== null ? "pointer" : "";
  let text = null;
  if (V.hoverNeuron !== null) {
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
  V.camAnim = { pos: pos.clone().addScaledVector(B, 4.4), target: pos.clone() };
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
  if (V.savedView) V.camAnim = V.savedView;
  V.savedView = null;
}

function deselect() {
  if (!V) return;
  V.selected = null;
  V.ring.removeFromParent();
  V.tl.colourBy(null);
}

async function openNeuron(l, n, x, y) {
  const reply = S.lastReply();
  const d = V.data;
  V.selected = { l, n };
  V.ring.position.set(V.local[n * 3], 0.004, V.local[n * 3 + 2]);
  V.layers[l].g.add(V.ring);
  const anchor = h("div", { class: "s3d-anchor", "data-pop-anchor": "" });
  anchor.style.left = `${x}px`;
  anchor.style.top = `${y}px`;
  document.body.append(anchor);

  let info = null;
  if (reply && d && !d.live && d.replyId === reply.id) info = await api.neuron(reply.id, l, n).catch(() => null);
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
    onClose: () => {
      anchor.remove();
      if (V.selected && V.selected.l === l && V.selected.n === n) {
        deselect();
        V.escAt = performance.now();
      }
    },
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
    putBack();
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
  const headEl = h("div", { class: "s3d-head", "aria-hidden": "true" });
  const track = h("div", { class: "s3d-track", tabindex: "0", title: "Click or drag to move through the reply" }, steps, headEl);
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
      st.prob != null ? ` (${Math.round(st.prob * 100)}%)` : "",
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
  // light the steps by one neuron's activity (null to clear)
  function colourBy(values) {
    const kids = stepEls();
    const max = values ? Math.max(1e-9, ...values) : 1;
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
    const x = e.clientX - steps.getBoundingClientRect().left;
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

  return { el, rebuild, append, update, head, colourBy, reveal: () => (revealNext = true) };
}
