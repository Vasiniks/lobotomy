/* lobotomy mockups: the only script. Draws data-heavy parts from data.js
   (strips, the neuron grid, heads) and handles the theme switch.
   Everything else on each page is static HTML. */
(function () {
  "use strict";
  var D = window.LOBO;

  /* ---------- theme ---------- */
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    document.querySelectorAll("[data-theme-toggle]").forEach(function (b) {
      b.textContent = t === "light" ? "Dark theme" : "Light theme";
    });
    try { localStorage.setItem("lobo-theme", t); } catch (e) {}
  }
  var saved = null;
  try { saved = localStorage.getItem("lobo-theme"); } catch (e) {}
  var asked = /[?&#]theme=(light|dark)/.exec(location.search + location.hash);   // e.g. main.html?theme=light
  if (asked) saved = asked[1];
  setTheme(saved === "light" ? "light" : "dark");
  document.addEventListener("click", function (e) {
    var b = e.target.closest("[data-theme-toggle]");
    if (b) setTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
  });

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  var isAttn = function (l) { return l % 4 === 3; };   // layers 3, 7, 11, 15, 19, 23
  function editsFor(layer) { return D.edits.filter(function (e) { return e.layer === layer; }); }
  function describe(e) {
    return (e.kind === "head" ? "head " : "neuron ") + e.index + " " + e.label;
  }

  /* ---------- layer stack ----------
     <div class="strips" data-run="healthy" data-col="5" data-layer="14" data-compact> */
  function renderStrips(root) {
    var run = D[root.dataset.run || "healthy"];
    var col = root.dataset.col != null ? +root.dataset.col : -1;
    var sel = root.dataset.layer != null ? +root.dataset.layer : -1;
    var compact = root.hasAttribute("data-compact");
    var href = root.dataset.href;
    for (var b = 0; b < 6; b++) {
      var block = el("div", "block");
      for (var l = b * 4; l < b * 4 + 4; l++) {
        var strip = el(href ? "a" : "button", "strip" + (isAttn(l) ? " is-attn" : "") + (l === sel ? " is-selected" : ""));
        if (href) strip.href = href; else strip.type = "button";
        var eds = editsFor(l);
        strip.setAttribute("aria-label", "Layer " + l + ", " + (isAttn(l) ? "full attention" : "Gated DeltaNet") +
          ", peak " + run.peaks[l] + (eds.length ? ", edits: " + eds.map(describe).join(", ") : "") + ". Open layer.");
        strip.appendChild(el("span", "strip-idx num", String(l)));
        if (!compact) strip.appendChild(el("span", "strip-type", isAttn(l) ? "Attention" : "DeltaNet"));
        var heat = el("span", "heat");
        var row = run.bins[l];
        for (var t = 0; t < row.length; t++) heat.appendChild(el("i", "act-" + row[t]));
        strip.appendChild(heat);
        if (!compact) strip.appendChild(el("span", "strip-peak num", run.peaks[l].toFixed(run.peaks[l] >= 10 ? 1 : 2)));
        var ed = el("span", "strip-edit");
        eds.forEach(function (e) {
          var s = el("i", "swatch " + e.cls);
          s.title = describe(e);
          ed.appendChild(s);
        });
        strip.appendChild(ed);
        block.appendChild(strip);
      }
      root.appendChild(block);
    }
    if (col >= 0) {
      requestAnimationFrame(function () { placeCursor(root, col); });
      window.addEventListener("resize", function () { placeCursor(root, col); });
    }
  }
  function placeCursor(root, col) {
    var first = root.querySelector(".heat i:nth-child(" + (col + 1) + ")");
    if (!first) return;
    var r0 = root.getBoundingClientRect(), r1 = first.getBoundingClientRect();
    var cur = root.querySelector(".col-cursor") || root.appendChild(el("div", "col-cursor"));
    cur.style.left = (r1.left - r0.left - 2) + "px";
    cur.style.width = (r1.width + 4) + "px";
    var mark = document.querySelector('.axis-mark[data-for="' + (root.id || "") + '"]');
    if (mark) {
      var track = mark.parentElement.getBoundingClientRect();
      mark.style.left = (r1.left - track.left + r1.width / 2) + "px";
    }
  }

  /* ---------- neuron grid ----------
     <div class="ngrid" data-layer="14" data-selected="938"> */
  function renderGrid(root) {
    var L = D["layer" + root.dataset.layer];
    var sel = root.dataset.selected != null ? +root.dataset.selected : -1;
    var edits = {};
    editsFor(+root.dataset.layer).forEach(function (e) { if (e.kind === "neuron") edits[e.index] = e; });
    var frag = document.createDocumentFragment();
    for (var i = 0; i < L.grid.length; i++) {
      var c = el("i", "act-" + L.grid[i]);
      if (edits[i]) { c.classList.add(edits[i].cls); c.title = "Neuron " + i + ", " + edits[i].label; }
      if (i === sel) c.classList.add("is-selected");
      frag.appendChild(c);
    }
    root.appendChild(frag);
    var anchor = document.querySelector('[data-anchor-to="' + sel + '"]');
    if (anchor) {
      var place = function () {
        var cell = root.children[sel];
        var host = anchor.offsetParent || document.body;
        var rc = cell.getBoundingClientRect(), rh = host.getBoundingClientRect();
        var left = rc.left - rh.left - anchor.offsetWidth + 40;
        anchor.style.left = Math.max(0, left) + "px";
        anchor.style.top = (rc.bottom - rh.top + 12) + "px";
      };
      requestAnimationFrame(place);
      window.addEventListener("resize", place);
    }
  }

  /* ---------- heads ----------
     <div class="heads" data-layer="15" data-selected="5"> */
  function renderHeads(root) {
    var layer = +root.dataset.layer;
    var L = D["layer" + layer];
    var edits = {};
    editsFor(layer).forEach(function (e) { if (e.kind === "head") edits[e.index] = e; });
    L.heads.forEach(function (v, i) {
      var cell = el("button", "head-cell");
      cell.type = "button";
      var h = el("span", "head act-" + L.headBins[i]);
      if (edits[i]) h.classList.add(edits[i].cls);
      h.textContent = edits[i] && edits[i].scale === 0 ? "" : v.toFixed(2).replace(/^0/, "");
      cell.setAttribute("aria-label", "Head " + i + ", output " + v.toFixed(2) + (edits[i] ? ", " + edits[i].label : ""));
      cell.appendChild(h);
      cell.appendChild(el("span", "head-idx num", String(i)));
      root.appendChild(cell);
    });
  }

  /* ---------- one token, every layer ----------
     <div class="lbars" data-col="5"> length and brightness are the token's
     column in the stack; the number is the raw peak |activation|. */
  function renderLayerBars(root) {
    var col = +root.dataset.col, run = D.healthy, raw = D.tokenPeaks[col];
    for (var b = 0; b < 6; b++) {
      var block = el("div", "block");
      for (var l = b * 4; l < b * 4 + 4; l++) {
        var bin = +run.bins[l][col];
        var row = el("a", "lbar" + (isAttn(l) ? " is-attn" : ""));
        row.href = "layer.html";
        row.setAttribute("aria-label", "Layer " + l + ", peak " + raw[l] + " on this token. Open layer.");
        row.appendChild(el("span", "strip-idx num", String(l)));
        row.appendChild(el("span", "strip-type", isAttn(l) ? "Attention" : "DeltaNet"));
        var track = el("span", "lbar-track act-" + bin);
        var fill = el("b"); fill.style.width = Math.max(4, (bin / 8) * 100) + "%";
        track.appendChild(fill); row.appendChild(track);
        row.appendChild(el("span", "lbar-val num", raw[l].toFixed(2)));
        var ed = el("span", "strip-edit");
        editsFor(l).forEach(function (e) { var s = el("i", "swatch " + e.cls); s.title = describe(e); ed.appendChild(s); });
        row.appendChild(ed);
        block.appendChild(row);
      }
      root.appendChild(block);
    }
  }

  document.querySelectorAll(".strips").forEach(renderStrips);
  document.querySelectorAll(".lbars").forEach(renderLayerBars);
  document.querySelectorAll(".ngrid").forEach(renderGrid);
  document.querySelectorAll(".heads").forEach(renderHeads);
})();
