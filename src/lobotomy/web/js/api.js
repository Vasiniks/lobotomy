// Thin client for the lobotomy JSON API (see src/lobotomy/server.py).

async function json(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// POST that streams newline-delimited JSON events; calls onEvent for each.
async function stream(path, body, onEvent) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let last = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.type === "error") throw new Error(ev.message);
      last = ev;
      onEvent(ev);
    }
  }
  return last;
}

export const api = {
  state: () => json("GET", "/api/state"),
  settings: (s) => json("PUT", "/api/settings", s),

  chat: (text, onEvent) => stream("/api/chat", { text }, onEvent),
  again: (opts, onEvent) => stream("/api/again", opts, onEvent),
  cancel: () => json("POST", "/api/cancel"),
  resetChat: () => json("POST", "/api/chat/reset"),

  addEdit: (target, op = "scale", value = 0) => json("POST", "/api/edits", { target, op, value }),
  addEditFull: (body) => json("POST", "/api/edits", body),
  addEdits: (edits) => json("POST", "/api/edits/batch", { edits }),
  updateEdit: (id, value) => json("PATCH", `/api/edits/${id}`, { value }),
  removeEdit: (id) => json("DELETE", `/api/edits/${id}`),
  resetEdits: () => json("DELETE", "/api/edits"),
  loadEdits: (edits) => json("POST", "/api/edits/load", { edits }),
  presets: () => json("GET", "/api/presets"),

  reply: (id) => json("GET", `/api/replies/${id}`),
  layers: (id, rows) => json("GET", `/api/replies/${id}/layers${rows ? `?rows=${rows}` : ""}`),
  // reduce=max: each neuron's strongest value over the selected tokens (the design's grid)
  layer: (id, layer, rows) => json("GET", `/api/replies/${id}/layer/${layer}?reduce=max${rows ? `&rows=${rows}` : ""}`),
  neuron: (id, layer, n) => json("GET", `/api/replies/${id}/neuron/${layer}/${n}`),
  token: (id, row) => json("GET", `/api/replies/${id}/token/${row}`),
  damage: (id) => json("GET", `/api/replies/${id}/damage`),
  lens: (id) => json("GET", `/api/replies/${id}/lens`),
  flow: (id, k = 96) => json("GET", `/api/replies/${id}/flow?k=${k}`),
  why: (id, row, k = 12) => json("GET", `/api/replies/${id}/why/${row}?k=${k}`),
  scan: (id, params) => json("GET", `/api/replies/${id}/scan?${new URLSearchParams(params)}`),
  tokenize: (text) => json("GET", `/api/tokenize?text=${encodeURIComponent(text)}`),
};
