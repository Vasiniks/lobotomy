"""Terminal interface: chat with the model and break it with slash commands.

    lobotomy chat                  # interactive
    lobotomy serve                 # web UI (see server.py)
    lobotomy download [FILE]       # a Qwen3.5 GGUF from Hugging Face into models/
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from pathlib import Path

from . import analysis as A
from .edits import Target
from .sampling import THINK_MAX_TOKENS, SamplingParams

DEFAULT_MODEL = Path("models/Qwen3.5-0.8B-Q4_K_M.gguf")
REPO_ROOT = Path(__file__).resolve().parents[2]  # the checkout, for an install made with `pip install -e .`


def download_model(file: str = DEFAULT_MODEL.name, dest: Path | None = None, attempts: int = 30) -> Path:
    """Download a Qwen3.5 GGUF from Unsloth on Hugging Face, like
    scripts/download_model.sh but in Python, so it also works on Windows.
    A partial file resumes, and a dropped connection retries."""
    import urllib.error
    import urllib.request

    m = re.match(r"^(Qwen3\.5-[0-9.]+B)-", file)
    if not m:
        sys.exit(f"don't know where to download {file!r} from; expected a name like Qwen3.5-2B-Q4_K_M.gguf")
    url = f"https://huggingface.co/unsloth/{m.group(1)}-GGUF/resolve/main/{file}"
    dest = dest or (REPO_ROOT / "models" if (REPO_ROOT / "pyproject.toml").exists() else Path("models"))
    dest.mkdir(parents=True, exist_ok=True)
    out, part = dest / file, dest / f"{file}.part"
    if out.exists():
        print(f"{out} already exists")
        return out
    print(f"downloading {url}")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "lobotomy-downloader"})
        total = 0
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # the server ignored the range: start over
                    have = 0
                # the whole file's size: "Content-Range: bytes a-b/total", or the length of a full reply
                cr = r.headers.get("Content-Range") or ""
                total = int(cr.rsplit("/", 1)[1]) if "/" in cr and cr.rsplit("/", 1)[1].isdigit() else have + int(
                    r.headers.get("Content-Length") or 0
                )
                done, last = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 0.5:
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(f"\r  {done / 1e6:7.1f} MB {pct}", end="", flush=True)
                            last = time.time()
            print()
        except urllib.error.HTTPError as e:
            if e.code != 416:  # 416: the partial file is already complete
                print(f"\ndownload failed ({e}); attempt {attempt} of {attempts}", file=sys.stderr)
                time.sleep(3)
                continue
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            print(f"\ndownload interrupted ({e}); resuming, attempt {attempt} of {attempts}", file=sys.stderr)
            time.sleep(3)
            continue
        # a connection can end early without an error: only a full-size file counts
        size = part.stat().st_size if part.exists() else 0
        if total and size != total:
            if size > total:
                part.unlink()
            print(f"download stopped at {size / 1e6:.1f} of {total / 1e6:.1f} MB; resuming, attempt {attempt} of {attempts}", file=sys.stderr)
            time.sleep(3)
            continue
        part.replace(out)
        print(f"saved {out}")
        return out
    sys.exit(f"couldn't download {file}; run the same command again to resume")

HELP = """\
Type a message to chat. Commands:

  Breaking things (VALUE: 0 = off, 0.5 = turn down, 2 = boost; default 0)
    /neuron L N [VALUE]        MLP neuron N in layer L
    /head L H [VALUE]          head H in layer L
    /layer L [all|mixer|mlp] [VALUE]
    /token TEXT [VALUE]        a token's embedding (also its output weight)
    /noise L [all|mixer|mlp] STD   add noise to a layer (STD relative, e.g. 0.5)
    /topoff L K                switch off layer L's K most active neurons in the last reply
    /edits                     list active edits
    /set ID VALUE              change an edit's value
    /undo [ID]                 undo an edit (the last one by default)
    /reset                     back to healthy (remove all edits)
    /save FILE | /load FILE    save or load edits as a recipe

  Asking again
    /again                     same question, same seed, current weights
    /healthy                   same question, healthy model
    /damage                    how far the last reply's model is from healthy

  Looking inside (about the last reply)
    /scan [layers | heads L | neurons L [K]]   what would cutting each part change?
    /layers                    activity per layer
    /top L [K]                 most active neurons in layer L
    /heads L                   head activity in layer L
    /lens [I]                  logit lens at reply token I (default: last)
    /says L N                  tokens neuron N in layer L pushes up and down

  Settings
    /temp X | /greedy | /seed N | /max N | /think on|off | /system TEXT
    /reread on|off             off = keep memory (old turns stay as read before your edits)
    /new                       start a new chat
    /info                      model info
    /quit
"""


def _value(args: list[str], i: int, default: float = 0.0) -> float:
    return float(args[i]) if len(args) > i else default


class Repl:
    def __init__(self, engine):
        self.eng = engine

    def out(self, *a):
        print(*a, flush=True)

    def stream(self, events):
        reply = None
        for ev in events:
            if ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "done":
                reply = ev["reply"]
        print()
        if reply is not None:
            s = reply.stats
            tag = " [healthy]" if reply.healthy else (f" [{len(reply.edits)} edits]" if reply.edits else "")
            self.out(f"  ({s['reply_tokens']} tokens, {s['tokens_per_second']} tok/s, prompt {s['prompt_tokens']}){tag}")
        return reply

    def reply_rows(self, r):
        return [t["row"] for t in r.recorded_tokens(self.eng.tokenizer) if t["role"] == "reply"]

    def need_reply(self):
        r = self.eng.last_reply()
        if r is None:
            raise ValueError("chat first")
        return r

    def command(self, line: str) -> bool:
        eng = self.eng
        parts = shlex.split(line[1:])
        if not parts:
            return True
        cmd, args = parts[0], parts[1:]
        if cmd in ("quit", "exit", "q"):
            return False
        if cmd == "help":
            self.out(HELP)
        elif cmd == "neuron":
            e = eng.add_edit(Target("neuron", layer=int(args[0]), index=int(args[1])), value=_value(args, 2))
            self.out(f"#{e.id} {e.label()}")
        elif cmd == "head":
            e = eng.add_edit(Target("head", layer=int(args[0]), index=int(args[1])), value=_value(args, 2))
            self.out(f"#{e.id} {e.label()}")
        elif cmd == "layer":
            part = args[1] if len(args) > 1 and args[1] in ("all", "mixer", "mlp") else "all"
            vi = 2 if len(args) > 1 and args[1] in ("all", "mixer", "mlp") else 1
            e = eng.add_edit(Target("layer", layer=int(args[0]), part=part), value=_value(args, vi))
            self.out(f"#{e.id} {e.label()}")
        elif cmd == "noise":
            part = args[1] if len(args) > 2 else "all"
            std = float(args[-1])
            e = eng.add_edit(Target("layer", layer=int(args[0]), part=part), op="noise", value=std)
            self.out(f"#{e.id} {e.label()}")
        elif cmd == "token":
            ids = eng.tokenizer.encode(args[0])
            for tid in ids:
                e = eng.add_edit(Target("token", index=tid), value=_value(args, 1))
                self.out(f"#{e.id} {e.label()} ({eng.tokenizer.token_str(tid)!r})")
        elif cmd == "topoff":
            r = self.need_reply()
            layer, k = int(args[0]), int(args[1])
            for n in A.top_neurons(r.recording, layer, k=k, rows=self.reply_rows(r)):
                eng.add_edit(Target("neuron", layer=layer, index=n["neuron"]), value=0.0)
            self.out(f"switched off {k} neurons in layer {layer}")
        elif cmd == "edits":
            if not eng.edits:
                self.out("no edits (healthy)")
            for e in eng.edits:
                self.out(f"#{e.id} {e.label()}")
        elif cmd == "set":
            e = eng.update_edit(int(args[0]), value=float(args[1]))
            self.out(f"#{e.id} {e.label()}")
        elif cmd == "undo":
            e = eng.undo(int(args[0])) if args else eng.undo_last()
            self.out(f"undid #{e.id} {e.label()}" if e else "nothing to undo")
        elif cmd == "reset":
            n = len(eng.reset_edits())
            self.out(f"removed {n} edits; healthy again")
        elif cmd == "save":
            Path(args[0]).write_text(json.dumps([e.to_dict() for e in eng.edits], indent=2))
            self.out(f"saved {len(eng.edits)} edits to {args[0]}")
        elif cmd == "load":
            edits = eng.load_edits(json.loads(Path(args[0]).read_text()))
            self.out(f"loaded {len(edits)} edits")
        elif cmd == "again":
            self.stream(eng.stream_again())
        elif cmd == "healthy":
            self.stream(eng.stream_again(healthy=True))
        elif cmd == "damage":
            r = self.need_reply()
            d = eng.damage(r)
            self.out(f"mean KL from healthy: {d['mean_kl']:.3f} nats   per token: {A.sparkline(d['kl'], 60)}")
        elif cmd == "scan":
            r = self.need_reply()
            what = args[0] if args else "layers"
            if what == "layers":
                res = eng.scan(r, "layers")
            elif what == "heads":
                res = eng.scan(r, "heads", layer=int(args[1]))
            elif what == "neurons":
                layer = int(args[1])
                cands = [n["neuron"] for n in A.top_neurons(r.recording, layer, k=int(args[2]) if len(args) > 2 else 32, rows=self.reply_rows(r))]
                res = eng.scan(r, "neurons", layer=layer, candidates=cands)
            else:
                raise ValueError("scan layers | heads L | neurons L [K]")
            self.out("cutting each one alone would change this reply by (KL, nats):")
            for x in sorted(res, key=lambda x: -x["mean_kl"])[:12]:
                bar = "█" * int(min(40, x["mean_kl"] * 10))
                self.out(f"  {x['label']:<22} {x['mean_kl']:7.3f}  top-1 kept {x['top1_agree']:.0%} {bar}")
        elif cmd == "layers":
            r = self.need_reply()
            act = A.layer_activity(r.recording, self.reply_rows(r))
            for a in act:
                kind = "attn" if eng.config.is_full_attention(a["layer"]) else "delta"
                bar = "█" * int(min(40, a["mlp_out"] * 4))
                self.out(f"L{a['layer']:>2} {kind:5} mlp {a['mlp_out']:7.2f} mixer {a['mixer_out']:7.2f} {bar}")
        elif cmd == "top":
            r = self.need_reply()
            layer, k = int(args[0]), int(args[1]) if len(args) > 1 else 10
            for n in A.top_neurons(r.recording, layer, k=k, rows=self.reply_rows(r)):
                self.out(f"  neuron {n['neuron']:>4}  activity {n['activity']:.3f}  mean {n['mean_signed']:+.3f}")
        elif cmd == "heads":
            r = self.need_reply()
            layer = int(args[0])
            act = A.layer_activity(r.recording, self.reply_rows(r))[layer]
            for h, v in enumerate(act["heads"]):
                self.out(f"  head {h:>2}  {v:7.3f} {'█' * int(min(40, v * 4))}")
        elif cmd == "lens":
            r = self.need_reply()
            rows = self.reply_rows(r)
            i = int(args[0]) if args else len(rows) - 1
            row = rows[i]
            tok = r.recorded_tokens(eng.tokenizer)[row]["text"]
            self.out(f"logit lens after reading {tok!r}:")
            for layer, top in enumerate(eng.logit_lens(r, row, k=3)):
                self.out(f"  L{layer:>2} " + "  ".join(f"{t!r} {p:.2f}" for t, p in top))
        elif cmd == "says":
            v = eng.neuron_vocab(int(args[0]), int(args[1]))
            self.out("  promotes:   " + " ".join(repr(t) for t in v["promotes"]))
            self.out("  suppresses: " + " ".join(repr(t) for t in v["suppresses"]))
        elif cmd == "temp":
            eng.sampling.temperature = float(args[0])
        elif cmd == "greedy":
            eng.sampling.temperature = 0.0
        elif cmd == "seed":
            eng.sampling.seed = int(args[0])
        elif cmd == "max":
            eng.sampling.max_tokens = int(args[0])
        elif cmd == "think":
            eng.enable_thinking = args[0] == "on"
            if eng.enable_thinking and eng.sampling.max_tokens < THINK_MAX_TOKENS:
                eng.sampling.max_tokens = THINK_MAX_TOKENS
                self.out(f"thinking on; max reply tokens raised to {THINK_MAX_TOKENS} so it can finish its thought (/max N to change)")
        elif cmd == "reread":
            eng.reread = args[0] != "off"
            self.out(
                "re-reading the whole chat with the current weights"
                if eng.reread
                else "keeping memory: earlier turns stay as read under the weights of the time"
            )
        elif cmd == "system":
            eng.system_prompt = " ".join(args) or None
        elif cmd == "new":
            eng.reset_chat()
            self.out("new chat")
        elif cmd == "info":
            i = eng.info()
            self.out(f"{i['name']} on {i['backend']}: {i['n_layers']} layers, {i['n_neurons']} neurons/layer, vocab {i['vocab_size']}")
        else:
            self.out(f"unknown command /{cmd} (try /help)")
        return True

    def run(self):
        self.out("lobotomy: type /help for commands")
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line.startswith("/"):
                try:
                    if not self.command(line):
                        return
                except (ValueError, KeyError, IndexError, RuntimeError) as e:
                    self.out(f"error: {e}")
                continue
            self.stream(self.eng.stream_chat(line))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lobotomy")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("chat", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--model", type=Path, default=None, help=f"a Qwen3.5 GGUF (default: {DEFAULT_MODEL})")
        p.add_argument(
            "--backend",
            default=None,
            help="mlx (Apple Silicon) or torch (an NVIDIA GPU through CUDA, or the CPU); default: mlx on Apple Silicon, torch elsewhere",
        )
        p.add_argument("--device", default=None, help="torch backend only: cuda, cuda:1, cpu, mps (default: cuda if available, else cpu)")
        p.add_argument("--dtype", default=None, help="bfloat16, float16 or float32 (default: bfloat16 on GPUs that support it, float32 on the CPU)")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=7860)
            p.add_argument("--open", action="store_true", help="open the app in the browser once it's ready")
    d = sub.add_parser("download", help="download a Qwen3.5 GGUF from Hugging Face into models/")
    d.add_argument("file", nargs="?", default=DEFAULT_MODEL.name, help=f"e.g. Qwen3.5-2B-Q4_K_M.gguf (default: {DEFAULT_MODEL.name})")
    d.add_argument("--dir", type=Path, default=None, help="where to save it (default: models/ in the checkout)")
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args(["chat", *(argv or sys.argv[1:])])
    if args.cmd == "download":
        download_model(args.file, args.dir)
        return

    if args.model is None:
        # the default model, looked for here and in the checkout this runs
        # from, so `~/lobotomy/.venv/bin/lobotomy serve` works from any folder
        candidates = [DEFAULT_MODEL, REPO_ROOT / DEFAULT_MODEL]
        args.model = next((p for p in candidates if p.exists()), DEFAULT_MODEL)
    if not args.model.exists():
        script = REPO_ROOT / "scripts" / "download_model.sh"
        sys.exit(
            f"model not found: {args.model}\n"
            f"download it with: {script if script.exists() else 'scripts/download_model.sh'}  (or pass --model PATH)"
        )

    from .engine import Engine

    engine = Engine(
        args.model, backend=args.backend, dtype=args.dtype, device=args.device, on_progress=lambda m: print(f"… {m}", flush=True)
    )
    if args.cmd == "serve":
        from .server import serve

        serve(engine, host=args.host, port=args.port, open_browser=args.open)
    else:
        Repl(engine).run()


if __name__ == "__main__":
    main()
