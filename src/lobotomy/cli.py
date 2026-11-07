"""Terminal interface: chat with the model and break it with slash commands.

    lobotomy chat                  # interactive
    lobotomy serve                 # web UI (see server.py)
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import analysis as A
from .edits import Target
from .sampling import SamplingParams

DEFAULT_MODEL = Path("models/Qwen3.5-0.8B-Q4_K_M.gguf")

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
    /layers                    activity per layer
    /top L [K]                 most active neurons in layer L
    /heads L                   head activity in layer L
    /lens [I]                  logit lens at reply token I (default: last)
    /says L N                  tokens neuron N in layer L pushes up and down

  Settings
    /temp X | /greedy | /seed N | /max N | /think on|off | /system TEXT
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
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        p.add_argument("--backend", default=None, help="mlx or cuda (default: mlx on Apple Silicon)")
        p.add_argument("--dtype", default="bfloat16")
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args(["chat", *(argv or sys.argv[1:])])

    if not args.model.exists():
        sys.exit(
            f"model not found: {args.model}\n"
            "download it with: scripts/download_model.sh  (or pass --model PATH)"
        )

    from .engine import Engine

    engine = Engine(args.model, backend=args.backend, dtype=args.dtype, on_progress=lambda m: print(f"… {m}", flush=True))
    if args.cmd == "serve":
        from .server import serve

        serve(engine, host=args.host, port=args.port)
    else:
        Repl(engine).run()


if __name__ == "__main__":
    main()
