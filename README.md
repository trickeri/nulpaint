# NulPaint — Voice-Controlled & Agentic AI Control for Krita

Drive a forked Krita by voice ("new layer", "select subject", "merge down",
"undo") with low-latency commands, plus a slower agentic/LLM path for
natural-language and compound edits — part of pushing Krita toward a
Photoshop-style workflow.

> **Status:** scaffold. Architecture is laid out; phases are stubbed. See
> [`docs/PLAN.md`](docs/PLAN.md) for the build plan and the Phase 0 research
> gate (do that before writing real logic).

Part of the Nuldrums Krita toolchain:

- **[trickeri/krita](https://github.com/trickeri/krita)** — forked Krita
  (GPL-3.0), `nuldrums` branch, with Photoshop-parity patches where the Python
  API can't reach.
- **NulPaint** (this repo) — voice + agent editing control (MIT).

## Why a socket bridge (not D-Bus)

Unlike Kdenlive, Krita has **no D-Bus scripting interface**. Instead it embeds a
**PyQt5 interpreter and runs plugins in-process**. So NulPaint is split in two:

```
┌─────────────────────────── Krita process ───────────────────────────┐
│  krita_plugin/nulpaint/   (pykrita Extension, stdlib-only)           │
│    └─ opens a local socket server, dispatches commands via the       │
│       `krita` (Libkis) API on Krita's GUI thread                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  loopback JSON-line protocol
┌──────────────────────────────┴───────────────────────────────────────┐
│  src/nulpaint/   (external process, full PyPI deps allowed)          │
│    bridge/   socket client → talks to the in-Krita server            │
│    voice/    constrained-grammar ASR → fast command intents          │
│    mcp/      MCP server → exposes Krita tools to an LLM agent         │
│    orchestrator/  routes voice + agent intents to the bridge         │
└───────────────────────────────────────────────────────────────────────┘
```

The in-Krita half must stay **stdlib-only** — it runs inside Krita's
interpreter, where you can't assume external PyPI packages are present.

## Install (dev)

```bash
# external half
pip install -e '.[mcp,voice,dev]'

# in-Krita half — symlink the plugin into Krita's resource folder
ln -s "$PWD/krita_plugin/nulpaint"        ~/.local/share/krita/pykrita/nulpaint
ln -s "$PWD/krita_plugin/nulpaint.desktop" ~/.local/share/krita/pykrita/nulpaint.desktop
# then enable: Krita → Settings → Configure Krita → Python Plugin Manager → NulPaint
```

## Layout

| Path | Runs where | Deps |
|------|-----------|------|
| `krita_plugin/nulpaint/` | inside Krita (PyQt5) | stdlib only |
| `src/nulpaint/bridge/`   | external | stdlib |
| `src/nulpaint/voice/`    | external | vosk, sounddevice |
| `src/nulpaint/mcp/`      | external | mcp |
| `src/nulpaint/orchestrator/` | external | — |
