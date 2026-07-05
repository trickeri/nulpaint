# NulPaint — Build Plan

Voice + agentic control for the forked Krita (`trickeri/krita`, `nuldrums`
branch), in service of a Photoshop-style workflow.

## Phase 0 — Research gate (do before writing real logic)

- [ ] Confirm Krita's Python API surface for the commands we want
      (`layer.*`, `select.*`, `image.*`). Use Scripter to probe live.
- [x] Verify a background thread can `socket.bind` inside Krita without
      tripping its event loop, and that the GUI-thread dispatch
      (`pyqtSignal` + `Qt.QueuedConnection`) actually serializes correctly.
      Confirmed 2026-06-16 — background `socket.listen` + `QueuedConnection`
      dispatch works. NB: the Qt6 fork runs plugins under **PyQt6** (scoped
      enums), not PyQt5.
- [ ] Decide which Photoshop-parity commands are reachable from Python vs.
      which need a C++ patch in the fork (adjustment layers, PS selection
      behavior, layer styles).

## Phase 1 — Bridge MVP

- [x] Wire protocol (newline JSON) — `config.py` + plugin mirror it.
- [x] In-Krita socket server + GUI-thread dispatcher.
- [x] External `BridgeClient`.
- [x] End-to-end smoke test: external `ping` → Krita → `pong` (2026-06-16).
- [x] Flesh out the command table — **38 verbs live** as of 2026-06-30 (full
      enumeration in the `/krita` skill §6.5): document.*, layer.*, node.*,
      selection.*, image.*, vector.*, text.set, tool.brush_stroke, app.*, edit.undo.
      Selections, layer/node management, vector/text, and AI in/out all landed.

## Phase 2 — Fast voice path

- [ ] vosk constrained grammar built from `voice/intent.py::PHRASES`.
- [ ] Push-to-talk capture (sounddevice) → intent → router → bridge.
- [ ] Latency budget: phrase-end → canvas change under ~150 ms.

## Phase 3 — Agentic path

- [ ] `nulpaint-mcp` exposes the bridge command set as MCP tools.
- [ ] Undo-grouping so a compound agent edit is one undo step.
- [ ] Dry-run / confirm mode for destructive ops.

## Open questions

- Port collision / multi-instance Krita → negotiate port, or unix socket.
- How much of "Photoshop parity" lives here vs. the C++ fork — keep a running
  list as we hit API walls.
