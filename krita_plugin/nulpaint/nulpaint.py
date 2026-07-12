"""NulPaint — in-Krita half.

Runs inside Krita's embedded Python interpreter. Opens a loopback socket server
on a background thread and dispatches each command on Krita's GUI thread (the
`krita` API is NOT thread-safe), then writes the JSON result back.

STDLIB + Qt bindings ONLY. This file cannot rely on anything pip-installed — it
runs in Krita's interpreter, not the project venv. The Qt6/KF6 fork ships PyQt6
(scoped enums); we import PyQt6 and fall back to PyQt5 for older builds.

Wire protocol (must match src/nulpaint/config.py):
  request:  {"id": int, "cmd": str, "args": {...}}\n
  response: {"id": int, "ok": bool, "result": any, "error": str|null}\n
"""

import atexit
import base64
import json
import os
import re
import socket
import subprocess
import threading
import time

from krita import Extension, Krita  # type: ignore

# Krita's Qt6/KF6 build ships PyQt6 (scoped enums); older Qt5 builds ship PyQt5.
# Import either and normalise the few enum constants we touch.
try:
    from PyQt6.QtCore import QObject, pyqtSignal, Qt  # type: ignore
    _QUEUED = Qt.ConnectionType.QueuedConnection
except ImportError:  # pragma: no cover — Qt5 fallback
    from PyQt5.QtCore import QObject, pyqtSignal, Qt  # type: ignore
    _QUEUED = Qt.QueuedConnection

# Loopback bridge. Each Krita instance serves its own socket so multiple windows
# can run at once, each driven by its own `nulpaint` CLI / Claude. The port comes
# from $NULPAINT_PORT (set it when launching a 2nd instance, or via
# `nulpaint launch --port N`). With no env, the FIRST instance takes the default
# 8765; a second instance whose default is taken auto-picks a free port (see
# _bind_server). An EXPLICIT $NULPAINT_PORT that's already in use is a hard error
# (the client is targeting that exact port, so we must not silently move).
HOST = os.environ.get("NULPAINT_HOST", "127.0.0.1")
PORT = int(os.environ.get("NULPAINT_PORT", "8765"))
PORT_EXPLICIT = bool(os.environ.get("NULPAINT_PORT"))
ENCODING = "utf-8"

# Each running bridge writes <port>.json here so the CLI can discover instances
# (mirrors NULPAINT_INSTANCE_DIR in config.py — this half can't import it).
INSTANCE_DIR = os.path.expanduser(
    os.environ.get("NULPAINT_INSTANCE_DIR", "~/.local/share/nulpaint/instances"))

# Keeps a running synthetic-stroke QTimer alive (would otherwise be GC'd).
_active_stroke = {}


def _pid_alive(pid):
    """True if a process with this pid exists (used to prune stale registry files)."""
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def _prune_instances():
    """Delete registry files whose owning process is gone."""
    try:
        names = os.listdir(INSTANCE_DIR)
    except OSError:
        return
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(INSTANCE_DIR, name)
        try:
            with open(path, encoding=ENCODING) as fh:
                pid = json.load(fh).get("pid")
        except (OSError, ValueError):
            pid = None
        if pid is None or not _pid_alive(pid):
            try:
                os.remove(path)
            except OSError:
                pass


def _instance_path(port):
    return os.path.join(INSTANCE_DIR, "%d.json" % port)


def _register_instance(host, port):
    """Announce this bridge so `nulpaint instances` / the CLI can find it."""
    try:
        os.makedirs(INSTANCE_DIR, exist_ok=True)
        with open(_instance_path(port), "w", encoding=ENCODING) as fh:
            json.dump({"pid": os.getpid(), "host": host, "port": port,
                       "started": int(time.time())}, fh)
    except OSError:
        pass


def _unregister_instance(port):
    try:
        os.remove(_instance_path(port))
    except OSError:
        pass


def _launcher():
    """Resolve the external `nulpaint` CLI (external venv owns cv2/torch)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in (os.path.expanduser("~/.local/bin/nulpaint"),
              os.path.join(here, "nulpaint")):
        if os.path.exists(p):
            return p
    return "nulpaint"


class _GuiDispatcher(QObject):
    """Marshals a (cmd, args, reply-box) job onto the GUI thread."""

    job = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        # QueuedConnection => slot runs on the thread that owns this QObject
        # (the GUI thread, where the Extension is created).
        self.job.connect(self._run, _QUEUED)

    def _run(self, box):
        cmd, args, result_box, done = box
        try:
            result_box["result"] = COMMANDS[cmd](args)
            result_box["ok"] = True
        except Exception as e:  # noqa: BLE001 — report any failure to the client
            result_box["ok"] = False
            result_box["error"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()


# --- command handlers (run on GUI thread) ----------------------------------
def _cmd_ping(_args):
    return "pong"


def _cmd_document_info(_args):
    doc = Krita.instance().activeDocument()
    if doc is None:
        return None
    return {"name": doc.name(), "width": doc.width(), "height": doc.height(),
            "colorModel": doc.colorModel(), "colorDepth": doc.colorDepth()}


def _cmd_layer_add(args):
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = doc.createNode(args.get("name") or "Layer", args.get("type", "paintlayer"))
    doc.rootNode().addChildNode(node, None)
    doc.refreshProjection()
    return {"name": node.name()}


def _cmd_edit_undo(_args):
    # Krita exposes undo via the action system.
    Krita.instance().action("edit_undo").trigger()
    return True


def _cmd_document_create(args):
    """Create a new document and open a view for it.

    args: width, height, name, resolution (ppi), colorModel, colorDepth.
    Dimensions are explicit pixels — the caller maps any named preset (e.g.
    "1080Land") to width/height before sending.
    """
    app = Krita.instance()
    w = int(args.get("width", 1920))
    h = int(args.get("height", 1080))
    name = args.get("name") or "Untitled"
    res = float(args.get("resolution", 72.0))
    model = args.get("colorModel", "RGBA")
    depth = args.get("colorDepth", "U8")
    doc = app.createDocument(w, h, name, model, depth, "", res)
    if doc is None:
        raise RuntimeError("createDocument returned None")
    win = app.activeWindow()
    if win is None:
        raise RuntimeError("no active Krita window to attach a view")
    win.addView(doc)
    doc.setBatchmode(False)
    doc.refreshProjection()
    return {"name": doc.name(), "width": doc.width(), "height": doc.height(),
            "colorModel": doc.colorModel(), "colorDepth": doc.colorDepth()}


def _cmd_shape_draw(args):
    """Draw filled shapes onto a paint layer (created if missing).

    args:
      shape: "ellipse" | "rectangle"            (default "ellipse")
      color: [r, g, b] or [r, g, b, a], 0-255   (default opaque red)
      items: [{x, y, w, h}, ...]                 bounding boxes, in px
      layer: layer name to draw on / create      (default "Shapes")

    Rendered with QPainter into a full-canvas ARGB32 image, then pushed to the
    node via setPixelData. ARGB32's little-endian byte order (B,G,R,A) matches
    Krita's 8-bit RGBA channel order, so no swizzle is needed.
    """
    try:
        from PyQt6.QtGui import QImage, QPainter, QColor
        from PyQt6.QtCore import QByteArray, Qt as _Qt
        _fmt = QImage.Format.Format_ARGB32
        _nopen = _Qt.PenStyle.NoPen
        _aa = QPainter.RenderHint.Antialiasing
    except ImportError:  # pragma: no cover — Qt5 fallback
        from PyQt5.QtGui import QImage, QPainter, QColor
        from PyQt5.QtCore import QByteArray, Qt as _Qt
        _fmt = QImage.Format_ARGB32
        _nopen = _Qt.NoPen
        _aa = QPainter.Antialiasing

    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    w, h = doc.width(), doc.height()

    layer_name = args.get("layer") or "Shapes"
    node = doc.nodeByName(layer_name)
    if node is None:
        node = doc.createNode(layer_name, "paintlayer")
        doc.rootNode().addChildNode(node, None)

    shape = args.get("shape", "ellipse")
    c = list(args.get("color", [255, 0, 0, 255]))
    if len(c) == 3:
        c.append(255)
    items = args.get("items", [])

    img = QImage(w, h, _fmt)
    img.fill(0)  # fully transparent
    p = QPainter(img)
    p.setRenderHint(_aa, True)
    p.setPen(_nopen)
    p.setBrush(QColor(int(c[0]), int(c[1]), int(c[2]), int(c[3])))
    for it in items:
        x, y, iw, ih = int(it["x"]), int(it["y"]), int(it["w"]), int(it["h"])
        if shape == "rectangle":
            p.drawRect(x, y, iw, ih)
        else:
            p.drawEllipse(x, y, iw, ih)
    p.end()

    n_bytes = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    ptr = img.constBits()
    ptr.setsize(n_bytes)
    node.setPixelData(QByteArray(bytes(ptr)), 0, 0, w, h)
    doc.refreshProjection()
    doc.waitForDone()
    return {"layer": node.name(), "shape": shape, "count": len(items)}


def _cmd_document_save(args):
    """Save the active document.

    args:
      path: destination file (extension picks the format, e.g. .kra/.png).
            If omitted, re-saves to the document's existing fileName.
    """
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    path = args.get("path")
    # Force batchmode so export/overwrite dialogs don't block the GUI thread.
    prev_batch = doc.batchmode()
    doc.setBatchmode(True)
    try:
        if path:
            doc.setFileName(path)
            ok = doc.saveAs(path)
        else:
            if not doc.fileName():
                raise RuntimeError("document has no path yet; pass 'path'")
            ok = doc.save()
        doc.waitForDone()
    finally:
        doc.setBatchmode(prev_batch)
    return {"ok": bool(ok), "fileName": doc.fileName(), "modified": doc.modified()}


def _cmd_document_close(args):
    """Close a document — the active one, or the one named args['name'].

    args:
      name: optional document name to close (default: the active document).
      save: save before closing (document must already have a path).
      discard: close WITHOUT saving even if modified.
    A modified doc with neither save nor discard is LEFT OPEN and reported
    {ok:False, unsaved:True} — never a modal "Save?" prompt (it blocks the GUI
    thread + bridge) and never a silent discard of work. We clear the modified
    flag right before close() so close itself can't pop a prompt.
    """
    app = Krita.instance()
    name = args.get("name")
    if name:
        doc = next((d for d in app.documents() if d.name() == name), None)
        if doc is None:
            raise RuntimeError(f"no open document named {name!r}")
    else:
        doc = app.activeDocument()
        if doc is None:
            raise RuntimeError("no active document")
    closed_name = doc.name()
    doc.setBatchmode(True)
    if args.get("save"):
        if not doc.fileName():
            raise RuntimeError("document has no path yet; can't save before close")
        doc.save()
        doc.waitForDone()
    elif doc.modified() and not args.get("discard"):
        return {"ok": False, "unsaved": True, "name": closed_name}
    doc.setModified(False)          # nothing to prompt about -> close won't block
    ok = doc.close()
    return {"ok": bool(ok), "closed": closed_name}


def _cmd_document_resize(args):
    """Resize the ACTIVE document to width x height (pixels).

    args:
      width, height: target size in pixels (required).
      mode: "scale" (default) resamples the whole image to the new size;
            "canvas" changes the canvas bounds only (anchor top-left,
            crop/extend) and never resamples pixels.
      filter: scale strategy when mode == "scale" (default "Bicubic").
    Returns the resulting name/size/mode.
    """
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    w = int(args.get("width", 0))
    h = int(args.get("height", 0))
    if w < 1 or h < 1:
        raise RuntimeError("width and height must be positive pixels")
    mode = (args.get("mode") or "scale").lower()
    if mode == "canvas":
        doc.resizeImage(0, 0, w, h)          # change bounds, keep pixel scale
    else:
        res = int(round(doc.resolution())) or 72
        doc.scaleImage(w, h, res, res, args.get("filter", "Bicubic"))
    doc.refreshProjection()
    doc.waitForDone()
    return {"name": doc.name(), "width": doc.width(), "height": doc.height(),
            "mode": mode}


def _cmd_grab_canvas(args):
    """Grab the canvas widget (incl. live tool decorations/preview) to a PNG.

    Unlike document.save this captures the on-screen canvas overlay, so it can
    show transient previews (e.g. the Flash Smooth in-progress line).
    """
    canvas = _find_canvas_widget()
    if canvas is None:
        raise RuntimeError("canvas widget not found")
    path = args.get("path", "/tmp/nulpaint-canvas.png")
    pix = canvas.grab()
    ok = pix.save(path)
    return {"ok": bool(ok), "path": path, "size": [pix.width(), pix.height()]}


def _cmd_list_windows(_args):
    """List visible top-level widgets — handy for spotting blocking dialogs."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover
        from PyQt5.QtWidgets import QApplication
    out = []
    for w in QApplication.topLevelWidgets():
        if w.isVisible():
            out.append({"class": w.metaObject().className(),
                        "title": w.windowTitle(), "modal": bool(w.isModal())})
    return out


def _cmd_close_dialogs(args):
    """Dismiss visible top-level dialogs (e.g. the autosave-recovery prompt).

    args: accept=bool — True clicks the default/accept button, False (default)
    rejects/cancels. Returns the titles/classes of what was closed.
    """
    try:
        from PyQt6.QtWidgets import QApplication, QDialog
    except ImportError:  # pragma: no cover
        from PyQt5.QtWidgets import QApplication, QDialog
    accept = bool(args.get("accept", False))
    closed = []
    for w in list(QApplication.topLevelWidgets()):
        if isinstance(w, QDialog) and w.isVisible():
            closed.append(w.windowTitle() or w.metaObject().className())
            (w.accept if accept else w.reject)()
    return {"closed": closed, "accept": accept}


def _find_canvas_widget():
    """The KisOpenGLCanvas2/KisQPainterCanvas widget — the input event receiver."""
    try:
        from PyQt6.QtWidgets import QApplication, QWidget
    except ImportError:  # pragma: no cover
        from PyQt5.QtWidgets import QApplication, QWidget
    win = Krita.instance().activeWindow()
    qwin = win.qwindow() if win else None
    roots = [qwin] if qwin is not None else list(QApplication.topLevelWidgets())
    found = []
    for root in roots:
        if root is None:
            continue
        for w in root.findChildren(QWidget):
            cn = w.metaObject().className()
            if "Canvas" in cn and ("OpenGL" in cn or "QPainter" in cn) \
                    and w.isVisible() and w.width() > 100 and w.height() > 100:
                found.append(w)
    found.sort(key=lambda w: w.width() * w.height(), reverse=True)
    return found[0] if found else None


def _gen_jitter_arc(w, h, n):
    """A smooth left→right arch with deterministic high-frequency jitter on top.
    Good smoothing should erase the jitter while keeping the arch."""
    import math
    x0, x1 = int(w * 0.12), int(w * 0.88)
    midy = int(h * 0.55)
    amp = min(160, int(h * 0.18))
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + (x1 - x0) * t
        arc = amp * math.sin(t * math.pi)                 # the signal: a clean arch
        jit = 13 * ((i % 2) * 2 - 1) + 9 * math.sin(i * 0.9)  # the noise (deterministic)
        pts.append((x, midy - arc + jit))
    return pts


def _cmd_brush_stroke(args):
    """Inject a freehand brush stroke through Krita's real tool + smoothing path.

    Posts synthetic mouse events (non-synthesized source, so the input manager
    won't eat them) along a jittery arc, paced by a QTimer so the stabilizer has
    real time to track. Returns immediately; the stroke finishes asynchronously.
    args: points [[x,y]...] in canvas-widget px (optional), n, interval_ms.
    """
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QMouseEvent, QFocusEvent
        from PyQt6.QtCore import Qt, QPointF, QEvent, QTimer
        _press = QEvent.Type.MouseButtonPress
        _move = QEvent.Type.MouseMove
        _release = QEvent.Type.MouseButtonRelease
        _focusin = QEvent.Type.FocusIn
        _otherreason = Qt.FocusReason.OtherFocusReason
        _LB, _NB = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        _nomod = Qt.KeyboardModifier.NoModifier
    except ImportError:  # pragma: no cover — Qt5 fallback
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QMouseEvent, QFocusEvent
        from PyQt5.QtCore import Qt, QPointF, QEvent, QTimer
        _press, _move, _release = (QEvent.MouseButtonPress, QEvent.MouseMove,
                                   QEvent.MouseButtonRelease)
        _focusin = QEvent.FocusIn
        _otherreason = Qt.OtherFocusReason
        _LB, _NB, _nomod = Qt.LeftButton, Qt.NoButton, Qt.NoModifier

    app = Krita.instance()
    win = app.activeWindow()
    doc = app.activeDocument()
    view = win.activeView() if win is not None else None
    if doc is None or view is None:
        raise RuntimeError("need an active document and view")

    diag = {}
    act = app.action("KritaShape/KisToolBrush")
    diag["tool_action_found"] = act is not None
    if act is not None:
        act.trigger()

    # Foreground -> opaque black so the stroke is visible on a white canvas.
    # Reuse the view's own colour object so the colour space matches.
    try:
        fg = view.foregroundColor()
        comps = fg.components()
        comps = [0.0] * len(comps)
        if comps:
            comps[-1] = 1.0  # alpha is the last channel
        fg.setComponents(comps)
        view.setForeGroundColor(fg)
        diag["fg_components"] = comps
    except Exception as e:  # noqa: BLE001
        diag["fg_error"] = f"{type(e).__name__}: {e}"

    # Ensure a brush preset is active.
    try:
        cur = view.currentBrushPreset()
        if cur is None:
            presets = app.resources("preset")
            name = next((k for k in presets if "Basic" in k), None) or \
                (next(iter(presets)) if presets else None)
            if name:
                view.setCurrentBrushPreset(presets[name])
                cur = view.currentBrushPreset()
        diag["preset"] = cur.name() if cur is not None else None
    except Exception as e:  # noqa: BLE001
        diag["preset_error"] = f"{type(e).__name__}: {e}"

    try:
        node = doc.activeNode()
        diag["active_node"] = node.name() if node is not None else None
    except Exception:  # noqa: BLE001
        pass

    canvas = _find_canvas_widget()
    if canvas is None:
        raise RuntimeError("canvas widget not found")
    w, h = canvas.width(), canvas.height()
    pts = args.get("points") or _gen_jitter_arc(w, h, int(args.get("n", 140)))
    interval = int(args.get("interval_ms", 12))

    def mk(kind, x, y, button, buttons):
        p = QPointF(float(x), float(y))
        return QMouseEvent(kind, p, p, button, buttons, _nomod)

    # The input manager only binds to a canvas on FocusIn. We launched without
    # focus, so the canvas was never bound — synthesize a FocusIn (a non-mouse
    # reason, else KisInputManager "eats" the first stroke) to bind it without
    # actually stealing OS focus.
    QApplication.sendEvent(canvas, QFocusEvent(_focusin, _otherreason))

    # Synchronous delivery through the input-manager event filter.
    QApplication.sendEvent(canvas, mk(_press, pts[0][0], pts[0][1], _LB, _LB))
    state = {"i": 1}
    timer = QTimer()

    def tick():
        i = state["i"]
        if i < len(pts):
            x, y = pts[i]
            QApplication.sendEvent(canvas, mk(_move, x, y, _NB, _LB))
            state["i"] = i + 1
        else:
            timer.stop()
            xe, ye = pts[-1]
            QApplication.sendEvent(canvas, mk(_release, xe, ye, _LB, _NB))
            doc.refreshProjection()
            _active_stroke.pop("timer", None)

    timer.timeout.connect(tick)
    timer.setInterval(interval)
    timer.start()
    _active_stroke["timer"] = timer
    diag.update({"canvas": [w, h], "points": len(pts), "interval_ms": interval,
                 "duration_ms": interval * len(pts)})
    return diag


# --- inpaint I/O (pixels in/out for the external sd.cpp orchestrator) --------
# The external half pulls a layer region + the selection mask, runs the
# diffusion inpaint, composites, then writes the region back. These four
# commands are the only Krita-side surface that needs; everything model-related
# lives outside Krita. All transfer images as base64 PNG over the JSON wire.
def _qt_imaging():
    """The QtGui/QtCore image classes we need — PyQt6 with a PyQt5 fallback.
    The QIODevice write-mode flag's enum path differs across the two bindings."""
    try:
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
        return (QImage, QByteArray, QBuffer, QIODevice.OpenModeFlag.WriteOnly,
                QImage.Format.Format_ARGB32, QImage.Format.Format_Grayscale8)
    except ImportError:  # pragma: no cover — Qt5 fallback
        from PyQt5.QtGui import QImage
        from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
        return (QImage, QByteArray, QBuffer, QIODevice.WriteOnly,
                QImage.Format_ARGB32, QImage.Format_Grayscale8)


def _png_b64(img, QBuffer, write_only):
    """Encode a QImage to a base64 PNG string."""
    buf = QBuffer()
    buf.open(write_only)
    img.save(buf, "PNG")
    data = bytes(buf.data())
    buf.close()
    return base64.b64encode(data).decode("ascii")


def _require_rgba8(doc):
    """Pixel I/O below assumes 8-bit RGBA byte order (ARGB32 ⇆ BGRA). Guard it."""
    if doc.colorModel() != "RGBA" or doc.colorDepth() != "U8":
        raise RuntimeError("inpaint v1 needs an 8-bit RGBA document; got "
                           f"{doc.colorModel()}/{doc.colorDepth()}")


def _target_node(doc, name=None):
    node = doc.nodeByName(name) if name else doc.activeNode()
    if node is None:
        raise RuntimeError(f"layer not found: {name!r}" if name else "no active layer")
    return node


def _cmd_selection_info(_args):
    """Active selection's bounding box in canvas px, or the full canvas if none."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    sel = doc.selection()
    if sel is None:
        return {"hasSelection": False,
                "bounds": {"x": 0, "y": 0, "w": doc.width(), "h": doc.height()}}
    return {"hasSelection": True,
            "bounds": {"x": sel.x(), "y": sel.y(), "w": sel.width(), "h": sel.height()}}


def _cmd_layer_get_region(args):
    """Base64 PNG (RGBA) of a layer region — the inpaint context image.
    args: x, y, w, h (canvas px); layer (name, default = active node)."""
    QImage, _QBA, QBuffer, write_only, fmt_argb, _ = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    _require_rgba8(doc)
    node = _target_node(doc, args.get("layer"))
    x, y, w, h = int(args["x"]), int(args["y"]), int(args["w"]), int(args["h"])
    raw = bytes(node.pixelData(x, y, w, h))          # BGRA, tightly packed
    img = QImage(raw, w, h, fmt_argb).copy()         # copy() detaches from raw
    return {"x": x, "y": y, "w": w, "h": h,
            "png_b64": _png_b64(img, QBuffer, write_only)}


def _cmd_selection_mask_region(args):
    """Base64 PNG (grayscale, white = inpaint here) of the selection over a region.
    No selection ⇒ all-white mask (whole region editable). args: x, y, w, h."""
    QImage, _QBA, QBuffer, write_only, _, fmt_gray = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    x, y, w, h = int(args["x"]), int(args["y"]), int(args["w"]), int(args["h"])
    sel = doc.selection()
    if sel is None:
        img = QImage(w, h, fmt_gray)
        img.fill(255)
    else:
        raw = bytes(sel.pixelData(x, y, w, h))       # 1 byte/px, 0..255
        img = QImage(raw, w, h, w, fmt_gray).copy()
    return {"x": x, "y": y, "w": w, "h": h,
            "png_b64": _png_b64(img, QBuffer, write_only)}


def _cmd_layer_set_region(args):
    """Write an RGBA image (base64 PNG) onto a layer at an offset — the result.
    args: x, y, png_b64; layer (name, default active). w/h come from the image."""
    QImage, QByteArray, _QBuf, _wo, fmt_argb, _ = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    _require_rgba8(doc)
    node = _target_node(doc, args.get("layer"))
    x, y = int(args["x"]), int(args["y"])
    img = QImage()
    if not img.loadFromData(base64.b64decode(args["png_b64"]), "PNG"):
        raise RuntimeError("could not decode PNG payload")
    img = img.convertToFormat(fmt_argb)
    w, h = img.width(), img.height()
    n = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    ptr = img.constBits()
    ptr.setsize(n)
    node.setPixelData(QByteArray(bytes(ptr)), x, y, w, h)
    doc.refreshProjection()
    doc.waitForDone()
    return {"layer": node.name(), "x": x, "y": y, "w": w, "h": h}


def _cmd_layer_list(_args):
    """List layers top-to-bottom (name, type, depth) — for reference-image pickers."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    out = []

    def walk(node, depth):
        for child in reversed(node.childNodes()):   # childNodes() is bottom-up; show top first
            out.append({"name": child.name(), "type": child.type(), "depth": depth})
            if child.childNodes():
                walk(child, depth + 1)

    walk(doc.rootNode(), 0)
    return {"layers": out}


def _cmd_layer_add_image(args):
    """Create a paint layer from an RGBA image (base64 PNG) and place it BELOW the
    active layer (default) or on top — used to drop a cloud generation in for review.
    args: png_b64; name; place ('below_active'|'top'); x, y (offset, default 0)."""
    QImage, QByteArray, _QBuf, _wo, fmt_argb, _ = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    _require_rgba8(doc)
    img = QImage()
    if not img.loadFromData(base64.b64decode(args["png_b64"]), "PNG"):
        raise RuntimeError("could not decode PNG payload")
    img = img.convertToFormat(fmt_argb)
    w, h = img.width(), img.height()

    node = doc.createNode(args.get("name") or "nulpaint gen", "paintlayer")
    active = doc.activeNode()
    parent = (active.parentNode() if active else None) or doc.rootNode()
    above = None
    if args.get("place", "below_active") == "below_active" and active is not None:
        # childNodes() and Node.index() share ordering (0 = bottom). The sibling
        # directly below active is at index-1; inserting our node ABOVE that sibling
        # lands it directly below active. (active at the bottom -> top fallback.)
        sibs = parent.childNodes()
        ai = active.index()
        if 0 < ai <= len(sibs):
            above = sibs[ai - 1]
    parent.addChildNode(node, above)

    x, y = int(args.get("x", 0)), int(args.get("y", 0))
    n = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    ptr = img.constBits()
    ptr.setsize(n)
    node.setPixelData(QByteArray(bytes(ptr)), x, y, w, h)
    doc.refreshProjection()
    doc.waitForDone()
    return {"layer": node.name(), "x": x, "y": y, "w": w, "h": h}


# --- subject-select (mask from a mattemodel/segmodel service -> selection) ---
def _cmd_image_get(_args):
    """Base64 PNG of the merged visible image (projection) — the matte input."""
    QImage, _QBA, QBuffer, write_only, _fa, _fg = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    w, h = doc.width(), doc.height()
    img = doc.projection(0, 0, w, h)
    if img is None or img.isNull():
        raise RuntimeError("projection unavailable")
    return {"w": w, "h": h, "png_b64": _png_b64(img, QBuffer, write_only)}


def _cmd_image_extend(args):
    """Grow the canvas for outpainting; existing content keeps its pixels, the new
    border is transparent. args: left, top, right, bottom (px). Returns new size."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    l, t = int(args.get("left", 0)), int(args.get("top", 0))
    r, b = int(args.get("right", 0)), int(args.get("bottom", 0))
    w, h = doc.width(), doc.height()
    # resizeImage(x, y, w, h): the new image's top-left sits at (x, y) in current
    # coords, so (-l, -t) shifts existing content to (l, t) and adds the border.
    doc.resizeImage(-l, -t, w + l + r, h + t + b)
    doc.refreshProjection()
    doc.waitForDone()
    return {"width": doc.width(), "height": doc.height(),
            "offset_x": l, "offset_y": t, "orig_w": w, "orig_h": h}


def _cmd_selection_set_from_mask(args):
    """Set the document selection from a grayscale mask (base64 PNG, white=selected).
    args: png_b64; x, y (offset, default 0). Mask size = the image's size."""
    from krita import Selection  # type: ignore
    QImage, QByteArray, _QBuf, _wo, _fa, fmt_gray = _qt_imaging()
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    img = QImage()
    if not img.loadFromData(base64.b64decode(args["png_b64"]), "PNG"):
        raise RuntimeError("could not decode mask PNG")
    img = img.convertToFormat(fmt_gray)
    w, h = img.width(), img.height()
    x, y = int(args.get("x", 0)), int(args.get("y", 0))

    # Tightly pack to w*h bytes (Grayscale8 scanlines are 4-byte aligned).
    n = img.sizeInBytes() if hasattr(img, "sizeInBytes") else img.byteCount()
    bits = img.constBits()
    bits.setsize(n)
    buf = bytes(bits)
    bpl = img.bytesPerLine()
    packed = buf if bpl == w else b"".join(buf[r * bpl:r * bpl + w] for r in range(h))

    sel = Selection()
    sel.setPixelData(QByteArray(packed), x, y, w, h)
    doc.setSelection(sel)
    # NB: deliberately NO doc.refreshProjection() here. A selection is an overlay,
    # not image pixels — refreshing the whole projection triggers a storm of partial
    # canvas repaints that erases the just-drawn marching-ants overlay (manual
    # selection tools never refresh the projection). setSelection already does the
    # right repaint (outline cache + notifySelectionChanged).
    return {"x": x, "y": y, "w": w, "h": h}


def _walk_nodes(node):
    """Depth-first walk of every descendant node."""
    for child in node.childNodes():
        yield child
        yield from _walk_nodes(child)


def _node_uuid(node):
    try:
        return node.uniqueId().toString()
    except Exception:
        return ""


def _cmd_vector_list(args):
    """List vector (shape) layers, optionally filtered by a name substring,
    reporting each node's uuid and current shape count. Used to locate a
    specific layer among duplicate-named ones."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    needle = (args.get("name_contains") or "").lower()
    out = []
    for node in _walk_nodes(doc.rootNode()):
        if node.type() != "vectorlayer":
            continue
        name = node.name()
        if needle and needle not in name.lower():
            continue
        try:
            count = len(node.shapes())
        except Exception:
            count = -1
        out.append({"name": name, "uuid": _node_uuid(node), "shapes": count})
    return {"layers": out}


def _cmd_vector_add_svg(args):
    """Inject SVG shapes into a vector layer identified by uuid (preferred) or
    exact name. `svg` is a full SVG document string (same form Krita stores in a
    shape layer's content.svg). Returns shape counts before/after."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    svg = args.get("svg")
    if not svg:
        raise RuntimeError("missing 'svg'")
    want_uuid = args.get("uuid")
    want_name = args.get("name")

    target = None
    matches = []
    for node in _walk_nodes(doc.rootNode()):
        if node.type() != "vectorlayer":
            continue
        if want_uuid:
            if _node_uuid(node) == want_uuid:
                target = node
                break
        elif want_name is not None and node.name() == want_name:
            matches.append(node)
    if target is None and want_uuid is None:
        if len(matches) == 1:
            target = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                "name '%s' matched %d vector layers; pass a uuid (use vector.list)"
                % (want_name, len(matches)))
    if target is None:
        raise RuntimeError("no vector layer found for uuid/name given")

    try:
        before = len(target.shapes())
    except Exception:
        before = -1
    target.addShapesFromSvg(svg)
    try:
        after = len(target.shapes())
    except Exception:
        after = -1
    doc.refreshProjection()
    return {"name": target.name(), "uuid": _node_uuid(target),
            "shapes_before": before, "shapes_after": after}


# --- node tree / visibility / text editing / export -------------------------
# Used by the stream-schedule automation skill. Nodes are addressed by either
# `uuid` (stable across text edits) or `path` (list of names from the document
# root, e.g. ["WeekdayPanels", "Online", "Wednesday"]) which disambiguates the
# many duplicate-named day groups.

def _resolve_node(doc, args):
    want_uuid = args.get("uuid")
    if want_uuid:
        for n in _walk_nodes(doc.rootNode()):
            if _node_uuid(n) == want_uuid:
                return n
        return None
    path = args.get("path")
    if path:
        cur = doc.rootNode()
        for name in path:
            nxt = None
            for c in cur.childNodes():
                if c.name() == name:
                    nxt = c
                    break
            if nxt is None:
                return None
            cur = nxt
        return cur
    return None


def _resolve_ident(doc, ident):
    """Resolve a single identifier — a uuid (braces optional) or an exact layer
    name (first match) — to a node. None if not found / empty."""
    if not ident:
        return None
    ident = str(ident)
    norm = ident.strip("{}").lower()
    for n in _walk_nodes(doc.rootNode()):
        u = _node_uuid(n)
        if u == ident or u.strip("{}").lower() == norm:
            return n
    for n in _walk_nodes(doc.rootNode()):
        if n.name() == ident:
            return n
    return None


def _resolve_target(doc, args):
    """Target node for the management commands: uuid/path (via _resolve_node) or a
    `node` field holding a uuid|name (via _resolve_ident)."""
    n = _resolve_node(doc, args)
    if n is not None:
        return n
    return _resolve_ident(doc, args.get("node"))


def _layer_text(node):
    """Concatenated tspan text of a vector layer (None for non-vector nodes)."""
    if node.type() != "vectorlayer":
        return None
    try:
        svg = node.toSvg()
    except Exception:
        return None
    return "".join(re.findall(r'>([^<]*)</tspan>', svg)).strip()


def _xml_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _node_to_dict(node, depth, maxdepth, want_text):
    d = {"name": node.name(), "uuid": _node_uuid(node),
         "type": node.type(), "visible": bool(node.visible())}
    if want_text and node.type() == "vectorlayer":
        d["text"] = _layer_text(node)
    if depth < maxdepth:
        d["children"] = [_node_to_dict(c, depth + 1, maxdepth, want_text)
                         for c in node.childNodes()]
    return d


def _cmd_node_tree(args):
    """Nested layer tree {name,uuid,type,visible,text?,children}. Optional
    `uuid`/`path` to start from a subtree, `depth` cap, `text` toggle."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    maxdepth = int(args.get("depth", 99))
    want_text = args.get("text", True)
    if args.get("uuid") or args.get("path"):
        start = _resolve_node(doc, args)
        if start is None:
            raise RuntimeError("start node not found")
        roots = [start]
    else:
        roots = list(doc.rootNode().childNodes())
    return {"tree": [_node_to_dict(n, 0, maxdepth, want_text) for n in roots]}


def _cmd_node_set_visible(args):
    """Show/hide a node by uuid or path. Used to flip a day's Online/Offline
    group and to switch between the main and Twitch panel sets."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_node(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    node.setVisible(bool(args.get("visible", True)))
    if args.get("refresh", True):
        doc.refreshProjection()
    return {"uuid": _node_uuid(node), "name": node.name(),
            "visible": bool(node.visible())}


# --- real-time layer/group management ---------------------------------------
# Nodes are addressed by uuid (preferred, stable) or name. A "true" move keeps the
# node object (blend mode, opacity, masks, styles) — it re-parents, never copies.

def _cmd_node_move(args):
    """Move/reparent a node live. Target: uuid|path|node(name|uuid). Optional:
    parent (group name|uuid; default keeps current parent), above (sit directly
    above this sibling), below (sit directly below this sibling). With neither
    above/below the node goes to the TOP of the parent."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found: %r" % (args.get("node") or args.get("uuid") or args.get("path")))
    new_parent = (_resolve_ident(doc, args.get("parent")) if args.get("parent")
                  else (node.parentNode() or doc.rootNode()))
    if new_parent is None:
        raise RuntimeError("parent not found: %r" % args.get("parent"))
    old_parent = node.parentNode() or doc.rootNode()
    old_parent.removeChildNode(node)               # detach FIRST; wrapper keeps it alive.
    # Resolve anchors + read siblings AFTER the detach: `node` is no longer in the
    # tree, so it can never resolve to its own anchor. addChildNode(node, above=node)
    # silently DROPS the layer when `above` isn't a current child of new_parent, so a
    # stale self-anchor (e.g. chained reorders) would otherwise lose it entirely.
    sibs = list(new_parent.childNodes())           # index 0 = bottom; `node` excluded
    above = _resolve_ident(doc, args.get("above")) if args.get("above") else None
    if above is None and args.get("below"):
        below = _resolve_ident(doc, args.get("below"))
        if below is not None:
            bi = next((i for i, s in enumerate(sibs)
                       if _node_uuid(s) == _node_uuid(below)), -1)
            above = sibs[bi - 1] if bi > 0 else None   # sit directly below the anchor
    elif above is None and not args.get("below"):
        # No anchor: default to the TOP of the parent (per this command's contract).
        # NB Krita's addChildNode(node, None) appends at the BOTTOM (index 0), so we
        # must explicitly anchor above the current top sibling to land on top.
        above = sibs[-1] if sibs else None
    new_parent.addChildNode(node, above)           # re-attach, preserving all props
    doc.refreshProjection()
    doc.waitForDone()
    return {"name": node.name(), "uuid": _node_uuid(node),
            "parent": new_parent.name(), "parent_uuid": _node_uuid(new_parent)}


def _cmd_node_create_group(args):
    """Create a group layer. args: name; parent (name|uuid, default root); above
    (sibling name|uuid). Returns the new group's uuid."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    group = doc.createNode(args.get("name") or "Group", "grouplayer")
    parent = (_resolve_ident(doc, args.get("parent")) if args.get("parent")
              else doc.rootNode())
    if parent is None:
        raise RuntimeError("parent not found: %r" % args.get("parent"))
    above = _resolve_ident(doc, args.get("above")) if args.get("above") else None
    parent.addChildNode(group, above)
    doc.refreshProjection()
    return {"name": group.name(), "uuid": _node_uuid(group), "parent": parent.name()}


def _cmd_node_delete(args):
    """Delete a node (and its children). Target via uuid|path|node(name|uuid)."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    name, uid = node.name(), _node_uuid(node)
    parent = node.parentNode() or doc.rootNode()
    parent.removeChildNode(node)
    doc.refreshProjection()
    return {"deleted": name, "uuid": uid}


def _cmd_node_rename(args):
    """Rename a node. Target via uuid|path|node; new name in 'new_name'."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    new = args.get("new_name") or args.get("to")
    if not new:
        raise RuntimeError("missing 'new_name'")
    node.setName(new)
    return {"name": node.name(), "uuid": _node_uuid(node)}


def _cmd_node_set_active(args):
    """Set the active node (so subsequent active-relative ops target it)."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    doc.setActiveNode(node)
    return {"active": node.name(), "uuid": _node_uuid(node)}


def _cmd_text_set(args):
    """Replace the text of a single-line vector text layer (by uuid or path),
    preserving font/style/transform. Round-trips the layer's own SVG and swaps
    only the tspan content, then refreshes."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_node(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    if node.type() != "vectorlayer":
        raise RuntimeError("not a vector layer: %s" % node.type())
    new_text = args.get("text")
    if new_text is None:
        raise RuntimeError("missing 'text'")
    svg = node.toSvg()
    tspans = re.findall(r'>([^<]*)</tspan>', svg)
    if len(tspans) == 0:
        raise RuntimeError("layer has no text shape")
    if len(tspans) != 1:
        raise RuntimeError("layer has %d text spans; only single-line supported"
                           % len(tspans))
    old = tspans[0].strip()
    new_svg = re.sub(r'(>)[^<]*(</tspan>)',
                     lambda m: m.group(1) + _xml_escape(new_text) + m.group(2),
                     svg, count=1)
    for sh in node.shapes():
        sh.remove()
    node.addShapesFromSvg(new_svg)
    if args.get("refresh", True):
        doc.refreshProjection()
    return {"uuid": _node_uuid(node), "name": node.name(),
            "old": old, "new": new_text}


# --- first-class text verbs (size / font / colour / position) ----------------
# So callers manipulate a text layer without hand-rolling SVG surgery. All work
# by round-tripping the layer's own toSvg() and rewriting in place (same proven
# mechanism as text.set), so untouched glyphs/kerning/per-run styling survive.

_TEXT_STYLE_PROPS = {                 # arg name -> SVG property
    "font_size": "font-size",
    "font_family": "font-family",
    "fill": "fill",
}


def _fmt(v):
    """Compact number formatting (drop trailing zeros)."""
    return ("%.4f" % float(v)).rstrip("0").rstrip(".")


def _parse_text_transform(svg):
    """(scale, tx, ty) from the <text> transform. Handles matrix(sx 0 0 sy e f)
    and translate(x, y); defaults to (1, 0, 0)."""
    m = re.search(r'<text[^>]*transform="([^"]*)"', svg)
    if not m:
        return 1.0, 0.0, 0.0
    t = m.group(1)
    mm = re.search(r'matrix\(([\-0-9.eE]+) 0 0 [\-0-9.eE]+ ([\-0-9.eE]+) ([\-0-9.eE]+)\)', t)
    if mm:
        return float(mm.group(1)), float(mm.group(2)), float(mm.group(3))
    tm = re.search(r'translate\(([\-0-9.eE]+),?\s*([\-0-9.eE]+)\)', t)
    if tm:
        return 1.0, float(tm.group(1)), float(tm.group(2))
    return 1.0, 0.0, 0.0


def _set_text_transform(svg, scale, tx, ty):
    """Rewrite the <text> transform to place the shape at (tx,ty) with uniform
    `scale`. translate() when scale==1 (how Krita serializes it), else matrix()."""
    if abs(float(scale) - 1.0) < 1e-9:
        new = "translate(%s, %s)" % (_fmt(tx), _fmt(ty))
    else:
        new = "matrix(%s 0 0 %s %s %s)" % (_fmt(scale), _fmt(scale), _fmt(tx), _fmt(ty))
    return re.subn(r'(<text[^>]*transform=")[^"]*"',
                   lambda m: m.group(1) + new + '"', svg, count=1)


def _text_props(node):
    """Structured read of a text vector layer: text, font_size, font_family,
    fill, anchor, align, scale, x, y."""
    svg = node.toSvg()
    def style(key):
        m = re.search(r'%s:\s*([^;"\']+)' % re.escape(key), svg)
        return m.group(1).strip() if m else None
    def attr(key):
        m = re.search(r'\b%s="([^"]*)"' % re.escape(key), svg)
        return m.group(1) if m else None
    scale, tx, ty = _parse_text_transform(svg)
    fs = style("font-size")
    return {
        "text": "".join(re.findall(r'>([^<]*)</tspan>', svg)).strip(),
        "font_size": float(fs) if fs else None,
        "font_family": style("font-family"),
        "fill": attr("fill"),
        "anchor": attr("text-anchor"),
        "align": style("text-align"),
        "scale": scale, "x": tx, "y": ty,
    }


def _apply_text_props(node, props):
    """Rewrite `node`'s SVG applying any of font_size/font_family/fill/text/
    scale/x/y/translate present (non-None) in `props`. Returns (new_svg, applied)."""
    svg = node.toSvg()
    applied = {}
    for key, prop in _TEXT_STYLE_PROPS.items():        # set every occurrence
        if props.get(key) is not None:
            svg, _ = _svg_set_prop(svg, prop, props[key])
            applied[key] = props[key]
    if props.get("text") is not None:
        svg = re.sub(r'(>)[^<]*(</tspan>)',
                     lambda m: m.group(1) + _xml_escape(str(props["text"])) + m.group(2),
                     svg, count=1)
        applied["text"] = props["text"]
    if any(props.get(k) is not None for k in ("scale", "x", "y", "translate")):
        cur_scale, cur_tx, cur_ty = _parse_text_transform(svg)
        scale = props.get("scale", cur_scale)
        tr = props.get("translate")
        if tr and len(tr) == 2:
            tx, ty = float(tr[0]), float(tr[1])
        else:
            tx = float(props["x"]) if props.get("x") is not None else cur_tx
            ty = float(props["y"]) if props.get("y") is not None else cur_ty
        svg, _ = _set_text_transform(svg, float(scale), tx, ty)
        applied.update(scale=float(scale), x=tx, y=ty)
    return svg, applied


def _write_text_shapes(node, new_svg, doc, refresh=True):
    for sh in node.shapes():
        sh.remove()
    node.addShapesFromSvg(new_svg)
    if refresh:
        doc.refreshProjection()


def _cmd_text_get(args):
    """Read a text vector layer's props (uuid|name|path|node): text, font_size,
    font_family, fill, anchor, align, scale, x, y."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    if node.type() != "vectorlayer":
        raise RuntimeError("not a vector layer: %s" % node.type())
    props = _text_props(node)
    props.update(uuid=_node_uuid(node), name=node.name())
    return props


def _cmd_text_set_props(args):
    """Set any of font_size (pt), font_family, fill, text, scale (uniform mult),
    x/y or translate=[x,y] on ONE text vector layer (uuid|name|path|node),
    preserving everything else."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    if node.type() != "vectorlayer":
        raise RuntimeError("not a vector layer: %s" % node.type())
    new_svg, applied = _apply_text_props(node, args)
    if not applied:
        raise RuntimeError("no text props given "
                           "(font_size/font_family/fill/text/scale/x/y)")
    _write_text_shapes(node, new_svg, doc, args.get("refresh", True))
    return {"uuid": _node_uuid(node), "name": node.name(), "applied": applied}


def _cmd_text_copy_props(args):
    """Copy text props from a source layer onto many targets. args: src
    (uuid|name|path|node); dst (list of uuid/name strings) and/or dst_name
    (EVERY vector layer with this exact name); props (which to copy; default
    ['font_size','font_family','fill','scale'] — position + text NOT copied so
    each target keeps its own). keep_position=False also copies x/y."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    src = _resolve_target(doc, args)
    if src is None or src.type() != "vectorlayer":
        raise RuntimeError("source is not a vector text layer")
    sp = _text_props(src)
    which = list(args.get("props") or ["font_size", "font_family", "fill", "scale"])
    if not args.get("keep_position", True):
        which += ["x", "y"]
    payload = {k: sp[k] for k in which if sp.get(k) is not None}

    targets, seen = [], set()
    for ident in (args.get("dst") or []):
        n = _resolve_ident(doc, ident)
        if n is not None and _node_uuid(n) not in seen:
            targets.append(n); seen.add(_node_uuid(n))
    dn = args.get("dst_name")
    if dn:
        for n in _walk_nodes(doc.rootNode()):
            if (n.type() == "vectorlayer" and n.name() == dn
                    and _node_uuid(n) != _node_uuid(src)
                    and _node_uuid(n) not in seen):
                targets.append(n); seen.add(_node_uuid(n))
    if not targets:
        raise RuntimeError("no targets (pass dst=[...] and/or dst_name)")

    results = []
    for n in targets:
        if n.type() != "vectorlayer":
            continue
        new_svg, applied = _apply_text_props(n, payload)
        _write_text_shapes(n, new_svg, doc, refresh=False)
        results.append({"uuid": _node_uuid(n), "name": n.name(), "applied": applied})
    doc.refreshProjection()
    return {"source": _node_uuid(src), "copied": payload, "targets": results}


def _activate_document_view(doc):
    """Bring `doc`'s view/tab to the front so activeDocument() follows it.

    Krita's `setActiveDocument()` sets the active-document POINTER but does NOT raise
    the MDI subwindow when several docs are open, so activeDocument() keeps returning
    whatever view still has focus. We raise the QMdiSubWindow that shows `doc`, which
    is what actually switches the active view. Returns how it resolved (for debugging).
    """
    try:
        from PyQt6.QtWidgets import QMdiArea  # type: ignore
    except ImportError:  # pragma: no cover — Qt5 fallback
        from PyQt5.QtWidgets import QMdiArea  # type: ignore
    app = Krita.instance()
    win = app.activeWindow()
    if win is None:
        app.setActiveDocument(doc)
        return "no-window"
    qwin = win.qwindow() if hasattr(win, "qwindow") else None
    mdi = qwin.findChild(QMdiArea) if qwin is not None else None
    if mdi is None:
        app.setActiveDocument(doc)
        return "no-mdi"

    want = os.path.abspath(doc.fileName() or "")
    wname = doc.name() or ""

    def _is_target(ad):
        if ad is None:
            return False
        try:
            if want and os.path.abspath(ad.fileName() or "") == want:
                return True
        except Exception:
            pass
        return ad is doc or (bool(wname) and ad.name() == wname)

    subs = mdi.subWindowList()
    # 1) Fast path: match the subwindow by title (no view churn), then verify.
    base = os.path.basename(want) if want else wname
    if base:
        for sub in subs:
            if base in sub.windowTitle():
                mdi.setActiveSubWindow(sub)
                app.setActiveDocument(doc)
                if _is_target(app.activeDocument()):
                    return "activated-by-title"
    # 2) Fallback: activate each subwindow and check which one yields `doc`.
    original = mdi.activeSubWindow()
    for sub in subs:
        mdi.setActiveSubWindow(sub)
        if _is_target(app.activeDocument()):
            app.setActiveDocument(doc)
            return "activated-by-probe"
    # Nothing matched: restore the original view, best-effort set the pointer.
    if original is not None:
        mdi.setActiveSubWindow(original)
    app.setActiveDocument(doc)
    return "pointer-only"


def _find_open_document(path=None, name=None):
    """The open Document matching an absolute `path` (by fileName) or `name`, else None."""
    app = Krita.instance()
    if path:
        p = os.path.abspath(os.path.expanduser(path))
        for d in app.documents():
            try:
                if os.path.abspath(d.fileName() or "") == p:
                    return d
            except Exception:
                continue
    if name:
        for d in app.documents():
            if d.name() == name:
                return d
    return None


def _cmd_document_open(args):
    """Open a document file and show it in a view/tab. args: path; set_active
    (default True). If the file is already open, activates (raises the tab of) that
    doc instead of opening a duplicate. Returns the doc's info (+ reused flag)."""
    path = args.get("path")
    if not path:
        raise RuntimeError("missing 'path'")
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise RuntimeError("no such file: %s" % path)
    app = Krita.instance()
    doc = _find_open_document(path=path)          # already open? -> reuse
    reused = doc is not None
    activated = None
    if doc is None:
        doc = app.openDocument(path)
        if doc is None:
            raise RuntimeError("failed to open: %s" % path)
        win = app.activeWindow()
        if win is not None:
            win.addView(doc)                     # show it in a tab (activates it)
    if args.get("set_active", True):
        # setActiveDocument alone won't raise an already-open doc's tab; do it right.
        activated = _activate_document_view(doc)
    return {"name": doc.name(), "fileName": doc.fileName(),
            "width": doc.width(), "height": doc.height(), "reused": reused,
            "activated": activated}


def _cmd_document_activate(args):
    """Raise an already-open document's tab/view to the front (switch the active
    document). args: path OR name. Errors if no open doc matches."""
    doc = _find_open_document(path=args.get("path"), name=args.get("name"))
    if doc is None:
        raise RuntimeError("no open document matching path/name: %r"
                           % (args.get("path") or args.get("name")))
    how = _activate_document_view(doc)
    return {"name": doc.name(), "fileName": doc.fileName(),
            "width": doc.width(), "height": doc.height(), "activated": how}


def _cmd_document_export_png(args):
    """Export the merged image to a PNG path (does not change the doc's URL)."""
    from krita import InfoObject  # type: ignore
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    path = args.get("path")
    if not path:
        raise RuntimeError("missing 'path'")
    doc.refreshProjection()
    doc.waitForDone()
    # Force batchmode so exportImage doesn't pop the interactive PNG-options
    # dialog (which blocks the GUI thread and hangs the bridge).
    prev_batch = doc.batchmode()
    doc.setBatchmode(True)
    try:
        cfg = InfoObject()
        cfg.setProperty("compression", 3)
        cfg.setProperty("alpha", False)
        ok = doc.exportImage(path, cfg)
    finally:
        doc.setBatchmode(prev_batch)
    return {"ok": bool(ok), "path": path}


# --- animation frame import / timeline scrub --------------------------------
# libkis can ONLY author the animation timeline via Document.importAnimation, which
# needs a GUI main window (it segfaults headless). So these run in the live-but-
# bridge-driven Krita: the whole per-character doc build is automated, no manual GUI.
def _cmd_document_import_animation(args):
    """Import image files as animation frames onto a NEW animated paint layer.

    args: files (list of abs paths, imported in the given order), first_frame
    (default 0), step (default 1), name (rename the created layer), fps (set the
    document fps). Returns the new layer's uuid/name + frame count.
    importAnimation adds one paint layer to the image root; we diff the root's
    children to find it (it isn't reliably the active node)."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    files = args.get("files") or []
    if not files:
        raise RuntimeError("no files to import")
    first_frame = int(args.get("first_frame", 0))
    step = int(args.get("step", 1))

    before = {_node_uuid(n) for n in doc.rootNode().childNodes()}
    prev_batch = doc.batchmode()
    doc.setBatchmode(True)                      # no import progress dialog
    try:
        ok = doc.importAnimation(list(files), first_frame, step)
    finally:
        doc.setBatchmode(prev_batch)
    if not ok:
        raise RuntimeError("importAnimation failed")

    new = [n for n in doc.rootNode().childNodes() if _node_uuid(n) not in before]
    node = new[-1] if new else doc.activeNode()
    if node is None:
        raise RuntimeError("could not locate the imported animation layer")
    if args.get("name"):
        node.setName(args["name"])
    if args.get("fps"):
        doc.setFramesPerSecond(int(args["fps"]))
    doc.setActiveNode(node)
    doc.refreshProjection()
    return {"uuid": _node_uuid(node), "name": node.name(),
            "frames": len(files), "animated": bool(node.animated())}


def _cmd_document_frame_info(_args):
    """Timeline state: current time, fps, and the full clip range."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    return {"currentTime": doc.currentTime(), "fps": doc.framesPerSecond(),
            "clipStart": doc.fullClipRangeStartTime(),
            "clipEnd": doc.fullClipRangeEndTime()}


def _cmd_document_set_frame(args):
    """Scrub the timeline: set the document's current time. args: time."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    doc.setCurrentTime(int(args["time"]))
    doc.refreshProjection()
    doc.waitForDone()
    return {"time": doc.currentTime()}


# --- filters (destructive apply of any registered Krita filter) --------------
# General primitive: apply a named Krita filter (by its registry id, e.g.
# "hsvadjustment", "levels", "crosschannel") with a property dict onto a node,
# baking the result into the layer's pixels. Config props are set through the
# filter's OWN default configuration object, so filter-specific property logic
# (e.g. the levels legacy blackvalue/whitevalue/gammavalue -> lightness-curve
# conversion) fires correctly. Region defaults to the whole document.
def _cmd_node_duplicate(args):
    """Duplicate a node and insert the copy directly above its source, keeping
    the same parent group and all layer properties. args: uuid|name|path to
    address the source, optional `name` for the copy (default "<src> nulpaint"),
    `set_active` (default True). Returns the new node's uuid/name."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    src = _resolve_target(doc, args)
    if src is None:
        raise RuntimeError("node not found")
    dup = src.duplicate()
    if dup is None:
        raise RuntimeError("duplicate failed for: %s" % src.name())
    # duplicate() copies an embedded layer style WITH THE SAME resource UUID, which
    # leaves two layers claiming one style id (a "Duplicated UUID for styles" load
    # warning + can confuse the projection). Round-trip the style through ASL so the
    # copy gets a fresh KisPSDLayerStyle (new uuid). No-op for unstyled layers.
    try:
        asl = dup.layerStyleToAsl()
        if asl:
            dup.setLayerStyleFromAsl(asl)
    except Exception:
        pass
    dup.setName(args.get("name") or (src.name() + " nulpaint"))
    parent = src.parentNode() or doc.rootNode()
    parent.addChildNode(dup, src)          # insert directly above the source
    if args.get("set_active", True):
        doc.setActiveNode(dup)
    if args.get("refresh", True):
        doc.refreshProjection()
    return {"uuid": _node_uuid(dup), "name": dup.name(),
            "source": _node_uuid(src)}


def _cmd_filter_read_config(args):
    """Read the filter id + configuration properties off a filter MASK or filter
    LAYER node (addressed by uuid|name|path). Returns {filter, config} where config
    can be fed straight back into filter.apply (or filter.add_mask) to reproduce it
    exactly — e.g. copying a hand-tuned Cross-channel curve onto other layers/docs."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    getf = getattr(node, "filter", None)
    if getf is None:
        raise RuntimeError("node has no filter (type=%s); need a filter mask/layer"
                           % node.type())
    flt = getf()
    if flt is None:
        raise RuntimeError("no filter on node %s" % node.name())
    cfg = flt.configuration()
    raw = cfg.properties()
    props = {}
    for k in (raw.keys() if hasattr(raw, "keys") else raw):
        v = raw[k]
        if not isinstance(v, (str, int, float, bool)) and v is not None:
            v = str(v)                    # keep JSON-safe (curves are strings already)
        props[k] = v
    return {"filter": flt.name(), "config": props, "node": node.name()}


def _cmd_filter_apply(args):
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None and not (args.get("uuid") or args.get("path") or args.get("node")):
        node = doc.activeNode()          # no identifier given -> active layer
    if node is None:
        raise RuntimeError("node not found")
    fid = args.get("filter")
    if not fid:
        raise RuntimeError("missing 'filter' (registry id)")
    flt = Krita.instance().filter(fid)
    if flt is None:
        raise RuntimeError("unknown filter id: %s" % fid)
    cfg = flt.configuration()
    config = args.get("config") or {}
    # Some configs are order-sensitive: multichannel/cross-channel ignore curveN
    # unless nTransfers (the channel count) is set FIRST. Apply it before the rest.
    if "nTransfers" in config:
        cfg.setProperty("nTransfers", config["nTransfers"])
    for k, v in config.items():
        if k == "nTransfers":
            continue
        cfg.setProperty(k, v)
    flt.setConfiguration(cfg)

    x = int(args.get("x", 0))
    y = int(args.get("y", 0))
    w = int(args.get("w", doc.width()))
    h = int(args.get("h", doc.height()))
    if node.locked():
        raise RuntimeError("node is locked: %s" % node.name())
    ok = flt.apply(node, x, y, w, h)
    if args.get("refresh", True):
        doc.refreshProjection()
        doc.waitForDone()
    return {"ok": bool(ok), "uuid": _node_uuid(node), "name": node.name(),
            "filter": fid}


# --- vector layer FX (stroke/fill/opacity) ----------------------------------
# Read/rewrite the SVG styling of vector layers. `vector.set_style` is the
# general "alter layer fx" verb: it edits presentation properties in place on a
# vector layer, or on every vector layer under a group, round-tripping each
# layer's own SVG so text/font/transform are preserved (same proven mechanism
# as text.set). Properties map to SVG presentation attrs/style props.

_STYLE_PROPS = {
    "stroke_width": "stroke-width",
    "stroke": "stroke",
    "stroke_opacity": "stroke-opacity",
    "fill": "fill",
    "fill_opacity": "fill-opacity",
    "opacity": "opacity",
}


def _vector_layers_in_scope(node):
    """Yield the node itself if it's a vector layer, else every descendant
    vector layer (depth-first)."""
    if node.type() == "vectorlayer":
        yield node
        return
    for c in node.childNodes():
        yield from _vector_layers_in_scope(c)


def _svg_set_prop(svg, prop, value):
    """Set SVG presentation property `prop` to `value` wherever it already
    appears, both as an XML attribute (prop="...") and inside an inline
    style="..." (prop:...). `\\b...\\s*[=:]` keeps 'stroke' from matching
    'stroke-width' etc. Returns (new_svg, n_changes)."""
    value = str(value)
    pat = re.escape(prop)
    svg, c1 = re.subn(r'(\b%s\s*=\s*")[^"]*"' % pat,
                      lambda m: m.group(1) + value + '"', svg)
    svg, c2 = re.subn(r'(\b%s\s*:\s*)[^;"\']*' % pat,
                      lambda m: m.group(1) + value, svg)
    return svg, c1 + c2


def _cmd_vector_get_svg(args):
    """Return the raw SVG of a vector layer (uuid|path|name|node). The read side
    of the vector-FX commands — lets a caller inspect styling before editing."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    if node.type() != "vectorlayer":
        raise RuntimeError("not a vector layer: %s" % node.type())
    return {"uuid": _node_uuid(node), "name": node.name(), "svg": node.toSvg()}


def _cmd_vector_set_svg(args):
    """Replace ALL shapes in a vector layer with the given `svg` (remove +
    addShapesFromSvg). Low-level write primitive behind the FX helpers."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    if node.type() != "vectorlayer":
        raise RuntimeError("not a vector layer: %s" % node.type())
    svg = args.get("svg")
    if not svg:
        raise RuntimeError("missing 'svg'")
    try:
        before = len(node.shapes())
    except Exception:
        before = -1
    for sh in node.shapes():
        sh.remove()
    node.addShapesFromSvg(svg)
    try:
        after = len(node.shapes())
    except Exception:
        after = -1
    if args.get("refresh", True):
        doc.refreshProjection()
    return {"uuid": _node_uuid(node), "name": node.name(),
            "shapes_before": before, "shapes_after": after}


def _cmd_vector_set_style(args):
    """Alter presentation FX on a vector layer, or on every vector layer under a
    group (uuid|path|name|node). Accepts any of: stroke_width, stroke,
    stroke_opacity, fill, fill_opacity, opacity. Paint values are SVG strings
    ('none', '#000000', ...). Rewrites each layer's own SVG so text/font/
    transform are preserved. Reports per-layer change counts."""
    doc = Krita.instance().activeDocument()
    if doc is None:
        raise RuntimeError("no active document")
    node = _resolve_target(doc, args)
    if node is None:
        raise RuntimeError("node not found")
    edits = {prop: args[key] for key, prop in _STYLE_PROPS.items()
             if args.get(key) is not None}
    if not edits:
        raise RuntimeError("no style props given (stroke_width/stroke/fill/...)")
    results = []
    for vl in _vector_layers_in_scope(node):
        svg = vl.toSvg()
        total = 0
        for prop, val in edits.items():
            svg, c = _svg_set_prop(svg, prop, val)
            total += c
        if total:
            for sh in vl.shapes():
                sh.remove()
            vl.addShapesFromSvg(svg)
        results.append({"uuid": _node_uuid(vl), "name": vl.name(),
                        "changes": total})
    if args.get("refresh", True):
        doc.refreshProjection()
    return {"target": node.name(), "edits": edits, "layers": results}


COMMANDS = {
    "ping": _cmd_ping,
    "document.info": _cmd_document_info,
    "document.create": _cmd_document_create,
    "document.save": _cmd_document_save,
    "document.close": _cmd_document_close,
    "document.resize": _cmd_document_resize,
    "layer.add": _cmd_layer_add,
    "shape.draw": _cmd_shape_draw,
    "tool.brush_stroke": _cmd_brush_stroke,
    "app.list_windows": _cmd_list_windows,
    "app.close_dialogs": _cmd_close_dialogs,
    "app.grab_canvas": _cmd_grab_canvas,
    "edit.undo": _cmd_edit_undo,
    "selection.info": _cmd_selection_info,
    "layer.get_region": _cmd_layer_get_region,
    "layer.list": _cmd_layer_list,
    "layer.add_image": _cmd_layer_add_image,
    "selection.mask_region": _cmd_selection_mask_region,
    "layer.set_region": _cmd_layer_set_region,
    "image.get": _cmd_image_get,
    "image.extend": _cmd_image_extend,
    "selection.set_from_mask": _cmd_selection_set_from_mask,
    "vector.list": _cmd_vector_list,
    "vector.add_svg": _cmd_vector_add_svg,
    "vector.get_svg": _cmd_vector_get_svg,
    "vector.set_svg": _cmd_vector_set_svg,
    "vector.set_style": _cmd_vector_set_style,
    "node.tree": _cmd_node_tree,
    "node.set_visible": _cmd_node_set_visible,
    "node.move": _cmd_node_move,
    "node.create_group": _cmd_node_create_group,
    "node.delete": _cmd_node_delete,
    "node.rename": _cmd_node_rename,
    "node.set_active": _cmd_node_set_active,
    "text.set": _cmd_text_set,
    "text.get": _cmd_text_get,
    "text.set_props": _cmd_text_set_props,
    "text.copy_props": _cmd_text_copy_props,
    "document.export_png": _cmd_document_export_png,
    "document.open": _cmd_document_open,
    "document.activate": _cmd_document_activate,
    "document.import_animation": _cmd_document_import_animation,
    "document.frame_info": _cmd_document_frame_info,
    "document.set_frame": _cmd_document_set_frame,
    "filter.apply": _cmd_filter_apply,
    "filter.read_config": _cmd_filter_read_config,
    "node.duplicate": _cmd_node_duplicate,
}


class NulPaintExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self._dispatcher = _GuiDispatcher()
        self._server = None
        self._thread = None
        self._port = PORT

    def setup(self):
        self._start_server()

    def createActions(self, window):
        # "Green-eat Selection": recolour leftover green edge fringe inside the
        # current lasso from the nearest clean colour (alpha untouched). The math
        # needs cv2, which lives in the external venv — so we shell the `nulpaint`
        # CLI, which connects BACK over this socket to pull/write pixels. It MUST
        # be non-blocking (Popen, no wait): the CLI's get/set calls are serviced
        # on this same GUI thread, so blocking here would deadlock.
        act = window.createAction("nulpaint_green_eat", "Green-eat Selection (despill)",
                                  "tools/scripts")
        act.triggered.connect(self._green_eat)

    def _green_eat(self):
        try:
            subprocess.Popen([_launcher(), "despill-selection"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001 — never let a UI action raise into Krita
            pass

    # -- socket server ------------------------------------------------------
    def _start_server(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, name="nulpaint-bridge", daemon=True)
        self._thread.start()

    def _bind_server(self):
        """Bind the bridge socket, returning (socket, port).

        Default port taken by another instance -> fall back to a free ephemeral
        port so a second Krita window still gets a bridge. An EXPLICIT
        $NULPAINT_PORT that's busy raises (the client targets that exact port)."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((HOST, PORT))
        except OSError:
            if PORT_EXPLICIT:
                srv.close()
                raise
            srv.bind((HOST, 0))  # 0 => let the OS pick a free port
        srv.listen(1)
        return srv, srv.getsockname()[1]

    def _serve(self):
        _prune_instances()
        try:
            srv, port = self._bind_server()
        except OSError as e:  # explicit port conflict — surface it, don't serve
            print("nulpaint: bridge could not bind %s:%d (%s) — "
                  "pick another NULPAINT_PORT" % (HOST, PORT, e))
            return
        self._server = srv
        self._port = port
        _register_instance(HOST, port)
        atexit.register(_unregister_instance, port)
        print("nulpaint: bridge listening on %s:%d (pid %d)" % (HOST, port, os.getpid()))
        while True:
            conn, _addr = srv.accept()
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn, conn.makefile("r", encoding=ENCODING) as rfile:
            for line in rfile:
                line = line.strip()
                if not line:
                    continue
                resp = self._dispatch(line)
                conn.sendall((json.dumps(resp) + "\n").encode(ENCODING))

    def _dispatch(self, line):
        try:
            req = json.loads(line)
            cmd = req["cmd"]
        except Exception as e:  # noqa: BLE001
            return {"id": None, "ok": False, "result": None, "error": f"bad request: {e}"}
        if cmd not in COMMANDS:
            return {"id": req.get("id"), "ok": False, "result": None,
                    "error": f"unknown command: {cmd}"}
        # Hand the job to the GUI thread and block this socket thread for it.
        result_box, done = {"ok": False, "result": None, "error": None}, threading.Event()
        self._dispatcher.job.emit((cmd, req.get("args", {}), result_box, done))
        # Most commands are quick; animation-frame import loads hundreds of PNGs on
        # the GUI thread, so allow a generous ceiling before the socket gives up.
        done.wait(timeout=600)
        return {"id": req.get("id"), "ok": result_box["ok"],
                "result": result_box.get("result"), "error": result_box.get("error")}
