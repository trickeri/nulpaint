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

import base64
import json
import re
import socket
import threading

from krita import Extension, Krita  # type: ignore

# Krita's Qt6/KF6 build ships PyQt6 (scoped enums); older Qt5 builds ship PyQt5.
# Import either and normalise the few enum constants we touch.
try:
    from PyQt6.QtCore import QObject, pyqtSignal, Qt  # type: ignore
    _QUEUED = Qt.ConnectionType.QueuedConnection
except ImportError:  # pragma: no cover — Qt5 fallback
    from PyQt5.QtCore import QObject, pyqtSignal, Qt  # type: ignore
    _QUEUED = Qt.QueuedConnection

HOST = "127.0.0.1"
PORT = 8765
ENCODING = "utf-8"

# Keeps a running synthetic-stroke QTimer alive (would otherwise be GC'd).
_active_stroke = {}


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
    above = _resolve_ident(doc, args.get("above")) if args.get("above") else None
    if above is None and args.get("below"):
        below = _resolve_ident(doc, args.get("below"))
        if below is not None:
            sibs = list(new_parent.childNodes())   # index 0 = bottom
            bi = next((i for i, s in enumerate(sibs)
                       if _node_uuid(s) == _node_uuid(below)), -1)
            if bi > 0:
                above = sibs[bi - 1]               # node above this anchor = below `below`
    old_parent = node.parentNode() or doc.rootNode()
    old_parent.removeChildNode(node)               # detaches; wrapper keeps it alive
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
    "document.export_png": _cmd_document_export_png,
}


class NulPaintExtension(Extension):
    def __init__(self, parent):
        super().__init__(parent)
        self._dispatcher = _GuiDispatcher()
        self._server = None
        self._thread = None

    def setup(self):
        self._start_server()

    def createActions(self, window):
        pass

    # -- socket server ------------------------------------------------------
    def _start_server(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._serve, name="nulpaint-bridge", daemon=True)
        self._thread.start()

    def _serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        self._server = srv
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
        done.wait(timeout=30)
        return {"id": req.get("id"), "ok": result_box["ok"],
                "result": result_box.get("result"), "error": result_box.get("error")}
