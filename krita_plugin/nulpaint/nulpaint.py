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

import json
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
    if path:
        doc.setFileName(path)
        ok = doc.saveAs(path)
    else:
        if not doc.fileName():
            raise RuntimeError("document has no path yet; pass 'path'")
        ok = doc.save()
    doc.waitForDone()
    return {"ok": bool(ok), "fileName": doc.fileName(), "modified": doc.modified()}


COMMANDS = {
    "ping": _cmd_ping,
    "document.info": _cmd_document_info,
    "document.create": _cmd_document_create,
    "document.save": _cmd_document_save,
    "layer.add": _cmd_layer_add,
    "shape.draw": _cmd_shape_draw,
    "edit.undo": _cmd_edit_undo,
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
