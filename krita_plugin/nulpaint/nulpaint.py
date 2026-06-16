"""NulPaint — in-Krita half.

Runs inside Krita's embedded PyQt5 interpreter. Opens a loopback socket server
on a background thread and dispatches each command on Krita's GUI thread (the
`krita` API is NOT thread-safe), then writes the JSON result back.

STDLIB + PyQt5 ONLY. This file cannot rely on anything pip-installed — it runs
in Krita's interpreter, not the project venv.

Wire protocol (must match src/nulpaint/config.py):
  request:  {"id": int, "cmd": str, "args": {...}}\n
  response: {"id": int, "ok": bool, "result": any, "error": str|null}\n
"""

import json
import socket
import threading

from krita import Extension, Krita  # type: ignore
from PyQt5.QtCore import QObject, pyqtSignal, Qt

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
        self.job.connect(self._run, Qt.QueuedConnection)

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


COMMANDS = {
    "ping": _cmd_ping,
    "document.info": _cmd_document_info,
    "layer.add": _cmd_layer_add,
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
