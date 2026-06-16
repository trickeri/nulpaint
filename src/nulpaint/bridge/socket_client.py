"""External-side bridge client.

Connects to the NulPaint server running *inside* Krita and issues commands
over the newline-delimited JSON protocol defined in ``nulpaint.config``.

This is deliberately stdlib-only and synchronous — the command set is small
and latency-sensitive, and a blocking round-trip keeps ordering trivial.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

from ..config import BRIDGE_HOST, BRIDGE_PORT, ENCODING


class BridgeError(RuntimeError):
    """Raised when the in-Krita server reports a command failure."""


class BridgeClient:
    def __init__(self, host: str = BRIDGE_HOST, port: int = BRIDGE_PORT, timeout: float = 5.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._lock = threading.Lock()
        self._next_id = 0

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.settimeout(self._timeout)
        self._sock = s
        self._rfile = s.makefile("r", encoding=ENCODING)

    def close(self) -> None:
        if self._rfile is not None:
            self._rfile.close()
        if self._sock is not None:
            self._sock.close()
        self._sock = self._rfile = None

    def __enter__(self) -> "BridgeClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- command path -------------------------------------------------------
    def call(self, cmd: str, **args: Any) -> Any:
        """Send one command, block for its response, return ``result``."""
        if self._sock is None:
            raise BridgeError("not connected; call connect() first")
        with self._lock:
            self._next_id += 1
            req_id = self._next_id
            payload = json.dumps({"id": req_id, "cmd": cmd, "args": args}) + "\n"
            self._sock.sendall(payload.encode(ENCODING))
            line = self._rfile.readline()
        if not line:
            raise BridgeError("connection closed by Krita")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise BridgeError(resp.get("error") or "unknown error")
        return resp.get("result")

    # -- convenience wrappers (grow as the command set firms up) ------------
    def ping(self) -> Any:
        return self.call("ping")

    def active_document(self) -> Any:
        return self.call("document.info")

    def undo(self) -> Any:
        return self.call("edit.undo")
