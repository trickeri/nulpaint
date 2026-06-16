"""Shared configuration + wire protocol constants.

This module is the ONE place the in-Krita plugin and the external process
agree on. The plugin half copies the protocol constants verbatim (it can't
import this package), so keep them dead simple and stable.
"""

from __future__ import annotations

import os

# --- Bridge transport -------------------------------------------------------
# Loopback only. The in-Krita server binds here; the external client connects.
BRIDGE_HOST = os.environ.get("NULPAINT_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("NULPAINT_PORT", "8765"))

# --- Wire protocol ----------------------------------------------------------
# Newline-delimited JSON. One request object per line, one response per line.
#   request:  {"id": <int>, "cmd": <str>, "args": {<...>}}
#   response: {"id": <int>, "ok": <bool>, "result": <any>, "error": <str|null>}
PROTOCOL_VERSION = 1
ENCODING = "utf-8"
