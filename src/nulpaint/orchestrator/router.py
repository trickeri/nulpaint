"""Route intents from the fast (voice) and slow (agent) paths to the bridge.

The fast path sends a known Intent straight through. The agent path produces
the same {cmd, args} shape, so both converge on a single dispatch point —
which also makes it the natural place to hang logging, undo-grouping, and a
dry-run mode later.
"""

from __future__ import annotations

from typing import Any

from ..bridge import BridgeClient
from ..voice.intent import Intent


class Router:
    def __init__(self, bridge: BridgeClient):
        self._bridge = bridge

    def dispatch(self, intent: Intent) -> Any:
        return self._bridge.call(intent.cmd, **intent.args)
