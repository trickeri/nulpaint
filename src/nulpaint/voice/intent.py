"""Map recognized phrases to bridge commands.

Phase 1 is a flat phrase table — deterministic, zero-latency, no model load.
The ASR layer (vosk) constrains its grammar to these keys so recognition is
fast and rarely wrong. Compound / natural-language requests fall through to
the agentic (MCP) path instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Intent:
    cmd: str
    args: dict[str, Any] = field(default_factory=dict)


# Phrase → (bridge command, args). Keep phrases short and acoustically distinct.
PHRASES: dict[str, Intent] = {
    "undo": Intent("edit.undo"),
    "redo": Intent("edit.redo"),
    "new layer": Intent("layer.add", {"type": "paint"}),
    "delete layer": Intent("layer.remove"),
    "merge down": Intent("layer.merge_down"),
    "select subject": Intent("select.subject"),
    "deselect": Intent("select.clear"),
    "invert selection": Intent("select.invert"),
    "flatten": Intent("image.flatten"),
}


def parse_intent(phrase: str) -> Intent | None:
    """Return the Intent for a recognized phrase, or None if unknown."""
    return PHRASES.get(phrase.strip().lower())
