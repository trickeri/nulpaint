"""Bridge: external-side client that talks to the in-Krita socket server."""

from .socket_client import BridgeClient, BridgeError

__all__ = ["BridgeClient", "BridgeError"]
