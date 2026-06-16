"""MCP server exposing Krita tools to an LLM agent.

Thin layer: each MCP tool validates args and forwards to the in-Krita bridge.
Keep the tool surface aligned with the bridge command set so the agent and the
voice grammar can't drift apart.

Run:  nulpaint-mcp        (after `pip install -e '.[mcp]'`)
"""

from __future__ import annotations

from ..bridge import BridgeClient, BridgeError


def build_server():  # -> mcp.server.FastMCP
    """Construct the FastMCP server. Imported lazily so the package stays
    importable without the optional `mcp` dependency installed."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("nulpaint")
    bridge = BridgeClient()

    def _ensure() -> BridgeClient:
        if bridge._sock is None:  # noqa: SLF001 — internal, single-process
            bridge.connect()
        return bridge

    @mcp.tool()
    def document_info() -> dict:
        """Return name/size/colorspace of Krita's active document."""
        return _ensure().call("document.info")

    @mcp.tool()
    def add_layer(name: str = "", layer_type: str = "paint") -> dict:
        """Add a layer to the active document."""
        return _ensure().call("layer.add", name=name, type=layer_type)

    @mcp.tool()
    def undo() -> dict:
        """Undo the last action."""
        return _ensure().call("edit.undo")

    return mcp


def main() -> None:
    try:
        build_server().run()
    except BridgeError as e:
        raise SystemExit(f"nulpaint: bridge error — is Krita running with the plugin enabled? ({e})")


if __name__ == "__main__":
    main()
