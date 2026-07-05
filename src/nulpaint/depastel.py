"""De-pastel / "punch" — bake more saturation + contrast into a layer.

Midjourney output tends to be washed-out and pastel; the Nuldrums look wants
bright, fully-saturated colour with real contrast. This applies a two-filter
recipe destructively to one or more layers via the `filter.apply` bridge verb:

  1. levels        — lift the input black point (+ optional white / gamma) so the
                     "milky" lifted blacks snap back to real contrast.
  2. hsvadjustment — global saturation (and optional value) boost.

Per-hue surgical control (FabFilter-style curve over the colour spectrum) is a
separate, interactive step: Filters -> Adjust -> Cross-channel colour
adjustment (driver = Hue, adjusted = Saturation). This command is only the
global base pass.

Runs the levels pass BEFORE saturation so the saturation boost acts on the
already-contrast-restored image.
"""
from __future__ import annotations

from .bridge import BridgeClient


def punch_layer(c: BridgeClient, node: str | None, *, saturation: int = 35,
                value: int = 0, black: int = 18, white: int = 245,
                gamma: float = 1.0) -> dict:
    """Apply the de-pastel recipe to a single layer (name/uuid, or None = active)."""
    target = {"node": node} if node else {}

    # 1. contrast / black-point restore (levels legacy props -> lightness curve).
    if black > 0 or white < 255 or abs(gamma - 1.0) > 1e-6:
        c.call("filter.apply", filter="levels",
               config={"blackvalue": int(black), "whitevalue": int(white),
                       "gammavalue": float(gamma)},
               refresh=False, **target)

    # 2. saturation (+ optional value) boost. type 1 = HSL.
    if saturation != 0 or value != 0:
        c.call("filter.apply", filter="hsvadjustment",
               config={"h": 0, "s": int(saturation), "v": int(value),
                       "type": 1, "colorize": False},
               **target)

    return {"node": node or "(active)", "saturation": saturation, "value": value,
            "black": black, "white": white, "gamma": gamma}
