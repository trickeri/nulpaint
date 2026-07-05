"""Green-eater: recolour leftover chroma-green edge fringe by pulling the nearest
CLEAN foreground colour into each green pixel, leaving alpha completely untouched.

Where matte + despill already ran, thin/wispy features still keep a green-tinted
edge ring, and pushing despill harder would erode those skinny parts. This does the
opposite of erosion: it keeps the exact silhouette (alpha bit-for-bit) and only
swaps the RGB under the fringe for the colour of the nearest non-green body pixel —
so a green hair edge becomes a hair-coloured edge, a green boot edge becomes boot-
coloured, etc. Deterministic and temporally stable (it copies stable interior
colour outward), so it barely flickers frame-to-frame.

Driven from the bridge exactly like inpaint: it pulls the active layer's selected
region + the selection mask (the lasso), recolours, and writes the region back. No
model, no daemon — just the four pixel-I/O bridge commands and OpenCV. The lasso
scopes WHERE green is eaten, so characters with intentionally green/teal parts are
safe (only recolour inside the selection).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import cv2
from PIL import Image

from .bridge import BridgeClient


def _b64_to_rgba(s: str) -> np.ndarray:
    im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA")
    return np.asarray(im)                       # HxWx4 uint8, RGBA


def _rgba_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def green_mask(rgba: np.ndarray, thr: int, amin: int) -> np.ndarray:
    """Spill pixels: opaque-ish AND the green channel dominates both R and B."""
    r = rgba[..., 0].astype(np.int16)
    g = rgba[..., 1].astype(np.int16)
    b = rgba[..., 2].astype(np.int16)
    a = rgba[..., 3]
    return (a > amin) & ((g - np.maximum(r, b)) > thr)


def green_eat(rgba: np.ndarray, thr: int = 22, amin: int = 8,
              grow: int = 0, region: np.ndarray | None = None):
    """Recolour green pixels from the nearest clean foreground pixel. Alpha kept.

    thr    green-dominance threshold (higher = only stronger green counts)
    grow   dilate the green mask this many px ALONG the soft edge (alpha<250)
           to catch a faint sub-threshold halo; never grows into the solid body
    region boolean lasso mask limiting WHERE we recolour (None = whole frame)
    Returns (fixed_rgba, green_mask_used).
    """
    out = rgba.copy()
    a = rgba[..., 3]
    gm = green_mask(rgba, thr, amin)
    target = gm.copy()
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        grown = cv2.dilate(target.astype(np.uint8), k).astype(bool)
        # only extend along the anti-aliased edge band, never into the solid interior
        target = grown & (a > amin) & (a < 250)
        target |= gm
    if region is not None:
        target &= region

    clean = (a > amin) & ~gm                     # clean = opaque, not-green (whole frame)
    if not target.any() or not clean.any():
        return out, target

    # Exact nearest CLEAN pixel for every pixel via distance transform + per-pixel
    # labels: seed = clean pixels (value 0), everything else 1 -> label of nearest seed.
    src = np.where(clean, 0, 1).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[clean]] = rgba[..., :3][clean]    # label -> that clean pixel's colour
    nearest = lut[labels]                        # every px -> nearest clean colour

    out[..., :3][target] = nearest[target]        # recolour only; alpha never touched
    return out, target


def despill_selection(client: BridgeClient, *, thr: int = 22, grow: int = 0,
                      amin: int = 8, layer: str | None = None) -> dict:
    """Green-eat the current selection on the active (or named) layer's active frame.

    Requires a selection (the lasso) — that's what scopes the recolour. With no
    selection it processes the whole canvas (fine for a subject with no real green).
    """
    sel = client.call("selection.info")
    b = sel["bounds"]
    x0, y0, rw, rh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
    if rw <= 0 or rh <= 0:
        raise RuntimeError("empty region — nothing to despill")

    rgba = _b64_to_rgba(client.call("layer.get_region", x=x0, y=y0, w=rw, h=rh,
                                    **({"layer": layer} if layer else {}))["png_b64"])
    region = None
    if sel.get("hasSelection"):
        m = _b64_to_rgba_mask(client.call("selection.mask_region",
                                          x=x0, y=y0, w=rw, h=rh)["png_b64"])
        region = m > 127

    fixed, target = green_eat(rgba, thr=thr, amin=amin, grow=grow, region=region)
    n = int(target.sum())
    if n:
        client.call("layer.set_region", x=x0, y=y0,
                    png_b64=_rgba_to_b64(fixed),
                    **({"layer": layer} if layer else {}))
    return {"recoloured": n, "x": x0, "y": y0, "w": rw, "h": rh,
            "scoped": bool(sel.get("hasSelection"))}


def _b64_to_rgba_mask(s: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("L"))
