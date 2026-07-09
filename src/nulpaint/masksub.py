"""Boolean layer subtract: erase from a TARGET layer everything covered by a
MASK layer's alpha footprint, with an optional grow so the cut fully surrounds
the mask (catching the anti-aliased edge halo).

Use case that motivated it: a merged/baked background layer still contains a
foreground element that has ALSO been lifted onto its own layer — e.g.
`GroundPatch1_Merged` still bakes in the grass that now lives on
`GroundPatch1_ForegroundGrass`. Painting them together double-draws the grass.
This punches the grass-shaped hole out of the merged layer using the grass
layer's own alpha as a stencil; the hole gets filled later (hand-paint / inpaint).

Driven from the bridge with the same pixel-I/O verbs as inpaint/despill —
`document.info` for the canvas size, `layer.get_region` -> numpy alpha op ->
`layer.set_region`. No model, no daemon. Non-destructive by default: it
duplicates the target, hides the pristine original, and cuts the copy, so
re-running with a different `grow` always starts from clean pixels. It can also
leave the grown stencil as the document selection so the SAME mask drives the
fill step.
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


def _mask_to_b64(mask: np.ndarray) -> str:
    """boolean HxW -> grayscale PNG b64 (white where True) for selection.set_from_mask."""
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate (px>0) or erode (px<0) the boolean mask. px==0 is a no-op.
    Negative px pulls the stencil IN, so the cut stays inside the mask edge."""
    if px == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * abs(px) + 1, 2 * abs(px) + 1))
    op = cv2.dilate if px > 0 else cv2.erode
    return op(mask.astype(np.uint8), k).astype(bool)


def _resolve(client: BridgeClient, ident: str) -> tuple[str, str]:
    """(uuid, name) for a layer addressed by exact name or uuid, via node.tree."""
    want = str(ident).strip("{}").lower()
    found: list[tuple[str, str]] = []

    def walk(n: dict) -> None:
        u = str(n.get("uuid", "")).strip("{}").lower()
        if u == want or n.get("name") == ident:
            found.append((n["uuid"], n["name"]))
        for ch in n.get("children", []):
            walk(ch)

    for top in client.call("node.tree")["tree"]:
        walk(top)
    if not found:
        raise RuntimeError(f"layer not found: {ident!r}")
    return found[0]


def _find_name(client: BridgeClient, name: str) -> str | None:
    """uuid of the first node with this exact name, or None."""
    try:
        return _resolve(client, name)[0]
    except RuntimeError:
        return None


def preview_mask(client: BridgeClient, *, mask_layer: str, grow: int = 0,
                 threshold: int = 8) -> dict:
    """Set the document selection to the (grown/eroded) stencil of `mask_layer`
    WITHOUT cutting anything — dial `grow` in to eyeball how far the cut reaches
    before committing. Negative grow pulls the mask in."""
    info = client.call("document.info")
    W, H = int(info["width"]), int(info["height"])
    _, mask_name = _resolve(client, mask_layer)
    mrgba = _b64_to_rgba(client.call("layer.get_region", x=0, y=0, w=W, h=H,
                                     layer=mask_name)["png_b64"])
    stencil = mrgba[..., 3] > threshold
    if not stencil.any():
        raise RuntimeError(f"mask layer {mask_name!r} has no pixels above alpha "
                           f"{threshold}")
    stencil = _grow(stencil, grow)
    client.call("selection.set_from_mask", x=0, y=0, png_b64=_mask_to_b64(stencil))
    return {"mask_layer": mask_name, "grow": grow, "threshold": threshold,
            "stencil_px": int(stencil.sum())}


def _b64_to_gray(s: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("L"))


def _nearest_opaque_rgb(rgb: np.ndarray, opaque: np.ndarray) -> np.ndarray:
    """Give every pixel the RGB of its nearest OPAQUE pixel (Voronoi by distance
    transform) — so transparent/black areas around a hole never bleed into a fill."""
    src = np.where(opaque, 0, 1).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[opaque]] = rgb[opaque]
    return lut[labels]


def fill_hole(client: BridgeClient, *, target: str | None = None,
              pad: float = 0.4, radius: int = 5,
              layer_name: str = "GrassFill (patch)",
              below: str | None = None) -> dict:
    """Content-aware fill the current selection (a hole in `target`, default active
    layer) by SAMPLING the surrounding real pixels — no model, no GPU. Writes the
    fill to its OWN new layer so it can be compared/toggled/discarded.

    Method: nearest-opaque prefill (kills transparent/black bleed) + Telea inpaint
    over the hole, giving a smooth continuation of the neighbouring grass/dirt. The
    new layer carries ONLY the hole pixels (feathered by the selection), transparent
    elsewhere, placed below `below` (a layer name, default = just under the foreground).
    """
    sel = client.call("selection.info")
    if not sel.get("hasSelection"):
        raise RuntimeError("no selection — the hole to fill must be selected")
    b = sel["bounds"]
    info = client.call("document.info")
    W, H = int(info["width"]), int(info["height"])
    px, py = int(b["w"] * pad), int(b["h"] * pad)
    x0 = max(0, b["x"] - px); y0 = max(0, b["y"] - py)
    x1 = min(W, b["x"] + b["w"] + px); y1 = min(H, b["y"] + b["h"] + py)
    rw, rh = x1 - x0, y1 - y0

    tgt_kw = {}
    if target:
        _, tname = _resolve(client, target)
        tgt_kw = {"layer": tname}
    rgba = _b64_to_rgba(client.call("layer.get_region", x=x0, y=y0, w=rw, h=rh,
                                    **tgt_kw)["png_b64"])
    holeL = _b64_to_gray(client.call("selection.mask_region",
                                     x=x0, y=y0, w=rw, h=rh)["png_b64"])

    rgb = rgba[..., :3].copy()
    opaque = rgba[..., 3] > 8
    hole = holeL > 8
    if not hole.any():
        raise RuntimeError("empty selection region")
    # prefill non-opaque with nearest real colour, then Telea-blend across the hole
    prefill = rgb.copy()
    prefill[~opaque] = _nearest_opaque_rgb(rgb, opaque)[~opaque]
    filled = cv2.inpaint(prefill, (hole.astype(np.uint8) * 255), radius,
                         cv2.INPAINT_TELEA)

    out = np.zeros((rh, rw, 4), np.uint8)
    out[..., :3] = filled
    out[..., 3] = holeL            # feathered by the selection, opaque only in the hole

    # place the patch layer just under the foreground (set that active, insert below it)
    if below:
        client.call("node.set_active", node=below)
    r = client.call("layer.add_image", name=layer_name, x=x0, y=y0,
                    png_b64=_rgba_to_b64(out), place="below_active")
    return {"fill_layer": r["layer"], "x": x0, "y": y0, "w": rw, "h": rh,
            "filled_px": int(hole.sum())}


def subtract_layer(client: BridgeClient, *, target: str,
                   mask_layer: str | None = None, from_selection: bool = False,
                   grow: int = 3, threshold: int = 8, in_place: bool = False,
                   suffix: str = " GrassRemoved", hide_source: bool = True,
                   clear_rgb: bool = False, set_selection: bool = True) -> dict:
    """Erase a stencil out of `target`. The stencil is either a MASK LAYER's alpha
    (> `threshold`, grown by `grow` px) OR — with `from_selection` — the live
    document selection, which erases with FEATHER (soft selection edges partially
    erase, so a hand-brushed selection keeps its softness).

    Non-destructive unless `in_place`: duplicates `target` to `<target><suffix>`,
    hides the original, and cuts the copy (a stale copy of that name is deleted
    first so re-runs start from pristine pixels). With a mask layer, `set_selection`
    leaves the grown stencil as the live selection; from a selection it is left as-is.
    """
    info = client.call("document.info")
    W, H = int(info["width"]), int(info["height"])
    tgt_uuid, tgt_name = _resolve(client, target)

    # erase map in [0,1]: 1 = fully erase, fractional = feathered erase
    if from_selection:
        seln = client.call("selection.info")
        if not seln.get("hasSelection"):
            raise RuntimeError("no active selection to subtract — make/brush one first")
        gray = _b64_to_gray(client.call("selection.mask_region",
                                        x=0, y=0, w=W, h=H)["png_b64"])
        if grow:                        # optional hard grow/shrink of the selection
            gray = np.where(_grow(gray > 0, grow), 255, gray).astype(np.uint8) \
                if grow > 0 else np.where(_grow(gray > 0, grow), gray, 0).astype(np.uint8)
        erase = gray.astype(np.float32) / 255.0
        src_desc = "selection"
    else:
        if not mask_layer:
            raise RuntimeError("need --mask <layer> or --from-selection")
        _, mask_name = _resolve(client, mask_layer)
        mrgba = _b64_to_rgba(client.call("layer.get_region", x=0, y=0, w=W, h=H,
                                         layer=mask_name)["png_b64"])
        stencil = mrgba[..., 3] > threshold
        if not stencil.any():
            raise RuntimeError(f"mask layer {mask_name!r} has no pixels above alpha "
                               f"{threshold} — nothing to subtract")
        erase = _grow(stencil, grow).astype(np.float32)
        src_desc = mask_name
    if not (erase > 0).any():
        raise RuntimeError("empty stencil — nothing to subtract")

    # pick / create the layer we actually cut
    if in_place:
        cut_name = tgt_name
    else:
        cut_name = f"{tgt_name}{suffix}"
        stale = _find_name(client, cut_name)
        if stale:
            client.call("node.delete", node=stale)
        r = client.call("node.duplicate", node=tgt_uuid, name=cut_name)
        cut_name = r["name"]
        if hide_source:
            client.call("node.set_visible", uuid=tgt_uuid, visible=False)

    # punch the hole: new_alpha = old_alpha * (1 - erase)  (hard when erase is 0/1)
    trgba = _b64_to_rgba(client.call("layer.get_region", x=0, y=0, w=W, h=H,
                                     layer=cut_name)["png_b64"]).copy()
    before = int((trgba[..., 3] > 0).sum())
    a = trgba[..., 3].astype(np.float32)
    trgba[..., 3] = np.clip(a * (1.0 - erase), 0, 255).round().astype(np.uint8)
    if clear_rgb:
        trgba[..., :3][erase > 0.5] = 0
    after = int((trgba[..., 3] > 0).sum())
    client.call("layer.set_region", x=0, y=0, png_b64=_rgba_to_b64(trgba),
                layer=cut_name)

    if set_selection and not from_selection:
        client.call("selection.set_from_mask", x=0, y=0,
                    png_b64=_mask_to_b64(erase > 0.5))

    return {"cut_layer": cut_name, "source": src_desc, "target": tgt_name,
            "from_selection": from_selection, "grow": grow, "threshold": threshold,
            "erased_px": before - after, "in_place": in_place,
            "hid_source": (not in_place and hide_source)}
