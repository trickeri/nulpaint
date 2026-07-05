"""Export a chibi character's final art from the open ChibiToonEdits.kra to the three
still deliverables, matching the EXISTING padding/size exactly:

  Chibi_<Name>.png              — NoBG, native 1024 position      -> Chibi_NoBG/
  Chibi_<Name>_Padded.png       — NoBG, scaled + repositioned      -> Chibi_NoBG/
  Chibi_<Name>_GreenBG_Padded.png — the padded art on green         -> Chibi_ChromaBG/

Punch/colour edits are colour-only (alpha/silhouette unchanged), so the padded
placement is read off whichever existing reference file exists and re-applied to the
new art — pixel-position-identical to the old, just recoloured. Existing outputs are
backed up to a `_prepunch_bak/` subfolder (once) before overwriting so results can be
compared. Colour-mask layers (crosschannel) are baked onto a temp layer via
filter.read_config + filter.apply so the projection is exact.
"""
from __future__ import annotations

import base64
import io
import os
import shutil

import numpy as np
from PIL import Image

from .bridge import BridgeClient

ST = "/mnt/storage1/Pictures/Nuldrums/StreamToons"
NOBG_DIR = f"{ST}/Chibi_NoBG"
CHROMA_DIR = f"{ST}/Chibi_ChromaBG"
OBJECT_DIR = f"{ST}/OtherImages_NoBG"
GREEN = (0, 177, 64)
W = H = 1024


def _b64_to_img(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGBA")


def _find_mask_uuid(client: BridgeClient, layer_base: str) -> str | None:
    """uuid of the filtermask under '<layer_base> (cut) nulpaint', or None."""
    want = f"{layer_base} (cut) nulpaint"

    def walk(n):
        if n["name"] == want:
            for c in n.get("children", []):
                if c["type"] == "filtermask":
                    return c["uuid"]
        for c in n.get("children", []):
            r = walk(c)
            if r:
                return r
        return None

    for t in client.call("node.tree", depth=6)["tree"]:
        r = walk(t)
        if r:
            return r
    return None


def _char_projection(client: BridgeClient, layer_name: str, mask_uuid: str | None) -> Image.Image:
    """Full-canvas RGBA of the character: punch pixels, with the colour mask baked in
    (via a temp layer) when the layer has one."""
    if not mask_uuid:
        r = client.call("layer.get_region", layer=layer_name, x=0, y=0, w=W, h=H)
        return _b64_to_img(r["png_b64"])
    cfg = client.call("filter.read_config", uuid=mask_uuid)
    punched = client.call("layer.get_region", layer=layer_name, x=0, y=0, w=W, h=H)["png_b64"]
    tmp = "_chibi_bake_tmp"
    client.call("layer.add_image", name=tmp, png_b64=punched, place="top", x=0, y=0)
    try:
        client.call("filter.apply", filter=cfg["filter"], node=tmp,
                    config=cfg["config"], refresh=False)
        baked = client.call("layer.get_region", layer=tmp, x=0, y=0, w=W, h=H)["png_b64"]
    finally:
        client.call("node.delete", node=tmp)
    return _b64_to_img(baked)


def _padded_bbox(exp: str) -> tuple[int, int, int, int] | None:
    """The character's placement box in the padded frame, from the existing reference:
    prefer the padded NoBG file's alpha bbox, else the non-green region of the green
    file. None if neither exists (caller falls back to a default placement)."""
    pad = f"{NOBG_DIR}/Chibi_{exp}_Padded.png"
    if os.path.exists(pad):
        return Image.open(pad).convert("RGBA").getbbox()
    green = f"{CHROMA_DIR}/Chibi_{exp}_GreenBG_Padded.png"
    if os.path.exists(green):
        arr = np.array(Image.open(green).convert("RGB")).astype(int)
        dist = np.abs(arr - np.array(GREEN)).sum(2)
        ys, xs = np.where(dist > 60)
        if len(xs):
            return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return None


def _backup(path: str, bakdir: str) -> bool:
    if os.path.exists(path):
        os.makedirs(bakdir, exist_ok=True)
        dst = os.path.join(bakdir, os.path.basename(path))
        if not os.path.exists(dst):
            shutil.copy2(path, dst)
        return True
    return False


def export_chibi(client: BridgeClient, *, layer_base: str, exp_name: str,
                 mask: bool = False, nobg_only: bool = False,
                 keep_aspect: bool = False) -> dict:
    layer_name = f"{layer_base} (cut) nulpaint"
    mask_uuid = _find_mask_uuid(client, layer_base) if mask else None
    if mask and not mask_uuid:
        raise RuntimeError(f"{layer_base}: --mask set but no filtermask found")

    proj = _char_projection(client, layer_name, mask_uuid)     # 1024 RGBA, new colours
    nb = proj.getbbox()
    if nb is None:
        raise RuntimeError(f"{layer_name}: empty projection")

    # NoBG-only: just the tight native-position PNG (website still), no padded/green.
    if nobg_only:
        tight_p = f"{NOBG_DIR}/Chibi_{exp_name}.png"
        backed = _backup(tight_p, f"{NOBG_DIR}/_prepunch_bak")
        proj.save(tight_p)
        return {"name": exp_name, "masked": bool(mask_uuid), "native_bbox": nb,
                "padded_bbox": None, "placement": "nobg-only", "backed_up": int(backed)}

    pb = _padded_bbox(exp_name)
    derived = "reference"
    if pb is None:                          # no reference -> default padding
        s = 0.5
        cw, ch = round((nb[2] - nb[0]) * s), round((nb[3] - nb[1]) * s)
        px0 = (W - cw) // 2
        py0 = 768 - ch
        pb = (px0, py0, px0 + cw, py0 + ch)
        derived = "default(0.5,feet@768)"
    elif keep_aspect:
        # The silhouette CHANGED (e.g. redrawn narrower), so fitting the new art into
        # the old padded box would re-stretch it to the old width. Instead keep the
        # reference's HEIGHT, feet line (bottom) and horizontal centre, and let WIDTH
        # follow the new art's aspect ratio.
        ref_h = pb[3] - pb[1]
        ref_cx = (pb[0] + pb[2]) / 2.0
        ref_bottom = pb[3]
        aw, ah = nb[2] - nb[0], nb[3] - nb[1]
        new_w = max(1, round(ref_h * aw / ah))
        x0 = round(ref_cx - new_w / 2.0)
        pb = (x0, ref_bottom - ref_h, x0 + new_w, ref_bottom)
        derived = "keep-aspect(ref height/feet/centre, new width)"

    char = proj.crop(nb).resize((pb[2] - pb[0], pb[3] - pb[1]), Image.LANCZOS)
    padded = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    padded.paste(char, (pb[0], pb[1]))
    green = Image.new("RGBA", (W, H), GREEN + (255,))
    green.alpha_composite(padded)

    tight_p = f"{NOBG_DIR}/Chibi_{exp_name}.png"
    pad_p = f"{NOBG_DIR}/Chibi_{exp_name}_Padded.png"
    green_p = f"{CHROMA_DIR}/Chibi_{exp_name}_GreenBG_Padded.png"
    backed = (_backup(tight_p, f"{NOBG_DIR}/_prepunch_bak")
              + _backup(pad_p, f"{NOBG_DIR}/_prepunch_bak")
              + _backup(green_p, f"{CHROMA_DIR}/_prepunch_bak"))
    proj.save(tight_p)
    padded.save(pad_p)
    green.save(green_p)
    return {"name": exp_name, "masked": bool(mask_uuid), "native_bbox": nb,
            "padded_bbox": pb, "placement": derived, "backed_up": backed}


def export_object(client: BridgeClient, *, layer_base: str, exp_name: str,
                  mask: bool = False, game_dir: str | None = None) -> dict:
    """Export a NON-character object's final '<base> (cut) nulpaint' layer from the OPEN
    ChibiToonEdits.kra to a plain full-canvas RGBA PNG at OtherImages_NoBG/<name>.png (the
    object convention, no chibi padding/greenscreen). Optionally copy into a game Textures
    dir. `--mask` bakes a Cross-channel colour mask if the layer has one (same path as
    export_chibi). Backs up any existing PNG to OtherImages_NoBG/_bak/ once."""
    layer_name = f"{layer_base} (cut) nulpaint"
    mask_uuid = _find_mask_uuid(client, layer_base) if mask else None
    if mask and not mask_uuid:
        raise RuntimeError(f"{layer_base}: --mask set but no filtermask found")

    proj = _char_projection(client, layer_name, mask_uuid)      # 1024 RGBA, final colours
    if proj.getbbox() is None:
        raise RuntimeError(f"{layer_name}: empty projection")

    os.makedirs(OBJECT_DIR, exist_ok=True)
    out = f"{OBJECT_DIR}/{exp_name}.png"
    backed = _backup(out, f"{OBJECT_DIR}/_bak")
    proj.save(out)

    game_out = None
    if game_dir:
        os.makedirs(game_dir, exist_ok=True)
        game_out = os.path.join(game_dir, f"{exp_name}.png")
        _backup(game_out, os.path.join(game_dir, "_bak"))
        proj.save(game_out)

    return {"name": exp_name, "masked": bool(mask_uuid), "bbox": proj.getbbox(),
            "out": out, "game_out": game_out, "backed_up": int(backed)}
