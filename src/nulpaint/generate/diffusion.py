"""Inpaint / outpaint via stable-diffusion.cpp (sd-cli), driven from the bridge.

Pulls the selection (or extends the canvas), runs sd-cli with the region as init
and a white=regenerate mask, then composites the result back onto the layer as a
single bridge write. No model lives here — sd-cli owns that. Needs Pillow.

Mask polarity matches Krita: selection white(255) = regenerate, black = keep.
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile

from PIL import Image, ImageDraw

from ..bridge import BridgeClient
from ..config import (SDCLI_BIN, SD_MODELS, SD_DEFAULT_MODEL, SD_NATIVE,
                      SD_IMG_CFG)


def _b64_to_img(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s)))


def _img_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _work_size(w: int, h: int, target: int) -> tuple[int, int]:
    """Scale (w,h) so the long side ≈ target, each snapped to a /64 multiple."""
    s = target / max(w, h)
    ww = max(64, int(round(w * s / 64.0)) * 64)
    hh = max(64, int(round(h * s / 64.0)) * 64)
    return ww, hh


def _run_sdcli(init_p, mask_p, out_p, prompt, negative, model_p, ww, hh,
               steps, cfg, strength, seed, img_cfg):
    cmd = [SDCLI_BIN, "-M", "img_gen", "-m", model_p, "-i", init_p,
           "--mask", mask_p, "-o", out_p, "-p", prompt,
           "-W", str(ww), "-H", str(hh), "--steps", str(steps),
           "--cfg-scale", str(cfg), "--strength", str(strength),
           "-s", str(seed), "--sampling-method", "euler_a"]
    if img_cfg is not None:        # inpaint/edit-model image guidance only
        cmd += ["--img-cfg-scale", str(img_cfg)]
    if negative:
        cmd += ["-n", negative]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_p):
        raise RuntimeError(f"sd-cli failed (exit {r.returncode}): {r.stderr[-600:]}")


def _generate(init_rgba: Image.Image, mask_L: Image.Image, prompt: str,
              negative: str, model: str, steps: int, cfg: float,
              strength: float, seed: int, img_cfg: float) -> Image.Image:
    """init (RGBA) + mask (L, white=regen) at region resolution → composited RGBA."""
    if model not in SD_MODELS:
        raise RuntimeError(f"unknown model {model!r} (have: {', '.join(SD_MODELS)})")
    rw, rh = init_rgba.size
    ww, hh = _work_size(rw, rh, SD_NATIVE.get(model, 512))
    with tempfile.TemporaryDirectory() as td:
        ip, mp, op = (os.path.join(td, n) for n in ("init.png", "mask.png", "out.png"))
        init_rgba.convert("RGB").resize((ww, hh), Image.LANCZOS).save(ip)
        mask_L.resize((ww, hh), Image.LANCZOS).save(mp)
        _run_sdcli(ip, mp, op, prompt, negative, SD_MODELS[model], ww, hh,
                   steps, cfg, strength, seed, img_cfg)
        result = Image.open(op).convert("RGB").resize((rw, rh), Image.LANCZOS)
    # Masked area = generated, rest = original (so only the selection changes).
    return Image.composite(result.convert("RGBA"), init_rgba, mask_L)


def inpaint(client: BridgeClient, prompt: str, *, negative: str = "",
            model: str | None = None, steps: int = 20, cfg: float = 7.0,
            strength: float = 1.0, seed: int = -1, pad: float = 0.25,
            img_cfg: float = SD_IMG_CFG) -> dict:
    """Inpaint the current selection. `pad` adds context margin around the bbox."""
    model = model or SD_DEFAULT_MODEL
    sel = client.call("selection.info")
    if not sel.get("hasSelection"):
        raise RuntimeError("no selection — select the area to inpaint first")
    info = client.call("document.info")
    W, H = info["width"], info["height"]
    b = sel["bounds"]
    px, py = int(b["w"] * pad), int(b["h"] * pad)
    x0, y0 = max(0, b["x"] - px), max(0, b["y"] - py)
    x1, y1 = min(W, b["x"] + b["w"] + px), min(H, b["y"] + b["h"] + py)
    rw, rh = x1 - x0, y1 - y0

    init = _b64_to_img(client.call("layer.get_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("RGBA")
    mask = _b64_to_img(client.call("selection.mask_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("L")
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg)
    client.call("layer.set_region", x=x0, y=y0, png_b64=_img_to_b64(out))
    return {"mode": "inpaint", "model": model, "x": x0, "y": y0, "w": rw, "h": rh}


def outpaint(client: BridgeClient, prompt: str, *, negative: str = "",
             model: str | None = None, pixels: int = 256, sides: str = "all",
             steps: int = 20, cfg: float = 7.0, strength: float = 1.0,
             seed: int = -1, img_cfg: float = SD_IMG_CFG) -> dict:
    """Extend the canvas and generate into the new border, using existing pixels
    as context. sides: 'all' or a comma list of left,right,top,bottom."""
    model = model or SD_DEFAULT_MODEL
    info = client.call("document.info")
    W, H = info["width"], info["height"]
    want = {"left", "right", "top", "bottom"} if sides == "all" else set(sides.split(","))
    l = pixels if "left" in want else 0
    t = pixels if "top" in want else 0
    r = pixels if "right" in want else 0
    bot = pixels if "bottom" in want else 0
    if not (l or t or r or bot):
        raise RuntimeError(f"no valid sides in {sides!r}")

    ext = client.call("image.extend", left=l, top=t, right=r, bottom=bot)
    NW, NH = ext["width"], ext["height"]
    init = _b64_to_img(client.call("layer.get_region", x=0, y=0, w=NW, h=NH)["png_b64"]).convert("RGBA")
    # White everywhere, black over the original content rect (l,t,W,H) = keep it.
    mask = Image.new("L", (NW, NH), 255)
    ImageDraw.Draw(mask).rectangle([l, t, l + W - 1, t + H - 1], fill=0)
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg)
    client.call("layer.set_region", x=0, y=0, png_b64=_img_to_b64(out))
    return {"mode": "outpaint", "model": model, "width": NW, "height": NH,
            "pixels": pixels, "sides": sorted(want)}


def style(client: BridgeClient, prompt: str, *, negative: str = "",
          model: str = "sdxl", strength: float = 0.55, steps: int = 24,
          cfg: float = 7.0, seed: int = -1) -> dict:
    """Restyle via img2img — the selection if there is one, else the whole layer.

    `strength` is the transform amount: ~0.3 subtle, ~0.55 balanced, ~0.8 strong
    (structure dissolves above that). Uses a base model (img2img); the sd15
    *inpaint* model is unsuitable for whole-image style, so default is sdxl.
    """
    info = client.call("document.info")
    W, H = info["width"], info["height"]
    sel = client.call("selection.info")
    if sel.get("hasSelection"):
        b = sel["bounds"]
        x0, y0, rw, rh = b["x"], b["y"], b["w"], b["h"]
        mask = _b64_to_img(client.call("selection.mask_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("L")
        scope = "selection"
    else:
        x0, y0, rw, rh = 0, 0, W, H
        mask = Image.new("L", (rw, rh), 255)   # whole layer
        scope = "layer"
    init = _b64_to_img(client.call("layer.get_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("RGBA")
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed,
                    img_cfg=None)   # base-model img2img: no inpaint image-guidance
    client.call("layer.set_region", x=x0, y=y0, png_b64=_img_to_b64(out))
    return {"mode": "style", "model": model, "scope": scope,
            "strength": strength, "x": x0, "y": y0, "w": rw, "h": rh}
