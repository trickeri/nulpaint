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
                      SD_IMG_CFG, LORA_DIR, CONTROLNET_DIR, CONTROL_MODELS,
                      DIFFUSION_URL)
from . import sdclient

# SDXL works at a 1024 long side. inpaint/outpaint/style all run on the warm SDXL
# daemon now (one resident model), so the per-op sd15/sdxl/sd15base split is gone;
# only `control` still cold-spawns sd-cli (its ControlNets are model-specific).
_SDXL_NATIVE = 1024


def _lora_tokens(lora: str) -> str:
    """'name' -> <lora:name:1.0>; 'name:0.7' -> <lora:name:0.7>; comma-separated."""
    out = []
    for part in (p.strip() for p in lora.split(",")):
        if not part:
            continue
        name, _, w = part.partition(":")
        out.append(f"<lora:{name}:{w or '1.0'}>")
    return " ".join(out)


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


def _generate(init_rgba: Image.Image, mask_L: Image.Image, prompt: str,
              negative: str, model: str, steps: int, cfg: float,
              strength: float, seed: int, img_cfg: float,
              lora: str = "") -> Image.Image:
    """init (RGBA) + mask (L, white=regen) at region resolution → composited RGBA.

    Runs on the warm SDXL daemon (one HTTP round-trip, no model reload). `model` is
    accepted for call-site compatibility but no longer selects a checkpoint — the
    daemon's resident SDXL serves every op (LoRAs via <lora:..> tokens in the prompt).
    """
    if lora:
        prompt = f"{prompt} {_lora_tokens(lora)}".strip()
    rw, rh = init_rgba.size
    ww, hh = _work_size(rw, rh, _SDXL_NATIVE)
    init = init_rgba.convert("RGB").resize((ww, hh), Image.LANCZOS)
    mask = mask_L.resize((ww, hh), Image.LANCZOS)
    result = sdclient.generate(
        DIFFUSION_URL, prompt=prompt, negative=negative,
        init_image=init, mask_image=mask, width=ww, height=hh,
        steps=steps, txt_cfg=cfg, img_cfg=img_cfg, strength=strength, seed=seed,
    ).resize((rw, rh), Image.LANCZOS)
    # Masked area = generated, rest = original (so only the selection changes).
    return Image.composite(result.convert("RGBA"), init_rgba, mask_L)


def inpaint(client: BridgeClient, prompt: str, *, negative: str = "",
            model: str | None = None, steps: int = 20, cfg: float = 7.0,
            strength: float = 1.0, seed: int = -1, pad: float = 0.25,
            img_cfg: float = SD_IMG_CFG, lora: str = "") -> dict:
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
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg, lora)
    client.call("layer.set_region", x=x0, y=y0, png_b64=_img_to_b64(out))
    return {"mode": "inpaint", "model": model, "x": x0, "y": y0, "w": rw, "h": rh}


def outpaint(client: BridgeClient, prompt: str, *, negative: str = "",
             model: str | None = None, pixels: int = 256, sides: str = "all",
             steps: int = 20, cfg: float = 7.0, strength: float = 1.0,
             seed: int = -1, img_cfg: float = SD_IMG_CFG, lora: str = "") -> dict:
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
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg, lora)
    client.call("layer.set_region", x=0, y=0, png_b64=_img_to_b64(out))
    return {"mode": "outpaint", "model": model, "width": NW, "height": NH,
            "pixels": pixels, "sides": sorted(want)}


def style(client: BridgeClient, prompt: str, *, negative: str = "",
          model: str = "sdxl", strength: float = 0.55, steps: int = 24,
          cfg: float = 7.0, seed: int = -1, lora: str = "") -> dict:
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
                    img_cfg=None, lora=lora)  # base-model img2img: no inpaint image-guidance
    client.call("layer.set_region", x=x0, y=y0, png_b64=_img_to_b64(out))
    return {"mode": "style", "model": model, "scope": scope,
            "strength": strength, "x": x0, "y": y0, "w": rw, "h": rh}


def _canny(img: Image.Image) -> Image.Image:
    """Canny edge map (OpenCV if present, else a PIL fallback) for ControlNet."""
    try:
        import cv2, numpy as np
        g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return Image.fromarray(cv2.Canny(g, 100, 200)).convert("RGB")
    except ImportError:
        from PIL import ImageFilter, ImageOps
        return ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES).convert("RGB")


def control(client: BridgeClient, prompt: str, *, kind: str = "canny",
            control_image: str | None = None, model: str = "sd15base",
            control_strength: float = 0.9, steps: int = 24, cfg: float = 7.0,
            seed: int = -1, negative: str = "", lora: str = "") -> dict:
    """Generate a new image conditioned on a structural control map, onto a new
    layer (original preserved). kind: 'canny' (composition lock, map derived from
    the canvas) or 'openpose' (repose — needs a skeleton via --control-image).
    Uses SD1.5 ControlNets, so model defaults to sd15base."""
    if kind not in CONTROL_MODELS:
        raise RuntimeError(f"unknown control {kind!r} (have: {', '.join(CONTROL_MODELS)})")
    cnet = os.path.join(CONTROLNET_DIR, CONTROL_MODELS[kind])
    if not os.path.exists(cnet):
        raise RuntimeError(f"ControlNet model missing: {cnet} — fetch it first")
    info = client.call("document.info")
    W, H = info["width"], info["height"]

    # Source image for the control map: a provided reference, else the canvas.
    src = (Image.open(control_image).convert("RGB") if control_image
           else _b64_to_img(client.call("image.get")["png_b64"]).convert("RGB"))
    if kind == "canny":
        ctrl = _canny(src)
    elif kind == "openpose":
        import numpy as np
        from ..vision.pose import pose_skeleton
        skel = pose_skeleton(np.ascontiguousarray(np.asarray(src)[:, :, ::-1]))  # RGB->BGR
        ctrl = Image.fromarray(skel[:, :, ::-1])  # BGR->RGB
    else:
        raise RuntimeError(f"unhandled control kind {kind!r}")

    ww, hh = _work_size(W, H, SD_NATIVE.get(model, 512))
    p = f"{prompt} {_lora_tokens(lora)}".strip() if lora else prompt
    with tempfile.TemporaryDirectory() as td:
        cp, op = os.path.join(td, "ctrl.png"), os.path.join(td, "out.png")
        ctrl.resize((ww, hh), Image.LANCZOS).save(cp)
        cmd = [SDCLI_BIN, "-M", "img_gen", "-m", SD_MODELS[model], "-p", p, "-o", op,
               "-W", str(ww), "-H", str(hh), "--steps", str(steps),
               "--cfg-scale", str(cfg), "-s", str(seed), "--sampling-method", "euler_a",
               "--control-net", cnet, "--control-image", cp,
               "--control-strength", str(control_strength)]
        if negative:
            cmd += ["-n", negative]
        if os.path.isdir(LORA_DIR):
            cmd += ["--lora-model-dir", LORA_DIR]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError(f"sd-cli controlnet failed (exit {r.returncode}): {r.stderr[-600:]}")
        result = Image.open(op).convert("RGBA").resize((W, H), Image.LANCZOS)

    layer = f"nulpaint {kind}"
    client.call("layer.add", name=layer)
    client.call("layer.set_region", layer=layer, x=0, y=0, png_b64=_img_to_b64(result))
    return {"mode": "control", "kind": kind, "model": model, "layer": layer,
            "w": W, "h": H}
