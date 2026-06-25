"""Inpaint / outpaint via stable-diffusion.cpp (sd-cli), driven from the bridge.

Pulls the selection (or extends the canvas), runs sd-cli with the region as init
and a white=regenerate mask, then composites the result back onto the layer as a
single bridge write. No model lives here — sd-cli owns that. Needs Pillow.

Mask polarity matches Krita: selection white(255) = regenerate, black = keep.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw

from ..bridge import BridgeClient
from ..config import (SDCLI_BIN, SD_MODELS, SD_DEFAULT_MODEL, SD_NATIVE,
                      SD_IMG_CFG, LORA_DIR, CONTROLNET_DIR, CONTROL_MODELS,
                      DIFFUSION_URL, INPAINT_URL, DIFFUSION_SERVICE,
                      INPAINT_SERVICE, MODELMANAGER_STATE)
from . import sdclient

# SDXL works at a 1024 long side. inpaint/outpaint/style all run on a warm SDXL
# daemon now (no per-call reload). Two checkpoints share the GPU one-at-a-time:
# the inpainting checkpoint (inpaint/outpaint) and the base (generation/style).
# Only `control` still cold-spawns sd-cli (its ControlNets are model-specific).
_SDXL_NATIVE = 1024


def _mm_placement(service: str) -> dict | None:
    try:
        with open(MODELMANAGER_STATE, encoding="utf-8") as fh:
            for m in (json.load(fh).get("models") or []):
                if m.get("service") == service:
                    return m
    except Exception:
        pass
    return None


# Friendly names for the two SDXL checkpoints (for the load prompt).
_SVC_NAME = {DIFFUSION_SERVICE: "SDXL Base", INPAINT_SERVICE: "SDXL Inpainting"}


class ModelNotLoadedError(RuntimeError):
    """Raised when a generate verb needs an SDXL checkpoint that the model manager
    hasn't placed on the GPU. We do NOT load it silently — the caller (docker / CLI)
    catches this and PROMPTS the user to load it. `parked` is the other checkpoint
    that the load would swap out to RAM (they're mutually exclusive in VRAM)."""
    def __init__(self, service: str, parked: str, mode: str):
        self.service = service
        self.parked = parked
        self.mode = mode
        self.name = _SVC_NAME.get(service, service)
        self.parked_name = _SVC_NAME.get(parked, parked)
        super().__init__(f"{self.name} is not loaded on the GPU (needed for {mode})")


def _image_mode_ready(active: str) -> bool:
    """True iff `active` is up and GPU-resident per the model manager state."""
    cur = _mm_placement(active)
    return bool(cur and cur.get("up") and cur.get("placement") == "gpu")


def _load_image_mode(active: str, parked: str, *, timeout: float = 45.0) -> None:
    """EXPLICIT load: make `active` the GPU-resident SDXL checkpoint and park `parked`
    in RAM via the modelmanager, then wait until `active` is serving on the GPU. No-op
    if it already is. Only ever called on a deliberate user action (the `mode` verb /
    a confirmed load prompt) — never as a silent side effect of generating."""
    if _image_mode_ready(active):
        return  # already warm on the GPU
    for svc, tgt in ((active, "gpu"), (parked, "ram")):
        subprocess.run(["qdbus6", "com.nuldrums.ModelManager", "/ModelManager",
                        "com.nuldrums.ModelManager.Move", svc, tgt],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.5)
        if _image_mode_ready(active):
            return
    # fall through — the generation call itself will surface any real failure


def _require_image_mode(active: str, parked: str, mode: str) -> None:
    """Gate at the top of every generate verb. The manual model manager is the source
    of truth for VRAM, so we never auto-load: if `active` isn't GPU-resident we raise
    ModelNotLoadedError so the UI can prompt. Set NULPAINT_AUTOLOAD=1 to opt back into
    automatic loading (e.g. the voice/headless path, which can't show a prompt)."""
    if _image_mode_ready(active):
        return
    if os.environ.get("NULPAINT_AUTOLOAD") == "1":
        _load_image_mode(active, parked)
        return
    raise ModelNotLoadedError(active, parked, mode)


def set_mode(mode: str) -> dict:
    """Image-model mode toggle: 'generate'|'style' -> base SDXL warm; 'inpaint'|
    'outpaint' -> SDXL-inpainting warm (the other parks in RAM). This is the EXPLICIT
    load path — the docker's confirmed load prompt and the `nulpaint mode` verb call
    it; selecting a mode in the docker no longer triggers it."""
    mode = (mode or "").lower()
    if mode in ("inpaint", "outpaint"):
        _load_image_mode(INPAINT_SERVICE, DIFFUSION_SERVICE)
        return {"mode": mode, "warm": "inpaint", "url": INPAINT_URL}
    _load_image_mode(DIFFUSION_SERVICE, INPAINT_SERVICE)
    return {"mode": mode or "generate", "warm": "base", "url": DIFFUSION_URL}


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


# --- Nano Banana Pro (cloud) engine ----------------------------------------
# An alternative `engine` for inpaint/outpaint/style: instruction-based editing via
# OpenRouter instead of local sd.cpp. The whole base image is always sent as a
# reference (so the result fits the existing style); the caller can add more refs
# (layers/files). NBP has no mask — we composite its full output back over just the
# target region, and drop the raw full generation on a review layer below.
_NANO_GUIDE = {
    "inpaint": ("Edit this image. {p}. Keep the overall style, lighting, colour "
                "palette and the rest of the scene consistent and unchanged."),
    "outpaint": ("Extend (outpaint) this image to fill the empty/transparent border "
                 "regions, continuing the scene naturally. {p}. Match the existing "
                 "style, lighting and perspective; leave existing content unchanged."),
    "style": ("Restyle this image: {p}. Preserve the composition and the layout of "
              "the subjects."),
}


def _nano_prompt(mode: str, prompt: str) -> str:
    return _NANO_GUIDE[mode].format(p=(prompt or "").strip() or "improve the image")


def _nano_refs(client: BridgeClient, ref_layers, ref_files, W: int, H: int) -> list:
    """Resolve chosen reference images to PIL: document layers (by name, at canvas
    bounds) plus image files on disk. The base image is added separately by callers."""
    refs = []
    for name in (ref_layers or []):
        r = client.call("layer.get_region", x=0, y=0, w=W, h=H, layer=name)
        refs.append(_b64_to_img(r["png_b64"]).convert("RGBA"))
    for path in (ref_files or []):
        refs.append(Image.open(path).convert("RGBA"))
    return refs


def _nano_generate(prompt: str, base_full: Image.Image, refs: list,
                   model: str | None, size: tuple[int, int]) -> Image.Image:
    """Run Nano Banana on base_full + refs; return RGBA resized to `size` so it
    aligns with the canvas for region compositing."""
    from .nanobanana import edit_image
    out = edit_image(prompt, [base_full.convert("RGBA")] + refs, model=model)
    return out.convert("RGBA").resize(size, Image.LANCZOS)


def _nano_apply_region(client: BridgeClient, nano_full: Image.Image, *,
                       x: int, y: int, w: int, h: int, mask_L: Image.Image,
                       review_name: str | None) -> None:
    """Composite the (masked) Nano output over the ACTIVE layer's region (x,y,w,h) and
    write it back — so unmasked pixels of the active layer are preserved. Optionally
    drop the full raw generation on a new review layer below the active one."""
    active = _b64_to_img(client.call("layer.get_region", x=x, y=y, w=w, h=h)["png_b64"]).convert("RGBA")
    nano_region = nano_full.crop((x, y, x + w, y + h))
    comp = Image.composite(nano_region, active, mask_L)
    client.call("layer.set_region", x=x, y=y, png_b64=_img_to_b64(comp))
    if review_name:
        client.call("layer.add_image", name=review_name, x=0, y=0,
                    png_b64=_img_to_b64(nano_full), place="below_active")


def _work_size(w: int, h: int, target: int) -> tuple[int, int]:
    """Scale (w,h) so the long side ≈ target, each snapped to a /64 multiple."""
    s = target / max(w, h)
    ww = max(64, int(round(w * s / 64.0)) * 64)
    hh = max(64, int(round(h * s / 64.0)) * 64)
    return ww, hh


def _generate(init_rgba: Image.Image, mask_L: Image.Image, prompt: str,
              negative: str, model: str, steps: int, cfg: float,
              strength: float, seed: int, img_cfg: float,
              lora: str = "", url: str = DIFFUSION_URL) -> Image.Image:
    """init (RGBA) + mask (L, white=regen) at region resolution → composited RGBA.

    Runs on the warm SDXL daemon at `url` (one HTTP round-trip, no model reload).
    `model` is accepted for call-site compatibility but no longer selects a checkpoint
    — which daemon (`url`) is hit picks the checkpoint (LoRAs via <lora:..> in the prompt).
    """
    if lora:
        prompt = f"{prompt} {_lora_tokens(lora)}".strip()
    rw, rh = init_rgba.size
    ww, hh = _work_size(rw, rh, _SDXL_NATIVE)
    init = init_rgba.convert("RGB").resize((ww, hh), Image.LANCZOS)
    mask = mask_L.resize((ww, hh), Image.LANCZOS)
    result = sdclient.generate(
        url, prompt=prompt, negative=negative,
        init_image=init, mask_image=mask, width=ww, height=hh,
        steps=steps, txt_cfg=cfg, img_cfg=img_cfg, strength=strength, seed=seed,
    ).resize((rw, rh), Image.LANCZOS)
    # Masked area = generated, rest = original (so only the selection changes).
    return Image.composite(result.convert("RGBA"), init_rgba, mask_L)


def inpaint(client: BridgeClient, prompt: str, *, negative: str = "",
            model: str | None = None, steps: int = 20, cfg: float = 7.0,
            strength: float = 1.0, seed: int = -1, pad: float = 0.25,
            img_cfg: float = SD_IMG_CFG, lora: str = "", engine: str = "local",
            ref_layers=None, ref_files=None, nano_model: str | None = None,
            review: bool = True) -> dict:
    """Inpaint the current selection. `pad` adds context margin around the bbox.

    engine='local' uses the SDXL inpainting daemon (mask-based). engine='nanobanana'
    uses Nano Banana Pro (OpenRouter): the whole base image + chosen references go in,
    its full output is composited back over just the selection, and the raw generation
    is dropped on a review layer below."""
    if engine == "nanobanana":
        info = client.call("document.info")
        W, H = info["width"], info["height"]
        sel = client.call("selection.info")
        if not sel.get("hasSelection"):
            raise RuntimeError("no selection — select the area to inpaint first")
        b = sel["bounds"]
        x0, y0, rw, rh = b["x"], b["y"], b["w"], b["h"]
        base_full = _b64_to_img(client.call("image.get")["png_b64"]).convert("RGBA")
        refs = _nano_refs(client, ref_layers, ref_files, W, H)
        nano = _nano_generate(_nano_prompt("inpaint", prompt), base_full, refs, nano_model, (W, H))
        mask = _b64_to_img(client.call("selection.mask_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("L")
        _nano_apply_region(client, nano, x=x0, y=y0, w=rw, h=rh, mask_L=mask,
                           review_name=("nano banana (raw)" if review else None))
        return {"mode": "inpaint", "engine": "nanobanana", "x": x0, "y": y0, "w": rw, "h": rh}

    model = model or SD_DEFAULT_MODEL
    _require_image_mode(INPAINT_SERVICE, DIFFUSION_SERVICE, "inpaint")  # gate before any work
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
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg, lora, url=INPAINT_URL)
    client.call("layer.set_region", x=x0, y=y0, png_b64=_img_to_b64(out))
    return {"mode": "inpaint", "model": model, "x": x0, "y": y0, "w": rw, "h": rh}


def outpaint(client: BridgeClient, prompt: str, *, negative: str = "",
             model: str | None = None, pixels: int = 256, sides: str = "all",
             steps: int = 20, cfg: float = 7.0, strength: float = 1.0,
             seed: int = -1, img_cfg: float = SD_IMG_CFG, lora: str = "",
             engine: str = "local", ref_layers=None, ref_files=None,
             nano_model: str | None = None, review: bool = True) -> dict:
    """Extend the canvas and generate into the new border, using existing pixels
    as context. sides: 'all' or a comma list of left,right,top,bottom.
    engine='nanobanana' fills the border via Nano Banana Pro instead of local SDXL."""
    info = client.call("document.info")
    W, H = info["width"], info["height"]
    want = {"left", "right", "top", "bottom"} if sides == "all" else set(sides.split(","))
    l = pixels if "left" in want else 0
    t = pixels if "top" in want else 0
    r = pixels if "right" in want else 0
    bot = pixels if "bottom" in want else 0
    if not (l or t or r or bot):
        raise RuntimeError(f"no valid sides in {sides!r}")

    if engine != "nanobanana":
        model = model or SD_DEFAULT_MODEL
        _require_image_mode(INPAINT_SERVICE, DIFFUSION_SERVICE, "outpaint")  # gate before image.extend

    ext = client.call("image.extend", left=l, top=t, right=r, bottom=bot)
    NW, NH = ext["width"], ext["height"]
    # White everywhere, black over the original content rect (l,t,W,H) = keep it.
    mask = Image.new("L", (NW, NH), 255)
    ImageDraw.Draw(mask).rectangle([l, t, l + W - 1, t + H - 1], fill=0)

    if engine == "nanobanana":
        base_full = _b64_to_img(client.call("image.get")["png_b64"]).convert("RGBA")
        refs = _nano_refs(client, ref_layers, ref_files, NW, NH)
        nano = _nano_generate(_nano_prompt("outpaint", prompt), base_full, refs, nano_model, (NW, NH))
        _nano_apply_region(client, nano, x=0, y=0, w=NW, h=NH, mask_L=mask,
                           review_name=("nano banana (raw)" if review else None))
        return {"mode": "outpaint", "engine": "nanobanana", "width": NW, "height": NH,
                "pixels": pixels, "sides": sorted(want)}

    init = _b64_to_img(client.call("layer.get_region", x=0, y=0, w=NW, h=NH)["png_b64"]).convert("RGBA")
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed, img_cfg, lora, url=INPAINT_URL)
    client.call("layer.set_region", x=0, y=0, png_b64=_img_to_b64(out))
    return {"mode": "outpaint", "model": model, "width": NW, "height": NH,
            "pixels": pixels, "sides": sorted(want)}


def style(client: BridgeClient, prompt: str, *, negative: str = "",
          model: str = "sdxl", strength: float = 0.55, steps: int = 24,
          cfg: float = 7.0, seed: int = -1, lora: str = "", engine: str = "local",
          ref_layers=None, ref_files=None, nano_model: str | None = None,
          review: bool = True) -> dict:
    """Restyle via img2img — the selection if there is one, else the whole layer.

    `strength` is the transform amount: ~0.3 subtle, ~0.55 balanced, ~0.8 strong
    (structure dissolves above that). Uses a base model (img2img); the sd15
    *inpaint* model is unsuitable for whole-image style, so default is sdxl.
    engine='nanobanana' restyles via Nano Banana Pro (whole base image as reference).
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

    if engine == "nanobanana":
        base_full = _b64_to_img(client.call("image.get")["png_b64"]).convert("RGBA")
        refs = _nano_refs(client, ref_layers, ref_files, W, H)
        nano = _nano_generate(_nano_prompt("style", prompt), base_full, refs, nano_model, (W, H))
        _nano_apply_region(client, nano, x=x0, y=y0, w=rw, h=rh, mask_L=mask,
                           review_name=("nano banana (raw)" if review else None))
        return {"mode": "style", "engine": "nanobanana", "scope": scope,
                "x": x0, "y": y0, "w": rw, "h": rh}

    _require_image_mode(DIFFUSION_SERVICE, INPAINT_SERVICE, "style")  # gate before any work
    init = _b64_to_img(client.call("layer.get_region", x=x0, y=y0, w=rw, h=rh)["png_b64"]).convert("RGBA")
    out = _generate(init, mask, prompt, negative, model, steps, cfg, strength, seed,
                    img_cfg=None, lora=lora, url=DIFFUSION_URL)  # base-model img2img
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
