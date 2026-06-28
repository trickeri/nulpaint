"""Subject selection: canvas → matte/seg service → Krita selection.

Grabs the merged canvas over the bridge, POSTs it to a local model service
(mattemodel for a person, segmodel for an arbitrary object), and applies the
returned alpha mask as the document's selection. No model code lives here — the
services own that; this just wires Krita to them.
"""
from __future__ import annotations

import base64
import socket
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

from ..bridge import BridgeClient
from ..config import MATTEMODEL_URL, SEGMODEL_URL


def _ensure_serving(service: str, url: str, timeout: float = 90.0) -> None:
    """Make sure the model service is loaded and listening before we use it.

    Asks the model-manager to auto-place it (GPU if VRAM is free, else RAM) when it's
    parked on M.2, then waits for its port. This is what lets you just pick the tool
    and click without manually loading the model first. Best-effort: if the manager
    isn't on the bus we skip straight to waiting on the port (the service may already
    be up / systemd-managed)."""
    subprocess.run(
        ["qdbus6", "com.nuldrums.ModelManager", "/ModelManager",
         "com.nuldrums.ModelManager.EnsureServing", service],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    parsed = urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    # fall through — the /matte POST will surface a clear error if it never came up


def _matte(service_url: str, png_bytes: bytes, timeout: float = 60.0) -> bytes:
    """POST an image to a model service's /matte?format=alpha; return mask PNG.

    Uses multipart "image" — accepted by both mattemodel (C++) and segmodel.
    """
    boundary = "----nulpaint"
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
        f"filename=\"canvas.png\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode()
    body = head + png_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{service_url}/matte?format=alpha", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fill_mask_holes(mask_png: bytes) -> bytes:
    """Fill fully-enclosed interior holes in a grayscale subject mask.

    The matte/seg services sometimes punch transparent "islands" inside an
    otherwise solid subject (a dark patch the salient-object model misreads as
    background), which then has to be repainted by hand. This solidifies any hole
    that is *completely surrounded* by subject — every enclosed interior pixel goes
    fully opaque — while leaving the subject's OUTER anti-aliased silhouette
    untouched (so edges stay soft, only interior islands are closed). A hole that
    touches the subject's outline (an open notch, not an island) is left alone.
    Takes and returns PNG bytes; a no-op mask is returned unchanged.
    """
    import io
    import numpy as np
    import cv2
    from PIL import Image

    m = np.asarray(Image.open(io.BytesIO(mask_png)).convert("L"))
    binary = (m >= 128).astype(np.uint8)
    if binary.max() == 0:
        return mask_png                                   # empty mask — nothing to fill
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(binary.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    holes = (filled == 1) & (binary == 0)                 # enclosed interior, sub-threshold
    if not holes.any():
        return mask_png
    out = m.copy()
    out[holes] = 255
    buf = io.BytesIO()
    Image.fromarray(out, "L").save(buf, "PNG")
    return buf.getvalue()


def select_subject(client: BridgeClient, kind: str = "person",
                   fill_holes: bool = True) -> dict:
    """Select the subject in the active document.

    kind: "person" → mattemodel (RVM human matte);
          "object" → segmodel (salient-object: BiRefNet@cuda / U2Net@vulkan).
    fill_holes (default on): close any transparent islands fully enclosed by the
    subject so the selection has no interior gaps (see fill_mask_holes).
    Returns the applied selection's bounds.
    """
    url = SEGMODEL_URL if kind == "object" else MATTEMODEL_URL
    service = "segmodel" if kind == "object" else "mattemodel"
    _ensure_serving(service, url)                         # auto-load from M.2 if parked
    img = client.call("image.get")                       # {w, h, png_b64}
    canvas_png = base64.b64decode(img["png_b64"])
    mask_png = _matte(url, canvas_png)
    if fill_holes:
        mask_png = fill_mask_holes(mask_png)
    res = client.call(
        "selection.set_from_mask",
        png_b64=base64.b64encode(mask_png).decode("ascii"),
    )
    return {"kind": kind, "service": url, **res}


def segment_layers(client: BridgeClient, kind: str = "object",
                   suffix: str = " (cut)", on_white: bool = True,
                   limit: int = 0, only: list | None = None,
                   beside: bool = True, fill_holes: bool = True) -> list:
    """Run subject segmentation on paint layers and add a trimmed copy per layer.

    Non-destructive: each source paint layer is left UNTOUCHED (the backup); a new
    '<name><suffix>' layer is added whose alpha = the layer's existing alpha × the seg
    mask — so it only ever *removes* background, never adds. Group layers and empty
    layers are skipped. `limit` (>0) processes only the first N paint layers (for a
    quick quality check before the full run). `only` (a list of layer names) scopes
    the run to just those layers — anywhere in the stack, including inside groups.
    `beside` (default) drops each cut directly BELOW its source (so it stays in the
    same group, original-above-cut); set False to push cuts to the document top.
    Returns a per-layer status report.
    """
    import io
    from PIL import Image, ImageChops

    url = SEGMODEL_URL if kind == "object" else MATTEMODEL_URL
    service = "segmodel" if kind == "object" else "mattemodel"
    _ensure_serving(service, url)
    info = client.call("document.info")
    W, H = info["width"], info["height"]
    layers = client.call("layer.list")["layers"]
    existing = {L["name"] for L in layers}

    report = []
    done = 0
    for L in layers:
        if L.get("type") != "paintlayer":
            continue
        name = L["name"]
        if only is not None and name not in only:
            continue
        if limit and done >= limit:
            break
        if name.endswith(suffix):
            continue   # this IS a cut layer (e.g. a prior run) — never re-cut it
        if (name + suffix) in existing:
            report.append({"layer": name, "status": "skipped (already has cut)"})
            continue
        try:
            r = client.call("layer.get_region", x=0, y=0, w=W, h=H, layer=name)
            rgba = Image.open(io.BytesIO(base64.b64decode(r["png_b64"]))).convert("RGBA")
            if rgba.getbbox() is None:
                report.append({"layer": name, "status": "skipped (empty)"})
                continue
            # Composite onto white so BiRefNet sees a clean fg/bg, then matte.
            comp = Image.new("RGB", (W, H), (255, 255, 255))
            comp.paste(rgba, (0, 0), rgba)
            if not on_white:
                comp = rgba.convert("RGB")
            buf = io.BytesIO()
            comp.save(buf, "PNG")
            mask_png = _matte(url, buf.getvalue())
            if fill_holes:
                mask_png = fill_mask_holes(mask_png)       # close interior islands
            mask = Image.open(io.BytesIO(mask_png)).convert("L")
            if mask.size != (W, H):
                mask = mask.resize((W, H))
            # New alpha = existing alpha ∧ seg mask (trim only — never paint outside).
            cut = rgba.copy()
            cut.putalpha(ImageChops.multiply(rgba.getchannel("A"), mask))
            out = io.BytesIO()
            cut.save(out, "PNG")
            if beside:
                # land the cut directly below its source (same group, PS-style pair)
                client.call("node.set_active", node=name)
                place = "below_active"
            else:
                place = "top"
            client.call("layer.add_image", name=name + suffix,
                        png_b64=base64.b64encode(out.getvalue()).decode("ascii"),
                        place=place)
            report.append({"layer": name, "status": "ok"})
            done += 1
        except Exception as e:  # noqa: BLE001 — keep going; report per-layer
            report.append({"layer": name, "status": "error: %s" % e})
    return report
