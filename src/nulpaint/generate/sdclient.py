"""HTTP client for the warm stable-diffusion.cpp daemon (sd-server).

Replaces the per-call ``sd-cli`` subprocess: the model stays resident in the
daemon, so a generation is just an HTTP round-trip instead of reloading the whole
checkpoint every call. Native async API:

    POST /sdcpp/v1/img_gen      -> {"id", "poll_url", ...}
    GET  /sdcpp/v1/jobs/{id}    -> {"status", "result": {"images": [{"b64_json"}]}}

init_image + mask_image perform inpaint / img2img (mask polarity is sd.cpp's:
white(255) = regenerate, black = keep — same as Krita's selection).
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

from PIL import Image

_TERMINAL = {"completed", "succeeded", "done", "failed", "error", "cancelled"}
_OK = {"completed", "succeeded", "done"}


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _http_json(url: str, body=None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(base_url: str, *, prompt: str, negative: str = "",
             init_image: Image.Image | None = None,
             mask_image: Image.Image | None = None,
             control_image: Image.Image | None = None,
             width: int, height: int, steps: int, txt_cfg: float,
             img_cfg: float | None = None, strength: float | None = None,
             control_strength: float | None = None, seed: int = -1,
             sampler: str = "euler_a", poll: float = 0.4,
             timeout: float = 600.0) -> Image.Image:
    """Run one img_gen on the daemon and return the result as a PIL RGB image."""
    guidance = {"txt_cfg": float(txt_cfg)}
    if img_cfg is not None:
        guidance["img_cfg"] = float(img_cfg)
    body: dict = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": int(width),
        "height": int(height),
        "seed": int(seed),
        "sample_params": {"sample_method": sampler, "sample_steps": int(steps),
                          "guidance": guidance},
        # SDXL's VAE decode compute buffer OOMs at >=768px without tiling.
        "vae_tiling_params": {"enabled": True},
    }
    if init_image is not None:
        body["init_image"] = _b64_png(init_image)
    if mask_image is not None:
        body["mask_image"] = _b64_png(mask_image)
    if control_image is not None:
        body["control_image"] = _b64_png(control_image)
        if control_strength is not None:
            body["control_strength"] = float(control_strength)
    if strength is not None:
        body["strength"] = float(strength)

    base = base_url.rstrip("/")
    job = _http_json(base + "/sdcpp/v1/img_gen", body)
    poll_url = job.get("poll_url") or ("/sdcpp/v1/jobs/" + job["id"])
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        j = _http_json(base + poll_url, timeout=30)
        st = (j.get("status") or "").lower()
        if st in _TERMINAL:
            if st not in _OK:
                err = j.get("error") or {}
                raise RuntimeError(f"sd daemon generation failed: {err.get('message', st)}")
            imgs = (j.get("result") or {}).get("images") or []
            if not imgs:
                raise RuntimeError("sd daemon returned no image")
            first = imgs[0]
            b64 = first["b64_json"] if isinstance(first, dict) else first
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    raise RuntimeError("sd daemon generation timed out")
