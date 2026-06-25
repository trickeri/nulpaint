"""Nano Banana Pro (Google Gemini 3 Pro Image) via OpenRouter — cloud image edit.

Unlike stable-diffusion.cpp, this is an *instruction-based* editor: it takes a text
prompt plus one or more reference images and returns a whole reimagined image. There
is no mask — region targeting is done caller-side (we composite the result back over
just the selection / new border, and also drop the full generation on its own layer).

OpenRouter is OpenAI-compatible; we hit chat/completions with modalities=["image",
"text"]. Stdlib only (urllib) so this stays dependency-light; images go in/out as
base64 PNG data URLs. Needs an OpenRouter API key (see config.openrouter_api_key()).
"""
from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image

from ..config import (OPENROUTER_URL, NANOBANANA_MODEL, NANOBANANA_MAX_REFS,
                      openrouter_api_key)


class NanoBananaError(RuntimeError):
    """A Nano Banana / OpenRouter call failed (missing key, HTTP error, no image)."""


def _img_to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_image(message: dict) -> Image.Image | None:
    """Pull the generated image out of an assistant message. Gemini-on-OpenRouter
    returns it in `message.images[].image_url.url` (a base64 data URL); be lenient
    and also scan `content` parts in case the shape differs."""
    candidates = []
    for entry in (message.get("images") or []):
        url = (entry.get("image_url") or {}).get("url") if isinstance(entry, dict) else None
        if url:
            candidates.append(url)
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "output_image"):
                url = (part.get("image_url") or {}).get("url") or part.get("image_url")
                if isinstance(url, str):
                    candidates.append(url)
    for url in candidates:
        if isinstance(url, str) and url.startswith("data:") and "base64," in url:
            raw = base64.b64decode(url.split("base64,", 1)[1])
            return Image.open(io.BytesIO(raw)).convert("RGBA")
    return None


def edit_image(prompt: str, images: list[Image.Image], *,
               model: str | None = None, timeout: float = 180.0) -> Image.Image:
    """Send `prompt` + `images` (base image first, then references) to Nano Banana Pro
    and return the generated image (RGBA). Raises NanoBananaError on any failure."""
    key = openrouter_api_key()
    if not key:
        raise NanoBananaError(
            "no OpenRouter API key — set $OPENROUTER_API_KEY or write it to "
            "~/.config/nulpaint/openrouter.key")
    if not images:
        raise NanoBananaError("no images to send (need at least the base image)")

    content: list[dict] = [{"type": "text", "text": prompt}]
    for im in images[:NANOBANANA_MAX_REFS]:
        content.append({"type": "image_url", "image_url": {"url": _img_to_data_url(im)}})

    body = json.dumps({
        "model": model or NANOBANANA_MODEL,
        "modalities": ["image", "text"],
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # OpenRouter attribution headers (optional but recommended).
            "HTTP-Referer": "https://github.com/trickeri/nulpaint",
            "X-Title": "nulpaint",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise NanoBananaError(f"OpenRouter HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise NanoBananaError(f"OpenRouter unreachable: {e.reason}") from e

    choices = payload.get("choices") or []
    if not choices:
        err = payload.get("error") or payload
        raise NanoBananaError(f"no choices in response: {json.dumps(err)[:400]}")
    out = _extract_image(choices[0].get("message") or {})
    if out is None:
        # Surface any text the model returned instead of an image (refusal/explanation).
        text = (choices[0].get("message") or {}).get("content")
        raise NanoBananaError(
            f"model returned no image. text={str(text)[:300]!r}")
    return out
