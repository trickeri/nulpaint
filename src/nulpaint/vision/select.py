"""Subject selection: canvas → matte/seg service → Krita selection.

Grabs the merged canvas over the bridge, POSTs it to a local model service
(mattemodel for a person, segmodel for an arbitrary object), and applies the
returned alpha mask as the document's selection. No model code lives here — the
services own that; this just wires Krita to them.
"""
from __future__ import annotations

import base64
import urllib.request

from ..bridge import BridgeClient
from ..config import MATTEMODEL_URL, SEGMODEL_URL


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


def select_subject(client: BridgeClient, kind: str = "person") -> dict:
    """Select the subject in the active document.

    kind: "person" → mattemodel (RVM human matte);
          "object" → segmodel (salient-object: BiRefNet@cuda / U2Net@vulkan).
    Returns the applied selection's bounds.
    """
    url = SEGMODEL_URL if kind == "object" else MATTEMODEL_URL
    img = client.call("image.get")                       # {w, h, png_b64}
    canvas_png = base64.b64decode(img["png_b64"])
    mask_png = _matte(url, canvas_png)
    res = client.call(
        "selection.set_from_mask",
        png_b64=base64.b64encode(mask_png).decode("ascii"),
    )
    return {"kind": kind, "service": url, **res}
