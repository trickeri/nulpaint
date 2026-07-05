"""Shared configuration + wire protocol constants.

This module is the ONE place the in-Krita plugin and the external process
agree on. The plugin half copies the protocol constants verbatim (it can't
import this package), so keep them dead simple and stable.
"""

from __future__ import annotations

import os

# --- Bridge transport -------------------------------------------------------
# Loopback only. The in-Krita server binds here; the external client connects.
BRIDGE_HOST = os.environ.get("NULPAINT_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("NULPAINT_PORT", "8765"))
# Multiple Krita windows can each serve a bridge on their own port. Every running
# instance writes <port>.json here (pid/host/port); `nulpaint instances` lists
# them. The plugin half copies this path (it can't import this module).
INSTANCE_DIR = os.path.expanduser(
    os.environ.get("NULPAINT_INSTANCE_DIR", "~/.local/share/nulpaint/instances"))

# --- Wire protocol ----------------------------------------------------------
# Newline-delimited JSON. One request object per line, one response per line.
#   request:  {"id": <int>, "cmd": <str>, "args": {<...>}}
#   response: {"id": <int>, "ok": <bool>, "result": <any>, "error": <str|null>}
PROTOCOL_VERSION = 1
ENCODING = "utf-8"

# --- Local model services (subject-select posts the canvas, gets a mask) ----
# mattemodel = human matte (RVM); segmodel = arbitrary salient object (BiRefNet/
# U2Net). Both expose POST /matte?format=alpha. See github.com/trickeri/mattemodel
# and .../segmodel.
MATTEMODEL_URL = os.environ.get("MATTEMODEL_URL", "http://127.0.0.1:48460")
SEGMODEL_URL = os.environ.get("SEGMODEL_URL", "http://127.0.0.1:48470")

# --- Local diffusion (inpaint / outpaint via stable-diffusion.cpp) ----------
# sd-cli is a subprocess; the model is switchable. Mask polarity for sd.cpp:
# white(255)=regenerate, black=keep — matches Krita's selection (selected=white).
_KRITA_ROOT = os.environ.get(
    "NULPAINT_KRITA_ROOT", os.path.expanduser("~/programming/Krita"))
SDCLI_BIN = os.environ.get(
    "SDCLI_BIN", os.path.join(_KRITA_ROOT, "stable-diffusion.cpp/build/bin/sd-cli"))
SD_MODELS = {
    "sd15": os.path.join(_KRITA_ROOT, "models/sd-v1-5-inpainting.ckpt"),
    "sdxl": os.path.join(_KRITA_ROOT, "models/sd_xl_base_1.0.safetensors"),
    # SD1.5 base — needed for the SD1.5 ControlNets (the inpaint sd15 won't do).
    "sd15base": os.path.join(_KRITA_ROOT, "models/sd-v1-5-base-fp16.safetensors"),
}
SD_DEFAULT_MODEL = os.environ.get("SD_DEFAULT_MODEL", "sd15")
# Native working resolution per model family (longest side, snapped to /64).
SD_NATIVE = {"sd15": 512, "sdxl": 1024, "sd15base": 512}
# Image-guidance for inpaint: lower => bolder prompt adherence (replace), higher
# (≈cfg) => seamless content-aware fill. 1.5 follows the prompt while staying coherent.
SD_IMG_CFG = float(os.environ.get("SD_IMG_CFG", "1.5"))

# Warm stable-diffusion.cpp daemon (sd-server). inpaint/outpaint/style POST here
# instead of cold-spawning sd-cli per call — the daemon holds SDXL resident, so a
# generation is one HTTP round-trip. See ~/programming/Models/diffusionmodel.
DIFFUSION_URL = os.environ.get("DIFFUSION_URL", "http://127.0.0.1:48480")
# SDXL inpainting daemon (the dedicated inpainting checkpoint) — inpaint/outpaint
# route here, generation/style to DIFFUSION_URL. Only one is GPU-resident at a time:
# the image-model "mode" swaps them via the modelmanager (the other parks in RAM).
INPAINT_URL = os.environ.get("INPAINT_URL", "http://127.0.0.1:48481")
DIFFUSION_SERVICE = "diffusionmodel"
INPAINT_SERVICE = "diffusionmodel-inpaint"
MODELMANAGER_STATE = os.path.expanduser("~/.cache/modelmanager/state.json")

# LoRAs: drop <name>.safetensors here, reference as <lora:name:weight> (or the
# --lora flag). ControlNets: <name>.safetensors here, selected by --control.
LORA_DIR = os.environ.get("NULPAINT_LORA_DIR", os.path.join(_KRITA_ROOT, "models/loras"))
CONTROLNET_DIR = os.environ.get(
    "NULPAINT_CONTROLNET_DIR", os.path.join(_KRITA_ROOT, "models/controlnet"))
# ControlNet model filenames (in CONTROLNET_DIR). These are SD1.5 ControlNets, so
# they pair with the sd15base model. canny=composition-lock, openpose=repose.
CONTROL_MODELS = {
    "canny": "control_v11p_sd15_canny_fp16.safetensors",
    "openpose": "control_v11p_sd15_openpose_fp16.safetensors",
}

# --- Cloud diffusion (Nano Banana Pro via OpenRouter) -----------------------
# An instruction-based image editor (NOT mask-based): prompt + up to ~5 reference
# images -> a full reimagined image. Used as an alternative `engine` for inpaint/
# outpaint/style. The whole base image is sent as a reference by default so the
# result fits the existing style. OpenAI-compatible chat/completions endpoint.
OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
NANOBANANA_MODEL = os.environ.get(
    "NANOBANANA_MODEL", "google/gemini-3-pro-image-preview")
# Max reference images the model accepts (base image + refs). Gemini 3 Pro Image
# keeps identity across ~5 subjects; keep the total attachment count sane.
NANOBANANA_MAX_REFS = int(os.environ.get("NANOBANANA_MAX_REFS", "6"))
_OPENROUTER_KEY_FILE = os.path.expanduser(
    os.environ.get("OPENROUTER_KEY_FILE", "~/.config/nulpaint/openrouter.key"))


def openrouter_api_key() -> str | None:
    """Resolve the OpenRouter API key: $OPENROUTER_API_KEY, else the key file
    (~/.config/nulpaint/openrouter.key, one line). None if neither is set."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        with open(_OPENROUTER_KEY_FILE, encoding="utf-8") as fh:
            line = fh.read().strip()
            return line or None
    except OSError:
        return None
