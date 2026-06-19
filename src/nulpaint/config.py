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
