"""Build a per-character animation-cleanup .kra by driving the bridge.

One doc per character; a GROUP per animation; each animation's frames live on
Krita's animation timeline (one animated paint layer per clip). Fully automated —
no manual GUI setup: we decode each `<Char>_<Anim>_4444.mov` (ProRes+alpha) to
transparent PNG frames with ffmpeg, then `document.import_animation` loads them
onto a new animated layer, which we wrap in a group named after the animation.

The timeline is document-global, so every clip's frames start at 0 and overlap on
the same frame axis; solo one animation-group at a time to review/clean it. Cleaned
frames go back out via the re-export step (Phase 3), which scrubs the timeline with
`document.set_frame` + `layer.get_region`.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile

from .bridge import BridgeClient


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def discover_clips(src_dir: str, char: str) -> list[tuple[str, str]]:
    """[(anim_name, mov_path)] for every `<char>_*_4444.mov`, sorted (Idle first)."""
    out = []
    for p in sorted(glob.glob(os.path.join(src_dir, f"{char}_*_4444.mov"))):
        stem = os.path.basename(p)[:-len("_4444.mov")]           # Trikeri_Idle_Loop
        anim = stem[len(char) + 1:] if stem.startswith(char + "_") else stem
        out.append((anim, p))
    # Idle/idle animations first, then the rest alphabetically — nicer stack order.
    out.sort(key=lambda t: (0 if "idle" in t[0].lower() else 1, t[0].lower()))
    return out


def probe(mov: str) -> dict:
    """fps (rounded int), frame count, and WxH of a clip."""
    fps_raw = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=r_frame_rate", "-of",
                    "default=noprint_wrappers=1:nokey=1", mov])
    num, den = (fps_raw.split("/") + ["1"])[:2]
    fps = round(float(num) / float(den or 1))
    wh = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", mov])
    w, h = (wh.split("x") + ["1024", "1024"])[:2]
    return {"fps": fps, "w": int(w), "h": int(h)}


def decode_frames(mov: str, out_dir: str) -> list[str]:
    """Decode a ProRes+alpha clip to 8-bit RGBA PNG frames (rgba = 8bpc, so no
    16-bit surprise). Returns the sorted frame paths."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mov,
                    "-vsync", "0", "-pix_fmt", "rgba",
                    os.path.join(out_dir, "%05d.png")], check=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def frame_count(mov: str) -> int:
    """Exact frame count of a clip (nb_frames, falling back to a decode count)."""
    n = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=nb_frames", "-of", "default=nk=1:nw=1", mov])
    if n.isdigit():
        return int(n)
    n = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
              "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1", mov])
    return int(n)


def _backup_once(path: str) -> bool:
    """Copy path -> path.bak, but only if no .bak exists yet (preserve the pristine
    original across repeated re-exports). Returns True if a backup was made."""
    if os.path.exists(path) and not os.path.exists(path + ".bak"):
        shutil.copy2(path, path + ".bak")
        return True
    return False


def _encode_prores(frames_dir: str, fps: int, out_mov: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", os.path.join(frames_dir, "%05d.png"),
                    "-c:v", "prores_ks", "-profile:v", "4444",
                    "-pix_fmt", "yuva444p10le", "-vendor", "apl0", out_mov], check=True)


def _encode_webm(src_mov: str, out_webm: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src_mov,
                    "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                    "-b:v", "0", "-crf", "20", "-auto-alt-ref", "0", out_webm], check=True)


def _verify_alpha(mov: str) -> dict:
    """Sample a few frames of the output, report opaque% and residual green%."""
    import numpy as np
    import cv2
    from .despill import green_mask
    n = frame_count(mov)
    sample = sorted({0, n // 2, max(n - 1, 0)})
    op_solid, green_frac = [], []
    for t in sample:
        d = tempfile.mkdtemp(prefix="nulpaint-verify-")
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mov,
                            "-vf", f"select=eq(n\\,{t})", "-frames:v", "1",
                            "-pix_fmt", "rgba", os.path.join(d, "f.png")], check=True)
            raw = cv2.imread(os.path.join(d, "f.png"), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.dtype != np.uint8:
                raw = (raw >> 8).astype(np.uint8)
            rgba = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
            a = rgba[..., 3]
            drawn = int((a > 16).sum())
            op_solid.append(100.0 * int((a == 255).sum()) / max(drawn, 1))
            green_frac.append(100.0 * int(green_mask(rgba, 22, 8).sum()) / max(drawn, 1))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    avg = lambda xs: round(float(sum(xs)) / len(xs), 2) if xs else None
    return {"opaque_pct": avg(op_solid), "residual_green_pct": avg(green_frac)}


def _require_active_doc(client: BridgeClient, char: str) -> None:
    """Guard: the anim-frame verbs operate on the ACTIVE doc, but per-character
    cleanup docs share group names (Idle_Forward, Attack1, …), so the group-match
    check alone can't tell e.g. Magi2 from LadyVermilia. Refuse unless the active
    document's name matches the requested character."""
    active = (client.call("document.info").get("name") or "").strip()
    if active.lower() != char.lower():
        raise RuntimeError(
            f"active document is '{active or '(unsaved/untitled)'}', not '{char}' — "
            f"switch to the {char} tab in Krita first")


def punch_anim_frames(client: BridgeClient, *, char: str, src_dir: str,
                      only: list[str] | None = None, saturation: int = 35,
                      value: int = 0, black: int = 18, white: int = 245,
                      gamma: float = 1.0) -> dict:
    """Bake the de-pastel "punch" (levels + HSL saturation) into EVERY keyframe of
    each animation in the open <Char>.kra. A filter mask would be invisible to the
    re-export (which reads raw layer pixels via layer.get_region → node.pixelData),
    so we scrub each animation's timeline and apply the filters destructively to the
    layer's current-frame device. Frame count per clip comes from its source
    `_4444.mov` (import was 1:1, all clips start at frame 0). Requires <Char>.kra to
    be the active document."""
    # anim group name -> inner animated paint layer uuid (same map as the exporter)
    layer_of = {}
    for g in client.call("node.tree")["tree"]:
        if g["type"] == "grouplayer" and g.get("children"):
            for c in g["children"]:
                if c["type"] == "paintlayer":
                    layer_of[g["name"]] = c["uuid"]
                    break

    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no matching clips for {char} in {src_dir}")
    missing = [a for a, _ in clips if a not in layer_of]
    if missing:
        raise RuntimeError(f"open doc has no group(s) for: {', '.join(missing)} "
                           f"— is {char}.kra the active document?")

    _require_active_doc(client, char)
    do_levels = black > 0 or white < 255 or abs(gamma - 1.0) > 1e-6
    do_hsv = saturation != 0 or value != 0
    orig_time = client.call("document.frame_info")["currentTime"]
    results = []
    try:
        for anim, mov in clips:
            n = frame_count(mov)
            uuid = layer_of[anim]
            for t in range(n):
                client.call("document.set_frame", time=t)
                if do_levels:
                    client.call("filter.apply", filter="levels", node=uuid,
                                refresh=False,
                                config={"blackvalue": int(black), "whitevalue": int(white),
                                        "gammavalue": float(gamma)})
                if do_hsv:
                    client.call("filter.apply", filter="hsvadjustment", node=uuid,
                                refresh=False,
                                config={"h": 0, "s": int(saturation), "v": int(value),
                                        "type": 1, "colorize": False})
            results.append({"anim": anim, "frames": n})
    finally:
        client.call("document.set_frame", time=orig_time)
    return {"char": char, "clips": results,
            "params": {"saturation": saturation, "value": value, "black": black,
                       "white": white, "gamma": gamma}}


def rebuild_clip(client: BridgeClient, *, char: str, src_dir: str, anim: str,
                 from_bak: bool = True) -> dict:
    """Replace a single animation clip in the open <Char>.kra with fresh frames from
    its PRISTINE source (the `_4444.mov.bak` written by the first export, i.e. the
    green-cleaned pre-punch original) — deletes the existing <anim> group and
    re-imports. Use to redo one clip cleanly (e.g. after a stray edit) without
    rebuilding the whole doc; follow with punch/apply-filter/despill/export scoped
    to --clips <anim>. Requires <Char>.kra active."""
    _require_active_doc(client, char)
    mov = os.path.join(src_dir, f"{char}_{anim}_4444.mov")
    bak = mov + ".bak"
    src_mov = bak if (from_bak and os.path.exists(bak)) else mov
    if not os.path.exists(src_mov):
        raise RuntimeError(f"no source clip for {anim}: {src_mov}")
    fps = probe(src_mov)["fps"]
    for g in client.call("node.tree")["tree"]:
        if g["name"] == anim and g["type"] == "grouplayer":
            client.call("node.delete", uuid=g["uuid"])
            break
    tmp = tempfile.mkdtemp(prefix=f"nulpaint-rebuild-{char}-{anim}-")
    try:
        frames = decode_frames(src_mov, os.path.join(tmp, anim))
        res = client.call("document.import_animation", files=frames,
                          first_frame=0, step=1, name=anim, fps=fps)
        grp = client.call("node.create_group", name=anim)
        client.call("node.move", node=res["uuid"], parent=grp["uuid"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"char": char, "anim": anim, "frames": res["frames"],
            "source": os.path.basename(src_mov)}


def apply_filter_anim_frames(client: BridgeClient, *, char: str, src_dir: str,
                             filter_id: str, config: dict,
                             only: list[str] | None = None) -> dict:
    """Bake an arbitrary Krita filter (id + config dict) into EVERY keyframe of each
    animation in the open <Char>.kra — e.g. a Cross-channel curve read off another
    doc's filter mask via `filter.read_config`. Same per-frame scrub as punch-anim.
    Requires <Char>.kra active."""
    _require_active_doc(client, char)
    layer_of = {}
    for g in client.call("node.tree")["tree"]:
        if g["type"] == "grouplayer" and g.get("children"):
            for c in g["children"]:
                if c["type"] == "paintlayer":
                    layer_of[g["name"]] = c["uuid"]
                    break

    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no matching clips for {char} in {src_dir}")
    missing = [a for a, _ in clips if a not in layer_of]
    if missing:
        raise RuntimeError(f"open doc has no group(s) for: {', '.join(missing)}")

    orig_time = client.call("document.frame_info")["currentTime"]
    results = []
    try:
        for anim, mov in clips:
            n = frame_count(mov)
            uuid = layer_of[anim]
            for t in range(n):
                client.call("document.set_frame", time=t)
                client.call("filter.apply", filter=filter_id, node=uuid,
                            refresh=False, config=config)
            results.append({"anim": anim, "frames": n})
    finally:
        client.call("document.set_frame", time=orig_time)
    return {"char": char, "filter": filter_id, "clips": results}


def despill_anim_frames(client: BridgeClient, *, char: str, src_dir: str,
                        only: list[str] | None = None, thr: int = 22,
                        edge: int = 8, grow: int = 2, amin: int = 8) -> dict:
    """Green-eat the outer alpha edge of EVERY keyframe of each animation in the open
    <Char>.kra. Run AFTER punch-anim: the contrast boost makes residual chroma-green
    fringe more green-dominant (easier to detect), and it's exactly the leftover
    matte+despill ring the green-eater is built to recolour (silhouette/alpha kept
    bit-for-bit, RGB pulled from the nearest clean body pixel). Scoped to a band of
    `edge` px just inside the alpha boundary so any intentionally-green interior is
    never touched. Requires <Char>.kra active."""
    import cv2  # local: only the external venv has cv2
    import numpy as np
    from .despill import green_eat, _b64_to_rgba, _rgba_to_b64

    _require_active_doc(client, char)
    layer_of = {}
    for g in client.call("node.tree")["tree"]:
        if g["type"] == "grouplayer" and g.get("children"):
            for c in g["children"]:
                if c["type"] == "paintlayer":
                    layer_of[g["name"]] = c["uuid"]
                    break

    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no matching clips for {char} in {src_dir}")
    missing = [a for a, _ in clips if a not in layer_of]
    if missing:
        raise RuntimeError(f"open doc has no group(s) for: {', '.join(missing)} "
                           f"— is {char}.kra the active document?")

    info = client.call("document.info")
    W, H = info["width"], info["height"]
    orig_time = client.call("document.frame_info")["currentTime"]
    results = []
    try:
        for anim, mov in clips:
            n = frame_count(mov)
            uuid = layer_of[anim]
            client.call("node.set_active", node=uuid)      # get/set_region -> this layer
            recol_frames, total_px = 0, 0
            for t in range(n):
                client.call("document.set_frame", time=t)
                rgba = _b64_to_rgba(client.call("layer.get_region", x=0, y=0, w=W, h=H)["png_b64"])
                a = rgba[..., 3]
                solid = a > amin
                if edge > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * edge + 1, 2 * edge + 1))
                    near_transparent = cv2.dilate((~solid).astype(np.uint8), k).astype(bool)
                    region = solid & near_transparent      # band just inside the edge
                else:
                    region = None
                fixed, target = green_eat(rgba, thr=thr, amin=amin, grow=grow, region=region)
                nn = int(target.sum())
                if nn:
                    client.call("layer.set_region", x=0, y=0, png_b64=_rgba_to_b64(fixed))
                    recol_frames += 1
                    total_px += nn
            results.append({"anim": anim, "frames": n,
                            "recoloured_frames": recol_frames, "px": total_px})
    finally:
        client.call("document.set_frame", time=orig_time)
    return {"char": char, "clips": results,
            "params": {"thr": thr, "edge": edge, "grow": grow}}


def decyan_anim_frames(client: BridgeClient, *, char: str, src_dir: str,
                       only: list[str] | None = None, thr: int = 22,
                       amin: int = 8) -> dict:
    """Remove green-screen spill that reads as CYAN on a blue/purple subject, on EVERY
    keyframe of each animation in the open <Char>.kra.

    Green light landing on the character's blue tones pushes G up to ~B, so the pixel
    goes cyan (G≈B, R suppressed) — which the green-eat / klingdespill despills can't see
    (they cap green OVER max(R,B), and here G≈B so it never bites). Instead we cap G to
    R+thr wherever G>R+thr, pulling the cyan back toward the subject's own blue/neutral
    tone. RGB-only, alpha untouched, self-scoping: only pixels with G>R+thr change, so
    blue-DOMINANT hair (B>G) and neutral/white areas are left alone. Run AFTER punch-anim
    (the punch amplifies the cast). Requires <Char>.kra to be the active document."""
    import numpy as np

    from .despill import _b64_to_rgba, _rgba_to_b64

    _require_active_doc(client, char)
    layer_of = {}
    for g in client.call("node.tree")["tree"]:
        if g["type"] == "grouplayer" and g.get("children"):
            for c in g["children"]:
                if c["type"] == "paintlayer":
                    layer_of[g["name"]] = c["uuid"]
                    break

    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no matching clips for {char} in {src_dir}")
    missing = [a for a, _ in clips if a not in layer_of]
    if missing:
        raise RuntimeError(f"open doc has no group(s) for: {', '.join(missing)} "
                           f"— is {char}.kra the active document?")

    info = client.call("document.info")
    W, H = info["width"], info["height"]
    orig_time = client.call("document.frame_info")["currentTime"]
    results = []
    try:
        for anim, mov in clips:
            n = frame_count(mov)
            uuid = layer_of[anim]
            client.call("node.set_active", node=uuid)      # get/set_region -> this layer
            recol_frames, total_px = 0, 0
            for t in range(n):
                client.call("document.set_frame", time=t)
                rgba = _b64_to_rgba(client.call("layer.get_region", x=0, y=0, w=W, h=H)["png_b64"])
                a = rgba.astype(np.int16)
                r, gch, al = a[..., 0], a[..., 1], a[..., 3]
                target = (al > amin) & (gch > r + thr)
                nn = int(target.sum())
                if nn:
                    a[..., 1] = np.where(target, r + thr, gch)   # cap G to R+thr (== min here)
                    client.call("layer.set_region", x=0, y=0,
                                png_b64=_rgba_to_b64(a.astype(np.uint8)))
                    recol_frames += 1
                    total_px += nn
            results.append({"anim": anim, "frames": n,
                            "recoloured_frames": recol_frames, "px": total_px})
    finally:
        client.call("document.set_frame", time=orig_time)
    return {"char": char, "clips": results, "params": {"thr": thr}}


def export_anim_doc(client: BridgeClient, *, char: str, src_dir: str,
                    out_dir: str | None = None, only: list[str] | None = None,
                    solidify: bool = True, fps_override: int | None = None) -> dict:
    """Re-export cleaned clips from the OPEN character .kra: scrub each animation's
    timeline layer back out to `_4444.mov` + `.webm`, solidify (character clips),
    verify alpha, backing up the pristine originals to `.bak` once. Requires the
    cleaned Character.kra to be the active document."""
    out_dir = out_dir or src_dir
    # Map animation name -> its inner animated paint layer uuid (group children).
    layer_of = {}
    for g in client.call("node.tree")["tree"]:
        if g["type"] == "grouplayer" and g.get("children"):
            for c in g["children"]:
                if c["type"] == "paintlayer":
                    layer_of[g["name"]] = c["uuid"]
                    break

    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no matching clips for {char} in {src_dir}")
    missing = [a for a, _ in clips if a not in layer_of]
    if missing:
        raise RuntimeError(f"open doc has no group(s) for: {', '.join(missing)} "
                           f"— is {char}.kra the active document?")

    info = client.call("document.info")
    W, H = info["width"], info["height"]
    tmp_root = tempfile.mkdtemp(prefix=f"nulpaint-export-{char}-")
    results = []
    try:
        for anim, mov in clips:
            fps = fps_override or probe(mov)["fps"]
            n = frame_count(mov)
            client.call("node.set_active", node=layer_of[anim])
            fdir = os.path.join(tmp_root, anim)
            os.makedirs(fdir, exist_ok=True)
            for t in range(n):
                client.call("document.set_frame", time=t)
                png = _b64_png(client.call("layer.get_region", x=0, y=0, w=W, h=H)["png_b64"])
                with open(os.path.join(fdir, "%05d.png" % t), "wb") as fh:
                    fh.write(png)

            out_mov = os.path.join(out_dir, f"{char}_{anim}_4444.mov")
            out_webm = os.path.join(out_dir, f"{char}_{anim}.webm")
            backed = _backup_once(out_mov) | _backup_once(out_webm)
            _encode_prores(fdir, fps, out_mov)
            if solidify:
                subprocess.run(["klingsolidify", out_mov], check=True)
            _encode_webm(out_mov, out_webm)
            shutil.rmtree(fdir, ignore_errors=True)
            results.append({"anim": anim, "frames": n, "fps": fps, "mov": out_mov,
                            "webm": out_webm, "backed_up": backed,
                            "verify": _verify_alpha(out_mov)})
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return {"char": char, "out_dir": out_dir, "solidified": solidify, "clips": results}


def _b64_png(s: str) -> bytes:
    import base64
    return base64.b64decode(s)


def build_anim_doc(client: BridgeClient, *, char: str, src_dir: str, out_path: str,
                   only: list[str] | None = None, fps: int | None = None,
                   canvas: tuple[int, int] | None = None) -> dict:
    clips = discover_clips(src_dir, char)
    if only:
        want = {c.lower() for c in only}
        clips = [(a, p) for (a, p) in clips if a.lower() in want]
    if not clips:
        raise RuntimeError(f"no {char}_*_4444.mov clips found in {src_dir}")

    meta = {a: probe(p) for a, p in clips}
    if canvas is None:
        first = meta[clips[0][0]]
        canvas = (first["w"], first["h"])
    if fps is None:
        seen = sorted({m["fps"] for m in meta.values()})
        fps = seen[0]                                            # clips for one char match
    w, h = canvas

    client.call("document.create", width=w, height=h, name=char,
                colorModel="RGBA", colorDepth="U8", resolution=72.0)
    # Drop the default opaque Background layer — these frames are transparent sprites.
    pre = client.call("node.tree")["tree"]
    for n in pre:
        if n["type"] == "paintlayer":
            client.call("node.delete", uuid=n["uuid"])

    tmp_root = tempfile.mkdtemp(prefix=f"nulpaint-anim-{char}-")
    built = []
    groups = []
    try:
        for anim, mov in clips:
            frames = decode_frames(mov, os.path.join(tmp_root, anim))
            res = client.call("document.import_animation", files=frames,
                              first_frame=0, step=1, name=anim, fps=fps)
            grp = client.call("node.create_group", name=anim)
            client.call("node.move", node=res["uuid"], parent=grp["uuid"])
            groups.append(grp["uuid"])
            built.append({"anim": anim, "frames": res["frames"],
                          "animated": res["animated"], "src_fps": meta[anim]["fps"]})
            shutil.rmtree(os.path.join(tmp_root, anim), ignore_errors=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # The timeline is document-global, so all clips overlap on the same frame axis.
    # Default to soloing the first (Idle) group; solo others in the UI while cleaning.
    for uid in groups[1:]:
        client.call("node.set_visible", uuid=uid, visible=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    client.call("document.save", path=out_path)
    return {"char": char, "out": out_path, "fps": fps, "canvas": [w, h],
            "clips": built}
