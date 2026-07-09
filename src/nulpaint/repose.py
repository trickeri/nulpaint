"""Repose a character still into a new fixed pose via Nano Banana Pro, headless.

Feed a transparent StreamToons chibi still (Chibi_<Name>.png) and get back a clean
transparent still of the same character in a new pose (first use: a MapleStory-style
prone "lie down" frame, side profile facing right). No Krita in the loop — both the
generator (Nano Banana / OpenRouter) and the background stripper (mattemodel/segmodel)
are plain HTTP, so this is a pure CLI batch step.

Pipeline per still:
  transparent still --composite on white--> Nano Banana edit_image(pose prompt)
    --> prone on flat gray --matte strip--> trim + ground-anchor --> transparent PNG

The composite-on-white is essential: edit_image flattens RGBA->RGB before sending,
so a raw transparent PNG would reach the model as a character on BLACK with dark
edge fringing. We isolate the character on white instead (same trick segment_layers
uses), and prompt for a flat gray output background we then strip.
"""
from __future__ import annotations

import io
import os

from PIL import Image

from .config import MATTEMODEL_URL, SEGMODEL_URL
from .generate.nanobanana import edit_image
from .vision.select import _ensure_serving, _matte, fill_mask_holes

# Pose prompts. Identity-lock first, concrete geometry second, automation
# constraints (framing/background) third, negatives last. Keep them explicit and
# identical across the roster so 37 characters land in the same pose.
POSE_PROMPTS = {
    "prone": (
        "Redraw the SAME character from the reference image LYING PRONE ON THEIR BELLY "
        "ON THE GROUND, but AWAKE, ALERT and looking at the viewer — like a MapleStory "
        "'lie down' pose, or a person lying on their stomach propped up on their "
        "elbows. Body orientation: MOSTLY SIDE-ON, oriented to the RIGHT. "
        "The LOWER body is FLAT and PRONE: belly, hips and legs rest flat along the "
        "floor, legs trailing back. The UPPER body is propped up on the forearms/"
        "elbows so the chest and head are lifted a little off the ground. The HEAD is "
        "held UP and ALERT with the EYES OPEN, turned toward the VIEWER at a 3/4 angle "
        "so the FACE IS CLEARLY VISIBLE and the character is clearly AWAKE and looking "
        "at us. "
        "IMPORTANT: the character is NOT asleep, NOT passed out, the eyes are NOT "
        "closed, and the head is NOT lying flat on the floor. But it is ALSO NOT "
        "sitting up, NOT kneeling, NOT standing — the hips and legs stay flat on the "
        "ground. It is an awake, propped-up prone rest pose. The arms/hands rest on "
        "the floor in front; do NOT sweep them straight out in a stiff 'Superman' line. "
        "Keep EVERYTHING about the character identical to the reference — the EXACT "
        "same art style, line-art, rendering, colours, outfit, hair, hat, accessories "
        "and chibi proportions. Do NOT restyle or redesign; only change the pose and "
        "camera angle. The character has EXACTLY ONE head and ONE face — do not "
        "duplicate the head, face, or any body part. "
        "One single figure, full body in frame, centred, resting on the floor at the "
        "bottom with a small margin. Plain flat solid light-gray background (#b0b0b0). "
        "No scenery, no cast shadow, no ground texture, no other characters, no text, "
        "no border."
    ),
    # Fallback for characters that refuse to lie prone (bipeds keep sitting up) — a
    # relaxed awake SIT on the ground, MapleStory rest-on-the-floor feel.
    "sit": (
        "Redraw the SAME character from the reference image SITTING DOWN ON THE GROUND "
        "in a relaxed, awake rest pose, like a MapleStory character sitting on the "
        "floor. Body MOSTLY SIDE-ON, oriented to the RIGHT. The character sits with "
        "its hips down on the ground — legs folded, crossed, or tucked to one side (or "
        "knees drawn up) — with one or both hands resting on the ground or on a knee. "
        "The torso is upright but relaxed and leaning. The HEAD is up and AWAKE with "
        "EYES OPEN, turned toward the VIEWER at a 3/4 angle so the FACE IS CLEARLY "
        "VISIBLE. Clearly sitting on the floor and resting — NOT standing, NOT lying "
        "flat, NOT kneeling to attack. "
        "Keep EVERYTHING about the character identical to the reference — the EXACT "
        "same art style, line-art, rendering, colours, outfit, hair, hat, accessories "
        "and chibi proportions. Do NOT restyle or redesign; only change the pose and "
        "camera angle. The character has EXACTLY ONE head and ONE face — do not "
        "duplicate the head, face, or any body part. "
        "One single figure, full body in frame, centred, sitting on the floor at the "
        "bottom with a small margin. Plain flat solid light-gray background (#b0b0b0). "
        "No scenery, no cast shadow, no ground texture, no other characters, no text, "
        "no border."
    ),
    # Start frame for a "drinking" animation: the character in its normal STANDING
    # idle stance, but now holding a wooden beer stein up in front of it. Fed to Kling
    # as the drink-anim start frame (return-to-idle or ping-pong for the loop).
    "drink": (
        "Redraw the SAME character from the reference image STANDING UPRIGHT in the "
        "SAME relaxed idle stance as the reference, but now HOLDING A WOODEN BEER STEIN "
        "raised UP IN FRONT of its chest, as if about to take a drink or toasting. "
        "The stein is a classic tankard/mug: a wooden barrel-staved body with metal "
        "bands and a handle, filled with frothy beer with foam on top. It is held "
        "FIRMLY IN ONE HAND, fingers gripping the handle or wrapped around the mug, at "
        "roughly chest height in front of the body. "
        "The character keeps standing on its feet in its normal idle posture with the "
        "SAME body orientation as the reference (facing/angled to the RIGHT); only ONE "
        "arm is changed to bring the stein up in front. If that hand already holds an "
        "item (a staff, weapon, etc.), keep that item in the OTHER hand and use the "
        "free hand for the stein — do NOT add extra arms or hands. The HEAD is up and "
        "AWAKE with EYES OPEN, turned toward the VIEWER at a 3/4 angle so the FACE IS "
        "CLEARLY VISIBLE. "
        "The facial EXPRESSION stays FROZEN and IDENTICAL to the reference — the exact "
        "same idle face, same eyes, same mouth. Do NOT change the expression; the ONLY "
        "change is that the character is now holding the stein. "
        "CRITICAL: the beer stein is a NORMAL hand-held size held IN the hand — it is "
        "NOT giant, NOT oversized, NOT floating, and NOT flying in from the side of the "
        "frame. It must look like the character is naturally holding it. "
        "Keep EVERYTHING about the character identical to the reference — the EXACT "
        "same art style, line-art, rendering, colours, outfit, hair, hat, accessories "
        "and chibi proportions. Do NOT restyle or redesign; only change the pose to "
        "hold the stein. The character has EXACTLY ONE head and ONE face — do not "
        "duplicate the head, face, or any body part. "
        "One single figure, full body in frame, standing centred at the bottom with a "
        "small margin. Plain flat solid CHROMA-KEY GREEN background (#00B140), the same "
        "green as the reference image. No scenery, no cast shadow, no ground texture, "
        "no other characters, no text, no border."
    ),
}


def _composite_on(im: Image.Image, color=(255, 255, 255)) -> Image.Image:
    """Flatten an RGBA still onto a solid colour so the model sees an isolated character
    (not the black field a raw RGBA->RGB flatten would produce). White by default; pass
    the chroma green to feed nano a greenscreen source (any bg it leaves then keys/reads
    as green, i.e. invisible on the final greenscreen, instead of a stray gray blob)."""
    im = im.convert("RGBA")
    bg = Image.new("RGB", im.size, tuple(color))
    bg.paste(im, (0, 0), im)
    return bg


def _composite_on_white(im: Image.Image) -> Image.Image:
    return _composite_on(im, (255, 255, 255))


def _strip_bg(gen: Image.Image, kind: str = "object",
              fill_holes: bool = True) -> Image.Image:
    """Matte the generated image's flat background off via the model service.

    kind='object' (default) → segmodel salient-object matte: pose- and species-
    agnostic, the safe pick for a mixed roster (humans + gremlins + robots).
    kind='person' → mattemodel (RVM human matte).
    """
    url = SEGMODEL_URL if kind == "object" else MATTEMODEL_URL
    service = "segmodel" if kind == "object" else "mattemodel"
    _ensure_serving(service, url)
    buf = io.BytesIO()
    gen.convert("RGB").save(buf, "PNG")
    mask_png = _matte(url, buf.getvalue())
    if fill_holes:
        mask_png = fill_mask_holes(mask_png)
    mask = Image.open(io.BytesIO(mask_png)).convert("L")
    if mask.size != gen.size:
        mask = mask.resize(gen.size, Image.LANCZOS)
    out = gen.convert("RGBA")
    out.putalpha(mask)
    return out


def _anchor(cut: Image.Image, canvas: int = 1024, margin: float = 0.06,
            ground_frac: float = 0.80) -> Image.Image:
    """Trim to content and place on a canvas×canvas transparent frame, horizontally
    centred with the silhouette BOTTOM sitting on a ground line (ground_frac of the
    canvas). Mirrors the standing stills' feet-anchored convention so the prone frame
    drops into the game at a matching scale/ground."""
    bb = cut.getbbox()
    if bb is None:
        return cut
    char = cut.crop(bb)
    cw, ch = char.size
    max_w = canvas * (1.0 - 2.0 * margin)
    ground_y = int(canvas * ground_frac)
    max_h = ground_y - int(canvas * margin)
    scale = min(max_w / cw, max_h / ch)
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    char = char.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.paste(char, ((canvas - nw) // 2, ground_y - nh))
    return out


def _register_to_ref(cut: Image.Image, ref_path: str,
                     green: tuple[int, int, int] | None = None) -> Image.Image:
    """Scale + place the matted character to MATCH a reference framed still (e.g. the
    idle Chibi_<Name>_Padded.png), so a generated pose registers with the existing
    Kling frame for a clean start<->end / ping-pong. The character is scaled so its
    silhouette height equals the reference's, then aligned feet-to-feet (bbox bottom)
    and centred horizontally on the reference's content centre. Output canvas matches
    the reference size; if `green` is given the background is filled with that solid
    colour (chroma key) instead of left transparent."""
    ref = Image.open(ref_path).convert("RGBA")
    rb = ref.getbbox()
    if rb is None:
        rb = (0, 0, ref.width, ref.height)
    rx0, ry0, rx1, ry1 = rb
    tgt_cx = (rx0 + rx1) / 2.0
    tgt_bottom = ry1
    tgt_h = ry1 - ry0

    bb = cut.getbbox()
    if bb is None:
        char = cut
        cw, ch = cut.size
    else:
        char = cut.crop(bb)
        cw, ch = char.size
    scale = tgt_h / ch if ch else 1.0
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    char = char.resize((nw, nh), Image.LANCZOS)

    canvas = Image.new("RGBA", ref.size,
                       (green[0], green[1], green[2], 255) if green else (0, 0, 0, 0))
    x = round(tgt_cx - nw / 2.0)
    y = round(tgt_bottom - nh)
    canvas.alpha_composite(char, (x, y))
    return canvas


ANIM_ROOT = "/mnt/storage1/Pictures/Nuldrums/StreamToons/Animations"
NOBG_DIR = "/mnt/storage1/Pictures/Nuldrums/StreamToons/Chibi_NoBG"
# Animation folders that are not a poseable character.
_SKIP_FOLDERS = {"Projectiles"}


def resolve_still(folder: str, nobg_dir: str = NOBG_DIR) -> str | None:
    """Find the source chibi still for an animation folder, bridging the naming
    drift (Gremlin1 -> Chibi_EnemyGremlin1, Nully -> Chibi_Nully1, etc.)."""
    for cand in (f"Chibi_{folder}.png", f"Chibi_{folder}1.png",
                 f"Chibi_Enemy{folder}.png", f"Chibi_Enemy{folder}1.png"):
        p = os.path.join(nobg_dir, cand)
        if os.path.exists(p):
            return p
    return None


def repose_roster(*, anim_root: str = ANIM_ROOT, nobg_dir: str = NOBG_DIR,
                  pose: str = "prone", only: list[str] | None = None,
                  limit: int = 0, dry_run: bool = False, kind: str = "object",
                  model: str | None = None, review_dir: str | None = None,
                  on_progress=None) -> dict:
    """For every character folder under `anim_root`, resolve its source still, generate
    a `<pose>` frame, and drop `<Char>/<Char>_<Pose>.png` into the folder. Raw gens +
    a contact sheet go to `review_dir` (asset folders keep only the deliverable). With
    `dry_run` only the folder->still mapping is resolved and returned (no API calls)."""
    from PIL import Image as _Image  # local: contact sheet only

    folders = sorted(d for d in os.listdir(anim_root)
                     if os.path.isdir(os.path.join(anim_root, d))
                     and d not in _SKIP_FOLDERS and not d.startswith("_"))
    if only:
        want = set(only)
        folders = [f for f in folders if f in want]
    if limit:
        folders = folders[:limit]
    if review_dir is None:
        review_dir = os.path.join(anim_root, "_prone_review")

    report, finals = [], []
    for folder in folders:
        still = resolve_still(folder, nobg_dir)
        row = {"char": folder, "still": still}
        if still is None:
            row["status"] = "NO SOURCE STILL"
            report.append(row)
            continue
        out = os.path.join(anim_root, folder, f"{folder}_{pose.capitalize()}.png")
        row["out"] = out
        if dry_run:
            row["status"] = "ok (dry-run)"
            report.append(row)
            continue
        try:
            os.makedirs(review_dir, exist_ok=True)
            raw = os.path.join(review_dir, f"{folder}_{pose}.raw.png")
            if os.path.exists(out):                       # back up any prior deliverable
                bak = os.path.join(anim_root, folder, "_prone_bak")
                os.makedirs(bak, exist_ok=True)
                Image.open(out).save(os.path.join(bak, os.path.basename(out)))
            res = repose_still(still, out, pose=pose, kind=kind, model=model,
                               raw_out=raw)
            row.update(status="ok", bbox=res["final_bbox"])
            finals.append((folder, out))
        except Exception as e:  # noqa: BLE001 — keep going; re-roll misses later
            row["status"] = f"error: {e}"
        report.append(row)
        if on_progress:
            on_progress(row)

    sheet = None
    if finals:                                            # contact sheet for review
        cols = 4
        cell = 256
        rows = (len(finals) + cols - 1) // cols
        sheet_img = _Image.new("RGBA", (cols * cell, rows * cell), (40, 40, 40, 255))
        for i, (name, path) in enumerate(finals):
            th = _Image.open(path).convert("RGBA")
            th.thumbnail((cell, cell), _Image.LANCZOS)
            cx = (i % cols) * cell + (cell - th.width) // 2
            cy = (i // cols) * cell + (cell - th.height) // 2
            sheet_img.alpha_composite(th, (cx, cy))
        sheet = os.path.join(review_dir, f"contact_sheet_{pose}.png")
        os.makedirs(review_dir, exist_ok=True)
        sheet_img.save(sheet)

    ok = sum(1 for r in report if r["status"] == "ok")
    return {"anim_root": anim_root, "count": len(report), "ok": ok,
            "review_dir": review_dir, "contact_sheet": sheet, "report": report}


def repose_still(in_path: str, out_path: str, *, pose: str = "prone",
                 prompt: str | None = None, extra: str | None = None,
                 refs: list[str] | None = None, kind: str = "object",
                 model: str | None = None, anchor: bool = True,
                 raw_out: str | None = None, pad_ref: str | None = None,
                 green_bg: tuple[int, int, int] | None = None) -> dict:
    """Generate a new-pose transparent still from `in_path` → `out_path`.

    refs: extra reference image paths (each isolated on white too) for identity.
    raw_out: where to keep the pre-strip generation (defaults to <out>.raw.png).
    pad_ref: register the result to this framed still's scale/placement (e.g. the
      idle Chibi_<Name>_Padded.png) instead of the default ground-anchor — for a
      pose that must line up with an existing Kling frame.
    green_bg: fill the output background with this solid RGB (chroma key) — for a
      Kling-ready greenscreen frame instead of transparency.
    """
    base = Image.open(in_path).convert("RGBA")
    text = prompt or POSE_PROMPTS.get(pose)
    if not text:
        raise ValueError(f"unknown pose {pose!r}; known: {sorted(POSE_PROMPTS)}")
    if extra:
        text = f"{text} {extra}"

    base_bg = green_bg if green_bg else (255, 255, 255)
    images = [_composite_on(base, base_bg)]
    for r in (refs or []):
        images.append(_composite_on_white(Image.open(r)))

    gen = edit_image(text, images, model=model)          # RGBA, opaque background

    if raw_out is None:
        rp, _ = os.path.splitext(out_path)
        raw_out = rp + ".raw.png"
    gen.save(raw_out)

    cut = _strip_bg(gen, kind=kind)
    if pad_ref:
        final = _register_to_ref(cut, pad_ref, green=green_bg)
    elif anchor:
        final = _anchor(cut)
        if green_bg:
            bg = Image.new("RGBA", final.size, (*green_bg, 255))
            bg.alpha_composite(final)
            final = bg
    else:
        final = cut
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    final.save(out_path)
    return {"in": in_path, "out": out_path, "raw": raw_out, "pose": pose,
            "kind": kind, "gen_size": gen.size, "final_bbox": cut.getbbox()}
