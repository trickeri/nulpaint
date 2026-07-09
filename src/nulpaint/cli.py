"""NulPaint command-line entry point.

A thin operator/dev front-end over the bridge:

  nulpaint launch [--no-focus]   launch the forked Krita (optionally without
                                 stealing window focus — see below)
  nulpaint ping                  round-trip the in-Krita socket server
  nulpaint info                  print the active document's info
  nulpaint demo [--preset NAME]  create a doc + draw circles/squares (smoke test)
  nulpaint call CMD [--args J]   send an arbitrary bridge command
  nulpaint no-focus-rule add|remove   manage the KWin focus rule directly

Focus stealing
--------------
Krita has no "don't activate me" launch flag, and on KDE Wayland focus is the
window manager's call, not the app's. So ``--no-focus`` installs a persistent
KWin window rule (focus-stealing-prevention = Extreme) scoped to Krita's window
class, then asks KWin to reload. Krita then opens in the background without
yanking focus from whatever you're doing. Remove it any time with
``nulpaint no-focus-rule remove``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .bridge import BridgeClient, BridgeError
from .config import BRIDGE_PORT, INSTANCE_DIR


def _current_port() -> int:
    """The bridge port to talk to: --port (mirrored into $NULPAINT_PORT by main)
    else $NULPAINT_PORT else the 8765 default. Re-read per call so `--port` and a
    per-shell env both work."""
    return int(os.environ.get("NULPAINT_PORT", str(BRIDGE_PORT)))

# --- document presets -------------------------------------------------------
# Krita's Python API has no "named built-in preset" call, so a preset here is
# just (width, height, resolution-ppi). "1080Land" == 1080p landscape.
PRESETS: dict[str, tuple[int, int, float]] = {
    "1080Land": (1920, 1080, 72.0),
    "1080Port": (1080, 1920, 72.0),
    "4KLand": (3840, 2160, 72.0),
    "Square": (1080, 1080, 72.0),
}

# --- KWin no-focus rule -----------------------------------------------------
_KWIN_FILE = "kwinrulesrc"
# Fixed group id so applying the rule is idempotent across runs.
_RULE_ID = "{f5a9c7e1-3b2d-4e8a-9f10-7c6b5a4d3e2f}"
_RULE_KEYS = {
    "Description": "nulpaint: krita no focus steal",
    "wmclass": "krita",
    "wmclassmatch": "1",   # 1 = exact match
    "fsplevel": "4",        # 4 = Extreme focus-stealing prevention
    "fsplevelrule": "2",    # 2 = Force
}


def _krita_binary(explicit: str | None) -> str:
    """Resolve the Krita binary, preferring the local fork install."""
    if explicit:
        return explicit
    env = os.environ.get("NULPAINT_KRITA")
    if env:
        return env
    fork = Path.home() / ".local/bin/krita"
    if fork.exists():
        return str(fork)
    found = shutil.which("krita")
    if not found:
        sys.exit("nulpaint: cannot find a 'krita' binary (set NULPAINT_KRITA or --krita)")
    return found


def _kreadconfig(group: str, key: str) -> str:
    try:
        out = subprocess.run(
            ["kreadconfig6", "--file", _KWIN_FILE, "--group", group, "--key", key],
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _kwriteconfig(group: str, key: str, value: str) -> None:
    subprocess.run(
        ["kwriteconfig6", "--file", _KWIN_FILE, "--group", group, "--key", key, value],
        check=True)


def _kwin_reconfigure() -> None:
    for tool in ("qdbus6", "qdbus"):
        if shutil.which(tool):
            subprocess.run([tool, "org.kde.KWin", "/KWin", "org.kde.KWin.reconfigure"],
                           check=False)
            return


def add_no_focus_rule() -> None:
    """Install (idempotently) the KWin rule that stops Krita stealing focus."""
    if not shutil.which("kwriteconfig6"):
        sys.exit("nulpaint: kwriteconfig6 not found — is this a KDE session?")
    ids = [r for r in _kreadconfig("General", "rules").split(",") if r]
    if _RULE_ID not in ids:
        ids.append(_RULE_ID)
        _kwriteconfig("General", "rules", ",".join(ids))
        _kwriteconfig("General", "count", str(len(ids)))
    for key, value in _RULE_KEYS.items():
        _kwriteconfig(_RULE_ID, key, value)
    _kwin_reconfigure()


def remove_no_focus_rule() -> None:
    ids = [r for r in _kreadconfig("General", "rules").split(",") if r and r != _RULE_ID]
    _kwriteconfig("General", "rules", ",".join(ids))
    _kwriteconfig("General", "count", str(len(ids)))
    subprocess.run(["kwriteconfig6", "--file", _KWIN_FILE, "--group", _RULE_ID, "--delete-group"],
                   check=False)
    _kwin_reconfigure()


def _connect(timeout: float, sock_timeout: float = 5.0) -> BridgeClient:
    """Connect to the in-Krita server, retrying until Krita is ready.

    `sock_timeout` bounds each command's round-trip — raise it for slow ops like
    animation-frame import (hundreds of PNGs loaded on Krita's GUI thread)."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        c = BridgeClient(port=_current_port(), timeout=sock_timeout)
        try:
            c.connect()
            return c
        except OSError as e:  # Krita not up / plugin not serving yet
            last = e
            time.sleep(0.4)
    sys.exit(f"nulpaint: could not reach Krita bridge within {timeout:g}s ({last})")


def _bridge_up() -> bool:
    """Quick one-shot probe: is the in-Krita bridge already reachable? Used by
    `new` to decide whether it must launch Krita first (avoids waiting the full
    --wait timeout when Krita simply isn't running yet)."""
    c = BridgeClient(port=_current_port())
    try:
        c.connect()
    except OSError:
        return False
    c.close()
    return True


# --- subcommands ------------------------------------------------------------
def cmd_launch(a: argparse.Namespace) -> None:
    if a.no_focus:
        add_no_focus_rule()
        print("nulpaint: KWin no-focus rule active for 'krita'")
    binary = _krita_binary(a.krita)
    argv = [binary] + (["--nosplash"] if a.no_splash else [])
    # `launch --port N` (dest launch_port) or the global `--port N` both work.
    port = a.launch_port if a.launch_port is not None else getattr(a, "port", None)
    env = os.environ.copy()
    # A distinct port makes this a SEPARATE Krita process (own instance key, see
    # main.cc) serving its own bridge — so multiple windows can run at once, each
    # driven by its own CLI/Claude. No --port => the default single instance.
    if port:
        env["NULPAINT_PORT"] = str(port)
    subprocess.Popen(argv, start_new_session=True, env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    where = f" on port {port}" if port else ""
    print(f"nulpaint: launched {binary}{where} (pid detached)")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def cmd_instances(_a: argparse.Namespace) -> None:
    """List running Krita bridges (one per window). Prunes dead registry files."""
    try:
        names = sorted(os.listdir(INSTANCE_DIR))
    except OSError:
        names = []
    rows = []
    for name in names:
        if not name.endswith(".json"):
            continue
        path = os.path.join(INSTANCE_DIR, name)
        try:
            with open(path, encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            info = {}
        pid, port = info.get("pid"), info.get("port")
        if pid is None or not _pid_alive(pid):
            try:
                os.remove(path)          # stale — its Krita is gone
            except OSError:
                pass
            continue
        doc = "?"
        try:
            c = BridgeClient(port=port, timeout=2.0)
            c.connect()
            try:
                d = c.active_document()
                doc = (d or {}).get("fileName") or (d or {}).get("name") or "(no document)"
            finally:
                c.close()
        except OSError:
            doc = "(not responding)"
        rows.append((port, pid, doc))
    if not rows:
        print("nulpaint: no running Krita bridges")
        return
    print(f"{'PORT':<7}{'PID':<9}DOCUMENT")
    for port, pid, doc in rows:
        print(f"{port:<7}{pid:<9}{doc}")


def cmd_ping(_a: argparse.Namespace) -> None:
    with _connect(_a.wait) as c:
        print(c.ping())


def cmd_info(_a: argparse.Namespace) -> None:
    with _connect(_a.wait) as c:
        print(json.dumps(c.active_document(), indent=2))


def cmd_save(a: argparse.Namespace) -> None:
    with _connect(a.wait) as c:
        print(json.dumps(c.save_document(a.path), indent=2))


def cmd_call(a: argparse.Namespace) -> None:
    args = json.loads(a.args) if a.args else {}
    with _connect(a.wait) as c:
        print(json.dumps(c.call(a.cmd_name, **args), indent=2))


def cmd_no_focus_rule(a: argparse.Namespace) -> None:
    if a.action == "add":
        add_no_focus_rule()
        print("nulpaint: no-focus rule added")
    else:
        remove_no_focus_rule()
        print("nulpaint: no-focus rule removed")


# Demo shape layouts (bounding boxes on a 1920x1080 canvas; scaled to preset).
_CIRCLES = [(160, 180, 200, 200), (520, 120, 150, 150), (900, 300, 260, 260),
            (380, 640, 180, 180), (1300, 520, 220, 220)]
_SQUARES = [(1480, 160, 180, 180), (300, 380, 140, 140), (760, 720, 200, 200),
            (1120, 120, 150, 150), (1580, 760, 160, 160)]


def _scale_items(boxes, sx, sy):
    return [{"x": int(x * sx), "y": int(y * sy), "w": int(w * sx), "h": int(h * sy)}
            for (x, y, w, h) in boxes]


def cmd_demo(a: argparse.Namespace) -> None:
    if a.preset not in PRESETS:
        sys.exit(f"nulpaint: unknown preset {a.preset!r} (have: {', '.join(PRESETS)})")
    w, h, res = PRESETS[a.preset]
    sx, sy = w / 1920.0, h / 1080.0
    with _connect(a.wait) as c:
        print("ping:", c.ping())
        doc = c.create_document(w, h, name=a.preset, resolution=res)
        print("document:", json.dumps(doc))
        circ = c.draw_shapes("ellipse", [40, 120, 255, 255],
                             _scale_items(_CIRCLES, sx, sy), layer="Circles")
        print("circles:", json.dumps(circ))
        sq = c.draw_shapes("rectangle", [255, 140, 30, 255],
                           _scale_items(_SQUARES, sx, sy), layer="Squares")
        print("squares:", json.dumps(sq))
    print("nulpaint: demo complete")


def cmd_new(a: argparse.Namespace) -> None:
    """Create a new blank document. Dimensions come from a preset (default
    1080Land) unless overridden by --width/--height/--resolution. Launches Krita
    first if the bridge isn't already up, so "create a new document" works even
    from a cold start."""
    if a.preset not in PRESETS:
        sys.exit(f"nulpaint: unknown preset {a.preset!r} (have: {', '.join(PRESETS)})")
    pw, ph, pres = PRESETS[a.preset]
    w, h, res = a.width or pw, a.height or ph, a.resolution or pres
    if not a.no_launch and not _bridge_up():
        binary = _krita_binary(a.krita)
        subprocess.Popen([binary, "--nosplash"], start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"nulpaint: launching {binary} …")
    with _connect(a.wait) as c:
        doc = c.create_document(w, h, name=a.name, resolution=res)
    print(f"nulpaint: new {doc['width']}x{doc['height']} document '{doc['name']}'")


def cmd_close(a: argparse.Namespace) -> None:
    """Close the active document (or --name). A modified doc is left open unless
    you pass --save or --discard (so close never blocks on a save prompt)."""
    with _connect(a.wait) as c:
        res = c.close_document(name=a.name, save=a.save, discard=a.discard)
    if res.get("unsaved"):
        sys.exit(f"nulpaint: '{res['name']}' has unsaved changes — "
                 f"close with --save or --discard")
    print(f"nulpaint: closed '{res['closed']}'")


def cmd_resize(a: argparse.Namespace) -> None:
    """Resize the active document — scale the image (default) or, with --canvas,
    change the canvas bounds (crop/extend, no resampling)."""
    mode = "canvas" if a.canvas else "scale"
    with _connect(a.wait) as c:
        res = c.resize_document(a.width, a.height, mode=mode, filter=a.filter)
    print(f"nulpaint: resized '{res['name']}' -> {res['width']}x{res['height']} ({res['mode']})")


def cmd_select_subject(a: argparse.Namespace) -> None:
    from .vision import select_subject
    kind = "object" if a.object else "person"
    with _connect(a.wait) as c:
        res = select_subject(c, kind, fill_holes=a.fill_holes)
    print(f"nulpaint: selected {res['kind']} via {res['service']} "
          f"({res['w']}x{res['h']})")


ANIM_ROOT = "/mnt/storage1/Pictures/Nuldrums/StreamToons/Animations"


def cmd_build_anim_doc(a: argparse.Namespace) -> None:
    from .anim import build_anim_doc
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    out = a.out or os.path.join(src, f"{a.char}.kra")
    only = [s.strip() for s in a.clips.split(",") if s.strip()] if a.clips else None
    canvas = tuple(int(v) for v in a.canvas.split("x")) if a.canvas else None
    # Frame import loads hundreds of PNGs on Krita's GUI thread -> long socket timeout.
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = build_anim_doc(c, char=a.char, src_dir=src, out_path=out,
                             only=only, fps=a.fps, canvas=canvas)
    print(f"nulpaint: built {res['out']}  ({res['canvas'][0]}x{res['canvas'][1]} "
          f"@ {res['fps']}fps)")
    for cl in res["clips"]:
        flag = "" if cl["animated"] else "  [NOT ANIMATED!]"
        print(f"  - {cl['anim']}: {cl['frames']} frames (src {cl['src_fps']}fps){flag}")


def cmd_punch_anim(a: argparse.Namespace) -> None:
    from .anim import punch_anim_frames
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    only = [s.strip() for s in a.clips.split(",") if s.strip()] if a.clips else None
    # Hundreds of per-frame set_frame + filter.apply round-trips -> long socket timeout.
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = punch_anim_frames(c, char=a.char, src_dir=src, only=only,
                                saturation=a.saturation, value=a.value,
                                black=a.black, white=a.white, gamma=a.gamma)
    total = sum(cl["frames"] for cl in res["clips"])
    print(f"nulpaint: punched {total} frames across {len(res['clips'])} "
          f"{a.char} animations (sat+{a.saturation})")
    for cl in res["clips"]:
        print(f"  - {cl['anim']}: {cl['frames']} frames")


def cmd_open(a: argparse.Namespace) -> None:
    path = str(Path(a.path).expanduser().resolve())
    with _connect(a.wait) as c:
        r = c.call("document.open", path=path)
    verb = "activated (already open)" if r.get("reused") else "opened"
    print(f"nulpaint: {verb} {r['fileName']}  ({r['width']}x{r['height']})")


def cmd_export_chibi(a: argparse.Namespace) -> None:
    from .chibi import export_chibi
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = export_chibi(c, layer_base=a.layer, exp_name=a.name or a.layer,
                           mask=a.mask, nobg_only=a.nobg_only, keep_aspect=a.keep_aspect)
    print(f"nulpaint: exported Chibi_{res['name']}  (masked={res['masked']}, "
          f"placement={res['placement']}, backed_up={res['backed_up']}/3)  "
          f"native_bbox={res['native_bbox']} padded_bbox={res['padded_bbox']}")


def cmd_export_object(a: argparse.Namespace) -> None:
    from .chibi import export_object
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = export_object(c, layer_base=a.layer, exp_name=a.name or a.layer,
                            mask=a.mask, game_dir=a.game_dir)
    extra = f" + {res['game_out']}" if res['game_out'] else ""
    print(f"nulpaint: exported {res['name']}.png -> {res['out']}{extra}  "
          f"(masked={res['masked']}, bbox={res['bbox']}, backed_up={res['backed_up']})")


def cmd_export_prone(a: argparse.Namespace) -> None:
    from .chibi import export_prone
    with _connect(a.wait, sock_timeout=600.0) as c:
        for char in a.chars:
            res = export_prone(c, char)
            print(f"nulpaint: exported {char}_Prone_final.png -> {res['out']}  "
                  f"(bbox={res['bbox']}, backed_up={res['backed_up']})")


def cmd_repose_still(a: argparse.Namespace) -> None:
    from .repose import repose_still
    out = a.out or (os.path.splitext(a.input)[0] + f"_{a.pose.capitalize()}.png")
    green = None
    if a.green:
        green = tuple(int(v) for v in a.green.split(",")) if "," in a.green \
            else (0, 177, 64)
    res = repose_still(a.input, out, pose=a.pose, prompt=a.prompt, extra=a.extra,
                       refs=a.ref, kind=("person" if a.person else "object"),
                       model=a.nano_model, anchor=not a.no_anchor,
                       pad_ref=a.pad_ref, green_bg=green)
    print(f"nulpaint: reposed {os.path.basename(a.input)} -> {res['out']}  "
          f"(pose={res['pose']}, gen={res['gen_size'][0]}x{res['gen_size'][1]}, "
          f"bbox={res['final_bbox']})\n  raw gen kept at {res['raw']}")


def cmd_repose_roster(a: argparse.Namespace) -> None:
    from .repose import repose_roster
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None

    def prog(row):
        print(f"  [{row['status']}] {row['char']} -> {os.path.basename(row.get('out',''))}",
              flush=True)

    res = repose_roster(pose=a.pose, only=only, limit=a.limit, dry_run=a.dry_run,
                        kind=("person" if a.person else "object"), model=a.nano_model,
                        on_progress=None if a.dry_run else prog)
    if a.dry_run:
        for r in res["report"]:
            src = os.path.basename(r["still"]) if r["still"] else "—"
            print(f"  {r['char']:<18} <- {src:<28} [{r['status']}]")
    print(f"nulpaint: {res['ok']}/{res['count']} reposed ({a.pose}).  "
          f"review: {res['contact_sheet'] or res['review_dir']}")


def cmd_rebuild_clip(a: argparse.Namespace) -> None:
    from .anim import rebuild_clip
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = rebuild_clip(c, char=a.char, src_dir=src, anim=a.anim,
                           from_bak=not a.from_mov)
    print(f"nulpaint: rebuilt {a.char}/{res['anim']} from {res['source']} "
          f"({res['frames']} frames)")


def cmd_apply_filter_anim(a: argparse.Namespace) -> None:
    import json
    from .anim import apply_filter_anim_frames
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    only = [s.strip() for s in a.clips.split(",") if s.strip()] if a.clips else None
    cfg = json.loads(Path(a.config_file).read_text()) if a.config_file else json.loads(a.config)
    # accept either a raw config dict, or the full {"filter","config"} from read_config
    filt = a.filter or cfg.get("filter")
    if isinstance(cfg, dict) and "config" in cfg and "filter" in cfg:
        filt, cfg = cfg["filter"], cfg["config"]
    if not filt:
        sys.exit("nulpaint: need --filter (or a read_config JSON with a 'filter' key)")
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = apply_filter_anim_frames(c, char=a.char, src_dir=src, filter_id=filt,
                                       config=cfg, only=only)
    total = sum(cl["frames"] for cl in res["clips"])
    print(f"nulpaint: baked '{filt}' into {total} frames across {len(res['clips'])} "
          f"{a.char} animations")


def cmd_despill_anim(a: argparse.Namespace) -> None:
    from .anim import despill_anim_frames
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    only = [s.strip() for s in a.clips.split(",") if s.strip()] if a.clips else None
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = despill_anim_frames(c, char=a.char, src_dir=src, only=only,
                                  thr=a.thr, edge=a.edge, grow=a.grow)
    total = sum(cl["px"] for cl in res["clips"])
    print(f"nulpaint: green-eat edge despill on {a.char}: recoloured {total} px "
          f"(thr={a.thr} edge={a.edge}px)")
    for cl in res["clips"]:
        print(f"  - {cl['anim']}: {cl['recoloured_frames']}/{cl['frames']} frames touched, "
              f"{cl['px']} px")


def cmd_export_anim_doc(a: argparse.Namespace) -> None:
    from .anim import export_anim_doc
    src = a.src or os.path.join(ANIM_ROOT, a.char)
    only = [s.strip() for s in a.clips.split(",") if s.strip()] if a.clips else None
    with _connect(a.wait, sock_timeout=600.0) as c:
        res = export_anim_doc(c, char=a.char, src_dir=src, out_dir=a.out_dir,
                              only=only, solidify=not a.no_solidify, fps_override=a.fps)
    print(f"nulpaint: re-exported {a.char} -> {res['out_dir']}  "
          f"(solidified: {res['solidified']})")
    for cl in res["clips"]:
        v = cl["verify"]
        bak = "  [orig backed up .bak]" if cl["backed_up"] else ""
        print(f"  - {cl['anim']}: {cl['frames']}f @{cl['fps']}fps -> "
              f"{os.path.basename(cl['mov'])} + .webm  "
              f"(opaque {v['opaque_pct']}%, green {v['residual_green_pct']}%){bak}")


def cmd_despill_selection(a: argparse.Namespace) -> None:
    from .despill import despill_selection
    with _connect(a.wait) as c:
        res = despill_selection(c, thr=a.thr, grow=a.grow, layer=a.layer)
    scope = "selection" if res["scoped"] else "whole canvas (no selection)"
    print(f"nulpaint: green-eat recoloured {res['recoloured']} px over {scope} "
          f"@({res['x']},{res['y']}) {res['w']}x{res['h']}")


def _collect_cut_layers(c: BridgeClient) -> list[tuple[str, str]]:
    """Every paint layer whose name ends with '(cut)', as (uuid, name)."""
    out: list[tuple[str, str]] = []

    def walk(node: dict) -> None:
        if node.get("type") == "paintlayer" and node.get("name", "").rstrip().endswith("(cut)"):
            out.append((node["uuid"], node["name"]))
        for child in node.get("children", []):
            walk(child)

    for top in c.call("node.tree")["tree"]:
        walk(top)
    return out


def cmd_punch(a: argparse.Namespace) -> None:
    from .depastel import punch_layer
    with _connect(a.wait) as c:
        if a.all_cuts:
            targets = _collect_cut_layers(c)          # (uuid, name)
            if not targets:
                print("nulpaint: no '(cut)' layers found")
                return
        elif a.layers:
            targets = [(None, s.strip()) for s in a.layers.split(",") if s.strip()]
        else:
            targets = [(None, None)]                  # active layer

        for uuid, name in targets:
            ident = uuid or name
            if a.new_layer:
                dup_name = f"{name} {a.suffix}" if name else None
                r = c.call("node.duplicate", node=ident, name=dup_name)
                ident, label = r["uuid"], r["name"]
            else:
                label = name or "(active layer)"
            punch_layer(c, ident, saturation=a.saturation, value=a.value,
                        black=a.black, white=a.white, gamma=a.gamma)
            print(f"nulpaint: punched '{label}' sat+{a.saturation} "
                  f"val+{a.value} black={a.black} white={a.white} gamma={a.gamma}")


def cmd_mask_subtract(a: argparse.Namespace) -> None:
    from .masksub import subtract_layer
    # default grow: 0 for a hand-tuned selection (use it as-is), 3 for a mask layer
    grow = a.grow if a.grow is not None else (0 if a.from_selection else 3)
    with _connect(a.wait) as c:
        r = subtract_layer(c, target=a.target, mask_layer=a.mask,
                           from_selection=a.from_selection, grow=grow,
                           threshold=a.threshold, in_place=a.in_place,
                           suffix=a.suffix, hide_source=not a.keep_source_visible,
                           clear_rgb=a.clear_rgb, set_selection=not a.no_selection)
    where = f"'{r['target']}' in place" if r["in_place"] else f"copy '{r['cut_layer']}'"
    grow_s = f" (grow {r['grow']}px)" if r["grow"] else ""
    print(f"nulpaint: subtracted {r['source']}{grow_s} from {where} "
          f"— erased {r['erased_px']} px"
          + (f", hid original '{r['target']}'" if r["hid_source"] else ""))


def cmd_mask_fill(a: argparse.Namespace) -> None:
    from .masksub import fill_hole
    with _connect(a.wait) as c:
        r = fill_hole(c, target=a.target, pad=a.pad, radius=a.radius,
                      layer_name=a.name, below=a.below)
    print(f"nulpaint: content-aware filled {r['filled_px']} px -> new layer "
          f"'{r['fill_layer']}' @({r['x']},{r['y']}) {r['w']}x{r['h']}")


def cmd_mask_preview(a: argparse.Namespace) -> None:
    from .masksub import preview_mask
    with _connect(a.wait) as c:
        r = preview_mask(c, mask_layer=a.mask, grow=a.grow, threshold=a.threshold)
    verb = "grew" if a.grow > 0 else ("pulled in" if a.grow < 0 else "used exact")
    print(f"nulpaint: previewed selection from '{r['mask_layer']}' — {verb} "
          f"{abs(a.grow)}px stencil ({r['stencil_px']} px). No pixels cut.")


def _nano_kwargs(a: argparse.Namespace) -> dict:
    return {"engine": a.engine, "ref_files": a.ref, "ref_layers": a.ref_layers,
            "nano_model": a.nano_model, "review": a.review}


def cmd_segment_layers(a: argparse.Namespace) -> None:
    from .vision.select import segment_layers
    kind = "person" if a.person else "object"
    only = [s.strip() for s in a.layers.split(",") if s.strip()] if a.layers else None
    with _connect(a.wait) as c:
        rep = segment_layers(c, kind=kind, suffix=a.suffix, limit=a.limit, only=only,
                             fill_holes=a.fill_holes)
    ok = [r for r in rep if r["status"] == "ok"]
    print(f"nulpaint: segmented {len(ok)}/{len(rep)} paint layers (kind={kind}) "
          f"-> added '<name>{a.suffix}' layers")
    for r in rep:
        if r["status"] != "ok":
            print(f"  - {r['layer']}: {r['status']}")


def _find_group(c: BridgeClient, name: str) -> str | None:
    """uuid of the first top-level group layer named `name`, or None."""
    for t in c.call("node.tree")["tree"]:
        if t["name"] == name and t["type"] == "grouplayer":
            return t["uuid"]
    return None


def cmd_import_image(a: argparse.Namespace) -> None:
    import base64
    paths = [Path(p).expanduser() for p in a.paths]
    for p in paths:
        if not p.is_file():
            sys.exit(f"nulpaint: no such file: {p}")
    with _connect(a.wait) as c:
        group_uuid = None
        if a.group:
            group_uuid = _find_group(c, a.group)
            if group_uuid is None:
                group_uuid = c.call("node.create_group", name=a.group)["uuid"]
                print(f"nulpaint: created group '{a.group}'")
        for p in paths:
            name = a.name if (a.name and len(paths) == 1) else p.stem
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            r = c.call("layer.add_image", name=name, png_b64=b64, place=a.place)
            if group_uuid:
                c.call("node.move", node=r["uuid"] if "uuid" in r else r["layer"],
                       parent=group_uuid)
            print(f"nulpaint: imported '{r['layer']}' {r['w']}x{r['h']}"
                  + (f" -> group '{a.group}'" if a.group else ""))


def cmd_inpaint(a: argparse.Namespace) -> None:
    from .generate import inpaint
    extra = {} if a.img_cfg is None else {"img_cfg": a.img_cfg}
    with _connect(a.wait) as c:
        res = inpaint(c, a.prompt, negative=a.negative, model=a.model,
                      steps=a.steps, cfg=a.cfg, strength=a.strength,
                      seed=a.seed, pad=a.pad, lora=a.lora, **_nano_kwargs(a), **extra)
    print(f"nulpaint: inpainted [{res.get('engine', res.get('model'))}] "
          f"{res['w']}x{res['h']} @({res['x']},{res['y']})")


def cmd_outpaint(a: argparse.Namespace) -> None:
    from .generate import outpaint
    extra = {} if a.img_cfg is None else {"img_cfg": a.img_cfg}
    with _connect(a.wait) as c:
        res = outpaint(c, a.prompt, negative=a.negative, model=a.model,
                       pixels=a.pixels, sides=a.sides, steps=a.steps,
                       cfg=a.cfg, strength=a.strength, seed=a.seed, lora=a.lora,
                       **_nano_kwargs(a), **extra)
    print(f"nulpaint: outpainted [{res.get('engine', res.get('model'))}] -> "
          f"{res['width']}x{res['height']} (+{res['pixels']}px {','.join(res['sides'])})")


def cmd_style(a: argparse.Namespace) -> None:
    from .generate import style
    with _connect(a.wait) as c:
        res = style(c, a.prompt, negative=a.negative, model=a.model,
                    strength=a.strength, steps=a.steps, cfg=a.cfg, seed=a.seed,
                    lora=a.lora, **_nano_kwargs(a))
    print(f"nulpaint: restyled [{res.get('engine', res.get('model'))}] "
          f"{res['scope']} {res['w']}x{res['h']}")


def cmd_mode(a: argparse.Namespace) -> None:
    # Pre-swap the SDXL checkpoint for an image-model mode (no bridge needed).
    from .generate import diffusion
    res = diffusion.set_mode(a.mode)
    print(f"nulpaint: image-model mode '{res['mode']}' -> {res['warm']} checkpoint warm")


def cmd_control(a: argparse.Namespace) -> None:
    from .generate import control
    with _connect(a.wait) as c:
        res = control(c, a.prompt, kind=a.kind, control_image=a.control_image,
                      model=a.model, control_strength=a.control_strength,
                      steps=a.steps, cfg=a.cfg, seed=a.seed, negative=a.negative,
                      lora=a.lora)
    print(f"nulpaint: {res['kind']} controlnet [{res['model']}] -> layer "
          f"'{res['layer']}' ({res['w']}x{res['h']})")


def _add_diffusion_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("prompt", help="text prompt")
    p.add_argument("-n", "--negative", default="", help="negative prompt")
    p.add_argument("--model", default=None, help="sd15 (default) | sdxl")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=7.0, help="cfg scale (prompt strength)")
    p.add_argument("--img-cfg", type=float, default=None, dest="img_cfg",
                   help="image guidance: low (~1.5)=bold replace, high (~cfg)=seamless fill")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=-1, help="-1 = random")
    p.add_argument("--lora", default="",
                   help="LoRA(s) as name[:weight], comma-separated (files in models/loras/)")


def _add_nano_args(p: argparse.ArgumentParser) -> None:
    """Shared engine/reference flags for the Nano Banana Pro (cloud) path."""
    p.add_argument("--engine", choices=["local", "nanobanana"], default="local",
                   help="local SDXL (default) or 'nanobanana' (Nano Banana Pro / OpenRouter)")
    p.add_argument("--ref", action="append", default=None, metavar="FILE",
                   help="reference image file for nanobanana (repeatable)")
    p.add_argument("--ref-layer", action="append", default=None, dest="ref_layers",
                   metavar="NAME", help="reference layer by name for nanobanana (repeatable)")
    p.add_argument("--nano-model", default=None, dest="nano_model",
                   help="override the OpenRouter model slug (default gemini-3-pro-image-preview)")
    p.add_argument("--no-review", action="store_false", dest="review", default=True,
                   help="don't add the raw nanobanana generation as a review layer below")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nulpaint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait", type=float, default=20.0,
                   help="seconds to wait for the Krita bridge (default 20)")
    p.add_argument("--port", type=int, default=None,
                   help="bridge port to talk to / launch on — selects one Krita "
                        "instance when several run (default $NULPAINT_PORT or 8765)")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("launch", help="launch the forked Krita")
    pl.add_argument("--no-focus", action="store_true",
                    help="install a KWin rule so Krita won't steal focus")
    pl.add_argument("--no-splash", action="store_true", help="suppress the splash screen")
    pl.add_argument("--krita", help="path to the krita binary (default: ~/.local/bin/krita)")
    pl.add_argument("--port", type=int, default=None, dest="launch_port",
                    help="start a SEPARATE instance on this bridge port (own window "
                         "+ own bridge; omit for the default single instance)")
    pl.set_defaults(func=cmd_launch)

    sub.add_parser("instances",
                   help="list running Krita bridges (one per window) + their docs"
                   ).set_defaults(func=cmd_instances)

    sub.add_parser("ping", help="ping the in-Krita server").set_defaults(func=cmd_ping)
    sub.add_parser("info", help="print active document info").set_defaults(func=cmd_info)

    ps = sub.add_parser("save", help="save the active document")
    ps.add_argument("path", nargs="?", help="destination file (e.g. foo.kra); "
                    "omit to re-save to the existing path")
    ps.set_defaults(func=cmd_save)

    pc = sub.add_parser("call", help="send an arbitrary bridge command")
    pc.add_argument("cmd_name", help="command name, e.g. document.info")
    pc.add_argument("--args", help="JSON object of command args")
    pc.set_defaults(func=cmd_call)

    pd = sub.add_parser("demo", help="create a doc + draw circles/squares")
    pd.add_argument("--preset", default="1080Land", help="document preset (default 1080Land)")
    pd.set_defaults(func=cmd_demo)

    pnew = sub.add_parser("new", help="create a new blank document (launches Krita if needed)")
    pnew.add_argument("--preset", default="1080Land",
                      help=f"document preset (default 1080Land; have: {', '.join(PRESETS)})")
    pnew.add_argument("--width", type=int, help="override preset width (px)")
    pnew.add_argument("--height", type=int, help="override preset height (px)")
    pnew.add_argument("--resolution", type=float, help="override resolution (ppi)")
    pnew.add_argument("--name", default="Untitled", help="document name")
    pnew.add_argument("--no-launch", action="store_true",
                      help="don't auto-launch Krita if the bridge is down")
    pnew.add_argument("--krita", help="path to the krita binary (default: ~/.local/bin/krita)")
    pnew.set_defaults(func=cmd_new)

    pcl = sub.add_parser("close", help="close the active document (or --name)")
    pcl.add_argument("--name", help="close the document with this name (default: active)")
    pcl.add_argument("--save", action="store_true", help="save before closing")
    pcl.add_argument("--discard", action="store_true",
                     help="close even if modified, discarding unsaved changes")
    pcl.set_defaults(func=cmd_close)

    prs = sub.add_parser("resize", help="resize the active document (scale, or --canvas)")
    prs.add_argument("width", type=int, help="target width in px")
    prs.add_argument("height", type=int, help="target height in px")
    prs.add_argument("--canvas", action="store_true",
                     help="change canvas bounds (crop/extend) instead of scaling the image")
    prs.add_argument("--filter", default="Bicubic", help="scale strategy (default Bicubic)")
    prs.set_defaults(func=cmd_resize)

    pn = sub.add_parser("no-focus-rule", help="manage the KWin no-focus rule")
    pn.add_argument("action", choices=["add", "remove"])
    pn.set_defaults(func=cmd_no_focus_rule)

    psl = sub.add_parser("segment-layers",
                         help="run subject segmentation on every paint layer, adding a trimmed '(cut)' copy")
    psl.add_argument("--person", action="store_true",
                     help="use mattemodel (person/RVM) instead of segmodel (object/BiRefNet)")
    psl.add_argument("--suffix", default=" (cut)", help="name suffix for the added cut layers")
    psl.add_argument("--limit", type=int, default=0,
                     help="only process the first N paint layers (0 = all) — for a quick test")
    psl.add_argument("--layers", default=None,
                     help="comma-separated layer names to scope the run to (anywhere in "
                          "the stack, incl. inside groups); default = every paint layer")
    psl.add_argument("--no-fill-holes", action="store_false", dest="fill_holes", default=True,
                     help="keep transparent islands enclosed by the subject (default: fill them)")
    psl.set_defaults(func=cmd_segment_layers)

    ppu = sub.add_parser("punch",
                         help="de-pastel: bake saturation + contrast into layer(s) "
                              "(then hand-tune hues via Cross-channel adjustment)")
    ppu.add_argument("--layers", default=None,
                     help="comma-separated layer names (uuid or name); default = active layer")
    ppu.add_argument("--all-cuts", action="store_true", dest="all_cuts",
                     help="target every paint layer whose name ends with '(cut)'")
    ppu.add_argument("--new-layer", action="store_true", dest="new_layer",
                     help="duplicate each target to a new layer and punch the copy "
                          "(leaves the original untouched)")
    ppu.add_argument("--suffix", default="nulpaint",
                     help="name suffix for --new-layer copies (default 'nulpaint')")
    ppu.add_argument("--saturation", type=int, default=35,
                     help="HSL saturation boost -100..100 (default 35)")
    ppu.add_argument("--value", type=int, default=0,
                     help="HSL value/brightness shift -100..100 (default 0)")
    ppu.add_argument("--black", type=int, default=18,
                     help="input black point 0..255 — raise to deepen darks (default 18)")
    ppu.add_argument("--white", type=int, default=245,
                     help="input white point 0..255 — lower to brighten highs (default 245)")
    ppu.add_argument("--gamma", type=float, default=1.0,
                     help="levels gamma; <1 darkens mids, >1 lightens (default 1.0)")
    ppu.set_defaults(func=cmd_punch)

    pms = sub.add_parser("mask-subtract",
                         help="erase from TARGET everything under MASK layer's alpha "
                              "(boolean layer subtract; e.g. cut foreground grass out "
                              "of a baked ground layer)")
    pms.add_argument("--target", required=True,
                     help="layer to erase FROM (name or uuid)")
    pms.add_argument("--mask", default=None,
                     help="layer whose alpha is the stencil (name or uuid); "
                          "omit when using --from-selection")
    pms.add_argument("--from-selection", action="store_true", dest="from_selection",
                     help="cut using the live document selection (feathered) instead "
                          "of a mask layer — use after hand-brushing the selection")
    pms.add_argument("--grow", type=int, default=None,
                     help="dilate (+) / shrink (-) the stencil px (default: 3 for a "
                          "mask layer, 0 for --from-selection so it's used as-is)")
    pms.add_argument("--threshold", type=int, default=8,
                     help="mask alpha above this counts as covered (default 8)")
    pms.add_argument("--in-place", action="store_true", dest="in_place",
                     help="cut the target directly instead of a hidden-original copy")
    pms.add_argument("--suffix", default=" GrassRemoved",
                     help="name suffix for the cut copy (default ' GrassRemoved')")
    pms.add_argument("--keep-source-visible", action="store_true",
                     dest="keep_source_visible",
                     help="don't hide the original when making a copy")
    pms.add_argument("--clear-rgb", action="store_true", dest="clear_rgb",
                     help="also zero RGB under the stencil (default: alpha only)")
    pms.add_argument("--no-selection", action="store_true", dest="no_selection",
                     help="don't leave the grown stencil as the document selection")
    pms.set_defaults(func=cmd_mask_subtract)

    pmp = sub.add_parser("mask-preview",
                         help="set the selection to a layer's (grown/eroded) alpha "
                              "stencil WITHOUT cutting — dial --grow to eyeball reach")
    pmp.add_argument("--mask", required=True,
                     help="layer whose alpha is the stencil (name or uuid)")
    pmp.add_argument("--grow", type=int, default=0,
                     help="dilate (+) or pull IN (-) the stencil this many px (default 0)")
    pmp.add_argument("--threshold", type=int, default=8,
                     help="mask alpha above this counts as covered (default 8)")
    pmp.set_defaults(func=cmd_mask_preview)

    pmf = sub.add_parser("mask-fill",
                         help="content-aware fill the selected hole by SAMPLING "
                              "surrounding real pixels (no GPU) -> new layer")
    pmf.add_argument("--target", default=None,
                     help="layer the hole is in (name or uuid); default = active")
    pmf.add_argument("--below", default="GroundPatch1_ForegroundGrass",
                     help="place the fill layer directly below this layer "
                          "(default GroundPatch1_ForegroundGrass); '' = below active")
    pmf.add_argument("--name", default="GrassFill (patch)",
                     help="name for the new fill layer")
    pmf.add_argument("--pad", type=float, default=0.4,
                     help="context margin around the hole bbox, fraction (default 0.4)")
    pmf.add_argument("--radius", type=int, default=5,
                     help="Telea inpaint radius px (default 5)")
    pmf.set_defaults(func=cmd_mask_fill)

    pim = sub.add_parser("import-image",
                         help="import image file(s) as paint layer(s), optionally into a group")
    pim.add_argument("paths", nargs="+", help="image file(s) to import")
    pim.add_argument("--name", default=None,
                     help="layer name (single file only; default = file stem)")
    pim.add_argument("--group", default=None,
                     help="add the layer(s) into this group (found by name, created if absent)")
    pim.add_argument("--place", default="top", choices=["top", "below_active"],
                     help="where to drop each new layer (default top)")
    pim.set_defaults(func=cmd_import_image)

    pbad = sub.add_parser("build-anim-doc",
                          help="build a per-character animation-cleanup .kra (group per "
                               "animation, frames on the timeline) from its _4444.mov clips")
    pbad.add_argument("char", help="character name, e.g. Trikeri")
    pbad.add_argument("--src", default=None,
                      help=f"source folder (default {ANIM_ROOT}/<char>)")
    pbad.add_argument("--out", default=None, help="output .kra path (default <src>/<char>.kra)")
    pbad.add_argument("--clips", default=None,
                      help="comma-separated animation names to include (default: all)")
    pbad.add_argument("--fps", type=int, default=None, help="override document fps")
    pbad.add_argument("--canvas", default=None, help="canvas WxH (default: clip dims)")
    pbad.set_defaults(func=cmd_build_anim_doc)

    ppa = sub.add_parser("punch-anim",
                         help="bake the de-pastel punch (levels+saturation) into EVERY keyframe "
                              "of each animation in the OPEN character .kra")
    ppa.add_argument("char", help="character name, e.g. Magi2")
    ppa.add_argument("--src", default=None, help=f"source folder (default {ANIM_ROOT}/<char>)")
    ppa.add_argument("--clips", default=None,
                     help="comma-separated animation names (default: all)")
    ppa.add_argument("--saturation", type=int, default=35, help="HSL saturation boost (default 35)")
    ppa.add_argument("--value", type=int, default=0, help="HSL value shift (default 0)")
    ppa.add_argument("--black", type=int, default=18, help="input black point 0..255 (default 18)")
    ppa.add_argument("--white", type=int, default=245, help="input white point 0..255 (default 245)")
    ppa.add_argument("--gamma", type=float, default=1.0, help="levels gamma (default 1.0)")
    ppa.set_defaults(func=cmd_punch_anim)

    popen = sub.add_parser("open", help="open a document file in Krita (activates it if already open)")
    popen.add_argument("path", help="path to the .kra (or any Krita-openable) file")
    popen.set_defaults(func=cmd_open)

    pec = sub.add_parser("export-chibi",
                         help="export a chibi character's final art (punch + optional colour mask) "
                              "from the OPEN ChibiToonEdits.kra to the 3 still deliverables "
                              "(NoBG, padded, greenscreen), matching existing padding; backs up first")
    pec.add_argument("layer", help="layer base name in ChibiToonEdits, e.g. WorldSynth3")
    pec.add_argument("--name", default=None, help="export name (default = layer), e.g. WorldSynths")
    pec.add_argument("--mask", action="store_true",
                     help="bake the layer's Cross-channel colour mask into the output")
    pec.add_argument("--nobg-only", action="store_true", dest="nobg_only",
                     help="only write the tight NoBG Chibi_<name>.png (skip padded + greenscreen)")
    pec.add_argument("--keep-aspect", action="store_true", dest="keep_aspect",
                     help="silhouette changed (e.g. redrawn narrower): keep the reference's "
                          "height/feet/centre but let padded WIDTH follow the new aspect ratio "
                          "(don't re-stretch to the old box)")
    pec.set_defaults(func=cmd_export_chibi)

    peo = sub.add_parser("export-object",
                         help="export a NON-character object's final '<base> (cut) nulpaint' layer "
                              "from the OPEN ChibiToonEdits.kra to a plain full-canvas RGBA PNG at "
                              "OtherImages_NoBG/<name>.png (no chibi padding); backs up existing first")
    peo.add_argument("layer", help="layer base name in ChibiToonEdits, e.g. BeerKeg")
    peo.add_argument("--name", default=None, help="export name (default = layer)")
    peo.add_argument("--mask", action="store_true",
                     help="bake the layer's Cross-channel colour mask if it has one")
    peo.add_argument("--game-dir", default=None, dest="game_dir",
                     help="also copy the PNG into this game Textures dir")
    peo.set_defaults(func=cmd_export_object)

    pep = sub.add_parser("export-prone",
                         help="export repositioned '<char>_Prone' layer(s) from the OPEN "
                              "ChibiToonEdits.kra to Animations/<char>/<char>_Prone_final.png "
                              "(full-canvas 1024 PNG, current transform preserved); backs up first")
    pep.add_argument("chars", nargs="+", help="character name(s), e.g. Gorlunk1 Magi1 Magi3")
    pep.set_defaults(func=cmd_export_prone)

    prc = sub.add_parser("rebuild-clip",
                         help="replace one animation clip in the OPEN character .kra with fresh "
                              "frames from its pristine _4444.mov.bak (then re-run the chain scoped "
                              "to that clip)")
    prc.add_argument("char", help="character name, e.g. Cyren3")
    prc.add_argument("anim", help="animation/clip name, e.g. Idle_Forward")
    prc.add_argument("--src", default=None, help=f"source folder (default {ANIM_ROOT}/<char>)")
    prc.add_argument("--from-mov", action="store_true", dest="from_mov",
                     help="import from the current _4444.mov instead of the pristine .bak")
    prc.set_defaults(func=cmd_rebuild_clip)

    pafa = sub.add_parser("apply-filter-anim",
                          help="bake an arbitrary Krita filter (id + config) into every keyframe "
                               "of each animation in the OPEN character .kra")
    pafa.add_argument("char", help="character name, e.g. Cyren3")
    pafa.add_argument("--filter", default=None, help="filter id (e.g. crosschannel); "
                      "optional if --config-file is a read_config JSON")
    pafa.add_argument("--config", default="{}", help="filter config as inline JSON")
    pafa.add_argument("--config-file", default=None, dest="config_file",
                      help="path to a JSON config (or a full read_config result)")
    pafa.add_argument("--src", default=None, help=f"source folder (default {ANIM_ROOT}/<char>)")
    pafa.add_argument("--clips", default=None, help="comma-separated animation names (default: all)")
    pafa.set_defaults(func=cmd_apply_filter_anim)

    pda = sub.add_parser("despill-anim",
                         help="green-eat the outer alpha edge of EVERY keyframe of each animation "
                              "in the OPEN character .kra (run after punch-anim, before re-export)")
    pda.add_argument("char", help="character name, e.g. LadyVermilia")
    pda.add_argument("--src", default=None, help=f"source folder (default {ANIM_ROOT}/<char>)")
    pda.add_argument("--clips", default=None, help="comma-separated animation names (default: all)")
    pda.add_argument("--thr", type=int, default=22, help="green-dominance threshold (default 22)")
    pda.add_argument("--edge", type=int, default=8,
                     help="edge-band width in px just inside the alpha boundary (default 8)")
    pda.add_argument("--grow", type=int, default=2,
                     help="dilate the green mask along the soft edge (default 2)")
    pda.set_defaults(func=cmd_despill_anim)

    pead = sub.add_parser("export-anim-doc",
                          help="re-export cleaned clips from the OPEN character .kra: timeline "
                               "-> _4444.mov + .webm, solidify, verify, back up originals")
    pead.add_argument("char", help="character name, e.g. Trikeri")
    pead.add_argument("--src", default=None,
                      help=f"source folder (default {ANIM_ROOT}/<char>)")
    pead.add_argument("--out-dir", default=None, dest="out_dir",
                      help="output folder (default = src)")
    pead.add_argument("--clips", default=None,
                      help="comma-separated animation names to export (default: all)")
    pead.add_argument("--no-solidify", action="store_true",
                      help="skip klingsolidify (use for pure-FX clips; default solidifies)")
    pead.add_argument("--fps", type=int, default=None, help="override output fps")
    pead.set_defaults(func=cmd_export_anim_doc)

    pds = sub.add_parser("despill-selection",
                         help="green-eater: recolour leftover green edge fringe inside the "
                              "selection from the nearest clean colour (alpha untouched)")
    pds.add_argument("--thr", type=int, default=22,
                     help="green-dominance threshold (higher = only stronger green; default 22)")
    pds.add_argument("--grow", type=int, default=0,
                     help="extend the recolour this many px along the soft edge (default 0)")
    pds.add_argument("--layer", default=None, help="layer name (default: active)")
    pds.set_defaults(func=cmd_despill_selection)

    pss = sub.add_parser("select-subject",
                         help="select the subject via the matte/seg services")
    pss.add_argument("--object", action="store_true",
                     help="arbitrary object (segmodel) instead of a person (mattemodel)")
    pss.add_argument("--no-fill-holes", action="store_false", dest="fill_holes", default=True,
                     help="keep transparent islands enclosed by the subject (default: fill them)")
    pss.set_defaults(func=cmd_select_subject)

    pmode = sub.add_parser("mode", help="pre-load the SDXL checkpoint for an image-model mode")
    pmode.add_argument("mode", choices=["generate", "style", "inpaint", "outpaint"],
                       help="generate/style -> base SDXL; inpaint/outpaint -> SDXL inpainting")
    pmode.set_defaults(func=cmd_mode)

    pip = sub.add_parser("inpaint", help="generative fill the current selection")
    _add_diffusion_args(pip)
    _add_nano_args(pip)
    pip.add_argument("--pad", type=float, default=0.25,
                     help="context margin around the selection (fraction)")
    pip.set_defaults(func=cmd_inpaint)

    pop = sub.add_parser("outpaint", help="extend the canvas with generated content")
    _add_diffusion_args(pop)
    _add_nano_args(pop)
    pop.add_argument("--pixels", type=int, default=256, help="border to add (px)")
    pop.add_argument("--sides", default="all",
                     help="'all' or comma list: left,right,top,bottom")
    pop.set_defaults(func=cmd_outpaint)

    pst = sub.add_parser("style", help="restyle the selection (or whole layer) via img2img")
    pst.add_argument("prompt", help="style prompt, e.g. 'watercolor painting'")
    pst.add_argument("-n", "--negative", default="")
    pst.add_argument("--model", default="sdxl", help="sdxl (default) | sd15")
    pst.add_argument("--strength", type=float, default=0.55,
                     help="transform amount: 0.3 subtle .. 0.8 strong (default 0.55)")
    pst.add_argument("--steps", type=int, default=24)
    pst.add_argument("--cfg", type=float, default=7.0)
    pst.add_argument("--seed", type=int, default=-1)
    pst.add_argument("--lora", default="",
                     help="LoRA(s) as name[:weight], comma-separated (files in models/loras/)")
    _add_nano_args(pst)
    pst.set_defaults(func=cmd_style)

    pcn = sub.add_parser("control",
                         help="ControlNet: generate following a structural map (canny/openpose)")
    pcn.add_argument("prompt")
    pcn.add_argument("--kind", default="canny", choices=["canny", "openpose"],
                     help="canny (composition lock, from canvas) | openpose (repose)")
    pcn.add_argument("--control-image", default=None, dest="control_image",
                     help="control map file (required for openpose; a pose skeleton)")
    pcn.add_argument("--control-strength", type=float, default=0.9, dest="control_strength")
    pcn.add_argument("--model", default="sd15base", help="SD1.5 base for SD1.5 ControlNets")
    pcn.add_argument("-n", "--negative", default="")
    pcn.add_argument("--steps", type=int, default=24)
    pcn.add_argument("--cfg", type=float, default=7.0)
    pcn.add_argument("--seed", type=int, default=-1)
    pcn.add_argument("--lora", default="", help="LoRA(s) as name[:weight], comma-separated")
    pcn.set_defaults(func=cmd_control)

    prp = sub.add_parser("repose-still",
                         help="generate a new-pose transparent still (Nano Banana), headless")
    prp.add_argument("input", help="a transparent chibi still, e.g. .../Chibi_NoBG/Chibi_Cyren2.png")
    prp.add_argument("-o", "--out", default=None,
                     help="output PNG (default: <input>_<Pose>.png beside the input)")
    prp.add_argument("--pose", default="prone",
                     help="pose preset (default: prone) — see repose.POSE_PROMPTS")
    prp.add_argument("--prompt", default=None,
                     help="override the pose prompt entirely")
    prp.add_argument("--extra", default=None,
                     help="append text to the preset prompt (e.g. per-character notes)")
    prp.add_argument("--ref", action="append", default=[],
                     help="extra reference image for identity (repeatable)")
    prp.add_argument("--person", action="store_true",
                     help="strip bg with mattemodel (person) instead of segmodel (object, default)")
    prp.add_argument("--nano-model", default=None, dest="nano_model",
                     help="override the OpenRouter model slug")
    prp.add_argument("--no-anchor", action="store_true",
                     help="skip trim+ground-anchor; save the raw matte-stripped cut")
    prp.add_argument("--pad-ref", default=None, dest="pad_ref", metavar="FILE",
                     help="register scale/placement to this framed still (e.g. the "
                          "idle Chibi_<Name>_Padded.png) so the pose lines up with an "
                          "existing Kling frame; overrides ground-anchor")
    prp.add_argument("--green", default=None, nargs="?", const="green", metavar="R,G,B",
                     help="fill background with a chroma-key colour for Kling "
                          "(bare flag = 0,177,64; or pass R,G,B)")
    prp.set_defaults(func=cmd_repose_still)

    prr = sub.add_parser("repose-roster",
                         help="repose every Animations/<Char> from its source still, "
                              "dropping <Char>_<Pose>.png into each folder")
    prr.add_argument("--pose", default="prone", help="pose preset (default: prone)")
    prr.add_argument("--only", default=None,
                     help="comma list of character folders to limit to")
    prr.add_argument("--limit", type=int, default=0, help="only the first N folders")
    prr.add_argument("--dry-run", action="store_true",
                     help="just resolve+print folder->still mapping, no API calls")
    prr.add_argument("--person", action="store_true",
                     help="strip bg with mattemodel (person) instead of segmodel (default)")
    prr.add_argument("--nano-model", default=None, dest="nano_model",
                     help="override the OpenRouter model slug")
    prr.set_defaults(func=cmd_repose_roster)

    return p


# Distinct exit code so the docker can tell "model not loaded" from a real failure
# and offer to load it, instead of just reporting an error.
EXIT_MODEL_NOT_LOADED = 10


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # Mirror --port into the env so _current_port() (and a launched Krita child)
    # both see it; an explicit --port wins over any inherited $NULPAINT_PORT.
    if getattr(args, "port", None):
        os.environ["NULPAINT_PORT"] = str(args.port)
    try:
        args.func(args)
    except BridgeError as e:
        sys.exit(f"nulpaint: bridge error — is Krita running with the plugin enabled? ({e})")
    except Exception as e:  # noqa: BLE001
        # The manual model manager is the source of truth — a generate verb won't
        # silently load a checkpoint. Surface the need to load with a parseable line
        # + a dedicated exit code so the docker can prompt; rethrow anything else.
        from .generate.diffusion import ModelNotLoadedError
        if isinstance(e, ModelNotLoadedError):
            print(f"NEEDS_LOAD mode={e.mode} model={e.name!r} parks={e.parked_name!r}",
                  file=sys.stderr)
            print(f"nulpaint: {e} — load it ({e.name} to GPU, {e.parked_name} to RAM) "
                  f"then retry, e.g.  nulpaint mode {e.mode}", file=sys.stderr)
            sys.exit(EXIT_MODEL_NOT_LOADED)
        raise


if __name__ == "__main__":
    main()
