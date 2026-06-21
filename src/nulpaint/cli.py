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


def _connect(timeout: float) -> BridgeClient:
    """Connect to the in-Krita server, retrying until Krita is ready."""
    deadline = time.monotonic() + timeout
    last: Exception | None = None
    while time.monotonic() < deadline:
        c = BridgeClient()
        try:
            c.connect()
            return c
        except OSError as e:  # Krita not up / plugin not serving yet
            last = e
            time.sleep(0.4)
    sys.exit(f"nulpaint: could not reach Krita bridge within {timeout:g}s ({last})")


# --- subcommands ------------------------------------------------------------
def cmd_launch(a: argparse.Namespace) -> None:
    if a.no_focus:
        add_no_focus_rule()
        print("nulpaint: KWin no-focus rule active for 'krita'")
    binary = _krita_binary(a.krita)
    argv = [binary] + (["--nosplash"] if a.no_splash else [])
    subprocess.Popen(argv, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"nulpaint: launched {binary} (pid detached)")


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


def cmd_select_subject(a: argparse.Namespace) -> None:
    from .vision import select_subject
    kind = "object" if a.object else "person"
    with _connect(a.wait) as c:
        res = select_subject(c, kind)
    print(f"nulpaint: selected {res['kind']} via {res['service']} "
          f"({res['w']}x{res['h']})")


def cmd_inpaint(a: argparse.Namespace) -> None:
    from .generate import inpaint
    extra = {} if a.img_cfg is None else {"img_cfg": a.img_cfg}
    with _connect(a.wait) as c:
        res = inpaint(c, a.prompt, negative=a.negative, model=a.model,
                      steps=a.steps, cfg=a.cfg, strength=a.strength,
                      seed=a.seed, pad=a.pad, lora=a.lora, **extra)
    print(f"nulpaint: inpainted [{res['model']}] {res['w']}x{res['h']} "
          f"@({res['x']},{res['y']})")


def cmd_outpaint(a: argparse.Namespace) -> None:
    from .generate import outpaint
    extra = {} if a.img_cfg is None else {"img_cfg": a.img_cfg}
    with _connect(a.wait) as c:
        res = outpaint(c, a.prompt, negative=a.negative, model=a.model,
                       pixels=a.pixels, sides=a.sides, steps=a.steps,
                       cfg=a.cfg, strength=a.strength, seed=a.seed, lora=a.lora, **extra)
    print(f"nulpaint: outpainted [{res['model']}] -> {res['width']}x{res['height']} "
          f"(+{res['pixels']}px {','.join(res['sides'])})")


def cmd_style(a: argparse.Namespace) -> None:
    from .generate import style
    with _connect(a.wait) as c:
        res = style(c, a.prompt, negative=a.negative, model=a.model,
                    strength=a.strength, steps=a.steps, cfg=a.cfg, seed=a.seed,
                    lora=a.lora)
    print(f"nulpaint: restyled [{res['model']}] {res['scope']} "
          f"{res['w']}x{res['h']} (strength {res['strength']})")


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nulpaint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait", type=float, default=20.0,
                   help="seconds to wait for the Krita bridge (default 20)")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("launch", help="launch the forked Krita")
    pl.add_argument("--no-focus", action="store_true",
                    help="install a KWin rule so Krita won't steal focus")
    pl.add_argument("--no-splash", action="store_true", help="suppress the splash screen")
    pl.add_argument("--krita", help="path to the krita binary (default: ~/.local/bin/krita)")
    pl.set_defaults(func=cmd_launch)

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

    pn = sub.add_parser("no-focus-rule", help="manage the KWin no-focus rule")
    pn.add_argument("action", choices=["add", "remove"])
    pn.set_defaults(func=cmd_no_focus_rule)

    pss = sub.add_parser("select-subject",
                         help="select the subject via the matte/seg services")
    pss.add_argument("--object", action="store_true",
                     help="arbitrary object (segmodel) instead of a person (mattemodel)")
    pss.set_defaults(func=cmd_select_subject)

    pmode = sub.add_parser("mode", help="pre-load the SDXL checkpoint for an image-model mode")
    pmode.add_argument("mode", choices=["generate", "style", "inpaint", "outpaint"],
                       help="generate/style -> base SDXL; inpaint/outpaint -> SDXL inpainting")
    pmode.set_defaults(func=cmd_mode)

    pip = sub.add_parser("inpaint", help="generative fill the current selection")
    _add_diffusion_args(pip)
    pip.add_argument("--pad", type=float, default=0.25,
                     help="context margin around the selection (fraction)")
    pip.set_defaults(func=cmd_inpaint)

    pop = sub.add_parser("outpaint", help="extend the canvas with generated content")
    _add_diffusion_args(pop)
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

    return p


# Distinct exit code so the docker can tell "model not loaded" from a real failure
# and offer to load it, instead of just reporting an error.
EXIT_MODEL_NOT_LOADED = 10


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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
