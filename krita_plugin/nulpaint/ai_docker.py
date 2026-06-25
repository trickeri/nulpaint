"""NulPaint AI docker — the in-Krita GUI for the local generative-AI suite.

A single scrolling panel of collapsible sections that surface everything the
external `nulpaint` CLI can do, all driven over the verified bridge:

  SELECT          Person / Object subject selection (matte / seg services)
  GENERATE        Fill · Style · Outpaint · Repose · Canny — prompt-driven
                  diffusion (sd.cpp) with per-mode knobs + a LoRA picker
  LORA TRAINING   kick off lora-training/train-lora.sh on a dataset folder

On any action the docker `subprocess.Popen`s the `nulpaint` launcher (which owns
the venv + sd-cli + model services) and that process drives THIS Krita back over
the loopback bridge. The docker just fires the job, polls for completion, and
refreshes the canvas — the heavy lifting stays external.

Stdlib + Qt only (runs in Krita's embedded interpreter). PyQt6 with a PyQt5
fallback; enum use is kept minimal so it works on either binding.
"""
import os
import subprocess
import time

from krita import DockWidget, Krita  # type: ignore

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit,
        QPushButton, QToolButton, QButtonGroup, QLabel, QLineEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QFileDialog,
        QMessageBox, QApplication, QProgressBar, QListWidget, QListWidgetItem,
        QMenu)
    from PyQt6.QtCore import QTimer, Qt, QSettings
except ImportError:  # pragma: no cover — Qt5 fallback
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit,
        QPushButton, QToolButton, QButtonGroup, QLabel, QLineEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QFileDialog,
        QMessageBox, QApplication, QProgressBar, QListWidget, QListWidgetItem,
        QMenu)
    from PyQt5.QtCore import QTimer, Qt, QSettings

# Mirror of nulpaint.cli.EXIT_MODEL_NOT_LOADED — the CLI exits with this code when
# a generate verb needs an SDXL checkpoint the model manager hasn't put on the GPU,
# so the docker can prompt to load it instead of reporting a generic failure.
_EXIT_MODEL_NOT_LOADED = 10

# sd-server (stable-diffusion.cpp fork) writes "<step> <steps> <epoch_ms>" here every
# sampling step so the docker can show a real determinate progress bar for SDXL gen.
_DIFFUSION_PROGRESS = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "nulpaint", "diffusion.progress")

# ── paths ────────────────────────────────────────────────────────────────────
_KRITA_ROOT = os.path.expanduser("~/programming/Krita")
_LORA_DIR = os.environ.get("NULPAINT_LORA_DIR",
                           os.path.join(_KRITA_ROOT, "models", "loras"))
_TRAIN_SH = os.path.join(_KRITA_ROOT, "lora-training", "train-lora.sh")


def _launcher():
    for p in (os.path.expanduser("~/.local/bin/nulpaint"),
              os.path.join(_KRITA_ROOT, "nulpaint", "nulpaint")):
        if os.path.exists(p):
            return p
    return "nulpaint"


def _list_loras():
    try:
        names = [os.path.splitext(f)[0] for f in sorted(os.listdir(_LORA_DIR))
                 if f.endswith(".safetensors")]
    except OSError:
        names = []
    return names


# ── generate modes ────────────────────────────────────────────────────────────
# key, label, CLI verb, fixed extra args, needs-a-selection, default model
_MODES = [
    ("fill",     "Fill",     "inpaint",  [],                          True,  "sd15"),
    ("style",    "Style",    "style",    [],                          False, "sdxl"),
    ("outpaint", "Outpaint", "outpaint", [],                          False, "sd15"),
    ("repose",   "Repose",   "control",  ["--kind", "openpose"],      False, None),
    ("canny",    "Canny",    "control",  ["--kind", "canny"],         False, None),
]
_MODE_KEYS = [m[0] for m in _MODES]

# The active GENERATE mode must read clearly. Breeze's default checked state on a
# QPushButton is too subtle, so we paint the accent (theme highlight = Troy's
# cornflower-blue) into the :checked state, with a matching border for hover.
_MODE_BTN_STYLE = """
QPushButton {
    padding: 6px 4px;
    border: 1px solid palette(mid);
    border-radius: 3px;
}
QPushButton:hover {
    border: 1px solid palette(highlight);
}
QPushButton:checked {
    background-color: palette(highlight);
    color: palette(highlighted-text);
    border: 1px solid palette(highlight);
    font-weight: bold;
}
"""


def _section(title, expanded=True):
    """A collapsible section: a checkable header button + a content QWidget.

    Returns (container, content_layout). Toggling the header shows/hides the
    body and flips a ▾/▸ glyph — no arrow enums, so it's binding-agnostic.
    """
    box = QWidget()
    outer = QVBoxLayout(box)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(2)

    header = QToolButton()
    header.setCheckable(True)
    header.setChecked(expanded)
    header.setText(("▾  " if expanded else "▸  ") + title)
    header.setStyleSheet("QToolButton { border: none; font-weight: bold; "
                         "text-align: left; padding: 4px 2px; }")

    body = QWidget()
    body_lay = QVBoxLayout(body)
    body_lay.setContentsMargins(8, 2, 2, 6)
    body_lay.setSpacing(4)
    body.setVisible(expanded)

    def _toggle(on):
        body.setVisible(on)
        header.setText(("▾  " if on else "▸  ") + title)
    header.toggled.connect(_toggle)

    outer.addWidget(header)
    outer.addWidget(body)
    return box, body_lay


def _row(label, widget):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lab = QLabel(label)
    lab.setMinimumWidth(64)
    lay.addWidget(lab)
    lay.addWidget(widget, 1)
    return w


def _hsep():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine if hasattr(QFrame, "Shape")
                       else QFrame.HLine)
    line.setStyleSheet("color: rgba(128,128,128,80);")
    return line


class NulPaintAIDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NulPaint AI")
        self._proc = None       # generate / select job
        self._proc_kind = None  # "gen" | "select" | "load" — what _proc is doing
        self._proc_err = None   # temp file capturing a gen job's stderr (for failure msgs)
        self._last_gen = None   # (args, label, verb) of the last generate spawned
        self._pending_gen = None  # (args, label) to retry after a confirmed model load
        self._train = None      # lora-training job
        self._busy_active = False   # window busy-cursor + status-bar progress shown
        self._sb_progress = None    # QProgressBar parked in the status bar
        self._gen_started = 0       # epoch-ms a generate spawned (ignores stale progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root = QWidget()
        self._lay = QVBoxLayout(root)
        self._lay.setContentsMargins(6, 6, 6, 6)
        self._lay.setSpacing(6)

        self._build_select_section()
        self._lay.addWidget(_hsep())
        self._build_generate_section()
        self._lay.addWidget(_hsep())
        self._build_training_section()

        self._lay.addStretch(1)
        self._status = QLabel("—")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: rgba(160,160,160,255); padding-top: 4px;")
        self._lay.addWidget(self._status)

        scroll.setWidget(root)
        self.setWidget(scroll)

        # Restore the last-used choices (select kind, generate mode, model, LoRA)
        # before the first _sync_mode so the right knobs show. _loaded gates saving
        # so the restore itself (and the init _sync_mode) doesn't clobber the store.
        self._loaded = False
        self._restore_settings()
        self._sync_mode()
        self._loaded = True
        # Persist whenever the user changes the model or LoRA picker too.
        self._model.currentIndexChanged.connect(lambda _i: self._save_settings())
        self._lora.currentIndexChanged.connect(lambda _i: self._save_settings())

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(700)

    # ── SELECT ────────────────────────────────────────────────────────────────
    def _build_select_section(self):
        box, lay = _section("SELECT", expanded=True)
        btns = QHBoxLayout()
        # Checkable + exclusive so the active kind stays lit (one is always on),
        # matching the GENERATE mode row. Clicking still runs the select immediately.
        self._sel_person = QPushButton("Person")
        self._sel_object = QPushButton("Object")
        self._sel_group = QButtonGroup(box)
        self._sel_group.setExclusive(True)
        for b, is_obj in ((self._sel_person, False), (self._sel_object, True)):
            b.setCheckable(True)
            b.setStyleSheet(_MODE_BTN_STYLE)
            self._sel_group.addButton(b)
            b.clicked.connect(lambda _checked=False, o=is_obj: self._run_select(o))
            btns.addWidget(b)
        self._sel_person.setChecked(True)   # default kind = person
        lay.addLayout(btns)
        hint = QLabel("Selects the subject as a mask you can fill, mask, or cut.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(140,140,140,255); font-size: 11px;")
        lay.addWidget(hint)
        self._lay.addWidget(box)

    # ── GENERATE ──────────────────────────────────────────────────────────────
    def _build_generate_section(self):
        box, lay = _section("GENERATE", expanded=True)

        # mode buttons (3 + 2 grid so the labels fit a narrow docker)
        grid = QGridLayout()
        grid.setSpacing(3)
        self._modes = QButtonGroup(box)
        self._modes.setExclusive(True)
        for i, (key, label, *_rest) in enumerate(_MODES):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(i == 0)
            # Breeze checkable buttons barely change when checked — force a clear
            # accent on the active mode so it stays obvious which one is selected.
            b.setStyleSheet(_MODE_BTN_STYLE)
            self._modes.addButton(b, i)
            grid.addWidget(b, i // 3, i % 3)
        try:                                   # PyQt6
            self._modes.idClicked.connect(lambda _i: self._sync_mode())
        except AttributeError:                 # PyQt5
            self._modes.buttonClicked.connect(lambda _b: self._sync_mode())
        lay.addLayout(grid)

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("prompt…")
        self._prompt.setFixedHeight(56)
        lay.addWidget(self._prompt)

        # ── engine: local SDXL vs Nano Banana Pro (cloud, fill/style/outpaint) ─
        self._engine = QComboBox()
        self._engine.addItem("Local SDXL", "local")
        self._engine.addItem("Nano Banana Pro", "nanobanana")
        self._engine.setToolTip("Local stable-diffusion.cpp, or Nano Banana Pro "
                                "(Gemini 3 Pro Image) via OpenRouter")
        self._engine.currentIndexChanged.connect(lambda _i: (self._sync_mode(), self._save_settings()))
        self._row_engine = _row("Engine", self._engine)
        lay.addWidget(self._row_engine)

        # ── references (Nano Banana): base image is auto-included; add more ────
        refs_w = QWidget()
        refs_l = QVBoxLayout(refs_w)
        refs_l.setContentsMargins(0, 0, 0, 0)
        refs_l.setSpacing(3)
        cap = QLabel("Reference images (base image auto-included)")
        cap.setStyleSheet("color: rgba(140,140,140,255); font-size: 11px;")
        refs_l.addWidget(cap)
        self._refs = []                       # list of ("file", path) | ("layer", name)
        self._refs_list = QListWidget()
        self._refs_list.setFixedHeight(64)
        refs_l.addWidget(self._refs_list)
        rb = QHBoxLayout()
        rb.setContentsMargins(0, 0, 0, 0)
        for text, slot in (("+ File", self._add_ref_file),
                           ("+ Layer", self._add_ref_layer),
                           ("Remove", self._remove_ref),
                           ("Clear", self._clear_refs)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            rb.addWidget(b)
        refs_l.addLayout(rb)
        self._row_refs = refs_w
        lay.addWidget(self._row_refs)

        # ── per-mode knobs (shown/hidden by _sync_mode) ──────────────────────
        self._img_cfg = QDoubleSpinBox()
        self._img_cfg.setRange(0.5, 15.0)
        self._img_cfg.setSingleStep(0.5)
        self._img_cfg.setValue(1.5)
        self._img_cfg.setToolTip("Image guidance: ~1.5 bold replace · high = seamless fill")
        self._row_img_cfg = _row("Boldness", self._img_cfg)
        lay.addWidget(self._row_img_cfg)

        self._strength = QDoubleSpinBox()
        self._strength.setRange(0.0, 1.0)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(0.55)
        self._strength.setToolTip("Transform amount: 0.3 subtle .. 0.8 strong")
        self._row_strength = _row("Strength", self._strength)
        lay.addWidget(self._row_strength)

        # outpaint: pixels + sides
        self._pixels = QSpinBox()
        self._pixels.setRange(32, 2048)
        self._pixels.setSingleStep(64)
        self._pixels.setValue(256)
        self._row_pixels = _row("Border px", self._pixels)
        lay.addWidget(self._row_pixels)

        sides_w = QWidget()
        sides_l = QHBoxLayout(sides_w)
        sides_l.setContentsMargins(0, 0, 0, 0)
        sides_l.addWidget(QLabel("Sides"))
        self._sides = {}
        for s in ("L", "R", "T", "B"):
            cb = QCheckBox(s)
            cb.setChecked(True)
            self._sides[s] = cb
            sides_l.addWidget(cb)
        sides_l.addStretch(1)
        self._row_sides = sides_w
        lay.addWidget(self._row_sides)

        # control: strength + optional skeleton image
        self._ctrl_strength = QDoubleSpinBox()
        self._ctrl_strength.setRange(0.0, 2.0)
        self._ctrl_strength.setSingleStep(0.05)
        self._ctrl_strength.setValue(0.9)
        self._row_ctrl = _row("Control", self._ctrl_strength)
        lay.addWidget(self._row_ctrl)

        ctrl_img_w = QWidget()
        ctrl_img_l = QHBoxLayout(ctrl_img_w)
        ctrl_img_l.setContentsMargins(0, 0, 0, 0)
        ctrl_img_l.addWidget(QLabel("Pose img"))
        self._ctrl_image = QLineEdit()
        self._ctrl_image.setPlaceholderText("auto from canvas (optional)")
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._pick_control_image)
        ctrl_img_l.addWidget(self._ctrl_image, 1)
        ctrl_img_l.addWidget(browse)
        self._row_ctrl_image = ctrl_img_w
        lay.addWidget(self._row_ctrl_image)

        # ── LoRA picker (shared across generate modes) ───────────────────────
        lora_w = QWidget()
        lora_l = QHBoxLayout(lora_w)
        lora_l.setContentsMargins(0, 0, 0, 0)
        lora_l.addWidget(QLabel("LoRA"))
        self._lora = QComboBox()
        self._refresh_loras()
        self._lora_weight = QDoubleSpinBox()
        self._lora_weight.setRange(0.0, 1.5)
        self._lora_weight.setSingleStep(0.1)
        self._lora_weight.setValue(0.8)
        refresh = QPushButton("⟳")
        refresh.setFixedWidth(28)
        refresh.clicked.connect(self._refresh_loras)
        lora_l.addWidget(self._lora, 1)
        lora_l.addWidget(self._lora_weight)
        lora_l.addWidget(refresh)
        self._row_lora = lora_w
        lay.addWidget(lora_w)

        # ── Advanced (nested collapsible) ────────────────────────────────────
        adv_box, adv = _section("Advanced", expanded=False)
        self._model = QComboBox()
        self._model.addItems(["sd15", "sdxl"])
        adv.addWidget(_row("Model", self._model))
        self._negative = QLineEdit()
        self._negative.setPlaceholderText("negative prompt (optional)")
        adv.addWidget(_row("Negative", self._negative))
        self._steps = QSpinBox()
        self._steps.setRange(1, 150)
        self._steps.setValue(20)
        adv.addWidget(_row("Steps", self._steps))
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(0.0, 30.0)
        self._cfg.setSingleStep(0.5)
        self._cfg.setValue(7.0)
        adv.addWidget(_row("CFG", self._cfg))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2_147_483_647)
        self._seed.setValue(-1)
        self._seed.setToolTip("-1 = random")
        adv.addWidget(_row("Seed", self._seed))
        self._pad = QDoubleSpinBox()
        self._pad.setRange(0.0, 1.0)
        self._pad.setSingleStep(0.05)
        self._pad.setValue(0.25)
        self._pad.setToolTip("Context margin around the selection (fill only)")
        self._row_pad = _row("Pad", self._pad)
        adv.addWidget(self._row_pad)
        self._row_adv = adv_box
        lay.addWidget(adv_box)

        self._go = QPushButton("Generate")
        self._go.clicked.connect(self._run_generate)
        lay.addWidget(self._go)

        self._lay.addWidget(box)

    # ── LORA TRAINING ─────────────────────────────────────────────────────────
    def _build_training_section(self):
        box, lay = _section("LORA TRAINING", expanded=False)
        self._tr_name = QLineEdit()
        self._tr_name.setPlaceholderText("lora name")
        lay.addWidget(_row("Name", self._tr_name))

        ds_w = QWidget()
        ds_l = QHBoxLayout(ds_w)
        ds_l.setContentsMargins(0, 0, 0, 0)
        ds_l.addWidget(QLabel("Dataset"))
        self._tr_dataset = QLineEdit()
        self._tr_dataset.setPlaceholderText("folder of images")
        ds_browse = QPushButton("…")
        ds_browse.setFixedWidth(28)
        ds_browse.clicked.connect(self._pick_dataset)
        ds_l.addWidget(self._tr_dataset, 1)
        ds_l.addWidget(ds_browse)
        lay.addWidget(ds_w)

        self._tr_base = QComboBox()
        self._tr_base.addItems(["sd15", "sdxl"])
        lay.addWidget(_row("Base", self._tr_base))
        self._tr_epochs = QSpinBox()
        self._tr_epochs.setRange(1, 200)
        self._tr_epochs.setValue(10)
        lay.addWidget(_row("Epochs", self._tr_epochs))

        self._tr_go = QPushButton("Train LoRA")
        self._tr_go.clicked.connect(self._run_train)
        lay.addWidget(self._tr_go)
        hint = QLabel("Self-trained LoRAs are sd.cpp-compatible. Output lands in "
                      "models/loras/ — hit ⟳ above to pick it up.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: rgba(140,140,140,255); font-size: 11px;")
        lay.addWidget(hint)
        self._lay.addWidget(box)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _refresh_loras(self):
        cur = self._lora.currentText() if self._lora.count() else ""
        self._lora.clear()
        self._lora.addItem("none")
        for n in _list_loras():
            self._lora.addItem(n)
        i = self._lora.findText(cur)
        if i >= 0:
            self._lora.setCurrentIndex(i)

    def _pick_control_image(self):
        path, _ = QFileDialog.getOpenFileName(self.widget(), "Pose / control image")
        if path:
            self._ctrl_image.setText(path)

    def _pick_dataset(self):
        path = QFileDialog.getExistingDirectory(self.widget(), "Training dataset folder")
        if path:
            self._tr_dataset.setText(path)

    # ── Nano Banana reference images ─────────────────────────────────────────
    def _refs_refresh_list(self):
        self._refs_list.clear()
        for kind, val in self._refs:
            label = ("file: " + val.rsplit("/", 1)[-1]) if kind == "file" else ("layer: " + val)
            self._refs_list.addItem(QListWidgetItem(label))

    def _add_ref_file(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self.widget(), "Reference image(s)", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*)")
        for p in paths:
            self._refs.append(("file", p))
        self._refs_refresh_list()

    def _add_ref_layer(self):
        doc = Krita.instance().activeDocument()
        if doc is None:
            return
        names = []

        def walk(node):
            for ch in reversed(node.childNodes()):   # top-to-bottom
                names.append(ch.name())
                walk(ch)
        walk(doc.rootNode())
        if not names:
            return
        menu = QMenu(self.widget())
        for n in names:
            menu.addAction(n, lambda nm=n: (self._refs.append(("layer", nm)),
                                            self._refs_refresh_list()))
        pos = self._refs_list.mapToGlobal(self._refs_list.rect().bottomLeft())
        (menu.exec if hasattr(menu, "exec") else menu.exec_)(pos)

    def _remove_ref(self):
        row = self._refs_list.currentRow()
        if 0 <= row < len(self._refs):
            del self._refs[row]
            self._refs_refresh_list()

    def _clear_refs(self):
        self._refs = []
        self._refs_refresh_list()

    def _cur_mode(self):
        return _MODES[self._modes.checkedId()]

    def _nano_capable(self, key):
        return key in ("fill", "style", "outpaint")

    def _sync_mode(self):
        self._save_settings()   # remember the active generate mode for next launch
        key = self._cur_mode()[0]
        is_ctrl = key in ("repose", "canny")
        # Nano Banana Pro (cloud) is only offered for fill/style/outpaint; control
        # modes (ControlNet) are local-only. When the cloud engine is active the
        # local sampler knobs (sd.cpp-only) are irrelevant, so hide them.
        nano = self._nano_capable(key) and self._engine.currentData() == "nanobanana"
        self._row_engine.setVisible(self._nano_capable(key))
        self._row_refs.setVisible(nano)
        self._row_img_cfg.setVisible(key == "fill" and not nano)
        self._row_strength.setVisible(key == "style" and not nano)
        self._row_pixels.setVisible(key == "outpaint")     # border size matters for both
        self._row_sides.setVisible(key == "outpaint")
        self._row_ctrl.setVisible(is_ctrl)
        self._row_ctrl_image.setVisible(key == "repose")
        self._row_pad.setVisible(key == "fill" and not nano)
        self._row_lora.setVisible(not nano)
        self._row_adv.setVisible(not nano)
        # default model per mode; control modes use the fixed sd15base ControlNet
        default_model = self._cur_mode()[5]
        self._model.setEnabled(not is_ctrl)
        if default_model:
            j = self._model.findText(default_model)
            if j >= 0:
                self._model.setCurrentIndex(j)
        # NB: selecting a mode no longer pre-warms its SDXL checkpoint. The manual
        # model manager is the source of truth for VRAM, so we don't move models on a
        # mere radio-button click — Generate prompts to load the checkpoint if it
        # isn't already on the GPU (see _tick's NEEDS_LOAD handling).

    def _has_selection(self):
        doc = Krita.instance().activeDocument()
        return doc is not None and doc.selection() is not None

    def _lora_arg(self):
        name = self._lora.currentText()
        if not name or name == "none":
            return []
        return ["--lora", "%s:%s" % (name, self._lora_weight.value())]

    # ── command assembly ────────────────────────────────────────────────────────
    def _build_generate_command(self):
        key, label, verb, extra, _needs_sel, _dm = self._cur_mode()
        prompt = self._prompt.toPlainText().strip()
        args = [_launcher(), verb, prompt, *extra]

        if not (key in ("repose", "canny")):
            args += ["--model", self._model.currentText()]
        neg = self._negative.text().strip()
        if neg:
            args += ["-n", neg]
        args += ["--steps", str(self._steps.value()),
                 "--cfg", str(self._cfg.value()),
                 "--seed", str(self._seed.value())]

        if key == "fill":
            args += ["--img-cfg", str(self._img_cfg.value()),
                     "--pad", str(self._pad.value())]
        elif key == "style":
            args += ["--strength", str(self._strength.value())]
        elif key == "outpaint":
            args += ["--pixels", str(self._pixels.value()),
                     "--sides", self._sides_arg()]
        elif key in ("repose", "canny"):
            args += ["--control-strength", str(self._ctrl_strength.value())]
            img = self._ctrl_image.text().strip()
            if key == "repose" and img:
                args += ["--control-image", img]
        args += self._lora_arg()

        # Cloud engine: route fill/style/outpaint through Nano Banana Pro + refs.
        # (The local sampler flags above are accepted but ignored by the nano path.)
        if self._nano_capable(key) and self._engine.currentData() == "nanobanana":
            args += ["--engine", "nanobanana"]
            for kind, val in self._refs:
                args += (["--ref", val] if kind == "file" else ["--ref-layer", val])
        return args, label

    def _sides_arg(self):
        on = [s for s, cb in self._sides.items() if cb.isChecked()]
        if len(on) == 4 or not on:
            return "all"
        m = {"L": "left", "R": "right", "T": "top", "B": "bottom"}
        return ",".join(m[s] for s in on)

    def _read_proc_error(self, rc):
        """Last meaningful line of a failed gen job's stderr (its exception message),
        falling back to the exit code."""
        path = getattr(self, "_proc_err", None)
        if path:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    lines = [ln.strip() for ln in fh if ln.strip()]
                if lines:
                    return lines[-1][:200]
            except OSError:
                pass
        return "exit %d" % rc

    def _cleanup_proc_error(self):
        path = getattr(self, "_proc_err", None)
        self._proc_err = None
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    # ── actions ──────────────────────────────────────────────────────────────────
    def _spawn(self, args, status, kind="gen"):
        if self._proc is not None:
            return False
        try:
            # Capture stderr for generate jobs so a failure (e.g. a Nano Banana /
            # OpenRouter error or missing API key) surfaces in the status line
            # instead of dying silently into DEVNULL.
            self._proc_err = None
            err = subprocess.DEVNULL
            if kind == "gen":
                import tempfile
                fh = tempfile.NamedTemporaryFile(
                    prefix="nulpaint-gen-", suffix=".log", delete=False)
                self._proc_err = fh.name
                fh.close()
                err = open(self._proc_err, "wb")
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=err,
                start_new_session=True)
            if err is not subprocess.DEVNULL:
                err.close()
            self._proc_kind = kind
            if kind == "gen":
                self._gen_started = int(time.time() * 1000)
            self._go.setEnabled(False)
            self._sel_person.setEnabled(False)
            self._sel_object.setEnabled(False)
            self._status.setText(status)
            self._sync_busy()
            return True
        except Exception as e:  # noqa: BLE001
            self._status.setText("error: %s" % e)
            return False

    # ── settings persistence ──────────────────────────────────────────────────
    def _settings(self):
        return QSettings("nuldrums", "nulpaint-ai")

    def _save_settings(self):
        if not getattr(self, "_loaded", False):
            return  # don't persist during restore / initial build
        s = self._settings()
        s.setValue("select_kind", "object" if self._sel_object.isChecked() else "person")
        s.setValue("gen_mode", self._modes.checkedId())
        s.setValue("model", self._model.currentText())
        s.setValue("lora", self._lora.currentText())
        s.setValue("engine", self._engine.currentData())

    def _restore_settings(self):
        s = self._settings()
        kind = s.value("select_kind", "person")
        (self._sel_object if kind == "object" else self._sel_person).setChecked(True)
        try:
            mode_id = int(s.value("gen_mode", 0))
        except (TypeError, ValueError):
            mode_id = 0
        btn = self._modes.button(mode_id)
        if btn:
            btn.setChecked(True)
        for combo, key in ((self._model, "model"), (self._lora, "lora")):
            val = s.value(key, "")
            if val:
                j = combo.findText(val)
                if j >= 0:
                    combo.setCurrentIndex(j)
        eng = s.value("engine", "local")
        je = self._engine.findData(eng)
        if je >= 0:
            self._engine.setCurrentIndex(je)

    def _run_select(self, is_object):
        self._save_settings()   # remember person/object for next launch
        args = [_launcher(), "select-subject"] + (["--object"] if is_object else [])
        self._spawn(args, "selecting %s…" % ("object" if is_object else "person"),
                    kind="select")

    def _run_generate(self):
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._status.setText("enter a prompt")
            return
        key = self._cur_mode()[0]
        if key == "fill" and not self._has_selection():
            self._status.setText("make a selection first")
            return
        args, label = self._build_generate_command()
        self._last_gen = (args, label, self._cur_mode()[2])
        self._spawn(args, "generating (%s)…" % label.lower(), kind="gen")

    # Model needed by each generate mode (for the load prompt). control modes
    # (repose/canny) cold-spawn sd-cli, so they're not gated here.
    _MODE_MODEL = {
        "inpaint":  ("SDXL Inpainting", "SDXL Base"),
        "outpaint": ("SDXL Inpainting", "SDXL Base"),
        "style":    ("SDXL Base",       "SDXL Inpainting"),
    }

    def _prompt_load(self, verb, args, label):
        """A generate verb reported its checkpoint isn't on the GPU. Ask before
        loading — the manual model manager owns VRAM, so we never load silently."""
        model, parked = self._MODE_MODEL.get(verb, (verb, "the other model"))
        ans = QMessageBox.question(
            self, "Load model?",
            "%s isn't loaded on the GPU.\n\nLoad it now for %s?\n"
            "(%s moves to the GPU; %s parks in system RAM.)"
            % (model, label.lower(), model, parked),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            self._status.setText("%s not loaded — skipped" % model)
            return
        self._pending_gen = (args, label)
        self._spawn([_launcher(), "mode", verb], "loading %s…" % model, kind="load")

    def _run_train(self):
        if self._train is not None:
            return
        name = self._tr_name.text().strip()
        dataset = self._tr_dataset.text().strip()
        if not name or not dataset:
            self._status.setText("training needs a name + dataset folder")
            return
        if not os.path.isdir(dataset):
            self._status.setText("dataset folder not found")
            return
        args = ["bash", _TRAIN_SH, name, dataset, self._tr_base.currentText(),
                str(self._tr_epochs.value())]
        try:
            self._train = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._tr_go.setEnabled(False)
            self._status.setText("training '%s'… this takes a while" % name)
            self._sync_busy()
        except Exception as e:  # noqa: BLE001
            self._status.setText("error: %s" % e)

    # ── busy feedback (window status bar) ─────────────────────────────────────────
    def _statusbar(self):
        """The Krita main-window bottom status bar, or None if unavailable."""
        try:
            return Krita.instance().activeWindow().qwindow().statusBar()
        except Exception:  # noqa: BLE001
            return None

    def _diffusion_progress(self):
        """(step, total) from sd-server's live progress file, or None.

        Honours it only while fresh (updated in the last 3 s) and newer than the
        current generate's start, so stale/previous-run data never shows."""
        try:
            with open(_DIFFUSION_PROGRESS, "r") as f:
                step_s, total_s, ms_s = f.read().split()[:3]
            step, total, ms = int(step_s), int(total_s), int(ms_s)
        except Exception:  # noqa: BLE001
            return None
        if total <= 0:
            return None
        now_ms = int(time.time() * 1000)
        if now_ms - ms > 3000 or ms < self._gen_started - 500:
            return None
        return (max(0, min(step, total)), total)

    def _sync_busy(self):
        """Reflect 'is any AI job running' into the window: a busy cursor + an
        indeterminate progress bar + a message in the bottom status bar. Driven by
        job presence so it survives the load→generate chain and ends only when
        everything is truly done."""
        active = self._proc is not None or self._train is not None
        sb = self._statusbar()
        if active:
            if not self._busy_active:
                self._busy_active = True
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                if sb is not None and self._sb_progress is None:
                    self._sb_progress = QProgressBar()
                    self._sb_progress.setRange(0, 0)        # indeterminate "busy" sweep
                    self._sb_progress.setTextVisible(False)
                    self._sb_progress.setMaximumWidth(140)
                    sb.addPermanentWidget(self._sb_progress)
            # SDXL generate exposes real step progress → show a determinate bar with
            # "step/total"; everything else (select, control, load) stays indeterminate.
            msg = self._status.text()
            prog = self._diffusion_progress() if self._proc_kind == "gen" else None
            if self._sb_progress is not None:
                if prog is not None:
                    step, total = prog
                    self._sb_progress.setRange(0, total)
                    self._sb_progress.setValue(step)
                    self._sb_progress.setTextVisible(True)
                    msg = "%s  %d/%d" % (self._status.text(), step, total)
                elif self._sb_progress.maximum() != 0:
                    self._sb_progress.setRange(0, 0)        # back to busy sweep
                    self._sb_progress.setTextVisible(False)
            if sb is not None:
                sb.showMessage("NulPaint: " + msg)
        elif self._busy_active:
            self._busy_active = False
            QApplication.restoreOverrideCursor()
            if sb is not None and self._sb_progress is not None:
                sb.removeWidget(self._sb_progress)
                self._sb_progress.deleteLater()
                self._sb_progress = None
            if sb is not None:
                sb.showMessage("NulPaint: " + self._status.text(), 4000)

    # ── polling ──────────────────────────────────────────────────────────────────
    def canvasChanged(self, canvas):  # required override
        pass

    def _tick(self):
        self._sync_busy()   # keep the window busy-state in sync with running jobs
        if self._train is not None and self._train.poll() is not None:
            rc = self._train.returncode
            self._train = None
            self._tr_go.setEnabled(True)
            self._refresh_loras()
            self._status.setText("training done ✓" if rc == 0
                                 else "training failed (exit %d)" % rc)
            return
        if self._proc is not None:
            if self._proc.poll() is None:
                return
            rc = self._proc.returncode
            kind = self._proc_kind
            self._proc = None
            self._proc_kind = None
            self._go.setEnabled(True)
            self._sel_person.setEnabled(True)
            self._sel_object.setEnabled(True)

            if kind == "load":
                # A confirmed model load finished. On success, run the generate it
                # was loaded for; otherwise report and drop it.
                pending, self._pending_gen = self._pending_gen, None
                if rc == 0 and pending:
                    args, label = pending
                    self._spawn(args, "generating (%s)…" % label.lower(), kind="gen")
                else:
                    self._status.setText("model load failed (exit %d)" % rc)
                return

            if kind == "gen" and rc == _EXIT_MODEL_NOT_LOADED and self._last_gen:
                # The checkpoint wasn't on the GPU. Prompt to load it, then retry —
                # the manual model manager stays the source of truth for VRAM.
                self._cleanup_proc_error()
                args, label, verb = self._last_gen
                self._prompt_load(verb, args, label)
                return

            doc = Krita.instance().activeDocument()
            if doc is not None:
                doc.refreshProjection()
            if rc == 0:
                self._status.setText("done ✓")
            else:
                self._status.setText("failed: " + self._read_proc_error(rc))
            self._cleanup_proc_error()
            return
        if self._train is not None:
            return  # training in flight; leave its status up
        if self._cur_mode()[0] == "fill" and not self._has_selection():
            self._status.setText("make a selection to fill")
        else:
            self._status.setText("ready")
