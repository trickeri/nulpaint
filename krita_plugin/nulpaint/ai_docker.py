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

from krita import DockWidget, Krita  # type: ignore

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit,
        QPushButton, QToolButton, QButtonGroup, QLabel, QLineEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QFileDialog)
    from PyQt6.QtCore import QTimer
except ImportError:  # pragma: no cover — Qt5 fallback
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPlainTextEdit,
        QPushButton, QToolButton, QButtonGroup, QLabel, QLineEdit, QComboBox,
        QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QFrame, QFileDialog)
    from PyQt5.QtCore import QTimer

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
        self._train = None      # lora-training job

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

        self._sync_mode()
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(700)

    # ── SELECT ────────────────────────────────────────────────────────────────
    def _build_select_section(self):
        box, lay = _section("SELECT", expanded=True)
        btns = QHBoxLayout()
        self._sel_person = QPushButton("Person")
        self._sel_object = QPushButton("Object")
        self._sel_person.clicked.connect(lambda: self._run_select(False))
        self._sel_object.clicked.connect(lambda: self._run_select(True))
        btns.addWidget(self._sel_person)
        btns.addWidget(self._sel_object)
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

    def _cur_mode(self):
        return _MODES[self._modes.checkedId()]

    def _sync_mode(self):
        key = self._cur_mode()[0]
        is_ctrl = key in ("repose", "canny")
        self._row_img_cfg.setVisible(key == "fill")
        self._row_strength.setVisible(key == "style")
        self._row_pixels.setVisible(key == "outpaint")
        self._row_sides.setVisible(key == "outpaint")
        self._row_ctrl.setVisible(is_ctrl)
        self._row_ctrl_image.setVisible(key == "repose")
        self._row_pad.setVisible(key == "fill")
        # default model per mode; control modes use the fixed sd15base ControlNet
        default_model = self._cur_mode()[5]
        self._model.setEnabled(not is_ctrl)
        if default_model:
            j = self._model.findText(default_model)
            if j >= 0:
                self._model.setCurrentIndex(j)
        # Pre-warm the SDXL checkpoint for this mode (swap VRAM<->RAM via the
        # modelmanager) so it's resident by the time you hit Generate. Control modes
        # (repose/canny) keep using sd-cli, so no daemon swap there.
        verb = self._cur_mode()[2]
        if verb in ("inpaint", "outpaint", "style"):
            try:
                subprocess.Popen([_launcher(), "mode", verb],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

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
        return args, label

    def _sides_arg(self):
        on = [s for s, cb in self._sides.items() if cb.isChecked()]
        if len(on) == 4 or not on:
            return "all"
        m = {"L": "left", "R": "right", "T": "top", "B": "bottom"}
        return ",".join(m[s] for s in on)

    # ── actions ──────────────────────────────────────────────────────────────────
    def _spawn(self, args, status):
        if self._proc is not None:
            return False
        try:
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._go.setEnabled(False)
            self._sel_person.setEnabled(False)
            self._sel_object.setEnabled(False)
            self._status.setText(status)
            return True
        except Exception as e:  # noqa: BLE001
            self._status.setText("error: %s" % e)
            return False

    def _run_select(self, is_object):
        args = [_launcher(), "select-subject"] + (["--object"] if is_object else [])
        self._spawn(args, "selecting %s…" % ("object" if is_object else "person"))

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
        self._spawn(args, "generating (%s)… first run loads the model" % label.lower())

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
        except Exception as e:  # noqa: BLE001
            self._status.setText("error: %s" % e)

    # ── polling ──────────────────────────────────────────────────────────────────
    def canvasChanged(self, canvas):  # required override
        pass

    def _tick(self):
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
            self._proc = None
            self._go.setEnabled(True)
            self._sel_person.setEnabled(True)
            self._sel_object.setEnabled(True)
            doc = Krita.instance().activeDocument()
            if doc is not None:
                doc.refreshProjection()
            self._status.setText("done ✓" if rc == 0 else "failed (exit %d)" % rc)
            return
        if self._train is not None:
            return  # training in flight; leave its status up
        if self._cur_mode()[0] == "fill" and not self._has_selection():
            self._status.setText("make a selection to fill")
        else:
            self._status.setText("ready")
