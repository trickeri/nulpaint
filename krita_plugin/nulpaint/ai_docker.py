"""NulPaint AI docker — the in-Krita GUI for generative editing.

A prompt field + Fill / Style / Repose mode toggles + Generate. On Generate it
shells out to the external `nulpaint` CLI (which has the venv + sd-cli), and that
process drives THIS Krita over the bridge (image.get → sd-cli → layer write). The
docker just fires it and polls for completion — the heavy lifting stays external.

Stdlib + Qt only (runs in Krita's interpreter). PyQt6 with a PyQt5 fallback.
"""
import os
import subprocess

from krita import DockWidget, Krita  # type: ignore

try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
                                 QPushButton, QButtonGroup, QLabel, QLineEdit)
    from PyQt6.QtCore import QTimer
except ImportError:  # pragma: no cover — Qt5 fallback
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
                                 QPushButton, QButtonGroup, QLabel, QLineEdit)
    from PyQt5.QtCore import QTimer

# (label, CLI verb, extra args, needs a selection)
_MODES = [
    ("Fill", "inpaint", [], True),
    ("Style", "style", [], False),
    ("Repose", "control", ["--kind", "openpose"], False),
]


def _launcher():
    for p in (os.path.expanduser("~/.local/bin/nulpaint"),
              os.path.expanduser("~/programming/Krita/nulpaint/nulpaint")):
        if os.path.exists(p):
            return p
    return "nulpaint"


class NulPaintAIDocker(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NulPaint AI")
        self._proc = None

        root = QWidget()
        lay = QVBoxLayout(root)

        modes = QHBoxLayout()
        self._modes = QButtonGroup(root)
        self._modes.setExclusive(True)
        for i, (label, *_rest) in enumerate(_MODES):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self._modes.addButton(b, i)
            modes.addWidget(b)
        lay.addLayout(modes)

        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("prompt…")
        self._prompt.setFixedHeight(64)
        lay.addWidget(self._prompt)

        self._lora = QLineEdit()
        self._lora.setPlaceholderText("LoRA  name:weight  (optional)")
        lay.addWidget(self._lora)

        self._go = QPushButton("Generate")
        self._go.clicked.connect(self._run)
        lay.addWidget(self._go)

        self._status = QLabel("—")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)
        lay.addStretch(1)
        self.setWidget(root)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(700)

    def canvasChanged(self, canvas):  # required override
        pass

    def _has_selection(self):
        doc = Krita.instance().activeDocument()
        return doc is not None and doc.selection() is not None

    def _tick(self):
        # Poll a running job, then reflect selection/readiness.
        if self._proc is not None:
            if self._proc.poll() is None:
                return
            rc = self._proc.returncode
            self._proc = None
            self._go.setEnabled(True)
            self._status.setText("done ✓" if rc == 0 else f"failed (exit {rc})")
            doc = Krita.instance().activeDocument()
            if doc is not None:
                doc.refreshProjection()
            return
        needs_sel = _MODES[self._modes.checkedId()][3]
        if needs_sel and not self._has_selection():
            self._status.setText("make a selection to fill")
        else:
            self._status.setText("ready")

    def _run(self):
        if self._proc is not None:
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._status.setText("enter a prompt")
            return
        label, verb, extra, needs_sel = _MODES[self._modes.checkedId()]
        if needs_sel and not self._has_selection():
            self._status.setText("make a selection first")
            return
        args = [_launcher(), verb, prompt, *extra]
        lora = self._lora.text().strip()
        if lora:
            args += ["--lora", lora]
        try:
            self._proc = subprocess.Popen(
                args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True)
            self._go.setEnabled(False)
            self._status.setText(f"generating ({label.lower()})… first run loads the model")
        except Exception as e:  # noqa: BLE001
            self._status.setText(f"error: {e}")
