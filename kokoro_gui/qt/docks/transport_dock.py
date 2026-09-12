"""Transport / Generate dock (bottom-right of the drawing's 2x2 grid, see
Claude/PLAN_ui_shell_redesign.md section 1).

What `QtTTSApp._build_action_bar` used to build as the central widget,
moved into a dock and reshaped into three rows:

1. play / pause / stop, elapsed / total time, loop toggle - driven by
   `kokoro_gui.audio.transport.Transport` through the app.
2. Preview, Generate (a `QToolButton` whose menu holds "Generate dirty
   clips", "Auto-split then generate" and the checkable "Split by
   paragraph"), Cancel.
3. One progress bar carrying the status/detail text via `setFormat`, in
   place of the three separate labels the old central widget had.

`set_status`/`set_progress`/`set_busy` are the app's only entry points for
feedback; `is_busy()` is the one-job-at-a-time guard every generation
trigger checks (it used to be `app.cancel_btn.isEnabled()`).
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget, QHBoxLayout, QLabel, QMenu, QProgressBar, QPushButton, QStyle, QToolButton, QVBoxLayout,
    QWidget,
)

from kokoro_gui.engine.time_utils import format_duration


def format_clock(seconds: float) -> str:
    """mm:ss.t for the transport readout (a tenth is enough for a playhead
    readout; `format_duration` is for the ETA/elapsed line)."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:04.1f}"


class TransportDock(QDockWidget):
    playRequested = Signal()
    pauseRequested = Signal()
    stopRequested = Signal()
    loopToggled = Signal(bool)

    def __init__(self, app, parent=None):
        super().__init__("Transport", parent)
        self.setObjectName("dock_transport")
        self.app = app
        self._status_text = "Ready"
        self._detail_text = ""
        self._busy = False

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Row 1: transport
        row1 = QHBoxLayout()
        # Standard media icons rather than the U+23F5-family glyphs: Fusion
        # ships them on every platform, while the glyphs depend on the
        # installed fonts (and are blank under the offscreen platform).
        style = self.style()
        self.play_btn = QPushButton()
        self.play_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.setToolTip("Play (Space)")
        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.pause_btn.setToolTip("Pause (Space)")
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.setToolTip("Stop")
        for btn in (self.play_btn, self.pause_btn, self.stop_btn):
            btn.setFixedWidth(36)
            row1.addWidget(btn)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row1.addSpacing(8)
        row1.addWidget(self.time_label)
        row1.addStretch(1)
        self.loop_btn = QPushButton("Loop")
        self.loop_btn.setCheckable(True)
        row1.addWidget(self.loop_btn)
        layout.addLayout(row1)

        self.play_btn.clicked.connect(self.playRequested)
        self.pause_btn.clicked.connect(self.pauseRequested)
        self.stop_btn.clicked.connect(self.stopRequested)
        self.loop_btn.toggled.connect(self.loopToggled)

        # Row 2: generate
        row2 = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.app.preview_conversion)
        row2.addWidget(self.preview_btn)

        self.generate_btn = QToolButton()
        self.generate_btn.setText("Generate")
        self.generate_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.generate_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.generate_btn.clicked.connect(self.app.on_generate_clicked)
        self.generate_menu = QMenu(self.generate_btn)
        self.generate_dirty_action = QAction("Generate dirty clips", self)
        self.generate_dirty_action.triggered.connect(self.app.on_generate_clicked)
        self.auto_split_action = QAction("Auto-split then generate", self)
        self.auto_split_action.triggered.connect(self.app.auto_split_and_generate)
        self.split_paragraph_action = QAction("Split by paragraph", self)
        self.split_paragraph_action.setCheckable(True)
        self.split_paragraph_action.setChecked(bool(self.app.settings.get("auto_split_by_paragraph", False)))
        self.split_paragraph_action.toggled.connect(self._on_split_paragraph_toggled)
        self.generate_menu.addAction(self.generate_dirty_action)
        self.generate_menu.addAction(self.auto_split_action)
        self.generate_menu.addSeparator()
        self.generate_menu.addAction(self.split_paragraph_action)
        self.generate_btn.setMenu(self.generate_menu)
        row2.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.app.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        row2.addWidget(self.cancel_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        # Row 3: progress with the status inside it
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        self.setWidget(content)
        self._refresh_format()

    # -- status / progress -------------------------------------------------

    def _refresh_format(self) -> None:
        parts = [self._status_text]
        if self._detail_text:
            parts.append(self._detail_text)
        text = "  |  ".join(p for p in parts if p)
        if self._busy:
            text = f"%p%  {text}"
        self.progress_bar.setFormat(text)

    def status_text(self) -> str:
        return self._status_text

    def detail_text(self) -> str:
        return self._detail_text

    def set_status(self, message: str, kind: str = "info") -> None:
        """`kind` is "info" | "error" | "warning" | "success" | "busy" and
        only changes the text color."""
        self._status_text = (message or "").split("\n")[0]
        colors = {"error": "#ff5555", "warning": "orange", "success": "#2e8b57", "busy": "#1a73e8"}
        color = colors.get(kind)
        self.progress_bar.setStyleSheet(f"QProgressBar {{ color: {color}; }}" if color else "")
        self._refresh_format()

    def set_progress(self, percent: float, detail: str = "", elapsed: float | None = None,
                     eta: str | None = None) -> None:
        self.progress_bar.setValue(int(max(0, min(100, percent))))
        pieces = []
        if detail:
            pieces.append(detail)
        if elapsed is not None:
            eta_text = eta if eta else "--:--"
            pieces.append(f"{format_duration(elapsed)} / ETA {eta_text}")
        self._detail_text = "  ".join(pieces)
        self._refresh_format()

    def set_progress_value(self, percent: float) -> None:
        self.progress_bar.setValue(int(max(0, min(100, percent))))

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.generate_btn.setEnabled(not busy)
        self.preview_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        if not busy:
            self._detail_text = ""
        self._refresh_format()

    def is_busy(self) -> bool:
        return self._busy

    # -- transport readout -------------------------------------------------

    def set_position(self, position_s: float, total_s: float) -> None:
        self.time_label.setText(f"{format_clock(position_s)} / {format_clock(total_s)}")

    def set_playing(self, playing: bool) -> None:
        self.play_btn.setEnabled(not playing)
        self.pause_btn.setEnabled(playing)

    # -- generate menu -----------------------------------------------------

    def _on_split_paragraph_toggled(self, checked: bool) -> None:
        self.app.settings["auto_split_by_paragraph"] = checked
        self.app.schedule_save()
        if self.app.transcript_dock is not None:
            self.app.transcript_dock.editor.refresh_split_rules()
