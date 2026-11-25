"""Transport / Generate dock (bottom-right of the drawing's 2x2 grid, see
Claude/PLAN_ui_shell_redesign.md section 1).

What `QtTTSApp._build_action_bar` used to build as the central widget,
moved into a dock and reshaped into three rows:

1. play / pause / stop (round `QToolButton`s with `kokoro_gui.qt.icons`
   glyphs, retinted on `themeChanged`), elapsed / total time, loop toggle -
   driven by `kokoro_gui.audio.transport.Transport` through the app, and
   the Dub / Original / Both monitor toggle (phase 5 D5): what the
   transport plays when the project has a source track. Without one the
   toggle is disabled on Dub.
   After the loop button sits the playback speed combo (0.5x to 2x, pitch
   unchanged, plan 27): it emits `rateRequested` and the app sets the
   transport's rate; `set_rate_choice` shows a rate set from elsewhere.
   Last is Rec, which opens the Record guide take panel on the selected
   clip (`QtTTSApp.record_guide_take`, plan 31); the input meter is in
   that panel.
2. Preview (the line under the row says what it would speak and as whom,
   `qt/preview_target.py`), Generate (the row's one `primary` button: a `QToolButton`
   whose menu holds "Generate stale clips", "Generate stale clips in
   selection", "Auto-split then generate" and
   the checkable "Split by paragraph"), Cancel (flat), and Pause (flat; the
   generation queue's, Resume while paused).
3. One progress bar carrying the status/detail text via `setFormat`, in
   place of the three separate labels the old central widget had.

Between row 1 and row 2 sits a `LevelMeter`: two horizontal bars (left and
right, -60 to 0 dBFS) filled to the RMS in `meter_ok` / `meter_warn` /
`meter_clip` zones, with a peak tick that holds for 1.5 s. The app feeds it
from `Transport.levelsChanged`; it clears when playback stops.

`set_status`/`set_progress`/`set_busy` are the app's only entry points for
feedback, with `set_generate_enabled` for the Generate button; a control
that is off says why in its tooltip. `is_busy()` is the one-job-at-a-time guard every generation
trigger checks (it used to be `app.cancel_btn.isEnabled()`).
"""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDockWidget, QHBoxLayout, QLabel, QMenu, QProgressBar, QPushButton, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)

from kokoro_gui.audio.transport import RATE_STEPS
from kokoro_gui.daw.timecode import format_position
from kokoro_gui.engine.time_utils import format_duration
from kokoro_gui.i18n import N_, tr
from kokoro_gui.qt import icons, preview_target, theme
from kokoro_gui.qt.scope_target import muted_label

MONITOR_LABELS = {"dub": N_("Dub"), "original": N_("Original"), "both": N_("Both")}
MONITOR_TIPS = {
    "dub": N_("Play the dub."),
    "original": N_("Play the original dialogue under each clip, from the source track."),
    "both": N_("Play the dub and the original together, each 6 dB down."),
}
NO_SOURCE_TRACK_TIP = N_("Import a source track (File menu) to hear the original.")
PLAY_TIP = N_("Play the timeline's audio (Space)")
PAUSE_TIP = N_("Pause (Space)")
GENERATE_TIP = (N_("Make audio for every stale clip. Text with no clips is generated as one file instead. "
                   "The arrow has Generate stale clips in selection and Auto-split."))
GENERATE_BUSY_TIP = N_("Generate is off until the running job ends.")
PREVIEW_BUSY_TIP = N_("Preview is off until the running job ends.")
CANCEL_TIP = N_("Stop the running job and drop what is queued. Finished clips keep their audio.")
CANCEL_IDLE_TIP = N_("Nothing is running.")
PAUSE_QUEUE_TIP = N_("Stop after the batch that is running. The rest stays queued.")
PAUSE_QUEUE_IDLE_TIP = N_("Nothing is queued.")
NO_STALE_SELECTED_TIP = N_("The selection has no stale clip.")


METER_FLOOR_DB = -60.0
METER_WARN_DB = -6.0
METER_CLIP_DB = -1.0
PEAK_HOLD_S = 1.5


def level_db(linear: float) -> float:
    """`linear` in dBFS, floored at the meter's bottom."""
    return max(METER_FLOOR_DB, 20.0 * math.log10(linear)) if linear > 0.0 else METER_FLOOR_DB


class LevelMeter(QWidget):
    """Two thin horizontal bars, left over right, from `METER_FLOOR_DB` to
    0 dBFS. The fill is the block's RMS, split into three colored zones
    (`theme` tokens `meter_ok`, `meter_warn`, `meter_clip`); a one-pixel-wide
    tick marks the peak and stays `PEAK_HOLD_S` after the peak that set it.
    `set_levels` takes the four linear values `Transport.levelsChanged`
    emits; `clear` zeroes everything (stopped)."""

    def __init__(self, parent=None, clock=time.monotonic):
        super().__init__(parent)
        self._clock = clock
        self.setFixedHeight(18)
        self.setMinimumWidth(80)
        self.paint_count = 0
        self.peak_db = [METER_FLOOR_DB, METER_FLOOR_DB]
        self.rms_db = [METER_FLOOR_DB, METER_FLOOR_DB]
        self.hold_db = [METER_FLOOR_DB, METER_FLOOR_DB]
        self._hold_at = [0.0, 0.0]
        self.setToolTip(tr("Output level: bar is RMS, tick is the peak (held 1.5 s). Left over right, -60 to 0 dBFS."))

    def set_levels(self, peak_l: float, peak_r: float, rms_l: float, rms_r: float) -> None:
        now = self._clock()
        for ch, (peak, rms) in enumerate(((peak_l, rms_l), (peak_r, rms_r))):
            self.peak_db[ch] = level_db(peak)
            self.rms_db[ch] = level_db(rms)
            if self.peak_db[ch] >= self.hold_db[ch] or now - self._hold_at[ch] > PEAK_HOLD_S:
                self.hold_db[ch] = self.peak_db[ch]
                self._hold_at[ch] = now
        self.update()

    def clear(self) -> None:
        for values in (self.peak_db, self.rms_db, self.hold_db):
            values[:] = [METER_FLOOR_DB, METER_FLOOR_DB]
        self.update()

    @staticmethod
    def _x(db: float, width: float) -> float:
        return (db - METER_FLOOR_DB) / -METER_FLOOR_DB * width

    def paintEvent(self, _event) -> None:
        self.paint_count += 1
        pal = theme.current()
        painter = QPainter(self)
        width, bar_h = float(self.width()), (self.height() - 2) / 2.0
        zones = ((METER_FLOOR_DB, METER_WARN_DB, pal.meter_ok), (METER_WARN_DB, METER_CLIP_DB, pal.meter_warn),
                 (METER_CLIP_DB, 0.0, pal.meter_clip))
        for ch in range(2):
            y = ch * (bar_h + 2)
            painter.fillRect(QRectF(0, y, width, bar_h), QColor(pal.panel_alt))
            for lo, hi, color in zones:
                top = min(hi, self.rms_db[ch])
                if top > lo:
                    painter.fillRect(QRectF(self._x(lo, width), y, self._x(top, width) - self._x(lo, width), bar_h),
                                     QColor(color))
            if self.hold_db[ch] > METER_FLOOR_DB:
                hold = self.hold_db[ch]
                color = pal.meter_clip if hold >= METER_CLIP_DB else pal.meter_warn if hold >= METER_WARN_DB                     else pal.meter_ok
                painter.fillRect(QRectF(min(width - 1.0, self._x(hold, width)), y, 1.0, bar_h), QColor(color))
        painter.end()


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
    monitorChanged = Signal(str)  # "dub" | "original" | "both"
    rateRequested = Signal(float)  # the speed combo, playback only

    def __init__(self, app, parent=None):
        super().__init__(tr("Transport"), parent)
        self.setObjectName("dock_transport")
        self.app = app
        self._status_text = tr("Ready")
        self._detail_text = ""
        self._busy = False
        # When the last job began (time.monotonic()), kept after it ends so
        # the finish handlers can tell a long job from a short one.
        self.busy_since: float | None = None

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Row 1: transport
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        # Painted glyphs rather than Fusion's SP_Media* pixmaps or the
        # U+23F5-family characters: the pixmaps are the 2000s look this
        # dock is trying to shed, the characters depend on installed fonts
        # (and are blank under the offscreen platform).
        self.play_btn = QToolButton()
        self.play_btn.setToolTip(tr(PLAY_TIP))
        self.play_btn.setAccessibleName(tr("Play"))
        self.pause_btn = QToolButton()
        self.pause_btn.setToolTip(tr("{pause_tip}. Nothing is playing.", pause_tip=tr(PAUSE_TIP)))
        self.pause_btn.setAccessibleName(tr("Pause"))
        self.pause_btn.setEnabled(False)
        self.stop_btn = QToolButton()
        self.stop_btn.setToolTip(tr("Stop"))
        self.stop_btn.setAccessibleName(tr("Stop"))
        for btn in (self.play_btn, self.pause_btn, self.stop_btn):
            btn.setProperty("transport", True)
            btn.setAutoRaise(True)
            row1.addWidget(btn)
        self._apply_icons()
        self.app.themeChanged.connect(self._apply_icons)
        self.time_label = QLabel("00:00.0 / 00:00.0")
        self.time_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row1.addSpacing(8)
        row1.addWidget(self.time_label)
        row1.addStretch(1)
        self.monitor_group = QButtonGroup(self)
        self.monitor_group.setExclusive(True)
        self.monitor_buttons: dict = {}
        for mode, label in MONITOR_LABELS.items():
            btn = QToolButton()
            btn.setText(tr(label))
            btn.setCheckable(True)
            btn.setToolTip(tr(MONITOR_TIPS[mode]))
            btn.setObjectName(f"monitor_{mode}")
            self.monitor_group.addButton(btn)
            self.monitor_buttons[mode] = btn
            row1.addWidget(btn)
        self.monitor_buttons["dub"].setChecked(True)
        self.set_monitor("dub", available=False)
        self.monitor_group.buttonClicked.connect(self._on_monitor_clicked)
        row1.addSpacing(8)
        self.loop_btn = QPushButton(tr("Loop"))
        self.loop_btn.setToolTip(tr("Repeat the loop region. Shift+drag on the ruler to set one."))
        self.loop_btn.setCheckable(True)
        row1.addWidget(self.loop_btn)
        self.rate_combo = QComboBox()
        self.rate_combo.setObjectName("playback_rate")
        for step in RATE_STEPS:
            self.rate_combo.addItem(f"{step:g}x", step)
        self.rate_combo.setCurrentIndex(RATE_STEPS.index(1.0))
        self.rate_combo.setToolTip(tr("Playback speed: voices keep their pitch. Playback only, never in an export. "
                                      "[ and ] step it, L steps 1x, 1.5x, 2x."))
        self.rate_combo.activated.connect(lambda _index: self.rateRequested.emit(self.rate_choice()))
        row1.addWidget(self.rate_combo)
        self.record_btn = QToolButton()
        self.record_btn.setText(tr("Rec"))
        self.record_btn.setObjectName("record_take")
        self.record_btn.setToolTip(tr("Record a guide take on the selected clip (R on the timeline)"))
        self.record_btn.clicked.connect(lambda _checked=False: self.app.record_guide_take())
        row1.addWidget(self.record_btn)
        layout.addLayout(row1)

        self.level_meter = LevelMeter()
        layout.addWidget(self.level_meter)

        self.play_btn.clicked.connect(self.playRequested)
        self.pause_btn.clicked.connect(self.pauseRequested)
        self.stop_btn.clicked.connect(self.stopRequested)
        self.loop_btn.toggled.connect(self.loopToggled)

        # Row 2: generate
        row2 = QHBoxLayout()
        self.preview_btn = QPushButton(tr("Preview"))
        self.preview_btn.setToolTip(preview_target.tooltip())
        self.preview_btn.clicked.connect(self.app.preview_conversion)
        row2.addWidget(self.preview_btn)

        self.generate_btn = QToolButton()
        self.generate_btn.setText(tr("Generate"))
        self.generate_btn.setToolTip(tr(GENERATE_TIP))
        self.generate_btn.setProperty("primary", True)
        self.generate_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.generate_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.generate_btn.clicked.connect(self.app.on_generate_clicked)
        self.generate_menu = QMenu(self.generate_btn)
        self.generate_menu.setToolTipsVisible(True)
        self.generate_dirty_action = QAction(tr("Generate stale clips"), self)
        self.generate_dirty_action.setToolTip(tr("Make audio for every stale clip."))
        self.generate_dirty_action.triggered.connect(self.app.on_generate_clicked)
        self.generate_selection_action = QAction(tr("Generate stale clips in selection"), self)
        self.generate_selection_action.setToolTip(tr("Make audio for the stale clips you selected."))
        self.generate_selection_action.triggered.connect(self.app.generate_selection)
        self.auto_split_action = QAction(tr("Auto-split then generate"), self)
        self.auto_split_action.setToolTip(tr("Turn each [Speaker]: tag into a clip of that character, then "
                                             "generate the clips."))
        self.auto_split_action.triggered.connect(self.app.auto_split_and_generate)
        self.split_paragraph_action = QAction(tr("Split by paragraph"), self)
        self.split_paragraph_action.setToolTip(tr("Auto-split also starts a new clip at each paragraph."))
        self.split_paragraph_action.setCheckable(True)
        self.split_paragraph_action.setChecked(bool(self.app.settings.get("auto_split_by_paragraph", False)))
        self.split_paragraph_action.toggled.connect(self._on_split_paragraph_toggled)
        self.generate_menu.addAction(self.generate_dirty_action)
        self.generate_menu.addAction(self.generate_selection_action)
        self.generate_menu.addAction(self.auto_split_action)
        self.generate_menu.addSeparator()
        self.generate_menu.addAction(self.split_paragraph_action)
        self.generate_menu.aboutToShow.connect(self._refresh_generate_menu)
        self.generate_btn.setMenu(self.generate_menu)
        row2.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setFlat(True)
        self.cancel_btn.clicked.connect(self.app.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip(tr(CANCEL_IDLE_TIP))
        row2.addWidget(self.cancel_btn)
        # Pause stops the generation queue after the running item; it turns
        # into Resume while the queue is paused (`set_queue_state`).
        self.queue_pause_btn = QPushButton(tr("Pause"))
        self.queue_pause_btn.setFlat(True)
        self.queue_pause_btn.setToolTip(tr(PAUSE_QUEUE_IDLE_TIP))
        self.queue_pause_btn.clicked.connect(self.app.toggle_queue_pause)
        self.queue_pause_btn.setEnabled(False)
        row2.addWidget(self.queue_pause_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        # What Preview would speak now. One line, cut rather than wrapped.
        self.preview_label = muted_label(wrap=False)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.preview_label)

        # Row 3: progress with the status inside it
        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip(tr("How far the running job is."))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        self.setWidget(content)
        self._refresh_format()

    def _refresh_generate_menu(self) -> None:
        """"Generate stale clips in selection" is on only while the
        selection holds a stale clip."""
        stale = bool(self.app.stale_selected_clips())
        self.generate_selection_action.setEnabled(stale)
        tip = tr("Make audio for the stale clips you selected.") if stale else tr(NO_STALE_SELECTED_TIP)
        self.generate_selection_action.setToolTip(tip)
        self.generate_selection_action.setStatusTip(tip)

    def refresh_preview_target(self) -> None:
        """The line under Preview, after the caret, the text or the active
        character changed."""
        editor = self.app.editor
        if editor is None:
            return
        text = preview_target.describe(editor, self.app.active_character())
        self.preview_label.setText(text)
        self.preview_label.setToolTip(f"{text} {preview_target.tooltip()}")
        self.preview_btn.setAccessibleDescription(text)

    def _apply_icons(self) -> None:
        pal = theme.current()
        self.play_btn.setIcon(icons.icon("play", pal.accent))
        self.pause_btn.setIcon(icons.icon("pause", pal.text))
        self.stop_btn.setIcon(icons.icon("stop", pal.text))

    # -- status / progress -------------------------------------------------

    def _refresh_format(self) -> None:
        parts = [self._status_text]
        if self._detail_text:
            parts.append(self._detail_text)
        text = "  |  ".join(p for p in parts if p)
        # The bar cuts a long line; the tooltip has it whole.
        self.progress_bar.setToolTip(text)
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
            pieces.append(tr("{duration} / ETA {eta_text}", duration=format_duration(elapsed), eta_text=eta_text))
        self._detail_text = "  ".join(pieces)
        self._refresh_format()

    def set_progress_value(self, percent: float) -> None:
        self.progress_bar.setValue(int(max(0, min(100, percent))))

    def set_busy(self, busy: bool) -> None:
        if busy and not self._busy:
            self.busy_since = time.monotonic()
        self._busy = busy
        self.set_generate_enabled(not busy)
        self.preview_btn.setEnabled(not busy)
        self.preview_btn.setToolTip(tr(PREVIEW_BUSY_TIP) if busy else preview_target.tooltip())
        self.cancel_btn.setEnabled(busy)
        self.cancel_btn.setToolTip(tr(CANCEL_TIP) if busy else tr(CANCEL_IDLE_TIP))
        if not busy:
            self._detail_text = ""
        self._refresh_format()

    def set_generate_enabled(self, enabled: bool) -> None:
        self.generate_btn.setEnabled(enabled)
        self.generate_btn.setToolTip(tr(GENERATE_TIP) if enabled else tr(GENERATE_BUSY_TIP))

    def is_busy(self) -> bool:
        return self._busy

    def set_queue_state(self, pending: bool, paused: bool) -> None:
        """The Pause button follows the generation queue: on while it has
        work, and Resume while it is paused."""
        self.queue_pause_btn.setEnabled(pending)
        self.queue_pause_btn.setText(tr("Resume") if paused else tr("Pause"))
        tip = tr("Start the queued batches again.") if paused else tr(PAUSE_QUEUE_TIP)
        self.queue_pause_btn.setToolTip(tip if pending else tr(PAUSE_QUEUE_IDLE_TIP))

    # -- transport readout -------------------------------------------------

    def set_position(self, position_s: float, total_s: float) -> None:
        """mm:ss.t, or timecode when the project enables it."""
        settings = self.app.document.settings if getattr(self.app, "document", None) is not None else {}
        now, total = format_position(settings, position_s), format_position(settings, total_s)
        if now is None or total is None:
            now, total = format_clock(position_s), format_clock(total_s)
        self.time_label.setText(f"{now} / {total}")

    def set_playing(self, playing: bool) -> None:
        self.play_btn.setEnabled(not playing)
        self.play_btn.setToolTip(tr("{play_tip}. Already playing.", play_tip=tr(PLAY_TIP)) if playing else tr(PLAY_TIP))
        self.pause_btn.setEnabled(playing)
        self.pause_btn.setToolTip(tr(PAUSE_TIP) if playing else tr("{pause_tip}. Nothing is playing.",
                                                                   pause_tip=tr(PAUSE_TIP)))
        if not playing:
            self.level_meter.clear()

    def set_levels(self, peak_l: float, peak_r: float, rms_l: float, rms_r: float) -> None:
        self.level_meter.set_levels(peak_l, peak_r, rms_l, rms_r)

    # -- playback speed ----------------------------------------------------

    def rate_choice(self) -> float:
        return float(self.rate_combo.currentData())

    def set_rate_choice(self, rate: float) -> None:
        """Shows the step nearest `rate` without emitting `rateRequested`."""
        nearest = min(range(self.rate_combo.count()), key=lambda i: abs(self.rate_combo.itemData(i) - rate))
        self.rate_combo.setCurrentIndex(nearest)

    # -- monitor toggle ----------------------------------------------------

    def monitor(self) -> str:
        return next((m for m, b in self.monitor_buttons.items() if b.isChecked()), "dub")

    def set_monitor(self, mode: str, available: bool = True) -> None:
        """Shows `mode` without emitting `monitorChanged`. Unavailable (no
        source track) shows Dub, disabled."""
        mode = mode if available and mode in self.monitor_buttons else "dub"
        self.monitor_buttons[mode].setChecked(True)
        for key, btn in self.monitor_buttons.items():
            btn.setEnabled(available)
            btn.setToolTip(tr(MONITOR_TIPS[key]) if available else tr(NO_SOURCE_TRACK_TIP))

    def _on_monitor_clicked(self, _button) -> None:
        self.monitorChanged.emit(self.monitor())

    # -- generate menu -----------------------------------------------------

    def _on_split_paragraph_toggled(self, checked: bool) -> None:
        self.app.settings["auto_split_by_paragraph"] = checked
        self.app.schedule_save()
        if self.app.transcript_dock is not None:
            self.app.transcript_dock.editor.refresh_split_rules()
