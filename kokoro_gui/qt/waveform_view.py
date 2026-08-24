"""Single-track waveform view spike (Workstream 3 of
Claude/PLAN_daw_ui_ux_redesign.md) - flagged, in the prior Tk->Qt migration
plan (Claude/PLAN_qt_and_engine_abstraction.md's Phase 3b), as "the point of
first real technical risk in the whole plan... treat it as a standalone
spike/prototype before wiring it into the docked shell."

Nothing here is wired into kokoro_gui/qt/app.py or any dock - `WaveformPanel`
is a freestanding widget, verified only by its own tests (and
scripts/manual_waveform_spike.py for a visual check automated tests can't
give). The full multi-track timeline (clip/track/document integration,
drag/trim, undo, auto-track-assignment, per-clip FX buttons) is deliberately
a separate, later pass once this spike is signed off.
"""
from __future__ import annotations

import time

import playback
from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsScene, QGraphicsView,
    QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
)

from kokoro_gui.qt import playhead_calc, waveform_data

WAVEFORM_BRUSH_COLOR = "#4a90d9"
PLAYHEAD_PEN_COLOR = "#e5484d"
PLAYHEAD_TIMER_INTERVAL_MS = 33  # ~30fps - smooth-looking without over-firing;
# actual OS timer granularity is coarser than this on most platforms, an
# accepted spike-level limitation (see playhead_calc.py's own docstring on
# why this is a wall-clock approximation in the first place).


class WaveformItem(QGraphicsItem):
    """Paints a (min, max) peak envelope as a filled path. The path is built
    once per `set_peaks()` call and cached - `paint()` only ever strokes/
    fills that cached path, never rebuilds it, since `paint()` fires on
    every scene repaint (including ones triggered by an unrelated sibling
    item like the playhead moving) and rebuilding a path across potentially
    thousands of buckets on every such repaint would be a real perf bug.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._width = 0.0
        self._height = 0.0
        self._peaks = None
        self._path = QPainterPath()

    def set_peaks(self, peaks, width: float, height: float) -> None:
        # prepareGeometryChange() must happen *before* the stored width/
        # height change - otherwise Qt's dirty-tracking can paint against a
        # now-stale boundingRect(), a classic source of clipped/ghosted
        # repaints after a resize.
        self.prepareGeometryChange()
        self._peaks = peaks
        self._width = width
        self._height = height
        self._path = self._build_path(peaks, width, height)
        self.update()

    @staticmethod
    def _build_path(peaks, width: float, height: float) -> QPainterPath:
        path = QPainterPath()
        n = len(peaks) if peaks is not None else 0
        if n == 0 or width <= 0 or height <= 0:
            return path

        half_height = height / 2.0
        bucket_width = width / n
        for i, (lo, hi) in enumerate(peaks):
            x = i * bucket_width
            y_top = half_height - hi * half_height
            y_bottom = half_height - lo * half_height
            path.addRect(x, y_top, bucket_width, max(y_bottom - y_top, 0.0))
        return path

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        painter.fillPath(self._path, QBrush(QColor(WAVEFORM_BRUSH_COLOR)))


class WaveformView(QGraphicsView):
    """Owns a `QGraphicsScene` with one `WaveformItem`. `load_audio()` reads
    a WAV via `waveform_data.load_peaks_from_file` at one bucket per
    horizontal pixel and feeds the result to the item; resizing recomputes
    peaks at the new bucket count (recompute-on-resize, not a cached
    multi-resolution pyramid - see waveform_data.py's module docstring for
    why that's an accepted spike-scoped simplification)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.waveform_item = WaveformItem()
        self._scene.addItem(self.waveform_item)

        self._loaded_path: str | None = None
        self.duration: float = 0.0

    def load_audio(self, path: str) -> None:
        self._loaded_path = path
        self._reload_peaks()

    def _current_bucket_count(self) -> int:
        return max(1, self.viewport().width())

    def _reload_peaks(self) -> None:
        if not self._loaded_path:
            return
        bucket_count = self._current_bucket_count()
        peaks, duration = waveform_data.load_peaks_from_file(self._loaded_path, bucket_count)
        self.duration = duration

        width = max(1, self.viewport().width())
        height = max(1, self.viewport().height())
        self.waveform_item.set_peaks(peaks, width, height)
        self._scene.setSceneRect(0, 0, width, height)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._reload_peaks()


class WaveformPanel(QWidget):
    """The standalone top-level widget for this spike: a `WaveformView` plus
    Play/Stop buttons and a `QTimer`-driven playhead line. Not wired into
    the app - see module docstring."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._view = WaveformView()
        self._play_btn = QPushButton("Play")
        self._stop_btn = QPushButton("Stop")

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._playhead_item = QGraphicsLineItem()
        self._playhead_item.setPen(QPen(QColor(PLAYHEAD_PEN_COLOR), 2))
        self._playhead_item.hide()
        self._view._scene.addItem(self._playhead_item)

        self._timer = QTimer(self)
        self._timer.setInterval(PLAYHEAD_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_timer_tick)
        self._play_started: float | None = None

        self._play_btn.clicked.connect(self._on_play_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

    def load_audio(self, path: str) -> None:
        self._view.load_audio(path)
        self._playhead_item.hide()

    # -- playback control ----------------------------------------------------

    def _on_play_clicked(self) -> None:
        if not self._view._loaded_path:
            return

        # Clicking Play again while already playing restarts cleanly rather
        # than being blocked - matches playback.play()'s own "replaces the
        # currently playing buffer" behavior, so nothing here needs to guard
        # against a second call.
        playback.play(self._view._loaded_path, blocking=False)

        if not playback.AVAILABLE:
            # No PortAudio (e.g. headless/CI) - nothing plays, so there's
            # nothing to animate a playhead against. Must not crash.
            return

        self._play_started = time.monotonic()
        self._timer.start()

    def _on_timer_tick(self) -> None:
        if self._play_started is None:
            return

        elapsed = time.monotonic() - self._play_started
        x = playhead_calc.playhead_x(elapsed, self._view.duration, self._view.viewport().width())
        if x is None:
            self._playhead_item.hide()
        else:
            height = max(1, self._view.viewport().height())
            self._playhead_item.setLine(x, 0, x, height)
            self._playhead_item.show()

        if elapsed >= self._view.duration:
            self._timer.stop()

    def _on_stop_clicked(self) -> None:
        playback.stop()
        self._timer.stop()
        self._playhead_item.hide()
