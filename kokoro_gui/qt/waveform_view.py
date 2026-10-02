"""Waveform rendering primitives: `WaveformItem` (the peak-envelope path the
timeline draws inside every generated clip block) and `WaveformView`, the
single-file view the original Workstream 3 spike validated.

The spike's `WaveformPanel` (Play/Stop plus a wall-clock playhead) and
`playhead_calc.py` are gone - the real transport
(`kokoro_gui.audio.transport.Transport`) tracks position from the audio
callback's frame counter, and the timeline draws the playhead.
"""
from __future__ import annotations

from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QColor, QPainterPath
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView
from shiboken6 import isValid

from kokoro_gui.qt import waveform_data

WAVEFORM_BRUSH_COLOR = "#4a90d9"
# Narrower than this a block draws no waveform and never loads its peaks.
WAVEFORM_MIN_WIDTH_PX = 4.0


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
        self._color = WAVEFORM_BRUSH_COLOR
        # A `() -> peaks | None` that `set_source` stores and the first
        # paint (or `loaded_peaks`) calls, so a block nobody scrolls to never
        # decodes its audio.
        self._loader = None
        # Like `_loader`, for a source whose peaks may arrive after the
        # first paint: `request(deliver)` calls `deliver(peaks)` now or later.
        # `_ticket` names the current request, so an answer to an older one
        # (the source was replaced meanwhile) is dropped.
        self._request = None
        self._ticket = None
        self._painting = False

    def set_source(self, loader, width: float, height: float) -> None:
        """Peaks come from `loader()` when the item is first painted."""
        self._reset_source(width, height)
        self._loader = loader

    def set_deferred_source(self, request, width: float, height: float) -> None:
        """Peaks come from `request(deliver)` when the item is first painted.
        `request` may call `deliver(peaks)` straight away or from the GUI
        thread later; the item draws nothing until it does, then repaints."""
        self._reset_source(width, height)
        self._request = request

    def _reset_source(self, width: float, height: float) -> None:
        self.prepareGeometryChange()
        self._loader = None
        self._request = None
        self._ticket = None
        self._peaks = None
        self._width = width
        self._height = height
        self._path = QPainterPath()
        self.update()

    def loaded_peaks(self):
        """The peaks, loading them now if the item hasn't been painted. A
        deferred source that hasn't answered yet gives None."""
        if self._loader is not None:
            loader, self._loader = self._loader, None
            try:
                peaks = loader()
            except Exception:
                peaks = None
            self._peaks = peaks
            self._path = self._build_path(peaks, self._width, self._height)
        elif self._request is not None and self._ticket is None:
            self._ticket = ticket = object()
            try:
                self._request(lambda peaks: self._deliver(ticket, peaks))
            except Exception:
                self._deliver(ticket, None)
        return self._peaks

    def _deliver(self, ticket, peaks) -> None:
        if ticket is not self._ticket or not isValid(self):
            return
        self._peaks = peaks
        self._path = self._build_path(peaks, self._width, self._height)
        if not self._painting:
            self.update()

    def set_peaks(self, peaks, width: float, height: float) -> None:
        # prepareGeometryChange() must happen *before* the stored width/
        # height change - otherwise Qt's dirty-tracking can paint against a
        # now-stale boundingRect(), a classic source of clipped/ghosted
        # repaints after a resize.
        self.prepareGeometryChange()
        self._loader = None
        self._request = None
        self._ticket = None
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

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color
        self.update()

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        if self._width < WAVEFORM_MIN_WIDTH_PX:
            # Too narrow to read: don't decode the audio for it.
            return
        self._painting = True
        try:
            if self._loader is not None or (self._request is not None and self._ticket is None):
                self.loaded_peaks()
        finally:
            self._painting = False
        painter.fillPath(self._path, QBrush(QColor(self._color)))


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
