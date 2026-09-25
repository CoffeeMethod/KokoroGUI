"""Multi-track timeline on a real seconds axis (UI9 of
Claude/PLAN_ui_shell_redesign.md, section 4).

`TimelineView` owns the scene: a ruler across the top, one lane per
track that has clips (`Document.used_tracks`; an unused track isn't
drawn), one `ClipBlockItem` per placed clip, and a playhead. x is
`seconds * self._zoom` (Ctrl+wheel, 20-400 px/s). Where a clip sits comes
from `kokoro_gui.daw.arrangement.compute_arrangement` - the same placement
the transport plays and the exporter writes - never from text offsets.
Generated clips draw a waveform; estimated (ungenerated) clips draw a
dashed outline and no waveform.

`TrackHeaderView` is the fixed 120px column on the left holding track names
and a color swatch, a separate `QGraphicsView` whose vertical scrollbar
follows the main view's, so labels never overlap clips. `TimelineWidget`
composes the two.

Mouse gestures on the main view (all resolved in `mouseReleaseEvent`):

- click a block: select it (`SelectionModel`); click its FX chip: FX menu.
- click the ruler: `seekRequested(seconds)`.
- drag a block horizontally: `clipMoved(clip_id, new_start_s)` - the dock
  pins the timestamp, and reorders the text if the drop lands before the
  clip's text-order predecessor (grill Q13). Snaps to other clips' edges
  and to the playhead within `SNAP_PX`.
- drag a block onto another lane: `clipDragReassigned` (the Q9 prompt).
- Shift+drag inside one block: `subRangeTtsRequested(clip_id, start, end)`
  - item 9's sub-range TTS replacement, now behind Shift so a plain drag
  can mean "move".
- drag a block's top-left or top-right corner handle: `fadeChanged(clip_id,
  "fade_in_s" | "fade_out_s", seconds)`.
- on the ruler: drag a marker flag to move it (`markerMoved`), Shift+drag
  to set a loop region (`loopRangeRequested`), right-click for "Add marker
  here" or a flag's Rename / Delete / Loop to next marker. Double-click
  seeks, like a plain click.
- in a lane whose automation is shown (the header's `A` toggle): double-click
  adds a breakpoint, drag moves one (clamped between its neighbours in
  time), right-click deletes one, Alt+drag on a segment moves both its
  ends. Each edit is one `automationChanged(track_id, points)`. Clips in
  that lane don't move while it's shown.

The header column has, per track, `M` (mute), `S` (solo), `A` (show the
automation lane), a fader and a pan slider (`trackFieldChanged`).
The block context menu adds Take (pick or delete a parked take), Status,
"Align words" and "Lock in time" (`Clip.pinned`: ripple on regenerate
won't move it). Two clips overlapping on one track get a red border
(`arrangement.overlaps`). The ruler labels in timecode when the document has it
enabled (`kokoro_gui/daw/timecode.py`). `set_status_filter` dims blocks
that don't match the timeline dock's filter.

The widget stays app-independent (no `self.app`): the dock owning
`app.document`/`app.engine` handles every signal. `render_document()` is a
full teardown-and-rebuild, fine for the clip counts a script has.
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsProxyWidget, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout, QMenu, QMessageBox, QSlider, QToolButton, QWidget,
)

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.arrangement import Arrangement, compute_arrangement, overlaps
from kokoro_gui.daw.models import CLIP_STATUSES
from kokoro_gui.daw.timecode import format_position
from kokoro_gui.qt import theme, waveform_data
from kokoro_gui.qt.fx_presets import list_fx_preset_names
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.waveform_view import WaveformItem

RULER_HEIGHT_PX = 22.0
MARKER_HIT_PX = 6.0
FADE_HANDLE_PX = 8.0
AUTOMATION_POINT_RADIUS_PX = 4.0
AUTOMATION_HIT_PX = 7.0
AUTOMATION_MAX_GAIN = 2.0
PAN_DETENT = 0.05
STATUS_LABELS = {"todo": "To do", "generated": "Generated", "approved": "Approved",
                 "needs_rewrite": "Needs rewrite"}
# Which statuses each timeline filter shows at full opacity.
STATUS_FILTERS = {
    "all": set(CLIP_STATUSES),
    "not_approved": set(CLIP_STATUSES) - {"approved"},
    "needs_rewrite": {"needs_rewrite"},
}
FILTERED_OUT_OPACITY = 0.3
LANE_HEIGHT_PX = 80.0
LANE_MARGIN_PX = 8.0
MIN_CLIP_WIDTH_PX = 20.0
DEFAULT_PIXELS_PER_SECOND = 50.0
MIN_PIXELS_PER_SECOND = 20.0
MAX_PIXELS_PER_SECOND = 400.0
HEADER_WIDTH_PX = 150
CLIP_RADIUS_PX = 4.0
# A clip's fill is its character color over the lane at this alpha: the
# label stays readable in both themes and the waveform (the color's darker
# shade) reads as one object with the block instead of a blue overlay.
CLIP_FILL_ALPHA = 200
SNAP_PX = 8.0
MIN_SCENE_SECONDS = 10.0
AUTO_SCROLL_GRACE_S = 2.0
FALLBACK_CLIP_COLOR = "#888888"
SELECTED_BORDER_WIDTH_PX = 3
OVERLAP_BORDER_WIDTH_PX = 2
FX_BUTTON_WIDTH_PX = 24.0
FX_BUTTON_HEIGHT_PX = 16.0
FX_BUTTON_ACTIVE_OPACITY = 0.9
FX_BUTTON_INACTIVE_OPACITY = 0.5


def seconds_to_x(seconds: float, zoom: float) -> float:
    return seconds * zoom


def x_to_seconds(x: float, zoom: float) -> float:
    return max(0.0, x / zoom) if zoom > 0 else 0.0


def lane_top(index: int) -> float:
    return RULER_HEIGHT_PX + index * LANE_HEIGHT_PX


def choose_tick_step(zoom: float, min_label_px: float = 60.0) -> float:
    """The tick spacing (seconds) that keeps labels at least
    `min_label_px` apart at `zoom`: 1, 2, 5, 10, 15, 30, 60, ..."""
    candidates = (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
    for step in candidates:
        if step * zoom >= min_label_px:
            return float(step)
    return float(candidates[-1])


def format_ruler_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if seconds < 60 and rest != int(rest):
        return f"{rest:.1f}s"
    if minutes == 0:
        return f"{int(rest)}s"
    return f"{minutes}:{int(rest):02d}"


def label_color_for(fill: QColor) -> QColor:
    """Black or white, whichever reads on `fill` (perceived luminance)."""
    lum = 0.299 * fill.red() + 0.587 * fill.green() + 0.114 * fill.blue()
    return QColor("#111111") if lum > 150 else QColor("#ffffff")


class ClipBlockItem(QGraphicsItem):
    """One clip's block on a lane: character-colored fill, label, optional
    child `WaveformItem`, FX chip in the bottom-right corner. Estimated
    clips get a dashed outline and no waveform."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self._width = 0.0
        self._height = 0.0
        self._color = FALLBACK_CLIP_COLOR
        self._label = ""
        self._waveform_item: WaveformItem | None = None
        self._selected = False
        self._fx_active = False
        self._estimated = False
        self._overlap = False
        # A subproject's block (phase 4): None for an ordinary clip, else
        # "ok", "stale" or "missing".
        self._nested_state = None
        self._fade_in_px = 0.0
        self._fade_out_px = 0.0
        self.clip_id: Optional[str] = None
        self.audio_path: Optional[str] = None
        self.start_s = 0.0
        self.duration_s = 0.0

    def set_clip_id(self, clip_id: str) -> None:
        self.clip_id = clip_id

    def set_audio_path(self, audio_path: Optional[str]) -> None:
        self.audio_path = audio_path

    def set_geometry(self, x: float, y: float, width: float, height: float) -> None:
        self.prepareGeometryChange()
        self.setPos(x, y)
        self._width = width
        self._height = height
        self.update()

    def set_color(self, hex_color: str) -> None:
        self._color = hex_color
        self.update()

    def set_label(self, text: str) -> None:
        self._label = text
        self.update()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update()

    def set_fx_active(self, active: bool) -> None:
        self._fx_active = active
        self.update()

    def set_estimated(self, estimated: bool) -> None:
        self._estimated = estimated
        self.update()

    @property
    def estimated(self) -> bool:
        return self._estimated

    def set_nested_state(self, state) -> None:
        self._nested_state = state
        self.update()

    @property
    def nested_state(self):
        return self._nested_state

    def set_overlap(self, overlap: bool) -> None:
        self._overlap = overlap
        self.update()

    @property
    def overlap(self) -> bool:
        return self._overlap

    def set_fades_px(self, fade_in_px: float, fade_out_px: float) -> None:
        self._fade_in_px = max(0.0, min(fade_in_px, self._width))
        self._fade_out_px = max(0.0, min(fade_out_px, self._width))
        self.update()

    def fade_in_handle_rect(self) -> QRectF:
        x = min(self._fade_in_px, max(0.0, self._width - FADE_HANDLE_PX))
        return QRectF(x, 0.0, FADE_HANDLE_PX, FADE_HANDLE_PX)

    def fade_out_handle_rect(self) -> QRectF:
        x = max(0.0, self._width - self._fade_out_px - FADE_HANDLE_PX)
        return QRectF(x, 0.0, FADE_HANDLE_PX, FADE_HANDLE_PX)

    def fx_button_rect(self) -> QRectF:
        width = min(FX_BUTTON_WIDTH_PX, self._width)
        height = min(FX_BUTTON_HEIGHT_PX, self._height)
        x = max(0.0, self._width - FX_BUTTON_WIDTH_PX)
        y = max(0.0, self._height - FX_BUTTON_HEIGHT_PX)
        return QRectF(x, y, width, height)

    def set_waveform(self, peaks, width: float, height: float) -> None:
        if self._waveform_item is None:
            self._waveform_item = WaveformItem(parent=self)
        self._waveform_item.set_color(QColor(self._color).darker(170).name())
        self._waveform_item.set_peaks(peaks, width, height)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        # The item's own opacity (the status filter dims it) is already on
        # the painter; the FX chip multiplies into it.
        base_opacity = painter.opacity()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.5, 0.5, self._width - 1, self._height - 1)
        base = QColor(self._color)
        fill = QColor(base)
        fill.setAlpha(70 if self._estimated else CLIP_FILL_ALPHA)
        if self._selected:
            painter.setPen(QPen(QColor(pal.selection_border), SELECTED_BORDER_WIDTH_PX))
        elif self._overlap:
            painter.setPen(QPen(QColor(pal.overlap_border), OVERLAP_BORDER_WIDTH_PX))
        elif self._estimated:
            painter.setPen(QPen(QColor(pal.estimated_outline), 1, Qt.PenStyle.DashLine))
        else:
            painter.setPen(QPen(base.darker(135), 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, CLIP_RADIUS_PX, CLIP_RADIUS_PX)
        label_left = 6
        if self._nested_state is not None:
            # A subproject: a folder glyph before the title; a stale one is
            # hatched like a stale clip, a missing one crossed out.
            painter.save()
            if self._nested_state in ("stale", "missing"):
                pattern = Qt.BrushStyle.DiagCrossPattern if self._nested_state == "missing" \
                    else Qt.BrushStyle.BDiagPattern
                hatch = QColor(pal.dirty_underline if self._nested_state == "missing" else pal.estimated_outline)
                hatch.setAlpha(150)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(hatch, pattern))
                painter.drawRoundedRect(rect, CLIP_RADIUS_PX, CLIP_RADIUS_PX)
            painter.setPen(QPen(QColor(pal.text), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            folder = QRectF(6, 6, 12, 9)
            painter.drawRect(folder)
            painter.drawLine(QPointF(6, 6), QPointF(9, 3.5))
            painter.drawLine(QPointF(9, 3.5), QPointF(12, 3.5))
            painter.drawLine(QPointF(12, 3.5), QPointF(13, 6))
            painter.restore()
            label_left = 22
        if self._label:
            painter.setPen(label_color_for(base) if not self._estimated else QColor(pal.text))
            painter.drawText(rect.adjusted(label_left, 3, -4, -2), 0, self._label)

        if not self._estimated:
            # Fade ramps: a line from the bottom corner up to where the fade
            # ends on the top edge, and the corner handles.
            painter.setPen(QPen(label_color_for(base), 1))
            if self._fade_in_px > 0:
                painter.drawLine(QPointF(0.5, self._height - 0.5), QPointF(self._fade_in_px, 0.5))
            if self._fade_out_px > 0:
                painter.drawLine(QPointF(self._width - self._fade_out_px, 0.5),
                                 QPointF(self._width - 0.5, self._height - 0.5))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(pal.fx_badge_bg))
            painter.drawRect(self.fade_in_handle_rect())
            painter.drawRect(self.fade_out_handle_rect())

        painter.setOpacity(base_opacity * (FX_BUTTON_ACTIVE_OPACITY if self._fx_active
                                           else FX_BUTTON_INACTIVE_OPACITY))
        fx_rect = self.fx_button_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal.fx_badge_bg))
        painter.drawRoundedRect(fx_rect, 4, 4)
        painter.setPen(QPen(QColor(pal.fx_badge_text)))
        painter.drawText(fx_rect, Qt.AlignmentFlag.AlignCenter, "FX")
        painter.setOpacity(base_opacity)


class _RulerItem(QGraphicsItem):
    """Ticks and labels across the top of the scene (mm:ss, or timecode
    when the document enables it), the loop region, marker flags, and the
    playhead's triangle."""

    def __init__(self):
        super().__init__()
        self._width = 0.0
        self._zoom = DEFAULT_PIXELS_PER_SECOND
        self._playhead_x: Optional[float] = None
        self._markers: list = []
        self._settings: dict = {}
        self._loop_s: Optional[tuple] = None
        self.setZValue(5)

    def set_span(self, width: float, zoom: float) -> None:
        self.prepareGeometryChange()
        self._width = width
        self._zoom = zoom
        self.update()

    def set_playhead_x(self, x: Optional[float]) -> None:
        self._playhead_x = x
        self.update()

    def set_markers(self, markers: list) -> None:
        self._markers = list(markers)
        self.update()

    def set_document_settings(self, settings: dict) -> None:
        self._settings = dict(settings or {})
        self.update()

    def set_loop_s(self, loop_s: Optional[tuple]) -> None:
        self._loop_s = loop_s
        self.update()

    def marker_at_x(self, x: float) -> Optional[dict]:
        """The marker whose flag is within `MARKER_HIT_PX` of scene `x`."""
        best, best_d = None, MARKER_HIT_PX
        for marker in self._markers:
            d = abs(seconds_to_x(marker["seconds"], self._zoom) - x)
            if d <= best_d:
                best, best_d = marker, d
        return best

    def label_for(self, seconds: float) -> str:
        return format_position(self._settings, seconds) or format_ruler_label(seconds)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, RULER_HEIGHT_PX)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        painter.fillRect(self.boundingRect(), QColor(pal.ruler_bg))
        if self._loop_s is not None:
            loop = QColor(pal.playhead)
            loop.setAlpha(60)
            x0, x1 = (seconds_to_x(t, self._zoom) for t in self._loop_s)
            painter.fillRect(QRectF(x0, 0, max(1.0, x1 - x0), RULER_HEIGHT_PX), loop)
        painter.setPen(QPen(QColor(pal.ruler_text)))
        timecode = format_position(self._settings, 0.0) is not None
        step = choose_tick_step(self._zoom, min_label_px=90.0 if timecode else 60.0)
        total_s = self._width / self._zoom if self._zoom > 0 else 0.0
        t = 0.0
        while t <= total_s + 1e-6:
            x = seconds_to_x(t, self._zoom)
            painter.drawLine(QPointF(x, RULER_HEIGHT_PX - 6), QPointF(x, RULER_HEIGHT_PX))
            painter.drawText(QRectF(x + 2, 0, step * self._zoom - 4, RULER_HEIGHT_PX - 4),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.label_for(t))
            minor = t + step / 2
            if minor <= total_s:
                mx = seconds_to_x(minor, self._zoom)
                painter.drawLine(QPointF(mx, RULER_HEIGHT_PX - 3), QPointF(mx, RULER_HEIGHT_PX))
            t += step
        painter.setPen(QPen(QColor(pal.lane_border)))
        painter.drawLine(QPointF(0, RULER_HEIGHT_PX - 0.5), QPointF(self._width, RULER_HEIGHT_PX - 0.5))
        for marker in self._markers:
            x = seconds_to_x(marker["seconds"], self._zoom)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(pal.selection_border))
            painter.drawPolygon(QPolygonF([QPointF(x, 2), QPointF(x + 8, 6), QPointF(x, 10)]))
            painter.setPen(QPen(QColor(pal.selection_border)))
            painter.drawLine(QPointF(x, 2), QPointF(x, RULER_HEIGHT_PX))
            if marker.get("name"):
                painter.drawText(QRectF(x + 10, 0, 120, 12),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop), marker["name"])
        if self._playhead_x is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(pal.playhead))
            x = self._playhead_x
            painter.drawPolygon(QPolygonF([
                QPointF(x - 6, RULER_HEIGHT_PX - 12), QPointF(x + 6, RULER_HEIGHT_PX - 12), QPointF(x, RULER_HEIGHT_PX - 1),
            ]))


class AutomationLaneItem(QGraphicsItem):
    """A track's volume automation drawn over its lane: gain 0 at the lane's
    bottom margin, `AUTOMATION_MAX_GAIN` at its top, 1.0 halfway. No points
    draws a dashed line at unity. Geometry lives here; the view owns the
    mouse."""

    def __init__(self, track_id: str, top: float, width: float, zoom: float, points: list):
        super().__init__()
        self.track_id = track_id
        self._top = top
        self._width = width
        self._zoom = zoom
        self.points = sorted([float(p[0]), float(p[1])] for p in points or [])
        self.setZValue(4)

    def _band(self) -> tuple:
        return self._top + LANE_MARGIN_PX, self._top + LANE_HEIGHT_PX - LANE_MARGIN_PX

    def gain_to_y(self, gain: float) -> float:
        lo, hi = self._band()
        return hi - (max(0.0, min(AUTOMATION_MAX_GAIN, gain)) / AUTOMATION_MAX_GAIN) * (hi - lo)

    def y_to_gain(self, y: float) -> float:
        lo, hi = self._band()
        return round(max(0.0, min(AUTOMATION_MAX_GAIN, (hi - y) / (hi - lo) * AUTOMATION_MAX_GAIN)), 3)

    def point_pos(self, point) -> QPointF:
        return QPointF(seconds_to_x(point[0], self._zoom), self.gain_to_y(point[1]))

    def point_index_at(self, scene_pos: QPointF) -> Optional[int]:
        for index, point in enumerate(self.points):
            p = self.point_pos(point)
            if abs(p.x() - scene_pos.x()) <= AUTOMATION_HIT_PX and abs(p.y() - scene_pos.y()) <= AUTOMATION_HIT_PX:
                return index
        return None

    def segment_index_at(self, scene_pos: QPointF) -> Optional[int]:
        """The index of the left point of the segment under `scene_pos`."""
        seconds = x_to_seconds(scene_pos.x(), self._zoom)
        for index in range(len(self.points) - 1):
            if self.points[index][0] <= seconds <= self.points[index + 1][0]:
                return index
        return None

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, self._top, self._width, LANE_HEIGHT_PX)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(pal.playhead)
        if not self.points:
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            y = self.gain_to_y(1.0)
            painter.drawLine(QPointF(0, y), QPointF(self._width, y))
            return
        painter.setPen(QPen(color, 2))
        first, last = self.point_pos(self.points[0]), self.point_pos(self.points[-1])
        path = QPainterPath(QPointF(0, first.y()))
        for point in self.points:
            path.lineTo(self.point_pos(point))
        path.lineTo(QPointF(self._width, last.y()))
        painter.drawPath(path)
        painter.setBrush(color)
        for point in self.points:
            painter.drawEllipse(self.point_pos(point), AUTOMATION_POINT_RADIUS_PX, AUTOMATION_POINT_RADIUS_PX)


class _TrackLabelItem(QGraphicsSimpleTextItem):
    """A track's name in the header column, click-selectable as that
    lane's character. `shape()` returns the full rect so hit-testing near
    the glyph edges doesn't miss."""

    def __init__(self, text: str, character_id: Optional[str]):
        super().__init__(text)
        self.character_id = character_id

    def shape(self) -> QPainterPath:  # noqa: N802 (Qt override)
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path


def gain_to_slider(gain: float) -> int:
    return int(round(max(0.0, min(AUTOMATION_MAX_GAIN, float(gain))) * 100))


def pan_to_slider(pan: float) -> int:
    return int(round(max(-1.0, min(1.0, float(pan))) * 100))


def slider_to_pan(value: int) -> float:
    """Slider units to pan, snapping to centre within `PAN_DETENT`."""
    pan = value / 100.0
    return 0.0 if abs(pan) <= PAN_DETENT else pan


class TrackHeaderView(QGraphicsView):
    """The fixed-width track header column: per track a color swatch, the
    name, `M` / `S` / `A` toggles, a fader (0-200%) and a pan slider
    (centre detent). The toggles and sliders are plain widgets on
    `QGraphicsProxyWidget`s; edits go out as `trackFieldChanged(track_id,
    field, value)` and `A` as `automationToggled(track_id, shown)`."""

    trackFieldChanged = Signal(str, str, object)
    automationToggled = Signal(str, bool)

    def __init__(self, parent=None, selection_model: Optional[SelectionModel] = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._selection_model = selection_model
        self._labels: list = []  # keep-alive for Python-subclassed items
        self.controls: dict = {}  # track_id -> {"mute", "solo", "auto", "gain", "pan"} widgets
        self.setFixedWidth(HEADER_WIDTH_PX)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def _add_widget(self, widget, x: float, y: float):
        proxy = QGraphicsProxyWidget()
        proxy.setWidget(widget)
        proxy.setPos(x, y)
        self._scene.addItem(proxy)
        return widget

    def _toggle(self, text: str, tip: str, checked: bool) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setFixedSize(22, 18)
        return button

    def _slider(self, lo: int, hi: int, value: int, tip: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.setToolTip(tip)
        slider.setFixedSize(HEADER_WIDTH_PX - 30, 14)
        return slider

    def render_tracks(self, tracks: list, document, automation_shown=frozenset()) -> None:
        pal = theme.current()
        self._scene.clear()
        self._labels = []
        self.controls = {}
        total_height = RULER_HEIGHT_PX + max(len(tracks), 1) * LANE_HEIGHT_PX
        corner = QGraphicsRectItem(0, 0, HEADER_WIDTH_PX, RULER_HEIGHT_PX)
        corner.setBrush(QColor(pal.ruler_bg))
        corner.setPen(QPen(QColor(pal.lane_border)))
        self._scene.addItem(corner)
        for i, track in enumerate(tracks):
            y = lane_top(i)
            bg = QGraphicsRectItem(0, y, HEADER_WIDTH_PX, LANE_HEIGHT_PX)
            bg.setBrush(QColor(pal.panel_alt))
            bg.setPen(QPen(QColor(pal.lane_border)))
            bg.setZValue(-1)
            self._scene.addItem(bg)
            character = document.get_character(track.character_id) if document is not None else None
            swatch = QGraphicsRectItem(6, y + 8, 10, 10)
            swatch.setBrush(QColor(character.highlight_color if character else FALLBACK_CLIP_COLOR))
            swatch.setPen(Qt.PenStyle.NoPen)
            self._scene.addItem(swatch)
            label = _TrackLabelItem(track.name, track.character_id)
            label.setBrush(QColor(pal.text))
            label.setPos(22, y + 5)
            self._scene.addItem(label)
            self._labels.append(label)

            mute = self._add_widget(self._toggle("M", "Mute", bool(track.mute)), 6, y + 24)
            solo = self._add_widget(self._toggle("S", "Solo", bool(track.solo)), 30, y + 24)
            auto = self._add_widget(self._toggle("A", "Show volume automation", track.id in automation_shown),
                                    54, y + 24)
            gain = self._add_widget(self._slider(0, 200, gain_to_slider(track.gain), "Fader"), 6, y + 46)
            pan = self._add_widget(self._slider(-100, 100, pan_to_slider(track.pan), "Pan"), 6, y + 62)
            tid = track.id
            mute.toggled.connect(lambda on, t=tid: self.trackFieldChanged.emit(t, "mute", bool(on)))
            solo.toggled.connect(lambda on, t=tid: self.trackFieldChanged.emit(t, "solo", bool(on)))
            auto.toggled.connect(lambda on, t=tid: self.automationToggled.emit(t, bool(on)))
            # sliderReleased, not valueChanged: one undo step per drag.
            gain.sliderReleased.connect(
                lambda g=gain, t=tid: self.trackFieldChanged.emit(t, "gain", g.value() / 100.0))
            pan.sliderReleased.connect(lambda p=pan, t=tid: self._emit_pan(t, p))
            self.controls[tid] = {"mute": mute, "solo": solo, "auto": auto, "gain": gain, "pan": pan}
        self._scene.setSceneRect(0, 0, HEADER_WIDTH_PX, total_height)
        self.setBackgroundBrush(QColor(pal.panel))

    def _emit_pan(self, track_id: str, slider: QSlider) -> None:
        pan = slider_to_pan(slider.value())
        if pan == 0.0 and slider.value() != 0:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self.trackFieldChanged.emit(track_id, "pan", pan)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton or self._selection_model is None:
            return
        item = self.itemAt(event.position().toPoint())
        character_id = getattr(item, "character_id", None) if item is not None else None
        if character_id is not None:
            self._selection_model.select_character(character_id)


class TimelineView(QGraphicsView):
    generateClipRequested = Signal(str)
    playClipRequested = Signal(str)  # context-menu Play: seek the transport to the clip and play
    fxPresetRequested = Signal(str, str)  # (clip_id, preset_name); "" clears
    clipDragReassigned = Signal(str, str, bool)  # (clip_id, target_track_id, reassign_character)
    subRangeTtsRequested = Signal(str, int, int)  # (clip_id, sub_start, sub_end) text offsets
    clipMoved = Signal(str, float)  # (clip_id, new_start_s)
    unpinRequested = Signal(str)
    lockInTimeRequested = Signal(str, bool)  # (clip_id, pinned)
    # Phase 4: a subproject block's menu and double-click. The action is
    # "enter", "render", "relink", "detach", "embed" or "remove".
    subprojectActionRequested = Signal(str, str)  # (clip_id, action)
    seekRequested = Signal(float)
    zoomChanged = Signal(float)
    fadeChanged = Signal(str, str, float)  # (clip_id, "fade_in_s" | "fade_out_s", seconds)
    takeSelected = Signal(str, int)
    takeDeleteRequested = Signal(str, int)
    statusChangeRequested = Signal(str, str)
    alignWordsRequested = Signal(str)
    markerAddRequested = Signal(float)
    markerMoved = Signal(str, float)
    markerRenameRequested = Signal(str)
    markerDeleteRequested = Signal(str)
    loopRangeRequested = Signal(float, float)
    loopClearRequested = Signal()
    automationChanged = Signal(str, object)  # (track_id, [[seconds, gain], ...])

    def __init__(self, parent=None, selection_model: Optional[SelectionModel] = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self._selection_model = selection_model
        self._blocks_by_clip_id: dict = {}
        self._selected_block: Optional[ClipBlockItem] = None
        self._document = None
        self._arrangement: Optional[Arrangement] = None
        # Optional `(clip) -> (samples, rate) | None` the owner passes so the
        # waveform shows the post-processed audio the transport plays
        # (`QtTTSApp.rendered_clip_samples`); None draws the raw file.
        self._clip_samples = None
        self._zoom = DEFAULT_PIXELS_PER_SECOND
        self._playhead_s: Optional[float] = None
        self._playhead_item: Optional[QGraphicsLineItem] = None
        self._ruler: Optional[_RulerItem] = None
        self._lane_count = 0
        self._last_user_scroll = 0.0
        self._programmatic_scroll = False

        self._drag_clip_id: Optional[str] = None
        self._drag_start_pos = None
        self._drag_threshold_px = 8
        self._drag_clip_x_range: Optional[tuple] = None
        self._drag_shift = False
        # What a press started, resolved on release: None (a clip move or
        # sub-range drag, the default), "fade_in"/"fade_out", "marker",
        # "loop", "automation" or "automation_segment".
        self._drag_mode: Optional[str] = None
        self._drag_payload = None

        self._automation_shown: set = set()
        self._nested_state_fn = None
        self._automation_items: dict = {}
        self._status_filter = "all"
        self._loop_s: Optional[tuple] = None

        self.header: Optional[TrackHeaderView] = None
        self.horizontalScrollBar().valueChanged.connect(self._on_hscroll)
        if selection_model is not None:
            selection_model.changed.connect(self._on_selection_changed)

    # -- zoom ----------------------------------------------------------------

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, pixels_per_second: float) -> None:
        new_zoom = max(MIN_PIXELS_PER_SECOND, min(MAX_PIXELS_PER_SECOND, float(pixels_per_second)))
        if new_zoom == self._zoom:
            return
        self._zoom = new_zoom
        if self._document is not None:
            self.render_document(self._document, self._arrangement)
        self.zoomChanged.emit(self._zoom)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else (1 / 1.15)
            self.set_zoom(self._zoom * factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _on_hscroll(self, _value: int) -> None:
        if not self._programmatic_scroll:
            self._last_user_scroll = time.monotonic()

    # -- context menu ----------------------------------------------------------

    def _clip_block_at(self, pos) -> Optional[ClipBlockItem]:
        """The clip block under `pos`, looking through the playhead/grid
        lines drawn above it and up from a child WaveformItem."""
        for item in self.items(pos):
            while item is not None and not isinstance(item, ClipBlockItem):
                item = item.parentItem()
            if item is not None:
                return item
        return None

    def _build_context_menu(self, pos) -> Optional[QMenu]:
        block = self._clip_block_at(pos)
        if block is None or block.clip_id is None:
            return None

        menu = QMenu(self)
        clip = self._document.get_clip(block.clip_id) if self._document is not None else None
        if clip is not None and clip.is_nested:
            return self._build_subproject_menu(menu, clip)
        generate_action = menu.addAction("Generate")
        generate_action.triggered.connect(
            lambda checked=False, cid=block.clip_id: self.generateClipRequested.emit(cid)
        )

        if block.audio_path:
            play_action = menu.addAction("Play")
            play_action.triggered.connect(
                lambda checked=False, cid=block.clip_id: self.playClipRequested.emit(cid)
            )

        clip = self._document.get_clip(block.clip_id) if self._document is not None else None
        if clip is not None and clip.timeline_timestamp is not None:
            menu.addSeparator()
            unpin = menu.addAction("Unpin from timeline")
            unpin.triggered.connect(lambda checked=False, cid=block.clip_id: self.unpinRequested.emit(cid))

        if clip is not None:
            lock = menu.addAction("Lock in time")
            lock.setCheckable(True)
            lock.setChecked(bool(clip.pinned))
            lock.setToolTip("Ripple on regenerate won't move this clip.")
            lock.triggered.connect(lambda checked=False, cid=clip.id: self.lockInTimeRequested.emit(cid, bool(checked)))
            menu.addSeparator()
            self._add_take_menu(menu, clip)
            status_menu = menu.addMenu("Status")
            for status in CLIP_STATUSES:
                action = status_menu.addAction(STATUS_LABELS[status])
                action.setCheckable(True)
                action.setChecked(clip.status == status)
                action.triggered.connect(
                    lambda checked=False, cid=clip.id, st=status: self.statusChangeRequested.emit(cid, st))
            if any(s.audio_path for s in clip.segments):
                align = menu.addAction("Align words")
                align.triggered.connect(lambda checked=False, cid=clip.id: self.alignWordsRequested.emit(cid))

        return menu

    def _build_subproject_menu(self, menu: QMenu, clip) -> QMenu:
        """A nested block: Enter, Generate and render, Detach to file... or
        Embed, Relink..., Remove."""
        kind = (clip.child or {}).get("kind", "embedded")
        items = [("Enter", "enter"), ("Generate and render", "render")]
        items.append(("Detach to file...", "detach") if kind == "embedded" else ("Embed", "embed"))
        items.append(("Relink...", "relink"))
        items.append(("Remove", "remove"))
        for label, action_name in items:
            action = menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, cid=clip.id, a=action_name: self.subprojectActionRequested.emit(cid, a))
        return menu

    def _add_take_menu(self, menu: QMenu, clip) -> None:
        """Take > one entry per take (the active one ticked, each with its
        length), then Delete take > the parked ones. A parked take whose
        text differs from the clip's text now is marked "old text", its
        text in the tooltip: picking it makes the clip stale at once."""
        active = int(clip.overrides.get("take", 0) or 0)
        takes = {active: clip.segments, **clip.takes}
        if len(takes) < 2:
            return
        current_text = " ".join(self._document.clip_text(clip).split())
        take_menu = menu.addMenu("Take")
        take_menu.setToolTipsVisible(True)
        for index in sorted(takes):
            segments = takes[index]
            seconds = sum(float(s.duration or 0.0) for s in segments)
            text = " ".join(" ".join(s.text for s in segments).split())
            label = f"Take {index + 1} ({seconds:.1f}s)"
            if text != current_text:
                label += " - old text"
            action = take_menu.addAction(label)
            action.setToolTip(text)
            action.setCheckable(True)
            action.setChecked(index == active)
            if index != active:
                action.triggered.connect(
                    lambda checked=False, cid=clip.id, i=index: self.takeSelected.emit(cid, i))
        delete_menu = menu.addMenu("Delete take")
        for index in sorted(clip.takes):
            action = delete_menu.addAction(f"Take {index + 1}")
            action.triggered.connect(
                lambda checked=False, cid=clip.id, i=index: self.takeDeleteRequested.emit(cid, i))

    def _build_ruler_menu(self, scene_x: float) -> QMenu:
        menu = QMenu(self)
        marker = self._ruler.marker_at_x(scene_x) if self._ruler is not None else None
        if marker is not None:
            mid = marker["id"]
            menu.addAction("Rename marker...").triggered.connect(
                lambda checked=False: self.markerRenameRequested.emit(mid))
            menu.addAction("Delete marker").triggered.connect(
                lambda checked=False: self.markerDeleteRequested.emit(mid))
            later = [m for m in self._markers() if m["seconds"] > marker["seconds"]]
            if later:
                end_s = later[0]["seconds"]
                menu.addAction("Loop to next marker").triggered.connect(
                    lambda checked=False: self.loopRangeRequested.emit(marker["seconds"], end_s))
        else:
            seconds = x_to_seconds(scene_x, self._zoom)
            menu.addAction("Add marker here").triggered.connect(
                lambda checked=False: self.markerAddRequested.emit(seconds))
        if self._loop_s is not None:
            menu.addSeparator()
            menu.addAction("Clear loop").triggered.connect(lambda checked=False: self.loopClearRequested.emit())
        return menu

    def _markers(self) -> list:
        return marker_ops.list_markers(self._document.settings if self._document is not None else {})

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        scene_pos = self.mapToScene(event.pos())
        if scene_pos.y() < RULER_HEIGHT_PX:
            self._build_ruler_menu(scene_pos.x()).exec(event.globalPos())
            return
        if self._delete_automation_point_at(scene_pos):
            return
        menu = self._build_context_menu(event.pos())
        if menu is not None:
            menu.exec(event.globalPos())

    # -- automation lanes ------------------------------------------------------

    def set_automation_visible(self, track_id: str, shown: bool) -> None:
        if shown:
            self._automation_shown.add(track_id)
        else:
            self._automation_shown.discard(track_id)
        if self._document is not None:
            self.render_document(self._document, self._arrangement)

    def automation_item(self, track_id: str) -> Optional[AutomationLaneItem]:
        return self._automation_items.get(track_id)

    def _automation_lane_at(self, scene_pos: QPointF) -> Optional[AutomationLaneItem]:
        if self._document is None or scene_pos.y() < RULER_HEIGHT_PX:
            return None
        track = self._track_at_y(self._document, scene_pos.y())
        return self._automation_items.get(track.id) if track is not None else None

    def _delete_automation_point_at(self, scene_pos: QPointF) -> bool:
        lane = self._automation_lane_at(scene_pos)
        if lane is None:
            return False
        index = lane.point_index_at(scene_pos)
        if index is None:
            return False
        points = [list(p) for p in lane.points]
        del points[index]
        self.automationChanged.emit(lane.track_id, points)
        return True

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        block = self._clip_block_at(event.position().toPoint())
        if block is not None and block.nested_state is not None and block.clip_id is not None:
            # A subproject's block: enter it (NP6).
            self.subprojectActionRequested.emit(block.clip_id, "enter")
            event.accept()
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton:
            if scene_pos.y() < RULER_HEIGHT_PX:
                self.seekRequested.emit(x_to_seconds(scene_pos.x(), self._zoom))
                return
            lane = self._automation_lane_at(scene_pos)
            if lane is not None:
                seconds = round(x_to_seconds(scene_pos.x(), self._zoom), 3)
                points = sorted([list(p) for p in lane.points] + [[seconds, lane.y_to_gain(scene_pos.y())]])
                self.automationChanged.emit(lane.track_id, points)
                return
        super().mouseDoubleClickEvent(event)

    def set_status_filter(self, name: str) -> None:
        self._status_filter = name if name in STATUS_FILTERS else "all"
        self._apply_status_filter()

    def _apply_status_filter(self) -> None:
        shown = STATUS_FILTERS[self._status_filter]
        for clip_id, block in self._blocks_by_clip_id.items():
            clip = self._document.get_clip(clip_id) if self._document is not None else None
            dim = clip is not None and clip.status not in shown
            block.setOpacity(FILTERED_OUT_OPACITY if dim else 1.0)

    def set_loop_s(self, loop_s: Optional[tuple]) -> None:
        self._loop_s = loop_s
        if self._ruler is not None:
            self._ruler.set_loop_s(loop_s)

    # -- FX menu -----------------------------------------------------------------

    def _build_fx_menu(self, block: ClipBlockItem) -> QMenu:
        menu = QMenu(self)
        for name in list_fx_preset_names():
            action = menu.addAction(name)
            action.triggered.connect(
                lambda checked=False, cid=block.clip_id, n=name: self.fxPresetRequested.emit(cid, n)
            )
        menu.addSeparator()
        clear_action = menu.addAction("Clear FX")
        clear_action.triggered.connect(
            lambda checked=False, cid=block.clip_id: self.fxPresetRequested.emit(cid, "")
        )
        return menu

    def _handle_fx_button_click(self, block: ClipBlockItem, global_pos) -> None:
        self._build_fx_menu(block).exec(global_pos)

    # -- mouse -------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        scene_pos = self.mapToScene(pos)
        self._drag_mode = None
        self._drag_payload = None
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if scene_pos.y() < RULER_HEIGHT_PX:
            self._drag_clip_id = None
            self._drag_start_pos = pos
            marker = self._ruler.marker_at_x(scene_pos.x()) if self._ruler is not None else None
            if marker is not None:
                self._drag_mode, self._drag_payload = "marker", marker["id"]
            elif shift:
                self._drag_mode = "loop"
            else:
                self._drag_start_pos = None
                self.seekRequested.emit(x_to_seconds(scene_pos.x(), self._zoom))
            return

        lane = self._automation_lane_at(scene_pos)
        if lane is not None:
            self._drag_clip_id = None
            self._drag_start_pos = pos
            index = lane.point_index_at(scene_pos)
            if index is not None:
                self._drag_mode, self._drag_payload = "automation", (lane.track_id, index)
            elif event.modifiers() & Qt.KeyboardModifier.AltModifier:
                segment = lane.segment_index_at(scene_pos)
                if segment is not None:
                    self._drag_mode, self._drag_payload = "automation_segment", (lane.track_id, segment)
            return

        block = self._clip_block_at(pos)
        if block is not None and block.clip_id is not None:
            local_pos = block.mapFromScene(scene_pos)
            if block.fx_button_rect().contains(local_pos):
                self._handle_fx_button_click(block, event.globalPosition().toPoint())
                return
            if not block.estimated:
                if block.fade_in_handle_rect().contains(local_pos):
                    self._drag_mode, self._drag_payload = "fade_in", block
                elif block.fade_out_handle_rect().contains(local_pos):
                    self._drag_mode, self._drag_payload = "fade_out", block

        self._drag_start_pos = pos
        self._drag_clip_id = block.clip_id if block is not None else None
        self._drag_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if block is not None and block.clip_id is not None:
            left = block.pos().x()
            self._drag_clip_x_range = (left, left + block.boundingRect().width())
        else:
            self._drag_clip_x_range = None

        if self._selection_model is None:
            return
        self._select_clip_block_at(pos)

    def _track_at_y(self, document, y: float):
        tracks = document.used_tracks()
        index = int((y - RULER_HEIGHT_PX) // LANE_HEIGHT_PX)
        if y < RULER_HEIGHT_PX:
            return None
        if 0 <= index < len(tracks):
            return tracks[index]
        return None

    def _snap_seconds(self, seconds: float, moving_clip_id: str) -> float:
        """Snap to other clips' edges and the playhead within SNAP_PX."""
        candidates = []
        if self._arrangement is not None:
            for placed in self._arrangement.placed:
                if placed.clip.id == moving_clip_id:
                    continue
                candidates.extend((placed.start_s, placed.end_s))
        if self._playhead_s is not None:
            candidates.append(self._playhead_s)
        candidates.append(0.0)
        best = seconds
        best_dist = SNAP_PX / self._zoom
        for c in candidates:
            d = abs(c - seconds)
            if d <= best_dist:
                best, best_dist = c, d
        return max(0.0, best)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)

        clip_id = self._drag_clip_id
        start_pos = self._drag_start_pos
        clip_x_range = self._drag_clip_x_range
        shift = self._drag_shift
        mode, payload = self._drag_mode, self._drag_payload
        self._drag_clip_id = None
        self._drag_start_pos = None
        self._drag_clip_x_range = None
        self._drag_shift = False
        self._drag_mode = None
        self._drag_payload = None

        if mode is not None and start_pos is not None and event.button() == Qt.MouseButton.LeftButton:
            self._finish_drag(mode, payload, start_pos, event.position().toPoint())
            return

        if clip_id is None or start_pos is None or self._document is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()
        delta = pos - start_pos
        distance = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
        if distance <= self._drag_threshold_px:
            return  # a plain click, already handled on press

        document = self._document
        clip = document.get_clip(clip_id)
        if clip is None:
            return

        scene_pos = self.mapToScene(pos)
        press_scene = self.mapToScene(start_pos)
        target_track = self._track_at_y(document, scene_pos.y())

        # Shift+drag inside one block: sub-range TTS replacement (item 9).
        if shift and clip_x_range is not None:
            extent = document.clip_extent(clip.id)
            left, right = clip_x_range
            if extent is not None and left <= press_scene.x() <= right and left <= scene_pos.x() <= right:
                clip_start, clip_end = extent
                span = max(right - left, 1e-6)
                chars = clip_end - clip_start
                press_offset = clip_start + round((press_scene.x() - left) / span * chars)
                release_offset = clip_start + round((scene_pos.x() - left) / span * chars)
                press_offset = max(clip_start, min(clip_end, press_offset))
                release_offset = max(clip_start, min(clip_end, release_offset))
                sub_start, sub_end = sorted((press_offset, release_offset))
                if sub_start != sub_end:
                    self.subRangeTtsRequested.emit(clip.id, sub_start, sub_end)
            return

        moved_horizontally = abs(delta.x()) > self._drag_threshold_px
        if moved_horizontally and clip_x_range is not None:
            left, _right = clip_x_range
            new_left = left + (scene_pos.x() - press_scene.x())
            new_start = self._snap_seconds(x_to_seconds(new_left, self._zoom), clip.id)
            self.clipMoved.emit(clip.id, new_start)

        if target_track is None or target_track.id == clip.track_id:
            return

        if target_track.character_id is None or target_track.character_id == clip.character_id:
            self.clipDragReassigned.emit(clip.id, target_track.id, False)
            return

        msg = QMessageBox(self)
        msg.setWindowTitle("Reassign Character?")
        msg.setText(
            "This track belongs to a different character. Reassign this "
            "clip's character to match the track, or just move it to this "
            "lane while keeping its own character?"
        )
        reassign_btn = msg.addButton("Reassign", QMessageBox.ButtonRole.AcceptRole)
        move_btn = msg.addButton("Just Move", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()

        if clicked is reassign_btn:
            self.clipDragReassigned.emit(clip.id, target_track.id, True)
        elif clicked is move_btn:
            self.clipDragReassigned.emit(clip.id, target_track.id, False)

    def _finish_drag(self, mode: str, payload, start_pos, pos) -> None:
        """Release half of the ruler, fade and automation drags."""
        press_scene, scene_pos = self.mapToScene(start_pos), self.mapToScene(pos)
        moved = abs(pos.x() - start_pos.x()) > self._drag_threshold_px \
            or abs(pos.y() - start_pos.y()) > self._drag_threshold_px
        seconds = x_to_seconds(scene_pos.x(), self._zoom)
        if mode == "marker":
            if moved:
                self.markerMoved.emit(payload, round(seconds, 3))
            else:
                self.seekRequested.emit(x_to_seconds(press_scene.x(), self._zoom))
        elif mode == "loop":
            start_s = x_to_seconds(press_scene.x(), self._zoom)
            if moved and abs(seconds - start_s) > 1e-3:
                self.loopRangeRequested.emit(*sorted((start_s, seconds)))
        elif mode in ("fade_in", "fade_out") and moved:
            block = payload
            local_x = block.mapFromScene(scene_pos).x()
            width_s = block.duration_s
            if mode == "fade_in":
                fade = local_x / self._zoom
            else:
                fade = (block.boundingRect().width() - local_x) / self._zoom
            fade = round(max(0.0, min(width_s, fade)), 3)
            self.fadeChanged.emit(block.clip_id, f"{mode}_s", fade)
        elif mode == "automation" and moved:
            track_id, index = payload
            lane = self._automation_items.get(track_id)
            if lane is None:
                return
            points = [list(p) for p in lane.points]
            lo = points[index - 1][0] if index > 0 else 0.0
            hi = points[index + 1][0] if index + 1 < len(points) else float("inf")
            points[index] = [round(max(lo, min(hi, seconds)), 3), lane.y_to_gain(scene_pos.y())]
            self.automationChanged.emit(track_id, points)
        elif mode == "automation_segment" and moved:
            track_id, index = payload
            lane = self._automation_items.get(track_id)
            if lane is None:
                return
            delta = lane.y_to_gain(scene_pos.y()) - lane.y_to_gain(press_scene.y())
            points = [list(p) for p in lane.points]
            for i in (index, index + 1):
                points[i][1] = round(max(0.0, min(AUTOMATION_MAX_GAIN, points[i][1] + delta)), 3)
            self.automationChanged.emit(track_id, points)

    def _select_clip_block_at(self, pos) -> None:
        block = self._clip_block_at(pos)
        if block is not None:
            if block.clip_id is not None:
                self._selection_model.select_clip(block.clip_id)
            return
        item = self.itemAt(pos)
        character_id = getattr(item, "character_id", None) if item is not None else None
        if character_id is not None:
            self._selection_model.select_character(character_id)
        else:
            self._selection_model.clear()

    def _on_selection_changed(self) -> None:
        if self._selected_block is not None:
            self._selected_block.set_selected(False)
            self._selected_block = None

        clip_id = self._selection_model.selected_clip_id
        block = self._blocks_by_clip_id.get(clip_id) if clip_id is not None else None
        if block is not None:
            block.set_selected(True)
            self.ensureVisible(block)
            self._selected_block = block

    # -- playhead --------------------------------------------------------------------

    def set_playhead(self, seconds: Optional[float]) -> None:
        self._playhead_s = seconds
        if self._playhead_item is None:
            return
        if seconds is None:
            self._playhead_item.hide()
            if self._ruler is not None:
                self._ruler.set_playhead_x(None)
            return
        x = seconds_to_x(seconds, self._zoom)
        height = RULER_HEIGHT_PX + max(self._lane_count, 1) * LANE_HEIGHT_PX
        self._playhead_item.setLine(x, RULER_HEIGHT_PX, x, height)
        self._playhead_item.show()
        if self._ruler is not None:
            self._ruler.set_playhead_x(x)
        if time.monotonic() - self._last_user_scroll > AUTO_SCROLL_GRACE_S:
            viewport_left = self.mapToScene(0, 0).x()
            viewport_right = self.mapToScene(self.viewport().width(), 0).x()
            if x < viewport_left or x > viewport_right - 20:
                self._programmatic_scroll = True
                try:
                    self.horizontalScrollBar().setValue(int(x - 40))
                finally:
                    self._programmatic_scroll = False

    # -- rendering -----------------------------------------------------------------

    def set_arrangement(self, arrangement: Arrangement) -> None:
        if self._document is None:
            return
        self.render_document(self._document, arrangement)

    def _peaks_for(self, clip, fallback_path: str, bucket_count: int):
        """Waveform peaks from the owner's rendered samples when it gave us a
        `clip_samples` callable, else from the raw first-segment file."""
        if self._clip_samples is not None:
            rendered = self._clip_samples(clip)
            if rendered is not None:
                samples, rate = rendered
                return waveform_data.compute_peaks(samples, rate, bucket_count)
        peaks, _duration = waveform_data.load_peaks_from_file(fallback_path, bucket_count)
        return peaks

    def render_document(self, document, arrangement: Optional[Arrangement] = None,
                        clip_samples=None, nested_state=None) -> None:
        """`nested_state(clip)` gives a subproject block's state ("ok",
        "stale", "missing"); unset, every nested block paints "stale"."""
        pal = theme.current()
        if nested_state is not None:
            self._nested_state_fn = nested_state
        self._document = document
        if arrangement is None:
            arrangement = compute_arrangement(document)
        self._arrangement = arrangement
        if clip_samples is not None:
            self._clip_samples = clip_samples

        self._scene.clear()
        self._blocks_by_clip_id = {}
        self._selected_block = None
        self._playhead_item = None
        self._ruler = None

        # Only tracks with clips are drawn (grill PR4); an unused one keeps
        # its mixer settings in the model.
        tracks = document.used_tracks()
        overlapping = {clip_id for pair in overlaps(arrangement) for clip_id in pair}
        lane_index_by_track_id = {track.id: i for i, track in enumerate(tracks)}
        self._lane_count = len(tracks)

        visible_seconds = self.viewport().width() / self._zoom if self._zoom > 0 else 0.0
        total_seconds = max(arrangement.total_duration_s + 2.0, visible_seconds, MIN_SCENE_SECONDS)
        total_width = seconds_to_x(total_seconds, self._zoom)
        total_height = RULER_HEIGHT_PX + max(len(tracks), 1) * LANE_HEIGHT_PX

        for i, _track in enumerate(tracks):
            y = lane_top(i)
            lane_rect = QGraphicsRectItem(0, y, total_width, LANE_HEIGHT_PX)
            lane_rect.setBrush(QColor(pal.lane_bg if i % 2 == 0 else pal.lane_alt_bg))
            lane_rect.setPen(QPen(QColor(pal.lane_border)))
            lane_rect.setZValue(-1)
            self._scene.addItem(lane_rect)

        # Grid lines at the ruler's ticks, under the clips.
        step = choose_tick_step(self._zoom)
        t = step
        while t < total_seconds:
            x = seconds_to_x(t, self._zoom)
            grid = QGraphicsLineItem(x, RULER_HEIGHT_PX, x, total_height)
            grid_color = QColor(pal.lane_border)
            grid_color.setAlpha(120)
            grid.setPen(QPen(grid_color, 1, Qt.PenStyle.DotLine))
            grid.setZValue(-0.5)
            self._scene.addItem(grid)
            t += step

        for placed in arrangement.placed:
            clip = placed.clip
            track = document.get_track(clip.track_id)
            if track is None:
                continue
            character = document.get_character(clip.character_id)
            color = character.highlight_color if character is not None else FALLBACK_CLIP_COLOR
            if clip.is_nested:
                color = pal.accent

            x = seconds_to_x(placed.start_s, self._zoom)
            width = max(seconds_to_x(placed.duration_s, self._zoom), MIN_CLIP_WIDTH_PX)
            y = lane_top(lane_index_by_track_id[track.id]) + LANE_MARGIN_PX
            height = LANE_HEIGHT_PX - 2 * LANE_MARGIN_PX

            block = ClipBlockItem()
            block.set_overlap(clip.id in overlapping)
            block.set_clip_id(clip.id)
            block.set_color(color)
            if clip.is_nested:
                block.set_label(document.clip_text(clip))
                state_fn = getattr(self, "_nested_state_fn", None)
                block.set_nested_state(state_fn(clip) if state_fn is not None else "stale")
            else:
                block.set_label(character.name if character is not None else "")
            block.set_geometry(x, y, width, height)
            block.set_estimated(placed.estimated)
            block.start_s = placed.start_s
            block.duration_s = placed.duration_s
            block.set_fades_px(seconds_to_x(float(clip.fade_in_s or 0.0), self._zoom),
                               seconds_to_x(float(clip.fade_out_s or 0.0), self._zoom))
            has_character_fx = bool(character is not None and character.preset_data.get("fx_preset")
                                    and character.preset_data.get("fx_preset") != "Select FX Preset...")
            block.set_fx_active(bool(clip.fx_override) or bool(clip.overrides.get("fx_preset")) or has_character_fx)
            self._blocks_by_clip_id[clip.id] = block
            self._scene.addItem(block)

            audio_segment = next((s for s in clip.segments if s.audio_path), None)
            block.set_audio_path(audio_segment.audio_path if audio_segment is not None else None)
            if audio_segment is not None and not placed.estimated:
                try:
                    peaks = self._peaks_for(clip, audio_segment.audio_path, max(1, int(width)))
                except Exception:
                    peaks = None
                if peaks is not None:
                    block.set_waveform(peaks, width, height)

        self._automation_items = {}
        for track in tracks:
            if track.id in self._automation_shown:
                lane = AutomationLaneItem(track.id, lane_top(lane_index_by_track_id[track.id]), total_width,
                                          self._zoom, track.automation)
                self._automation_items[track.id] = lane
                self._scene.addItem(lane)

        self._ruler = _RulerItem()
        self._ruler.set_span(total_width, self._zoom)
        self._ruler.set_document_settings(document.settings)
        self._ruler.set_markers(marker_ops.list_markers(document.settings))
        self._ruler.set_loop_s(self._loop_s)
        self._scene.addItem(self._ruler)
        self._apply_status_filter()

        self._playhead_item = QGraphicsLineItem()
        self._playhead_item.setPen(QPen(QColor(pal.playhead), 2))
        self._playhead_item.setZValue(10)
        self._playhead_item.hide()
        self._scene.addItem(self._playhead_item)

        self._scene.setSceneRect(0, 0, total_width, total_height)
        self.setBackgroundBrush(QColor(pal.panel))
        if self.header is not None:
            self.header.render_tracks(tracks, document, frozenset(self._automation_shown))

        if self._playhead_s is not None:
            self.set_playhead(self._playhead_s)
        if self._selection_model is not None:
            self._on_selection_changed()


class TimelineWidget(QWidget):
    """Header column + timeline view with linked vertical scrolling."""

    def __init__(self, parent=None, selection_model: Optional[SelectionModel] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = TrackHeaderView(selection_model=selection_model)
        self.view = TimelineView(selection_model=selection_model)
        self.view.header = self.header
        self.header.automationToggled.connect(self.view.set_automation_visible)
        layout.addWidget(self.header)
        layout.addWidget(self.view, 1)
        self.view.verticalScrollBar().valueChanged.connect(self.header.verticalScrollBar().setValue)

    def render_document(self, document, arrangement: Optional[Arrangement] = None,
                        clip_samples=None, nested_state=None) -> None:
        self.view.render_document(document, arrangement, clip_samples=clip_samples, nested_state=nested_state)
