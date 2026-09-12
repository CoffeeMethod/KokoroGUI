"""Multi-track timeline on a real seconds axis (UI9 of
Claude/PLAN_ui_shell_redesign.md, section 4).

`TimelineView` owns the scene: a ruler across the top, one lane per
`Document.track`, one `ClipBlockItem` per placed clip, and a playhead. x is
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

The widget stays app-independent (no `self.app`): the dock owning
`app.document`/`app.engine` handles every signal. `render_document()` is a
full teardown-and-rebuild, fine for the clip counts a script has.
"""
from __future__ import annotations

import time
from typing import Optional

import playback
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsLineItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QHBoxLayout, QMenu, QMessageBox, QWidget,
)

from kokoro_gui.daw.arrangement import Arrangement, compute_arrangement
from kokoro_gui.qt import theme, waveform_data
from kokoro_gui.qt.fx_presets import list_fx_preset_names
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.waveform_view import WaveformItem

RULER_HEIGHT_PX = 22.0
LANE_HEIGHT_PX = 80.0
LANE_MARGIN_PX = 8.0
MIN_CLIP_WIDTH_PX = 20.0
DEFAULT_PIXELS_PER_SECOND = 50.0
MIN_PIXELS_PER_SECOND = 20.0
MAX_PIXELS_PER_SECOND = 400.0
HEADER_WIDTH_PX = 120
SNAP_PX = 8.0
MIN_SCENE_SECONDS = 10.0
AUTO_SCROLL_GRACE_S = 2.0
FALLBACK_CLIP_COLOR = "#888888"
SELECTED_BORDER_WIDTH_PX = 3
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

    def fx_button_rect(self) -> QRectF:
        width = min(FX_BUTTON_WIDTH_PX, self._width)
        height = min(FX_BUTTON_HEIGHT_PX, self._height)
        x = max(0.0, self._width - FX_BUTTON_WIDTH_PX)
        y = max(0.0, self._height - FX_BUTTON_HEIGHT_PX)
        return QRectF(x, y, width, height)

    def set_waveform(self, peaks, width: float, height: float) -> None:
        if self._waveform_item is None:
            self._waveform_item = WaveformItem(parent=self)
        self._waveform_item.set_peaks(peaks, width, height)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        rect = QRectF(0, 0, self._width, self._height)
        fill = QColor(self._color)
        if self._estimated:
            fill.setAlpha(90)
        painter.fillRect(rect, fill)
        if self._selected:
            painter.setPen(QPen(QColor(pal.selection_border), SELECTED_BORDER_WIDTH_PX))
        elif self._estimated:
            painter.setPen(QPen(QColor(pal.estimated_outline), 1, Qt.PenStyle.DashLine))
        else:
            painter.setPen(QPen(QColor(pal.lane_border)))
        painter.drawRect(rect)
        if self._label:
            painter.setPen(QColor(pal.text))
            painter.drawText(rect.adjusted(4, 2, -4, -2), 0, self._label)

        painter.setOpacity(FX_BUTTON_ACTIVE_OPACITY if self._fx_active else FX_BUTTON_INACTIVE_OPACITY)
        fx_rect = self.fx_button_rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(pal.fx_badge_bg))
        painter.drawRoundedRect(fx_rect, 4, 4)
        painter.setPen(QPen(QColor(pal.fx_badge_text)))
        painter.drawText(fx_rect, Qt.AlignmentFlag.AlignCenter, "FX")
        painter.setOpacity(1.0)


class _RulerItem(QGraphicsItem):
    """Ticks and mm:ss labels across the top of the scene, plus the
    playhead's triangle."""

    def __init__(self):
        super().__init__()
        self._width = 0.0
        self._zoom = DEFAULT_PIXELS_PER_SECOND
        self._playhead_x: Optional[float] = None
        self.setZValue(5)

    def set_span(self, width: float, zoom: float) -> None:
        self.prepareGeometryChange()
        self._width = width
        self._zoom = zoom
        self.update()

    def set_playhead_x(self, x: Optional[float]) -> None:
        self._playhead_x = x
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, RULER_HEIGHT_PX)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        pal = theme.current()
        painter.fillRect(self.boundingRect(), QColor(pal.ruler_bg))
        painter.setPen(QPen(QColor(pal.ruler_text)))
        step = choose_tick_step(self._zoom)
        total_s = self._width / self._zoom if self._zoom > 0 else 0.0
        t = 0.0
        while t <= total_s + 1e-6:
            x = seconds_to_x(t, self._zoom)
            painter.drawLine(QPointF(x, RULER_HEIGHT_PX - 6), QPointF(x, RULER_HEIGHT_PX))
            painter.drawText(QRectF(x + 2, 0, step * self._zoom - 4, RULER_HEIGHT_PX - 4),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), format_ruler_label(t))
            minor = t + step / 2
            if minor <= total_s:
                mx = seconds_to_x(minor, self._zoom)
                painter.drawLine(QPointF(mx, RULER_HEIGHT_PX - 3), QPointF(mx, RULER_HEIGHT_PX))
            t += step
        painter.setPen(QPen(QColor(pal.lane_border)))
        painter.drawLine(QPointF(0, RULER_HEIGHT_PX - 0.5), QPointF(self._width, RULER_HEIGHT_PX - 0.5))
        if self._playhead_x is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(pal.playhead))
            x = self._playhead_x
            painter.drawPolygon(QPolygonF([
                QPointF(x - 6, RULER_HEIGHT_PX - 12), QPointF(x + 6, RULER_HEIGHT_PX - 12), QPointF(x, RULER_HEIGHT_PX - 1),
            ]))


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


class TrackHeaderView(QGraphicsView):
    """The fixed-width track header column."""

    def __init__(self, parent=None, selection_model: Optional[SelectionModel] = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._selection_model = selection_model
        self._labels: list = []  # keep-alive for Python-subclassed items
        self.setFixedWidth(HEADER_WIDTH_PX)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    def render_tracks(self, tracks: list, document) -> None:
        pal = theme.current()
        self._scene.clear()
        self._labels = []
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
            swatch.setPen(QPen(QColor(pal.lane_border)))
            self._scene.addItem(swatch)
            label = _TrackLabelItem(track.name, track.character_id)
            label.setBrush(QColor(pal.text))
            label.setPos(22, y + 5)
            self._scene.addItem(label)
            self._labels.append(label)
        self._scene.setSceneRect(0, 0, HEADER_WIDTH_PX, total_height)
        self.setBackgroundBrush(QColor(pal.panel))

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
    fxPresetRequested = Signal(str, str)  # (clip_id, preset_name); "" clears
    clipDragReassigned = Signal(str, str, bool)  # (clip_id, target_track_id, reassign_character)
    subRangeTtsRequested = Signal(str, int, int)  # (clip_id, sub_start, sub_end) text offsets
    clipMoved = Signal(str, float)  # (clip_id, new_start_s)
    unpinRequested = Signal(str)
    seekRequested = Signal(float)
    zoomChanged = Signal(float)

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
        generate_action = menu.addAction("Generate")
        generate_action.triggered.connect(
            lambda checked=False, cid=block.clip_id: self.generateClipRequested.emit(cid)
        )

        if block.audio_path:
            play_action = menu.addAction("Play")
            play_action.triggered.connect(
                lambda checked=False, path=block.audio_path: playback.play(path, blocking=False)
            )

        clip = self._document.get_clip(block.clip_id) if self._document is not None else None
        if clip is not None and clip.timeline_timestamp is not None:
            menu.addSeparator()
            unpin = menu.addAction("Unpin from timeline")
            unpin.triggered.connect(lambda checked=False, cid=block.clip_id: self.unpinRequested.emit(cid))

        return menu

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        menu = self._build_context_menu(event.pos())
        if menu is not None:
            menu.exec(event.globalPos())

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

        if scene_pos.y() < RULER_HEIGHT_PX:
            self.seekRequested.emit(x_to_seconds(scene_pos.x(), self._zoom))
            self._drag_clip_id = None
            self._drag_start_pos = None
            return

        block = self._clip_block_at(pos)
        if block is not None and block.clip_id is not None:
            local_pos = block.mapFromScene(scene_pos)
            if block.fx_button_rect().contains(local_pos):
                self._handle_fx_button_click(block, event.globalPosition().toPoint())
                return

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
        tracks = sorted(document.tracks, key=lambda t: t.order_index)
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
        self._drag_clip_id = None
        self._drag_start_pos = None
        self._drag_clip_x_range = None
        self._drag_shift = False

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

    def render_document(self, document, arrangement: Optional[Arrangement] = None) -> None:
        pal = theme.current()
        self._document = document
        if arrangement is None:
            arrangement = compute_arrangement(document)
        self._arrangement = arrangement

        self._scene.clear()
        self._blocks_by_clip_id = {}
        self._selected_block = None
        self._playhead_item = None
        self._ruler = None

        tracks = sorted(document.tracks, key=lambda t: t.order_index)
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

            x = seconds_to_x(placed.start_s, self._zoom)
            width = max(seconds_to_x(placed.duration_s, self._zoom), MIN_CLIP_WIDTH_PX)
            y = lane_top(lane_index_by_track_id[track.id]) + LANE_MARGIN_PX
            height = LANE_HEIGHT_PX - 2 * LANE_MARGIN_PX

            block = ClipBlockItem()
            block.set_clip_id(clip.id)
            block.set_color(color)
            block.set_label(character.name if character is not None else "")
            block.set_geometry(x, y, width, height)
            block.set_estimated(placed.estimated)
            block.start_s = placed.start_s
            block.duration_s = placed.duration_s
            has_character_fx = bool(character is not None and character.preset_data.get("fx_preset")
                                    and character.preset_data.get("fx_preset") != "Select FX Preset...")
            block.set_fx_active(bool(clip.fx_override) or bool(clip.overrides.get("fx_preset")) or has_character_fx)
            self._blocks_by_clip_id[clip.id] = block
            self._scene.addItem(block)

            audio_segment = next((s for s in clip.segments if s.audio_path), None)
            block.set_audio_path(audio_segment.audio_path if audio_segment is not None else None)
            if audio_segment is not None and not placed.estimated:
                try:
                    peaks, _duration = waveform_data.load_peaks_from_file(
                        audio_segment.audio_path, max(1, int(width))
                    )
                except Exception:
                    pass
                else:
                    block.set_waveform(peaks, width, height)

        self._ruler = _RulerItem()
        self._ruler.set_span(total_width, self._zoom)
        self._scene.addItem(self._ruler)

        self._playhead_item = QGraphicsLineItem()
        self._playhead_item.setPen(QPen(QColor(pal.playhead), 2))
        self._playhead_item.setZValue(10)
        self._playhead_item.hide()
        self._scene.addItem(self._playhead_item)

        self._scene.setSceneRect(0, 0, total_width, total_height)
        self.setBackgroundBrush(QColor(pal.panel))
        if self.header is not None:
            self.header.render_tracks(tracks, document)

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
        layout.addWidget(self.header)
        layout.addWidget(self.view, 1)
        self.view.verticalScrollBar().valueChanged.connect(self.header.verticalScrollBar().setValue)

    def render_document(self, document, arrangement: Optional[Arrangement] = None) -> None:
        self.view.render_document(document, arrangement)
