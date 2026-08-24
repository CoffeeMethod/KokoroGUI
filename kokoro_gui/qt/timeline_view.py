"""Multi-track timeline view, wiring Workstream 3's waveform spike
(kokoro_gui/qt/waveform_view.py) into a real, document-backed rendering
widget (Claude/PLAN_daw_ui_ux_redesign.md).

New sibling file rather than an addition to waveform_view.py: that module's
docstring explicitly scopes it as the already-signed-off spike, with zero
knowledge of kokoro_gui.daw.models. This file reuses `WaveformItem`/
`load_peaks_from_file` unmodified via ordinary imports, keeping that
boundary honest.

**Placeholder positioning, stated plainly**: `Clip.timeline_timestamp` is
always `None` today, and nothing populates real audio (hence real duration)
into a `Clip`'s `segments` yet. Clips are positioned/sized purely from their
`start_offset`/`end_offset` into `Document.text` - i.e. "where they fall in
the document," not "when they play." This is an explicit, temporary stand-in
a later workstream replaces outright once real audio durations exist - it is
NOT a real time axis. A fixed pixels-per-character scale is used rather than
auto-fitting to the viewport width (unlike a single waveform's one-bucket-
per-pixel scaling): a clip's rendered size should mean the same thing
regardless of window size, which auto-fit would break.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QMenu,
)

from kokoro_gui.qt import waveform_data
from kokoro_gui.qt.waveform_view import WaveformItem

PLACEHOLDER_PIXELS_PER_CHAR = 4.0
MIN_CLIP_WIDTH_PX = 20.0
LANE_HEIGHT_PX = 80.0
LANE_MARGIN_PX = 8.0
FALLBACK_CLIP_COLOR = "#888888"
LANE_BACKGROUND_COLOR = "#2b2b2b"
LANE_LABEL_COLOR = "#dddddd"


class ClipBlockItem(QGraphicsItem):
    """One clip's visual block on a timeline lane. Composition over
    inheriting `WaveformItem`: a clip block needs a background fill keyed
    off a per-character color plus a label, and only *optionally* an actual
    waveform overlay - a materially different, wider contract than
    `WaveformItem`'s narrow "one cached path, one fixed color" job. The
    optional waveform is a real child `QGraphicsItem`, painted by Qt's own
    scene graph automatically once added - this class never has to draw it
    itself.

    `ItemClipsChildrenToShape` is set so a child waveform can never visually
    bleed past its block into a neighboring clip on the same lane - blocks
    are packed side-by-side here, unlike the spike's single full-viewport
    item, so this clipping guarantee (rather than requiring exact-match
    arithmetic everywhere) is the one genuinely new rendering risk beyond
    what the spike already validated.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self._width = 0.0
        self._height = 0.0
        self._color = FALLBACK_CLIP_COLOR
        self._label = ""
        self._waveform_item: WaveformItem | None = None
        self.clip_id: Optional[str] = None

    def set_clip_id(self, clip_id: str) -> None:
        self.clip_id = clip_id

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

    def set_waveform(self, peaks, width: float, height: float) -> None:
        if self._waveform_item is None:
            self._waveform_item = WaveformItem(parent=self)
        self._waveform_item.set_peaks(peaks, width, height)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt override)
        return QRectF(0, 0, self._width, self._height)

    def paint(self, painter, option, widget=None) -> None:  # noqa: N802 (Qt override)
        rect = QRectF(0, 0, self._width, self._height)
        painter.fillRect(rect, QColor(self._color))
        painter.setPen(QPen(QColor("#000000")))
        painter.drawRect(rect)
        if self._label:
            painter.drawText(rect.adjusted(4, 2, -4, -2), 0, self._label)


class TimelineView(QGraphicsView):
    """Owns the scene: one lane per `Document.track`, one `ClipBlockItem`
    per resolvable `Document.clip`. `render_document()` does a full
    teardown-and-rebuild on every call - a deliberate simplification (no
    drag/trim/incremental state to preserve yet, and document/clip counts
    are small today); revisit only if a later workstream needs frequent
    refreshes over large documents.
    """

    generateClipRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

    # -- per-clip Generate context menu -------------------------------------
    # Introduces no persistent selection/highlight state - a one-shot QMenu
    # triggered by hit-testing, same self-contained pattern
    # TranscriptEditor's Characters menu already uses for text selections,
    # just anchored to a clip block instead. Deliberately not the
    # click-to-select sync layer (Workstream 4) - nothing here is
    # remembered after the menu closes.

    def _clip_block_at(self, pos) -> Optional[ClipBlockItem]:
        item = self.itemAt(pos)
        while item is not None and not isinstance(item, ClipBlockItem):
            # A right-click can land on a clip's child WaveformItem (drawn
            # on top of its parent block), so walk up to find the block.
            item = item.parentItem()
        return item

    def _build_context_menu(self, pos) -> Optional[QMenu]:
        """Split out from `contextMenuEvent` so tests can inspect/trigger it
        without ever calling the blocking `.exec()` - same precedent as
        TranscriptEditor._build_context_menu."""
        block = self._clip_block_at(pos)
        if block is None or block.clip_id is None:
            return None

        menu = QMenu(self)
        action = menu.addAction("Generate")
        action.triggered.connect(lambda checked=False, cid=block.clip_id: self.generateClipRequested.emit(cid))
        return menu

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        menu = self._build_context_menu(event.pos())
        if menu is not None:
            menu.exec(event.globalPos())

    def render_document(self, document) -> None:
        self._scene.clear()

        tracks = sorted(document.tracks, key=lambda t: t.order_index)
        lane_index_by_track_id = {track.id: i for i, track in enumerate(tracks)}

        total_width = max(len(document.text) * PLACEHOLDER_PIXELS_PER_CHAR, MIN_CLIP_WIDTH_PX)
        total_height = max(len(tracks), 1) * LANE_HEIGHT_PX

        for i, track in enumerate(tracks):
            y = i * LANE_HEIGHT_PX
            lane_rect = QGraphicsRectItem(0, y, total_width, LANE_HEIGHT_PX)
            lane_rect.setBrush(QColor(LANE_BACKGROUND_COLOR))
            lane_rect.setPen(QPen(QColor("#444444")))
            self._scene.addItem(lane_rect)

            label = QGraphicsSimpleTextItem(track.name)
            label.setBrush(QColor(LANE_LABEL_COLOR))
            label.setPos(4, y + 2)
            self._scene.addItem(label)

        for clip in document.clips:
            track = document.get_track(clip.track_id)
            if track is None:
                # Not reachable via UI today (nothing lets a track be
                # deleted out from under a clip), but defended against
                # anyway rather than crashing a whole refresh over it.
                continue

            character = document.get_character(clip.character_id)
            color = character.highlight_color if character is not None else FALLBACK_CLIP_COLOR

            x = clip.start_offset * PLACEHOLDER_PIXELS_PER_CHAR
            width = max((clip.end_offset - clip.start_offset) * PLACEHOLDER_PIXELS_PER_CHAR, MIN_CLIP_WIDTH_PX)
            y = lane_index_by_track_id[track.id] * LANE_HEIGHT_PX + LANE_MARGIN_PX
            height = LANE_HEIGHT_PX - 2 * LANE_MARGIN_PX

            block = ClipBlockItem()
            block.set_clip_id(clip.id)
            block.set_color(color)
            block.set_label(character.name if character is not None else "")
            block.set_geometry(x, y, width, height)
            self._scene.addItem(block)

            audio_segment = next((s for s in clip.segments if s.audio_path), None)
            if audio_segment is not None:
                try:
                    peaks, _duration = waveform_data.load_peaks_from_file(
                        audio_segment.audio_path, max(1, int(width))
                    )
                except Exception:
                    # A missing/stale/corrupt audio_path must degrade to the
                    # flat-color block already set above, not crash the
                    # whole refresh - this branch isn't reachable via the
                    # live app yet (nothing populates real segments), but
                    # must still hold once a later workstream does.
                    pass
                else:
                    block.set_waveform(peaks, width, height)

        self._scene.setSceneRect(0, 0, total_width, total_height)
