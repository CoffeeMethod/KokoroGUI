"""Multi-track timeline view, wiring Workstream 3's waveform spike
(kokoro_gui/qt/waveform_view.py) into a real, document-backed rendering
widget (Claude/PLAN_daw_ui_ux_redesign.md).

New sibling file rather than an addition to waveform_view.py: that module's
docstring explicitly scopes it as the already-signed-off spike, with zero
knowledge of kokoro_gui.daw.models. This file reuses `WaveformItem`/
`load_peaks_from_file` unmodified via ordinary imports, keeping that
boundary honest.

**Placeholder positioning, stated plainly**: `Clip.timeline_timestamp` is
always `None` today. A per-clip Generate action (the timeline's right-click
menu) can populate real audio into a `Clip`'s `segments` now, but nothing
yet *repositions* a clip using that real duration - clips are still
positioned/sized purely from `Document.clip_extent(clip.id)` (a run-list
walk, per Claude/PLAN_text_editor_redesign.md - not a stored offset), i.e.
"where they fall in the document," not "when they play." This is an
explicit, temporary stand-in a later workstream replaces
outright once the timeline is taught to use real audio durations - it is
NOT a real time axis. A fixed pixels-per-character scale is used rather than
auto-fitting to the viewport width (unlike a single waveform's one-bucket-
per-pixel scaling): a clip's rendered size should mean the same thing
regardless of window size, which auto-fit would break.
"""
from __future__ import annotations

from typing import Optional

import playback
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QMenu,
    QMessageBox,
)

from kokoro_gui.qt import waveform_data
from kokoro_gui.qt.fx_presets import list_fx_preset_names
from kokoro_gui.qt.selection import SelectionModel
from kokoro_gui.qt.waveform_view import WaveformItem

PLACEHOLDER_PIXELS_PER_CHAR = 4.0
MIN_CLIP_WIDTH_PX = 20.0
# Item 6 ("Real time-based positioning"): once a clip has generated segments
# with real (non-None) durations, its rendered width grows to fit
# `total_duration_seconds * PIXELS_PER_SECOND` when that exceeds the
# char-based placeholder width - never smaller than the placeholder, only
# ever wider. Purely cosmetic and freely tunable; the exact value doesn't
# matter beyond "looks reasonable at the app's default zoom".
PIXELS_PER_SECOND = 50.0
LANE_HEIGHT_PX = 80.0
LANE_MARGIN_PX = 8.0
FALLBACK_CLIP_COLOR = "#888888"
LANE_BACKGROUND_COLOR = "#2b2b2b"
LANE_LABEL_COLOR = "#dddddd"
# Item 1 ("Sync layer"): a clip block selected via the transcript editor or a
# timeline click gets a thicker, brighter border instead of the normal thin
# black one.
SELECTED_BORDER_COLOR = "#ffd700"
SELECTED_BORDER_WIDTH_PX = 3
# Item 5 ("Per-clip FX button"): a small hit-testable "FX" glyph anchored to
# a block's bottom-right corner. 90% opacity when the clip actually carries
# an fx_override, 50% otherwise - per the design doc's "still discoverable,
# but visually recedes when inactive" convention.
FX_BUTTON_WIDTH_PX = 24.0
FX_BUTTON_HEIGHT_PX = 16.0
FX_BUTTON_BACKGROUND_COLOR = "#000000"
FX_BUTTON_TEXT_COLOR = "#ffffff"
FX_BUTTON_ACTIVE_OPACITY = 0.9
FX_BUTTON_INACTIVE_OPACITY = 0.5


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
        self._selected = False
        self._fx_active = False
        self.clip_id: Optional[str] = None
        self.audio_path: Optional[str] = None

    def set_clip_id(self, clip_id: str) -> None:
        self.clip_id = clip_id

    def set_audio_path(self, audio_path: Optional[str]) -> None:
        """The path `render_document` already resolved for this clip's
        waveform overlay (if any) - exposed here too so the context menu can
        offer "Play" without `TimelineView` needing its own separate lookup
        into `Document`/`Clip`."""
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
        """Item 5 ("Per-clip FX button"): whether this clip currently
        carries an `fx_override` - drives the FX glyph's opacity in
        `paint()`. Called by `render_document()` for every block on every
        rebuild, same pattern as `set_color`/`set_label`."""
        self._fx_active = active
        self.update()

    def fx_button_rect(self) -> QRectF:
        """A small, fixed-size hit-testable rect anchored to this block's
        bottom-right corner (in the block's own local coordinates - the same
        space `boundingRect()` is defined in), clamped to never exceed the
        block's own bounds for a clip narrower/shorter than the button
        itself (e.g. a `MIN_CLIP_WIDTH_PX`-floored clip)."""
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
        rect = QRectF(0, 0, self._width, self._height)
        painter.fillRect(rect, QColor(self._color))
        if self._selected:
            painter.setPen(QPen(QColor(SELECTED_BORDER_COLOR), SELECTED_BORDER_WIDTH_PX))
        else:
            painter.setPen(QPen(QColor("#000000")))
        painter.drawRect(rect)
        if self._label:
            painter.drawText(rect.adjusted(4, 2, -4, -2), 0, self._label)

        # Item 5 ("Per-clip FX button"): a small "FX" glyph, more opaque
        # when the clip actually has an fx_override. setOpacity is scoped to
        # just this one draw (restored to 1.0 immediately after) so it never
        # bleeds into the block fill/border/label painted above.
        painter.setOpacity(FX_BUTTON_ACTIVE_OPACITY if self._fx_active else FX_BUTTON_INACTIVE_OPACITY)
        fx_rect = self.fx_button_rect()
        painter.fillRect(fx_rect, QColor(FX_BUTTON_BACKGROUND_COLOR))
        painter.setPen(QPen(QColor(FX_BUTTON_TEXT_COLOR)))
        painter.drawText(fx_rect, Qt.AlignmentFlag.AlignCenter, "FX")
        painter.setOpacity(1.0)


class _TrackLabelItem(QGraphicsSimpleTextItem):
    """A `Track`'s lane-name label, click-selectable as that lane's
    character (item 1, "Sync layer" - the cross-workstream resolution that a
    `Character` can be selected by clicking its lane label, not just a clip
    block). Carries `character_id` for `TimelineView._select_clip_block_at`'s
    hit-test.

    Overrides `shape()` to return the full `boundingRect()` rather than
    `QGraphicsSimpleTextItem`'s default (a tight outline of the rendered
    glyphs only) - relying on exact glyph coverage made `itemAt()` hit-
    testing near a label's edges observably miss, and got measurably less
    reliable once the scene's spatial index rebuilt after a real paint.
    A full-rect shape is simply a more forgiving (and more correctly
    label-sized) click target either way.
    """

    def __init__(self, text: str, character_id: Optional[str]):
        super().__init__(text)
        self.character_id = character_id

    def shape(self) -> QPainterPath:  # noqa: N802 (Qt override)
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path


class TimelineView(QGraphicsView):
    """Owns the scene: one lane per `Document.track`, one `ClipBlockItem`
    per resolvable `Document.clip`. `render_document()` does a full
    teardown-and-rebuild on every call - a deliberate simplification (no
    drag/trim/incremental state to preserve yet, and document/clip counts
    are small today); revisit only if a later workstream needs frequent
    refreshes over large documents.
    """

    generateClipRequested = Signal(str)
    # Item 5 ("Per-clip FX button"): (clip_id, preset_name) - an empty
    # preset_name means "Clear FX" (clip_id, ""). Bare-payload, app-
    # independent signal, same precedent as generateClipRequested - the dock
    # owning self.app.document/self.app.engine handles it, this widget never
    # reaches for either itself.
    fxPresetRequested = Signal(str, str)
    # Item 8 ("Drag-to-reassign a clip to a different track"): (clip_id,
    # target_track_id, should_reassign_character). Emitted only once a real
    # drag+drop onto a *different, valid* track lane resolves (same-track
    # drops, drops off any lane, and plain clicks never emit this at all).
    # `should_reassign_character` is True only when the user picked
    # "Reassign" from the Q9 prompt; it's False for "Just Move" and for the
    # no-ambiguity case (target track's character already matches, or has
    # none) where there's nothing to ask about. TimelineDock re-resolves
    # clip/track objects from these ids itself and pushes the actual
    # MoveClipCommand/ReassignTrackCommand - same "bare ids across the
    # signal, dock owns document mutation" pattern fxPresetRequested
    # already establishes.
    clipDragReassigned = Signal(str, str, bool)
    # Item 9 ("Sub-range TTS replacement"): (clip_id, sub_start, sub_end) -
    # document-text offsets, sub_start < sub_end always (a zero-width
    # selection never emits). A FOURTH mouse gesture layered on the same
    # same-track drag branch item 8 already established: a drag that starts
    # AND ends within one clip block's own x-range (never crossing into a
    # neighboring clip or empty lane space) is disambiguated from item 8's
    # plain same-track no-op right there in mouseReleaseEvent - see that
    # method. TimelineDock resolves the clip and shows the replacement
    # dialog; this widget never touches Document/app, same pattern every
    # other signal here already establishes.
    subRangeTtsRequested = Signal(str, int, int)

    def __init__(self, parent=None, selection_model: Optional[SelectionModel] = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # Item 1 ("Sync layer"): NOT read via self.app - this widget stays
        # deliberately app-independent (see test_timeline_view.py, which
        # constructs it standalone with no app object at all).
        self._selection_model = selection_model
        self._blocks_by_clip_id: dict = {}
        self._selected_block: Optional[ClipBlockItem] = None
        # Item 8 ("Drag-to-reassign a clip to a different track"): the
        # `Document` last handed to render_document() - kept around (not
        # `self.app`, preserving app-independence) purely so a later
        # mouseReleaseEvent can resolve a drop position back to a track lane
        # without the dock needing to inject one separately.
        self._document = None
        # Drag-tracking state, set on mousePressEvent when the press lands
        # on a clip block, cleared on mouseReleaseEvent. Whether a given
        # press+release pair is "a drag" (moved past _drag_threshold_px) or
        # "just a click" (item 1's click-to-select, already fired on press)
        # is decided lazily in mouseReleaseEvent - see that method's
        # docstring-equivalent comment block below.
        self._drag_clip_id: Optional[str] = None
        self._drag_start_pos = None
        self._drag_threshold_px = 8
        # Item 9 ("Sub-range TTS replacement"): the origin clip block's own
        # (left, right) x-range in scene coordinates, captured alongside
        # _drag_clip_id at press time - None whenever _drag_clip_id is None.
        # mouseReleaseEvent uses this to tell item 9's sub-range gesture
        # apart from item 8's plain same-track no-op (see that method).
        self._drag_clip_x_range: Optional[tuple] = None
        # QGraphicsItem (unlike QObject) has no parent-owns-child lifetime
        # guarantee Shiboken can rely on for a *Python-subclassed* item: a
        # _TrackLabelItem added via addItem() with no other live Python
        # reference was observed to actually get garbage-collected out of
        # the scene (silently, no error) the moment a hit-test loop
        # (_clip_block_at below) let go of its own local reference - unlike
        # the plain (non-subclassed) QGraphicsRectItem/QGraphicsSimpleTextItem
        # this scene also holds, which aren't affected. This list is that
        # explicit keep-alive, rebuilt every render_document() call.
        self._track_labels: list = []
        if selection_model is not None:
            selection_model.changed.connect(self._on_selection_changed)

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
        generate_action = menu.addAction("Generate")
        generate_action.triggered.connect(
            lambda checked=False, cid=block.clip_id: self.generateClipRequested.emit(cid)
        )

        if block.audio_path:
            play_action = menu.addAction("Play")
            play_action.triggered.connect(
                lambda checked=False, path=block.audio_path: playback.play(path, blocking=False)
            )

        return menu

    def contextMenuEvent(self, event) -> None:  # noqa: N802 (Qt override)
        menu = self._build_context_menu(event.pos())
        if menu is not None:
            menu.exec(event.globalPos())

    # -- per-clip FX preset menu (item 5, "Per-clip FX button") -------------
    # A click on a block's small fx_button_rect() (see mousePressEvent
    # below), not the click-to-select gesture. Same "dock owns the Signal
    # handling, this widget stays app-independent" pattern generateClipRequested
    # already establishes - fxPresetRequested carries (clip_id, preset_name),
    # with an empty preset_name meaning "Clear FX".

    def _build_fx_menu(self, block: ClipBlockItem) -> QMenu:
        """Split out from `_handle_fx_button_click` so tests can inspect/
        trigger it without ever calling the blocking `.exec()` - same
        precedent as `_build_context_menu`."""
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

    # -- click-to-select (item 1, "Sync layer") ------------------------------
    # A separate gesture from the context menu above: left-click selects (and
    # is remembered/highlighted across re-renders), while right-click's menu
    # stays the one-shot, nothing-remembered action it always was.

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()

        # Item 5 ("Per-clip FX button"): a click landing inside a block's
        # small fx_button_rect() opens the FX-preset menu INSTEAD of
        # click-to-select for that one click - checked first, before the
        # selection_model-gated branch below, since the FX button must work
        # even on a standalone TimelineView with no selection_model wired up
        # (see test_timeline_view.py's app-independent construction). Every
        # other part of the block keeps today's click-to-select behavior.
        block = self._clip_block_at(pos)
        if block is not None and block.clip_id is not None:
            local_pos = block.mapFromScene(self.mapToScene(pos))
            if block.fx_button_rect().contains(local_pos):
                self._handle_fx_button_click(block, event.globalPosition().toPoint())
                return

        # Item 8 ("Drag-to-reassign a clip to a different track"): record
        # the press position/clip (if any) unconditionally - whether this
        # turns into a drag is only known once mouseReleaseEvent sees how
        # far the mouse actually moved. This doesn't change today's
        # click-to-select behavior below at all; a press that never moves
        # past the threshold still resolves as an ordinary click there, and
        # a press that DOES move still selected the clip here on press,
        # same as before this item.
        self._drag_start_pos = pos
        self._drag_clip_id = block.clip_id if block is not None else None
        if block is not None and block.clip_id is not None:
            left = block.pos().x()
            self._drag_clip_x_range = (left, left + block.boundingRect().width())
        else:
            self._drag_clip_x_range = None

        if self._selection_model is None:
            return
        self._select_clip_block_at(pos)

    # -- drag-to-reassign (item 8, "Drag-to-reassign a clip to a different
    # track") --------------------------------------------------------------
    # A third mouse gesture on top of click-to-select (item 1) and the FX
    # button (item 5): a press on a clip block followed by real movement
    # past _drag_threshold_px, released over a different track's lane,
    # reassigns that clip to the new lane - with a reassign-vs-move prompt
    # (Q9) when the target lane's character differs from the clip's own.

    def _track_at_y(self, document, y: float):
        """The `Track` whose lane covers scene y-coordinate `y`, using the
        exact same order_index-sorted, LANE_HEIGHT_PX-per-lane layout
        render_document() lays lanes out with - `None` if `y` falls outside
        every lane (above the first, below the last, or there are no
        tracks at all)."""
        tracks = sorted(document.tracks, key=lambda t: t.order_index)
        index = int(y // LANE_HEIGHT_PX)
        if 0 <= index < len(tracks):
            return tracks[index]
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseMoveEvent(event)
        # No drag in progress (no press landed on a clip block) - nothing to
        # do. Visual drag feedback is intentionally not implemented here;
        # the functional drop behavior is fully resolved in
        # mouseReleaseEvent below, which recomputes the same threshold
        # check independently.
        if self._drag_clip_id is None or self._drag_start_pos is None:
            return

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)

        clip_id = self._drag_clip_id
        start_pos = self._drag_start_pos
        clip_x_range = self._drag_clip_x_range
        self._drag_clip_id = None
        self._drag_start_pos = None
        self._drag_clip_x_range = None

        if clip_id is None or start_pos is None or self._document is None:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = event.position().toPoint()
        delta = pos - start_pos
        distance = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
        if distance <= self._drag_threshold_px:
            # Real movement never exceeded the threshold - a plain click,
            # already fully handled by click-to-select on press. Not a drag.
            return

        document = self._document
        clip = document.get_clip(clip_id)
        if clip is None:
            return

        scene_pos = self.mapToScene(pos)
        target_track = self._track_at_y(document, scene_pos.y())
        if target_track is None:
            # No valid target lane - a no-op, snap back to where it was.
            return

        if target_track.id == clip.track_id:
            # Item 8's plain same-track no-op, EXTENDED by item 9
            # ("Sub-range TTS replacement"): if both the press and release
            # x-coordinates stayed inside the origin clip's own x-range
            # (clip_x_range, captured at press time) - i.e. the whole drag
            # never crossed into a neighboring clip or empty lane space -
            # this is a sub-range-selection gesture instead, mapping the
            # x-coordinates back to document-text offsets via the same
            # clip_extent()-plus-pixels/PLACEHOLDER_PIXELS_PER_CHAR arithmetic
            # render_document() uses to position clips in the first place.
            # A drag that exits the origin clip's own bounds at either end
            # keeps today's plain no-op unchanged - cross-clip range
            # selection is explicitly out of scope.
            extent = document.clip_extent(clip.id)
            if clip_x_range is not None and extent is not None:
                clip_start, clip_end = extent
                left, right = clip_x_range
                press_scene_x = self.mapToScene(start_pos).x()
                release_scene_x = scene_pos.x()
                if left <= press_scene_x <= right and left <= release_scene_x <= right:
                    press_offset = clip_start + round((press_scene_x - left) / PLACEHOLDER_PIXELS_PER_CHAR)
                    release_offset = clip_start + round((release_scene_x - left) / PLACEHOLDER_PIXELS_PER_CHAR)
                    press_offset = max(clip_start, min(clip_end, press_offset))
                    release_offset = max(clip_start, min(clip_end, release_offset))
                    sub_start, sub_end = sorted((press_offset, release_offset))
                    if sub_start != sub_end:
                        self.subRangeTtsRequested.emit(clip.id, sub_start, sub_end)
            return

        if target_track.character_id is None or target_track.character_id == clip.character_id:
            # No ambiguity: either the target lane has no character of its
            # own, or it already matches the clip's - "just move" and
            # "reassign" are equivalent here, so no prompt is needed.
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
        # Any other outcome (Cancel, or the dialog dismissed some other way)
        # is a deliberate no-op: nothing emitted, clip stays where it was.

    def _select_clip_block_at(self, pos) -> None:
        block = self._clip_block_at(pos)
        if block is not None:
            if block.clip_id is not None:
                self._selection_model.select_clip(block.clip_id)
            return

        # No clip block hit - check whether the click landed on a lane label
        # instead (the cross-workstream resolution: clicking a lane label
        # selects that lane's character). render_document() below tags each
        # label item with a `character_id` attribute for exactly this check.
        item = self.itemAt(pos)
        character_id = getattr(item, "character_id", None) if item is not None else None
        if character_id is not None:
            self._selection_model.select_character(character_id)
        else:
            self._selection_model.clear()

    def _on_selection_changed(self) -> None:
        """Reacts to `selection_model.changed` by moving the highlighted
        border to whichever block (if any) matches the newly-selected clip.
        Also called at the end of every `render_document()` (see below) to
        re-apply the current selection against the freshly-rebuilt scene."""
        if self._selected_block is not None:
            self._selected_block.set_selected(False)
            self._selected_block = None

        clip_id = self._selection_model.selected_clip_id
        block = self._blocks_by_clip_id.get(clip_id) if clip_id is not None else None
        if block is not None:
            block.set_selected(True)
            self.ensureVisible(block)
            self._selected_block = block

    def render_document(self, document) -> None:
        # Item 8 ("Drag-to-reassign a clip to a different track"): kept
        # around for mouseReleaseEvent's lane-resolution - see __init__'s
        # comment on _document.
        self._document = document

        self._scene.clear()
        self._blocks_by_clip_id = {}
        self._selected_block = None
        self._track_labels = []

        tracks = sorted(document.tracks, key=lambda t: t.order_index)
        lane_index_by_track_id = {track.id: i for i, track in enumerate(tracks)}

        total_width = max(len(document.text) * PLACEHOLDER_PIXELS_PER_CHAR, MIN_CLIP_WIDTH_PX)
        total_height = max(len(tracks), 1) * LANE_HEIGHT_PX

        for i, track in enumerate(tracks):
            y = i * LANE_HEIGHT_PX
            lane_rect = QGraphicsRectItem(0, y, total_width, LANE_HEIGHT_PX)
            lane_rect.setBrush(QColor(LANE_BACKGROUND_COLOR))
            lane_rect.setPen(QPen(QColor("#444444")))
            # Explicit Z-value below the default (0) that labels/clip blocks
            # sit at: this item spans the whole lane and would otherwise only
            # be *below* them by insertion order, an easy invariant for a
            # future change to accidentally break. An explicit lower
            # Z-value here removes the ambiguity outright, for every item
            # drawn on top of a lane, not just labels.
            lane_rect.setZValue(-1)
            self._scene.addItem(lane_rect)

            label = _TrackLabelItem(track.name, track.character_id)
            label.setBrush(QColor(LANE_LABEL_COLOR))
            label.setPos(4, y + 2)
            self._scene.addItem(label)
            self._track_labels.append(label)  # keep-alive - see __init__

        # Item 6 ("Real time-based positioning"): ascending extent-start, not
        # document.clips's incidental list order (not guaranteed sorted after
        # repeated splits/edits) - so that when a clip's width legitimately
        # overruns into where the next clip on its lane starts (its audio
        # runs long relative to its text), the later-starting clip is always
        # added to the scene last and therefore painted on top,
        # deterministically rather than by incidental list order. A clip
        # with no run pointing at it any more (extent is None - shouldn't
        # normally happen, but not reachable-via-UI isn't the same as
        # impossible) is skipped rather than crashing a whole refresh.
        clips_with_extent = [
            (clip, document.clip_extent(clip.id)) for clip in document.clips
        ]
        clips_with_extent = [(c, e) for c, e in clips_with_extent if e is not None]
        for clip, (clip_start, clip_end) in sorted(clips_with_extent, key=lambda pair: pair[1][0]):
            track = document.get_track(clip.track_id)
            if track is None:
                # Not reachable via UI today (nothing lets a track be
                # deleted out from under a clip), but defended against
                # anyway rather than crashing a whole refresh over it.
                continue

            character = document.get_character(clip.character_id)
            color = character.highlight_color if character is not None else FALLBACK_CLIP_COLOR

            x = clip_start * PLACEHOLDER_PIXELS_PER_CHAR
            width = max((clip_end - clip_start) * PLACEHOLDER_PIXELS_PER_CHAR, MIN_CLIP_WIDTH_PX)
            total_duration = sum(s.duration for s in clip.segments if s.duration is not None)
            if total_duration > 0:
                width = max(width, total_duration * PIXELS_PER_SECOND)
            y = lane_index_by_track_id[track.id] * LANE_HEIGHT_PX + LANE_MARGIN_PX
            height = LANE_HEIGHT_PX - 2 * LANE_MARGIN_PX

            block = ClipBlockItem()
            block.set_clip_id(clip.id)
            block.set_color(color)
            block.set_label(character.name if character is not None else "")
            block.set_geometry(x, y, width, height)
            block.set_fx_active(bool(clip.fx_override))
            self._blocks_by_clip_id[clip.id] = block
            self._scene.addItem(block)

            audio_segment = next((s for s in clip.segments if s.audio_path), None)
            block.set_audio_path(audio_segment.audio_path if audio_segment is not None else None)
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

        # Full teardown/rebuild above means a previously-selected clip's
        # highlight is gone until re-applied - required, not optional, since
        # render_document() runs on every keystroke-triggered refresh
        # elsewhere in the document (see refresh_timeline()).
        if self._selection_model is not None:
            self._on_selection_changed()
