"""Timeline dock: renders `app.document`'s clips/tracks/characters as a
multi-track timeline (Claude/PLAN_daw_ui_ux_redesign.md). Wires
kokoro_gui/qt/timeline_view.py's `TimelineView` into the docked shell -
unconditional, not capability-gated, since it renders Document state, which
is engine-independent.

Also owns per-clip Generate: a right-click on a clip block runs the
existing, unmodified `process_chunk_task`/`compute_cache_key` machinery
(via `KokoroEngine.generate_clip_audio`, kokoro_gui/engine/conversion.py)
for just that clip's text/config, and populates its `Segment`s with real
audio - the same "a dock owns its own Signals and reaches into
self.app.engine/self.app.document directly" pattern MixingDock's
preview/mix flow already establishes.

Ripple on regenerate: when a generate gives a clip that already had audio a
different length, every later clip placed by timestamp moves by the
difference (`arrangement.plan_ripple`, one undoable `RippleCommand`), unless
it is locked in time (`Clip.pinned`) or the project turned ripple off
(`document.settings["ripple"]`, on by default).
"""
from __future__ import annotations

import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDockWidget, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from kokoro_gui.daw import markers as marker_ops
from kokoro_gui.daw.arrangement import OVERLAP_EPSILON_S, plan_ripple
from kokoro_gui.daw.dirty import build_segments_from_results, carry_segment_timing, take_from_results
from kokoro_gui.daw.undo import (
    AssignCharacterCommand, DeleteTakeCommand, MoveClipBeforeCommand, MoveClipCommand, ReassignTrackCommand,
    RippleCommand, SetActiveTakeCommand, SetClipTimestampCommand, SetFieldCommand, TextEditCommand,
)
from kokoro_gui.qt.timeline_view import TimelineWidget


class TimelineDock(QDockWidget):
    clipGenerationFinished = Signal(str, bool, str)

    # Item 3 ("Consolidated action bar + batch dirty-scoped generation") of
    # the DAW-for-text remaining-work roadmap. batchGenerationProgress:
    # (completed_count, total_count, current_clip_label). batchGenerationFinished:
    # (succeeded_count, failed_count, failed_clip_ids) - app.py connects both
    # to update the existing status_label/progress_bar rather than routing
    # through EngineSignalBridge, whose shape is built around a single
    # character-throughput run with no per-clip identity.
    batchGenerationProgress = Signal(int, int, str)
    batchGenerationFinished = Signal(int, int, list)

    # Internal-only, bare signal: marshals a completed batch's raw outcomes
    # from the background engine-worker thread (where future.add_done_callback
    # runs its callback) onto the GUI thread, same reasoning as
    # clipGenerationFinished/_pending_results below - document mutation and
    # schedule_save()/refresh_timeline() must happen on the GUI thread.
    _batchGenerationRaw = Signal()

    def __init__(self, app, parent=None):
        super().__init__("Timeline", parent)
        self.setObjectName("dock_timeline")
        self.app = app

        self.timeline_widget = TimelineWidget(selection_model=self.app.selection)
        self.timeline_view = self.timeline_widget.view
        self.timeline_view.generateClipRequested.connect(self.on_generate_clip_requested)
        self.timeline_view.fxPresetRequested.connect(self.on_fx_preset_requested)
        self.timeline_view.clipDragReassigned.connect(self.on_clip_drag_reassigned)
        self.timeline_view.subRangeTtsRequested.connect(self.on_sub_range_tts_requested)
        self.timeline_view.clipMoved.connect(self.on_clip_moved)
        self.timeline_view.unpinRequested.connect(self.on_clip_unpin_requested)
        self.timeline_view.lockInTimeRequested.connect(self.on_lock_in_time_requested)
        self.timeline_view.playClipRequested.connect(self.on_play_clip_requested)
        self.timeline_view.fadeChanged.connect(self.on_fade_changed)
        self.timeline_view.takeSelected.connect(self.on_take_selected)
        self.timeline_view.takeDeleteRequested.connect(self.on_take_delete_requested)
        self.timeline_view.statusChangeRequested.connect(self.on_status_change_requested)
        self.timeline_view.alignWordsRequested.connect(self.on_align_words_requested)
        self.timeline_view.markerAddRequested.connect(self.on_marker_add_requested)
        self.timeline_view.markerMoved.connect(self.on_marker_moved)
        self.timeline_view.markerRenameRequested.connect(self.on_marker_rename_requested)
        self.timeline_view.markerDeleteRequested.connect(self.on_marker_delete_requested)
        self.timeline_view.loopRangeRequested.connect(self.on_loop_range_requested)
        self.timeline_view.loopClearRequested.connect(self.on_loop_clear_requested)
        self.timeline_view.automationChanged.connect(self.on_automation_changed)
        self.timeline_widget.header.trackFieldChanged.connect(self.on_track_field_changed)

        # Review filter (phase 2, A5): dims the blocks that don't match.
        self.status_filter_combo = QComboBox()
        for label, key in (("All clips", "all"), ("Not approved", "not_approved"),
                           ("Needs rewrite", "needs_rewrite")):
            self.status_filter_combo.addItem(label, key)
        self.status_filter_combo.currentIndexChanged.connect(
            lambda _i: self.timeline_view.set_status_filter(self.status_filter_combo.currentData()))
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(4, 2, 4, 0)
        filter_row.addWidget(QLabel("Show:"))
        filter_row.addWidget(self.status_filter_combo)
        filter_row.addStretch(1)
        column.addLayout(filter_row)
        column.addWidget(self.timeline_widget, 1)
        self.setWidget(content)
        if hasattr(self.app, "themeChanged"):
            self.app.themeChanged.connect(self.refresh)

        self.clipGenerationFinished.connect(self._on_clip_generation_finished)
        self._batchGenerationRaw.connect(self._on_batch_generation_raw)

        # results for a clip id whose generation future hasn't been picked
        # up by _on_clip_generation_finished yet - avoids widening
        # clipGenerationFinished's argument types just to carry the result
        # list across the thread-safe emit/handle boundary. The engine
        # reports the key, take and engine version it generated under in
        # each result dict, so nothing is predicted before dispatch.
        self._pending_results: dict = {}

        # Outcome list for a batch whose future hasn't been picked up by
        # _on_batch_generation_raw yet - same reasoning as _pending_results.
        self._pending_batch: list | None = None
        self._batch_progress_lock = threading.Lock()
        self._batch_completed = 0

        self.refresh()

    def refresh(self) -> None:
        arrangement = self.app.build_arrangement()
        self.timeline_view.render_document(self.app.document, arrangement,
                                           clip_samples=self.app.rendered_clip_samples)

    # -- seconds-axis drags (UI9) ------------------------------------------------

    def on_clip_moved(self, clip_id: str, new_start_s: float) -> None:
        """Handles `TimelineView.clipMoved`. Pins the clip's timestamp; if
        the drop lands at or before the start of the clip that precedes it in
        text order, the clip's text moves too (grill Q13) - to just before the
        first clip in text order that now starts at or after it."""
        document = self.app.document
        clip = document.get_clip(clip_id)
        if clip is None:
            return
        arrangement = self.app.build_arrangement()
        order = [p for p in arrangement.placed]
        index = next((i for i, p in enumerate(order) if p.clip.id == clip_id), None)
        predecessor = order[index - 1] if index is not None and index > 0 else None

        if predecessor is not None and new_start_s <= predecessor.start_s:
            before = next((p for p in order if p.clip.id != clip_id and p.start_s >= new_start_s), None)
            if before is not None:
                document.undo_stack.push(MoveClipBeforeCommand(clip_id, before.clip.id, timestamp=new_start_s))
                self.app.editor.load_text(document.text)
                self.app.schedule_save()
                self.app.refresh_timeline()
                return
        document.undo_stack.push(SetClipTimestampCommand(clip_id, new_start_s))
        self.app.schedule_save()
        self.app.refresh_timeline()

    def on_play_clip_requested(self, clip_id: str) -> None:
        """Context-menu Play: seek the transport to the clip and play, so
        the clip is heard with its read-time post-processing."""
        placed = self.app.current_arrangement().by_clip_id().get(clip_id)
        if placed is None:
            return
        self.app.transport.seek(placed.start_s)
        self.app.transport.play()

    # -- phase 2 edits: fades, takes, status, markers, loop, track controls ------

    def _push(self, command, rehighlight: bool = False) -> None:
        self.app.document.undo_stack.push(command)
        if rehighlight and self.app.editor is not None:
            self.app.editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    def on_fade_changed(self, clip_id: str, field: str, seconds: float) -> None:
        if self.app.document.get_clip(clip_id) is not None and field in ("fade_in_s", "fade_out_s"):
            self._push(SetFieldCommand("clip", clip_id, field, float(seconds)))

    def on_take_selected(self, clip_id: str, index: int) -> None:
        if self.app.document.get_clip(clip_id) is not None:
            self._push(SetActiveTakeCommand(clip_id, index), rehighlight=True)

    def on_take_delete_requested(self, clip_id: str, index: int) -> None:
        if self.app.document.get_clip(clip_id) is not None:
            self._push(DeleteTakeCommand(clip_id, index))

    def on_status_change_requested(self, clip_id: str, status: str) -> None:
        if self.app.document.get_clip(clip_id) is not None:
            self._push(SetFieldCommand("clip", clip_id, "status", status))

    def on_align_words_requested(self, clip_id: str) -> None:
        self.app.schedule_word_alignment([clip_id], force=True)

    def _set_markers(self, new_list: list) -> None:
        self._push(SetFieldCommand("document", None, "settings", new_list, key=marker_ops.MARKERS_KEY))

    def on_marker_add_requested(self, seconds: float) -> None:
        new_list, _marker = marker_ops.add_marker(self.app.document.settings, seconds)
        self._set_markers(new_list)

    def on_marker_moved(self, marker_id: str, seconds: float) -> None:
        self._set_markers(marker_ops.move_marker(self.app.document.settings, marker_id, seconds))

    def _ask_marker_text(self, marker: dict):
        """`(name, note)` from two input boxes, or None on cancel. Its own
        method so tests answer it without a modal."""
        name, ok = QInputDialog.getText(self, "Rename marker", "Name:", text=marker["name"])
        if not ok:
            return None
        note, ok = QInputDialog.getText(self, "Marker note", "Note (a listen-through flag):", text=marker["note"])
        return (name, note) if ok else None

    def on_marker_rename_requested(self, marker_id: str) -> None:
        marker = marker_ops.get_marker(self.app.document.settings, marker_id)
        if marker is None:
            return
        answer = self._ask_marker_text(marker)
        if answer is not None:
            self._set_markers(marker_ops.rename_marker(self.app.document.settings, marker_id, *answer))

    def on_marker_delete_requested(self, marker_id: str) -> None:
        self._set_markers(marker_ops.delete_marker(self.app.document.settings, marker_id))

    def on_loop_range_requested(self, start_s: float, end_s: float) -> None:
        """Runtime only: the transport loops there and the ruler shades it;
        the project dir's session.json remembers it for this machine."""
        self.app.set_loop_range(start_s, end_s)

    def on_loop_clear_requested(self) -> None:
        self.app.set_loop_range(None, None)

    def on_automation_changed(self, track_id: str, points) -> None:
        if self.app.document.get_track(track_id) is not None:
            self._push(SetFieldCommand("track", track_id, "automation", [list(p) for p in points]))

    def on_track_field_changed(self, track_id: str, field: str, value) -> None:
        if self.app.document.get_track(track_id) is not None and field in ("mute", "solo", "gain", "pan"):
            self._push(SetFieldCommand("track", track_id, field, value))

    def on_clip_unpin_requested(self, clip_id: str) -> None:
        if self.app.document.get_clip(clip_id) is None:
            return
        self.app.document.undo_stack.push(SetClipTimestampCommand(clip_id, None))
        self.app.schedule_save()
        self.app.refresh_timeline()

    # -- per-clip Generate ---------------------------------------------------

    def on_generate_clip_requested(self, clip_id: str, regenerate: bool = False) -> None:
        """`regenerate` is what the gutter button sends for a clip that is
        already clean; the engine then bumps the take instead of returning
        the present file (grill TB8). The dirty batch path never sets it."""
        if self.app.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before generating a clip.")
            return

        clip = self.app.document.get_clip(clip_id)
        if clip is None:
            return

        text = self.app.document.clip_text(clip)
        config = self.app._assemble_clip_config(clip)
        if regenerate:
            config["regenerate"] = True

        self.app.set_ui_state(True)

        def _done(future):
            try:
                results = future.result()
                success = bool(results)
                error = "" if success else "Generation produced no audio (cancelled or empty text)."
            except Exception as e:
                results, success, error = [], False, str(e)

            if success:
                self._pending_results[clip_id] = results
            self.clipGenerationFinished.emit(clip_id, success, error)

        engine = self.app.backend_for(clip).engine
        future = engine.worker.run_coro(engine.generate_clip_audio((0, text, config)))
        future.add_done_callback(_done)

    def on_lock_in_time_requested(self, clip_id: str, pinned: bool) -> None:
        if self.app.document.get_clip(clip_id) is not None:
            self._push(SetFieldCommand("clip", clip_id, "pinned", bool(pinned)))

    def _ripple(self, before, clip_ids) -> int:
        """Ripple on regenerate for `clip_ids`, whose new audio is already
        applied; `before` is the arrangement from just before. Only a clip
        that had audio counts (a first generate replaces an estimate, not a
        take). Returns how many clips moved."""
        document = self.app.document
        if not document.settings.get("ripple", True):
            return 0
        old = before.by_clip_id()
        deltas = {}
        for clip_id in clip_ids:
            placed = old.get(clip_id)
            clip = document.get_clip(clip_id)
            if placed is None or placed.estimated or clip is None:
                continue
            duration = self.app.clip_duration_s(clip)
            if duration is None:
                continue
            delta = duration - placed.duration_s
            if abs(delta) > OVERLAP_EPSILON_S:
                deltas[clip_id] = delta
        shifts = plan_ripple(before, deltas)
        if shifts:
            document.undo_stack.push(RippleCommand(shifts))
            self.app.set_status(f"Ripple: moved {len(shifts)} clip(s) after the regenerated audio.")
        return len(shifts)

    def _on_clip_generation_finished(self, clip_id: str, success: bool, error: str) -> None:
        self.app.set_ui_state(False)

        clip = self.app.document.get_clip(clip_id)
        if success and clip is not None:
            results = self._pending_results.pop(clip_id)
            before = self.app.build_arrangement()
            self._apply_results(clip, results)
            self._ripple(before, [clip_id])
            self.app.editor.rehighlight()
            self.app.schedule_save()
            self.app.refresh_timeline()
            self.app.schedule_word_alignment([clip_id])
        elif not success:
            self._pending_results.pop(clip_id, None)
            self.app.set_status(f"Clip generation failed: {error}", "error")

    def _apply_results(self, clip, results: list) -> None:
        """Stamps `clip.segments` and `clip.overrides["take"]` from what the
        engine reported. A result without a `cache_key` (a hand-built one
        in tests) falls back to the key the app would compute now."""
        fallback = None
        if any(not r.get("cache_key") for r in results):
            from kokoro_gui.daw.dirty import compute_expected_cache_hash

            fallback = compute_expected_cache_hash(
                self.app.document.clip_text(clip), self.app._assemble_clip_config(clip),
                key_fn=self.app.document.segment_key_fn, clip=clip,
            )
        current_take = int(clip.overrides.get("take", 0) or 0)
        take = take_from_results(results, default=current_take)
        new_segments = build_segments_from_results(fallback, results)
        carry_segment_timing(new_segments, [clip.segments, *clip.takes.values()])
        # A new take parks the one it replaces; re-rendering a parked take
        # takes it out of the parked list.
        if take != current_take and clip.segments:
            clip.takes[current_take] = clip.segments
        clip.takes.pop(take, None)
        clip.segments = new_segments
        if take:
            clip.overrides["take"] = take
        else:
            clip.overrides.pop("take", None)
        clip.status = "generated"

    # -- per-clip FX preset menu (item 5, "Per-clip FX button") --------------

    def on_fx_preset_requested(self, clip_id: str, preset_name: str) -> None:
        """Handles `TimelineView.fxPresetRequested` - an empty `preset_name`
        is the "Clear FX" case. UI6: also selects the clip and raises the
        Audio FX tab, so the tab shows the override that was just set. The
        undoable `SetClipFxCommand` push lives in
        `TranscriptDock.apply_fx_preset_to_clip`, shared with the transcript
        header's FX combo."""
        if self.app.document.get_clip(clip_id) is None:
            return
        self.app.selection.select_clip(clip_id)
        self.app.transcript_dock.apply_fx_preset_to_clip(clip_id, preset_name)
        self.app.raise_fx_tab()

    # -- drag-to-reassign (item 8, "Drag-to-reassign a clip to a different
    # track") ------------------------------------------------------------

    def on_clip_drag_reassigned(self, clip_id: str, target_track_id: str, should_reassign_character: bool) -> None:
        """Handles `TimelineView.clipDragReassigned`. `TimelineView` only
        ever hands over bare ids and the already-resolved Reassign/Just-Move
        choice (Q9) - it never touches `self.app.document` itself, keeping
        it app-independent per this file's module docstring - so clip/track
        are re-resolved fresh here before pushing the actual undoable
        command (`MoveClipCommand` for "just move", `ReassignTrackCommand`
        for "reassign", per item 4)."""
        clip = self.app.document.get_clip(clip_id)
        target_track = self.app.document.get_track(target_track_id)
        if clip is None or target_track is None:
            return

        if should_reassign_character:
            command = ReassignTrackCommand(clip_id, target_track_id, target_track.character_id)
        else:
            command = MoveClipCommand(clip_id, target_track_id)

        self.app.document.undo_stack.push(command)
        if should_reassign_character:
            # character_id changed, which changes the transcript's
            # highlight color for this clip's run(s) too.
            self.app.editor.rehighlight()
        self.app.schedule_save()
        self.app.refresh_timeline()

    # -- sub-range TTS replacement (item 9, "Sub-range TTS replacement") -----
    # Per Q27: any sub-range of any clip (including imported audio) can be
    # carved out and replaced by fresh TTS under ANY character, not
    # necessarily the clip's own. `assign_character_to_range` is already the
    # split-or-create primitive for this (tested by
    # tests/daw/test_assign_character.py) - the only new work here is the
    # dialog and correctly sequencing a real text edit (if the user edits
    # the sub-range's transcript) before the character assignment.

    def _build_sub_range_dialog(self, clip, original_text: str) -> QDialog:
        """Split out from `on_sub_range_tts_requested` so tests can build and
        inspect/drive the dialog without ever calling the blocking `.exec()`
        themselves - same precedent as `TimelineView._build_context_menu`/
        `_build_fx_menu`. Widgets are found back via `QDialog.findChild` by
        type (there's exactly one `QPlainTextEdit` and one `QComboBox` in
        this dialog), the same way a monkeypatched `.exec()` can reach in and
        supply canned "user typed X and picked character Y" input."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Replace with TTS")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Text:"))
        text_edit = QPlainTextEdit(original_text)
        layout.addWidget(text_edit)

        layout.addWidget(QLabel("Character:"))
        character_combo = QComboBox()
        default_index = 0
        for index, character in enumerate(self.app.document.characters):
            character_combo.addItem(character.name, character.id)
            if character.id == clip.character_id:
                default_index = index
        if character_combo.count():
            character_combo.setCurrentIndex(default_index)
        layout.addWidget(character_combo)

        button_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_row.addWidget(ok_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

        return dialog

    def on_sub_range_tts_requested(self, clip_id: str, sub_start: int, sub_end: int) -> None:
        """Handles `TimelineView.subRangeTtsRequested`. `TimelineView` only
        ever hands over the bare clip id and document-text offsets - it
        never touches `self.app.document` itself (same app-independence
        pattern every other signal on that widget already establishes), so
        the clip is re-resolved here before showing the dialog.

        On OK: if the dialog's (possibly edited) text differs from the
        original sub-range text, a `TextEditCommand` for that exact
        replacement is pushed FIRST - the new sub-range's end offset is then
        recomputed from the edited text's actual length, not the original
        `sub_end` (text length may have changed). Then an
        `AssignCharacterCommand` carves out `[sub_start, sub_end)` under
        whichever character was chosen (Q27: any character, not necessarily
        the parent clip's) - `assign_character_to_range`'s existing split
        logic already produces correct leftover fragments for the parent
        clip's remainder, retaining its `source`/`original_audio_path`
        unmodified. Cancel pushes nothing.
        """
        clip = self.app.document.get_clip(clip_id)
        if clip is None or clip.is_nested:
            return

        document = self.app.document
        original_text = document.text[sub_start:sub_end]
        dialog = self._build_sub_range_dialog(clip, original_text)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        text_edit = dialog.findChild(QPlainTextEdit)
        character_combo = dialog.findChild(QComboBox)
        new_text = text_edit.toPlainText()
        character_id = character_combo.currentData()

        if new_text != original_text:
            old_text = document.text
            new_full_text = old_text[:sub_start] + new_text + old_text[sub_end:]
            document.undo_stack.push(TextEditCommand(
                position=sub_start,
                chars_removed=sub_end - sub_start,
                chars_added=len(new_text),
                new_text=new_full_text,
            ))
            sub_end = sub_start + len(new_text)

        if sub_end > sub_start:
            document.undo_stack.push(AssignCharacterCommand(sub_start, sub_end, character_id))

        self.app.schedule_save()
        self.app.editor.load_text(document.text)
        self.app.refresh_timeline()
        self.generate_dirty_clips_requested()

    # -- batch dirty-scoped Generate (item 3) --------------------------------

    def generate_dirty_clips_requested(self) -> None:
        """Dispatches `KokoroEngine.generate_dirty_clips` for every clip
        `Document.dirty_clips()` currently reports as stale. Guarded by the
        same one-job-at-a-time check `on_generate_clip_requested` already
        uses. Callers (`QtTTSApp.on_generate_clicked`) are expected to have
        already checked `dirty_clips()` themselves for the "nothing to do"
        message - this method silently no-ops on an empty dirty list so it
        stays safe to call directly too."""
        if self.app.is_busy():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before generating.")
            return

        dirty = self.app.document.dirty_clips()
        if not dirty:
            return

        # One batch per engine: each clip generates with its character's
        # backend, and the backends run side by side on their own workers.
        groups: dict = {}
        clips_with_configs = []
        for clip in dirty:
            text = self.app.document.clip_text(clip)
            config = self.app._assemble_clip_config(clip)
            clips_with_configs.append((clip.id, text, config))
            engine = self.app.backend_for(clip).engine
            groups.setdefault(id(engine), (engine, []))[1].append((clip.id, text, config))

        total = len(clips_with_configs)
        self._batch_completed = 0
        self.app.set_ui_state(True)
        self.batchGenerationProgress.emit(0, total, "")

        def _on_clip_progress(clip_id, _success):
            with self._batch_progress_lock:
                self._batch_completed += 1
                completed = self._batch_completed
            self.batchGenerationProgress.emit(completed, total, clip_id)

        outcomes: list = []
        remaining = [len(groups)]

        def _done_for(group):
            def _done(future):
                try:
                    group_outcomes = future.result()
                except Exception as e:
                    # An exception here means the batch never even ran a
                    # single clip (e.g. the coroutine itself failed to
                    # schedule) - generate_dirty_clips already catches every
                    # per-clip exception internally via
                    # return_exceptions=True, so this branch is the "total
                    # failure" case, not a per-clip one.
                    group_outcomes = [
                        {"clip_id": cid, "success": False, "results": [], "error": str(e), "cancelled": False}
                        for cid, _text, _cfg in group
                    ]
                with self._batch_progress_lock:
                    outcomes.extend(group_outcomes)
                    remaining[0] -= 1
                    last = remaining[0] == 0
                if last:
                    self._pending_batch = outcomes
                    self._batchGenerationRaw.emit()
            return _done

        for engine, group in groups.values():
            future = engine.worker.run_coro(engine.generate_dirty_clips(group, progress_callback=_on_clip_progress))
            future.add_done_callback(_done_for(group))

    def _on_batch_generation_raw(self) -> None:
        self.app.set_ui_state(False)

        pending = self._pending_batch
        self._pending_batch = None
        if pending is None:
            return

        succeeded_ids = []
        failed_ids = []
        any_segments_updated = False
        before = self.app.build_arrangement()

        for outcome in pending:
            clip_id = outcome["clip_id"]
            clip = self.app.document.get_clip(clip_id)
            if outcome["success"] and clip is not None:
                self._apply_results(clip, outcome["results"])
                succeeded_ids.append(clip_id)
                any_segments_updated = True
            else:
                # Failed and cancelled clips alike: leave .segments
                # untouched (still whatever they were before this batch -
                # possibly empty/dirty, possibly stale-but-present).
                failed_ids.append(clip_id)

        if any_segments_updated:
            self._ripple(before, succeeded_ids)
            self.app.editor.rehighlight()
            self.app.schedule_save()
            self.app.refresh_timeline()
            self.app.schedule_word_alignment(succeeded_ids)

        self.batchGenerationFinished.emit(len(succeeded_ids), len(failed_ids), failed_ids)
