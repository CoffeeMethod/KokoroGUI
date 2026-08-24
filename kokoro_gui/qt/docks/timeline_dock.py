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
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDockWidget, QMessageBox

from kokoro_gui.daw.dirty import compute_expected_cache_hash
from kokoro_gui.daw.models import Segment
from kokoro_gui.qt.timeline_view import TimelineView


class TimelineDock(QDockWidget):
    clipGenerationFinished = Signal(str, bool, str)

    def __init__(self, app, parent=None):
        super().__init__("Timeline", parent)
        self.setObjectName("dock_timeline")
        self.app = app

        self.timeline_view = TimelineView()
        self.timeline_view.generateClipRequested.connect(self.on_generate_clip_requested)
        self.setWidget(self.timeline_view)

        self.clipGenerationFinished.connect(self._on_clip_generation_finished)

        # (results, expected_cache_hash) for a clip id whose generation
        # future hasn't been picked up by _on_clip_generation_finished yet -
        # avoids widening clipGenerationFinished's argument types just to
        # carry the result list across the thread-safe emit/handle boundary.
        self._pending_results: dict = {}

        self.refresh()

    def refresh(self) -> None:
        self.timeline_view.render_document(self.app.document)

    # -- per-clip Generate ---------------------------------------------------

    def on_generate_clip_requested(self, clip_id: str) -> None:
        if self.app.cancel_btn.isEnabled():
            QMessageBox.warning(self, "Busy", "Finish or cancel the current job before generating a clip.")
            return

        clip = self.app.document.get_clip(clip_id)
        if clip is None:
            return

        text = self.app.document.clip_text(clip)
        config = self.app._assemble_clip_config(clip)
        expected_hash = compute_expected_cache_hash(text, config)

        self.app.set_ui_state(True)

        def _done(future):
            try:
                results = future.result()
                success = bool(results)
                error = "" if success else "Generation produced no audio (cancelled or empty text)."
            except Exception as e:
                results, success, error = [], False, str(e)

            if success:
                self._pending_results[clip_id] = (results, expected_hash)
            self.clipGenerationFinished.emit(clip_id, success, error)

        future = self.app.engine.worker.run_coro(self.app.engine.generate_clip_audio((0, text, config)))
        future.add_done_callback(_done)

    def _on_clip_generation_finished(self, clip_id: str, success: bool, error: str) -> None:
        self.app.set_ui_state(False)

        clip = self.app.document.get_clip(clip_id)
        if success and clip is not None:
            results, expected_hash = self._pending_results.pop(clip_id)
            # enumerate(results) for order_index, NOT each dict's "seg_idx" -
            # process_chunk_task sets seg_idx to the same outer chunk index
            # (always 0 here) for every sub-segment of one call; using it
            # directly would give every Segment order_index=0, breaking
            # dirty.is_clip_dirty's segment-count comparison.
            clip.segments = [
                Segment(order_index=i, text=result["text"], cache_key=expected_hash,
                        audio_path=result["path"], duration=result["duration"])
                for i, result in enumerate(results)
            ]
            self.app.schedule_save()
            self.app.refresh_timeline()
        elif not success:
            self._pending_results.pop(clip_id, None)
            self.app.status_label.setText(f"Clip generation failed: {error}")
            self.app.status_label.setStyleSheet("color: #ff5555;")
