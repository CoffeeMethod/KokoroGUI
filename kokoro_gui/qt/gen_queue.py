"""Running the generation queue (plan 28): the app half of `daw/genqueue.py`.

Generate, Generate selection and a subproject's Generate enqueue work instead
of starting it. `pump_queue` runs the queued items one at a time:

- a "clips" item is one `TimelineDock.start_batch` over at most
  `genqueue.BATCH_CLIPS` stale clips, so its results land on the timeline and
  in the transcript when it ends;
- a "subproject" item opens the child, generates its stale clips in batches of
  the same size, then renders its mixdown (what `generate_subproject` did).

The whole queue is one job to the rest of the window: `queue_active` holds the
busy state from the first item to the last, and `set_ui_state(False)` from a
step inside it (a project IO, a batch) is ignored until the queue goes idle.
Pause stops after the running item; Cancel cancels the running item and every
queued one.

While the queue runs the user keeps editing (plan 29): Generate, a gutter
generate and Fit to slot add to it instead of being refused (a single clip
or a fit is a "task" item, put ahead of the queued ones), an item re-reads
which of its clips are still stale when it starts, and New Subproject hands
the queued clips it moves to the child (`requeue_moved_clips`).
`queued_clip_ids` tells the gutter and the timeline what is queued or
generating.

The queue of the root project is written to `session.json["queue"]` after
every change (runtime state, never in `document.json` or a bundle) and read
back on Open, paused, with the clips that are still stale. Task items are
left out: they are code, not data.

Mixed into `QtTTSApp`; expects its docks, `root`, `children`, `set_ui_state`.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from kokoro_gui.daw import genqueue
from kokoro_gui.daw.genqueue import GenerationQueue, QueueItem
from kokoro_gui.daw.lanes import text_ordered_clips
from kokoro_gui.qt import project as project_io

QUEUE_KEY = "queue"


class GenerationQueueMixin:
    def _init_generation_queue(self) -> None:
        self.generation_queue = GenerationQueue()
        # Pause: take no new item after the running one.
        self.queue_paused = False
        # A run is on: from its first item to the moment the queue goes idle.
        self.queue_active = False
        self._queue_item: QueueItem | None = None
        # Cancel was asked during the run.
        self._queue_cancelled = False
        # item id -> `then(ok)` of a subproject item (`generate_subproject`).
        self._queue_then: dict = {}
        # This run's clip counts, for the final status line.
        self._queue_ok = 0
        self._queue_failed = 0
        # Clips of subproject children, which have no item of their own.
        self._queue_extra_total = 0
        self._queue_extra_done = 0
        # Clips the running batch has finished, and how far into the running
        # item that is (0 to 1), for the progress bar and the time left.
        self._queue_inflight = 0
        self._queue_fraction = 0.0
        self._queue_child_total = 0
        self._queue_child_done = 0
        self._queue_rates: dict = {}
        self._queue_written: str | None = None
        self._queue_restored = 0
        # item id -> `start(done)` of a task item (`enqueue_task`).
        self._queue_tasks: dict = {}
        # `(project_id, clip ids)` of the batch a subproject item is running.
        self._queue_batch: tuple | None = None
        self.queue_dock = None

    # -- looking things up ------------------------------------------------------

    def project_by_id(self, project_id):
        """The open project with this id, or None."""
        for project in self.open_projects():
            if project.project_id == project_id:
                return project
        return None

    def _clip_entries(self, project, clips) -> list:
        """`[(clip_id, chars, engine_id), ...]` in text order, for `plan_items`."""
        wanted = {clip.id: clip for clip in clips}
        document = project.document
        ordered = [clip for clip in text_ordered_clips(document) if clip.id in wanted]
        placed = {clip.id for clip in ordered}
        ordered += [clip for clip in clips if clip.id not in placed]
        return [(clip.id, len(document.clip_text(clip)), self.backend_for(clip, project).id) for clip in ordered]

    # -- enqueueing --------------------------------------------------------------

    def enqueue_clips(self, project, clips, title: str | None = None, front: bool = False,
                      at: int | None = None) -> int:
        """Queues `clips` (stale, generatable clips of `project`) as items of
        at most eight, in text order, after the items already queued (ahead
        of them with `front`, or at index `at`). A clip already in a queued or
        running item is not added twice. Starts the queue. Returns how many
        clips were added."""
        queue = self.generation_queue
        waiting = queue.pending_clip_ids(project.project_id)
        fresh = [clip for clip in clips if clip.id not in waiting]
        items = genqueue.plan_items(project.project_id, self._clip_entries(project, fresh), title or project.title())
        if at is not None:
            index = max(queue.first_queued_index(), min(len(queue.items), at))
        else:
            index = queue.first_queued_index() if front else len(queue.items)
        for offset, item in enumerate(items):
            queue.add(item, index + offset)
        self._kick_queue()
        return sum(item.clip_count for item in items)

    def enqueue_subproject(self, clip, project=None, then=None, front: bool = False) -> QueueItem | None:
        """Queues the nested `clip` (open the child, generate its stale clips,
        render its mixdown). One already waiting is not added twice; `then(ok)`
        replaces its callback."""
        project = project or self.project_for(clip)
        queue = self.generation_queue
        item = next((i for i in queue.items if i.state == genqueue.QUEUED and i.kind == genqueue.SUBPROJECT
                     and i.project_id == project.project_id and i.clip_ids == [clip.id]), None)
        if item is None:
            item = QueueItem(kind=genqueue.SUBPROJECT, project_id=project.project_id, clip_ids=[clip.id],
                             title=project.document.clip_text(clip) or "Subproject")
            queue.add(item, queue.first_queued_index() if front else None)
        if then is not None:
            self._queue_then[item.id] = then
        self._kick_queue()
        return item

    def enqueue_task(self, project, clip_ids, title: str, start, front: bool = True) -> QueueItem:
        """Queues work the app runs itself: a single clip's generate or a fit
        to slot asked for while a generate runs. `start(done)` begins it when
        its turn comes and returns True (it calls `done(succeeded_ids,
        failed_ids)` when it ends), False (it couldn't start) or None (there
        was nothing left to do). Goes ahead of the queued items unless
        `front` is False. Never written to `session.json`."""
        clips = [project.document.get_clip(clip_id) for clip_id in clip_ids]
        entries = self._clip_entries(project, [clip for clip in clips if clip is not None])
        engine_chars: dict = {}
        for _clip_id, chars, engine_id in entries:
            engine_chars[engine_id] = engine_chars.get(engine_id, 0) + chars
        item = QueueItem(kind=genqueue.TASK, project_id=project.project_id, clip_ids=list(clip_ids), title=title,
                         chars=sum(engine_chars.values()), engine_chars=engine_chars)
        queue = self.generation_queue
        queue.add(item, queue.first_queued_index() if front else None)
        self._queue_tasks[item.id] = start
        self._kick_queue()
        return item

    def requeue_moved_clips(self, parent, child, clip_ids) -> int:
        """New Subproject moved `clip_ids` from `parent` into `child`. Their
        queued work leaves `parent`'s items; the ones still stale in `child`
        are queued there, where those items were. A running batch's result
        for a moved clip finds no clip in `parent` and is skipped. Returns how
        many clips were queued under `child`."""
        queue = self.generation_queue
        moved = set(clip_ids) & queue.pending_clip_ids(parent.project_id)
        if not moved:
            return 0
        at = queue.take_clips(parent.project_id, moved)
        stale = [clip for clip in child.document.dirty_clips() if clip.id in moved and not clip.is_nested
                 and not self.cannot_generate(clip, child)]
        if not stale:
            self._queue_changed()
            return 0
        return self.enqueue_clips(child, stale, at=at if at >= 0 else None, front=at < 0)

    def queued_clip_ids(self, project=None) -> dict:
        """`{clip_id: "queued" | "running"}` for the clips of `project` (the
        focus by default) that queued or running work will generate: the
        gutter reads it once per paint, the timeline once per refresh. A
        queued subproject is its nested clip in the parent."""
        project = project or self.focus
        project_id = project.project_id
        states: dict = {}
        for item in self.generation_queue.items:
            if item.project_id != project_id:
                continue
            if item.state == genqueue.QUEUED:
                for clip_id in item.clip_ids:
                    states.setdefault(clip_id, "queued")
            elif item.state == genqueue.RUNNING:
                for clip_id in item.clip_ids:
                    states[clip_id] = "running"
        batch = self._queue_batch
        if batch is not None and batch[0] == project_id:
            for clip_id in batch[1]:
                states[clip_id] = "running"
        dock = getattr(self, "timeline_dock", None)
        if dock is not None:
            for clip_id in dock.clips_generating(project):
                states[clip_id] = "running"
        return states

    def _kick_queue(self) -> None:
        """Something was queued by an action of the user's: it runs, even
        after a Pause."""
        if self.generation_queue.next_queued() is None:
            self._queue_changed()
            return
        self.queue_paused = False
        self._queue_changed()
        self.pump_queue()

    # -- running -------------------------------------------------------------------

    def pump_queue(self) -> None:
        """Starts the next queued item when nothing runs; goes idle when there
        is none (or the queue is paused)."""
        if self._queue_item is not None or getattr(self, "_closed", False):
            return
        queue = self.generation_queue
        item = None if self.queue_paused or self._queue_cancelled else queue.next_queued()
        if item is None:
            self._queue_go_idle()
            return
        if not self.queue_active:
            if self.is_busy():
                # Another job holds the window; `set_ui_state(False)` pumps again.
                return
            self._queue_begin_run()
        self._start_item(item)

    def _queue_begin_run(self) -> None:
        self.generation_queue.drop_finished()
        waiting = {item.id for item in self.generation_queue.items}
        self._queue_tasks = {k: v for k, v in self._queue_tasks.items() if k in waiting}
        self.queue_active = True
        self._queue_cancelled = False
        self._queue_ok = self._queue_failed = 0
        self._queue_extra_total = self._queue_extra_done = 0
        self._queue_inflight = 0
        self._queue_fraction = 0.0
        self.set_ui_state(True)

    def _start_item(self, item: QueueItem) -> None:
        queue = self.generation_queue
        queue.mark(item, genqueue.RUNNING)
        self._queue_item = item
        self._queue_inflight = 0
        self._queue_fraction = 0.0
        self._queue_child_total = self._queue_child_done = 0
        self._queue_rates.clear()
        self._queue_changed()
        project = self.project_by_id(item.project_id)
        if item.kind == genqueue.TASK:
            self._run_task_item(item, project)
        elif project is None:
            self._finish_later(item, ok=False)
        elif item.kind == genqueue.CLIPS:
            self._run_clips_item(item, project)
        else:
            self._run_subproject_item(item, project)

    def _run_task_item(self, item: QueueItem, project) -> None:
        start = self._queue_tasks.pop(item.id, None)
        started = False
        if start is not None and project is not None:
            started = start(lambda ok_ids, bad_ids: self._item_finished(item, ok_ids, bad_ids))
        if started is None:
            self._finish_later(item)
        elif not started:
            self._finish_later(item, bad_ids=item.clip_ids)

    def _finish_later(self, item, ok_ids=(), bad_ids=(), ok=True) -> None:
        """Ends an item that did no work, on the next event loop pass, so a
        queue of such items doesn't nest one call per item."""
        QTimer.singleShot(0, lambda: self._item_finished(item, list(ok_ids), list(bad_ids), ok))

    def _replan_item(self, item: QueueItem) -> bool:
        """Keeps in `item` only what is still stale in its project; returns
        whether anything is left."""
        project = self.project_by_id(item.project_id)
        if project is None:
            return False
        document = project.document
        if item.kind == genqueue.SUBPROJECT:
            clip = document.get_clip(item.clip_ids[0])
            return clip is not None and clip.is_nested and self.nested_state(clip, project) == "stale"
        stale = {clip.id: clip for clip in document.dirty_clips() if not clip.is_nested}
        clips = [stale[clip_id] for clip_id in item.clip_ids if clip_id in stale
                 and not self.cannot_generate(stale[clip_id], project)]
        if not clips:
            return False
        # The item keeps the order it was planned in (text order).
        entries = [(clip.id, len(document.clip_text(clip)), self.backend_for(clip, project).id) for clip in clips]
        item.clip_ids = [clip_id for clip_id, _chars, _engine in entries]
        item.engine_chars = {}
        for _clip_id, chars, engine_id in entries:
            item.engine_chars[engine_id] = item.engine_chars.get(engine_id, 0) + chars
        item.chars = sum(item.engine_chars.values())
        return True

    def _run_clips_item(self, item: QueueItem, project) -> None:
        if not self._replan_item(item):
            self._finish_later(item)
            return
        started = self.timeline_dock.start_batch(
            project, item.clip_ids, lambda ok_ids, bad_ids: self._item_finished(item, ok_ids, bad_ids))
        if not started:
            self._finish_later(item, bad_ids=item.clip_ids)

    def _run_subproject_item(self, item: QueueItem, project) -> None:
        clip = project.document.get_clip(item.clip_ids[0])
        if clip is None or not clip.is_nested:
            self._finish_later(item, ok=False)
            return

        def _finished(ok):
            self._item_finished(item, [], [], ok)

        def _opened(child):
            if self._queue_cancelled or child is None:
                _finished(False)
                return
            stale = [c for c in child.document.dirty_clips() if not c.is_nested
                     and not self.cannot_generate(c, child)]
            if stale:
                entries = self._clip_entries(child, stale)
                item.engine_chars = {}
                for _clip_id, chars, engine_id in entries:
                    item.engine_chars[engine_id] = item.engine_chars.get(engine_id, 0) + chars
                item.chars = sum(item.engine_chars.values())
                self._queue_extra_total += len(entries)
                self._queue_child_total, self._queue_child_done = len(entries), 0
                batches = [i.clip_ids for i in genqueue.plan_items(child.project_id, entries, child.title())]
                self._run_child_batches(item, child, batches, _finished)
                return
            nested_stale = [c for c in child.document.nested_clips() if self.nested_state(c, child) == "stale"]
            if nested_stale:
                # Grandchildren first, then this child again.
                queue = self.generation_queue
                at = queue.first_queued_index()
                for offset, grand in enumerate(nested_stale):
                    queue.add(QueueItem(kind=genqueue.SUBPROJECT, project_id=child.project_id, clip_ids=[grand.id],
                                        title=child.document.clip_text(grand) or "Subproject"), at + offset)
                again = queue.add(QueueItem(kind=genqueue.SUBPROJECT, project_id=item.project_id,
                                            clip_ids=list(item.clip_ids), title=item.title), at + len(nested_stale))
                then = self._queue_then.pop(item.id, None)
                if then is not None:
                    self._queue_then[again.id] = then
                self._item_finished(item, [], [], True, run_then=False)
                return
            if not self.render_subproject(child, then=_finished):
                _finished(False)

        self.open_child(clip, then=_opened)

    def _run_child_batches(self, item, child, batches: list, finished) -> None:
        """Generates a child's stale clips batch by batch; the last batch's
        end renders the mixdown (`_after_project_generated` calls `finished`)."""

        def _step(index: int) -> None:
            if self._queue_cancelled:
                finished(False)
                return
            last = index == len(batches) - 1
            if last:
                self._pending_render_after_generate[child.project_id] = finished

            def _after(ok_ids, bad_ids):
                self._queue_ok += len(ok_ids)
                self._queue_failed += len(bad_ids)
                self._queue_extra_done += len(ok_ids) + len(bad_ids)
                self._queue_child_done += len(ok_ids) + len(bad_ids)
                self._queue_inflight = 0
                self._queue_fraction = self._queue_child_done / max(1, self._queue_child_total)
                if last:
                    if not ok_ids:
                        # Nothing landed, so `_after_project_generated` never ran.
                        self._pending_render_after_generate.pop(child.project_id, None)
                        finished(False)
                else:
                    QTimer.singleShot(0, lambda: _step(index + 1))

            self._queue_batch = (child.project_id, set(batches[index]))
            self.refresh_queue_marks()
            started = self.timeline_dock.start_batch(child, batches[index], _after)
            if not started:
                self._pending_render_after_generate.pop(child.project_id, None)
                finished(False)

        _step(0)

    def _item_finished(self, item: QueueItem, ok_ids: list, bad_ids: list, ok: bool = True,
                       run_then: bool = True) -> None:
        """The running item ended (its results are applied): marks it, then
        runs the next one or goes idle."""
        queue = self.generation_queue
        self._queue_batch = None
        if item.kind in genqueue.CLIP_KINDS:
            self._queue_ok += len(ok_ids)
            self._queue_failed += len(bad_ids)
            ok = not bad_ids
        state = genqueue.DONE
        if not ok:
            state = genqueue.CANCELLED if self._queue_cancelled else genqueue.FAILED
        queue.mark(item, state)
        self._queue_item = None
        self._queue_inflight = 0
        self._queue_fraction = 0.0
        then = self._queue_then.pop(item.id, None) if run_then else None
        if then is not None:
            then(ok)
        if not self._queue_cancelled and not self.queue_paused and queue.next_queued() is not None:
            self._queue_changed()
            QTimer.singleShot(0, self.pump_queue)
        else:
            self._queue_go_idle()

    def _queue_go_idle(self) -> None:
        """No item runs and none will start: the window is free again."""
        if not self.queue_active:
            self._queue_changed()
            return
        self.queue_active = False
        cancelled, self._queue_cancelled = self._queue_cancelled, False
        queue = self.generation_queue
        ok, failed = self._queue_ok, self._queue_failed
        waiting = queue.pending_clip_count()
        self._queue_changed()
        self.set_ui_state(False)
        if cancelled:
            self.set_status("Cancelled." if not ok else f"Cancelled after {ok} clip(s).", "warning")
        elif self.queue_paused and queue.next_queued() is not None:
            self.set_status(f"Paused. {waiting} clip(s) still queued.", "info")
        elif failed == 0 and ok:
            self.set_status(f"Generated {ok} clip(s).", "success")
        elif ok == 0 and failed:
            self.set_status(f"Batch generation failed for all {failed} clip(s).", "error")
        elif ok:
            self.set_status(f"Generated {ok} of {ok + failed} clips ({failed} failed)", "warning")
        self._notify_if_long_job()

    def wait_for_queue(self, timeout_s: float = 60.0) -> None:
        """Runs the event loop until the queue is not between two items (the
        next has started, or it is idle) and no project IO is in flight.
        For tests and scripts."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.wait_for_project_io(timeout_s)
            QApplication.processEvents()
            if self._io_thread is None and not (self.queue_active and self._queue_item is None):
                break
        QApplication.processEvents()

    # -- pause, cancel -------------------------------------------------------------

    def pause_queue(self) -> None:
        """Takes no new item after the running one."""
        if self.generation_queue.next_queued() is None and self._queue_item is None:
            return
        self.queue_paused = True
        if self._queue_item is not None:
            self.set_status("Pausing after this batch...", "warning")
        self._queue_changed()
        if self._queue_item is None:
            self._queue_go_idle()

    def resume_queue(self) -> None:
        self.queue_paused = False
        self._queue_changed()
        self.pump_queue()

    def toggle_queue_pause(self) -> None:
        if self.queue_paused:
            self.resume_queue()
        else:
            self.pause_queue()

    def cancel_queue(self) -> None:
        """Every queued item is cancelled; the running one stops the way
        `cancel_conversion` stops it. Called by `cancel_conversion`."""
        queue = self.generation_queue
        queue.cancel_queued()
        if self.queue_active:
            self._queue_cancelled = True
        self.queue_paused = False
        self._queue_changed()
        if self.queue_active and self._queue_item is None:
            # Between two items (a pump is pending): nothing left to wait for.
            self._queue_go_idle()

    def cancel_queue_clicked(self) -> None:
        """The Queue dock's Cancel all: stops a running generate the way the
        Transport dock's Cancel does; with nothing running it clears the
        queued items."""
        if self.queue_active:
            self.cancel_conversion()
        else:
            self.cancel_queue()

    def remove_queue_item(self, item: QueueItem) -> bool:
        if self.generation_queue.remove(item):
            self._queue_tasks.pop(item.id, None)
            self._queue_changed()
            return True
        return False

    def move_queue_item(self, item: QueueItem, new_index: int) -> bool:
        queue = self.generation_queue
        index = queue.index_of(item)
        if index >= 0 and queue.move(index, new_index):
            self._queue_changed()
            return True
        return False

    def queue_item_to_top(self, item: QueueItem) -> bool:
        if self.generation_queue.move_to_top(item):
            self._queue_changed()
            return True
        return False

    # -- progress and the time left ------------------------------------------------

    def _rate_for(self, engine_id):
        if engine_id not in self._queue_rates:
            from kokoro_gui.daw.arrangement import recorded_chars_per_second

            self._queue_rates[engine_id] = recorded_chars_per_second(engine_id)
        return self._queue_rates[engine_id]

    def queue_eta_text(self) -> str:
        """"about 2 h 10 min left", "at least 12 min left" when some item's
        size or speed isn't known, or "" when nothing can be said."""
        seconds, complete = self.generation_queue.eta_s(self._rate_for, self._queue_fraction)
        if seconds <= 0:
            return ""
        return f"{'about' if complete else 'at least'} {genqueue.format_eta(seconds)} left"

    def queue_progress_text(self) -> tuple:
        """`(percent, "12 of 340 clips, about 2 h 10 min left")`."""
        queue = self.generation_queue
        total = queue.clip_total() + self._queue_extra_total
        done = queue.clips_finished() + self._queue_extra_done + self._queue_inflight
        percent = int(done / total * 100) if total else 0
        text = f"{done} of {total} clips"
        eta = self.queue_eta_text()
        return percent, f"{text}, {eta}" if eta else text

    def on_queue_batch_progress(self, completed: int, total: int) -> None:
        """One clip of the running batch finished."""
        self._queue_inflight = completed
        item = self._queue_item
        if item is not None and item.kind in genqueue.CLIP_KINDS and total:
            self._queue_fraction = completed / total
        elif item is not None and self._queue_child_total:
            self._queue_fraction = (self._queue_child_done + completed) / self._queue_child_total
        percent, text = self.queue_progress_text()
        self.transport_dock.set_progress(percent, text)

    def queue_pending_clip_count(self) -> int:
        """Clips waiting or running, counting each queued subproject as one."""
        queue = self.generation_queue
        return sum(max(1, i.clip_count) if i.kind in genqueue.CLIP_KINDS else 1 for i in queue.pending())

    # -- session.json["queue"] -----------------------------------------------------

    def _queue_changed(self) -> None:
        self._persist_queue()
        self.refresh_queue_ui()

    def refresh_queue_ui(self) -> None:
        if getattr(self, "transport_dock", None) is not None:
            self.transport_dock.set_queue_state(bool(self.generation_queue.pending()), self.queue_paused)
        if self.queue_dock is not None:
            self.queue_dock.refresh()
        self.refresh_queue_marks()

    def refresh_queue_marks(self) -> None:
        """The queued and generating marks in the gutter and on the timeline
        follow the queue."""
        dock = getattr(self, "timeline_dock", None)
        if dock is not None:
            dock.refresh_queue_marks()
        if getattr(self, "transcript_dock", None) is not None and self.editor is not None:
            self.editor.update_gutter()

    def _persist_queue(self) -> None:
        """Writes the queued and running items to the root project's
        `session.json` when they changed; drops the key when none are left."""
        import json

        root = self.root
        if not root.project_dir:
            return
        pending = [i.to_dict() for i in self.generation_queue.pending() if i.kind in genqueue.KINDS]
        text = json.dumps(pending)
        if text == self._queue_written:
            return
        session = project_io.read_session(root.project_dir) or {}
        if pending:
            session[QUEUE_KEY] = {"items": pending}
        elif QUEUE_KEY in session:
            del session[QUEUE_KEY]
        else:
            self._queue_written = text
            return
        try:
            project_io.write_session(root.project_dir, session)
        except OSError:
            return
        self._queue_written = text

    def restore_queue(self, session) -> None:
        """Open: the queue of the project that was left with work to do comes
        back paused, with the clips that are still stale. Nothing starts until
        the user resumes it."""
        if self.queue_active:
            return
        self.generation_queue = GenerationQueue()
        self.queue_paused = False
        self._queue_written = None
        self._queue_restored = 0
        saved = GenerationQueue.from_dict((session or {}).get(QUEUE_KEY))
        for item in saved.items:
            if item.state == genqueue.QUEUED and self._replan_item(item):
                self.generation_queue.add(item)
        if self.generation_queue.items:
            self.queue_paused = True
            self._queue_restored = self.queue_pending_clip_count()
        self._queue_changed()

    def announce_restored_queue(self) -> None:
        """The status line and the Queue dock offer to resume what the last
        session left queued. Not a dialog."""
        if not self._queue_restored or not self.generation_queue.pending():
            return
        count = self.queue_pending_clip_count()
        self.set_status(f"{count} clip(s) were still queued when this project closed. "
                        "Resume them in the Queue dock.", "info")
        if self.queue_dock is not None:
            self.queue_dock.show()
            self.queue_dock.raise_()

    def carry_queue_over(self, previous: dict | None, project_dir: str) -> None:
        """Open rewrites `session.json` and keeps only `RUNTIME_STATE_KEYS`.
        When the same file reopens into its own dir, puts the previous
        session's `queue` back."""
        session = project_io.read_session(project_dir)
        if not previous or not session or QUEUE_KEY in session or QUEUE_KEY not in previous:
            return
        if previous.get("source_path") != session.get("source_path"):
            return
        session[QUEUE_KEY] = previous[QUEUE_KEY]
        try:
            project_io.write_session(project_dir, session)
        except OSError:
            pass
