"""The generation queue (plan 28): Generate plans stale clips into items of
eight, each lands when it ends, and the queue pauses, cancels, reorders and
survives a reopen (kokoro_gui/qt/gen_queue.py, daw/genqueue.py)."""
import concurrent.futures
import os

import numpy as np
import soundfile as sf

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw import genqueue
from kokoro_gui.qt import project as project_io


class Batches:
    """Gives every engine call its own Future, so a test resolves the
    queue's batches one at a time, in the order they were dispatched."""

    def __init__(self, qt_app, tmp_path):
        self.qt_app = qt_app
        self.dir = tmp_path
        self.futures = []
        qt_app.engine.worker.run_coro.side_effect = self._run

    def _run(self, _coro):
        future = concurrent.futures.Future()
        self.futures.append(future)
        return future

    @property
    def dispatched(self) -> int:
        return len(self.futures)

    def group(self, index):
        """The `(clip_id, text, config)` triples of batch `index`."""
        return self.qt_app.engine.generate_dirty_clips.call_args_list[index][0][0]

    def ids(self, index):
        return [cid for cid, _t, _c in self.group(index)]

    def resolve(self, index, fail=(), cancelled=()):
        outcomes = []
        for clip_id, text, _config in self.group(index):
            if clip_id in cancelled:
                outcomes.append({"clip_id": clip_id, "success": False, "results": [], "error": "", "cancelled": True})
            elif clip_id in fail:
                outcomes.append({"clip_id": clip_id, "success": False, "results": [], "error": "boom",
                                 "cancelled": False})
            else:
                path = self.dir / f"{clip_id}.wav"
                path.write_bytes(b"RIFF")
                outcomes.append({"clip_id": clip_id, "success": True, "error": "", "cancelled": False,
                                 "results": [{"path": str(path), "text": text, "duration": 1.0, "seg_idx": 0}]})
        self.futures[index].set_result(outcomes)


def _many_clips(qt_app, count):
    pieces = [f"Clip number {i:02d} is here." for i in range(count)]
    text = " ".join(pieces)
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clips = []
    for piece in pieces:
        start = text.index(piece)
        clips.append(qt_app.document.assign_character_to_range(start, start + len(piece), character.id))
    return clips


def _make_clean(app, clip, tmp_path):
    """Gives `clip` its audio the way a finished generate does."""
    path = tmp_path / f"clean-{clip.id}.wav"
    path.write_bytes(b"RIFF")
    results = [{"path": str(path), "text": app.document.clip_text(clip), "duration": 1.0, "seg_idx": 0}]
    app.timeline_dock._apply_results(clip, results, app.level)


def _stale_ids(qt_app):
    return {c.id for c in qt_app.document.dirty_clips()}


def _states(qt_app):
    return [i.state for i in qt_app.generation_queue.items]


# -- enqueue and run -------------------------------------------------------------


def test_generate_on_twenty_clips_runs_three_items_in_order_and_generates_everything(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)

    qt_app.on_generate_clicked()

    items = qt_app.generation_queue.items
    assert [i.clip_count for i in items] == [8, 8, 4]
    assert [c for i in items for c in i.clip_ids] == [c.id for c in clips]
    assert batches.dispatched == 1  # one item at a time
    assert qt_app.is_busy() and qt_app.queue_active

    for index in range(3):
        assert batches.dispatched == index + 1
        batches.resolve(index)
        qt_app.wait_for_queue()

    assert [batches.ids(i) for i in range(3)] == [i.clip_ids for i in items]
    assert _stale_ids(qt_app) == set()
    assert _states(qt_app) == [genqueue.DONE] * 3
    assert not qt_app.is_busy() and not qt_app.queue_active
    assert "Generated 20 clip(s)." in qt_app.transport_dock.status_text()


def test_the_first_items_clips_land_while_the_later_ones_are_still_stale(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    batches.resolve(0)
    qt_app.wait_for_queue()

    first, rest = {c.id for c in clips[:8]}, {c.id for c in clips[8:]}
    assert _stale_ids(qt_app) == rest
    assert all(clip.segments for clip in clips[:8])
    assert batches.dispatched == 2 and qt_app.is_busy()  # the next item has begun
    assert _states(qt_app) == [genqueue.DONE, genqueue.RUNNING, genqueue.QUEUED]
    assert first.isdisjoint(_stale_ids(qt_app))


def test_the_progress_bar_counts_clips_across_items(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.daw import arrangement

    monkeypatch.setattr(arrangement, "recorded_chars_per_second", lambda engine_id: 10.0)
    clips = _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    batches.resolve(0)
    qt_app.wait_for_queue()

    qt_app.timeline_dock.batchGenerationProgress.emit(3, 8, clips[10].id)

    dock = qt_app.transport_dock
    assert dock.detail_text().startswith("11 of 20 clips, about ")
    assert dock.detail_text().endswith(" left")
    assert dock.progress_bar.value() == 55


def test_a_failed_clip_fails_its_item_and_the_run_goes_on(qt_app, tmp_path):
    clips = _many_clips(qt_app, 10)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    batches.resolve(0, fail={clips[1].id})
    qt_app.wait_for_queue()
    batches.resolve(1)
    qt_app.wait_for_queue()

    assert _states(qt_app) == [genqueue.FAILED, genqueue.DONE]
    assert _stale_ids(qt_app) == {clips[1].id}
    assert "Generated 9 of 10 clips (1 failed)" in qt_app.transport_dock.status_text()


def test_generate_selection_queues_only_the_selected_clips(qt_app, tmp_path):
    clips = _many_clips(qt_app, 5)
    batches = Batches(qt_app, tmp_path)
    qt_app.selection.selected_clip_id = clips[3].id

    qt_app.generate_selection()

    assert [i.clip_ids for i in qt_app.generation_queue.items] == [[clips[3].id]]
    assert batches.ids(0) == [clips[3].id]


def test_a_clip_already_queued_or_running_is_not_queued_twice(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    added = qt_app.enqueue_clips(qt_app.level, qt_app.document.dirty_clips())

    # Eight are running and the other twelve are queued.
    assert added == 0
    assert sum(i.clip_count for i in qt_app.generation_queue.items) == 20


def test_a_job_that_is_not_the_queue_holds_the_window_until_it_ends(qt_app, tmp_path):
    _many_clips(qt_app, 3)
    batches = Batches(qt_app, tmp_path)
    qt_app.transport_dock.set_busy(True)  # an export, say

    qt_app.enqueue_clips(qt_app.level, qt_app.document.dirty_clips())

    assert batches.dispatched == 0 and not qt_app.queue_active
    qt_app.set_ui_state(False)
    qt_app.wait_for_queue()
    assert batches.dispatched == 1


def test_a_clip_made_clean_since_it_was_queued_is_dropped_when_its_item_starts(qt_app, tmp_path):
    clips = _many_clips(qt_app, 10)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    _make_clean(qt_app, clips[9], tmp_path)

    batches.resolve(0)
    qt_app.wait_for_queue()

    assert batches.ids(1) == [clips[8].id]  # clips[9] was clean by then
    assert qt_app.generation_queue.items[1].clip_ids == [clips[8].id]


# -- pause and cancel ----------------------------------------------------------------


def test_pause_after_the_first_item_stops_before_the_second(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    qt_app.toggle_queue_pause()
    assert qt_app.is_busy()  # the running item finishes
    assert qt_app.transport_dock.queue_pause_btn.text() == "Resume"
    batches.resolve(0)
    qt_app.wait_for_queue()

    assert batches.dispatched == 1
    assert _states(qt_app) == [genqueue.DONE, genqueue.QUEUED, genqueue.QUEUED]
    assert not qt_app.is_busy() and not qt_app.queue_active
    assert "Paused. 12 clip(s) still queued." in qt_app.transport_dock.status_text()
    assert len(_stale_ids(qt_app)) == 12


def test_resume_runs_the_rest(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0)
    qt_app.wait_for_queue()

    qt_app.toggle_queue_pause()

    assert batches.dispatched == 2 and qt_app.is_busy()
    assert qt_app.transport_dock.queue_pause_btn.text() == "Pause"
    batches.resolve(1)
    qt_app.wait_for_queue()
    batches.resolve(2)
    qt_app.wait_for_queue()
    assert _stale_ids(qt_app) == set()


def test_generate_after_a_pause_resumes_instead_of_queueing_a_copy(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0)
    qt_app.wait_for_queue()

    qt_app.on_generate_clicked()

    # The finished item went when the new run began.
    assert [i.clip_count for i in qt_app.generation_queue.items] == [8, 4]
    assert not qt_app.queue_paused
    assert batches.dispatched == 2


def test_cancel_stops_the_running_item_and_cancels_the_rest(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    qt_app.cancel_conversion()
    assert qt_app.engine.cancel.called
    assert _states(qt_app) == [genqueue.RUNNING, genqueue.CANCELLED, genqueue.CANCELLED]
    batches.resolve(0, cancelled={c.id for c in clips[4:8]})
    qt_app.wait_for_queue()

    assert batches.dispatched == 1
    assert not qt_app.is_busy() and not qt_app.queue_active
    assert _states(qt_app) == [genqueue.CANCELLED] * 3
    assert len(_stale_ids(qt_app)) == 16
    assert "Cancelled after 4 clip(s)." in qt_app.transport_dock.status_text()


def test_cancel_all_clears_a_paused_queue(qt_app, tmp_path):
    _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0)
    qt_app.wait_for_queue()

    qt_app.cancel_queue()

    assert _states(qt_app) == [genqueue.DONE, genqueue.CANCELLED, genqueue.CANCELLED]
    assert not qt_app.queue_paused
    assert qt_app.generation_queue.pending() == []


def test_reorder_runs_the_moved_item_first(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0)
    qt_app.wait_for_queue()
    last = qt_app.generation_queue.items[2]

    assert qt_app.queue_item_to_top(last)
    qt_app.resume_queue()

    assert batches.ids(1) == [c.id for c in clips[16:]]


# -- session.json["queue"] ---------------------------------------------------------------


def _saved(qt_app, tmp_path, count=20):
    clips = _many_clips(qt_app, count)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    assert not qt_app.is_project_dirty()
    return clips


def test_the_queue_is_written_to_session_json_and_cleared_when_it_ends(qt_app, tmp_path):
    _saved(qt_app, tmp_path)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    saved = project_io.read_session(qt_app.project_dir)["queue"]["items"]
    assert [len(i["clip_ids"]) for i in saved] == [8, 8, 4]
    batches.resolve(0)
    qt_app.wait_for_queue()
    assert len(project_io.read_session(qt_app.project_dir)["queue"]["items"]) == 2
    batches.resolve(1)
    qt_app.wait_for_queue()
    batches.resolve(2)
    qt_app.wait_for_queue()
    assert "queue" not in project_io.read_session(qt_app.project_dir)


def test_the_queue_is_never_bundled(qt_app, tmp_path):
    import zipfile

    clips = _saved(qt_app, tmp_path, count=10)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0, fail={c.id for c in clips})
    qt_app.wait_for_queue()
    assert project_io.read_session(qt_app.project_dir)["queue"]["items"]  # the second item waits
    path = qt_app.project_path
    qt_app.save_project()
    qt_app.wait_for_project_io()
    with zipfile.ZipFile(path) as zf:
        assert not any("queue" in name for name in zf.namelist())
        assert b"queue" not in zf.read("document.json")


def test_a_saved_queue_comes_back_paused_after_a_relaunch_and_resume_skips_clean_clips(qt_app, qtbot, tmp_path):
    clips = _saved(qt_app, tmp_path)
    batches = Batches(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.pause_queue()
    batches.resolve(0, fail={c.id for c in clips[:8]})  # nothing lands, so the project stays clean
    qt_app.wait_for_queue()
    assert not qt_app.is_project_dirty()
    queued = [c.id for c in clips[8:]]

    qt_app.close()
    second = qt_app_module.QtTTSApp()
    qtbot.addWidget(second)
    second.wait_for_project_io()

    assert [c for i in second.generation_queue.items for c in i.clip_ids] == queued
    assert second.queue_paused and not second.queue_active and not second.is_busy()
    assert "12 clip(s) were still queued" in second.transport_dock.status_text()
    # One of them was generated by hand meanwhile.
    _make_clean(second, second.document.get_clip(queued[0]), tmp_path)
    again = Batches(second, tmp_path)

    second.resume_queue()

    assert again.dispatched == 1
    assert queued[0] not in again.ids(0)
    assert again.ids(0) == queued[1:8]
    second._generating = False
    second.queue_active = False


def test_a_damaged_saved_queue_is_ignored_on_open(qt_app, qtbot, tmp_path):
    _saved(qt_app, tmp_path, count=3)
    session = project_io.read_session(qt_app.project_dir)
    session["queue"] = {"items": [None, {"kind": "clips"}, {"kind": "clips", "project_id": 7, "clip_ids": "x"}]}
    project_io.write_session(qt_app.project_dir, session)

    qt_app.restore_queue(project_io.read_session(qt_app.project_dir))

    assert qt_app.generation_queue.items == []
    assert not qt_app.queue_paused
    assert "queue" not in project_io.read_session(qt_app.project_dir)


# -- subprojects ----------------------------------------------------------------------------


def test_stale_subprojects_are_queue_items_that_run_in_order(qt_app):
    from tests.gui_qt.test_subprojects import _book

    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    assert qt_app.nested_state(nested) == "stale"

    qt_app.on_generate_clicked()
    qt_app.wait_for_project_io()
    qt_app.wait_for_queue()

    items = qt_app.generation_queue.items
    assert [(i.kind, i.state) for i in items] == [(genqueue.SUBPROJECT, genqueue.DONE)]
    assert items[0].clip_ids == [nested.id]
    assert qt_app.nested_state(nested) == "ok"
    assert not qt_app.is_busy()


def test_generate_subproject_runs_then_after_its_item(qt_app):
    from tests.gui_qt.test_subprojects import _book

    _intro, _chapter, child = _book(qt_app)
    nested = qt_app.document.get_clip(child.clip_id)
    results = []

    qt_app.generate_subproject(nested, then=results.append)
    qt_app.wait_for_project_io()
    qt_app.wait_for_queue()

    assert results == [True]


def test_a_childs_stale_clips_land_batch_by_batch_before_the_mixdown(qt_app, tmp_path):
    from tests.gui_qt.test_subprojects import _generated_clip

    # "Intro." and a chapter of twelve stale clips, which become a subproject.
    pieces = [f"Part {i:02d} of the chapter." for i in range(12)]
    body = " ".join(pieces)
    text = f"Intro. {body} Outro."
    qt_app.document.text = text
    qt_app.editor.load_text(text)
    _generated_clip(qt_app, 0, 6)
    character = qt_app.document.characters[0]
    for piece in pieces:
        start = text.index(piece)
        qt_app.document.assign_character_to_range(start, start + len(piece), character.id)
    child = qt_app.new_subproject(7, 7 + len(body), title="Chapter 1")
    batches = Batches(qt_app, tmp_path)
    nested = qt_app.document.get_clip(child.clip_id)

    qt_app.generate_subproject(nested)
    stale_before = len(child.document.dirty_clips())
    assert stale_before == 12
    first = batches.ids(0)
    assert len(first) == genqueue.BATCH_CLIPS
    batches.resolve(0)
    qt_app.wait_for_queue()

    assert batches.dispatched == 2  # the second batch of the same child
    assert len(child.document.dirty_clips()) == stale_before - genqueue.BATCH_CLIPS
    assert project_io.read_mixdown_info(child.project_dir) is None  # not rendered yet
    assert qt_app.queue_active
