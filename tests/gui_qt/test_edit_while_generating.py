"""Editing while the generation queue runs (plan 29): `is_generating` and
`is_io_busy` split the busy flag, Generate, a gutter generate and Fit to slot
queue instead of being refused, New Subproject takes its clips' queued work
along, a device change waits, Save and an export run beside the queue, a
result for a clip edited or deleted meanwhile never overwrites its newer
text, and the gutter and the timeline mark what is queued.

The engine is a stub whose every job is a Future the test resolves, so a
generate stays running until the test lets it end."""
import concurrent.futures
import os
import threading

from PySide6.QtWidgets import QApplication

from kokoro_gui.daw import genqueue
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt.transcript_editor import TranscriptGutter


class Jobs:
    """Every engine job is a Future held here, in dispatch order. A job is
    `("batch", [(clip_id, text, config), ...])` or `("clip", (index, text,
    config))`."""

    def __init__(self, qt_app, tmp_path):
        self.qt_app = qt_app
        self.dir = tmp_path
        self.jobs = []
        engine = qt_app.engine
        engine.generate_dirty_clips.side_effect = lambda items, progress_callback=None: ("batch", list(items))
        engine.generate_clip_audio.side_effect = lambda chunk: ("clip", chunk)
        engine.worker.run_coro.side_effect = self._run

    def _run(self, job):
        future = concurrent.futures.Future()
        self.jobs.append((job, future))
        return future

    def kind(self, index):
        return self.jobs[index][0][0]

    def ids(self, index):
        return [cid for cid, _t, _c in self.jobs[index][0][1]]

    def _result(self, name, text, key=None):
        path = self.dir / f"{name}.wav"
        path.write_bytes(b"RIFF")
        result = {"path": str(path), "text": text, "duration": 1.0, "seg_idx": 0}
        if key is not None:
            result.update(cache_key=key, engine_version=self.qt_app.backend.engine_version())
        return result

    def resolve(self, index, keys=None):
        """Ends job `index` with one segment per clip; `keys` gives a clip's
        `cache_key` (the key its text had at dispatch)."""
        (kind, payload), future = self.jobs[index]
        keys = keys or {}
        if kind == "clip":
            _i, text, _config = payload
            future.set_result([self._result(f"clip-{index}", text)])
            return
        outcomes = [{"clip_id": cid, "success": True, "error": "", "cancelled": False,
                     "results": [self._result(f"{cid}-{index}", text, keys.get(cid))]}
                    for cid, text, _config in payload]
        future.set_result(outcomes)


def _many_clips(qt_app, count):
    pieces = [f"Clip number {i:02d} is here." for i in range(count)]
    text = " ".join(pieces)
    qt_app.editor.load_text(text)
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clips = []
    for piece in pieces:
        start = text.index(piece)
        clips.append(qt_app.document.assign_character_to_range(start, start + len(piece), character.id))
    return clips


def _stale_ids(qt_app, project=None):
    return {c.id for c in (project or qt_app.root).document.dirty_clips()}


# -- the two busy flags ------------------------------------------------------------


def test_a_running_queue_is_generating_and_not_io_busy(qt_app, tmp_path):
    _many_clips(qt_app, 4)
    Jobs(qt_app, tmp_path)
    assert not qt_app.is_generating() and not qt_app.is_io_busy() and not qt_app.is_busy()

    qt_app.on_generate_clicked()

    assert qt_app.is_generating() and qt_app.is_busy()
    assert not qt_app.is_io_busy() and not qt_app.generate_blocked()
    assert qt_app.transport_dock.generate_btn.isEnabled()


def test_a_job_that_is_not_a_generate_is_io_busy_and_blocks_generate(qt_app, tmp_path):
    _many_clips(qt_app, 2)
    qt_app.begin_side_job()

    assert qt_app.is_io_busy() and not qt_app.is_generating() and qt_app.generate_blocked()
    assert not qt_app.transport_dock.generate_btn.isEnabled()
    assert qt_app.timeline_dock.generate_dirty_clips_requested() == 0

    qt_app.end_side_job()
    assert not qt_app.is_busy()


# -- queue instead of refuse ---------------------------------------------------------


def test_generate_while_the_queue_runs_adds_the_new_stale_clips(qt_app, tmp_path):
    clips = _many_clips(qt_app, 12)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.timeline_dock.generate_dirty_clips_requested(clip_ids={c.id for c in clips[:4]})
    assert qt_app.is_generating() and len(qt_app.generation_queue.items) == 1

    qt_app.on_generate_clicked()

    queued = [c for i in qt_app.generation_queue.items[1:] for c in i.clip_ids]
    assert queued == [c.id for c in clips[4:]]
    for index in range(2):
        jobs.resolve(index)
        qt_app.wait_for_queue()
    assert _stale_ids(qt_app) == set() and not qt_app.is_busy()


def test_a_gutter_generate_during_a_running_queue_is_queued_at_the_front(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    assert len(qt_app.generation_queue.items) == 3

    qt_app.generate_clip(clips[19].id)

    items = qt_app.generation_queue.items
    assert [i.kind for i in items] == [genqueue.CLIPS, genqueue.TASK, genqueue.CLIPS, genqueue.CLIPS]
    assert items[1].clip_ids == [clips[19].id] and items[1].state == genqueue.QUEUED
    assert len(jobs.jobs) == 1  # nothing ran beside the running batch

    jobs.resolve(0)
    qt_app.wait_for_queue()

    assert jobs.kind(1) == "clip"  # the user's clip went next
    jobs.resolve(1)
    qt_app.wait_for_queue()
    assert clips[19].id not in _stale_ids(qt_app)
    assert jobs.kind(2) == "batch" and jobs.ids(2) == [c.id for c in clips[8:16]]


def test_a_queued_task_is_never_written_to_session_json(qt_app, tmp_path):
    clips = _many_clips(qt_app, 10)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.generate_clip(clips[9].id)

    saved = project_io.read_session(qt_app.root.project_dir)["queue"]["items"]
    assert genqueue.TASK not in {item["kind"] for item in saved}
    assert genqueue.GenerationQueue.from_dict({"items": [dict(saved[0], kind=genqueue.TASK)]}).items == []


def test_fit_to_slot_while_the_queue_runs_is_queued_at_the_front(qt_app, tmp_path):
    clips = _many_clips(qt_app, 10)
    clips[9].overrides["target_duration_s"] = 0.5
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    assert qt_app.timeline_dock.on_fit_to_slot_requested(clips[9].id) is True

    items = qt_app.generation_queue.items
    assert items[1].kind == genqueue.TASK and items[1].clip_ids == [clips[9].id]
    assert not qt_app.timeline_dock._fits  # nothing runs beside the batch
    jobs.resolve(0)
    qt_app.wait_for_queue()
    assert jobs.kind(1) == "clip" and qt_app.timeline_dock._fits


def test_auto_split_while_the_queue_runs_queues_the_new_clips(qt_app, tmp_path):
    _many_clips(qt_app, 3)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    text = qt_app.document.text + "\n\n[Default]: A tagged line at the end."
    qt_app.editor.load_text(text)
    qt_app.document.text = text

    qt_app.auto_split_and_generate()

    new = [c for c in qt_app.document.clips if "tagged line" in qt_app.document.clip_text(c)]
    assert new and new[0].id in qt_app.generation_queue.pending_clip_ids()


# -- allowed beside a generate ---------------------------------------------------------


def test_new_subproject_takes_the_moved_clips_queued_work_along(qt_app, tmp_path):
    clips = _many_clips(qt_app, 20)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    root = qt_app.root
    text = root.document.text
    # Clips 4-11: four in the running batch, four in the next queued item.
    start = text.index("Clip number 04")
    end = text.index("Clip number 11") + len("Clip number 11 is here.")

    child = qt_app.new_subproject(start, end, title="Chapter")

    assert child is not None
    moved = {c.id for c in clips[4:12]}
    root_waiting = qt_app.generation_queue.queued_clip_ids(root.project_id)
    assert moved.isdisjoint(root_waiting)
    assert qt_app.generation_queue.queued_clip_ids(child.project_id) == moved
    assert qt_app.generation_queue.items[1].project_id == child.project_id  # where those items were

    jobs.resolve(0)  # the running batch: clips 4-7 aren't in the root any more
    qt_app.wait_for_queue()
    assert "failed" not in qt_app.transport_dock.status_text()
    assert jobs.kind(1) == "batch" and set(jobs.ids(1)) == moved
    jobs.resolve(1)
    qt_app.wait_for_queue()
    assert not [c for c in child.document.dirty_clips() if not c.is_nested]
    while qt_app.queue_active:
        qt_app.wait_for_project_io()
        index = len(jobs.jobs) - 1
        if not jobs.jobs[index][1].done():
            jobs.resolve(index)
        qt_app.wait_for_queue()
    assert {c.id for c in clips[:4] + clips[12:]}.isdisjoint(_stale_ids(qt_app))


def test_save_runs_beside_a_generate_and_a_result_that_lands_meanwhile_keeps_it_dirty(qt_app, tmp_path,
                                                                                         monkeypatch):
    _many_clips(qt_app, 4)
    qt_app.save_project_as(str(tmp_path / "book.tbaw"))
    qt_app.wait_for_project_io()
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    release = threading.Event()
    real_write = project_io.write_bundle

    def slow_write(*args, **kwargs):
        release.wait(10)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(project_io, "write_bundle", slow_write)
    qt_app.save_project()
    assert qt_app._io_thread is not None  # the Save started beside the queue

    jobs.resolve(0)  # a result lands while the zip is being written
    QApplication.processEvents()
    qt_app._autosave_one(qt_app.root)
    release.set()
    qt_app.wait_for_project_io()

    assert "Saved" in qt_app.transport_dock.status_text() or not qt_app.is_io_busy()
    assert qt_app.root.dirty  # the bundle doesn't hold that result


def test_save_as_is_refused_while_generating(qt_app, tmp_path):
    # Open and New park the project instead (plan 30,
    # tests/gui_qt/test_cross_project_generation.py).
    _many_clips(qt_app, 2)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    path = qt_app.project_path

    qt_app.save_project_as(str(tmp_path / "elsewhere.tbaw"))

    assert qt_app.project_path == path
    assert not os.path.exists(tmp_path / "elsewhere.tbaw")


def test_an_export_runs_beside_a_generate(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from kokoro_gui.qt.docks.export_dialog import run_export

    # The stale-clips prompt (they export silent): Export anyway.
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Export anyway"))
    _many_clips(qt_app, 2)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    values = {"out_dir": str(tmp_path / "out"), "filename": "mix", "format": "wav", "srt": False,
              "keep_clip_files": False}

    assert run_export(qt_app, values) is True
    assert qt_app.is_io_busy() and qt_app.is_generating()
    assert run_export(qt_app, values) is False  # one export at a time

    jobs.resolve(0)
    qt_app.wait_for_queue()
    assert not qt_app.queue_active and qt_app.is_busy()  # the export still holds the window
    qt_app.exportFinished.emit(True, "Exported.")
    assert not qt_app.is_busy()


# -- still refused ---------------------------------------------------------------------


def test_a_device_change_is_refused_while_generating(qt_app, tmp_path):
    _many_clips(qt_app, 2)
    Jobs(qt_app, tmp_path)
    device = qt_app.settings.get("device", "auto")
    qt_app.on_generate_clicked()
    qt_app.engine.init_pipeline_async.reset_mock()

    qt_app.set_device("cpu")

    assert qt_app.settings.get("device", "auto") == device
    assert not qt_app.device_actions["cpu"].isChecked()
    assert "generate is running" in qt_app.transport_dock.status_text()
    qt_app.engine.init_pipeline_async.assert_not_called()


def test_an_engine_change_is_refused_for_an_engine_the_queue_uses(qt_app, tmp_path):
    _many_clips(qt_app, 2)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    character = qt_app.document.characters[0]
    engine = character.backend_id

    assert qt_app.set_character_engine(character, "dummy") is False
    assert character.backend_id == engine


# -- results that land on a changed clip -------------------------------------------------


def test_a_result_for_a_clip_edited_mid_generate_leaves_it_stale(qt_app, tmp_path):
    clips = _many_clips(qt_app, 3)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    document = qt_app.document
    keys = {cid: document.segment_key_fn(text, document.get_clip(cid)) for cid, text, _c in jobs.jobs[0][0][1]}

    start, _end = document.clip_extent(clips[1].id)
    at = start + len("Clip ")
    document.replace_text(at, len("number"), len("numeral"),
                          document.text[:at] + "numeral" + document.text[at + len("number"):])
    jobs.resolve(0, keys=keys)
    qt_app.wait_for_queue()

    assert clips[1].segments  # stored: a valid cache entry
    assert _stale_ids(qt_app) == {clips[1].id}  # but its key is the old text's
    assert document.clip_text(clips[1]) == "Clip numeral 01 is here."
    assert qt_app.generation_queue.pending() == []  # not regenerated on its own


def test_a_result_for_a_clip_deleted_mid_generate_is_skipped(qt_app, tmp_path):
    clips = _many_clips(qt_app, 3)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.document.clips.remove(clips[2])

    jobs.resolve(0)
    qt_app.wait_for_queue()

    assert qt_app.transport_dock.status_text().startswith("Generated 2 clip(s).")
    assert qt_app.generation_queue.items[0].state == genqueue.DONE


# -- queued and generating marks ------------------------------------------------------


def test_queued_clip_ids_names_the_running_and_queued_clips(qt_app, tmp_path):
    clips = _many_clips(qt_app, 10)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()

    states = qt_app.queued_clip_ids()

    assert {c.id for c in clips[:8]} == {k for k, v in states.items() if v == "running"}
    assert {c.id for c in clips[8:]} == {k for k, v in states.items() if v == "queued"}


def test_the_gutter_reads_the_queue_once_per_paint_and_the_timeline_hatches_queued_blocks(qt_app, tmp_path,
                                                                                         monkeypatch):
    clips = _many_clips(qt_app, 10)
    jobs = Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    qt_app.refresh_timeline()
    view = qt_app.timeline_dock.timeline_view
    assert view._blocks_by_clip_id[clips[0].id].queue_state == "running"
    assert view._blocks_by_clip_id[clips[9].id].queue_state == "queued"

    calls = []
    real = type(qt_app).queued_clip_ids
    monkeypatch.setattr(type(qt_app), "queued_clip_ids", lambda self, *a: calls.append(1) or real(self, *a))
    marks = []
    monkeypatch.setattr(TranscriptGutter, "_draw_queue_mark", staticmethod(lambda p, r, pal, state: marks.append(state)))
    gutter = qt_app.editor.findChild(TranscriptGutter)
    gutter.grab()  # one paint, shown or not
    assert calls == [1]
    # The one line's mark stands where its play button was, and still clicks.
    assert marks == ["running"]
    assert [cid for _rect, cid in gutter._button_rects] == [clips[0].id]

    jobs.resolve(0)
    qt_app.wait_for_queue()
    assert view._blocks_by_clip_id[clips[0].id].queue_state is None
    assert view._blocks_by_clip_id[clips[9].id].queue_state == "running"
