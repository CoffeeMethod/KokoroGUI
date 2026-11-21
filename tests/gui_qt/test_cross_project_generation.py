"""Generation goes on after another project opens (plan 30, grill WF11,
PG13, PG14): Open and New park a project the running queue still has work
for instead of closing it. Its results land in its own document and project
dir, its queue in its own `session.json`, and it is released (autosave, GC,
unlock) when its last item ends. Opening it again while it is parked brings
back the same objects. The Queue dock greys its items and the progress line
reads "<name>: done/total in background" while one runs.

The engine is a stub whose every batch is a Future the test resolves; a
resolved clip's file is written where the real engine writes it, the
`out_dir` of the clip's config."""
import concurrent.futures
import json
import os
import zipfile

import numpy as np
import soundfile as sf

import kokoro_gui.qt.app  # noqa: F401 (the docks package imports it back)
from kokoro_gui.daw import genqueue
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt.docks.queue_dock import ELSEWHERE
from tests.gui_qt.test_generation_queue import _many_clips


class Engine:
    def __init__(self, qt_app):
        self.qt_app = qt_app
        self.futures = []
        qt_app.engine.worker.run_coro.side_effect = self._run

    def _run(self, _coro):
        future = concurrent.futures.Future()
        self.futures.append(future)
        return future

    def group(self, index):
        return self.qt_app.engine.generate_dirty_clips.call_args_list[index][0][0]

    def resolve(self, index, cancelled=False, fail=False):
        outcomes = []
        for clip_id, text, config in self.group(index):
            if cancelled or fail:
                outcomes.append({"clip_id": clip_id, "success": False, "results": [], "error": "boom",
                                 "cancelled": cancelled})
                continue
            os.makedirs(config["out_dir"], exist_ok=True)
            path = os.path.join(config["out_dir"], f"{clip_id}.wav")
            sf.write(path, np.full(2400, 0.2, dtype=np.float32), 24000)
            outcomes.append({"clip_id": clip_id, "success": True, "error": "", "cancelled": False,
                             "results": [{"path": path, "text": text, "duration": 0.1, "seg_idx": 0}]})
        self.futures[index].set_result(outcomes)
        self.qt_app.wait_for_queue()


def _two_books(qt_app, tmp_path, count=16):
    """book_b.tbaw (two stale clips) on disk, then book_a.tbaw open and
    saved with `count` stale clips. Returns A's path and B's."""
    _many_clips(qt_app, 2)
    path_b = str(tmp_path / "book_b.tbaw")
    qt_app.save_project_as(path_b)
    qt_app.wait_for_project_io()
    qt_app.new_project()
    _many_clips(qt_app, count)
    path_a = str(tmp_path / "book_a.tbaw")
    qt_app.save_project_as(path_a)
    qt_app.wait_for_project_io()
    assert not qt_app.is_project_dirty()
    return path_a, path_b


def _start_and_open_b(qt_app, tmp_path):
    path_a, path_b = _two_books(qt_app, tmp_path)
    engine = Engine(qt_app)
    qt_app.on_generate_clicked()
    assert qt_app.queue_active
    a = qt_app.root
    qt_app.open_project(path_b)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == os.path.abspath(path_b)
    return engine, a, path_a, path_b


def _document_json(project_dir):
    with open(os.path.join(project_dir, project_io.DOCUMENT), encoding="utf-8") as f:
        return json.load(f)


# -- parking -----------------------------------------------------------------------


def test_open_parks_a_generating_project_and_its_results_reach_its_own_files(qt_app, tmp_path, monkeypatch):
    engine, a, path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    a_dir = a.project_dir

    # Not torn down: same object, its lock held, its dir kept.
    assert qt_app.parked_projects[a.project_id].root is a
    assert a.lock is not None and project_io.is_locked(a_dir)
    assert os.path.isfile(os.path.join(a_dir, project_io.DOCUMENT))
    assert qt_app.root is not a and qt_app.root.project_dir != a_dir

    engine.resolve(0)
    # The first batch landed in A while B is open; A is still parked.
    assert len(a.document.dirty_clips()) == 8
    assert qt_app.parked_projects
    engine.resolve(1)

    assert a.document.dirty_clips() == []
    on_disk = _document_json(a_dir)
    assert all(clip["segments"] for clip in on_disk["clips"])
    assert all(os.path.isfile(os.path.join(a_dir, project_io.AUDIO_GENERATED, f"{c.id}.wav"))
               for c in a.document.clips)
    # The last item ended: released, and its results wait in its dir.
    assert not qt_app.parked_projects
    assert a.lock is None and not project_io.is_locked(a_dir)
    assert os.path.isdir(a_dir)
    assert not qt_app.is_busy()
    # B saw none of it.
    assert len(qt_app.document.dirty_clips()) == 2

    # Opening A again offers the unsaved results back, and its clips are clean.
    monkeypatch.setattr(type(qt_app), "_ask_recover_choice", lambda self, *a: "keep")
    qt_app.open_project(path_a)
    qt_app.wait_for_project_io()
    assert qt_app.project_path == os.path.abspath(path_a)
    assert len(qt_app.document.clips) == 16
    assert qt_app.document.dirty_clips() == []


def test_a_project_with_no_work_left_closes_as_before(qt_app, tmp_path):
    path_a, path_b = _two_books(qt_app, tmp_path)
    engine = Engine(qt_app)
    qt_app.on_generate_clicked()
    # B's items run after A's, and B is the one left with nothing to do.
    a = qt_app.root
    qt_app.open_project(path_b)
    qt_app.wait_for_project_io()
    qt_app.on_generate_clicked()
    assert [i.project_id for i in qt_app.generation_queue.pending()] == [a.project_id] * 2 + [qt_app.project_id]

    qt_app.new_project()

    # B had queued work, so it is parked too; the new Untitled one is open.
    assert len(qt_app.parked_projects) == 2 and qt_app.project_path is None
    for index in range(3):
        engine.resolve(index)
    assert not qt_app.parked_projects and not qt_app.is_busy()


def test_new_parks_the_generating_project(qt_app, tmp_path):
    path_a, _path_b = _two_books(qt_app, tmp_path)
    engine = Engine(qt_app)
    qt_app.on_generate_clicked()
    a = qt_app.root

    qt_app.new_project()

    assert qt_app.project_path is None and qt_app.root is not a
    assert a.project_id in qt_app.parked_projects
    engine.resolve(0)
    engine.resolve(1)
    assert a.document.dirty_clips() == [] and not qt_app.parked_projects


def test_reopening_a_parked_project_brings_back_the_same_objects(qt_app, tmp_path, monkeypatch):
    engine, a, path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_recover_choice", lambda self, *x: asked.append(1) or "keep")

    qt_app.open_project(path_a)

    assert qt_app.root is a and qt_app.document is a.document
    assert not qt_app.parked_projects and not asked
    assert qt_app.queue_active
    engine.resolve(0)
    engine.resolve(1)
    assert qt_app.document.dirty_clips() == []
    assert a.lock is not None  # open again, so it keeps its lock


def test_a_copy_with_the_same_project_id_waits_for_the_parked_one(qt_app, tmp_path):
    import shutil

    engine, a, path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    copy = str(tmp_path / "copy_of_a.tbaw")
    shutil.copy(path_a, copy)
    path_b = qt_app.project_path

    qt_app.open_project(copy)

    assert qt_app.project_path == path_b
    assert a.project_id in qt_app.parked_projects


def test_a_parked_projects_subproject_item_generates_and_renders_in_the_background(qt_app, tmp_path):
    from tests.gui_qt.test_subprojects import _generated_clip

    _many_clips(qt_app, 2)
    path_b = str(tmp_path / "book_b.tbaw")
    qt_app.save_project_as(path_b)
    qt_app.wait_for_project_io()
    qt_app.new_project()
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
    qt_app.save_project_as(str(tmp_path / "book_a.tbaw"))
    qt_app.wait_for_project_io()
    engine = Engine(qt_app)
    a = qt_app.root
    qt_app.generate_subproject(qt_app.document.get_clip(child.clip_id))
    assert len(engine.futures) == 1

    qt_app.open_project(path_b)
    qt_app.wait_for_project_io()

    assert qt_app.parked_projects[a.project_id].children[child.project_id] is child
    assert qt_app.children == {}
    engine.resolve(0)
    assert len(engine.futures) == 2  # the child's second batch, still in the background
    engine.resolve(1)
    qt_app.wait_for_queue()

    assert child.document.dirty_clips() == []
    assert project_io.read_mixdown_info(child.project_dir) is not None
    assert not qt_app.parked_projects and child.lock is None and a.lock is None
    assert qt_app.project_path == os.path.abspath(path_b)


# -- the close prompt --------------------------------------------------------------


def test_discard_is_refused_while_the_project_still_generates(qt_app, tmp_path):
    path_a, path_b = _two_books(qt_app, tmp_path)
    Engine(qt_app)
    qt_app.on_generate_clicked()
    a = qt_app.root
    a.document.settings["edited"] = True
    qt_app.schedule_save()
    assert qt_app.is_project_dirty()  # the fixture answers Discard

    qt_app.open_project(path_b)
    qt_app.wait_for_project_io()

    assert qt_app.root is a and not qt_app.parked_projects
    assert os.path.isdir(a.project_dir) and a.lock is not None


def test_save_then_park(qt_app, tmp_path, monkeypatch):
    path_a, path_b = _two_books(qt_app, tmp_path)
    engine = Engine(qt_app)
    qt_app.on_generate_clicked()
    a = qt_app.root
    a.document.settings["edited"] = True
    qt_app.schedule_save()
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: "save")

    qt_app.open_project(path_b)
    qt_app.wait_for_project_io()

    assert qt_app.project_path == os.path.abspath(path_b)
    assert a.project_id in qt_app.parked_projects
    with zipfile.ZipFile(path_a) as zf:
        assert json.loads(zf.read(project_io.DOCUMENT))["settings"]["edited"] is True
    engine.resolve(0)
    engine.resolve(1)
    assert a.document.dirty_clips() == [] and not qt_app.parked_projects


def test_an_untitled_project_with_work_is_asked_about_even_when_clean(qt_app, tmp_path, monkeypatch):
    _path_a, path_b = _two_books(qt_app, tmp_path)
    qt_app.new_project()
    _many_clips(qt_app, 4)
    Engine(qt_app)
    qt_app.on_generate_clicked()
    a = qt_app.root
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: asked.append(1) or "cancel")
    # Clean or not, its results need a file to be saved to.
    monkeypatch.setattr(type(qt_app), "is_project_dirty", lambda self: False)

    qt_app.open_project(path_b)

    assert asked and qt_app.root is a and not qt_app.parked_projects


# -- the queue's session.json --------------------------------------------------------


def test_a_parked_projects_queue_goes_to_its_own_session(qt_app, tmp_path):
    engine, a, path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    b_dir = qt_app.project_dir

    saved = project_io.read_session(a.project_dir)["queue"]["items"]
    assert [i["project_id"] for i in saved] == [a.project_id] * 2
    assert "queue" not in (project_io.read_session(b_dir) or {})

    qt_app.pause_queue()
    engine.resolve(0)
    # Paused with A's second item queued: A stays parked, the queue idle.
    assert not qt_app.queue_active and a.project_id in qt_app.parked_projects
    assert len(project_io.read_session(a.project_dir)["queue"]["items"]) == 1

    qt_app.close()

    # Quit releases it; its queue waits in its own session for the next Open.
    assert a.lock is None
    assert len(project_io.read_session(a.project_dir)["queue"]["items"]) == 1
    assert "queue" not in (project_io.read_session(b_dir) or {})


def test_the_opened_projects_saved_queue_waits_for_the_background_run(qt_app, tmp_path):
    engine, a, _path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    b_dir = qt_app.project_dir
    # B's session holds a queue an earlier run left (restored on Open).
    left = {"id": "left0", "kind": genqueue.CLIPS, "project_id": qt_app.project_id,
            "clip_ids": [c.id for c in qt_app.document.clips], "title": "B", "state": genqueue.QUEUED}
    session = project_io.read_session(b_dir)
    session["queue"] = {"items": [left]}
    project_io.write_session(b_dir, session)

    qt_app.restore_queue(project_io.read_session(b_dir))

    # It neither runs nor is lost while A's run goes on.
    assert "left0" not in {i.id for i in qt_app.generation_queue.items}
    assert [i["id"] for i in project_io.read_session(b_dir)["queue"]["items"]] == ["left0"]
    engine.resolve(0)
    engine.resolve(1)

    # A's run ended: B's queue is back, paused, as on Open.
    assert not qt_app.queue_active and qt_app.queue_paused and not qt_app.parked_projects
    assert [i.id for i in qt_app.generation_queue.pending()] == ["left0"]
    assert "were still queued" in qt_app.transport_dock.status_text()
    assert a.lock is None


def test_quitting_with_parked_work_asks_and_cancels_it_all(qt_app, tmp_path, monkeypatch):
    engine, a, _path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_cancel_generate_to_quit", lambda self: asked.append(1) or True)

    qt_app.close()
    assert asked and qt_app.isVisible() is not None  # the close waits for the cancel
    assert all(i.state in (genqueue.CANCELLED, genqueue.RUNNING) for i in qt_app.generation_queue.items)
    engine.resolve(0, cancelled=True)
    qt_app.wait_for_queue()

    assert qt_app._closed
    assert not qt_app.parked_projects and a.lock is None


# -- the indicator (grill PG13) --------------------------------------------------------


def test_the_queue_dock_greys_a_parked_projects_items_and_the_progress_line_names_it(qt_app, tmp_path):
    engine, a, _path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    dock = qt_app.queue_dock
    dock.rebuild()

    rows = [dock.tree.topLevelItem(i) for i in range(dock.tree.topLevelItemCount())]
    assert rows and all(row.text(0).endswith(ELSEWHERE) for row in rows)
    assert all("book_a" in row.toolTip(0) for row in rows)

    qt_app.timeline_dock.batchGenerationProgress.emit(3, 8, "x")
    assert qt_app.transport_dock.detail_text() == "book_a: 3/16 in background"
    dock.rebuild()
    assert dock.summary.text() == "book_a: 3/16 in background"

    # B's own work is not greyed, and its progress reads as before.
    qt_app.on_generate_clicked()
    dock.rebuild()
    last = dock.tree.topLevelItem(dock.tree.topLevelItemCount() - 1)
    assert not last.text(0).endswith(ELSEWHERE)

    engine.resolve(0)
    engine.resolve(1)
    qt_app.timeline_dock.batchGenerationProgress.emit(1, 2, "x")
    assert "in background" not in qt_app.transport_dock.detail_text()
    engine.resolve(2)
    assert "in background" not in qt_app.transport_dock.detail_text()
    assert not qt_app.is_busy()


def test_the_background_line_clears_when_the_run_ends(qt_app, tmp_path):
    engine, _a, _path_a, _path_b = _start_and_open_b(qt_app, tmp_path)
    engine.resolve(0)
    qt_app.timeline_dock.batchGenerationProgress.emit(4, 8, "x")
    assert "in background" in qt_app.transport_dock.detail_text()

    engine.resolve(1)

    assert "in background" not in qt_app.transport_dock.detail_text()
