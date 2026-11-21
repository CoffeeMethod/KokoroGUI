"""A finished Save marks a project clean only when the document still
matches the snapshot it wrote (plan 46 F2). The editor stays writable while
the zip is written, so an edit that autosaves meanwhile isn't in the bundle:
it must stay Unsaved, survive a recover and keep its dir from eviction. The
same holds for an embedded subproject and for the Save that runs before a
New or an Open.

Each test holds `write_bundle` on an event, so the edit lands inside the
Save without a timing guess."""
import json
import os
import threading
import zipfile

import pytest

from kokoro_gui.qt import project as project_io
from tests.gui_qt.test_subprojects import _book


@pytest.fixture
def held_save(monkeypatch):
    """`write_bundle` waits while the returned event is clear (it starts set)."""
    release = threading.Event()
    release.set()
    real_write = project_io.write_bundle

    def slow_write(*args, **kwargs):
        release.wait(10)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(project_io, "write_bundle", slow_write)
    return release


def _edit(project, app, gap):
    project.document.settings["gap_s"] = gap
    app.save_settings()  # the debounced autosave runs now


def _bundle_gap(path):
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("document.json")).get("settings", {}).get("gap_s")


def _saved(qt_app, tmp_path):
    qt_app.save_project_as(str(tmp_path / "book.tbaw"))
    qt_app.wait_for_project_io()
    assert not qt_app.any_project_dirty()


def _assert_unsaved_and_recoverable(qt_app, project_dir, path, monkeypatch, gap):
    assert qt_app.root.dirty and qt_app.windowTitle().startswith("book*")
    assert project_io.read_session(project_dir)["dirty"]
    assert _bundle_gap(path) != gap  # the bundle has the snapshot, not the edit

    # A clean close and eviction keep the dir: it holds the only copy.
    qt_app._teardown_project(discard=False)
    assert project_dir not in [os.path.realpath(d) for d in project_io.evict_project_dirs(None)]
    assert os.path.isdir(project_dir)

    # Reopening offers to recover, and keeping the session brings the edit back.
    asked = []
    monkeypatch.setattr(type(qt_app), "_ask_recover_choice", lambda self, *a: asked.append(1) or "keep")
    qt_app.open_project(path)
    qt_app.wait_for_project_io()
    assert asked
    assert qt_app.document.settings["gap_s"] == gap


@pytest.mark.parametrize("how", ["save", "save_as"])
def test_an_edit_made_during_save_stays_unsaved(qt_app, tmp_path, held_save, monkeypatch, how):
    qt_app.document.text = "One. Two."
    qt_app.editor.load_text(qt_app.document.text)
    _saved(qt_app, tmp_path)
    path, project_dir = qt_app.project_path, os.path.realpath(qt_app.project_dir)
    held_save.clear()

    _edit(qt_app.root, qt_app, 0.25)  # Unsaved before the Save, so Save has work
    if how == "save":
        qt_app.save_project()
    else:
        qt_app.save_project_as(path)
    assert qt_app._io_thread is not None
    _edit(qt_app.root, qt_app, 0.75)  # lands while the zip is being written
    held_save.set()
    qt_app.wait_for_project_io()

    assert _bundle_gap(path) == 0.25
    _assert_unsaved_and_recoverable(qt_app, project_dir, path, monkeypatch, 0.75)


def test_a_child_edit_made_during_save_stays_unsaved(qt_app, tmp_path, held_save):
    _intro, _chapter, child = _book(qt_app)
    qt_app.save_project_as(str(tmp_path / "book"))
    qt_app.wait_for_project_io()
    held_save.clear()

    _edit(child, qt_app, 0.25)
    qt_app.save_project_as(qt_app.project_path)
    _edit(child, qt_app, 0.75)
    held_save.set()
    qt_app.wait_for_project_io()

    assert child.dirty and qt_app.any_project_dirty()
    assert project_io.read_session(child.project_dir)["dirty"]
    inner = project_io.load_project(os.path.join(qt_app.project_dir, "projects", f"{child.project_id}.tbaw"))
    assert inner.document.settings["gap_s"] == 0.25


@pytest.mark.parametrize("then", ["new", "open"])
def test_save_before_new_or_open_asks_again_when_an_edit_lands_during_it(qt_app, tmp_path, held_save,
                                                                         monkeypatch, then):
    other = str(tmp_path / "other.tbaw")
    qt_app.save_project_as(other)
    qt_app.wait_for_project_io()
    _saved(qt_app, tmp_path)
    held_save.clear()
    path, project_dir = qt_app.project_path, qt_app.project_dir

    answers = ["save", "cancel"]
    monkeypatch.setattr(type(qt_app), "_ask_close_choice", lambda self: answers.pop(0) if answers else "discard")
    _edit(qt_app.root, qt_app, 0.25)
    if then == "new":
        qt_app.new_project()
    else:
        qt_app.open_project(other)
    _edit(qt_app.root, qt_app, 0.75)  # lands during the Save the close asked for
    held_save.set()
    qt_app.wait_for_project_io()

    assert answers == []  # asked a second time instead of putting the project away
    assert qt_app.project_path == path and qt_app.project_dir == project_dir
    assert qt_app.root.dirty and project_io.read_session(project_dir)["dirty"]
    assert qt_app.document.settings["gap_s"] == 0.75
    assert _bundle_gap(path) == 0.25
