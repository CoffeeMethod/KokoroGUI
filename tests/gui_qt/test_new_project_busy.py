"""File > New, Ctrl+N, New from Template and an Import Text "new" replace the
open project, so they are refused while a generate runs or project IO is busy,
the way Open and Save As are (plan 46, F4). Nothing about the open project
moves: not its lock, directory, document or id."""
import os

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.qt import project as project_io
from tests.gui_qt.test_edit_while_generating import Jobs, _many_clips


def _snapshot(qt_app):
    root = qt_app.root
    return (root.lock, root.project_dir, root.document, root.project_id, qt_app.project_path)


def _arm_generate(qt_app, tmp_path):
    _many_clips(qt_app, 2)
    Jobs(qt_app, tmp_path)
    qt_app.on_generate_clicked()
    assert qt_app.is_generating()


def _arm_io(qt_app, tmp_path):
    _many_clips(qt_app, 2)
    qt_app.begin_side_job()
    assert qt_app.is_io_busy()


def _watch_close(qt_app, monkeypatch):
    """Records a close or teardown, then lets it run (the fixture's own
    closeEvent uses them too)."""
    calls = []
    for name in ("_close_current_project", "_teardown_project"):
        real = getattr(qt_app, name)
        monkeypatch.setattr(qt_app, name, lambda *a, _r=real, _n=name, **k: (calls.append(_n), _r(*a, **k))[1])
    return calls


@pytest.mark.parametrize("arm", [_arm_generate, _arm_io])
@pytest.mark.parametrize("how", ["new", "shortcut", "template", "import_new"])
def test_replacing_the_project_is_refused_while_busy(qt_app, tmp_path, monkeypatch, arm, how):
    arm(qt_app, tmp_path)
    before = _snapshot(qt_app)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    calls = _watch_close(qt_app, monkeypatch)

    if how == "new":
        qt_app.new_project()
    elif how == "shortcut":
        assert qt_app.new_action.shortcut() == QKeySequence(QKeySequence.StandardKey.New)
        qt_app.new_action.trigger()
    elif how == "template":
        from kokoro_gui.daw import templates
        qt_app.new_from_template(templates.Template(name="t", settings={}, characters=[], sections=[]))
    else:
        text_before = qt_app.document.text
        qt_app._finish_import_text("x.txt", "Hello there.", "new")
        assert qt_app.document.text == text_before

    assert _snapshot(qt_app) == before
    assert warned and not calls
    assert os.path.isdir(qt_app.root.project_dir)


def test_a_whole_document_generate_carries_the_projects_dir(qt_app, monkeypatch):
    """Plan 46, F3: the config the whole-document and JIT paths resolve a
    bundled voice from names the open project's directory."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(qt_app.engine, "pipeline", True)
    monkeypatch.setattr(qt_app.backend, "convert_document", MagicMock(), raising=False)
    qt_app.editor.setPlainText("hello")
    qt_app.start_conversion()

    config = qt_app.backend.convert_document.call_args[0][1]
    assert config["project_dir"] == qt_app.project_dir
    qt_app._converting = False
    qt_app.set_ui_state(False)
