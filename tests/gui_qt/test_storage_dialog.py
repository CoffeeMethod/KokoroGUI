"""Options > Storage... and the cache trim (kokoro_gui/qt/storage_dialog.py,
plan 37). Everything runs in the tmp dir the qt_app fixture changes into."""
import os

import pytest
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.engine import cache_admin, runtime
from kokoro_gui.engines import audio8_tts
from kokoro_gui.qt import storage_dialog

KEY_A = "a" * 64
KEY_B = "b" * 64
MIB = cache_admin.MIB


def _write(path, size=100, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def stored(qt_app):
    """Segment files, a project working copy, a log, and two reference codes."""
    root = os.path.abspath(runtime.CACHE_DIR)
    refs = os.path.abspath(audio8_tts._ref_codes_cache_dir())
    files = {
        "seg_a": _write(os.path.join(root, f"{KEY_A}_0.wav"), 100, mtime=1000),
        "seg_b": _write(os.path.join(root, f"{KEY_B}_0.wav"), 200, mtime=2000),
        "doc": _write(os.path.join(root, "projects", "p1", "document.json"), 300),
        "log": _write(os.path.join(root, "logs", "kokorogui.log"), 20),
        "ref": _write(os.path.join(refs, "0123456789abcdef.npy"), 64),
    }
    return files


@pytest.fixture
def dialog(qt_app, stored):
    dlg = qt_app.open_storage_dialog()
    dlg.wait_idle()
    yield dlg
    dlg.wait_idle()
    dlg.reject()


def _answer(monkeypatch, button):
    asked = []

    def _question(parent, title, text, *a, **k):
        asked.append((title, text))
        return button

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_question))
    return asked


# -- the menu entry and the sizes --------------------------------------------------


def test_the_options_menu_has_storage(qt_app):
    assert qt_app.storage_action in qt_app.options_menu.actions()
    assert qt_app.storage_action.text() == "Storage..."


def test_sizes_for_the_three_rows_are_measured_on_open(dialog):
    assert dialog.size_labels["segments"].text() == "2 files, 300 B"
    assert dialog.size_labels["refs"].text() == "1 file, 64 B"
    # The open Untitled project has a working copy of its own in there too.
    projects = cache_admin.projects_usage()
    assert projects.count >= 1 and projects.bytes >= 300
    assert dialog.size_labels["projects"].text() == cache_admin.describe(projects)


def test_project_working_copies_have_no_clear_button(dialog):
    assert set(dialog.clear_buttons) == {"segments", "refs"}
    assert dialog.clear("projects") is False


# -- Clear -------------------------------------------------------------------------


def test_clear_asks_with_the_size_then_removes_only_segment_files(dialog, stored, monkeypatch):
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    assert dialog.clear("segments") is True
    dialog.wait_idle()
    dialog.wait_idle()

    assert "2 files, 300 B" in asked[0][1]
    assert not os.path.exists(stored["seg_a"]) and not os.path.exists(stored["seg_b"])
    for kept in ("doc", "log", "ref"):
        assert os.path.exists(stored[kept])
    assert dialog.size_labels["segments"].text() == "0 files, 0 B"
    assert "Cleared 2 files, 300 B" in dialog.note.text()


def test_clear_does_nothing_when_the_user_cancels(dialog, stored, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)

    assert dialog.clear("segments") is False

    assert os.path.exists(stored["seg_a"]) and os.path.exists(stored["seg_b"])


def test_clear_reference_codes_leaves_the_segment_cache(dialog, stored, monkeypatch):
    _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    assert dialog.clear("refs") is True
    dialog.wait_idle()
    dialog.wait_idle()

    assert not os.path.exists(stored["ref"])
    assert os.path.exists(stored["seg_a"]) and os.path.exists(stored["doc"])


def test_clear_is_refused_while_a_job_runs(qt_app, dialog, stored, monkeypatch):
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    qt_app.transport_dock.set_busy(True)
    dialog._sync_buttons()

    assert not dialog.clear_buttons["segments"].isEnabled()
    assert dialog.clear("segments") is False
    assert asked == []
    assert os.path.exists(stored["seg_a"])


def test_clear_with_nothing_to_clear_says_so(dialog, stored, monkeypatch):
    os.remove(stored["ref"])
    dialog.refresh()
    dialog.wait_idle()
    asked = _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    assert dialog.clear("refs") is False
    assert asked == [] and "Nothing to clear" in dialog.note.text()


# -- the limit ---------------------------------------------------------------------


def test_the_limit_defaults_to_2048_mb_and_is_saved_in_the_program_settings(qt_app, dialog):
    assert dialog.limit_spin.value() == 2048
    dialog.limit_spin.setValue(512)
    assert qt_app.settings["segment_cache_max_mb"] == 512
    assert "segment_cache_max_mb" not in qt_app.document.settings


def test_zero_reads_as_no_limit(dialog):
    dialog.limit_spin.setValue(0)
    assert dialog.limit_spin.text() == "No limit"


def test_applying_a_limit_trims_the_oldest_key(qt_app, dialog, stored):
    _write(stored["seg_a"], MIB, mtime=1000)
    _write(stored["seg_b"], MIB, mtime=2000)

    dialog.limit_spin.setValue(1)
    dialog.apply_limit()
    dialog.wait_idle()
    dialog.wait_idle()

    assert not os.path.exists(stored["seg_a"])
    assert os.path.exists(stored["seg_b"])
    assert os.path.exists(stored["doc"]) and os.path.exists(stored["log"])
    assert "Removed 1 file" in dialog.note.text()


def test_a_limit_of_zero_trims_nothing(dialog, stored):
    dialog.limit_spin.setValue(0)
    dialog.apply_limit()
    dialog.wait_idle()
    assert os.path.exists(stored["seg_a"]) and os.path.exists(stored["seg_b"])


# -- the trim at launch and after a generate -----------------------------------------


def test_the_app_trims_after_a_generate_ends(qt_app, stored):
    _write(stored["seg_a"], MIB, mtime=1000)
    _write(stored["seg_b"], MIB, mtime=2000)
    qt_app.settings["segment_cache_max_mb"] = 1

    qt_app.on_engine_finish()
    qt_app.wait_for_cache_trim()

    assert not os.path.exists(stored["seg_a"])
    assert os.path.exists(stored["seg_b"])
    assert os.path.exists(stored["doc"]) and os.path.exists(stored["log"]) and os.path.exists(stored["ref"])


def test_no_limit_means_no_trim_thread(qt_app, stored):
    qt_app.settings["segment_cache_max_mb"] = 0
    qt_app._cache_trim_thread = None

    qt_app.trim_segment_cache_async()

    assert qt_app._cache_trim_thread is None


def test_a_failing_trim_is_reported_not_raised(tmp_path, monkeypatch):
    lines = []

    def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(cache_admin, "trim_to_setting", _boom)

    thread = storage_dialog.trim_in_background(5, str(tmp_path), report=lines.append)
    thread.join(5)

    assert lines and "disk gone" in lines[0]


def test_the_default_limit_is_a_program_setting():
    from kokoro_gui.qt import spec

    assert spec.SETTINGS_DEFAULTS["segment_cache_max_mb"] == 2048


def test_about_shows_the_cache_size(qt_app, stored):
    from kokoro_gui.qt import about_dialog

    assert "Generated audio cache: 2 files, 300 B" in about_dialog.about_text("config_qt.json")
