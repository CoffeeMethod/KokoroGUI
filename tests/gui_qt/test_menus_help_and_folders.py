"""File > Show in Folder, File > Recent > Clear list, and the Help menu
(plan 06)."""
import os

import pytest

import kokoro_gui.qt.app as app_module
from kokoro_gui.qt import project as project_io


@pytest.fixture
def revealed(monkeypatch):
    paths = []
    monkeypatch.setattr(app_module, "reveal", lambda path: paths.append(path) or True)
    return paths


# -- File > Show in Folder --------------------------------------------------------------


def test_show_in_folder_entries_are_disabled_until_their_target_exists(qt_app):
    qt_app.project_path = None
    qt_app._sync_show_in_folder_actions()
    assert not qt_app.show_project_file_action.isEnabled()
    assert not qt_app.show_last_export_action.isEnabled()


def test_show_project_file_reveals_the_saved_project(qt_app, tmp_path, revealed):
    path = tmp_path / "story.tbaw"
    qt_app.save_project_as(str(path))
    qt_app.wait_for_project_io()
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_project_file_action.isEnabled()

    qt_app.show_project_file_action.trigger()

    assert revealed == [qt_app.project_path]


def test_show_working_folder_reveals_the_root_project_dir(qt_app, tmp_path, revealed):
    qt_app.root.project_dir = str(tmp_path)
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_working_folder_action.isEnabled()

    qt_app.show_working_folder_action.trigger()

    assert revealed == [str(tmp_path)]


def test_last_export_is_remembered_when_an_export_finishes(qt_app, tmp_path, revealed):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"x")
    qt_app.exportWrote.emit(str(mix))
    qt_app.exportFinished.emit(True, f"Exported {mix}")
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert qt_app.show_last_export_action.isEnabled()

    qt_app.show_last_export_action.trigger()

    assert revealed == [str(mix)]


def test_a_vanished_export_disables_its_entry_on_the_next_open(qt_app, tmp_path):
    mix = tmp_path / "mix.wav"
    mix.write_bytes(b"x")
    qt_app.exportWrote.emit(str(mix))
    mix.unlink()
    qt_app.show_in_folder_menu.aboutToShow.emit()
    assert not qt_app.show_last_export_action.isEnabled()


def test_revealing_a_missing_path_says_so_in_the_status_line(qt_app, tmp_path):
    qt_app._reveal_path(str(tmp_path / "gone.wav"))
    assert "gone" in qt_app.transport_dock.status_text()


# -- File > Recent > Clear list ---------------------------------------------------------


def test_recent_menu_has_no_clear_entry_when_empty(qt_app):
    qt_app.settings["recent_projects"] = []
    qt_app._rebuild_recent_menu()
    assert [a.text() for a in qt_app.recent_menu.actions()] == ["(empty)"]


def test_recent_clear_list_empties_the_menu_and_the_setting(qt_app, tmp_path):
    project_io.remember_recent(qt_app.settings, str(tmp_path / "one.tbaw"))
    project_io.remember_recent(qt_app.settings, str(tmp_path / "two.tbaw"))
    qt_app._rebuild_recent_menu()
    actions = qt_app.recent_menu.actions()
    assert actions[-1].text() == "Clear list"
    assert actions[-2].isSeparator()

    actions[-1].trigger()

    assert qt_app.settings["recent_projects"] == []
    assert [a.text() for a in qt_app.recent_menu.actions()] == ["(empty)"]


# -- Help menu, About, shortcut sheet --------------------------------------------------


def test_help_menu_lists_documentation_shortcuts_and_about(qt_app):
    texts = [a.text() for a in qt_app.help_menu.actions()]
    assert texts == ["&Documentation", "&Keyboard Shortcuts", "", "&About KokoroGUI"]
    assert qt_app.help_menu.actions()[2].isSeparator()
    assert [a.text() for a in qt_app.menuBar().actions()][-1] == "&Help"


def test_documentation_opens_the_docs_site(qt_app, monkeypatch):
    opened = []
    monkeypatch.setattr(app_module.QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url.toString()) or True))
    qt_app.documentation_action.trigger()
    assert opened == [app_module.DOCS_URL]
    assert app_module.DOCS_URL.startswith("https://coffeemethod.github.io/KokoroGUI")


def test_shortcut_sheet_reads_menu_actions_and_loose_shortcuts(qt_app):
    dialog = qt_app.show_shortcuts()
    text = dialog.text()
    assert "Ctrl+S" in text
    assert "Space" in text
    assert "Play / pause" in text
    assert text.index("File") < text.index("Ctrl+S")


def test_shortcut_sheet_picks_up_a_new_shortcut_attribute(qt_app):
    from PySide6.QtGui import QKeySequence, QShortcut

    qt_app.my_new_shortcut = QShortcut(QKeySequence("Ctrl+Alt+J"), qt_app)
    assert "Ctrl+Alt+J" in qt_app.show_shortcuts().text()


def test_about_lists_versions_engines_paths_and_whisper(qt_app, monkeypatch):
    import kokoro_gui
    from kokoro_gui.engine import asr
    from kokoro_gui.qt import about_dialog

    monkeypatch.setattr(asr, "whisper_model_cached", lambda *a, **k: True)
    monkeypatch.setattr(about_dialog, "device_summary", lambda: "CPU")
    text = qt_app.show_about().text()
    assert kokoro_gui.APP_VERSION in text
    assert "PySide6" in text and "torch" in text
    assert "Device: CPU" in text
    assert "kokoro" in text.lower()
    assert "Cache folder:" in text and "Custom voices folder:" in text
    assert "config_qt.json" in text
    assert f"Whisper model: {asr.get_whisper_model_name()} (cached)" in text


def test_about_names_the_reason_an_engine_is_unavailable(qt_app, monkeypatch):
    from kokoro_gui.engines import registry
    from kokoro_gui.qt import about_dialog

    monkeypatch.setattr(registry, "unavailable_reason", lambda engine_id: "needs a package")
    assert "unavailable. needs a package" in about_dialog.about_text("config_qt.json")


def test_about_copy_button_puts_the_text_on_the_clipboard(qt_app):
    from PySide6.QtGui import QGuiApplication

    dialog = qt_app.show_about()
    dialog.copy_button.click()
    assert QGuiApplication.clipboard().text() == dialog.text()


# -- device line -------------------------------------------------------------------------


def _fake_torch(monkeypatch, cuda=False, mps=False, name="RTX Test"):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: name)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_device_summary_names_cuda_mps_or_cpu(monkeypatch):
    from kokoro_gui.qt.about_dialog import device_summary

    _fake_torch(monkeypatch, cuda=True)
    assert device_summary() == "CUDA (RTX Test)"
    _fake_torch(monkeypatch, mps=True)
    assert device_summary() == "MPS (Apple)"
    _fake_torch(monkeypatch)
    assert device_summary() == "CPU"


def test_device_summary_without_torch_is_cpu(monkeypatch):
    import builtins

    from kokoro_gui.qt.about_dialog import device_summary

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device_summary() == "CPU (torch not installed)"


def test_device_menu_starts_with_a_disabled_detected_line(qt_app):
    first = qt_app.device_menu.actions()[0]
    assert first.text().startswith("Detected: ")
    assert not first.isEnabled()
    assert set(qt_app.device_actions) == {"auto", "cpu", "cuda"}


def test_device_notice_names_cuda_and_how_to_change_it(qt_app, monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    notice = qt_app.device_notice()
    assert "Engines will run on CUDA (RTX Test)" in notice
    assert "Options > Device" in notice


def test_device_notice_for_mps_says_engines_use_the_cpu(qt_app, monkeypatch):
    _fake_torch(monkeypatch, mps=True)
    notice = qt_app.device_notice()
    assert "MPS (Apple)" in notice and "CPU" in notice


def test_device_notice_follows_a_cpu_choice(qt_app, monkeypatch):
    _fake_torch(monkeypatch, cuda=True)
    qt_app.settings["device"] = "cpu"
    assert qt_app.device_notice().startswith("Engines will run on the CPU")


def test_first_launch_shows_the_device_notice_once(qt_app):
    # The fixture's launch already showed it.
    assert qt_app.settings["device_notice_shown"] is True
    assert qt_app._first_launch_device_notice() is None
    qt_app.settings["device_notice_shown"] = False
    assert "Options > Device" in qt_app._first_launch_device_notice()
    assert qt_app.settings["device_notice_shown"] is True
