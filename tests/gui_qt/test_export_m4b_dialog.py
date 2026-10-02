"""The Export dialog's M4B format (plan 18): hidden without ffmpeg, its bitrate,
the Tags tab for it and an export through a mocked ffmpeg. kokoro_gui/qt/docks/
export_dialog.py with kokoro_gui/daw/m4b.py."""
import os
import subprocess

import pytest

import kokoro_gui.qt.app  # noqa: F401 - app.py must load before any docks module (circular import)
from kokoro_gui.daw import m4b, tagging
from kokoro_gui.qt.docks import export_dialog
from kokoro_gui.qt.docks.export_dialog import (
    ExportDialog, export_defaults, export_formats, run_export, tag_options, tags_usable,
)
from tests.daw.test_m4b import FAKE, FakeFfmpeg
from tests.gui_qt.test_export_and_characters import _generated_clip, _type
from tests.gui_qt.test_export_presets_dialog import _scheduled_result, _two_subprojects


def _formats(dialog):
    return [dialog.format_combo.itemText(i) for i in range(dialog.format_combo.count())]


def _with_ffmpeg(monkeypatch, fail=None):
    fake = FakeFfmpeg(fail=fail)
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(m4b.subprocess, "run", fake)
    return fake


def test_m4b_is_in_the_format_list_only_when_ffmpeg_is_found(qt_app, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: None)
    assert export_formats() == ("wav", "mp3", "flac", "ogg") and _formats(ExportDialog(qt_app)) == list(export_formats())

    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    dialog = ExportDialog(qt_app)
    assert _formats(dialog) == ["wav", "mp3", "flac", "ogg", "m4b"]
    assert "ffmpeg" in dialog.format_combo.itemData(4, export_dialog.Qt.ItemDataRole.ToolTipRole)


def test_a_stored_m4b_falls_back_to_wav_while_ffmpeg_is_missing(qt_app, monkeypatch):
    qt_app.project_settings["export"] = {"format": "m4b"}
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: None)
    assert ExportDialog(qt_app).format_combo.currentText() == "wav"

    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    assert ExportDialog(qt_app).format_combo.currentText() == "m4b"


def test_the_m4b_bitrate_shows_for_m4b_alone_and_persists(qt_app, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    dialog = ExportDialog(qt_app)
    assert [dialog.m4b_bitrate_combo.itemData(i) for i in range(dialog.m4b_bitrate_combo.count())] == [64, 96, 128]
    assert dialog.m4b_bitrate_combo.currentData() == 96
    assert dialog.audio_form.isRowVisible(dialog.m4b_bitrate_combo) is False

    dialog.format_combo.setCurrentText("m4b")
    assert dialog.audio_form.isRowVisible(dialog.m4b_bitrate_combo) is True
    assert dialog.audio_form.isRowVisible(dialog.bitrate_combo) is False
    dialog.m4b_bitrate_combo.setCurrentIndex(dialog.m4b_bitrate_combo.findData(64))
    qt_app.project_settings["export"] = dialog.values()

    again = ExportDialog(qt_app)
    assert again.format_combo.currentText() == "m4b" and again.m4b_bitrate_combo.currentData() == 64
    dialog.format_combo.setCurrentText("mp3")
    assert dialog.audio_form.isRowVisible(dialog.m4b_bitrate_combo) is False
    assert dialog.audio_form.isRowVisible(dialog.bitrate_combo) is True


def test_a_bad_stored_m4b_bitrate_goes_back_to_the_default(qt_app):
    qt_app.project_settings["export"] = {"m4b_bitrate_kbps": 9999}
    assert export_defaults(qt_app)["m4b_bitrate_kbps"] == 96
    qt_app.project_settings["export"] = {"m4b_bitrate_kbps": "96"}
    assert export_defaults(qt_app)["m4b_bitrate_kbps"] == 96


def test_the_tags_tab_explains_that_an_m4b_always_has_chapters(qt_app, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    dialog = ExportDialog(qt_app)

    dialog.format_combo.setCurrentText("m4b")

    assert dialog.tags_check.isEnabled() and dialog.tag_artist_edit.isEnabled() and dialog.cover_edit.isEnabled()
    assert not dialog.tag_chapters_check.isEnabled()
    assert "always carries its chapters" in dialog.tags_note_label.text()
    assert "WAV" not in dialog.tags_note_label.text()


def test_without_mutagen_the_tags_tab_still_works_for_an_m4b(qt_app, monkeypatch):
    qt_app.project_settings["export"] = {"format": "m4b", "tags": {"enabled": True, "title": "Kept"}}
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(tagging, "available", lambda: False)
    dialog = ExportDialog(qt_app)
    index = dialog.tabs.indexOf(dialog.tags_form.parentWidget())

    assert dialog.format_combo.currentText() == "m4b" and tags_usable("m4b") and not tags_usable("mp3")
    assert dialog.tabs.isTabEnabled(index) and dialog.tags_check.isEnabled() and dialog.tags_check.isChecked()
    assert tag_options(qt_app, dialog.values())["title"] == "Kept"

    dialog.format_combo.setCurrentText("mp3")
    assert not dialog.tabs.isTabEnabled(index) and "mutagen" in dialog.tabs.tabToolTip(index)
    assert not dialog.tags_check.isChecked() and dialog.values()["tags"]["enabled"] is True  # not lost
    assert tag_options(qt_app, dialog.values()) is None

    dialog.format_combo.setCurrentText("m4b")
    assert dialog.tabs.isTabEnabled(index) and dialog.tags_check.isChecked()


def test_switching_tags_off_survives_a_trip_through_a_format_that_cannot_use_them(qt_app, monkeypatch):
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    monkeypatch.setattr(tagging, "available", lambda: False)
    dialog = ExportDialog(qt_app)
    dialog.format_combo.setCurrentText("m4b")
    dialog.tags_check.setChecked(False)

    dialog.format_combo.setCurrentText("wav")
    dialog.format_combo.setCurrentText("m4b")

    assert not dialog.tags_check.isChecked() and dialog.values()["tags"]["enabled"] is False


def _tone_project(qt_app, tmp_path):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    return _generated_clip(qt_app, tmp_path, 0, 5, seconds=2.0, name="tone")


def test_an_m4b_export_sends_the_chapters_the_tags_and_the_bitrate_to_ffmpeg(qt_app, tmp_path, monkeypatch):
    fake = _with_ffmpeg(monkeypatch)
    _two_subprojects(qt_app)
    qt_app.project_settings["title"] = "The Book"
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.filename_edit.setText("book")
    dialog.format_combo.setCurrentText("m4b")
    dialog.keep_clips_check.setChecked(False)
    dialog.m4b_bitrate_combo.setCurrentIndex(dialog.m4b_bitrate_combo.findData(64))
    dialog.tag_artist_edit.setText("A. Writer")

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert result.audio_path == str(tmp_path / "out" / "book.m4b") and os.path.exists(result.audio_path)
    assert fake.calls[0][fake.calls[0].index("-b:a") + 1] == "64k"
    meta = fake.seen["meta"]
    assert "title=The Book\nartist=A. Writer\n" in meta
    assert [c.splitlines()[-1] for c in meta.split("[CHAPTER]")[1:]] == ["title=Alpha", "title=Beta"]
    assert result.files[0].tagged is True
    assert qt_app.project_settings["export"]["format"] == "m4b"
    assert qt_app.project_settings["export"]["m4b_bitrate_kbps"] == 64


def test_an_m4b_split_export_writes_one_m4b_per_subproject(qt_app, tmp_path, monkeypatch):
    fake = _with_ffmpeg(monkeypatch)
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("m4b")
    dialog.keep_clips_check.setChecked(False)
    dialog.split_combo.setCurrentIndex(dialog.split_combo.findData("subprojects"))

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert [os.path.basename(f.path) for f in result.files] == ["01 - Alpha.m4b", "02 - Beta.m4b"]
    assert len(fake.calls) == 2


def test_an_ffmpeg_failure_fails_the_export_and_leaves_no_files(qt_app, tmp_path, monkeypatch):
    _with_ffmpeg(monkeypatch, fail=subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"Invalid argument"))
    _tone_project(qt_app, tmp_path)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("m4b")
    dialog.keep_clips_check.setChecked(False)

    assert run_export(qt_app, dialog.values()) is True
    with pytest.raises(m4b.M4bError, match="Invalid argument"):
        _scheduled_result(qt_app)

    assert os.listdir(tmp_path / "out") == []  # the run's own `_done` turns the error into "Export failed: ..."


def test_run_export_refuses_an_m4b_when_ffmpeg_has_gone(qt_app, tmp_path, monkeypatch):
    _tone_project(qt_app, tmp_path)
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: FAKE)
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.format_combo.setCurrentText("m4b")
    values = dialog.values()
    monkeypatch.setattr(m4b, "find_ffmpeg", lambda: None)
    warned = []
    monkeypatch.setattr(export_dialog.QMessageBox, "warning", lambda *a, **k: warned.append(a[2]))
    qt_app.engine.worker.run_coro.reset_mock()

    assert run_export(qt_app, values) is False

    assert warned and "ffmpeg" in warned[0] and not qt_app.engine.worker.run_coro.called
