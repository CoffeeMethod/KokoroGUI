"""The Exports dock (plan 38): kokoro_gui/qt/docks/output_dock.py."""
import asyncio
import os

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication, QMessageBox

import kokoro_gui.qt.app  # noqa: F401 - app.py must load before any docks module (circular import)
from kokoro_gui.daw import exports_log
from kokoro_gui.qt.docks import output_dock as output_dock_module
from kokoro_gui.qt.docks.export_dialog import run_export
from tests.gui_qt.test_export_and_characters import _generated_clip, _type


def _export(qt_app, tmp_path, name="mix", **extra):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name=name + "_clip")
    values = {"out_dir": str(tmp_path / "out"), "filename": name, "format": "wav", "srt": True,
              "keep_clip_files": False, **extra}
    assert run_export(qt_app, values) is True
    result = asyncio.run(qt_app.engine.worker.run_coro.call_args[0][0])
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    return result


def _show(qt_app):
    qt_app.show()
    dock = qt_app.output_dock
    dock.raise_()
    QApplication.processEvents()
    assert dock.isVisible()
    dock.rebuild()
    return dock


def test_dock_is_tabbed_with_the_settings_group(qt_app):
    dock = qt_app.output_dock
    assert dock.objectName() == "dock_output" and dock.windowTitle() == "Exports"
    assert dock in qt_app.tabifiedDockWidgets(qt_app.settings_dock)


def test_an_export_shows_up(qt_app, tmp_path):
    result = _export(qt_app, tmp_path)
    dock = _show(qt_app)

    assert dock.tree.topLevelItemCount() == 1
    item = dock.tree.topLevelItem(0)
    assert "mix.wav" in item.text(0) and item.text(3) == "WAV"
    entry = item.data(0, output_dock_module._ENTRY)
    assert entry["path"] == os.path.abspath(result.audio_path)
    assert entry["extras"] == [os.path.abspath(result.srt_path)]


def test_delete_removes_the_audio_and_its_srt_and_nothing_else(qt_app, tmp_path):
    result = _export(qt_app, tmp_path)
    unrelated = os.path.join(os.path.dirname(result.audio_path), "notes.txt")
    with open(unrelated, "w") as f:
        f.write("keep me")
    dock = _show(qt_app)
    dock.tree.setCurrentItem(dock.tree.topLevelItem(0))

    assert dock.delete_selected() is True

    assert not os.path.exists(result.audio_path) and not os.path.exists(result.srt_path)
    assert os.path.exists(unrelated)
    assert exports_log.entries(qt_app.exports_log_file) == []
    assert dock.tree.topLevelItemCount() == 0


def test_declining_the_delete_keeps_every_file(qt_app, tmp_path, monkeypatch):
    result = _export(qt_app, tmp_path)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    dock = _show(qt_app)
    dock.tree.setCurrentItem(dock.tree.topLevelItem(0))

    assert dock.delete_selected() is False
    assert os.path.exists(result.audio_path) and os.path.exists(result.srt_path)


def test_a_file_deleted_outside_the_app_is_marked_missing(qt_app, tmp_path):
    result = _export(qt_app, tmp_path)
    os.remove(result.audio_path)
    dock = _show(qt_app)
    assert "(missing)" in dock.tree.topLevelItem(0).text(0)
    dock.tree.setCurrentItem(dock.tree.topLevelItem(0))
    assert not dock.play_button.isEnabled() and not dock.delete_button.isEnabled()


def test_the_project_filter(qt_app, tmp_path):
    here = os.path.abspath(tmp_path / "a.wav")
    sf.write(here, np.zeros(10, dtype=np.float32), 24000)
    exports_log.record(qt_app.exports_log_file, here, project_title="Here", project_path="")
    exports_log.record(qt_app.exports_log_file, here + "2.wav", project_title="Other", project_path="/o/other.tbaw")
    dock = _show(qt_app)
    assert dock.tree.topLevelItemCount() == 2

    dock.project_only.setChecked(True)

    assert dock.tree.topLevelItemCount() == 1
    assert dock.tree.topLevelItem(0).text(0).startswith("Here")


def test_play_and_a_second_click_stops(qt_app, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(output_dock_module.playback, "play", lambda path, blocking=False: calls.append(("play", path)))
    monkeypatch.setattr(output_dock_module.playback, "stop", lambda: calls.append(("stop",)))
    result = _export(qt_app, tmp_path)
    dock = _show(qt_app)
    dock.tree.setCurrentItem(dock.tree.topLevelItem(0))

    dock.toggle_play()
    assert dock.play_button.text() == "Stop"
    dock.toggle_play()

    assert calls == [("play", os.path.abspath(result.audio_path)), ("stop",)]
    assert dock.play_button.text() == "Play"


def test_show_in_folder_reveals_the_file(qt_app, tmp_path, monkeypatch):
    revealed = []
    monkeypatch.setattr(output_dock_module.reveal, "reveal", lambda path: revealed.append(path))
    result = _export(qt_app, tmp_path)
    dock = _show(qt_app)
    dock.tree.setCurrentItem(dock.tree.topLevelItem(0))
    dock.reveal_selected()
    assert revealed == [os.path.abspath(result.audio_path)]


def test_the_whole_document_path_is_logged(qt_app, tmp_path):
    combined = str(tmp_path / "out" / "output_0_combined.wav")
    os.makedirs(os.path.dirname(combined))
    sf.write(combined, np.zeros(24000, dtype=np.float32), 24000)

    qt_app.on_engine_status(f"Done! Saved: {combined}", False)

    rows = exports_log.entries(qt_app.exports_log_file)
    assert len(rows) == 1 and rows[0]["path"] == os.path.abspath(combined)
    assert rows[0]["duration_s"] == 1.0 and rows[0]["format"] == "wav"


def test_a_split_export_logs_one_entry_per_file(qt_app, tmp_path):
    from kokoro_gui.daw.mixdown import ExportFile, ExportResult

    files = [ExportFile(str(tmp_path / f"{n}.wav"), str(n), 1.0, extras=[str(tmp_path / f"{n}.srt")]) for n in (1, 2)]
    qt_app._on_export_files(ExportResult(audio_path=files[0].path, files=files), {"title": "T", "path": ""})
    rows = exports_log.entries(qt_app.exports_log_file)
    assert [os.path.basename(r["path"]) for r in rows] == ["2.wav", "1.wav"]
    assert os.path.basename(rows[0]["extras"][0]) == "2.srt"
