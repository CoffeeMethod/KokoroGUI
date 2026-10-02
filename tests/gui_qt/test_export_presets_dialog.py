"""The Export dialog's preset, split and silence fields, and what a preset or
split export does (plan 14): kokoro_gui/qt/docks/export_dialog.py."""
import asyncio
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtGui import QTextCursor

import kokoro_gui.qt.app  # noqa: F401 - app.py must load before any docks module (circular import)
from kokoro_gui.audio import loudness  # noqa: E402
from kokoro_gui.daw import markers  # noqa: E402
from kokoro_gui.daw.export_presets import PRESETS  # noqa: E402
from kokoro_gui.qt.docks.export_dialog import (  # noqa: E402
    ExportDialog, ExportReportDialog, chapter_plan, export_defaults, run_export,
)
from tests.gui_qt.test_export_and_characters import _generated_clip, _type  # noqa: E402
from tests.gui_qt import test_subprojects as sub  # noqa: E402


def _pick(dialog, preset_id):
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData(preset_id))


def _tone_project(qt_app, tmp_path, seconds=2.0):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    return _generated_clip(qt_app, tmp_path, 0, 5, seconds=seconds, name="tone", tone_hz=440)


def _scheduled_result(qt_app):
    coro = qt_app.engine.worker.run_coro.call_args[0][0]
    result = asyncio.run(coro)
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    return result


def _two_subprojects(qt_app):
    """Intro, then two embedded subprojects "Alpha" and "Beta", then Outro; both rendered."""
    qt_app.document.text = "Intro. Part A. Part B. Outro."
    qt_app.editor.load_text(qt_app.document.text)
    sub._generated_clip(qt_app, 0, 6)
    sub._generated_clip(qt_app, 7, 14)
    sub._generated_clip(qt_app, 15, 22)
    sub._generated_clip(qt_app, 23, 29)
    beta = qt_app.new_subproject(15, 22, title="Beta")  # later text first, so the earlier offsets hold
    alpha = qt_app.new_subproject(7, 14, title="Alpha")
    for child in (alpha, beta):
        sub._render(qt_app, child)
    return alpha, beta


# -- the dialog ---------------------------------------------------------------------------


def test_the_preset_combo_lists_custom_then_each_preset(qt_app):
    dialog = ExportDialog(qt_app)

    assert dialog.preset_combo.itemData(0) == "custom" and dialog.preset_combo.currentData() == "custom"
    assert [dialog.preset_combo.itemData(i) for i in range(1, dialog.preset_combo.count())] == [p.id for p in PRESETS]
    assert dialog.preset_combo.itemText(1) == "ACX (Audible)"


def test_picking_acx_fills_the_fields(qt_app):
    dialog = ExportDialog(qt_app)

    _pick(dialog, "acx")

    values = dialog.values()
    assert (values["format"], values["bitrate_kbps"], values["sample_rate"], values["channels"]) == ("mp3", 192, 44100, 1)
    assert values["normalize_loudness"] is True and values["normalize_mode"] == "rms"
    assert (values["target_rms_dbfs"], values["limiter_dbfs"]) == (-20.0, -3.5)
    assert (values["head_s"], values["tail_s"]) == (1.0, 2.0)
    assert values["preset"] == "acx"
    assert dialog.preset_combo.currentData() == "acx"  # filling the fields didn't count as an edit
    assert not dialog.bitrate_combo.isHidden()
    assert dialog.split_combo.currentData() is None  # no subprojects here, so that choice is greyed out


def test_picking_a_podcast_preset_switches_to_lufs_and_forgets_acx(qt_app):
    dialog = ExportDialog(qt_app)
    _pick(dialog, "acx")

    _pick(dialog, "apple")

    values = dialog.values()
    assert values["normalize_mode"] == "lufs" and values["target_lufs"] == -16.0 and values["ceiling_dbtp"] == -1.0
    assert (values["channels"], values["head_s"], values["tail_s"], values["sample_rate"]) == (2, 0.0, 0.0, 44100)
    assert not dialog.target_spin.isHidden() and dialog.rms_spin.isHidden()
    _pick(dialog, "youtube")
    assert (dialog.values()["format"], dialog.values()["sample_rate"]) == ("wav", 48000)


def test_editing_a_field_after_a_preset_goes_back_to_custom(qt_app):
    for edit in (lambda d: d.bitrate_combo.setCurrentIndex(2),
                 lambda d: d.format_combo.setCurrentText("flac"),
                 lambda d: d.channels_combo.setCurrentIndex(1),
                 lambda d: d.target_spin.setValue(-10.0),
                 lambda d: d.head_spin.setValue(0.25),
                 lambda d: d.normalize_check.setChecked(False)):
        dialog = ExportDialog(qt_app)
        _pick(dialog, "apple")
        assert dialog.preset_combo.currentData() == "apple"
        edit(dialog)
        assert dialog.preset_combo.currentData() == "custom"
        assert dialog.values()["preset"] == "custom"


def test_fields_that_arent_part_of_a_preset_leave_it_alone(qt_app):
    dialog = ExportDialog(qt_app)
    _pick(dialog, "spotify")

    dialog.out_dir_edit.setText("elsewhere")
    dialog.filename_edit.setText("renamed")
    dialog.srt_check.setChecked(True)

    assert dialog.preset_combo.currentData() == "spotify"


def test_the_preset_and_new_fields_persist_per_project(qt_app):
    dialog = ExportDialog(qt_app)
    _pick(dialog, "acx")
    qt_app.project_settings["export"] = dialog.values()

    again = ExportDialog(qt_app)

    assert again.preset_combo.currentData() == "acx"
    assert (again.head_spin.value(), again.tail_spin.value()) == (1.0, 2.0)
    assert again.normalize_mode_combo.currentData() == "rms" and again.rms_spin.value() == -20.0


def test_export_defaults_ignore_bad_preset_split_and_silence_values(qt_app):
    qt_app.project_settings["export"] = {"preset": "podcastx", "split": "chapters", "head_s": "long", "tail_s": 99,
                                         "normalize_mode": "peak", "target_rms_dbfs": 5, "limiter_dbfs": None}
    values = export_defaults(qt_app)

    assert (values["preset"], values["split"], values["head_s"], values["tail_s"]) == ("custom", None, 0.0, 10.0)
    assert (values["normalize_mode"], values["target_rms_dbfs"], values["limiter_dbfs"]) == ("lufs", -6.0, -3.5)


def test_presets_are_greyed_out_without_pyloudnorm(qt_app, monkeypatch):
    monkeypatch.setattr(loudness, "available", lambda: False)
    qt_app.project_settings["export"] = {"preset": "acx"}

    dialog = ExportDialog(qt_app)

    model = dialog.preset_combo.model()
    assert dialog.preset_combo.currentData() == "custom"
    assert model.item(0).isEnabled() and not any(model.item(i).isEnabled() for i in range(1, dialog.preset_combo.count()))


def test_the_normalize_mode_swaps_the_rows(qt_app):
    dialog = ExportDialog(qt_app)
    assert not dialog.target_spin.isHidden() and dialog.rms_spin.isHidden() and dialog.limiter_spin.isHidden()

    dialog.normalize_mode_combo.setCurrentIndex(dialog.normalize_mode_combo.findData("rms"))

    assert dialog.target_spin.isHidden() and dialog.ceiling_spin.isHidden()
    assert not dialog.rms_spin.isHidden() and not dialog.limiter_spin.isHidden()
    assert (dialog.rms_spin.minimum(), dialog.rms_spin.maximum()) == (-40.0, -6.0)
    assert (dialog.limiter_spin.minimum(), dialog.limiter_spin.maximum()) == (-12.0, 0.0)


def test_split_choices_need_subprojects_or_two_markers(qt_app):
    dialog = ExportDialog(qt_app)
    model = dialog.split_combo.model()
    assert [model.item(i).isEnabled() for i in range(3)] == [True, False, False]

    qt_app.document.settings["markers"], _a = markers.add_marker(qt_app.document.settings, 1.0, name="A")
    qt_app.document.settings["markers"], _b = markers.add_marker(qt_app.document.settings, 3.0, name="B")
    model = ExportDialog(qt_app).split_combo.model()
    assert [model.item(i).isEnabled() for i in range(3)] == [True, False, True]


def test_a_stored_split_that_no_longer_applies_falls_back_to_one_file(qt_app):
    qt_app.project_settings["export"] = {"split": "subprojects"}

    assert ExportDialog(qt_app).split_combo.currentData() is None


def test_splitting_by_subproject_previews_the_files_and_greys_the_range(qt_app):
    _two_subprojects(qt_app)
    qt_app.document.settings["markers"], _a = markers.add_marker(qt_app.document.settings, 0.0, name="A")
    qt_app.document.settings["markers"], _b = markers.add_marker(qt_app.document.settings, 1.0, name="B")
    dialog = ExportDialog(qt_app)
    assert dialog.range_combo.isEnabled()

    dialog.split_combo.setCurrentIndex(dialog.split_combo.findData("subprojects"))

    assert not dialog.range_combo.isEnabled()
    assert dialog.name_preview_label.text() == "Writes 01 - Alpha.wav and 1 more"
    dialog.split_combo.setCurrentIndex(0)
    assert dialog.range_combo.isEnabled() and dialog.name_preview_label.text() == "Writes output.wav"


def test_acx_picks_one_file_per_subproject_when_there_are_subprojects(qt_app):
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)

    _pick(dialog, "acx")

    assert dialog.split_combo.currentData() == "subprojects"
    assert dialog.preset_combo.currentData() == "acx"
    dialog.split_combo.setCurrentIndex(0)
    assert dialog.preset_combo.currentData() == "custom"


# -- running an export ---------------------------------------------------------------------


def test_an_acx_export_pads_normalizes_and_names_the_check_it_failed(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path, seconds=30.0)  # the head and tail silence dilute the RMS, a little over 30 s
    dialog = ExportDialog(qt_app)
    _pick(dialog, "acx")
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    values = dialog.values()

    assert run_export(qt_app, values) is True
    result = _scheduled_result(qt_app)

    info = sf.info(result.audio_path)
    assert result.audio_path.endswith(".mp3") and info.samplerate == 44100 and info.channels == 1
    assert info.duration == pytest.approx(30.0 + 1.0 + 2.0, abs=0.1)
    (entry,) = result.files
    assert entry.report.rms_dbfs == pytest.approx(-20.0, abs=0.5)
    assert entry.report.sample_peak_dbfs <= -3.5 + 0.1
    # A steady tone has no quiet stretch, so only the noise floor check fails.
    assert [f.split(" ")[0] for f in entry.failed] == ["Noise"]
    assert "Failed 1 of the ACX (Audible) checks (see report)" in qt_app.transport_dock.status_text()
    report = qt_app._export_report_dialog
    assert isinstance(report, ExportReportDialog) and report.table.rowCount() == 1
    assert report.table.item(0, 6).text().startswith("Failed: Noise floor")
    assert qt_app.project_settings["export"]["preset"] == "acx"


def test_a_clean_single_file_export_opens_no_report(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path)
    dialog = ExportDialog(qt_app)
    _pick(dialog, "spotify")
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.normalize_check.setChecked(True)
    dialog.preset_combo.setCurrentIndex(dialog.preset_combo.findData("custom"))  # no checks to fail

    assert run_export(qt_app, dialog.values()) is True
    _scheduled_result(qt_app)

    assert getattr(qt_app, "_export_report_dialog", None) is None
    assert "checks" not in qt_app.transport_dock.status_text()


def test_a_preset_export_that_passes_says_so(qt_app, tmp_path):
    _tone_project(qt_app, tmp_path, seconds=3.0)
    dialog = ExportDialog(qt_app)
    _pick(dialog, "youtube")  # -14 LUFS under -1 dBTP: a sine has no peaks to stop it
    dialog.out_dir_edit.setText(str(tmp_path / "out"))

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert result.files[0].failed == []
    assert "Passed the YouTube checks" in qt_app.transport_dock.status_text()
    assert getattr(qt_app, "_export_report_dialog", None) is None


def test_a_split_export_writes_one_file_per_subproject_and_reports_them(qt_app, tmp_path):
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)
    dialog.split_combo.setCurrentIndex(dialog.split_combo.findData("subprojects"))
    dialog.out_dir_edit.setText(str(tmp_path / "out"))
    dialog.srt_check.setChecked(True)
    dialog.keep_clips_check.setChecked(False)

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert [os.path.basename(f.path) for f in result.files] == ["01 - Alpha.wav", "02 - Beta.wav"]
    assert all(sf.info(f.path).frames > 0 for f in result.files)
    assert sorted(os.listdir(tmp_path / "out")) == ["01 - Alpha.srt", "01 - Alpha.wav", "02 - Beta.srt", "02 - Beta.wav"]
    assert qt_app.project_settings["export"]["split"] == "subprojects"
    status = qt_app.transport_dock.status_text()
    assert "Exported 2 files" in status and "2 clips outside any subproject weren't exported" in status
    assert qt_app._export_report_dialog.table.rowCount() == 2
    assert qt_app._last_export_path == result.audio_path


def test_acx_split_export_reports_failures_per_file(qt_app, tmp_path):
    _two_subprojects(qt_app)
    dialog = ExportDialog(qt_app)
    _pick(dialog, "acx")
    dialog.out_dir_edit.setText(str(tmp_path / "out"))

    assert run_export(qt_app, dialog.values()) is True
    result = _scheduled_result(qt_app)

    assert [os.path.basename(f.path) for f in result.files] == ["01 - Alpha.mp3", "02 - Beta.mp3"]
    assert all(f.duration_s == pytest.approx(f.report.duration_s, abs=0.01) for f in result.files)
    status = qt_app.transport_dock.status_text()
    assert "Exported 2 files" in status and "failed the ACX (Audible) checks (see report)" in status
    summary = qt_app._export_report_dialog.summary_label.text()
    assert "files passed the ACX (Audible) checks" in summary


def test_a_split_with_nothing_to_split_refuses(qt_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[1]))
    _tone_project(qt_app, tmp_path)
    values = {**export_defaults(qt_app), "out_dir": str(tmp_path / "out"), "split": "subprojects"}

    assert run_export(qt_app, values) is False
    assert shown == ["Nothing to split"] and not qt_app.is_busy()


def test_a_split_export_asks_once_when_files_exist(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt.docks import export_dialog

    _two_subprojects(qt_app)
    out = tmp_path / "out"
    out.mkdir()
    (out / "01 - Alpha.wav").write_bytes(b"old")
    (out / "02 - Beta.wav").write_bytes(b"old")
    asked = []
    monkeypatch.setattr(export_dialog, "_ask_existing", lambda parent, path, more=0: asked.append((path, more)) or "number")
    values = {**export_defaults(qt_app), "out_dir": str(out), "split": "subprojects", "srt": False}

    assert run_export(qt_app, values) is True
    result = _scheduled_result(qt_app)

    assert asked == [(str(out / "01 - Alpha.wav"), 1)]
    assert [os.path.basename(f.path) for f in result.files] == ["01 - Alpha (2).wav", "02 - Beta (2).wav"]
    assert (out / "01 - Alpha.wav").read_bytes() == b"old"


def test_cancelling_the_overwrite_prompt_of_a_split_leaves_the_settings_alone(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt.docks import export_dialog

    _two_subprojects(qt_app)
    out = tmp_path / "out"
    out.mkdir()
    (out / "02 - Beta.wav").write_bytes(b"old")
    monkeypatch.setattr(export_dialog, "_ask_existing", lambda parent, path, more=0: None)
    values = {**export_defaults(qt_app), "out_dir": str(out), "split": "subprojects"}

    assert run_export(qt_app, values) is False
    assert "split" not in qt_app.project_settings.get("export", {})


def test_chapter_plan_reads_the_project_the_timeline_shows(qt_app):
    _two_subprojects(qt_app)

    plan = chapter_plan(qt_app, "subprojects")

    assert [c.name for c in plan.chapters] == ["01 - Alpha", "02 - Beta"]
