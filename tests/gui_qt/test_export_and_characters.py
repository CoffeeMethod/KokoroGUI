"""Tests for the Export dialog (section 6) and the Edit > Characters dialog
(UI13) of Claude/PLAN_ui_shell_redesign.md, plus the transport -> playhead
-> transcript follow chain (section 5) at the app level."""
import math
import os

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMessageBox

from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash

import kokoro_gui.qt.app  # noqa: F401 - app.py must load before any docks module (circular import)
from kokoro_gui.qt.characters_dialog import CharactersDialog  # noqa: E402
from kokoro_gui.qt.docks.export_dialog import ExportDialog, export_defaults, run_export  # noqa: E402


def _type(editor, text):
    cursor = editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _generated_clip(qt_app, tmp_path, start, end, seconds=1.0, name="a", tone_hz=None, character=None):
    alice = character or qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, end, alice.id)
    path = str(tmp_path / f"{name}.wav")
    if tone_hz:  # loudness tests need something K-weighting does not remove
        samples = (0.25 * np.sin(2 * np.pi * tone_hz * np.arange(int(24000 * seconds)) / 24000)).astype(np.float32)
    else:
        samples = np.full(int(24000 * seconds), 0.25, dtype=np.float32)
    sf.write(path, samples, 24000)
    text = qt_app.document.clip_text(clip)
    expected = compute_expected_cache_hash(text, qt_app.document.effective_config_for_clip(clip))
    clip.segments = build_segments_from_results(expected, [{"text": text, "path": path, "duration": seconds}])
    return clip


# -- export -------------------------------------------------------------------------


def test_export_defaults_fall_back_to_legacy_settings_then_project(qt_app):
    qt_app.settings["out_dir"] = "legacy_dir"
    qt_app.settings["export_subtitles"] = True
    assert export_defaults(qt_app)["out_dir"] == "legacy_dir"
    assert export_defaults(qt_app)["srt"] is True

    qt_app.project_settings["export"] = {"out_dir": "proj_dir", "format": "flac"}
    values = export_defaults(qt_app)
    assert values["out_dir"] == "proj_dir" and values["format"] == "flac"


def test_export_dialog_reads_back_sanitized_values(qt_app):
    dialog = ExportDialog(qt_app)
    dialog.out_dir_edit.setText("out")
    dialog.filename_edit.setText("../../evil")
    dialog.format_combo.setCurrentText("flac")
    dialog.srt_check.setChecked(True)
    dialog.keep_clips_check.setChecked(True)

    values = dialog.values()

    assert values == {"out_dir": "out", "filename": "evil", "format": "flac", "srt": True, "keep_clip_files": True,
                      "channels": 2, "srt_words": False, "cue_sheet": False,
                      "normalize_loudness": False, "target_lufs": -16.0, "ceiling_dbtp": -1.0,
                      "bitrate_kbps": 192, "sample_rate": None, "normalize_mode": "lufs", "target_rms_dbfs": -20.0,
                      "limiter_dbfs": -3.5, "head_s": 0.0, "tail_s": 0.0, "split": None, "preset": "custom",
                      "stems": None, "dialogue_stem": False, "extras": [], "transcript_speakers": True}


def test_run_export_refuses_without_clips(qt_app, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a)))
    assert run_export(qt_app, export_defaults(qt_app)) is False
    assert infos


def test_run_export_with_dirty_clips_offers_generate_first(qt_app, monkeypatch):
    _type(qt_app.editor, "hello world")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 11, alice.id)  # dirty
    generated = []
    monkeypatch.setattr(qt_app, "on_generate_clicked", lambda: generated.append(True))
    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    monkeypatch.setattr(QMessageBox, "clickedButton",
                        lambda self: next(b for b in self.buttons() if b.text() == "Generate first"))

    assert run_export(qt_app, export_defaults(qt_app)) is False
    assert generated == [True]


def test_run_export_schedules_mixdown_on_the_worker_and_writes_the_file(qt_app, tmp_path):
    qt_app.document.settings["gap_s"] = 0.0  # about the export path, not gaps
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a")
    _generated_clip(qt_app, tmp_path, 6, 11, seconds=0.5, name="b")
    values = {"out_dir": str(tmp_path / "out"), "filename": "mix", "format": "wav", "srt": True,
              "keep_clip_files": True}

    assert run_export(qt_app, values) is True
    assert qt_app.is_busy()
    assert qt_app.project_settings["export"] == values

    coro = qt_app.engine.worker.run_coro.call_args[0][0]
    import asyncio

    result = asyncio.run(coro)
    assert os.path.exists(result.audio_path)
    data, rate = sf.read(result.audio_path)
    assert rate == 24000 and len(data) == 36000
    assert result.srt_path.endswith("mix.srt")
    assert len(result.clip_files) == 2

    # StubEngine hands out a real Future; resolving it runs run_export's
    # done-callback (the worker thread in real life), which emits
    # exportFinished back onto the GUI thread.
    future = qt_app.engine.worker.run_coro.return_value
    future.set_result(result)
    assert not qt_app.is_busy()
    assert "Exported" in qt_app.transport_dock.status_text()
    assert qt_app._last_export_path == result.audio_path
    assert qt_app.show_last_export_action.isEnabled()


# -- characters dialog ------------------------------------------------------------------


def test_characters_dialog_lists_and_edits_name_color_voice_fx(qt_app):
    alice = qt_app.document.characters[0]
    assert qt_app.document.tracks == []  # a track is made on first use
    _type(qt_app.editor, "hello")
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    dialog = CharactersDialog(qt_app)
    assert [dialog.list.item(i).text() for i in range(dialog.list.count())] == ["Default"]

    dialog.name_edit.setText("Narrator")
    dialog.name_edit.textEdited.emit("Narrator")
    dialog.set_color("#123456")
    dialog.voice_combo.setCurrentText("af_bella")
    dialog.fx_combo.addItem("Echo")
    dialog.fx_combo.setCurrentText("Echo")

    assert alice.name == "Narrator"
    assert alice.highlight_color == "#123456"
    assert alice.preset_data["voice"] == "af_bella"
    assert alice.preset_data["fx_preset"] == "Echo"
    assert qt_app.document.tracks[0].name == "Narrator"
    assert qt_app.transcript_dock.character_combo.findText("Narrator") >= 0


def test_characters_dialog_add_and_remove(qt_app, monkeypatch):
    dialog = CharactersDialog(qt_app)
    new = dialog.add_character()
    assert new in qt_app.document.characters
    assert new.library_id is None
    assert not any(t.character_id == new.id for t in qt_app.document.tracks)

    dialog.remove_current()
    assert new not in qt_app.document.characters
    assert not any(t.character_id == new.id for t in qt_app.document.tracks)


def test_characters_dialog_refuses_to_remove_a_character_in_use(qt_app, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    _type(qt_app.editor, "hello")
    alice = qt_app.document.characters[0]
    qt_app.document.assign_character_to_range(0, 5, alice.id)
    dialog = CharactersDialog(qt_app)

    dialog.remove_current()

    assert alice in qt_app.document.characters
    assert warned


# -- transport follow ---------------------------------------------------------------------


def test_transport_position_drives_playhead_readout_and_playing_clip(qt_app, tmp_path):
    qt_app.document.settings["gap_s"] = 0.0  # about the readout, not gaps
    _type(qt_app.editor, "hello world")
    first = _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a")
    second = _generated_clip(qt_app, tmp_path, 6, 11, seconds=1.0, name="b")
    qt_app._rebuild_transport_schedule()
    assert qt_app.transport.duration() == 2.0

    qt_app.transport._set_state("playing")
    qt_app._on_transport_position(0.5)
    assert qt_app.selection.playing_clip_id == first.id
    assert "00:00.5 / 00:02.0" == qt_app.transport_dock.time_label.text()
    assert qt_app.timeline_dock.timeline_view._playhead_item.isVisible()

    qt_app._on_transport_position(1.5)
    assert qt_app.selection.playing_clip_id == second.id
    assert qt_app.selection.selected_clip_id is None

    qt_app._on_transport_state("stopped")
    assert qt_app.selection.playing_clip_id is None


def test_generation_finishing_reloads_the_transport(qt_app, tmp_path, monkeypatch):
    reloads = []
    monkeypatch.setattr(qt_app.transport, "load", lambda *a, **k: reloads.append(True))
    qt_app.on_batch_generation_finished(1, 0, [])
    qt_app.on_engine_finish()
    assert reloads == [True, True]


def test_ruler_seek_moves_the_transport(qt_app, tmp_path):
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 11, seconds=3.0, name="a")
    qt_app._rebuild_transport_schedule()

    qt_app.timeline_dock.timeline_view.seekRequested.emit(1.25)

    assert abs(qt_app.transport.position() - 1.25) < 1e-6
    assert qt_app.transport_dock.time_label.text().startswith("00:01.2")


def test_characters_changed_refreshes_header_and_timeline(qt_app):
    dialog = CharactersDialog(qt_app)
    added = dialog.add_character()
    assert qt_app.transcript_dock.character_combo.findData(added.id) >= 0
    _type(qt_app.editor, "hello")
    qt_app.document.assign_character_to_range(0, 5, added.id)
    qt_app.on_characters_changed()
    labels = [item.text() for item in qt_app.timeline_dock.timeline_widget.header._scene.items()
              if hasattr(item, "text")]
    assert added.name in labels



def test_export_dialog_offers_channels_ranges_cue_sheet_and_word_srt(qt_app):
    from kokoro_gui.daw import markers
    from kokoro_gui.qt.docks.export_dialog import ExportDialog

    qt_app.document.settings["markers"], _a = markers.add_marker(qt_app.document.settings, 1.0, name="A")
    qt_app.document.settings["markers"], _b = markers.add_marker(qt_app.document.settings, 3.0, name="B")
    dialog = ExportDialog(qt_app)
    dialog.channels_combo.setCurrentIndex(dialog.channels_combo.findData(1))
    dialog.cue_sheet_check.setChecked(True)
    dialog.srt_words_check.setChecked(True)
    dialog.range_combo.setCurrentIndex(1)

    values = dialog.values()
    assert (values["channels"], values["cue_sheet"], values["srt_words"]) == (1, True, True)
    assert dialog.range_s() == (1.0, 3.0)
    assert dialog.range_combo.itemText(1) == "A to B"


def test_characters_dialog_variants_table_edits_the_character(qt_app, monkeypatch):
    import dataclasses

    from kokoro_gui.qt.characters_dialog import CharactersDialog

    cloning = dataclasses.replace(qt_app.backend.capabilities, supports_voice_cloning=True)
    monkeypatch.setattr(qt_app.backend, "capabilities", cloning)
    character = qt_app.document.characters[0]
    character.backend_id = qt_app.backend.id
    dialog = CharactersDialog(qt_app)
    assert not dialog.variants_box.isHidden()

    dialog.add_variant()
    dialog.variants_table.item(0, 0).setText("angry")
    dialog.variants_table.cellWidget(0, 1).setCurrentText("angry_ref")

    assert character.variants == {"angry": "angry_ref"}
    dialog.remove_variant()
    assert character.variants == {}


def test_transcript_variant_combo_sets_the_override_undoably(qt_app):
    character = qt_app.document.characters[0]
    character.variants = {"angry": "angry_ref"}
    qt_app.document.text = "hello"
    clip = qt_app.document.assign_character_to_range(0, 5, character.id)
    qt_app.editor.load_text(qt_app.document.text)
    qt_app.selection.select_clip(clip.id)
    dock = qt_app.transcript_dock
    dock.sync_header()
    assert not dock.variant_combo.isHidden()

    dock.set_clip_variant(clip.id, "angry")
    assert clip.overrides["variant"] == "angry"
    qt_app.document.undo_stack.undo()
    assert "variant" not in clip.overrides


# -- the character library in the dialog (phase 3 step 5) -----------------------------


def _scopes(dialog):
    from kokoro_gui.qt.characters_dialog import _SCOPE_ROLE
    return [dialog.list.item(i).data(_SCOPE_ROLE) for i in range(dialog.list.count())]


def test_promote_makes_a_library_entry_and_links_the_record(qt_app):
    from kokoro_gui.qt.characters_dialog import SCOPE_LIBRARY, SCOPE_LOCAL

    character = qt_app.document.characters[0]
    dialog = CharactersDialog(qt_app)
    assert _scopes(dialog) == [SCOPE_LOCAL]
    assert dialog.promote_btn.isEnabled()
    assert not dialog.rename_in_library_btn.isEnabled()

    library_id = dialog.promote_current()

    assert character.library_id == library_id
    entry = qt_app.character_library.get(library_id)
    assert entry.name == character.name
    assert entry.preset_data == character.preset_data
    assert _scopes(dialog) == [SCOPE_LIBRARY]
    assert dialog.scope_label.text() == SCOPE_LIBRARY
    assert not dialog.promote_btn.isEnabled()  # greyed once linked
    assert dialog.promote_current() is None


def test_editing_a_linked_character_writes_the_library_entry(qt_app):
    character = qt_app.document.characters[0]
    dialog = CharactersDialog(qt_app)
    library_id = dialog.promote_current()

    dialog.voice_combo.setCurrentText("am_adam")
    dialog.set_color("#654321")

    entry = qt_app.character_library.get(library_id)
    assert entry.preset_data["voice"] == "am_adam"
    assert entry.highlight_color == "#654321"
    assert character.preset_data["voice"] == "am_adam"


def test_renaming_edits_the_project_only_until_rename_in_library(qt_app):
    dialog = CharactersDialog(qt_app)
    library_id = dialog.promote_current()
    original = qt_app.character_library.get(library_id).name

    dialog.name_edit.setText("Narrator here")
    dialog.name_edit.textEdited.emit("Narrator here")
    assert qt_app.document.characters[0].name == "Narrator here"
    assert qt_app.character_library.get(library_id).name == original

    assert dialog.rename_in_library_btn.isEnabled()
    assert dialog.rename_in_library() is True
    assert qt_app.character_library.get(library_id).name == "Narrator here"


def test_add_from_library_inlines_a_linked_copy(qt_app):
    from kokoro_gui.daw.models import Character

    library = qt_app.character_library
    host_id = library.save(Character.from_preset_dict("Host", {"voice": "am_adam"}, highlight_color="#3fae7a"))
    guest_id = library.save(Character.from_preset_dict("Guest", {"voice": "af_bella"}))
    dialog = CharactersDialog(qt_app)
    assert {e.library_id for e in dialog.library_entries_to_add()} == {host_id, guest_id}

    added = dialog.add_from_library([host_id])

    assert len(added) == 1
    host = added[0]
    assert host in qt_app.document.characters
    assert host.library_id == host_id
    assert host.id != host_id
    assert host.preset_data == {"voice": "am_adam"}
    assert host.highlight_color == "#3fae7a"
    assert not any(t.character_id == host.id for t in qt_app.document.tracks)
    assert [e.library_id for e in dialog.library_entries_to_add()] == [guest_id]
    assert qt_app.transcript_dock.character_combo.findData(host.id) >= 0


def test_a_character_missing_from_this_library_plays_its_snapshot(qt_app):
    from kokoro_gui.daw.models import Character
    from kokoro_gui.qt.characters_dialog import SCOPE_MISSING

    orphan = Character.from_preset_dict("Visitor", {"voice": "bf_emma"}, library_id="from-another-machine")
    qt_app.document.characters.append(orphan)
    report = qt_app.resolve_library()
    assert report.missing == [orphan.id]
    assert orphan.id in qt_app.library_missing
    assert orphan.preset_data == {"voice": "bf_emma"}

    dialog = CharactersDialog(qt_app)
    dialog.list.setCurrentRow(len(qt_app.document.characters) - 1)
    assert dialog.scope_label.text() == SCOPE_MISSING
    assert not dialog.promote_btn.isEnabled()
    assert not dialog.rename_in_library_btn.isEnabled()

    # Edits stay on the snapshot; nothing is written to this library.
    dialog.voice_combo.setCurrentText("am_adam")
    assert orphan.preset_data["voice"] == "am_adam"
    assert qt_app.character_library.get("from-another-machine") is None


# -- engine pickers (grill EN1/EN2/EN4) ----------------------------------------


def test_add_uses_the_default_engine_not_the_active_characters(qt_app):
    qt_app.set_default_engine("dummy")
    assert qt_app.backend.id == "kokoro"

    new = CharactersDialog(qt_app).add_character()

    assert new.backend_id == "dummy"
    assert new.preset_data["voice"] == "dummy"


def test_a_settings_tab_edit_on_a_linked_character_reaches_the_library(qt_app):
    character = qt_app.document.characters[0]
    library_id = CharactersDialog(qt_app).promote_current()
    qt_app.selection.select_character(character.id)
    assert qt_app.settings_dock._mode == "character"

    qt_app.settings_dock.schema_form.widget_for("speed").setValue(1.4)
    qt_app.settings_dock.volume_spin.setValue(0.6)

    entry = qt_app.character_library.get(library_id)
    assert entry.preset_data["speed"] == 1.4
    assert entry.preset_data["volume"] == 0.6


def test_both_pickers_list_the_same_engines_in_the_same_order(qt_app):
    dialog = CharactersDialog(qt_app)
    combo = qt_app.settings_dock.engine_combo
    dialog_ids = [dialog.engine_combo.itemData(i) for i in range(dialog.engine_combo.count())]
    dock_ids = [combo.itemData(i) for i in range(combo.count())]
    assert dialog_ids == dock_ids == [engine_id for _label, engine_id in qt_app.engine_choices()]


# -- loudness (plan 12) -------------------------------------------------------------------


def test_export_dialog_loudness_rows_default_off_and_persist(qt_app):
    pytest.importorskip("pyloudnorm")
    dialog = ExportDialog(qt_app)
    assert not dialog.normalize_check.isChecked()
    assert not dialog.target_spin.isEnabled() and not dialog.ceiling_spin.isEnabled()
    assert (dialog.target_spin.value(), dialog.ceiling_spin.value()) == (-16.0, -1.0)
    assert (dialog.target_spin.minimum(), dialog.target_spin.maximum()) == (-30.0, -5.0)
    assert (dialog.ceiling_spin.minimum(), dialog.ceiling_spin.maximum()) == (-6.0, 0.0)

    dialog.normalize_check.setChecked(True)
    dialog.target_spin.setValue(-19.0)
    dialog.ceiling_spin.setValue(-3.0)
    assert dialog.target_spin.isEnabled() and dialog.ceiling_spin.isEnabled()
    qt_app.project_settings["export"] = dialog.values()

    again = ExportDialog(qt_app)
    assert again.normalize_check.isChecked()
    assert (again.target_spin.value(), again.ceiling_spin.value()) == (-19.0, -3.0)


def test_export_defaults_ignore_a_bad_loudness_value(qt_app):
    qt_app.project_settings["export"] = {"target_lufs": "loud", "ceiling_dbtp": 40, "normalize_loudness": 1}
    values = export_defaults(qt_app)
    assert values["target_lufs"] == -16.0 and values["ceiling_dbtp"] == 0.0 and values["normalize_loudness"] is True


def test_export_dialog_disables_normalize_without_pyloudnorm(qt_app, monkeypatch):
    from kokoro_gui.audio import loudness

    monkeypatch.setattr(loudness, "available", lambda: False)
    qt_app.project_settings["export"] = {"normalize_loudness": True}
    dialog = ExportDialog(qt_app)
    assert not dialog.normalize_check.isEnabled() and not dialog.normalize_check.isChecked()
    assert "pyloudnorm" in dialog.normalize_check.toolTip()


def _export_with_loudness(qt_app, tmp_path, **loud):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a", tone_hz=440)
    _generated_clip(qt_app, tmp_path, 6, 11, seconds=1.0, name="b", tone_hz=440)
    values = {"out_dir": str(tmp_path / "out"), "filename": "mix", "format": "wav", "srt": False,
              "keep_clip_files": False, "normalize_loudness": True, **loud}
    assert run_export(qt_app, values) is True
    coro = qt_app.engine.worker.run_coro.call_args[0][0]
    import asyncio

    result = asyncio.run(coro)
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    return result


def test_export_with_loudness_reports_what_it_measured(qt_app, tmp_path):
    pytest.importorskip("pyloudnorm")
    result = _export_with_loudness(qt_app, tmp_path, target_lufs=-20.0, ceiling_dbtp=-1.0)

    assert not result.loudness_limited
    assert result.loudness_after.integrated_lufs == pytest.approx(-20.0, abs=0.5)
    message = qt_app.transport_dock.status_text()
    assert "Measured" in message and "LUFS" in message and "dBTP" in message


def test_export_message_says_when_the_peak_ceiling_limited_the_gain(qt_app, tmp_path):
    pytest.importorskip("pyloudnorm")
    result = _export_with_loudness(qt_app, tmp_path, target_lufs=-5.0, ceiling_dbtp=-6.0)

    assert result.loudness_limited
    assert "target not reached: peak-limited" in qt_app.transport_dock.status_text()


# -- Measure Loudness ---------------------------------------------------------------------


def test_measure_loudness_refuses_without_clips(qt_app, monkeypatch):
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a)))
    qt_app.measure_loudness()
    assert infos and not qt_app.is_busy()


def test_measure_loudness_shows_numbers_for_two_clips_and_writes_no_file(qt_app, tmp_path):
    pytest.importorskip("pyloudnorm")
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a", tone_hz=440)
    _generated_clip(qt_app, tmp_path, 6, 11, seconds=1.0, name="b", tone_hz=440)
    before = sorted(p.name for p in tmp_path.rglob("*"))

    qt_app.measure_loudness()
    assert qt_app.is_busy()
    coro = qt_app.engine.worker.run_coro.call_args[0][0]
    import asyncio

    report = asyncio.run(coro)
    qt_app.engine.worker.run_coro.return_value.set_result(report)

    assert not qt_app.is_busy()
    dialog = qt_app._loudness_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.report.duration_s == pytest.approx(2.0, abs=0.05)
    assert math.isfinite(dialog.report.integrated_lufs)
    assert "LUFS" in dialog.value_labels["integrated"].text()
    assert "dBTP" in dialog.value_labels["true_peak"].text()
    assert sorted(p.name for p in tmp_path.rglob("*")) == before
    dialog.close()


# -- export options: tabs, mp3 bitrate, sample rate, name template, existing files ------------


def test_export_dialog_has_three_tabs_and_keeps_its_attribute_names(qt_app):
    dialog = ExportDialog(qt_app)

    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == ["Audio", "Extras", "Project file"]

    def tab_of(widget):
        page = widget
        while page is not None and dialog.tabs.indexOf(page) < 0:
            page = page.parentWidget()
        return dialog.tabs.tabText(dialog.tabs.indexOf(page))

    for name in ("out_dir_edit", "filename_edit", "format_combo", "bitrate_combo", "sample_rate_combo",
                 "channels_combo", "normalize_check", "target_spin", "ceiling_spin", "range_combo"):
        assert tab_of(getattr(dialog, name)) == "Audio", name
    for name in ("srt_check", "srt_words_check", "cue_sheet_check", "keep_clips_check"):
        assert tab_of(getattr(dialog, name)) == "Extras", name
    for name in ("bundle_audio_check", "bundle_imported_check", "bundle_format_combo", "bundle_video_check"):
        assert tab_of(getattr(dialog, name)) == "Project file", name


def test_export_dialog_shows_the_bitrate_only_for_mp3(qt_app):
    dialog = ExportDialog(qt_app)
    assert dialog.format_combo.currentText() == "wav" and dialog.bitrate_combo.isHidden()

    dialog.format_combo.setCurrentText("mp3")
    assert not dialog.bitrate_combo.isHidden()
    assert [dialog.bitrate_combo.itemData(i) for i in range(dialog.bitrate_combo.count())] == [128, 192, 256, 320]
    assert dialog.bitrate_combo.currentData() == 192

    dialog.format_combo.setCurrentText("flac")
    assert dialog.bitrate_combo.isHidden()

    qt_app.project_settings["export"] = {"format": "mp3"}
    assert not ExportDialog(qt_app).bitrate_combo.isHidden()


def test_export_dialog_remembers_bitrate_and_sample_rate_per_project(qt_app):
    dialog = ExportDialog(qt_app)
    assert dialog.sample_rate_combo.currentData() is None
    assert dialog.sample_rate_combo.itemText(0) == f"Project rate ({qt_app.project_sample_rate()} Hz)"
    assert [dialog.sample_rate_combo.itemData(i) for i in range(1, dialog.sample_rate_combo.count())] == [
        22050, 24000, 44100, 48000]

    dialog.format_combo.setCurrentText("mp3")
    dialog.bitrate_combo.setCurrentIndex(dialog.bitrate_combo.findData(320))
    dialog.sample_rate_combo.setCurrentIndex(dialog.sample_rate_combo.findData(44100))
    values = dialog.values()
    assert values["bitrate_kbps"] == 320 and values["sample_rate"] == 44100
    qt_app.project_settings["export"] = values

    again = ExportDialog(qt_app)
    assert again.bitrate_combo.currentData() == 320 and again.sample_rate_combo.currentData() == 44100


def test_export_defaults_ignore_a_bad_bitrate_or_rate(qt_app):
    qt_app.project_settings["export"] = {"bitrate_kbps": 999, "sample_rate": "fast"}
    values = export_defaults(qt_app)
    assert values["bitrate_kbps"] == 192 and values["sample_rate"] is None
    qt_app.project_settings["export"] = {"bitrate_kbps": True, "sample_rate": 12345}
    values = export_defaults(qt_app)
    assert values["bitrate_kbps"] == 192 and values["sample_rate"] is None


def test_export_dialog_previews_the_expanded_filename(qt_app):
    dialog = ExportDialog(qt_app)
    dialog.filename_edit.setText("{project}-{date}")
    shown = dialog.name_preview_label.text()
    assert shown.startswith("Writes Untitled-") and shown.endswith(".wav")
    assert "{" not in shown

    dialog.format_combo.setCurrentText("flac")
    dialog.filename_edit.setText("take-{range}-{nope}")
    assert dialog.name_preview_label.text() == "Writes take-full-{nope}.flac"
    assert dialog.values()["filename"] == "take-{range}-{nope}"  # the template is what gets stored


def _one_clip(qt_app, tmp_path):
    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a")


def _scheduled_result(qt_app):
    import asyncio

    coro = qt_app.engine.worker.run_coro.call_args[0][0]
    result = asyncio.run(coro)
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    return result


def test_run_export_expands_the_filename_template(qt_app, tmp_path):
    _one_clip(qt_app, tmp_path)
    values = {"out_dir": str(tmp_path / "out"), "filename": "{project}_{range}", "format": "wav",
              "srt": False, "keep_clip_files": False}

    assert run_export(qt_app, values, range_label="Intro to Ch1") is True

    result = _scheduled_result(qt_app)
    assert os.path.basename(result.audio_path) == "Untitled_Intro to Ch1.wav"
    assert qt_app.project_settings["export"]["filename"] == "{project}_{range}"


def test_run_export_passes_the_output_rate_to_the_mixdown(qt_app, tmp_path):
    _one_clip(qt_app, tmp_path)
    values = {"out_dir": str(tmp_path / "out"), "filename": "mix", "format": "wav", "sample_rate": 48000,
              "srt": False, "keep_clip_files": False}

    assert run_export(qt_app, values) is True

    result = _scheduled_result(qt_app)
    assert sf.info(result.audio_path).samplerate == 48000
    assert result.duration_s == pytest.approx(1.0, abs=0.01)


def _existing_mix(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "mix.wav").write_bytes(b"old")
    return out


def test_run_export_asks_before_overwriting_and_can_cancel(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt.docks import export_dialog

    _one_clip(qt_app, tmp_path)
    out = _existing_mix(tmp_path)
    asked = []
    monkeypatch.setattr(export_dialog, "_ask_existing", lambda parent, path: asked.append(path) or None)

    assert run_export(qt_app, {"out_dir": str(out), "filename": "mix", "format": "wav",
                                  "srt": False, "keep_clip_files": False}) is False

    assert [os.path.basename(p) for p in asked] == ["mix.wav"]
    assert (out / "mix.wav").read_bytes() == b"old"
    assert not qt_app.is_busy()


def test_run_export_replace_overwrites_the_file(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt.docks import export_dialog

    _one_clip(qt_app, tmp_path)
    out = _existing_mix(tmp_path)
    monkeypatch.setattr(export_dialog, "_ask_existing", lambda parent, path: "replace")

    assert run_export(qt_app, {"out_dir": str(out), "filename": "mix", "format": "wav",
                                  "srt": False, "keep_clip_files": False}) is True

    result = _scheduled_result(qt_app)
    assert result.audio_path == str(out / "mix.wav")
    assert sf.info(result.audio_path).frames > 0


def test_run_export_add_number_keeps_the_old_file_and_numbers_the_extras(qt_app, tmp_path, monkeypatch):
    from kokoro_gui.qt.docks import export_dialog

    _one_clip(qt_app, tmp_path)
    out = _existing_mix(tmp_path)
    (out / "mix (2).wav").write_bytes(b"older")
    monkeypatch.setattr(export_dialog, "_ask_existing", lambda parent, path: "number")

    assert run_export(qt_app, {"out_dir": str(out), "filename": "mix", "format": "wav", "srt": True,
                                  "keep_clip_files": False}) is True

    result = _scheduled_result(qt_app)
    assert os.path.basename(result.audio_path) == "mix (3).wav"
    assert os.path.basename(result.srt_path) == "mix (3).srt"
    assert (out / "mix.wav").read_bytes() == b"old" and (out / "mix (2).wav").read_bytes() == b"older"
    assert qt_app.project_settings["export"]["filename"] == "mix"


def test_a_filename_template_reaches_the_engine_config_expanded(qt_app):
    qt_app.project_settings["export"] = {"filename": "{project}-{range}"}
    assert qt_app._assemble_config()["filename"] == "Untitled-full"


# -- stems (plan 15) ------------------------------------------------------------------------


def _two_voices(qt_app, tmp_path):
    """The default character says "hello", Bob says "world"; both rendered, one second each."""
    from kokoro_gui.daw.models import Character

    qt_app.document.settings["gap_s"] = 0.0
    _type(qt_app.editor, "hello world")
    _generated_clip(qt_app, tmp_path, 0, 5, seconds=1.0, name="a")
    bob = Character.from_preset_dict("Bob", {"voice": "am_michael"})
    qt_app.document.characters.append(bob)
    _generated_clip(qt_app, tmp_path, 6, 11, seconds=1.0, name="b", character=bob)


def test_export_dialog_has_the_stem_fields_on_the_extras_tab_and_remembers_them(qt_app):
    dialog = ExportDialog(qt_app)
    assert dialog.stems_combo.currentData() is None and not dialog.dialogue_stem_check.isChecked()
    assert [dialog.stems_combo.itemData(i) for i in range(dialog.stems_combo.count())] == [None, "track", "character"]
    for widget in (dialog.stems_combo, dialog.dialogue_stem_check):
        page = widget
        while page is not None and dialog.tabs.indexOf(page) < 0:
            page = page.parentWidget()
        assert dialog.tabs.tabText(dialog.tabs.indexOf(page)) == "Extras"

    dialog.stems_combo.setCurrentIndex(dialog.stems_combo.findData("character"))
    dialog.dialogue_stem_check.setChecked(True)
    qt_app.project_settings["export"] = dialog.values()

    again = ExportDialog(qt_app)
    assert again.stems_combo.currentData() == "character" and again.dialogue_stem_check.isChecked()


def test_export_defaults_ignore_a_bad_stems_value(qt_app):
    qt_app.project_settings["export"] = {"stems": "drums", "dialogue_stem": "yes"}
    values = export_defaults(qt_app)
    assert values["stems"] is None and values["dialogue_stem"] is True


@pytest.mark.parametrize("mode, names", [
    ("track", None),
    ("character", ["mix_Default.wav", "mix_Bob.wav", "mix_Dialogue.wav"]),
])
def test_a_stem_export_writes_the_expected_files(qt_app, tmp_path, mode, names):
    _two_voices(qt_app, tmp_path)
    values = dict(export_defaults(qt_app), out_dir=str(tmp_path / "out"), filename="mix", format="wav",
                  srt=False, keep_clip_files=False, stems=mode, dialogue_stem=True)

    assert not qt_app.document.dirty_clips(), [(c.character_id, c.segments) for c in qt_app.document.dirty_clips()]
    assert run_export(qt_app, values) is True
    assert qt_app.project_settings["export"] == values

    import asyncio

    result = asyncio.run(qt_app.engine.worker.run_coro.call_args[0][0])
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    written = sorted(os.listdir(tmp_path / "out"))
    stems = [os.path.basename(p) for p in result.stem_files]
    if names is not None:
        assert stems == names
    else:
        assert len(stems) >= 3 and stems[-1] == "mix_Dialogue.wav"
    assert written == sorted(["mix.wav", *stems])
    assert {sf.info(p).frames for p in result.stem_files} == {sf.info(result.audio_path).frames}
    assert f"{len(stems)} stems" in qt_app.transport_dock.status_text()


# -- transcripts and chapters (plan 16) -----------------------------------------------------


def test_export_dialog_has_the_text_file_checkboxes_on_the_extras_tab_and_remembers_them(qt_app):
    from kokoro_gui.daw.transcripts import TEXT_EXTRAS

    dialog = ExportDialog(qt_app)
    assert list(dialog.extra_checks) == list(TEXT_EXTRAS)
    assert not any(check.isChecked() for check in dialog.extra_checks.values())
    assert dialog.speakers_check.isChecked()
    for widget in (*dialog.extra_checks.values(), dialog.speakers_check):
        page = widget
        while page is not None and dialog.tabs.indexOf(page) < 0:
            page = page.parentWidget()
        assert dialog.tabs.tabText(dialog.tabs.indexOf(page)) == "Extras"

    dialog.extra_checks["show_notes"].setChecked(True)
    dialog.extra_checks["vtt"].setChecked(True)
    dialog.speakers_check.setChecked(False)
    values = dialog.values()
    assert values["extras"] == ["vtt", "show_notes"] and values["transcript_speakers"] is False
    qt_app.project_settings["export"] = values

    again = ExportDialog(qt_app)
    assert [k for k, c in again.extra_checks.items() if c.isChecked()] == ["vtt", "show_notes"]
    assert not again.speakers_check.isChecked()


def test_export_defaults_ignore_a_bad_extras_value(qt_app):
    qt_app.project_settings["export"] = {"extras": ["vtt", "bogus", 7], "transcript_speakers": 0}
    values = export_defaults(qt_app)
    assert values["extras"] == ["vtt"] and values["transcript_speakers"] is False
    qt_app.project_settings["export"] = {"extras": "vtt"}
    assert export_defaults(qt_app)["extras"] == []


def test_an_export_with_text_extras_writes_them_and_says_so(qt_app, tmp_path):
    _two_voices(qt_app, tmp_path)
    from kokoro_gui.daw import markers

    qt_app.document.settings["markers"], _m = markers.add_marker(qt_app.document.settings, 1.0, "Bob speaks",
                                                                 "Retake the breath.")
    values = dict(export_defaults(qt_app), out_dir=str(tmp_path / "out"), filename="mix", format="wav",
                  srt=False, keep_clip_files=False, extras=["vtt", "txt", "chapters_json", "show_notes"])

    assert run_export(qt_app, values) is True
    assert qt_app.project_settings["export"]["extras"] == ["vtt", "txt", "chapters_json", "show_notes"]

    import asyncio

    result = asyncio.run(qt_app.engine.worker.run_coro.call_args[0][0])
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    out = tmp_path / "out"
    assert sorted(os.listdir(out)) == ["mix.chapters.json", "mix.show-notes.md", "mix.txt", "mix.vtt", "mix.wav"]
    name = qt_app.document.characters[0].name
    vtt = (out / "mix.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT") and f"<v {name}>hello" in vtt and "<v Bob>world" in vtt
    assert (out / "mix.show-notes.md").read_text(encoding="utf-8") == "- (00:01) Bob speaks\n  Retake the breath.\n"
    assert "4 text files" in qt_app.transport_dock.status_text()


def test_turning_the_speaker_names_off_leaves_them_out_of_the_files(qt_app, tmp_path):
    _two_voices(qt_app, tmp_path)
    values = dict(export_defaults(qt_app), out_dir=str(tmp_path / "out"), filename="mix", format="wav",
                  srt=False, keep_clip_files=False, extras=["txt"], transcript_speakers=False)

    assert run_export(qt_app, values) is True

    import asyncio

    result = asyncio.run(qt_app.engine.worker.run_coro.call_args[0][0])
    qt_app.engine.worker.run_coro.return_value.set_result(result)
    assert (tmp_path / "out" / "mix.txt").read_text(encoding="utf-8") == "hello\n\nworld\n"
