"""Tests for the Export dialog (section 6) and the Edit > Characters dialog
(UI13) of Claude/PLAN_ui_shell_redesign.md, plus the transport -> playhead
-> transcript follow chain (section 5) at the app level."""
import os

import numpy as np
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


def _generated_clip(qt_app, tmp_path, start, end, seconds=1.0, name="a"):
    alice = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, end, alice.id)
    path = str(tmp_path / f"{name}.wav")
    sf.write(path, np.full(int(24000 * seconds), 0.25, dtype=np.float32), 24000)
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
                      "channels": 2, "srt_words": False, "cue_sheet": False}


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
