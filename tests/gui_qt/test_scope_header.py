"""Plan 47 pass 2: what an edit and an action will touch, said before it
happens. The Voice & Clip title (grill PG30), the target headers and hints of
Voice & Clip and Audio FX, marks on values set on a clip, folding sections,
the Preview line, empty states, and tooltips on controls that are off."""
import json
import os

import pytest
from PySide6.QtGui import QTextCursor

import kokoro_gui.qt.app as qt_app_module
from kokoro_gui.daw.models import Character, Segment
from kokoro_gui.qt import preview_target, scope_target
from kokoro_gui.qt.characters_dialog import CharactersDialog
from kokoro_gui.qt.collapsible import CollapsibleSection
from kokoro_gui.qt.docks import settings_dock as settings_dock_module
from kokoro_gui.qt.docks import timeline_dock, transcript_dock, transport_dock


def _type(qt_app, text):
    cursor = qt_app.editor.textCursor()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.insertText(text)


def _clip(qt_app, text="hello world", start=0, end=None, character=None):
    _type(qt_app, text)
    character = character or qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, len(text) if end is None else end, character.id)
    qt_app.editor.rehighlight()
    return clip, character


def _titles(qt_app):
    return [dock.scope_header.title_label.text() for dock in (qt_app.settings_dock, qt_app.fx_dock)]


# -- the rename (PG30) -------------------------------------------------------------


def test_the_dock_is_voice_and_clip_and_keeps_its_object_name(qt_app):
    dock = qt_app.settings_dock
    assert dock.objectName() == "dock_settings"
    # Doubled: the title bar, the tab and the Panels menu all read a mnemonic.
    assert dock.windowTitle() == "Voice && Clip"
    assert dock.toggleViewAction().text() == "Voice && Clip"


def test_the_settings_window_keeps_its_name(qt_app):
    window = qt_app.open_settings_window()
    assert window.windowTitle() == "Settings"
    window.reject()


def test_a_floating_dock_and_its_panels_entry_read_right(qt_app):
    dock = qt_app.settings_dock
    dock.setFloating(True)
    assert dock.windowTitle() == "Voice & Clip"
    assert dock.toggleViewAction().text() == "Voice && Clip"
    dock.setFloating(False)
    assert dock.windowTitle() == "Voice && Clip"


# -- target headers ----------------------------------------------------------------


def test_both_headers_name_the_defaults_a_character_and_a_clip(qt_app):
    clip, character = _clip(qt_app)
    qt_app.selection.clear()
    assert _titles(qt_app) == ["Defaults", "Defaults"]

    qt_app.selection.select_character(character.id)
    assert _titles(qt_app) == [f"Character: {character.name}"] * 2

    qt_app.selection.select_clip(clip.id)
    assert _titles(qt_app) == [f"Clip: {character.name}"] * 2
    for dock in (qt_app.settings_dock, qt_app.fx_dock):
        assert dock.scope_header.detail_label.text() == "hello world"


def test_the_headers_are_outside_the_scroll_areas(qt_app):
    from PySide6.QtWidgets import QScrollArea

    for dock in (qt_app.settings_dock, qt_app.fx_dock):
        parent = dock.scope_header.parentWidget()
        while parent is not None and parent is not dock:
            assert not isinstance(parent, QScrollArea)
            parent = parent.parentWidget()


def test_several_clips_are_counted_with_their_characters(qt_app):
    first, alice = _clip(qt_app, "one. two.", 0, 4)
    bob = Character(name="Bob", preset_data={"voice": "af_bella"})
    qt_app.document.characters.append(bob)
    second = qt_app.document.assign_character_to_range(5, 9, bob.id)
    qt_app.selection.select_clips([first.id, second.id])
    assert _titles(qt_app) == [f"2 clips: {alice.name}, Bob"] * 2
    assert "all 2 clips, as one undo step" in qt_app.settings_dock.scope_hint.text()
    assert "all 2 clips, as one undo step" in qt_app.fx_dock.scope_hint.text()


def test_text_outside_a_clip_says_the_defaults_change(qt_app):
    _type(qt_app, "plain text with no clip")
    qt_app.selection.select_range(0, 5)
    assert _titles(qt_app) == ["Defaults", "Defaults"]
    assert "outside a clip" in qt_app.settings_dock.scope_hint.text()


def test_defaults_are_not_called_the_projects(qt_app):
    qt_app.selection.clear()
    for dock in (qt_app.settings_dock, qt_app.fx_dock):
        assert dock.scope_hint.text().startswith("Shared by every project.")


def test_a_subproject_is_named_above_the_target(qt_app):
    _type(qt_app, "Intro. Chapter text. Outro.")
    child = qt_app.new_subproject(7, 20, title="Chapter 1")
    qt_app.selection.select_clip(child.clip_id)
    for dock in (qt_app.settings_dock, qt_app.fx_dock):
        assert dock.scope_header.context_label.text() == "Subproject: Chapter 1"
        assert not dock.scope_header.context_label.isHidden()
    assert _titles(qt_app) == ["Defaults", "Defaults"]
    assert qt_app.settings_dock.scope_hint.text().startswith("Select a clip inside the subproject")

    # Entered, the timeline shows it too; the header still says where.
    qt_app.set_level(child)
    qt_app.selection.clear()
    assert qt_app.settings_dock.scope_header.context_label.text() == "Subproject: Chapter 1"

    qt_app.set_level(qt_app.root)
    assert qt_app.settings_dock.scope_header.context_label.isHidden()


def test_a_rename_and_a_reassignment_reach_the_header_without_a_new_selection(qt_app):
    clip, alice = _clip(qt_app)
    bob = Character(name="Bob", preset_data={"voice": "af_bella", "speed": 1.6})
    qt_app.document.characters.append(bob)
    qt_app.selection.select_clip(clip.id)

    alice.name = "Alicia"
    qt_app.on_characters_changed()
    assert _titles(qt_app) == ["Clip: Alicia"] * 2

    # A whole-clip assignment makes a new clip; the selection moves to it.
    qt_app.editor._push_assign_character(0, 11, bob.id)
    assert qt_app.selection.selected_clip_id in {c.id for c in qt_app.document.clips}
    assert _titles(qt_app) == ["Clip: Bob"] * 2
    # The inherited values follow the new character.
    assert qt_app.settings_dock.schema_form.values()["speed"] == 1.6


def test_a_deleted_clip_takes_the_tabs_back_to_the_defaults(qt_app):
    clip, _character = _clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    qt_app.document.clips.remove(clip)
    qt_app.refresh_timeline()
    assert _titles(qt_app) == ["Defaults", "Defaults"]


def test_a_long_name_is_cut_and_does_not_widen_the_dock(qt_app):
    character = qt_app.document.characters[0]
    before = qt_app.settings_dock.minimumSizeHint().width()
    character.name = "N" * 200
    qt_app.selection.select_character(character.id)
    title = qt_app.settings_dock.scope_header.title_label.text()
    assert len(title) <= len("Character: ") + scope_target.NAME_LIMIT
    assert qt_app.settings_dock.minimumSizeHint().width() <= max(before, 400)


def test_a_name_is_shown_as_typed_not_as_markup(qt_app):
    character = qt_app.document.characters[0]
    character.name = "<b>Al</b>"
    qt_app.selection.select_character(character.id)
    from PySide6.QtCore import Qt

    label = qt_app.settings_dock.scope_header.title_label
    assert label.text() == "Character: <b>Al</b>" and label.textFormat() == Qt.TextFormat.PlainText


# -- scope hints -------------------------------------------------------------------


def test_the_engine_row_says_it_changes_the_character_in_clip_scope(qt_app):
    clip, character = _clip(qt_app)
    dock = qt_app.settings_dock
    assert dock.engine_hint.isHidden()
    qt_app.selection.select_clip(clip.id)
    assert not dock.engine_hint.isHidden()
    assert f"changes the character {character.name}" in dock.engine_hint.text()
    qt_app.selection.select_character(character.id)
    assert dock.engine_hint.isHidden()


def test_a_library_characters_write_through_is_said(qt_app):
    clip, character = _clip(qt_app)
    CharactersDialog(qt_app).promote_current()
    qt_app.selection.select_character(character.id)
    assert "every project that uses it changes" in qt_app.settings_dock.scope_hint.text()
    qt_app.selection.select_clip(clip.id)
    assert "every project that uses it changes" in qt_app.settings_dock.engine_hint.text()


@pytest.mark.parametrize("scope,words", [("local", ""), ("missing", "only this project changes"),
                                         ("project", "this project's subprojects"),
                                         ("global", "every project that uses it")])
def test_library_note_follows_the_characters_scope(qt_app, monkeypatch, scope, words):
    monkeypatch.setattr(qt_app, "character_scope", lambda _c: scope)
    note = scope_target.library_note(qt_app, qt_app.document.characters[0])
    assert (words in note) if words else note == ""


def test_values_set_on_the_clip_are_marked_and_inherited_ones_say_from_where(qt_app):
    clip, character = _clip(qt_app)
    character.preset_data = {"voice": "af_bella"}
    qt_app.selection.select_clip(clip.id)
    dock = qt_app.settings_dock
    assert dock.scope_hint.text().startswith("Nothing is set on this clip")
    voice, speed = dock.schema_form.label_for("voice"), dock.schema_form.label_for("speed")
    assert voice.toolTip() == f"From {character.name}." and speed.toolTip() == "From the defaults."
    assert speed.text() == "Speed"

    dock.schema_form.widget_for("speed").setValue(1.3)
    dock.volume_spin.setValue(0.7)
    dock.normalize_check.setChecked(True)

    assert speed.text() == "Speed (clip)" and speed.property("overridden") is True
    assert dock._hand_built_labels["volume"].text() == "Volume (clip):"
    assert dock.normalize_check.text() == "Normalize (clip)"
    assert dock.trim_check.text() == "Trim Silence"
    assert dock.scope_hint.text().startswith("Set on this clip: Speed, Volume and Normalize. The rest follows")

    qt_app.selection.clear()
    assert dock._hand_built_labels["volume"].text() == "Volume:"
    assert dock._hand_built_labels["pitch"].text() == "Pitch (st):"
    assert dock.normalize_check.text() == "Normalize"


def test_several_clips_mark_some_or_all(qt_app):
    first, character = _clip(qt_app, "one. two.", 0, 4)
    second = qt_app.document.assign_character_to_range(5, 9, character.id)
    first.overrides["volume"] = 0.5
    first.overrides["pitch"] = 1.0
    second.overrides["pitch"] = 1.0
    qt_app.selection.select_clips([first.id, second.id])
    labels = qt_app.settings_dock._hand_built_labels
    assert labels["volume"].text() == "Volume (some clips):"
    assert labels["pitch"].text() == "Pitch (st) (clips):"


def test_fx_hints_say_where_an_edit_goes(qt_app, monkeypatch):
    clip, character = _clip(qt_app)
    fx = qt_app.fx_dock
    qt_app.selection.select_character(character.id)
    assert fx.scope_hint.text() == f"{character.name} has no FX yet. An edit saves one under that name."
    character.preset_data["fx_preset"] = "Radio"
    qt_app.selection.clear()
    qt_app.selection.select_character(character.id)
    assert fx.scope_hint.text() == 'Edits are saved to the FX "Radio", so everything that uses it changes.'
    qt_app.selection.select_clip(clip.id)
    assert fx.scope_hint.text() == 'This clip follows the FX "Radio". An edit gives it its own FX.'
    fx._value_widgets["gain_db"].setValue(4.0)
    fx._flush_clip_edit()
    assert fx.scope_hint.text() == "This clip has its own FX. Edits change this clip only."


# -- groups ------------------------------------------------------------------------


def test_a_folded_section_keeps_its_values(qtbot):
    section = CollapsibleSection("EQ & Filters")
    qtbot.addWidget(section)
    seen = []
    section.collapsedChanged.connect(seen.append)
    section.header.click()
    assert section.is_collapsed() and section.body.isHidden() and seen == [True]
    section.set_state("EQ on")
    assert section.header.text() == "EQ && Filters: EQ on"
    assert section.header.accessibleName() == "EQ & Filters: EQ on"


def test_settings_groups_are_voice_audio_and_clip(qt_app):
    dock = qt_app.settings_dock
    assert [g.title() for g in (dock.schema_group, dock.audio_group, dock.scope_group)] == ["Voice", "Audio", "Clip"]
    assert dock.schema_group.isAncestorOf(dock.engine_combo)
    assert dock.apply_fx_check.text() == "Enable FX"


def test_clip_status_stays_in_view_and_the_rest_folds(qt_app):
    clip, _character = _clip(qt_app)
    clip.source_text = "bonjour"
    clip.gap_before_s = 0.5
    qt_app.selection.select_clip(clip.id)
    fields = qt_app.settings_dock.scope_fields
    for key in ("status", "note", "take"):
        assert not fields.timing_section.isAncestorOf(fields.widgets[key])
        assert not fields.source_section.isAncestorOf(fields.widgets[key])
    assert fields.timing_section.isAncestorOf(fields.widgets["gap_before_s"])
    assert fields.source_section.isAncestorOf(fields.widgets["reference_range"])
    assert fields.timing_section.is_collapsed() and fields.source_section.is_collapsed()
    assert fields.timing_section.state == "gap 0.5 s" and fields.source_section.state == "source text"

    fields.timing_section.set_collapsed(False)
    qt_app.selection.clear()
    qt_app.selection.select_clip(clip.id)
    assert not fields.timing_section.is_collapsed()


def test_the_model_group_folds_and_stays_as_left(qt_app, monkeypatch):
    from kokoro_gui.engines import audio8_tts

    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    character = qt_app.document.characters[0]
    assert qt_app.set_character_engine(character, "audio8")
    qt_app.selection.select_character(character.id)
    dock = qt_app.settings_dock
    assert dock.schema_form.sections["Model"].is_collapsed()
    field = dock.schema_form.widget_for("temperature")
    assert not field.isEnabled() and "Edit it with nothing selected" in field.toolTip()
    dock.schema_form.sections["Model"].set_collapsed(False)
    dock.rebuild_schema_form()
    assert not dock.schema_form.sections["Model"].is_collapsed()


def test_fx_groups_show_what_is_on_and_fold_without_turning_it_off(qt_app):
    fx = qt_app.fx_dock
    dynamics = fx.group_sections["Dynamics"]
    assert all(s.is_collapsed() and s.state == "" for s in fx.group_sections.values())

    fx._enabled_checks["comp_enabled"].setChecked(True)
    fx._value_widgets["eq_bass"].setValue(3.0)
    assert dynamics.state == "Compressor on"
    assert fx.group_sections["EQ & Filters"].state == "EQ on"

    dynamics.set_collapsed(False)
    dynamics.set_collapsed(True)
    assert fx.project_fx_state()["comp_enabled"] is True


def test_fx_group_state_shows_mixed_and_a_missing_file(qt_app):
    first, character = _clip(qt_app, "one. two.", 0, 4)
    second = qt_app.document.assign_character_to_range(5, 9, character.id)
    first.fx_override = {"delay_enabled": True}
    qt_app.selection.select_clips([first.id, second.id])
    assert qt_app.fx_dock.group_sections["Spatial & Time"].state == "Delay mixed"
    first.fx_override = {"convolution_ir": "gone", "convolution_mix": 0.5}
    qt_app.selection.select_clip(first.id)
    assert qt_app.fx_dock.group_sections["Spatial & Time"].state == "Convolution Reverb missing file"


# -- Preview, Generate and Play ----------------------------------------------------


def test_the_preview_line_names_what_it_speaks_and_as_whom(qt_app):
    label = qt_app.transport_dock.preview_label
    name = qt_app.document.characters[0].name
    assert label.text() == f"Preview speaks a sample line as {name}."
    _type(qt_app, "Some words here.")
    assert label.text() == f"Preview speaks the transcript as {name}."
    cursor = qt_app.editor.textCursor()
    cursor.setPosition(0)
    cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
    qt_app.editor.setTextCursor(cursor)
    assert label.text() == f"Preview speaks the selected text as {name}."
    _type(qt_app, "word " * 200)
    assert label.text() == f"Preview speaks the start of the transcript as {name}."


def test_the_preview_line_follows_the_active_character(qt_app):
    clip, _alice = _clip(qt_app)
    bob = Character(name="Bob", preset_data={"voice": "af_bella"})
    qt_app.document.characters.append(bob)
    qt_app.selection.select_character(bob.id)
    assert qt_app.transport_dock.preview_label.text().endswith("as Bob.")


@pytest.mark.parametrize("text,select,expected", [
    ("", False, preview_target.SAMPLE_TEXT),
    ("  \n ", False, preview_target.SAMPLE_TEXT),
    ("All of it.", False, "All of it."),
    ("All of it.", True, "All"),
    ("x" * 3000, False, "x" * 1000),
])
def test_preview_speaks_what_the_line_says(qt_app, text, select, expected):
    _type(qt_app, text)
    if select:
        cursor = qt_app.editor.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        qt_app.editor.setTextCursor(cursor)
    qt_app.preview_conversion()
    assert qt_app.engine.generate_preview.call_args[0][0] == expected


def test_play_preview_and_generate_tell_themselves_apart(qt_app):
    dock = qt_app.transport_dock
    assert "timeline's audio" in dock.play_btn.toolTip()
    assert "does not play the timeline's audio" in dock.preview_btn.toolTip()
    assert "stale" in dock.generate_btn.toolTip()
    assert not any("dirty" in text.lower() for text in (
        dock.generate_btn.toolTip(), *(a.text() for a in dock.generate_menu.actions())))


# -- empty states ------------------------------------------------------------------


def test_an_empty_transcript_offers_import_text(qt_app, monkeypatch):
    strip = qt_app.transcript_dock.empty_state
    assert strip.text() == transcript_dock.EMPTY_TEXT and not strip.isHidden()
    asked = []
    monkeypatch.setattr(qt_app, "import_text", lambda path, target=None: asked.append(path))
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("book.txt", "")))
    strip.run("import")
    assert asked == ["book.txt"]
    _type(qt_app, "words")
    assert strip.isHidden()


def test_the_timeline_says_what_is_missing(qt_app):
    strip = qt_app.timeline_dock.empty_state
    assert strip.text() == timeline_dock.EMPTY_NO_TEXT
    _type(qt_app, "some words")
    qt_app.refresh_timeline()
    assert strip.text() == timeline_dock.EMPTY_NO_CLIPS
    clip = qt_app.document.assign_character_to_range(0, 10, qt_app.document.characters[0].id)
    qt_app.refresh_timeline()
    assert strip.text() == timeline_dock.EMPTY_NO_AUDIO
    clip.segments = [Segment(order_index=0, text="some words", cache_key="k", duration=1.0)]
    qt_app.timeline_dock.refresh_empty_state()
    assert strip.isHidden()


def test_the_timelines_links_press_the_transports_own_controls(qt_app):
    pressed = []
    dock = qt_app.transport_dock
    dock.generate_btn.clicked.disconnect()
    dock.generate_btn.clicked.connect(lambda: pressed.append("generate"))
    dock.auto_split_action.triggered.disconnect()
    dock.auto_split_action.triggered.connect(lambda: pressed.append("split"))
    qt_app.timeline_dock.empty_state.run("generate")
    qt_app.timeline_dock.empty_state.run("split")
    dock.set_generate_enabled(False)
    qt_app.timeline_dock.empty_state.run("generate")
    assert pressed == ["generate", "split"]


def test_manage_beside_the_character_combo_opens_the_dialog(qt_app, monkeypatch):
    opened = []
    monkeypatch.setattr(qt_app, "open_characters_dialog", lambda: opened.append(True))
    qt_app.transcript_dock.manage_btn.click()
    assert opened == [True]


# -- controls that are off ---------------------------------------------------------


def test_disabled_transport_controls_say_why(qt_app):
    dock = qt_app.transport_dock
    assert dock.cancel_btn.toolTip() == transport_dock.CANCEL_IDLE_TIP
    assert dock.queue_pause_btn.toolTip() == transport_dock.PAUSE_QUEUE_IDLE_TIP
    assert all(b.toolTip() == transport_dock.NO_SOURCE_TRACK_TIP for b in dock.monitor_buttons.values())
    assert "Nothing is playing" in dock.pause_btn.toolTip()

    dock.set_busy(True)
    assert dock.preview_btn.toolTip() == transport_dock.PREVIEW_BUSY_TIP
    assert dock.generate_btn.toolTip() == transport_dock.GENERATE_BUSY_TIP
    assert dock.cancel_btn.toolTip() == transport_dock.CANCEL_TIP
    dock.set_queue_state(pending=True, paused=False)
    assert dock.queue_pause_btn.toolTip() == transport_dock.PAUSE_QUEUE_TIP
    dock.set_busy(False)
    assert dock.preview_btn.toolTip() == preview_target.TOOLTIP

    dock._refresh_generate_menu()
    assert not dock.generate_selection_action.isEnabled()
    assert dock.generate_selection_action.toolTip() == transport_dock.NO_STALE_SELECTED_TIP


def test_the_whole_job_status_is_in_the_bars_tooltip(qt_app):
    dock = qt_app.transport_dock
    dock.set_status("Generating a very long line of status")
    dock.set_progress(40, "A: 40/300 in background")
    assert "Generating a very long line of status" in dock.progress_bar.toolTip()
    assert "A: 40/300 in background" in dock.progress_bar.toolTip()


def test_disabled_dock_controls_say_why(qt_app):
    _type(qt_app, "no clip here")
    qt_app.transcript_dock.sync_header()
    assert qt_app.transcript_dock.fx_combo.toolTip() == transcript_dock.FX_NO_CLIP_TIP

    dock = qt_app.settings_dock
    dock.set_generating(True)
    assert not dock.volume_spin.isEnabled()
    assert dock.volume_spin.toolTip() == settings_dock_module.BUSY_FIELD_TIP
    dock.set_generating(False)
    assert dock.volume_spin.isEnabled() and dock.volume_spin.toolTip() == ""

    clip = qt_app.document.assign_character_to_range(0, 2, qt_app.document.characters[0].id)
    clip.character_id = "gone"
    qt_app.selection.select_clip(clip.id)
    assert not dock.engine_combo.isEnabled()
    assert dock.engine_combo.toolTip().startswith("This clip has no character.")
    assert dock.scope_header.title_label.text() == "Clip: no character"


def test_greyed_menu_entries_say_why(qt_app):
    qt_app._sync_add_credits_action()
    assert not qt_app.add_credits_action.isEnabled()
    assert qt_app.add_credits_action.toolTip() == "Add some text or a subproject first."
    qt_app._sync_split_join_actions()
    assert not qt_app.split_clip_action.isEnabled()
    assert qt_app.split_clip_action.toolTip() == "Put the playhead on a clip."
    assert qt_app.join_clip_action.toolTip() == "Select a clip with another clip after it."
    qt_app._sync_show_in_folder_actions()
    assert qt_app.show_project_file_action.toolTip() == "The project has no file yet. Save it first."
    assert qt_app.file_menu.toolTipsVisible() and qt_app.edit_menu.toolTipsVisible()
    _type(qt_app, "words")
    qt_app._sync_add_credits_action()
    assert qt_app.add_credits_action.toolTip() == qt_app_module.ADD_CREDITS_TIP


# -- saved layouts -----------------------------------------------------------------


def test_a_layout_saved_before_the_rename_still_restores(qt_app):
    """A layout names docks by object name, so one saved with the dock closed
    under its old title closes it under the new one."""
    qt_app.settings_dock.setWindowTitle("Settings")
    qt_app.settings_dock.hide()
    state = qt_app.saveState()
    qt_app.settings_dock.show()
    qt_app.settings_dock._apply_title(False)
    assert qt_app.restoreState(state)
    assert qt_app.settings_dock.isHidden()
    assert qt_app.settings_dock.windowTitle() == "Voice && Clip"


def test_header_text_is_never_saved(qt_app):
    clip, _character = _clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    qt_app.settings_dock.scope_fields.timing_section.set_collapsed(False)
    qt_app.save_settings()
    with open(qt_app_module.CONFIG_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    assert not any("collaps" in key or "section" in key for key in saved)
    assert os.path.basename(qt_app_module.CONFIG_FILE) == "config_qt.json"
