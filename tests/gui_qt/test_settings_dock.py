"""Tests for kokoro_gui/qt/docks/settings_dock.py's SettingsDock - item 2
("Settings panel rescoping") of the DAW-for-text redesign's remaining-work
roadmap. Three states keyed off `SelectionModel.kind` ("none"/"clip"/
"character"), each sourcing/writing a different backing store - see that
module's docstring for the full contract.
"""
from kokoro_gui.daw import dirty
from kokoro_gui.daw.models import Segment
from kokoro_gui.qt import spec


def _make_clip(qt_app, start=0, end=5, text="hello world"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(start, end, character.id)
    return clip, character


# ---------------------------------------------------------------------------
# "none" mode - regression-pin today's migrated (pre-rescoping) behavior
# ---------------------------------------------------------------------------

def test_none_mode_is_the_default_selection_state(qt_app):
    assert qt_app.selection.kind == "none"
    assert qt_app.settings_dock._mode == "none"


def test_none_mode_get_state_reflects_live_widget_edits(qt_app):
    qt_app.settings_dock.volume_spin.setValue(1.75)
    qt_app.settings_dock.schema_form.set_values({"speed": 1.3})

    state = qt_app.settings_dock.get_state()

    assert state["volume"] == 1.75
    assert state["speed"] == 1.3


def test_none_mode_fields_are_all_enabled(qt_app):
    for key in ("lang_code", "voice", "speed", "num_threads", "caching"):
        widget = qt_app.settings_dock.schema_form.widget_for(key)
        assert widget is not None
        assert widget.isEnabled() is True


# ---------------------------------------------------------------------------
# "clip" mode
# ---------------------------------------------------------------------------

def test_clip_mode_renders_effective_config_for_clip(qt_app):
    clip, character = _make_clip(qt_app)
    character.preset_data["voice"] = "af_bella"
    character.preset_data["speed"] = 1.4

    qt_app.selection.select_clip(clip.id)

    assert qt_app.settings_dock._mode == "clip"
    values = qt_app.settings_dock.schema_form.values()
    assert values["voice"] == "af_bella"
    assert values["speed"] == 1.4


def test_clip_mode_disables_non_allowed_preset_fields_but_not_allowed_ones(qt_app):
    clip, _character = _make_clip(qt_app)

    qt_app.selection.select_clip(clip.id)

    threads_widget = qt_app.settings_dock.schema_form.widget_for("num_threads")
    caching_widget = qt_app.settings_dock.schema_form.widget_for("caching")
    lang_widget = qt_app.settings_dock.schema_form.widget_for("lang_code")
    voice_widget = qt_app.settings_dock.schema_form.widget_for("voice")
    speed_widget = qt_app.settings_dock.schema_form.widget_for("speed")

    assert threads_widget.isEnabled() is False
    assert caching_widget.isEnabled() is False
    assert lang_widget.isEnabled() is False
    assert voice_widget.isEnabled() is True
    assert speed_widget.isEnabled() is True


def test_editing_clip_mode_schema_field_writes_to_overrides_not_settings(qt_app):
    clip, _character = _make_clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    original_settings_speed = qt_app.settings.get("speed")

    qt_app.settings_dock.schema_form.set_values({"speed": 1.9})
    # set_values() drives the same on_change path as a real user edit.
    qt_app.settings_dock._on_schema_field_changed("speed", 1.9)

    assert clip.overrides.get("speed") == 1.9
    assert qt_app.settings.get("speed") == original_settings_speed


def test_editing_clip_mode_hand_built_widget_writes_to_overrides_not_settings(qt_app):
    clip, _character = _make_clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    original_settings_volume = qt_app.settings.get("volume")

    qt_app.settings_dock.volume_spin.setValue(1.6)

    assert clip.overrides.get("volume") == 1.6
    assert qt_app.settings.get("volume") == original_settings_volume


def test_stale_clip_selection_falls_back_to_none(qt_app):
    qt_app.selection.select_clip("does-not-exist")

    assert qt_app.settings_dock._mode == "none"
    assert qt_app.settings_dock.schema_form is not None


# ---------------------------------------------------------------------------
# "character" mode + Q7 propagation
# ---------------------------------------------------------------------------

def test_character_mode_renders_preset_data(qt_app):
    character = qt_app.document.characters[0]
    character.preset_data["voice"] = "af_sarah"

    qt_app.selection.select_character(character.id)

    assert qt_app.settings_dock._mode == "character"
    assert qt_app.settings_dock.schema_form.values()["voice"] == "af_sarah"


def test_character_edit_propagates_to_non_overridden_clip_but_not_overridden_one(qt_app):
    qt_app.document.text = "hello world here"
    character = qt_app.document.characters[0]
    clip_no_override = qt_app.document.assign_character_to_range(0, 5, character.id)
    clip_with_override = qt_app.document.assign_character_to_range(6, 11, character.id)
    clip_with_override.overrides["speed"] = 2.0

    qt_app.selection.select_character(character.id)
    qt_app.settings_dock.schema_form.set_values({"speed": 1.7})
    qt_app.settings_dock._on_schema_field_changed("speed", 1.7)

    assert character.preset_data["speed"] == 1.7
    assert qt_app.document.effective_config_for_clip(clip_no_override)["speed"] == 1.7
    assert qt_app.document.effective_config_for_clip(clip_with_override)["speed"] == 2.0


def test_stale_character_selection_falls_back_to_none(qt_app):
    qt_app.selection.select_character("does-not-exist")

    assert qt_app.settings_dock._mode == "none"
    assert qt_app.settings_dock.schema_form is not None


# ---------------------------------------------------------------------------
# dirty-state flips with no explicit dirty-marking call
# ---------------------------------------------------------------------------

def test_editing_clip_config_flips_dirty_state_with_no_explicit_marking(qt_app):
    clip, _character = _make_clip(qt_app)
    text = qt_app.document.clip_text(clip)
    config = qt_app.document.effective_config_for_clip(clip)
    cache_hash = dirty.compute_expected_cache_hash(text, config)
    clip.segments = [Segment(order_index=0, text=text, cache_key=cache_hash)]
    assert dirty.is_clip_dirty(clip, text, qt_app.document.effective_config_for_clip(clip)) is False

    qt_app.selection.select_clip(clip.id)
    qt_app.settings_dock.schema_form.set_values({"speed": 1.5})
    qt_app.settings_dock._on_schema_field_changed("speed", 1.5)

    assert dirty.is_clip_dirty(
        clip, qt_app.document.clip_text(clip), qt_app.document.effective_config_for_clip(clip)
    ) is True


# ---------------------------------------------------------------------------
# GenerationDock.get_state() still supplies everything app.py's config
# assembly reads directly by key.
# ---------------------------------------------------------------------------

def test_generation_dock_state_covers_base_keys_minus_settings_owned(qt_app):
    state = qt_app.settings_dock.get_state()
    settings_owned = {"engine_id", "time_id", "lexicon"}
    # Output/format/subtitles/keep-segments live in the Export dialog now
    # (kokoro_gui/qt/docks/export_dialog.py), not the Settings tab.
    export_owned = {"filename", "out_dir", "separate", "combine", "export_subtitles"}
    assert set(state.keys()) | settings_owned | export_owned == set(spec.GENERATION_BASE_KEYS)


def test_assemble_config_does_not_raise_key_error(qt_app):
    config = qt_app._assemble_config()
    assert config["voice"]


def test_assemble_clip_config_does_not_raise_key_error(qt_app):
    clip, _character = _make_clip(qt_app)
    config = qt_app._assemble_clip_config(clip)
    assert config["voice"]


def test_assemble_config_unaffected_by_a_selected_clip(qt_app):
    """Whole-document config assembly must keep using project-wide defaults
    even while a clip happens to be selected in the UI, not whatever that
    clip's character happens to resolve to."""
    qt_app.settings_dock.schema_form.set_values({"voice": "af_sarah"})
    clip, character = _make_clip(qt_app)
    character.preset_data["voice"] = "af_bella"

    qt_app.selection.select_clip(clip.id)
    config = qt_app._assemble_config()

    assert config["voice"] == "af_sarah"



def test_project_scope_pacing_fields_write_document_settings_undoably(qt_app):
    dock = qt_app.settings_dock
    qt_app.selection.clear()
    fields = dock.scope_fields.widgets
    assert dock.scope_group.title() == "Project"

    fields["gap_s"].setValue(0.5)
    fields["gap_s"].editingFinished.emit()
    assert qt_app.document.settings["gap_s"] == 0.5
    fields["auto_crossfade"].setChecked(True)
    assert qt_app.document.settings["auto_crossfade"] is True

    qt_app.document.undo_stack.undo()
    qt_app.document.undo_stack.undo()
    assert "gap_s" not in qt_app.document.settings
    assert "auto_crossfade" not in qt_app.document.settings


def test_project_scope_ripple_checkbox_defaults_on_and_writes_the_setting(qt_app):
    qt_app.selection.clear()
    ripple = qt_app.settings_dock.scope_fields.widgets["ripple"]
    assert ripple.isChecked()
    ripple.setChecked(False)
    assert qt_app.document.settings["ripple"] is False
    qt_app.document.undo_stack.undo()
    assert "ripple" not in qt_app.document.settings


def test_project_scope_track_layout_switches_both_ways(qt_app):
    """Grill PR4: Unified puts clips on "Lane N" tracks by the lane rule;
    One per character puts them back on character tracks. Each switch is
    one undo step with its relane."""
    from kokoro_gui.daw.models import Character

    doc = qt_app.document
    alice = doc.characters[0]
    bob = Character.from_preset_dict("Bob", {})
    doc.characters.append(bob)
    doc.text = "one two three"
    first = doc.assign_character_to_range(0, 3, alice.id)
    second = doc.assign_character_to_range(4, 7, bob.id)
    third = doc.assign_character_to_range(8, 13, alice.id)
    character_track_ids = [c.track_id for c in (first, second, third)]
    qt_app.selection.clear()
    fields = qt_app.settings_dock.scope_fields.widgets
    assert fields["track_layout"].currentData() == "character"
    assert not fields["track_lanes"].isEnabled()

    fields["track_lanes"].setValue(2)
    fields["track_layout"].setCurrentIndex(fields["track_layout"].findData("unified"))
    fields["track_layout"].activated.emit(fields["track_layout"].currentIndex())

    assert doc.settings["track_layout"] == {"mode": "unified", "lanes": 2}
    assert [doc.get_track(c.track_id).name for c in (first, second, third)] == ["Lane 1", "Lane 2", "Lane 1"]
    header_names = [t.name for t in doc.used_tracks()]
    assert header_names == ["Lane 1", "Lane 2"]

    fields = qt_app.settings_dock.scope_fields.widgets
    fields["track_layout"].setCurrentIndex(fields["track_layout"].findData("character"))
    fields["track_layout"].activated.emit(fields["track_layout"].currentIndex())
    assert [c.track_id for c in (first, second, third)] == character_track_ids

    doc.undo_stack.undo()
    assert [doc.get_track(c.track_id).lane for c in (first, second, third)] == [1, 2, 1]
    doc.undo_stack.undo()
    assert "track_layout" not in doc.settings
    assert [c.track_id for c in (first, second, third)] == character_track_ids


def test_project_scope_timecode_fields_store_one_dict(qt_app):
    dock = qt_app.settings_dock
    qt_app.selection.clear()
    fields = dock.scope_fields.widgets
    fields["tc_fps"].setCurrentIndex(fields["tc_fps"].findData(29.97))
    fields["tc_drop"].setChecked(True)
    fields["tc_start"].setText("01:00:00;00")
    fields["tc_start"].editingFinished.emit()
    fields["tc_enabled"].setChecked(True)

    tc = qt_app.document.settings["timecode"]
    assert tc == {"enabled": True, "frame_rate": 29.97, "start": "01:00:00;00", "drop_frame": True}
    assert qt_app.transport_dock.time_label.text().startswith("01:00:00;00")

    fields["tc_start"].setText("garbage")
    fields["tc_start"].editingFinished.emit()
    assert qt_app.document.settings["timecode"]["start"] == "01:00:00;00"


def test_clip_scope_fields_edit_status_note_gap_and_source_text(qt_app):
    qt_app.document.text = "Hello there friend."
    character = qt_app.document.characters[0]
    clip = qt_app.document.assign_character_to_range(0, 18, character.id)
    qt_app.selection.select_clip(clip.id)
    dock = qt_app.settings_dock
    fields = dock.scope_fields.widgets
    assert dock.scope_group.title() == "Clip"

    fields["status"].setCurrentIndex(fields["status"].findData("approved"))
    fields["status"].activated.emit(fields["status"].currentIndex())
    fields["note"].setText("breath at 2s")
    fields["note"].editingFinished.emit()
    fields["gap_before_s"].setValue(1.25)
    fields["gap_before_s"].editingFinished.emit()
    fields["source_edit"].setChecked(True)
    fields["source_text"].setPlainText("Bonjour mon ami.")
    fields["source_edit"].setChecked(False)

    assert (clip.status, clip.note, clip.gap_before_s) == ("approved", "breath at 2s", 1.25)
    assert clip.source_text == "Bonjour mon ami."
    assert fields["syllables"].text() == "Syllables: source 5 / dub 5"

    fields["gap_before_s"].setValue(fields["gap_before_s"].minimum())
    fields["gap_before_s"].editingFinished.emit()
    assert clip.gap_before_s is None


def test_clip_scope_take_combo_picks_a_parked_take(qt_app):
    from kokoro_gui.daw.models import Segment

    qt_app.document.text = "hello"
    clip = qt_app.document.assign_character_to_range(0, 5, qt_app.document.characters[0].id)
    clip.segments = [Segment(text="hello", audio_path="t1.wav", duration=1.0)]
    clip.overrides["take"] = 1
    clip.takes = {0: [Segment(text="hello", audio_path="t0.wav", duration=2.0)]}
    qt_app.selection.select_clip(clip.id)
    take = qt_app.settings_dock.scope_fields.widgets["take"]
    assert [take.itemText(i) for i in range(take.count())] == ["Take 1 (2.0s)", "Take 2 (1.0s)"]

    take.setCurrentIndex(0)
    take.activated.emit(0)

    assert clip.segments[0].audio_path == "t0.wav"
    assert "take" not in clip.overrides


def test_syllable_count_is_rough_but_stable():
    from kokoro_gui.qt.docks.scope_fields import syllable_count

    assert syllable_count("Hello there friend.") == 5
    assert syllable_count("") == 0
    assert syllable_count("rhythm 42") == 1
