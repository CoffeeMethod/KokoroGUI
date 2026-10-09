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
    for key in ("lang_code", "voice", "speed"):
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

    lang_widget = qt_app.settings_dock.schema_form.widget_for("lang_code")
    voice_widget = qt_app.settings_dock.schema_form.widget_for("voice")
    speed_widget = qt_app.settings_dock.schema_form.widget_for("speed")

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
    # Kept per engine (grill EN5): `app.engine_settings(engine_id)`.
    engine_owned = {"lang_code", "num_threads", "voice"}
    assert set(state.keys()) | settings_owned | export_owned | engine_owned == set(spec.GENERATION_BASE_KEYS)


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


def test_clip_scope_target_duration_edits_the_override_and_blank_clears_it(qt_app):
    qt_app.document.text = "Hello there friend."
    clip = qt_app.document.assign_character_to_range(0, 18, qt_app.document.characters[0].id)
    clip.overrides["target_duration_s"] = 1.5
    qt_app.selection.select_clip(clip.id)
    target = qt_app.settings_dock.scope_fields.widgets["target_duration_s"]
    stack = qt_app.document.undo_stack
    assert target.value() == 1.5

    steps = len(stack._undo)
    target.setValue(2.25)
    target.editingFinished.emit()
    assert clip.overrides["target_duration_s"] == 2.25
    assert len(stack._undo) == steps + 1
    target.editingFinished.emit()  # no change, no step
    assert len(stack._undo) == steps + 1

    target.setValue(target.minimum())
    target.editingFinished.emit()
    assert "target_duration_s" not in clip.overrides
    stack.undo()
    assert clip.overrides["target_duration_s"] == 2.25
    # Not a generation input: editing it never dirties the clip.
    assert "target_duration_s" not in qt_app._assemble_generation_config(clip)


def test_syllable_count_of_a_clip_ignores_its_tag():
    from kokoro_gui.engine.text_extraction import strip_markup
    from kokoro_gui.qt.docks.scope_fields import syllable_count

    assert syllable_count(strip_markup("[Alice:Radio]: Hello there friend.")) == 5


def test_syllable_count_is_rough_but_stable():
    from kokoro_gui.qt.docks.scope_fields import syllable_count

    assert syllable_count("Hello there friend.") == 5
    assert syllable_count("") == 0
    assert syllable_count("rhythm 42") == 1


# ---------------------------------------------------------------------------
# The Engine row (grill EN1, EN2)
# ---------------------------------------------------------------------------

def test_engine_row_in_project_scope_sets_the_engine_for_new_characters(qt_app):
    """Grill UI16: the voice fields below follow it, so they show what a
    new character gets."""
    dock = qt_app.settings_dock
    assert dock._mode == "none"
    assert dock.engine_label.text() == "Engine for new characters:"
    assert dock.engine_combo.isEnabled()
    assert dock.engine_combo.currentData() == "kokoro"

    dock.engine_combo.setCurrentIndex(dock.engine_combo.findData("dummy"))
    dock._on_engine_picked()

    assert qt_app.settings["default_engine"] == "dummy"
    assert qt_app.default_engine_id == "dummy"
    assert qt_app.document.characters[0].backend_id == "kokoro"
    assert dock.shown_backend().id == "dummy"
    assert dock.schema_form.values()["voice"] == "dummy"


def test_a_project_scope_voice_edit_lands_in_the_new_character_engines_bucket(qt_app):
    dock = qt_app.settings_dock
    dock.engine_combo.setCurrentIndex(dock.engine_combo.findData("dummy"))
    dock._on_engine_picked()
    dock.schema_form.set_values({"speed": 1.2})
    lang = dock.schema_form.widget_for("lang_code")
    lang.setCurrentIndex(lang.count() - 1)
    assert qt_app.engine_settings("dummy")["lang_code"] == lang.currentData()


def test_engine_row_in_clip_scope_switches_the_clips_character(qt_app, qtbot):
    clip, character = _make_clip(qt_app)
    qt_app.selection.select_clip(clip.id)
    dock = qt_app.settings_dock
    assert dock.engine_combo.currentData() == "kokoro"
    assert dock.engine_combo.isEnabled()

    dock.engine_combo.setCurrentIndex(dock.engine_combo.findData("dummy"))
    dock._on_engine_picked()
    qtbot.waitUntil(lambda: character.backend_id == "dummy")

    assert dock._mode == "clip"
    assert dock.engine_combo.currentData() == "dummy"
    assert dock.schema_form.values()["voice"] == "dummy"


def test_engine_row_in_character_scope_switches_the_characters_engine(qt_app, qtbot):
    character = qt_app.document.characters[0]
    qt_app.selection.select_character(character.id)
    dock = qt_app.settings_dock
    assert dock.engine_combo.isEnabled()

    dock.engine_combo.setCurrentIndex(dock.engine_combo.findData("dummy"))
    dock._on_engine_picked()
    # Deferred: the switch rebuilds the dock the combo sits in.
    assert character.backend_id == "kokoro"
    qtbot.waitUntil(lambda: character.backend_id == "dummy")

    assert dock.engine_combo.currentData() == "dummy"
    assert qt_app.backend.id == "dummy"
    assert dock.schema_form.values()["voice"] == "dummy"


def test_engine_switch_is_refused_while_a_job_runs(qt_app, monkeypatch):
    character = qt_app.document.characters[0]
    qt_app.selection.select_character(character.id)
    monkeypatch.setattr(qt_app, "is_busy", lambda: True)

    assert qt_app.settings_dock.pick_character_engine(character, "dummy") is False
    assert character.backend_id == "kokoro"
    assert qt_app.settings_dock.engine_combo.currentData() == "kokoro"


def test_a_voice_the_new_engine_lists_is_kept(qt_app):
    character = qt_app.document.characters[0]
    character.preset_data["voice"] = "bm_daniel"
    assert qt_app.set_character_engine(character, "dummy")
    assert character.preset_data["voice"] == "dummy"  # Dummy has no bm_daniel

    assert qt_app.set_character_engine(character, "kokoro")
    assert character.preset_data["voice"] == "af_heart"  # the schema default
    character.preset_data["voice"] = "bm_daniel"
    assert qt_app.set_character_engine(character, "kokoro") is True
    assert character.preset_data["voice"] == "bm_daniel"  # listed (British English)


def test_an_engine_with_no_voices_leaves_the_character_without_one(qt_app, monkeypatch, tmp_path):
    from kokoro_gui.engines import audio8_tts

    monkeypatch.setattr(audio8_tts, "_get_model", lambda: (object(), object()))
    monkeypatch.setattr(audio8_tts, "AUDIO8_REFS_DIR", str(tmp_path / "no_refs"))
    character = qt_app.document.characters[0]
    character.variants = {"angry": "angry_ref"}

    assert qt_app.set_character_engine(character, "audio8")

    assert "voice" not in character.preset_data
    assert character.variants == {"angry": "angry_ref"}


def test_clip_scope_overlap_field_sets_and_clears_the_override_undoably(qt_app):
    import pytest

    qt_app.document.text = "A long first line here. Right."
    character = qt_app.document.characters[0]
    first = qt_app.document.assign_character_to_range(0, 23, character.id)
    second = qt_app.document.assign_character_to_range(24, 30, character.id)
    qt_app.selection.select_clip(second.id)
    fields = qt_app.settings_dock.scope_fields.widgets
    assert fields["overlap_s"].value() == fields["overlap_s"].minimum()  # blank

    before = qt_app.build_arrangement().by_clip_id()[second.id].start_s
    fields["overlap_s"].setValue(0.3)
    fields["overlap_s"].editingFinished.emit()

    assert second.overrides["overlap_s"] == 0.3
    placed = qt_app.build_arrangement().by_clip_id()
    assert placed[second.id].start_s == pytest.approx(placed[first.id].end_s - 0.3)
    assert placed[second.id].start_s < before
    qt_app.undo()
    assert "overlap_s" not in second.overrides

    fields = qt_app.settings_dock.scope_fields.widgets
    fields["overlap_s"].setValue(0.0)
    fields["overlap_s"].editingFinished.emit()
    assert second.overrides["overlap_s"] == 0.0  # zero is an overlap of nothing, no gap
    fields["overlap_s"].setValue(fields["overlap_s"].minimum())
    fields["overlap_s"].editingFinished.emit()
    assert "overlap_s" not in second.overrides
