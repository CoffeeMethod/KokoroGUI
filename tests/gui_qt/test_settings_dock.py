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
    state = qt_app.generation_dock.get_state()
    settings_owned = {"engine_id", "time_id", "lexicon"}
    assert set(state.keys()) | settings_owned == set(spec.GENERATION_BASE_KEYS)


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
