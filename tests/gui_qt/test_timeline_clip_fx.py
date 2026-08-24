"""Tests for per-clip FX (item 5, "Per-clip FX button"): choosing an FX
preset (or "Clear FX") from a timeline clip block's FX menu
(kokoro_gui/qt/timeline_view.py) via kokoro_gui/qt/docks/timeline_dock.py's
`on_fx_preset_requested`, and `_assemble_clip_config`'s (kokoro_gui/qt/app.py)
"clip-level FX always wins" merge - mirrors test_timeline_clip_generate.py's
conventions (qt_app fixture, StubEngine's mocked engine calls)."""
from kokoro_gui.daw.undo import SetClipFxCommand


def _make_clip(qt_app, start=0, end=5, text="hello world"):
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    return qt_app.document.assign_character_to_range(start, end, character.id)


# ---------------------------------------------------------------------------
# TimelineDock.on_fx_preset_requested
# ---------------------------------------------------------------------------

def test_choosing_a_preset_sets_clip_fx_override_via_undoable_command(qt_app):
    clip = _make_clip(qt_app)
    qt_app.engine.load_fx_preset.return_value = {
        "reverb_enabled": True, "reverb_room_size": 0.5, "not_an_fx_key": "dropped",
    }

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "Warm")

    assert clip.fx_override == {"reverb_enabled": True, "reverb_room_size": 0.5}
    assert qt_app.document.undo_stack.can_undo() is True


def test_choosing_a_preset_calls_load_fx_preset_with_the_chosen_name(qt_app):
    clip = _make_clip(qt_app)
    qt_app.engine.load_fx_preset.return_value = {"reverb_enabled": True}

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "Telephone")

    qt_app.engine.load_fx_preset.assert_called_once_with("Telephone")


def test_clear_fx_on_a_clip_with_an_override_sets_it_back_to_none(qt_app):
    clip = _make_clip(qt_app)
    clip.fx_override = {"reverb_enabled": True}

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "")

    assert clip.fx_override is None
    assert qt_app.document.undo_stack.can_undo() is True


def test_clear_fx_pushes_undoable_command_restoring_previous_override(qt_app):
    clip = _make_clip(qt_app)
    clip.fx_override = {"reverb_enabled": True}

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "")
    assert clip.fx_override is None

    qt_app.document.undo_stack.undo()

    assert clip.fx_override == {"reverb_enabled": True}


def test_choosing_a_preset_that_fails_to_load_clears_the_override(qt_app):
    clip = _make_clip(qt_app)
    qt_app.engine.load_fx_preset.return_value = None  # missing/unreadable preset file

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "GoneNow")

    assert clip.fx_override is None


def test_unknown_clip_id_is_a_noop(qt_app):
    qt_app.timeline_dock.on_fx_preset_requested("nonexistent-clip-id", "Warm")

    assert qt_app.document.undo_stack.can_undo() is False


def test_on_fx_preset_requested_calls_refresh_timeline_and_schedule_save(qt_app, monkeypatch):
    clip = _make_clip(qt_app)
    qt_app.engine.load_fx_preset.return_value = {"reverb_enabled": True}
    refresh_calls = []
    save_calls = []
    monkeypatch.setattr(qt_app, "refresh_timeline", lambda: refresh_calls.append(True))
    monkeypatch.setattr(qt_app, "schedule_save", lambda: save_calls.append(True))

    qt_app.timeline_dock.on_fx_preset_requested(clip.id, "Warm")

    assert refresh_calls
    assert save_calls


def test_fx_preset_requested_signal_is_connected_to_the_dock_handler(qt_app):
    clip = _make_clip(qt_app)
    qt_app.engine.load_fx_preset.return_value = {"reverb_enabled": True}

    qt_app.timeline_dock.timeline_view.fxPresetRequested.emit(clip.id, "Warm")

    assert clip.fx_override == {"reverb_enabled": True}


# ---------------------------------------------------------------------------
# _assemble_clip_config: clip-level fx_override wins over the character's
# resolved fx_preset for the same key(s).
# ---------------------------------------------------------------------------

def test_assemble_clip_config_includes_fx_override_values(qt_app):
    clip = _make_clip(qt_app)
    clip.fx_override = {"reverb_enabled": True, "reverb_room_size": 0.8}

    config = qt_app._assemble_clip_config(clip)

    assert config["reverb_enabled"] is True
    assert config["reverb_room_size"] == 0.8


def test_assemble_clip_config_fx_override_wins_over_character_fx_preset(qt_app, monkeypatch):
    clip = _make_clip(qt_app)
    character = qt_app.document.characters[0]
    character.preset_data["apply_fx"] = True
    character.preset_data["fx_preset"] = "CharacterPreset"

    def _fake_load_fx_preset(name):
        assert name == "CharacterPreset"
        return {"reverb_enabled": True, "reverb_room_size": 0.2}

    monkeypatch.setattr(qt_app.engine, "load_fx_preset", _fake_load_fx_preset)
    clip.fx_override = {"reverb_room_size": 0.9}  # conflicts with the character preset's value

    config = qt_app._assemble_clip_config(clip)

    # The character preset's other key still applies...
    assert config["reverb_enabled"] is True
    # ...but the clip's own override wins on the conflicting key.
    assert config["reverb_room_size"] == 0.9


def test_assemble_clip_config_with_no_fx_override_is_unaffected(qt_app):
    clip = _make_clip(qt_app)
    assert clip.fx_override is None

    config = qt_app._assemble_clip_config(clip)

    assert "reverb_enabled" not in config
