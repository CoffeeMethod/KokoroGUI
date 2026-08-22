"""Config-dict contract parity between the Qt and Tk frontends.

`tests/test_gui_config_assembly.py` is the existing source of truth for what
gui.py's `start_conversion` actually assembles. This file asserts Qt's
`_assemble_config()` produces the identical key set - both against the
mirrored constants in kokoro_gui/qt/spec.py *and* against a live capture of
Tk's own assembled dict via the existing `tts_app` fixture - so a future edit
to either frontend's config dict that isn't mirrored in the other is a test
failure, not a silent drift.
"""
from kokoro_gui.qt import spec


def test_assembled_config_matches_mirrored_spec_keys(qt_app):
    config = qt_app._assemble_config()
    expected = set(spec.GENERATION_BASE_KEYS) | set(spec.FX_PRESET_KEYS)
    assert set(config.keys()) == expected


def test_assembled_config_omits_fx_keys_when_apply_fx_off(qt_app):
    qt_app.generation_dock.apply_fx_check.setChecked(False)
    config = qt_app._assemble_config()
    assert set(config.keys()) == set(spec.GENERATION_BASE_KEYS)
    assert "reverb_enabled" not in config
    assert "gain_db" not in config


def test_assembled_config_time_id_is_timecode(qt_app):
    import re
    config = qt_app._assemble_config()
    assert re.match(r"^\d{14}$", config["time_id"])


def test_assembled_config_matches_tk_live_capture(tts_app, qt_app):
    # tts_app must be resolved before qt_app: both share one tmp_path (pytest's
    # tmp_path fixture is function-scoped) and each fixture creates
    # "custom_voices" in it - tts_app's own mkdir() (unmodifiable, see
    # tests/conftest.py) has no exist_ok, so it must run first; qt_app's does
    # tolerate the directory already existing (see tests/gui_qt/conftest.py).
    """Cross-frontend parity: whatever key set gui.py's start_conversion
    actually builds (captured live through the Tk `tts_app` fixture +
    StubEngine, exactly like tests/test_gui_config_assembly.py does) must
    equal the key set Qt assembles - not just what spec.py claims."""
    tts_app.text_entry.insert("1.0", "Hello world")
    tts_app.start_conversion()

    assert tts_app.engine.start_conversion.called
    tk_text, tk_config = tts_app.engine.start_conversion.call_args[0]

    qt_config = qt_app._assemble_config()

    assert set(tk_config.keys()) == set(qt_config.keys())


def test_generation_dock_state_covers_base_keys_minus_settings_owned(qt_app):
    """Everything _assemble_config adds on top of the Generation dock's own
    get_state() (engine_id/time_id/lexicon) is intentionally settings-owned,
    not dock-owned - see app.py's _assemble_config."""
    state = qt_app.generation_dock.get_state()
    settings_owned = {"engine_id", "time_id", "lexicon"}
    assert set(state.keys()) | settings_owned == set(spec.GENERATION_BASE_KEYS)


def test_fx_dock_state_covers_all_fx_preset_keys(qt_app):
    state = qt_app.fx_dock.get_state()
    assert set(state.keys()) == set(spec.FX_PRESET_KEYS)
