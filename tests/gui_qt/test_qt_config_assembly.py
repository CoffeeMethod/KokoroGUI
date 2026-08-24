"""Config-dict assembly contract for the Qt frontend's `_assemble_config()`,
checked against the mirrored constants in kokoro_gui/qt/spec.py."""
from kokoro_gui.qt import spec


def test_assembled_config_matches_mirrored_spec_keys(qt_app):
    config = qt_app._assemble_config()
    expected = set(spec.GENERATION_BASE_KEYS) | set(spec.FX_PRESET_KEYS)
    assert set(config.keys()) == expected


def test_assembled_config_omits_fx_keys_when_apply_fx_off(qt_app):
    qt_app.settings_dock.apply_fx_check.setChecked(False)
    config = qt_app._assemble_config()
    assert set(config.keys()) == set(spec.GENERATION_BASE_KEYS)
    assert "reverb_enabled" not in config
    assert "gain_db" not in config


def test_assembled_config_time_id_is_timecode(qt_app):
    import re
    config = qt_app._assemble_config()
    assert re.match(r"^\d{14}$", config["time_id"])


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
