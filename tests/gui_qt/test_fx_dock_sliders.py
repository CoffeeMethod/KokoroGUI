"""Every FX preset key has a control in the Audio FX tab. The seven keys
that used to be saved by presets and read by the engine but had no widget
(reverb_dry_level, chorus_mix, phaser_depth, phaser_mix, comp_attack,
comp_release, limiter_release) now each have a slider."""
import json
import os

import pytest

from kokoro_gui.qt import spec

NEW_KEYS = ["reverb_dry_level", "chorus_mix", "phaser_depth", "phaser_mix",
            "comp_attack", "comp_release", "limiter_release"]


def _slider_spec(key):
    return next(s for s in spec.FX_FIELD_SPECS if s.key == key)


def test_every_fx_preset_key_has_a_control():
    """A key added to FX_PRESET_KEYS with no slider, enable checkbox, toggle
    or file combo fails here, instead of living as a value nobody can edit."""
    sliders = {s.key for s in spec.FX_FIELD_SPECS}
    enables = {s.enabled_key for s in spec.FX_FIELD_SPECS if s.enabled_key}
    toggles = {t[0] for t in spec.FX_STANDALONE_TOGGLES}
    assert set(spec.FX_PRESET_KEYS) - sliders - enables - toggles == set()


def test_the_dock_builds_a_widget_for_every_preset_key(qt_app):
    fx = qt_app.fx_dock
    widgets = set(fx._value_widgets) | set(fx._enabled_checks) | set(fx._file_combos)
    assert set(spec.FX_PRESET_KEYS) - widgets == set()
    assert set(fx.get_state()) == set(spec.FX_PRESET_KEYS)


@pytest.mark.parametrize("key", NEW_KEYS)
def test_a_new_slider_shows_the_default_and_moves_the_state(qt_app, key):
    fx = qt_app.fx_dock
    s = _slider_spec(key)
    spin = fx._value_widgets[key]
    assert (spin.minimum(), spin.maximum()) == (s.minimum, s.maximum)
    assert spin.value() == spec.SETTINGS_DEFAULTS[key]
    target = s.minimum if spin.value() != s.minimum else s.maximum
    spin.setValue(target)
    assert fx.get_state()[key] == pytest.approx(target)
    assert fx.project_fx_state()[key] == pytest.approx(target)
    assert qt_app._assemble_config()[key] == pytest.approx(target)


@pytest.mark.parametrize("key", NEW_KEYS)
def test_a_preset_value_loads_into_the_new_slider(qt_app, key):
    import kokoro_gui.qt.app as qt_app_module

    fx = qt_app.fx_dock
    s = _slider_spec(key)
    value = s.minimum if spec.SETTINGS_DEFAULTS[key] != s.minimum else s.maximum
    os.makedirs(qt_app_module.FX_PRESETS_DIR, exist_ok=True)
    with open(os.path.join(qt_app_module.FX_PRESETS_DIR, "Sliders.json"), "w", encoding="utf-8") as f:
        json.dump({key: value}, f)
    fx.refresh_presets()
    fx.load_preset("Sliders")
    assert fx._value_widgets[key].value() == pytest.approx(value)
    assert fx.get_state()[key] == pytest.approx(value)


def test_a_new_slider_is_gated_under_its_effect_checkbox(qt_app):
    fx = qt_app.fx_dock
    for key in NEW_KEYS:
        assert _slider_spec(key).enabled_key in fx._enabled_checks


def test_an_out_of_range_preset_value_is_clamped_into_the_new_slider(qt_app):
    fx = qt_app.fx_dock
    fx.set_values({"reverb_dry_level": 5.0, "comp_attack": 0.0, "limiter_release": 1e9})
    state = fx.get_state()
    assert state["reverb_dry_level"] == 1.0
    assert state["comp_attack"] == pytest.approx(0.1)
    assert state["limiter_release"] == 1000
