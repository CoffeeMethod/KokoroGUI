"""Tests for the engine-level load_preset/load_fx_preset (kokoro_engine.py:491-515).

These read from hardcoded relative paths ("presets/...", "presets/fx/...")
rather than a module constant, so isolation here uses monkeypatch.chdir
instead of the isolated_dirs fixture.
"""
import json

import pytest


def test_load_preset_reads_json(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "MyPreset.json").write_text(json.dumps({"voice": "af_heart"}), encoding="utf-8")

    assert engine.load_preset("MyPreset") == {"voice": "af_heart"}


def test_load_preset_missing_returns_none(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets").mkdir()

    assert engine.load_preset("DoesNotExist") is None


def test_load_preset_malformed_json_returns_none(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "Bad.json").write_text("{not valid json", encoding="utf-8")

    assert engine.load_preset("Bad") is None


def test_load_preset_path_traversal_sanitized(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "secret.json").write_text(json.dumps({"voice": "x"}), encoding="utf-8")

    # os.path.basename() strips any path components before the lookup.
    assert engine.load_preset("../../secret") == {"voice": "x"}


def test_load_fx_preset_reads_json(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fx_dir = tmp_path / "presets" / "fx"
    fx_dir.mkdir(parents=True)
    (fx_dir / "MyFx.json").write_text(json.dumps({"reverb_enabled": True}), encoding="utf-8")

    assert engine.load_fx_preset("MyFx") == {"reverb_enabled": True}


def test_load_fx_preset_missing_returns_none(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets" / "fx").mkdir(parents=True)

    assert engine.load_fx_preset("Nope") is None


def test_load_fx_preset_path_traversal_sanitized(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fx_dir = tmp_path / "presets" / "fx"
    fx_dir.mkdir(parents=True)
    (fx_dir / "s.json").write_text(json.dumps({"gain_db": 3.0}), encoding="utf-8")

    assert engine.load_fx_preset("../../s") == {"gain_db": 3.0}


# -- FX value types and the impulse-response store (grill Q31) -------------

def test_fx_filter_accepts_a_string_ir_name_and_numbers():
    from kokoro_gui.engine.presets import filter_fx_preset_values

    data = {"convolution_ir": "Hall", "convolution_mix": 0.4, "reverb_enabled": True, "gain_db": 3,
            "out_dir": "/etc"}
    assert filter_fx_preset_values(data) == {"convolution_ir": "Hall", "convolution_mix": 0.4,
                                             "reverb_enabled": True, "gain_db": 3}


def test_fx_filter_drops_wrong_types():
    from kokoro_gui.engine.presets import filter_fx_preset_values

    data = {"convolution_ir": ["../../etc/passwd"], "convolution_mix": "0.5",
            "reverb_room_size": {"x": 1}, "gain_db": None}
    assert filter_fx_preset_values(data) == {}
    assert filter_fx_preset_values({"convolution_ir": 5}) == {}
    # A string is only accepted for the one string key.
    assert filter_fx_preset_values({"reverb_room_size": "Hall"}) == {}


def _wav(path, value=1.0):
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.array([value], dtype=np.float32), 24000, subtype="FLOAT")
    return str(path)


def test_resolve_ir_prefers_the_project_copy_then_the_global_store(tmp_path, monkeypatch):
    import os

    from kokoro_gui.engine.presets import list_ir_names, resolve_ir

    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / "project"
    global_hall = _wav(tmp_path / "presets" / "fx" / "ir" / "Hall.wav")
    _wav(tmp_path / "presets" / "fx" / "ir" / "Room.wav")

    assert resolve_ir("Hall", str(project_dir)) == os.path.realpath(global_hall)
    local_hall = _wav(project_dir / "fx" / "ir" / "Hall.wav")
    _wav(project_dir / "fx" / "ir" / "Cave.wav")
    assert resolve_ir("Hall", str(project_dir)) == os.path.realpath(local_hall)
    assert resolve_ir("Hall", None) == os.path.realpath(global_hall)
    assert resolve_ir("Nope", str(project_dir)) is None
    assert resolve_ir("", str(project_dir)) is None
    assert resolve_ir(None, str(project_dir)) is None
    assert list_ir_names(str(project_dir)) == ["Cave", "Hall", "Room"]
    assert list_ir_names(None) == ["Hall", "Room"]


def test_resolve_ir_sanitises_the_name_with_basename(tmp_path, monkeypatch):
    import os

    from kokoro_gui.engine.presets import resolve_ir

    monkeypatch.chdir(tmp_path)
    _wav(tmp_path / "presets" / "fx" / "secret.wav")
    hall = _wav(tmp_path / "presets" / "fx" / "ir" / "Hall.wav")

    assert resolve_ir("../secret") is None
    assert resolve_ir("../../elsewhere/Hall") == os.path.realpath(hall)


@pytest.mark.parametrize("name", ["hall\x00x", "Ha\nll", "a\tb", "a?b", "a*b", 'a"b', "a|b", "a<b", "a:b", "."])
def test_an_ir_name_with_an_unsafe_character_is_rejected(tmp_path, monkeypatch, name):
    from kokoro_gui.engine.presets import ir_safe_name, list_ir_names, resolve_ir

    monkeypatch.chdir(tmp_path)
    _wav(tmp_path / "presets" / "fx" / "ir" / "Hall.wav")
    assert ir_safe_name(name) is None
    assert resolve_ir(name, str(tmp_path / "project")) is None
    assert ir_safe_name(" Hall ") == "Hall"
    assert list_ir_names(None) == ["Hall"]


def test_list_ir_names_leaves_out_a_file_whose_name_is_unsafe(tmp_path, monkeypatch):
    from kokoro_gui.engine.presets import list_ir_names

    monkeypatch.chdir(tmp_path)
    _wav(tmp_path / "presets" / "fx" / "ir" / "Hall.wav")
    # DEL, not a tab: a Windows file name can hold it.
    _wav(tmp_path / "presets" / "fx" / "ir" / "a\x7fb.wav")
    assert list_ir_names(None) == ["Hall"]


# -- FX preset values are coerced and clamped (plan 03) -------------------------

def test_fx_filter_drops_a_loud_gain_and_other_wrong_kinds():
    from kokoro_gui.engine.presets import filter_fx_preset_values

    data = {"gain_db": "loud", "reverb_enabled": 1, "comp_ratio": True, "delay_time": float("nan"),
            "eq_bass": float("inf"), "bitcrush_depth": 10**400, "gsm_enabled": "true", "eq_treble": 2.5}
    assert filter_fx_preset_values(data) == {"eq_treble": 2.5}


def test_fx_filter_clamps_to_the_slider_range():
    from kokoro_gui.engine.presets import filter_fx_preset_values

    data = {"gain_db": 99, "comp_threshold": -500, "reverb_room_size": 1.5, "chorus_rate": 0.0,
            "reverb_dry_level": 1e9}
    assert filter_fx_preset_values(data) == {"gain_db": 20, "comp_threshold": -60, "reverb_room_size": 1,
                                             "chorus_rate": 0.1, "reverb_dry_level": 1e9}


def test_fx_filter_gives_an_empty_dict_for_a_non_dict():
    from kokoro_gui.engine.presets import filter_fx_preset_values

    assert filter_fx_preset_values([1, 2]) == {}
    assert filter_fx_preset_values(None) == {}


def test_fx_value_ranges_mirror_the_sliders():
    from kokoro_gui.engine.presets import FX_VALUE_RANGES
    from kokoro_gui.qt import spec

    sliders = {s.key: (s.minimum, s.maximum) for s in spec.FX_FIELD_SPECS if hasattr(s, "minimum")}
    assert FX_VALUE_RANGES == sliders


def test_every_default_fx_setting_is_inside_its_range():
    from kokoro_gui.engine.presets import filter_fx_preset_values
    from kokoro_gui.qt import spec

    defaults = {k: spec.SETTINGS_DEFAULTS[k] for k in spec.FX_PRESET_KEYS}
    assert filter_fx_preset_values(defaults) == defaults


def test_load_fx_preset_with_a_loud_gain_loads_without_error(engine, tmp_path, monkeypatch):
    from kokoro_gui.engine.presets import filter_fx_preset_values

    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets" / "fx").mkdir(parents=True)
    (tmp_path / "presets" / "fx" / "Loud.json").write_text(
        json.dumps({"gain_db": "loud", "gain_enabled": True}), encoding="utf-8")

    loaded = engine.load_fx_preset("Loud")
    assert filter_fx_preset_values(loaded) == {"gain_enabled": True}


def test_load_fx_preset_of_a_json_list_is_none(engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "presets" / "fx").mkdir(parents=True)
    (tmp_path / "presets" / "fx" / "List.json").write_text("[1, 2]", encoding="utf-8")
    assert engine.load_fx_preset("List") is None
