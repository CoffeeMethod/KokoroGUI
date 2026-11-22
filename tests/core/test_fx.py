"""kokoro_gui/core/fx.py: the FX layers a clip or character resolves to,
with plain values in place of the Audio FX and Settings tabs."""
import json

import pytest

from kokoro_gui.core import fx as core_fx
from kokoro_gui.daw.models import Character, Clip, Document
from kokoro_gui.engine import presets


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    presets_dir = tmp_path / "presets" / "fx"
    project_dir = tmp_path / "project"
    presets_dir.mkdir(parents=True)
    (project_dir / "fx").mkdir(parents=True)
    return presets_dir, project_dir


def _write(folder, name, values):
    (folder / f"{name}.json").write_text(json.dumps(values), encoding="utf-8")


def _resolve(document, dirs, clip=None, character=None, project_fx=None, master_apply=True, project_preset=None):
    presets_dir, project_dir = dirs
    return core_fx.resolve_fx(document, clip, character, project_fx=project_fx or {}, master_apply=master_apply,
                              project_preset=project_preset, project_dir=str(project_dir),
                              presets_dir=str(presets_dir))


def _scene(character_preset=None, clip_overrides=None, fx_override=None, character_apply=None):
    preset_data = {}
    if character_preset:
        preset_data["fx_preset"] = character_preset
    if character_apply is not None:
        preset_data["apply_fx"] = character_apply
    character = Character(name="Sam", preset_data=preset_data)
    clip = Clip(character_id=character.id, overrides=dict(clip_overrides or {}), fx_override=fx_override)
    return Document(runs=[], clips=[clip], characters=[character]), character, clip


def test_layers_apply_lowest_first(dirs):
    presets_dir, _project_dir = dirs
    _write(presets_dir, "Room", {"eq_bass": 2.0, "eq_treble": 1.0, "reverb_enabled": True})
    _write(presets_dir, "Phone", {"eq_treble": 5.0})
    document, _character, clip = _scene("Room", {"fx_preset": "Phone"}, fx_override={"eq_bass": 9.0})
    resolution = _resolve(document, dirs, clip=clip, project_fx={"eq_bass": 0.0, "reverb_wet_level": 0.4})
    assert resolution.values == {"eq_bass": 9.0, "eq_treble": 5.0, "reverb_enabled": True, "reverb_wet_level": 0.4}
    assert resolution.preset_name == "Phone"
    assert resolution.apply_fx is True


def test_an_fx_override_without_a_clip_preset_is_custom(dirs):
    document, _character, clip = _scene(fx_override={"eq_bass": 3.0})
    assert _resolve(document, dirs, clip=clip).preset_name == "custom"


def test_project_scope_names_the_project_preset_and_others_do_not(dirs):
    document, character, clip = _scene()
    assert _resolve(document, dirs, project_preset="Hall").preset_name == "Hall"
    assert _resolve(document, dirs, project_preset=core_fx.PLACEHOLDER).preset_name is None
    assert _resolve(document, dirs, character=character, project_preset="Hall").preset_name is None
    assert _resolve(document, dirs, clip=clip, project_preset="Hall").preset_name is None


@pytest.mark.parametrize("master, character_apply, clip_apply, expected", [
    (True, None, None, True),
    (False, None, None, False),
    (True, False, None, False),
    (True, False, True, True),
    (False, True, True, False),
    (True, True, False, False),
])
def test_apply_fx_is_the_master_switch_and_the_scope(dirs, master, character_apply, clip_apply, expected):
    overrides = {} if clip_apply is None else {"apply_fx": clip_apply}
    document, _character, clip = _scene(clip_overrides=overrides, character_apply=character_apply)
    assert _resolve(document, dirs, clip=clip, master_apply=master).apply_fx is expected


def test_fx_override_counts_as_on_unless_the_clip_says_off(dirs):
    document, _character, clip = _scene(fx_override={"eq_bass": 1.0}, character_apply=False)
    assert _resolve(document, dirs, clip=clip).apply_fx is True
    clip.overrides["apply_fx"] = False
    assert _resolve(document, dirs, clip=clip).apply_fx is False


def test_project_local_preset_wins_over_the_global_one(dirs):
    presets_dir, project_dir = dirs
    _write(presets_dir, "Radio", {"eq_bass": 1.0})
    _write(project_dir / "fx", "Radio", {"eq_bass": 7.0})
    assert core_fx.load_fx_preset_values("Radio", str(project_dir), str(presets_dir)) == {"eq_bass": 7.0}
    assert core_fx.load_fx_preset_values("Radio", None, str(presets_dir)) == {"eq_bass": 1.0}


def test_preset_files_are_read_directly_when_the_loader_finds_none(dirs, monkeypatch, tmp_path):
    _presets_dir, project_dir = dirs
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(presets, "load_fx_preset", lambda *a, **k: None)
    _write(elsewhere, "Warm", {"eq_bass": 2.0, "not_an_fx_key": 1})
    assert core_fx.load_fx_preset_values("Warm", str(project_dir), str(elsewhere)) == {"eq_bass": 2.0}
    _write(project_dir / "fx", "Warm", {"eq_bass": 4.0})
    assert core_fx.load_fx_preset_values("Warm", str(project_dir), str(elsewhere)) == {"eq_bass": 4.0}


def test_placeholder_and_empty_names_load_nothing(dirs):
    presets_dir, project_dir = dirs
    assert core_fx.real_preset_name(core_fx.PLACEHOLDER) is None
    assert core_fx.real_preset_name("") is None
    assert core_fx.load_fx_preset_values(core_fx.PLACEHOLDER, str(project_dir), str(presets_dir)) is None
