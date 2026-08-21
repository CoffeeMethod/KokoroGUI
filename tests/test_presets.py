"""Tests for the engine-level load_preset/load_fx_preset (kokoro_engine.py:491-515).

These read from hardcoded relative paths ("presets/...", "presets/fx/...")
rather than a module constant, so isolation here uses monkeypatch.chdir
instead of the isolated_dirs fixture.
"""
import json


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
