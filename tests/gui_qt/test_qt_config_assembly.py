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
    state = qt_app.settings_dock.get_state()
    settings_owned = {"engine_id", "time_id", "lexicon"}
    # Output/format/subtitles/keep-segments live in the Export dialog now
    # (kokoro_gui/qt/docks/export_dialog.py), not the Settings tab.
    export_owned = {"filename", "out_dir", "separate", "combine", "export_subtitles"}
    assert set(state.keys()) | settings_owned | export_owned == set(spec.GENERATION_BASE_KEYS)


def test_fx_dock_state_covers_all_fx_preset_keys(qt_app):
    state = qt_app.fx_dock.get_state()
    assert set(state.keys()) == set(spec.FX_PRESET_KEYS)


# --- generation config and the segment key closure (Claude/PLAN_tbaw_bundle.md 2.3) --

def _clip_for(qt_app, text="hello world", preset=None):
    from kokoro_gui.daw.models import Character

    qt_app.document.text = text
    character = Character.from_preset_dict("NoLang", preset or {"voice": "af_sarah"})
    qt_app.document.characters.append(character)
    return qt_app.document.assign_character_to_range(0, len(text), character.id)


def test_dirty_check_and_generate_hash_the_same_config_without_lang_code_in_the_preset(qt_app):
    """The character's preset carries no `lang_code`, so the app default
    fills it in for both paths: `_assemble_generation_config` decides what
    the key hashes, `_assemble_clip_config` is built on it."""
    from kokoro_gui.engine.caching import segment_key

    clip = _clip_for(qt_app)
    text = qt_app.document.clip_text(clip)
    generation = qt_app._assemble_generation_config(clip)
    full = qt_app._assemble_clip_config(clip)

    assert "lang_code" in generation and generation["voice"] == "af_sarah"
    assert {k: full[k] for k in generation} == generation
    assert qt_app.document.segment_key_fn(text, clip) == segment_key(text, full, qt_app.backend)


def test_generation_config_reads_take_off_the_clip_and_carries_project_dir(qt_app):
    clip = _clip_for(qt_app)
    assert qt_app._assemble_generation_config(clip)["take"] == 0
    clip.overrides["take"] = 3
    config = qt_app._assemble_generation_config(clip)
    assert config["take"] == 3
    assert "project_dir" in config
    # `take` never comes through the preset whitelist.
    assert "take" not in qt_app.document.effective_config_for_clip(clip)


def test_segment_key_fn_is_memoized_and_notices_a_rewritten_voice_file(qt_app, tmp_path, monkeypatch):
    import os

    import kokoro_engine
    from kokoro_gui.engine import caching

    voices = tmp_path / "custom_voices"
    voices.mkdir(exist_ok=True)
    monkeypatch.setattr(kokoro_engine, "CUSTOM_VOICES_DIR", str(voices))
    mix = voices / "Mix.pt"
    mix.write_bytes(b"v1")
    clip = _clip_for(qt_app, preset={"voice": "Mix"})
    text = qt_app.document.clip_text(clip)

    calls = []
    real = caching.segment_key
    monkeypatch.setattr(caching, "segment_key", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    first = qt_app.document.segment_key_fn(text, clip)
    again = qt_app.document.segment_key_fn(text, clip)
    assert first == again and len(calls) == 1

    mix.write_bytes(b"v2 longer")
    future = os.path.getmtime(mix) + 5
    os.utime(mix, (future, future))
    assert qt_app.document.segment_key_fn(text, clip) != first
    assert len(calls) == 2


def test_dirty_clips_on_many_clips_reads_no_files(qt_app, tmp_path, monkeypatch):
    """A rehighlight of a book: stats only, no `open()` on any segment,
    voice or transcript file."""
    import builtins
    import os

    from kokoro_gui.daw.dirty import build_segments_from_results

    text = " ".join(f"line{i}." for i in range(200))
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    clips = []
    pos = 0
    for i in range(200):
        end = text.index(" ", pos) if i < 199 else len(text)
        clips.append(qt_app.document.assign_character_to_range(pos, end, character.id))
        pos = end + 1
    for i, clip in enumerate(clips):
        path = tmp_path / f"seg{i}.wav"
        path.write_bytes(b"RIFF")
        key = qt_app.document.segment_key_fn(qt_app.document.clip_text(clip), clip)
        clip.segments = build_segments_from_results(key, [{
            "text": qt_app.document.clip_text(clip), "path": str(path), "duration": 1.0, "cache_key": key,
        }])

    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (opened.append(a[0]), real_open(*a, **k))[1])
    assert qt_app.document.dirty_clips() == []
    assert opened == []
    assert os.path.isfile(str(tmp_path / "seg0.wav"))
