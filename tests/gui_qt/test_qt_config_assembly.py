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


# --- generation config and the segment key closure (Claude/old/PLAN_tbaw_bundle.md 2.3) --

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


def test_generation_config_carries_the_lexicon(qt_app):
    qt_app.settings["lexicon"] = {"Nguyen": "Win"}
    clip = _clip_for(qt_app)
    assert qt_app._assemble_generation_config(clip)["lexicon"] == {"Nguyen": "Win"}


def test_lexicon_rule_dirties_only_the_clip_it_rewrites(qt_app, tmp_path):
    from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash

    text = "Mr Nguyen arrived. Nobody else."
    qt_app.document.text = text
    character = qt_app.document.characters[0]
    first = qt_app.document.assign_character_to_range(0, 18, character.id)
    second = qt_app.document.assign_character_to_range(19, len(text), character.id)
    for i, clip in enumerate((first, second)):
        path = tmp_path / f"seg{i}.wav"
        path.write_bytes(b"RIFF")
        clip_text = qt_app.document.clip_text(clip)
        key = compute_expected_cache_hash(clip_text, qt_app._assemble_generation_config(clip),
                                          key_fn=qt_app.document.segment_key_fn, clip=clip)
        clip.segments = build_segments_from_results(key, [{
            "text": clip_text, "path": str(path), "duration": 1.0, "cache_key": key,
        }])
    assert qt_app.document.dirty_clips() == []

    qt_app.lexicon_dock.orig_edit.setText("Nguyen")
    qt_app.lexicon_dock.replace_edit.setText("Win")
    qt_app.lexicon_dock.add_rule()

    assert qt_app.document.dirty_clips() == [first]


def test_both_paths_carry_the_segmentation_settings(qt_app):
    from kokoro_gui.qt import spec

    qt_app.settings["segment_target_words"] = 25
    qt_app.settings["segment_at_pauses"] = False
    qt_app.settings_dock._build_schema_form()
    clip = _clip_for(qt_app)
    for config in (qt_app._assemble_generation_config(clip), qt_app._assemble_config()):
        assert {k: config[k] for k in spec.SEGMENTATION_KEYS} == {
            "segment_target_words": 25, "segment_at_paragraphs": True,
            "segment_at_sentences": True, "segment_at_pauses": False,
        }
    assert "split_pattern" not in qt_app._assemble_generation_config(clip)


def test_changing_the_word_target_restales_clips_whose_pieces_move(qt_app, tmp_path):
    from kokoro_gui.daw.dirty import build_segments_from_results, compute_expected_cache_hash, predict_segment_texts

    text = "w1 w2 w3 w4. w5 w6. w7 w8 w9 w10."
    qt_app.settings["segment_target_words"] = 5
    qt_app.settings_dock._build_schema_form()
    clip = _clip_for(qt_app, text=text)
    config = qt_app._assemble_generation_config(clip)
    key = compute_expected_cache_hash(text, config, key_fn=qt_app.document.segment_key_fn, clip=clip)
    results = []
    for i, piece in enumerate(predict_segment_texts(text, config)):
        path = tmp_path / f"seg{i}.wav"
        path.write_bytes(b"RIFF")
        results.append({"text": piece, "path": str(path), "duration": 1.0, "cache_key": key})
    clip.segments = build_segments_from_results(key, results)
    assert qt_app.document.dirty_clips() == []

    qt_app.settings_dock.schema_form.widget_for("segment_target_words").setValue(6)

    assert qt_app.document.dirty_clips() == [clip]



def test_variant_override_swaps_the_voice_on_a_cloning_backend(qt_app, monkeypatch):
    import dataclasses

    qt_app.document.text = "hello"
    character = qt_app.document.characters[0]
    character.preset_data["voice"] = "calm_ref"
    character.variants = {"angry": "angry_ref"}
    clip = qt_app.document.assign_character_to_range(0, 5, character.id)
    clip.overrides["variant"] = "angry"

    # Kokoro has no variants: the override is ignored.
    assert qt_app._assemble_generation_config(clip)["voice"] == "calm_ref"

    cloning = dataclasses.replace(qt_app.backend.capabilities, supports_voice_cloning=True)
    monkeypatch.setattr(qt_app.backend, "capabilities", cloning)
    assert qt_app._assemble_generation_config(clip)["voice"] == "angry_ref"
    assert qt_app._assemble_clip_config(clip)["voice"] == "angry_ref"

    clip.overrides["variant"] = "missing"
    assert qt_app._assemble_generation_config(clip)["voice"] == "calm_ref"
