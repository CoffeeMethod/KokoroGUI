"""Tests for the config-dict assembly contract in start_conversion/
preview_conversion (gui.py:1528-1747)."""
import os
import re
import tempfile


def _set_text(app, text):
    app.text_entry.delete("1.0", "end")
    app.text_entry.insert("1.0", text)


BASE_KEYS = {
    "lang_code", "voice", "speed", "split_pattern", "filename", "format",
    "out_dir", "separate", "combine", "export_subtitles", "caching",
    "time_id", "num_threads", "volume", "pitch", "normalize",
    "trim_silence", "lexicon",
}

FX_KEYS = {
    "reverb_enabled", "reverb_room_size", "reverb_wet_level", "reverb_damping",
    "reverb_dry_level", "reverb_width", "eq_bass", "eq_treble",
    "comp_enabled", "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
    "distortion_enabled", "distortion_drive",
    "chorus_enabled", "chorus_rate", "chorus_depth", "chorus_mix",
    "phaser_enabled", "phaser_rate", "phaser_depth", "phaser_mix",
    "clipping_enabled", "clipping_thresh",
    "bitcrush_enabled", "bitcrush_depth", "gsm_enabled",
    "highpass_enabled", "highpass_freq", "lowpass_enabled", "lowpass_freq",
    "delay_enabled", "delay_time", "delay_feedback", "delay_mix",
    "pitch_shift_enabled", "pitch_shift_semitones",
    "limiter_enabled", "limiter_threshold", "limiter_release",
    "gain_enabled", "gain_db",
}


def test_start_conversion_assembles_full_key_set(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.apply_fx_var.set(True)
    tts_app.start_conversion()

    assert tts_app.engine.start_conversion.called
    text_arg, config = tts_app.engine.start_conversion.call_args[0]
    assert text_arg == "Hello world."
    assert BASE_KEYS <= config.keys()
    assert FX_KEYS <= config.keys()
    assert re.fullmatch(r"\d{14}", config["time_id"])


def test_start_conversion_apply_fx_false_omits_fx_keys(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.apply_fx_var.set(False)
    tts_app.start_conversion()

    _, config = tts_app.engine.start_conversion.call_args[0]
    assert "reverb_enabled" not in config
    assert "gain_db" not in config


def test_start_conversion_jit_enabled_routes_to_start_jit_conversion(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.jit_enabled.set(True)
    tts_app.start_conversion()

    assert tts_app.engine.start_jit_conversion.called
    assert not tts_app.engine.start_conversion.called


def test_start_conversion_blocks_when_pipeline_not_ready(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.engine.pipeline = None
    tts_app.start_conversion()

    assert not tts_app.engine.start_conversion.called
    assert not tts_app.engine.start_jit_conversion.called


def test_start_conversion_empty_text_shows_warning(tts_app):
    _set_text(tts_app, "")
    tts_app.start_conversion()

    assert not tts_app.engine.start_conversion.called
    assert not tts_app.engine.start_jit_conversion.called


def test_preview_conversion_assembles_smaller_extra_config(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.apply_fx_var.set(False)
    tts_app.preview_conversion()

    assert tts_app.engine.generate_preview.called
    args, kwargs = tts_app.engine.generate_preview.call_args
    preview_text, voice, speed, out_path, extra_config = args[:5]
    assert set(extra_config.keys()) == {"volume", "pitch", "normalize", "trim_silence", "lexicon"}
    assert voice == tts_app.voice_var.get()
    assert speed == tts_app.speed_var.get()


def test_preview_conversion_apply_fx_true_adds_fx_keys(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.apply_fx_var.set(True)
    tts_app.preview_conversion()

    args, kwargs = tts_app.engine.generate_preview.call_args
    extra_config = args[4]
    assert FX_KEYS <= extra_config.keys()
    assert "voice" not in extra_config
    assert "out_dir" not in extra_config


def test_preview_conversion_uses_tempdir_wav_path(tts_app):
    _set_text(tts_app, "Hello world.")
    tts_app.preview_conversion()

    args, kwargs = tts_app.engine.generate_preview.call_args
    out_path = args[3]
    assert out_path == os.path.join(tempfile.gettempdir(), "kokoro_preview.wav")
