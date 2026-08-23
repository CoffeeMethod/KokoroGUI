"""Loading speaker presets (`presets/*.json`) and FX presets (`presets/fx/*.json`)
used by multi-speaker script parsing. Directory names are fixed constants, not
monkeypatched by any test, so no `import kokoro_engine` qualification is needed here.
"""
import json
import os

# Keys a *speaker* preset (presets/*.json) is allowed to merge into a
# trusted per-segment config. Mirrors exactly what the Generation dock's
# `_save_preset_dialog` writes (kokoro_gui/qt/docks/generation_dock.py) -
# notably never out_dir/filename/time_id, which a preset file must not be
# able to steer (presets are shareable JSON with no import/export vetting -
# see Claude/SECURITY_AUDIT.md).
ALLOWED_PRESET_KEYS = frozenset({
    "voice", "speed", "volume", "pitch", "split_pattern", "normalize",
    "trim", "format", "apply_fx", "fx_preset",
})

# Keys an *FX* preset (presets/fx/*.json) is allowed to merge. Mirrors
# kokoro_gui/qt/spec.py's FX_PRESET_KEYS (duplicated here rather than
# imported, since kokoro_gui/engine is meant to stay independent of the Qt
# frontend - see CLAUDE.md).
ALLOWED_FX_PRESET_KEYS = frozenset({
    "reverb_enabled", "reverb_room_size", "reverb_wet_level", "reverb_damping",
    "reverb_dry_level", "reverb_width",
    "eq_bass", "eq_treble",
    "comp_enabled", "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
    "distortion_enabled", "distortion_drive",
    "chorus_enabled", "chorus_rate", "chorus_depth", "chorus_mix",
    "phaser_enabled", "phaser_rate", "phaser_depth", "phaser_mix",
    "clipping_enabled", "clipping_thresh",
    "bitcrush_enabled", "bitcrush_depth",
    "gsm_enabled",
    "highpass_enabled", "highpass_freq",
    "lowpass_enabled", "lowpass_freq",
    "delay_enabled", "delay_time", "delay_feedback", "delay_mix",
    "pitch_shift_enabled", "pitch_shift_semitones",
    "limiter_enabled", "limiter_threshold", "limiter_release",
    "gain_enabled", "gain_db",
})


def filter_allowed_keys(preset_dict, allowed_keys):
    """Returns a copy of `preset_dict` containing only keys in
    `allowed_keys` - used to whitelist which fields a loaded preset JSON is
    allowed to merge into a trusted config dict, since preset files are
    untrusted, shareable input (see Claude/SECURITY_AUDIT.md)."""
    return {k: v for k, v in preset_dict.items() if k in allowed_keys}


class PresetsMixin:
    def load_preset(self, name):
        """Loads a preset from the presets directory."""
        # Sanitize name to prevent path traversal
        safe_name = os.path.basename(name)
        preset_path = os.path.join("presets", f"{safe_name}.json")
        if os.path.exists(preset_path):
            try:
                with open(preset_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading preset {name}: {e}")
        return None

    def load_fx_preset(self, name):
        """Loads an FX preset from the presets/fx directory."""
        # Sanitize name to prevent path traversal
        safe_name = os.path.basename(name)
        fx_path = os.path.join("presets", "fx", f"{safe_name}.json")
        if os.path.exists(fx_path):
            try:
                with open(fx_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading FX preset {name}: {e}")
        return None
