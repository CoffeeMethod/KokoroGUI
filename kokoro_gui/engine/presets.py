"""Loading speaker presets (`presets/*.json`) and FX presets (`presets/fx/*.json`)
used by multi-speaker script parsing. Directory names are fixed constants, not
monkeypatched by any test, so no `import kokoro_engine` qualification is needed here.
"""
import json
import math
import os

# Keys a *speaker* preset (presets/*.json) is allowed to merge into a
# trusted per-segment config. Mirrors exactly what the Generation dock's
# `_save_preset_dialog` writes (kokoro_gui/qt/docks/generation_dock.py) -
# notably never out_dir/filename/time_id, which a preset file must not be
# able to steer (presets are shareable JSON with no import/export vetting -
# see Claude/SECURITY_AUDIT.md).
ALLOWED_PRESET_KEYS = frozenset({
    "voice", "speed", "volume", "pitch", "normalize",
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
    "convolution_ir", "convolution_mix",
})

# FX keys whose value is a string (a name). Every other FX key is a number or
# a bool; `filter_fx_preset_values` drops a value of the wrong kind either way.
FX_STRING_KEYS = frozenset({"convolution_ir"})

# The (min, max) of each FX key that has a slider: mirrors the `minimum` and
# `maximum` of kokoro_gui/qt/spec.py's `FX_FIELD_SPECS` (duplicated for the
# same reason as the key set above; tests/test_presets.py pins the two
# together). A numeric key with no entry here is only checked to be a finite
# number; today every numeric FX key has a slider and so an entry.
FX_VALUE_RANGES = {
    "comp_threshold": (-60, 0),
    "comp_ratio": (1, 20),
    "comp_attack": (0.1, 100),
    "comp_release": (10, 1000),
    "limiter_threshold": (-12, 0),
    "limiter_release": (10, 1000),
    "gain_db": (-20, 20),
    "eq_bass": (-20, 20),
    "eq_treble": (-20, 20),
    "highpass_freq": (20, 1000),
    "lowpass_freq": (1000, 20000),
    "reverb_room_size": (0, 1),
    "reverb_wet_level": (0, 1),
    "reverb_dry_level": (0, 1),
    "reverb_damping": (0, 1),
    "reverb_width": (0, 1),
    "delay_time": (0, 2),
    "delay_feedback": (0, 1),
    "delay_mix": (0, 1),
    "convolution_mix": (0, 1),
    "chorus_rate": (0.1, 10),
    "chorus_depth": (0, 1),
    "chorus_mix": (0, 1),
    "distortion_drive": (0, 60),
    "phaser_rate": (0.1, 10),
    "phaser_depth": (0, 1),
    "phaser_mix": (0, 1),
    "clipping_thresh": (-20, 0),
    "pitch_shift_semitones": (-12, 12),
    "bitcrush_depth": (2, 16),
}

# Global impulse-response store for the convolution reverb (grill Q31). A
# project dir's `fx/ir/` is looked at first (grill TB3).
FX_IR_DIR = os.path.join("presets", "fx", "ir")
FX_IR_SUBDIR = os.path.join("fx", "ir")


def filter_allowed_keys(preset_dict, allowed_keys):
    """Returns a copy of `preset_dict` containing only keys in
    `allowed_keys` - used to whitelist which fields a loaded preset JSON is
    allowed to merge into a trusted config dict, since preset files are
    untrusted, shareable input (see Claude/SECURITY_AUDIT.md)."""
    return {k: v for k, v in preset_dict.items() if k in allowed_keys}


def _is_number(value):
    """An int or float that is not a bool, NaN or infinite (and an int small
    enough to be a float: `math.isfinite(10**400)` raises)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def filter_fx_preset_values(preset_dict):
    """`filter_allowed_keys(preset_dict, ALLOWED_FX_PRESET_KEYS)` plus a type
    check: a `FX_STRING_KEYS` value must be a str, a key ending `_enabled` a
    bool, and every other value a finite number, clamped to its slider's
    range (`FX_VALUE_RANGES`). A value of the wrong type is dropped, so a
    crafted preset or `fx_override` can't hand a dict to Pedalboard, a list
    to the impulse-response resolver or a string to a Qt spin box, and the
    key falls back to its default. The Audio FX dock and the engine both
    read presets through this."""
    if not isinstance(preset_dict, dict):
        return {}
    out = {}
    for key, value in filter_allowed_keys(preset_dict, ALLOWED_FX_PRESET_KEYS).items():
        if key in FX_STRING_KEYS:
            if isinstance(value, str):
                out[key] = value
        elif key.endswith("_enabled"):
            if isinstance(value, bool):
                out[key] = value
        elif _is_number(value):
            low, high = FX_VALUE_RANGES.get(key, (None, None))
            if low is not None:
                value = max(low, min(high, value))
            out[key] = value
    return out


# Characters no IR name may hold: a NUL makes every os.path call raise, and
# the rest can't be in a file name on Windows, where a bundle made
# elsewhere may be opened.
_IR_UNSAFE_CHARS = frozenset('<>:"|?*\\') | frozenset(chr(c) for c in range(32)) | {"\x7f"}


def ir_safe_name(name):
    """The file stem an impulse-response name maps to: `os.path.basename` of
    it, or None for a non-string or empty name, "." or "..", or one holding
    a control character (NUL included) or one of `<>:"|?*\\`. The characters
    are checked before the basename too: on Windows `basename("a:b")` is
    "b", a drive-relative read of the name."""
    if not isinstance(name, str):
        return None
    name = name.strip()
    if any(c in _IR_UNSAFE_CHARS and c != "\\" for c in name):
        return None
    safe = os.path.basename(name)
    if not safe or safe in (".", "..") or any(c in _IR_UNSAFE_CHARS for c in safe):
        return None
    return safe


def _ir_dirs(project_dir, global_dir):
    dirs = []
    if project_dir:
        dirs.append(os.path.join(project_dir, FX_IR_SUBDIR))
    dirs.append(global_dir or FX_IR_DIR)
    return dirs


def resolve_ir(name, project_dir=None, global_dir=None):
    """The absolute path of impulse response `name`:
    `<project_dir>/fx/ir/<name>.wav` first, then `<global_dir>/<name>.wav`
    (`FX_IR_DIR` by default), else None. The name is reduced to its basename
    and the result must stay inside the directory it was looked up in."""
    safe = ir_safe_name(name)
    if safe is None:
        return None
    for directory in _ir_dirs(project_dir, global_dir):
        try:
            root = os.path.realpath(directory)
            candidate = os.path.realpath(os.path.join(root, f"{safe}.wav"))
        except (OSError, ValueError):
            continue
        if candidate.startswith(root + os.sep) and os.path.isfile(candidate):
            return candidate
    return None


def list_ir_names(project_dir=None, global_dir=None):
    """Every impulse response's name (no extension), sorted: the project's
    `fx/ir/*.wav` plus the global store's, the union. A file whose name
    `ir_safe_name` refuses is left out, since it could never resolve."""
    names = set()
    for directory in _ir_dirs(project_dir, global_dir):
        if os.path.isdir(directory):
            names.update(f[:-4] for f in os.listdir(directory)
                         if f.endswith(".wav") and len(f) > 4 and ir_safe_name(f[:-4]) == f[:-4])
    return sorted(names)


def load_fx_preset(name, project_dir=None):
    """Loads an FX preset: the open project's `fx/<name>.json` first (a
    `.tbaw` bundles the presets it names), then presets/fx. A module
    function: it never needed an engine (the GUI calls it without one)."""
    # Sanitize name to prevent path traversal
    safe_name = os.path.basename(name)
    candidates = []
    if project_dir:
        candidates.append(os.path.join(project_dir, "fx", f"{safe_name}.json"))
    candidates.append(os.path.join("presets", "fx", f"{safe_name}.json"))
    for fx_path in candidates:
        if os.path.exists(fx_path):
            try:
                with open(fx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading FX preset {name}: {e}")
                return None
            if not isinstance(data, dict):
                print(f"Error loading FX preset {name}: expected a JSON object")
                return None
            return data
    return None


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

    def load_fx_preset(self, name, project_dir=None):
        return load_fx_preset(name, project_dir)
