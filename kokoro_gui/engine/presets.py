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


def filter_fx_preset_values(preset_dict):
    """`filter_allowed_keys(preset_dict, ALLOWED_FX_PRESET_KEYS)` plus a type
    check: a `FX_STRING_KEYS` value must be a str, and every other value a
    number or a bool. A value of the wrong type is dropped, so a crafted
    preset or `fx_override` can't hand a dict to Pedalboard or a list to the
    impulse-response resolver."""
    out = {}
    for key, value in filter_allowed_keys(preset_dict, ALLOWED_FX_PRESET_KEYS).items():
        if key in FX_STRING_KEYS:
            if isinstance(value, str):
                out[key] = value
        elif isinstance(value, (bool, int, float)):
            out[key] = value
    return out


# Characters no IR name may hold: a NUL makes every os.path call raise, and
# the rest can't be in a file name on Windows, where a bundle made
# elsewhere may be opened.
_IR_UNSAFE_CHARS = frozenset('<>:"|?*\\') | frozenset(chr(c) for c in range(32)) | {"\x7f"}


def ir_safe_name(name):
    """The file stem an impulse-response name maps to: `os.path.basename` of
    it, or None for a non-string or empty name, "." or "..", or one holding
    a control character (NUL included) or one of `<>:"|?*\\`."""
    if not isinstance(name, str):
        return None
    safe = os.path.basename(name.strip())
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
        """Loads an FX preset: the open project's `fx/<name>.json` first
        (a `.tbaw` bundles the presets it names), then presets/fx."""
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
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading FX preset {name}: {e}")
                    return None
        return None
