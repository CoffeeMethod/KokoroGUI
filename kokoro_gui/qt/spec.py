"""Pure-data constants for the Qt frontend's field lists (generation config
keys, FX preset keys/slider specs, settings defaults). Engine data (voices,
languages) lives on each backend in `kokoro_gui/engines/`.

This module has no Qt imports so both `kokoro_gui/qt/*` and the test suite can
import it standalone.

These constants originated as a mirror of the now-retired Tk frontend's
(`gui.py`, `kokoro_gui/ui/*.py`) hard-coded field lists, kept in sync via
`tests/gui_qt/test_qt_config_assembly.py`'s cross-frontend check during the
migration (see PLAN_qt_and_engine_abstraction.md, workstream 3a). Now that Tk
has been removed, this module is simply the canonical source of truth for the
Qt frontend.

- Every FX_PRESET_KEYS field has a control: a slider in FX_FIELD_SPECS, an
  enable checkbox, a standalone toggle or a file combo. The assert at the
  bottom fails when a key is added to FX_PRESET_KEYS with none of those.
"""
from dataclasses import dataclass
from typing import Optional

# --- Generation config dict (non-FX keys) --------------------------------

GENERATION_BASE_KEYS = [
    "engine_id", "lang_code", "voice", "speed", "filename",
    "format", "out_dir", "separate", "combine", "export_subtitles", "caching",
    "time_id", "num_threads", "volume", "pitch", "normalize", "trim_silence",
    "lexicon", "segment_target_words", "segment_at_paragraphs", "segment_at_sentences",
    "segment_at_pauses",
]

# Project-wide segmentation settings (kokoro_gui/engine/segmenting.py): what
# decides the pieces a text is generated as, on every path.
SEGMENTATION_KEYS = (
    "segment_target_words", "segment_at_paragraphs", "segment_at_sentences", "segment_at_pauses",
)

# Schema fields that belong to the program, not to a voice: Options >
# Settings... shows them and the Settings tab leaves them out. That's the
# segmentation keys, the default output format, and every field in a
# backend's "Advanced" group (threads, the segment cache, Audio8's
# reference-encoding cache) except the lexicon, which has its own tab.
PROGRAM_SCHEMA_KEYS = (*SEGMENTATION_KEYS, "format")
PROGRAM_SCHEMA_GROUP = "Advanced"


def is_program_field(field) -> bool:
    """True for a `ConfigField` the Settings window owns (see above)."""
    if field.key == "lexicon":
        return False
    return field.key in PROGRAM_SCHEMA_KEYS or field.group == PROGRAM_SCHEMA_GROUP


# Options > Transcript details: the overlay toggles under "Show details"
# (`transcript_details`), as (settings key, menu label).
DETAIL_LAYERS = (
    ("details_segments", "Segment boundaries"),
    ("details_clip_info", "Clip info"),
    ("details_lexicon", "Lexicon rewrites"),
    ("details_gaps", "Gaps"),
)

# --- FX preset / config-merge keys ----------------------------------------

FX_PRESET_KEYS = [
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
]


@dataclass(frozen=True)
class FXSliderSpec:
    """One numeric FX field the UI exposes a control for. `steps` mirrors the
    Tk `CTkSlider(number_of_steps=...)` value so `(maximum - minimum) / steps`
    reproduces the same granularity in a QDoubleSpinBox's singleStep."""
    key: str
    label: str
    minimum: float
    maximum: float
    steps: int
    group: str          # dock section heading, e.g. "Spatial & Time"
    section: str         # sub-heading, e.g. "Reverb"
    enabled_key: Optional[str] = None   # bool field this is gated under, if any
    unit: str = ""
    decimals: int = 2


@dataclass(frozen=True)
class FXFileSpec:
    """One FX field that names a file in an asset store instead of holding a
    number: the dock shows a combo of the names in `store` (project-local
    first, then global) with a "None" entry that stores "". `store` is
    "ir" for impulse responses (`presets/fx/ir/*.wav`, grill Q31), the only
    store so far."""
    key: str
    label: str
    group: str
    section: str
    store: str = "ir"
    enabled_key: Optional[str] = None


FX_FIELD_SPECS = [
    # --- Dynamics ---
    FXSliderSpec("comp_threshold", "Threshold", -60, 0, 60, "Dynamics", "Compressor", "comp_enabled", "dB", 1),
    FXSliderSpec("comp_ratio", "Ratio", 1, 20, 19, "Dynamics", "Compressor", "comp_enabled", ":1", 1),
    FXSliderSpec("comp_attack", "Attack", 0.1, 100, 999, "Dynamics", "Compressor", "comp_enabled", "ms", 1),
    FXSliderSpec("comp_release", "Release", 10, 1000, 99, "Dynamics", "Compressor", "comp_enabled", "ms", 0),
    FXSliderSpec("limiter_threshold", "Threshold", -12, 0, 24, "Dynamics", "Limiter", "limiter_enabled", "dB", 1),
    FXSliderSpec("limiter_release", "Release", 10, 1000, 99, "Dynamics", "Limiter", "limiter_enabled", "ms", 0),
    FXSliderSpec("gain_db", "dB", -20, 20, 80, "Dynamics", "Gain", "gain_enabled", "dB", 1),
    # --- EQ & Filters ---
    FXSliderSpec("eq_bass", "Bass (LowShelf)", -20, 20, 40, "EQ & Filters", "EQ", None, "dB", 1),
    FXSliderSpec("eq_treble", "Treble (HighShelf)", -20, 20, 40, "EQ & Filters", "EQ", None, "dB", 1),
    FXSliderSpec("highpass_freq", "Freq", 20, 1000, 100, "EQ & Filters", "HighPass Filter", "highpass_enabled", "Hz", 0),
    FXSliderSpec("lowpass_freq", "Freq", 1000, 20000, 100, "EQ & Filters", "LowPass Filter", "lowpass_enabled", "Hz", 0),
    # --- Spatial & Time ---
    FXSliderSpec("reverb_room_size", "Room Size", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_wet_level", "Wet Level", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_dry_level", "Dry Level", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_damping", "Damping", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_width", "Width", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("delay_time", "Time", 0, 2, 100, "Spatial & Time", "Delay", "delay_enabled", "s", 2),
    FXSliderSpec("delay_feedback", "Feedback", 0, 1, 100, "Spatial & Time", "Delay", "delay_enabled", "", 2),
    FXSliderSpec("delay_mix", "Mix", 0, 1, 100, "Spatial & Time", "Delay", "delay_enabled", "", 2),
    # Convolution reverb (grill Q31): an impulse response by name, "" for none.
    FXFileSpec("convolution_ir", "Impulse response", "Spatial & Time", "Convolution Reverb"),
    FXSliderSpec("convolution_mix", "Mix", 0, 1, 100, "Spatial & Time", "Convolution Reverb", None, "", 2),
    # --- Guitar / Modulation ---
    FXSliderSpec("chorus_rate", "Rate", 0.1, 10, 50, "Guitar / Modulation", "Chorus", "chorus_enabled", "Hz", 1),
    FXSliderSpec("chorus_depth", "Depth", 0, 1, 50, "Guitar / Modulation", "Chorus", "chorus_enabled", "", 2),
    FXSliderSpec("chorus_mix", "Mix", 0, 1, 100, "Guitar / Modulation", "Chorus", "chorus_enabled", "", 2),
    FXSliderSpec("distortion_drive", "Drive", 0, 60, 60, "Guitar / Modulation", "Distortion", "distortion_enabled", "dB", 1),
    FXSliderSpec("phaser_rate", "Rate", 0.1, 10, 50, "Guitar / Modulation", "Phaser", "phaser_enabled", "Hz", 1),
    FXSliderSpec("phaser_depth", "Depth", 0, 1, 100, "Guitar / Modulation", "Phaser", "phaser_enabled", "", 2),
    FXSliderSpec("phaser_mix", "Mix", 0, 1, 100, "Guitar / Modulation", "Phaser", "phaser_enabled", "", 2),
    FXSliderSpec("clipping_thresh", "Threshold", -20, 0, 40, "Guitar / Modulation", "Clipping", "clipping_enabled", "dB", 1),
    # --- Quality / Pitch ---
    FXSliderSpec("pitch_shift_semitones", "Semitones", -12, 12, 48, "Quality / Pitch", "Pitch Shift (High Quality)", "pitch_shift_enabled", "st", 1),
    FXSliderSpec("bitcrush_depth", "Bit Depth", 2, 16, 28, "Quality / Pitch", "Bitcrush", "bitcrush_enabled", "", 1),
]

# Standalone checkbox with no slider at all (Quality / Pitch group).
FX_STANDALONE_TOGGLES = [
    ("gsm_enabled", "GSM Compressor (Phone Quality)", "Quality / Pitch"),
]

FX_GROUP_ORDER = ["Dynamics", "EQ & Filters", "Spatial & Time", "Guitar / Modulation", "Quality / Pitch"]

# --- App-settings defaults (config_qt.json) --------------------------------

SETTINGS_DEFAULTS = {
    "filename": "output",
    "format": "wav",
    "out_dir": "audio_output",
    "speed": 1.0,
    "volume": 1.0,
    "pitch": 0.0,
    "num_threads": 1,
    # Where text is cut before synthesis (kokoro_gui/engine/segmenting.py).
    "segment_target_words": 40,
    "segment_at_paragraphs": True,
    "segment_at_sentences": True,
    "segment_at_pauses": True,
    "separate": True,
    "combine": True,
    "export_subtitles": False,
    "caching": True,
    "jit_enabled": False,
    "auto_split_by_paragraph": False,
    "character_fx_paste_splits": True,
    "character_fx_copy": True,
    "notify_sound": True,       # Options > Sound when a long job finishes
    "snap_to_grid": False,      # the Timeline dock's Snap to grid button and the G key
    "spellcheck": False,        # Options > Spellcheck: underline words the dictionary lacks
    # Options > Transcript details: the master switch, then one toggle per
    # overlay (each counts only while the master is on).
    "transcript_details": False,
    "details_segments": True,
    "details_clip_info": True,
    "details_lexicon": True,
    "details_gaps": True,
    "theme": "dark",            # Options > Theme: "light" | "dark"
    "device": "auto",           # Options > Device: "auto" | "cpu" | "cuda"
    "last_project": None,       # File menu: the project launch reopens
    "recent_projects": [],      # File > Recent, most recent first (max 10)
    "show_welcome": True,       # Welcome dialog on launch (File > Welcome... reopens it)
    "import_rules": {},         # Import wizard cleanup rules: {rule_id: bool}, a missing id keeps its default
    "normalize": False,
    "trim": False,
    "apply_fx": True,
    "reverb_enabled": False,
    "reverb_room_size": 0.5,
    "reverb_wet_level": 0.3,
    "reverb_damping": 0.5,
    "reverb_dry_level": 1.0,
    "reverb_width": 1.0,
    "eq_bass": 0.0,
    "eq_treble": 0.0,
    "comp_enabled": False,
    "comp_threshold": -20.0,
    "comp_ratio": 4.0,
    "comp_attack": 1.0,
    "comp_release": 100.0,
    "distortion_enabled": False,
    "distortion_drive": 25.0,
    "chorus_enabled": False,
    "chorus_rate": 1.0,
    "chorus_depth": 0.25,
    "chorus_mix": 0.5,
    "phaser_enabled": False,
    "phaser_rate": 1.0,
    "phaser_depth": 0.5,
    "phaser_mix": 0.5,
    "clipping_enabled": False,
    "clipping_thresh": -6.0,
    "bitcrush_enabled": False,
    "bitcrush_depth": 8.0,
    "gsm_enabled": False,
    "highpass_enabled": False,
    "highpass_freq": 50.0,
    "lowpass_enabled": False,
    "lowpass_freq": 10000.0,
    "delay_enabled": False,
    "delay_time": 0.5,
    "delay_feedback": 0.0,
    "delay_mix": 0.5,
    "pitch_shift_enabled": False,
    "pitch_shift_semitones": 0.0,
    "limiter_enabled": False,
    "limiter_threshold": -1.0,
    "limiter_release": 100.0,
    "gain_enabled": False,
    "gain_db": 0.0,
    "convolution_ir": "",
    "convolution_mix": 0.5,
    # The engine new characters get (grill EN1/EN4); the Settings tab's
    # project scope sets it.
    "default_engine": "kokoro",
    # Per-engine settings (lang_code, num_threads, a backend's "Model"
    # group): {engine_id: {key: value}} (grill EN5, `QtTTSApp.engine_settings`).
    "engines": {},
    "asr_engine": "whisper",
    "lexicon": [],  # rules: {"find", "replace", "mode", "case"}, in order
    # Workspace layouts: {"Advanced": {"state": b64, "geometry": b64}, ...}
    # (kokoro_gui/qt/workspace.py). The old flat dock_state/geometry keys
    # migrate into workspaces.Advanced on first load.
    "workspaces": {},
    "active_workspace": "Advanced",
}

_FX_ENABLED_KEYS = {s.enabled_key for s in FX_FIELD_SPECS if s.enabled_key}
assert set(FX_PRESET_KEYS) == (
    {s.key for s in FX_FIELD_SPECS}
    | {t[0] for t in FX_STANDALONE_TOGGLES} | _FX_ENABLED_KEYS
)
