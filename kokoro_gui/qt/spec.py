"""Pure-data constants for the Qt frontend's field lists (generation config
keys, FX preset keys/slider specs, language/voice tables, settings defaults).

This module has no Qt imports so both `kokoro_gui/qt/*` and the test suite can
import it standalone.

These constants originated as a mirror of the now-retired Tk frontend's
(`gui.py`, `kokoro_gui/ui/*.py`) hard-coded field lists, kept in sync via
`tests/gui_qt/test_qt_config_assembly.py`'s cross-frontend check during the
migration (see PLAN_qt_and_engine_abstraction.md, workstream 3a). Now that Tk
has been removed, this module is simply the canonical source of truth for the
Qt frontend.

- FX_FIELD_SPECS has no widget for seven FX_PRESET_KEYS fields
  (reverb_dry_level, chorus_mix, phaser_depth, phaser_mix, comp_attack,
  comp_release, limiter_release) — a pre-existing gap inherited from Tk, not
  yet closed (see ROADMAP.md).
"""
from dataclasses import dataclass
from typing import Optional

# --- Generation config dict (non-FX keys) --------------------------------

GENERATION_BASE_KEYS = [
    "engine_id", "lang_code", "voice", "speed", "split_pattern", "filename",
    "format", "out_dir", "separate", "combine", "export_subtitles", "caching",
    "time_id", "num_threads", "volume", "pitch", "normalize", "trim_silence",
    "lexicon",
]

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


FX_FIELD_SPECS = [
    # --- Dynamics ---
    FXSliderSpec("comp_threshold", "Threshold", -60, 0, 60, "Dynamics", "Compressor", "comp_enabled", "dB", 1),
    FXSliderSpec("comp_ratio", "Ratio", 1, 20, 19, "Dynamics", "Compressor", "comp_enabled", ":1", 1),
    FXSliderSpec("limiter_threshold", "Threshold", -12, 0, 24, "Dynamics", "Limiter", "limiter_enabled", "dB", 1),
    FXSliderSpec("gain_db", "dB", -20, 20, 80, "Dynamics", "Gain", "gain_enabled", "dB", 1),
    # --- EQ & Filters ---
    FXSliderSpec("eq_bass", "Bass (LowShelf)", -20, 20, 40, "EQ & Filters", "EQ", None, "dB", 1),
    FXSliderSpec("eq_treble", "Treble (HighShelf)", -20, 20, 40, "EQ & Filters", "EQ", None, "dB", 1),
    FXSliderSpec("highpass_freq", "Freq", 20, 1000, 100, "EQ & Filters", "HighPass Filter", "highpass_enabled", "Hz", 0),
    FXSliderSpec("lowpass_freq", "Freq", 1000, 20000, 100, "EQ & Filters", "LowPass Filter", "lowpass_enabled", "Hz", 0),
    # --- Spatial & Time ---
    FXSliderSpec("reverb_room_size", "Room Size", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_wet_level", "Wet Level", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_damping", "Damping", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("reverb_width", "Width", 0, 1, 100, "Spatial & Time", "Reverb", "reverb_enabled", "", 2),
    FXSliderSpec("delay_time", "Time", 0, 2, 100, "Spatial & Time", "Delay", "delay_enabled", "s", 2),
    FXSliderSpec("delay_feedback", "Feedback", 0, 1, 100, "Spatial & Time", "Delay", "delay_enabled", "", 2),
    FXSliderSpec("delay_mix", "Mix", 0, 1, 100, "Spatial & Time", "Delay", "delay_enabled", "", 2),
    # --- Guitar / Modulation ---
    FXSliderSpec("chorus_rate", "Rate", 0.1, 10, 50, "Guitar / Modulation", "Chorus", "chorus_enabled", "Hz", 1),
    FXSliderSpec("chorus_depth", "Depth", 0, 1, 50, "Guitar / Modulation", "Chorus", "chorus_enabled", "", 2),
    FXSliderSpec("distortion_drive", "Drive", 0, 60, 60, "Guitar / Modulation", "Distortion", "distortion_enabled", "dB", 1),
    FXSliderSpec("phaser_rate", "Rate", 0.1, 10, 50, "Guitar / Modulation", "Phaser", "phaser_enabled", "Hz", 1),
    FXSliderSpec("clipping_thresh", "Threshold", -20, 0, 40, "Guitar / Modulation", "Clipping", "clipping_enabled", "dB", 1),
    # --- Quality / Pitch ---
    FXSliderSpec("pitch_shift_semitones", "Semitones", -12, 12, 48, "Quality / Pitch", "Pitch Shift (High Quality)", "pitch_shift_enabled", "st", 1),
    FXSliderSpec("bitcrush_depth", "Bit Depth", 2, 16, 28, "Quality / Pitch", "Bitcrush", "bitcrush_enabled", "", 1),
]

# Standalone checkbox with no slider at all (Quality / Pitch group).
FX_STANDALONE_TOGGLES = [
    ("gsm_enabled", "GSM Compressor (Phone Quality)", "Quality / Pitch"),
]

# FX_PRESET_KEYS entries with no matching widget in either frontend today
# (see module docstring) - still valid dict keys, just not user-editable.
FX_KEYS_WITHOUT_WIDGET = {"reverb_dry_level", "chorus_mix", "phaser_depth", "phaser_mix", "comp_attack", "comp_release", "limiter_release"}

FX_GROUP_ORDER = ["Dynamics", "EQ & Filters", "Spatial & Time", "Guitar / Modulation", "Quality / Pitch"]

# --- Voice / language display data ----------------------------------------

LANGUAGES = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Italian": "i",
    "Portuguese": "p",
    "Japanese": "j",
    "Chinese": "z",
}

VOICE_DB = {
    "a": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"],
    "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "i": ["if_sara", "im_nicola"],
    "p": ["pf_dora", "pm_alex"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunjian"],
}

MIX_PREVIEW_TEXT = {
    "f": "Ceci est un aperçu de votre voix personnalisée.",
    "e": "Esta es una vista previa de su voz personalizada.",
    "i": "Questa è un'anteprima della tua voce personalizzata.",
    "p": "Esta é uma prévia da sua voz personalizada.",
    "j": "これはカスタム合成音声のプレビューです。",
    "z": "这是您的自定义混合语音预览。",
}
MIX_PREVIEW_TEXT_DEFAULT = "This is a preview of your custom mixed voice."

# --- App-settings defaults (config_qt.json) --------------------------------

SETTINGS_DEFAULTS = {
    "lang_code": "a",
    "voice": "af_heart",
    "filename": "output",
    "format": "wav",
    "out_dir": "audio_output",
    "speed": 1.0,
    "volume": 1.0,
    "pitch": 0.0,
    "num_threads": 1,
    "split_pattern": r"\n+",
    "separate": True,
    "combine": True,
    "export_subtitles": False,
    "caching": True,
    "jit_enabled": False,
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
    "engine_id": "kokoro",
    "lexicon": {},
    "dock_state": None,   # base64 QMainWindow.saveState() bytes, set at runtime
    "geometry": None,     # base64 QMainWindow.saveGeometry() bytes, set at runtime
}

_FX_ENABLED_KEYS = {s.enabled_key for s in FX_FIELD_SPECS if s.enabled_key}
assert set(FX_PRESET_KEYS) == (
    {s.key for s in FX_FIELD_SPECS} | FX_KEYS_WITHOUT_WIDGET
    | {t[0] for t in FX_STANDALONE_TOGGLES} | _FX_ENABLED_KEYS
)
