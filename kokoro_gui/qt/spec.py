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

from kokoro_gui.i18n import N_



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
PROGRAM_SCHEMA_GROUP = N_("Advanced")


def is_program_field(field) -> bool:
    """True for a `ConfigField` the Settings window owns (see above)."""
    if field.key == "lexicon":
        return False
    return field.key in PROGRAM_SCHEMA_KEYS or field.group == PROGRAM_SCHEMA_GROUP


# Options > Transcript details: the overlay toggles under "Show details"
# (`transcript_details`), as (settings key, menu label).
DETAIL_LAYERS = (
    ("details_segments", N_("Segment boundaries")),
    ("details_clip_info", N_("Clip info")),
    ("details_lexicon", N_("Lexicon rewrites")),
    ("details_gaps", N_("Gaps")),
)

DETAIL_LAYER_TIPS = {
    "details_segments": N_("Mark where each clip is cut into the pieces the engine speaks."),
    "details_clip_info": N_("Show each clip's character, voice, length and status."),
    "details_lexicon": N_("Show the words the Lexicon reads in another way."),
    "details_gaps": N_("Show the silence before each clip."),
}

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
    FXSliderSpec("comp_threshold", N_("Threshold"), -60, 0, 60, "Dynamics", N_("Compressor"), "comp_enabled", "dB", 1),
    FXSliderSpec("comp_ratio", N_("Ratio"), 1, 20, 19, "Dynamics", N_("Compressor"), "comp_enabled", ":1", 1),
    FXSliderSpec("comp_attack", N_("Attack"), 0.1, 100, 999, "Dynamics", N_("Compressor"), "comp_enabled", "ms", 1),
    FXSliderSpec("comp_release", N_("Release"), 10, 1000, 99, "Dynamics", N_("Compressor"), "comp_enabled", "ms", 0),
    FXSliderSpec("limiter_threshold", N_("Threshold"), -12, 0, 24, "Dynamics", N_("Limiter"), "limiter_enabled", "dB",
                 1),
    FXSliderSpec("limiter_release", N_("Release"), 10, 1000, 99, "Dynamics", N_("Limiter"), "limiter_enabled", "ms", 0),
    FXSliderSpec("gain_db", "dB", -20, 20, 80, "Dynamics", N_("Gain"), "gain_enabled", "dB", 1),
    # --- EQ & Filters ---
    FXSliderSpec("eq_bass", N_("Bass (LowShelf)"), -20, 20, 40, "EQ & Filters", N_("EQ"), None, "dB", 1),
    FXSliderSpec("eq_treble", N_("Treble (HighShelf)"), -20, 20, 40, "EQ & Filters", N_("EQ"), None, "dB", 1),
    FXSliderSpec("highpass_freq", N_("Freq"), 20, 1000, 100, "EQ & Filters", N_("HighPass Filter"), "highpass_enabled",
                 "Hz", 0),
    FXSliderSpec("lowpass_freq", N_("Freq"), 1000, 20000, 100, "EQ & Filters", N_("LowPass Filter"), "lowpass_enabled",
                 "Hz", 0),
    # --- Spatial & Time ---
    FXSliderSpec("reverb_room_size", N_("Room Size"), 0, 1, 100, "Spatial & Time", N_("Reverb"), "reverb_enabled", "",
                 2),
    FXSliderSpec("reverb_wet_level", N_("Wet Level"), 0, 1, 100, "Spatial & Time", N_("Reverb"), "reverb_enabled", "",
                 2),
    FXSliderSpec("reverb_dry_level", N_("Dry Level"), 0, 1, 100, "Spatial & Time", N_("Reverb"), "reverb_enabled", "",
                 2),
    FXSliderSpec("reverb_damping", N_("Damping"), 0, 1, 100, "Spatial & Time", N_("Reverb"), "reverb_enabled", "", 2),
    FXSliderSpec("reverb_width", N_("Width"), 0, 1, 100, "Spatial & Time", N_("Reverb"), "reverb_enabled", "", 2),
    FXSliderSpec("delay_time", N_("Time"), 0, 2, 100, "Spatial & Time", N_("Delay"), "delay_enabled", "s", 2),
    FXSliderSpec("delay_feedback", N_("Feedback"), 0, 1, 100, "Spatial & Time", N_("Delay"), "delay_enabled", "", 2),
    FXSliderSpec("delay_mix", N_("Mix"), 0, 1, 100, "Spatial & Time", N_("Delay"), "delay_enabled", "", 2),
    # Convolution reverb (grill Q31): an impulse response by name, "" for none.
    FXFileSpec("convolution_ir", N_("Impulse response"), "Spatial & Time", N_("Convolution Reverb")),
    FXSliderSpec("convolution_mix", N_("Mix"), 0, 1, 100, "Spatial & Time", N_("Convolution Reverb"), None, "", 2),
    # --- Guitar / Modulation ---
    FXSliderSpec("chorus_rate", N_("Rate"), 0.1, 10, 50, "Guitar / Modulation", N_("Chorus"), "chorus_enabled", "Hz",
                 1),
    FXSliderSpec("chorus_depth", N_("Depth"), 0, 1, 50, "Guitar / Modulation", N_("Chorus"), "chorus_enabled", "", 2),
    FXSliderSpec("chorus_mix", N_("Mix"), 0, 1, 100, "Guitar / Modulation", N_("Chorus"), "chorus_enabled", "", 2),
    FXSliderSpec("distortion_drive", N_("Drive"), 0, 60, 60, "Guitar / Modulation", N_("Distortion"),
                 "distortion_enabled", "dB", 1),
    FXSliderSpec("phaser_rate", N_("Rate"), 0.1, 10, 50, "Guitar / Modulation", N_("Phaser"), "phaser_enabled", "Hz",
                 1),
    FXSliderSpec("phaser_depth", N_("Depth"), 0, 1, 100, "Guitar / Modulation", N_("Phaser"), "phaser_enabled", "", 2),
    FXSliderSpec("phaser_mix", N_("Mix"), 0, 1, 100, "Guitar / Modulation", N_("Phaser"), "phaser_enabled", "", 2),
    FXSliderSpec("clipping_thresh", N_("Threshold"), -20, 0, 40, "Guitar / Modulation", N_("Clipping"),
                 "clipping_enabled", "dB", 1),
    # --- Quality / Pitch ---
    FXSliderSpec("pitch_shift_semitones", N_("Semitones"), -12, 12, 48, "Quality / Pitch",
                 N_("Pitch Shift (High Quality)"), "pitch_shift_enabled", "st", 1),
    FXSliderSpec("bitcrush_depth", N_("Bit Depth"), 2, 16, 28, "Quality / Pitch", N_("Bitcrush"), "bitcrush_enabled",
                 "", 1),
]

# Standalone checkbox with no slider at all (Quality / Pitch group).
FX_STANDALONE_TOGGLES = [
    ("gsm_enabled", N_("GSM Compressor (Phone Quality)"), N_("Quality / Pitch")),
]

# Tooltips: what an effect does (by its enable key) and what a control changes.
FX_TIPS = {
    "comp_enabled": N_("Evens out loud and quiet parts."),
    "comp_threshold": N_("Sound louder than this is turned down."),
    "comp_ratio": N_("How hard sound over the threshold is turned down. 4 lets a quarter of the excess through."),
    "comp_attack": N_("How fast the compressor reacts when the sound gets loud."),
    "comp_release": N_("How fast it lets go when the sound gets quiet again."),
    "limiter_enabled": N_("Keeps peaks from going over a level."),
    "limiter_threshold": N_("The level that peaks are held at."),
    "limiter_release": N_("How fast the limiter lets go after a peak."),
    "gain_enabled": N_("Makes the sound louder or quieter."),
    "gain_db": N_("How much louder or quieter."),
    "eq_bass": N_("Raises or lowers the low end."),
    "eq_treble": N_("Raises or lowers the high end."),
    "highpass_enabled": N_("Cuts low rumble."),
    "highpass_freq": N_("Everything below this frequency is cut."),
    "lowpass_enabled": N_("Cuts hiss and brightness."),
    "lowpass_freq": N_("Everything above this frequency is cut."),
    "reverb_enabled": N_("Puts the voice in a room."),
    "reverb_room_size": N_("How big the room is."),
    "reverb_wet_level": N_("How loud the room is."),
    "reverb_dry_level": N_("How loud the voice itself is."),
    "reverb_damping": N_("How much the walls soak up the high end."),
    "reverb_width": N_("How wide the room sounds in stereo."),
    "delay_enabled": N_("Repeats the sound as an echo."),
    "delay_time": N_("The time between echoes."),
    "delay_feedback": N_("How much of each echo is repeated."),
    "delay_mix": N_("How loud the echoes are against the voice."),
    "convolution_ir": N_("The recording of a room or device that shapes the reverb. None turns it off."),
    "convolution_mix": N_("How loud the reverb is against the voice."),
    "chorus_enabled": N_("Thickens the voice with slightly detuned copies."),
    "chorus_rate": N_("How fast the copies drift."),
    "chorus_depth": N_("How far the copies drift."),
    "chorus_mix": N_("How loud the copies are against the voice."),
    "distortion_enabled": N_("Overdrives the sound."),
    "distortion_drive": N_("How hard the sound is overdriven."),
    "phaser_enabled": N_("Adds a sweeping, hollow sound."),
    "phaser_rate": N_("How fast the sweep goes."),
    "phaser_depth": N_("How far the sweep goes."),
    "phaser_mix": N_("How loud the sweep is against the voice."),
    "clipping_enabled": N_("Chops off the peaks for a harsh sound."),
    "clipping_thresh": N_("The level the peaks are chopped at."),
    "pitch_shift_enabled": N_("Raises or lowers the voice without changing its speed."),
    "pitch_shift_semitones": N_("How far, in semitones. 12 is an octave."),
    "bitcrush_enabled": N_("Lowers the sample resolution for a lo-fi sound."),
    "bitcrush_depth": N_("The resolution in bits. Lower is rougher."),
    "gsm_enabled": N_("Cuts the sound to the quality of a mobile phone call."),
}

FX_GROUP_ORDER = [N_("Dynamics"), N_("EQ & Filters"), N_("Spatial & Time"), N_("Guitar / Modulation"),
                  N_("Quality / Pitch")]

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
    "playback_rate": 1.0,       # the Transport dock's speed combo and the [ ] keys, 0.5 to 2
    "segment_cache_max_mb": 2048,  # Options > Storage: the generated audio cache is trimmed to this, 0 = no limit
    "spellcheck": False,        # Options > Spellcheck: underline words the dictionary lacks
    "proof_threshold": 0.92,    # Proof dock: a clip matching its text less closely than this is flagged
    # Options > Transcript details: the master switch, then one toggle per
    # overlay (each counts only while the master is on).
    "transcript_details": False,
    "details_segments": True,
    "details_clip_info": True,
    "details_lexicon": True,
    "details_gaps": True,
    "theme": "dark",            # Options > Theme: "light" | "dark"
    "language": "system",       # Options > Language: "system" or a code in kokoro_gui/locales, read at launch
    "device": "auto",           # Options > Device: "auto" | "cpu" | "cuda"
    "last_project": None,       # File menu: the project launch reopens
    "recent_projects": [],      # File > Recent, most recent first (max 10)
    "show_welcome": True,       # Welcome dialog on launch (File > Welcome... reopens it)
    "clone_consent_confirm": True,  # the consent checkbox when cloning from a recording (plan 33, grill PG9)
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
    # Named snapshots {name: {"state", "geometry"}}, written only by Save
    # workspace as and Update workspace (grill UI17), and the active custom
    # workspace's live arrangement {"workspace", "state", "geometry",
    # "modified"} that a restart brings back (PG31).
    "custom_workspaces": {},
    "workspace_live": {},
    "active_workspace": "Advanced",
}

_FX_ENABLED_KEYS = {s.enabled_key for s in FX_FIELD_SPECS if s.enabled_key}
assert set(FX_PRESET_KEYS) == (
    {s.key for s in FX_FIELD_SPECS}
    | {t[0] for t in FX_STANDALONE_TOGGLES} | _FX_ENABLED_KEYS
)
