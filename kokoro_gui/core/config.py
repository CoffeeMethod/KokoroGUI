"""The config dicts a clip generates and post-processes with.

`generation_config` is exactly what decides what a clip's audio *is* (the
segment key hashes it and nothing else); `clip_config` adds everything that
doesn't change the audio (output location and naming, threads, caching,
the post keys). The app's `_assemble_generation_config` and
`_assemble_clip_config` read the Settings tab, the Lexicon and the export
settings into `GenerationInputs` and `OutputInputs` and call these; a
headless caller uses `GenerationInputs.defaults()` and
`OutputInputs.defaults()`.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from kokoro_gui.audio import post
from kokoro_gui.core.engines import model_fields
from kokoro_gui.daw.arrangement import heading_clip_id, heading_speed
from kokoro_gui.engine.lexicon import normalize_rules
from kokoro_gui.qt import project as project_io, spec

# The `time_id` format every config carries (`QtTTSApp.timecode_format`).
TIMECODE_FORMAT = "%Y%m%d%H%M%S"


def segmentation_config(state: dict) -> dict:
    """The segmentation keys (`spec.SEGMENTATION_KEYS`) from `state`,
    defaulted: where every path cuts text into pieces."""
    return {key: state.get(key, spec.SETTINGS_DEFAULTS[key]) for key in spec.SEGMENTATION_KEYS}


@dataclass(frozen=True)
class GenerationInputs:
    """The project-wide values a clip's config starts from: the Settings
    tab's speed, pitch, volume, normalize, trim and caching, the
    segmentation keys, and the Lexicon's rules (raw, as stored)."""
    speed: float
    pitch: float
    volume: float
    normalize: bool
    trim_silence: bool
    caching: bool
    lexicon: object = None
    segmentation: dict = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: dict, lexicon) -> "GenerationInputs":
        """From the Settings tab's `get_state()` and `settings["lexicon"]`.
        Speed and pitch are read by index, as the config code always has."""
        defaults = spec.SETTINGS_DEFAULTS
        return cls(
            speed=state["speed"],
            pitch=state["pitch"],
            volume=state.get("volume", defaults["volume"]),
            normalize=state.get("normalize", defaults["normalize"]),
            trim_silence=state.get("trim_silence", defaults["trim"]),
            caching=state.get("caching", defaults["caching"]),
            lexicon=lexicon,
            segmentation=segmentation_config(state),
        )

    @classmethod
    def defaults(cls) -> "GenerationInputs":
        """`spec.SETTINGS_DEFAULTS`' values: what a fresh install's Settings
        tab holds."""
        defaults = spec.SETTINGS_DEFAULTS
        return cls(
            speed=defaults["speed"],
            pitch=defaults["pitch"],
            volume=defaults["volume"],
            normalize=defaults["normalize"],
            trim_silence=defaults["trim"],
            caching=defaults["caching"],
            lexicon=list(defaults["lexicon"]),
            segmentation=segmentation_config({}),
        )


@dataclass(frozen=True)
class OutputInputs:
    """Where a clip's file goes outside a project dir and what it's called:
    the export settings' format and folder, the expanded filename and the
    `time_id` stamp."""
    format: str
    out_dir: str
    filename: str
    time_id: str

    @classmethod
    def defaults(cls) -> "OutputInputs":
        defaults = spec.SETTINGS_DEFAULTS
        return cls(format=defaults["format"], out_dir=defaults["out_dir"], filename=defaults["filename"],
                   time_id=time.strftime(TIMECODE_FORMAT))


def variant_voice(document, clip, backend):
    """The reference name `clip.overrides["variant"]` picks from its
    character's `variants`, or None. Only a cloning backend has variants;
    on any other the override is ignored."""
    variant = (clip.overrides or {}).get("variant")
    if not variant or not getattr(backend.capabilities, "supports_voice_cloning", False):
        return None
    character = document.get_character(clip.character_id)
    if character is None:
        return None
    return (character.variants or {}).get(variant) or None


def generation_config(inputs: GenerationInputs, engine_settings: dict, document, clip, backend,
                      project_dir, memo=None) -> dict:
    """Exactly the inputs that decide what a clip's audio *is*: the segment
    key hashes these and nothing else, and `clip_config` is built on top.
    The project defaults (`inputs`), the clip's own engine's `lang_code`
    and "Model" schema group (Audio8's sampling knobs) from
    `engine_settings`, the Lexicon (the key is over the text after
    substitution), the segmentation settings (they decide the pieces), then
    the clip's `effective_config_for_clip` (character preset, then
    overrides) on top, then `project_dir` and the clip's take. The take is
    read off `clip.overrides` directly, not through the
    `ALLOWED_PRESET_KEYS` whitelist, which is for untrusted preset files and
    shouldn't widen for a runtime counter. `memo` is the schema memo
    `model_fields` keeps."""
    config = {
        "engine_id": backend.id,
        "lang_code": engine_settings.get("lang_code"),
        "voice": engine_settings.get("voice"),
        "speed": inputs.speed,
        "pitch": inputs.pitch,
        "lexicon": normalize_rules(inputs.lexicon),
        **inputs.segmentation,
    }
    for key, default in model_fields(backend, memo):
        config[key] = engine_settings.get(key, default)
    own = dict(document.effective_config_for_clip(clip))
    for key in ("voice", "speed", "pitch", "lang_code"):
        if key in own:
            config[key] = own[key]
    variant = variant_voice(document, clip, backend)
    if variant:
        config["voice"] = variant
    factor = heading_speed(document)
    if factor != 1.0 and heading_clip_id(document) == clip.id:
        # A chapter title reads at its own pace (the project's `heading_speed`).
        config["speed"] = round(float(config.get("speed", 1.0) or 1.0) * factor, 4)
    config["project_dir"] = project_dir
    config["take"] = int(clip.overrides.get("take", 0) or 0)
    return config


def clip_config(inputs: GenerationInputs, output: OutputInputs, engine_settings: dict, document, clip,
                backend, project_dir, project_settings, fx, memo=None) -> dict:
    """The config dict for a per-clip generate and, through `post_config`,
    for read-time post-processing: `generation_config` plus everything that
    doesn't change the audio, with the clip's character/override settings
    merged on top so the clip's own values win. Defaults come first:
    `process_chunk_task` reads `config['voice']` by direct indexing, so a
    clip with no character still ends up with usable defaults. `fx` is the
    clip's `core.fx.FxResolution`."""
    config = {
        # The clip's engine's own settings (threads, Audio8's
        # cache_reference_codes, ...); the generation keys among them
        # are set again below by generation_config.
        "num_threads": 1,
        **engine_settings,
        "format": output.format,
        "out_dir": output.out_dir,
        "caching": inputs.caching,
        "time_id": output.time_id,
        "volume": inputs.volume,
        "normalize": inputs.normalize,
        "trim_silence": inputs.trim_silence,
        "filename": output.filename,
    }

    own = dict(document.effective_config_for_clip(clip))
    # ALLOWED_PRESET_KEYS whitelists "trim", but process_audio reads
    # "trim_silence" - same inline rename every other caller does.
    if "trim" in own:
        own["trim_silence"] = own.pop("trim")
    config.update(own)
    config.update(generation_config(inputs, engine_settings, document, clip, backend, project_dir, memo))
    if project_dir:
        # Clips generate straight into the project dir, once, named by
        # their segment key (Claude/old/PLAN_tbaw_bundle.md section 3). The
        # bundle's audio format decides the extension of new segments,
        # over a preset's `format` (that one is for the export path).
        config["out_dir"] = os.path.join(project_dir, *project_io.AUDIO_GENERATED.split("/"))
        config["segment_naming"] = "cache_key"
        config["format"] = project_io.bundle_options(project_settings)["audio_format"]

    # effective_config_for_clip only ever carries the FX preset's *name*;
    # the resolver turned project values + character preset + clip preset
    # + clip.fx_override into the actual FX keys and the ANDed apply_fx.
    config.update(fx.values)
    config["apply_fx"] = fx.apply_fx
    # Fit to slot's stretch factor (a post key). Read off the overrides
    # directly, like the take: it is per clip, never a preset value.
    stretch = (clip.overrides or {}).get("time_stretch")
    if stretch is not None:
        config["time_stretch"] = stretch
    return config


def post_config(config: dict) -> dict:
    """The `POST_KEYS` subset of a `clip_config`: what the transport, the
    exporter and the timeline waveform apply on top of the raw segment
    files."""
    return post.extract_post_config(config)
