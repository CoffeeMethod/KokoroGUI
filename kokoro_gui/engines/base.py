"""Cross-engine surface for TTS backends.

This is workstream 1 of PLAN_qt_and_engine_abstraction.md ("Abstract the
model-specific parts"): a thin, engine-agnostic description of what a TTS
backend *is* (id, display name, capability flags) and what settings it takes
(`get_config_schema()`), so the GUI can eventually render per-engine panels
and gate engine-specific tabs/features without hard-coding "Kokoro" anywhere.

Deliberately thin. `KokoroEngine`'s actual generation/streaming entry points
(`start_conversion`, `start_jit_conversion`, `generate_preview`) stay
callback-driven and scheduled onto `AsyncLoopThread` - per the plan's
"Explicitly out of scope" note, that execution model stays Kokoro-backend-
private for now. A uniform async `generate()`/`start_jit()` request/response
surface every backend implements the same way is real design work that's
premature until a second backend actually exists to validate it against
(see migration step 5); inventing it here, unvalidated, is exactly the kind
of over-fit-to-Kokoro abstraction the plan warns against for voice mixing.
So this Protocol only covers what's true for *any* backend today: identity,
capabilities, its config schema, its voice list, and cancellation, plus the
five `.tbaw` hooks (Claude/old/PLAN_tbaw_bundle.md section 5) that
`BackendHooksMixin` gives working defaults for: `engine_version`,
`cache_key_extra`, `resolve_voice_file` (the segment-key trio, forwarded to
the wrapped engine because `process_chunk_task` runs there without an
adapter reference), `collect_project_assets` and `on_project_opened` (adapter
only). Nothing in this package imports `kokoro_gui/daw/`: the project layer
walks the document and hands each backend the voice names it uses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable


class ConfigFieldType(str, Enum):
    """Widget shape a GUI should render for a `ConfigField`."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    TEXT = "text"
    FILE = "file"
    CHOICE = "choice"
    SLIDER = "slider"


@dataclass(frozen=True)
class ConfigField:
    """One entry in a backend's `get_config_schema()`.

    `choices`, when set, is a list of `(label, value)` pairs for CHOICE/SLIDER
    fields with a fixed, known set of options (languages, output format).
    "voice" leaves `choices=None`: its options depend on the language and on
    the files on disk, so the GUI asks `get_voices(lang_code)`.
    """
    key: str
    label: str
    type: ConfigFieldType
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list] = None
    group: str = "General"


@dataclass(frozen=True)
class EngineCapabilities:
    """Flags the GUI uses to show/hide whole panels rather than special-
    casing engine names/ids."""
    supports_voice_mixing: bool = False       # show the Mixing tab at all
    supports_voice_cloning: bool = False      # show an upload-a-sample panel
    supports_multi_speaker_script: bool = False  # [Speaker:FX]: syntax
    is_local_model: bool = True               # device/GPU picker vs API-key field
    supports_jit_streaming: bool = True
    # Generation stamps `Segment.words` from the model's own timings; without
    # it the app aligns words with Whisper after a generate.
    supports_word_timing: bool = False
    # The model honours the `speed` config key. Fit to slot regenerates at a
    # new speed when it does and time-stretches the render when it doesn't.
    supports_speed: bool = True


# Choice presets for schema fields whose *meaning* isn't actually
# model-specific, just conventionally offered by more than one backend:
# which container formats get written to disk. Backends are free to ignore
# these or offer their own instead - they're shared defaults, not part of
# the Protocol.
COMMON_OUTPUT_FORMAT_CHOICES = [("wav", "wav"), ("flac", "flac"), ("mp3", "mp3"), ("ogg", "ogg")]

# Schema keys every engine shares one value for: where text is cut, the
# output format, caching, the lexicon, and the speed and pitch a character
# overrides. Every other key in a backend's schema is per engine and lives in
# `config_qt.json`'s `engines[<id>]` bucket (grill EN5): `lang_code`,
# `num_threads`, the "Model" group, and the default `voice`, since one
# engine's voice names mean nothing to another.
SHARED_CONFIG_KEYS = frozenset({
    "segment_target_words", "segment_at_paragraphs", "segment_at_sentences", "segment_at_pauses",
    "format", "caching", "lexicon", "speed", "pitch",
})


def per_engine_fields(schema: list) -> list:
    """The fields of `schema` whose value is kept per engine."""
    return [f for f in schema if f.key not in SHARED_CONFIG_KEYS]


def segmentation_fields() -> list:
    """The Settings fields for where text is cut before synthesis
    (kokoro_gui/engine/segmenting.py, grill PR6): a word target and the
    three boundary toggles. Every backend lists them; they apply
    project-wide, not per character."""
    from kokoro_gui.engine import segmenting

    return [
        ConfigField("segment_target_words", "Target Words per Segment", ConfigFieldType.INT,
                    default=segmenting.DEFAULT_TARGET_WORDS, min=segmenting.MIN_TARGET_WORDS,
                    max=segmenting.MAX_TARGET_WORDS, step=5, group="Generation"),
        ConfigField("segment_at_paragraphs", "Split at Paragraphs", ConfigFieldType.BOOL,
                    default=True, group="Generation"),
        ConfigField("segment_at_sentences", "Split at Sentences", ConfigFieldType.BOOL,
                    default=True, group="Generation"),
        ConfigField("segment_at_pauses", "Split at Pauses (, ; : dashes)", ConfigFieldType.BOOL,
                    default=True, group="Generation"),
    ]


def common_fields(speed_range=(0.5, 2.0), max_threads: int = 32, pitch: bool = True,
                  caching_default: bool = True) -> list:
    """The fields every backend lists after its own `lang_code` and `voice`:
    speed, pitch (unless the backend has none), the segmentation fields,
    output format, parallel threads and the segment cache. A backend
    appends its own fields after these."""
    fields = [
        ConfigField("speed", "Speed", ConfigFieldType.SLIDER,
                    default=1.0, min=speed_range[0], max=speed_range[1], step=0.1, group="Generation"),
    ]
    if pitch:
        fields.append(ConfigField("pitch", "Pitch", ConfigFieldType.SLIDER,
                                  default=0.0, min=-12, max=12, step=1, group="Audio"))
    fields.extend([
        *segmentation_fields(),
        ConfigField("format", "Output Format", ConfigFieldType.CHOICE,
                    default="wav", choices=list(COMMON_OUTPUT_FORMAT_CHOICES), group="Generation"),
        ConfigField("num_threads", "Parallel Threads", ConfigFieldType.INT,
                    default=1, min=1, max=max_threads, step=1, group="Advanced"),
        ConfigField("caching", "Enable Segment Cache", ConfigFieldType.BOOL,
                    default=caching_default, group="Advanced"),
    ])
    return fields


@dataclass(frozen=True)
class VoiceInfo:
    """One selectable voice, as reported by a backend's `get_voices()`."""
    id: str
    display_name: str
    lang_code: Optional[str] = None
    is_custom: bool = False


@dataclass
class Synthesis:
    """One `SynthesisModel.synthesize` result: `audio` a 1-D float32 array at
    the model's `sample_rate`, `words` `[text, start_s, end_s]` relative to
    its start (`[]` for a model with no timings)."""
    audio: Any
    words: list = field(default_factory=list)


@runtime_checkable
class SynthesisModel(Protocol):
    """What an engine is once the shared machinery is taken away: the
    `EngineRunner` (kokoro_gui/engine/runner.py) supplies the worker, the
    cancel event, the callbacks, caching, FX, conversion, JIT and SRT around
    one of these. `ModelBase` in runner.py gives defaults for everything but
    `synthesize`.

    `concurrency` is "per_thread" (the model keeps one instance per worker
    thread itself, like Kokoro's thread-local `KPipeline`) or "shared" (one
    instance; the runner serializes `synthesize` calls through a lock)."""

    engine_id: str
    sample_rate: int
    concurrency: str

    def load(self, lang_code: Optional[str], device: Optional[str]) -> Any:
        """Loads the model for `lang_code` on `device` ("auto"/"cpu"/"cuda");
        returns a truthy readiness token. May download weights."""
        ...

    def synthesize(self, text: str, voice: str, speed: float, lang_code: Optional[str],
                   params: dict) -> Synthesis:
        """Speaks `text` (one piece, already segmented) with the resolved
        `voice`. `params` is the generation config plus `cancel_event` (a
        long synthesis may stop early and return what it has)."""
        ...

    def engine_version(self) -> str:
        ...

    def cache_key_extra(self, config: dict) -> dict:
        ...

    def resolve_voice_path(self, name: str, project_dir: Optional[str] = None) -> str:
        ...


@dataclass(frozen=True)
class BundleAsset:
    """One file a backend wants in a `.tbaw` bundle: where it goes inside the
    zip (`engines/<id>/...`, forward slashes) and where its bytes are now."""
    bundle_path: str
    source_path: str


class BackendHooksMixin:
    """Working defaults for the `.tbaw` hooks. An adapter that wraps an
    engine exposing `engine_version`/`cache_key_extra`/`resolve_voice_file`
    (every engine built on `CachingMixin`) forwards to it; otherwise the
    key gets the package version for the adapter's id, no extra inputs, and
    no voice file. `project_dir` is whatever `on_project_opened` last
    recorded, for listings that should show project-local assets first."""

    project_dir: Optional[str] = None
    # "named" (built into the model), "embedding" (a file per voice) or
    # "reference" (a wav + transcript per voice); the file kinds set
    # `voice_store` (kokoro_gui/engines/voice_store.py), which gives
    # `get_voices` and `collect_project_assets` below their defaults.
    voice_kind = "named"
    voice_store = None

    def builtin_voices(self, lang_code: Optional[str] = None) -> list:
        """The `VoiceInfo`s built into the model for `lang_code` (every
        language's when None). None by default."""
        return []

    def get_voices(self, lang_code: Optional[str] = None) -> list:
        """`builtin_voices(lang_code)`, then the voice store's (the open
        project's first), each name once."""
        voices = list(self.builtin_voices(lang_code))
        seen = {v.id for v in voices}
        if self.voice_store is not None:
            for name in self.voice_store.list_voices(self.project_dir):
                if name not in seen:
                    seen.add(name)
                    voices.append(VoiceInfo(id=name, display_name=name, lang_code=None, is_custom=True))
        return voices

    def get_languages(self) -> list:
        """`[(label, code), ...]`: the choices of the schema's `lang_code`
        field, or none when the backend has no such field."""
        field = next((f for f in self.get_config_schema() if f.key == "lang_code"), None)
        return list(field.choices or []) if field is not None else []

    def word_timing_for(self, lang_code: Optional[str]) -> bool:
        """Whether a generate in `lang_code` stamps `Segment.words` itself.
        Default: the `supports_word_timing` capability; a backend whose
        timings exist only for some languages narrows it."""
        return bool(getattr(self.capabilities, "supports_word_timing", False))

    def preview_text(self, lang_code: Optional[str] = None) -> str:
        """A short sentence to preview a voice with in `lang_code`."""
        return "This is a preview of your custom voice."

    def engine_version(self) -> str:
        """What goes into the segment key and `manifest.engines[id].version`.
        A backend that changes output without a package bump must change
        this string."""
        engine = getattr(self, "engine", None)
        hook = getattr(engine, "engine_version", None)
        if callable(hook):
            return hook()
        from kokoro_gui.engine.caching import get_engine_version

        return get_engine_version(self.id)

    def cache_key_extra(self, config: dict) -> dict:
        """Backend-specific generation inputs folded into `segment_key`."""
        engine = getattr(self, "engine", None)
        hook = getattr(engine, "cache_key_extra", None)
        return dict(hook(config) or {}) if callable(hook) else {}

    def resolve_voice_file(self, name: str, project_dir: Optional[str] = None) -> Optional[str]:
        """The file `name` resolves to, project-local first, or `None` for a
        built-in voice."""
        engine = getattr(self, "engine", None)
        hook = getattr(engine, "resolve_voice_file", None)
        return hook(name, project_dir) if callable(hook) else None

    def collect_project_assets(self, voice_names, project_dir=None) -> tuple:
        """`([BundleAsset, ...], meta)`: every file under `engines/<id>/`
        needed to reproduce the given voice names, plus the opaque `meta`
        dict written to `manifest.engines[id]`. A name that resolves to
        nothing is skipped (the project still saves; the character still
        names it). Default: the voice store's files for each name (none for
        a "named" engine), empty meta."""
        if self.voice_store is None:
            return [], {}
        assets = []
        for name in sorted(voice_names):
            assets.extend(self.voice_store.bundle_assets(name, project_dir))
        return assets, {}

    def on_project_opened(self, project_dir: Optional[str], meta: dict) -> None:
        """Called after a project is opened (or created) with this backend's
        manifest `meta`. Must not load a model: record what to do and do it
        on the first generate. The default remembers the dir for listings."""
        self.project_dir = project_dir

    # -- what the GUI calls (never `backend.engine.*`). Each job returns the
    # `concurrent.futures.Future` of a coroutine on the engine's worker.

    def run(self, coro):
        """Schedules `coro` on this engine's worker thread."""
        return self.engine.worker.run_coro(coro)

    def ensure_ready(self, lang_code: Optional[str], device: Optional[str] = None):
        """Loads the model (a download on first use) in `lang_code`."""
        return self.run(self.engine.init_pipeline_async(lang_code, device=device))

    def is_ready(self) -> bool:
        return bool(getattr(self.engine, "pipeline", None))

    def preview(self, text, voice, speed, output_path, extra_config=None, lang_code=None):
        """Speaks up to two segments of `text` into `output_path`."""
        return self.run(self.engine.generate_preview(text, voice, speed, output_path, extra_config,
                                                     lang_code=lang_code))

    def generate_clip(self, chunk_data):
        """One clip's `(index, text, config)`; resolves to its result dicts."""
        return self.run(self.engine.generate_clip_audio(chunk_data))

    def generate_clips(self, items, progress=None):
        """`[(clip_id, text, config), ...]`; resolves to one outcome per clip."""
        return self.run(self.engine.generate_dirty_clips(items, progress_callback=progress))

    def convert_document(self, text: str, config: dict, jit: bool = False) -> None:
        """The whole-document path (no clips yet): batch, or JIT streaming
        when `jit` and the backend supports it. Reports through the
        callbacks, not a Future."""
        if jit and self.capabilities.supports_jit_streaming:
            self.engine.start_jit_conversion(text, config)
        else:
            self.engine.start_conversion(text, config)

    def was_cancelled(self) -> bool:
        return self.engine.cancel_event.is_set()

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.engine, "SAMPLE_RATE", 24000) or 24000)

    def set_callbacks(self, on_status=None, on_progress=None, on_finish=None) -> None:
        """Where the engine reports status, progress and the end of a
        whole-document job (the app's signal bridge)."""
        self.engine.on_status = on_status
        self.engine.on_progress = on_progress
        self.engine.on_finish = on_finish

    def stop(self) -> None:
        """Stops the worker thread (app close)."""
        self.engine.worker.stop()


def bundle_asset_for(name: str, directory: str, extension: str, bundle_dir: str,
                     project_dir: Optional[str] = None) -> Optional[BundleAsset]:
    """Helper for `collect_project_assets`: `<name><extension>` looked up in
    the project-local `bundle_dir` first, then in the global `directory`,
    returned as a `BundleAsset` at `<bundle_dir>/<name><extension>`."""
    safe = os.path.basename(name)
    if not safe:
        return None
    candidates = []
    if project_dir:
        candidates.append(os.path.join(project_dir, *bundle_dir.split("/")))
    candidates.append(directory)
    for candidate_dir in candidates:
        path = os.path.join(candidate_dir, f"{safe}{extension}")
        if os.path.isfile(path):
            return BundleAsset(f"{bundle_dir}/{safe}{extension}", os.path.abspath(path))
    return None


@runtime_checkable
class TTSEngineBackend(Protocol):
    id: str
    display_name: str
    capabilities: EngineCapabilities

    def get_config_schema(self) -> list:
        """Return this backend's `ConfigField` list, describing the settings
        a generic GUI panel would need to render it."""
        ...

    def get_voices(self, lang_code: Optional[str] = None) -> list:
        """Return this backend's known `VoiceInfo` list, optionally filtered
        to a language code: built-in voices first, then the user's own."""
        ...

    def get_languages(self) -> list:
        """`[(label, code), ...]` this backend can speak."""
        ...

    def cancel(self) -> None:
        """Cancel any in-flight generation."""
        ...

    # `.tbaw` hooks - see BackendHooksMixin for the defaults and docs.

    def engine_version(self) -> str:
        ...

    def cache_key_extra(self, config: dict) -> dict:
        ...

    def resolve_voice_file(self, name: str, project_dir: Optional[str] = None) -> Optional[str]:
        ...

    def collect_project_assets(self, voice_names, project_dir=None) -> tuple:
        ...

    def on_project_opened(self, project_dir: Optional[str], meta: dict) -> None:
        ...


@runtime_checkable
class SupportsVoiceMixing(Protocol):
    """Optional extension for backends whose voices are locally-loadable
    tensors that can be blended (`capabilities.supports_voice_mixing=True`).
    Not part of `TTSEngineBackend` itself - per the plan, mixing has no
    equivalent in a cloud TTS API or a differently-shaped local model, so it
    is fenced off as an opt-in capability instead of forced into the shared
    protocol."""

    async def mix_voices(self, v1_name: str, v2_name: str, ratio: float,
                          new_name: str, op: str = "mix"):
        ...

    async def preview_mix(self, tensor, voice_name: str, text: str, output_path: str, lang_code: str):
        ...
