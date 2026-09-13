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
from dataclasses import dataclass
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
    fields with a fixed, known set of options (e.g. split-pattern presets,
    output format). Fields whose options are only known at runtime by the GUI
    (e.g. "voice", "lang_code" - today's Kokoro voice catalog is GUI display
    data, not engine data; see kokoro.py's `get_config_schema` docstring)
    leave `choices=None` and the GUI resolves them dynamically.
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


# Choice presets for schema fields whose *meaning* isn't actually
# model-specific, just conventionally offered by more than one backend: how
# raw input text gets split into chunks before parallel processing, and
# which container formats get written to disk. Backends are free to ignore
# these or offer their own instead - they're shared defaults, not part of
# the Protocol.
COMMON_SPLIT_PATTERN_CHOICES = [
    ("Natural (Newlines)", r"\n+"),
    ("Paragraphs (Double Newline)", r"\n\n+"),
    ("Sentences (.!?)", r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"),
]
COMMON_OUTPUT_FORMAT_CHOICES = [("wav", "wav"), ("flac", "flac"), ("mp3", "mp3"), ("ogg", "ogg")]


@dataclass(frozen=True)
class VoiceInfo:
    """One selectable voice, as reported by a backend's `get_voices()`."""
    id: str
    display_name: str
    lang_code: Optional[str] = None
    is_custom: bool = False


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
        names it). Default: no files, empty meta."""
        return [], {}

    def on_project_opened(self, project_dir: Optional[str], meta: dict) -> None:
        """Called after a project is opened (or created) with this backend's
        manifest `meta`. Must not load a model: record what to do and do it
        on the first generate. The default remembers the dir for listings."""
        self.project_dir = project_dir


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
        to a language code."""
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
