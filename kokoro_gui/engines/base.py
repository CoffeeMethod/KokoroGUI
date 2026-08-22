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
capabilities, its config schema, its voice list, and cancellation.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class VoiceInfo:
    """One selectable voice, as reported by a backend's `get_voices()`."""
    id: str
    display_name: str
    lang_code: Optional[str] = None
    is_custom: bool = False


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
