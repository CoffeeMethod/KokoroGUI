"""Engines without widgets: per-engine settings, voice lists, a new
character's defaults and the set of resident adapters.

`QtTTSApp.engine_settings`, `get_all_voices`, `default_voice`,
`make_character`, `backend_for_character` and `backend_for` are thin calls
into this module; the app's `EngineSet` callback wires each new adapter's
signals.
"""
from __future__ import annotations

from typing import Callable, Optional, Union

from kokoro_gui.daw.models import Character
from kokoro_gui.engines import registry as engine_registry
from kokoro_gui.engines.base import per_engine_fields
from kokoro_gui.engines.missing import MissingBackend


def engine_settings(settings: dict, engine_id: str, memo: Optional[dict] = None) -> dict:
    """The per-engine settings of `engine_id` (grill EN5): the schema
    defaults of every field that isn't shared (`lang_code`, `num_threads`,
    the "Model" group, ...), overlaid with `settings["engines"][engine_id]`.
    A stored value for a key the schema no longer has is dropped. `memo`
    keeps the field list per engine id (the app's `_schema_memo`, which
    Options > Force refresh clears)."""
    fields = memo.get(("per_engine", engine_id)) if memo is not None else None
    if fields is None:
        fields = tuple(per_engine_fields(engine_registry.get_config_schema(engine_id)))
        if memo is not None:
            memo[("per_engine", engine_id)] = fields
    stored = (settings.get("engines") or {}).get(engine_id) or {}
    return {f.key: stored.get(f.key, f.default) for f in fields}


def model_fields(backend, memo: Optional[dict] = None) -> tuple:
    """`(key, default)` for each field in the backend's "Model" schema
    group, memoized per adapter class in `memo` (every schema is a static
    list)."""
    memo_key = ("model", type(backend), backend.id)
    fields = memo.get(memo_key) if memo is not None else None
    if fields is None:
        fields = tuple((f.key, f.default) for f in backend.get_config_schema() if f.group == "Model")
        if memo is not None:
            memo[memo_key] = fields
    return fields


def voice_names(backend, lang_code) -> list:
    """The voice names `backend` lists for `lang_code`: built-ins first,
    then the user's own, each once."""
    names = []
    for voice in backend.get_voices(lang_code):
        if voice.id not in names:
            names.append(voice.id)
    return names


def default_voice(backend, settings_for_engine: dict) -> Optional[str]:
    """The default voice of `backend`'s engine: the stored per-engine
    `voice` when the engine lists it, else the schema default, else its
    first voice. `settings_for_engine` is `engine_settings(...)` for it."""
    voices = voice_names(backend, settings_for_engine.get("lang_code"))
    stored = settings_for_engine.get("voice")
    if stored and stored in voices:
        return stored
    field = next((f for f in backend.get_config_schema() if f.key == "voice"), None)
    if field is not None and field.default:
        return field.default
    return voices[0] if voices else stored


def new_character(name: str, highlight_color: str, backend, settings_for_engine: dict) -> Character:
    """A new local character on `backend`'s engine with that engine's
    default voice (grill EN4). Loads no model."""
    voice = default_voice(backend, settings_for_engine)
    return Character.from_preset_dict(name, {"voice": voice} if voice else {},
                                      highlight_color=highlight_color, backend_id=backend.id)


class EngineSet:
    """Every engine a character uses stays resident (grill V3, "engine
    follows the character"): engine id -> adapter, built once on first use.
    Building one starts its worker thread but loads no model.

    `default_engine_id` is the engine a clip with no character uses: an id,
    or a callable read per call (the app's default follows its settings).
    `on_new_backend(backend)` runs once for each adapter right after it is
    built and added to `resident` (the app wires its signal bridge there).
    """

    def __init__(self, default_engine_id: Union[str, Callable[[], str]],
                 primary_engine_id: str = engine_registry.DEFAULT_ENGINE_ID,
                 on_new_backend: Optional[Callable] = None):
        self._default_engine_id = default_engine_id
        self.primary_engine_id = primary_engine_id
        self.on_new_backend = on_new_backend
        self.resident: dict = {}
        self._missing: dict = {}

    @property
    def default_engine_id(self) -> str:
        value = self._default_engine_id
        return value() if callable(value) else value

    def get(self, engine_id: Optional[str]):
        """The resident adapter for `engine_id` (no id means the primary
        engine), built once and kept, or None for an engine that isn't
        registered."""
        engine_id = engine_id or self.primary_engine_id
        backend = self.resident.get(engine_id)
        if backend is not None:
            return backend
        try:
            backend = engine_registry.get_engine(engine_id)
        except Exception:  # noqa: BLE001 - an unregistered id is a warning, not a crash
            return None
        self.resident[backend.id] = backend
        if self.on_new_backend is not None:
            self.on_new_backend(backend)
        return backend

    def missing(self, engine_id: str) -> MissingBackend:
        """One `MissingBackend` per engine id, kept (not in `resident`:
        Save's asset collection and the job/cancel loops only see real
        engines)."""
        backend = self._missing.get(engine_id)
        if backend is None:
            backend = MissingBackend(engine_id, engine_registry.unavailable_reason(engine_id))
            self._missing[engine_id] = backend
        return backend

    def for_character(self, character):
        """The adapter a character generates with: its `backend_id` (no id
        means the primary engine). An engine this install doesn't have gets
        a `MissingBackend` (grill EN6). With no character at all, the
        default engine."""
        if character is None:
            return self.get(self.default_engine_id) or next(iter(self.resident.values()))
        engine_id = character.backend_id or self.primary_engine_id
        return self.get(engine_id) or self.missing(engine_id)

    def for_clip(self, document, clip):
        """The adapter `clip` generates with: its character's engine."""
        if clip is None:
            return self.for_character(None)
        return self.for_character(document.get_character(clip.character_id))
