"""Backend registry: `register_engine`/`get_engine`/`list_engines`.

A "factory" here is any callable that returns a `TTSEngineBackend` instance
- for the built-in `kokoro.py` adapter that's the `KokoroBackendAdapter`
class itself, called with the already-constructed `KokoroEngine` it wraps
(`get_engine("kokoro", engine=some_kokoro_engine)`), since the adapter is
composition over an existing engine instance, not a from-scratch factory.

Imports nothing from the backends, so `daw/` and `engine/` can import it
for `DEFAULT_ENGINE_ID` without loading a model stack. The built-in engines
(and the `kokorogui.engines` entry points) load on the first query that
needs them (`_ensure_loaded`), not at import.
"""
from __future__ import annotations

from typing import Callable, Dict

# The engine a character or config with no engine id means. Projects saved
# before characters carried an engine are Kokoro projects.
DEFAULT_ENGINE_ID = "kokoro"

_registry: Dict[str, Callable[..., object]] = {}
_display_names: Dict[str, str] = {}
# Engines that failed to load (a missing package, a broken plugin): id ->
# the reason, for the "(not installed)" labels (grill EN6).
_unavailable: Dict[str, str] = {}
_loaded = False


def _ensure_loaded() -> None:
    """Imports the built-in backends and the entry-point plugins once
    (`kokoro_gui/engines/__init__.py`). Each registers itself or is marked
    unavailable."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    from kokoro_gui.engines import load_engines

    load_engines()


def register_engine(engine_id: str, factory: Callable[..., object], display_name: str = None) -> None:
    """Register `factory` under `engine_id`. Re-registering the same id
    overwrites the previous factory (useful for tests that register a fake
    backend under a throwaway id)."""
    _registry[engine_id] = factory
    _unavailable.pop(engine_id, None)
    if display_name is not None:
        _display_names[engine_id] = display_name


def is_registered(engine_id: str) -> bool:
    """Whether `engine_id` is registered now (no loading)."""
    return engine_id in _registry


def mark_unavailable(engine_id: str, display_name: str, reason: str) -> None:
    """Records an engine that couldn't be loaded, so a project that uses it
    can still name it. An engine registered later (a plugin loading after
    all) clears this."""
    if engine_id in _registry:
        return
    _unavailable[engine_id] = reason
    _display_names.setdefault(engine_id, display_name)


def unavailable_reason(engine_id: str):
    """Why `engine_id` couldn't be loaded, or None when it's available or
    was never heard of."""
    _ensure_loaded()
    return _unavailable.get(engine_id)


def get_engine(engine_id: str, *args, **kwargs):
    """Construct and return the backend registered under `engine_id`."""
    _ensure_loaded()
    if engine_id not in _registry:
        raise KeyError(
            f"No engine backend registered under {engine_id!r}. "
            f"Known engines: {sorted(_registry)}"
        )
    return _registry[engine_id](*args, **kwargs)


def get_factory(engine_id: str):
    """The factory (usually the adapter class) registered under
    `engine_id`, or None."""
    _ensure_loaded()
    return _registry.get(engine_id)


def get_capabilities(engine_id: str):
    """The `EngineCapabilities` the factory registered under `engine_id`
    declares as a class attribute, read without constructing a backend, or
    None for an unknown id or a factory that declares none."""
    _ensure_loaded()
    return getattr(_registry.get(engine_id), "capabilities", None)


def get_config_schema(engine_id: str) -> list:
    """The `ConfigField` list the factory registered under `engine_id`
    declares, read without constructing a backend (every adapter's
    `get_config_schema` is a classmethod), or `[]` for an unknown id."""
    _ensure_loaded()
    factory = _registry.get(engine_id)
    getter = getattr(factory, "get_config_schema", None)
    if not callable(getter):
        return []
    try:
        return list(getter())
    except TypeError:  # an instance method on a factory that isn't a class
        return []


def package_version(engine_id: str):
    """The version string the factory registered under `engine_id` reports
    without a backend (`package_version()` classmethod, e.g. the installed
    `kokoro` package's), or None."""
    _ensure_loaded()
    getter = getattr(_registry.get(engine_id), "package_version", None)
    return getter() if callable(getter) else None


def list_engines() -> list:
    """Return the sorted list of registered (available) engine ids."""
    _ensure_loaded()
    return sorted(_registry)


def list_all_engines() -> list:
    """Every engine id this install knows of: the available ones and the
    ones that failed to load, sorted."""
    _ensure_loaded()
    return sorted(set(_registry) | set(_unavailable))


def get_display_name(engine_id: str) -> str:
    """Human-readable name for `engine_id`, falling back to the id itself if
    none was given at registration time."""
    _ensure_loaded()
    return _display_names.get(engine_id, engine_id)


def unregister_engine(engine_id: str) -> None:
    """Remove a registered engine id (mainly for test teardown)."""
    _registry.pop(engine_id, None)
    _unavailable.pop(engine_id, None)
    _display_names.pop(engine_id, None)
