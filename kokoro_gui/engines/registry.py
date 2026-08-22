"""Backend registry: `register_engine`/`get_engine`/`list_engines`.

A "factory" here is any callable that returns a `TTSEngineBackend` instance
- for the built-in `kokoro.py` adapter that's the `KokoroBackendAdapter`
class itself, called with the already-constructed `KokoroEngine` it wraps
(`get_engine("kokoro", engine=some_kokoro_engine)`), since the adapter is
composition over an existing engine instance, not a from-scratch factory.
"""
from __future__ import annotations

from typing import Callable, Dict

_registry: Dict[str, Callable[..., object]] = {}
_display_names: Dict[str, str] = {}


def register_engine(engine_id: str, factory: Callable[..., object], display_name: str = None) -> None:
    """Register `factory` under `engine_id`. Re-registering the same id
    overwrites the previous factory (useful for tests that register a fake
    backend under a throwaway id)."""
    _registry[engine_id] = factory
    if display_name is not None:
        _display_names[engine_id] = display_name


def get_engine(engine_id: str, *args, **kwargs):
    """Construct and return the backend registered under `engine_id`."""
    if engine_id not in _registry:
        raise KeyError(
            f"No engine backend registered under {engine_id!r}. "
            f"Known engines: {sorted(_registry)}"
        )
    return _registry[engine_id](*args, **kwargs)


def list_engines() -> list:
    """Return the sorted list of registered engine ids."""
    return sorted(_registry)


def unregister_engine(engine_id: str) -> None:
    """Remove a registered engine id (mainly for test teardown)."""
    _registry.pop(engine_id, None)
    _display_names.pop(engine_id, None)
