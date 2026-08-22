"""Engine backend abstraction (PLAN_qt_and_engine_abstraction.md workstream 1).

Importing this package registers the built-in "kokoro" and "dummy" backends
as a side effect (the `kokoro`/`dummy` submodule imports below).
"""
from kokoro_gui.engines import base, registry
from kokoro_gui.engines.dummy import DummyBackendAdapter
from kokoro_gui.engines.kokoro import KokoroBackendAdapter

__all__ = ["base", "registry", "KokoroBackendAdapter", "DummyBackendAdapter"]
