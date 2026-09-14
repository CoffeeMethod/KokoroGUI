"""Engine backend abstraction (PLAN_qt_and_engine_abstraction.md workstream 1).

Importing this package registers the built-in "kokoro", "dummy", and
"audio8" backends as a side effect (the submodule imports below).
"""
from kokoro_gui.engines import base, registry
from kokoro_gui.engines.audio8_tts import Audio8BackendAdapter
from kokoro_gui.engines.dummy import DummyBackendAdapter
from kokoro_gui.engines.kokoro import KokoroBackendAdapter

__all__ = ["base", "registry", "KokoroBackendAdapter", "DummyBackendAdapter", "Audio8BackendAdapter"]
