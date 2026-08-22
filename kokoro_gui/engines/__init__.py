"""Engine backend abstraction (PLAN_qt_and_engine_abstraction.md workstream 1).

Importing this package registers the built-in "kokoro" backend as a side
effect (`from kokoro_gui.engines.kokoro import KokoroBackendAdapter` below),
mirroring how `kokoro_gui/ui/__init__.py` collects the Tk tab mixins.
"""
from kokoro_gui.engines import base, registry
from kokoro_gui.engines.kokoro import KokoroBackendAdapter

__all__ = ["base", "registry", "KokoroBackendAdapter"]
