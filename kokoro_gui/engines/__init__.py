"""Engine backend abstraction (PLAN_qt_and_engine_abstraction.md workstream 1).

Importing this package loads no backend. The registry calls `load_engines`
on its first query: it imports the built-in "audio8", "dummy" and "kokoro"
backends, then every `kokorogui.engines` entry point (a plugin engine),
each on its own, so one that fails (the `kokoro` package isn't installed, a
broken plugin) is marked unavailable instead of taking the others down, and
a project that uses it opens with that engine named "(not installed)"
(grill EN6).

A plugin's entry point names a module that registers its adapter on import,
or the adapter class itself (registered here under its `id`). Adding an
engine: CONTRIBUTING.md, "Adding an engine".
"""
import importlib
import importlib.metadata

from kokoro_gui.engines import base, registry

# (engine id, display name, module): the display name is here as well as on
# the adapter, for when the module can't be imported.
BUILTIN_ENGINES = (
    ("audio8", "Audio8 TTS (voice cloning)", "kokoro_gui.engines.audio8_tts"),
    ("dummy", "Dummy (offline test tone)", "kokoro_gui.engines.dummy"),
    ("kokoro", "Kokoro (local)", "kokoro_gui.engines.kokoro"),
)

ENTRY_POINT_GROUP = "kokorogui.engines"


def _load_builtin(engine_id: str, display_name: str, module: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as e:
        registry.mark_unavailable(engine_id, display_name, f"{type(e).__name__}: {e}")


def _entry_points() -> list:
    try:
        return list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001 - broken metadata must not stop the app
        return []


def load_entry_point(entry_point) -> None:
    """Loads one plugin engine. The entry point's name is the engine id
    reported when it fails."""
    try:
        loaded = entry_point.load()
    except Exception as e:  # noqa: BLE001 - a plugin's failure is its own
        registry.mark_unavailable(entry_point.name, entry_point.name, f"{type(e).__name__}: {e}")
        return
    engine_id = getattr(loaded, "id", None)
    if isinstance(loaded, type) and engine_id and not registry.is_registered(engine_id):
        registry.register_engine(engine_id, loaded, display_name=getattr(loaded, "display_name", engine_id))


def load_engines() -> None:
    """Imports every built-in backend and then every plugin (each registers
    itself, or is marked unavailable)."""
    for engine_id, display_name, module in BUILTIN_ENGINES:
        _load_builtin(engine_id, display_name, module)
    for entry_point in _entry_points():
        load_entry_point(entry_point)


__all__ = ["base", "registry", "BUILTIN_ENGINES", "ENTRY_POINT_GROUP", "load_engines", "load_entry_point"]
