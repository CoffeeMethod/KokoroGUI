"""Cross-thread callback marshalling for the Qt frontend.

`KokoroEngine` calls `self.on_status`/`self.on_progress`/`self.on_finish` from
its own background `AsyncLoopThread` (see kokoro_engine.py's `AsyncLoopThread`
and kokoro_gui/engine/conversion.py's call sites). Qt's answer to that is a
`QObject` living on the main thread whose signals are emitted from the worker
thread: PySide6 detects the emitting thread differs from the receiving
QObject's thread and automatically queues the connected slot call onto that
thread's event loop (a `Qt.QueuedConnection`), with no `after()`-style
boilerplate needed - this works for any Python thread that emits into a
QObject with a running event loop, not just a `QThread`.

The same trick applies to the `concurrent.futures.Future.add_done_callback(...)`
pattern `preview_conversion`/mixing use: any Qt widget/dock is itself a
`QObject` that was constructed on the main thread, so defining a small
`Signal` directly on that dock/window class and emitting it from inside the
done-callback (which runs on the worker thread) is enough - no dedicated
bridge object needed for those one-off cases. See docks/mixing_dock.py's
`previewFinished`/`mixFinished` and app.py's `previewFinished` for examples.
"""
from PySide6.QtCore import QObject, Signal


class EngineSignalBridge(QObject):
    """One instance per resident engine (app.py's `_add_backend`): each
    backend's callbacks reach the app through its own bridge, so several
    engines can report at once."""

    # func(msg: str, is_error: bool) - kokoro_engine.py:72
    status = Signal(str, bool)
    # func(percentage: float, time_elapsed: float, eta: str, detail_text: str) - kokoro_engine.py:73
    progress = Signal(float, float, str, str)
    # func() - kokoro_engine.py:74
    finished = Signal()


def wire_engine(engine, bridge: EngineSignalBridge) -> None:
    """Point `engine`'s callback attributes at `bridge`'s signals - the Qt
    equivalent of a plain `engine.on_progress = self.on_engine_progress`
    wiring, just emitting a signal instead of calling a bound method directly."""
    engine.on_status = bridge.status.emit
    engine.on_progress = bridge.progress.emit
    engine.on_finish = bridge.finished.emit
