"""Process-wide state every engine shares, with no model behind it: the
storage directories, the playback module, and the asyncio worker thread
each engine runs its coroutines on.

These used to be `kokoro_engine`'s module globals, so importing any backend
imported `kokoro` too. `kokoro_engine` still answers for the old names (its
module `__getattr__` reads them from here), but code and tests read and
patch them here: `runtime.CACHE_DIR`, qualified, at call time, so a
monkeypatch (the `isolated_dirs` fixture) or the private-dir swap in
`prepare_storage` reaches every reader.
"""
import asyncio
import threading

import playback  # noqa: F401 - read qualified (`runtime.playback`); tests patch it

from kokoro_gui.engine.paths import ensure_private_dir

CUSTOM_VOICES_DIR = "custom_voices"
CACHE_DIR = "cache"
STATS_FILE = "generation_stats.json"  # per-engine generation history, see kokoro_gui/engine/stats.py


def prepare_storage() -> None:
    """Creates the voices and cache dirs private (0o700 on POSIX, see
    kokoro_gui/engine/paths.py). A cache dir owned by another user is
    swapped for a per-user one, and everything under it (project dirs
    included) follows, since every reader looks `CACHE_DIR` up here at call
    time. The voices dir stays put: moving a user's voices would lose them.
    Idempotent; the app calls it at startup and `KokoroEngine` on init."""
    global CACHE_DIR
    ensure_private_dir(CUSTOM_VOICES_DIR, fallback=False)
    CACHE_DIR = ensure_private_dir(CACHE_DIR)


class AsyncLoopThread(threading.Thread):
    """A daemon thread running its own asyncio loop; `run_coro` schedules a
    coroutine on it and returns a `concurrent.futures.Future`."""

    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.running = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()

    def run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
