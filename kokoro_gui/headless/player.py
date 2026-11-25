"""Playing files one after the other, and the lock that keeps two detached
players apart.

`Player` is a daemon thread over a queue. It goes through
`runtime.playback.play` and `play_range`, looked up per call, so the tests'
mock catches them. `stop(group)` drops one call's files and leaves the others.

A detached player (grill PG29) holds `audio_output/headless/player.lock`, an OS
lock that goes away with the process. Stopping one is a request, the file
`player.stop` beside it, which the holder polls: no process id is stored and
nothing is killed.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from kokoro_gui.engine import runtime
from kokoro_gui.qt.project import ProjectLock, ProjectLockedError

from . import paths
from .errors import HeadlessError

log = logging.getLogger(__name__)

PLAYBACK_ENV = "KOKOROGUI_PLAYBACK"
MODES = ("off", "after", "stream")
LOCK_NAME = "player.lock"
STOP_NAME = "player.stop"
POLL_S = 0.02
STOP_TIMEOUT_S = 5.0


def playback_reason() -> Optional[str]:
    """Why nothing can play: "disabled" (`KOKOROGUI_PLAYBACK=off`),
    "no_audio_device", or None."""
    if os.environ.get(PLAYBACK_ENV, "").strip().lower() == "off":
        return "disabled"
    return None if runtime.playback.AVAILABLE else "no_audio_device"


def check_mode(play) -> str:
    if play not in MODES:
        raise HeadlessError("bad_arguments", f"play must be one of {', '.join(MODES)}", valid=MODES)
    return play


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@dataclass(eq=False)
class Item:
    """One queued file. `start_s` and `end_s` play a range of it, `gap_s` is
    silence before it, `delete` removes the file when it is done or dropped."""

    path: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    gap_s: float = 0.0
    group: object = None
    delete: bool = False
    played: bool = False
    error: Optional[Exception] = None
    dropped: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class Player:
    """One file at a time, in the order they came. Thread-safe. The thread
    starts with the first file."""

    def __init__(self):
        self._cond = threading.Condition()
        self._queue: deque = deque()
        self._current: Optional[Item] = None
        self._started_at = 0.0
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def enqueue(self, path: str, start_s: Optional[float] = None, end_s: Optional[float] = None, *,
                interrupt: bool = False, gap_s: float = 0.0, group=None, delete: bool = False) -> Item:
        """`interrupt` stops what plays and drops what waits, first."""
        item = Item(path, start_s, end_s, max(0.0, float(gap_s)), group, delete)
        if interrupt:
            self.stop()
        with self._cond:
            if self._closed:
                raise HeadlessError("interrupted", "the player is closed")
            self._queue.append(item)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="headless-player", daemon=True)
                self._thread.start()
            self._cond.notify_all()
        return item

    def stop(self, group=None) -> None:
        """Drops what waits and stops what plays: everything, or one group's.
        Returns once the file that played has ended."""
        with self._cond:
            dropped = [i for i in self._queue if group is None or i.group is group]
            self._queue = deque(i for i in self._queue if i not in dropped)
            current = self._current
            if current is not None and group is not None and current.group is not group:
                current = None
            if current is not None:
                current.dropped = True
            self._cond.notify_all()
        for item in dropped:
            item.dropped = True
            self._finish(item)
        if current is None or threading.current_thread() is self._thread:
            return
        # Again until it ends: a stop that lands before the stream opens does
        # nothing.
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while not current.done.is_set() and time.monotonic() < deadline:
            try:
                runtime.playback.stop()
            except Exception:  # noqa: BLE001
                log.exception("stopping playback failed")
            current.done.wait(POLL_S)

    def status(self) -> dict:
        """`elapsed_s` is wall time since the file started: the playback
        module has no position."""
        with self._cond:
            current = self._current
            return {
                "playing": current.path if current is not None else None,
                "elapsed_s": round(time.monotonic() - self._started_at, 3) if current is not None else 0.0,
                "queued": len(self._queue),
            }

    def wait(self, group=None, cancel: Optional[Callable[[], bool]] = None) -> bool:
        """Until everything (or one group's) has played. A true `cancel()`
        stops it and returns False."""
        while True:
            with self._cond:
                items = ([self._current] if self._current is not None else []) + list(self._queue)
                if not any(group is None or i.group is group for i in items):
                    return True
                self._cond.wait(POLL_S)
            if cancel is not None and cancel():
                self.stop(group)
                return False

    def close(self) -> None:
        """Stops everything and joins the thread. Safe to call again."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self.stop()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(STOP_TIMEOUT_S)

    # -- the thread ------------------------------------------------------------

    def _finish(self, item: Item) -> None:
        if item.delete:
            _remove(item.path)
        item.done.set()

    def _pause(self, item: Item, seconds: float) -> None:
        """Waits, unless `item` is dropped meanwhile."""
        deadline = time.monotonic() + seconds
        with self._cond:
            while not item.dropped:
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                self._cond.wait(left)

    def _play(self, item: Item) -> None:
        self._pause(item, item.gap_s)
        if item.dropped:
            return
        self._started_at = time.monotonic()
        if item.start_s is None and item.end_s is None:
            runtime.playback.play(item.path, blocking=True)
        else:
            import soundfile as sf

            start = max(0.0, item.start_s or 0.0)
            length = sf.info(item.path).duration
            end = length if item.end_s is None else min(item.end_s, length)
            runtime.playback.play_range(item.path, start, end)
            self._pause(item, max(0.0, end - start))
        item.played = not item.dropped

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._closed:
                    self._cond.wait()
                if not self._queue:
                    return
                item = self._current = self._queue.popleft()
                self._started_at = time.monotonic()
            try:
                self._play(item)
            except Exception as e:  # noqa: BLE001 - the next file may play
                log.exception("playing %s failed", item.path)
                item.error = e
            with self._cond:
                self._current = None
                self._finish(item)
                self._cond.notify_all()


class Playback:
    """One call's use of a player: nothing at all when the mode is "off" or
    nothing can play. Without a `player` it makes one, waits for it and closes
    it, since a daemon thread ends with the process."""

    def __init__(self, mode: str, player: Optional[Player], *, wait: bool = False, interrupt: bool = False):
        self.mode = check_mode(mode)
        self.reason = playback_reason() if mode != "off" else None
        self.active = mode != "off" and self.reason is None
        self._own = player is None
        self._player = player
        self._wait = wait or self._own
        self._interrupt = interrupt
        self._items: list = []

    @property
    def streaming(self) -> bool:
        return self.active and self.mode == "stream"

    def add(self, path: str, *, gap_s: float = 0.0, delete: bool = False) -> None:
        if not self.active:
            if delete:
                _remove(path)
            return
        if self._player is None:
            self._player = Player()
        self._items.append(self._player.enqueue(
            path, gap_s=gap_s, delete=delete, group=self, interrupt=self._interrupt and not self._items))

    def finish(self, cancel: Optional[Callable[[], bool]] = None) -> tuple:
        """`(played, reason)`. "played" means queued when this doesn't wait."""
        if not self._items:
            return False, self.reason
        if not self._wait:
            return True, None
        try:
            finished = self._player.wait(self, cancel)
        finally:
            self._close()
        if not finished:
            return False, "interrupted"
        failed = next((i for i in self._items if i.error is not None), None)
        if failed is not None:
            return False, f"playback failed: {failed.error}"
        return all(i.played for i in self._items), None if all(i.played for i in self._items) else "interrupted"

    def abort(self) -> None:
        """The call failed: its files don't play."""
        if self._player is not None:
            self._player.stop(self)
        self._close()

    def _close(self) -> None:
        if self._own and self._player is not None:
            self._player.close()


# -- detached players ------------------------------------------------------------


class _Lock(ProjectLock):
    def __init__(self, root: str):
        super().__init__(root)
        self.path = os.path.join(root, LOCK_NAME)


class PlayerLock:
    """Held by a detached player for as long as it runs."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or paths.renders_root()
        self._lock = _Lock(self.root)

    def acquire(self, *, interrupt: bool = False, cancel: Optional[Callable[[], bool]] = None) -> "PlayerLock":
        """Waits for the holder. `interrupt` asks it to stop first."""
        asked = False
        while True:
            try:
                self._lock.acquire()
                break
            except ProjectLockedError:
                pass
            if cancel is not None and cancel():
                raise HeadlessError("interrupted", "cancelled while another player was running")
            if interrupt and not asked:
                _ask_to_stop(self.root)
                asked = True
            time.sleep(POLL_S)
        _remove(os.path.join(self.root, STOP_NAME))  # meant for the one before
        return self

    def stop_requested(self) -> bool:
        return os.path.exists(os.path.join(self.root, STOP_NAME))

    def release(self) -> None:
        if self._lock.held:
            _remove(os.path.join(self.root, STOP_NAME))
        self._lock.release()

    def __enter__(self) -> "PlayerLock":
        return self if self._lock.held else self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def _ask_to_stop(root: str) -> None:
    with open(os.path.join(root, STOP_NAME), "w", encoding="utf-8"):
        pass


def player_running(root: Optional[str] = None) -> bool:
    root = root or paths.renders_root()
    if not os.path.isfile(os.path.join(root, LOCK_NAME)):
        return False
    lock = _Lock(root)
    try:
        lock.acquire()
    except ProjectLockedError:
        return True
    lock.release()
    return False


def stop_detached(root: Optional[str] = None, timeout_s: float = STOP_TIMEOUT_S) -> bool:
    """Asks the detached player to stop and waits for it. False when there is
    none, or when it doesn't answer in time."""
    root = root or paths.renders_root()
    if not player_running(root):
        return False
    _ask_to_stop(root)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not player_running(root):
            return True
        time.sleep(POLL_S)
    _remove(os.path.join(root, STOP_NAME))
    return False
