"""Keep stdout for the protocol or the result.

Some engine paths `print()`. On MCP stdio a stray line corrupts the protocol,
and on the CLI it breaks `--json` and `-o -`. `guard()` runs first in both
entry points: it keeps the real stdout and points `sys.stdout` at stderr.

    with stdio.guard() as g:
        g.stdout.write(json_line)     # the real stdout
        g.binary.write(wav_bytes)     # its buffer, for `-o -` (None if absent)

`guard()` is idempotent. A second call, while one is active, returns a handle
to the same state, and only the handle that installed the guard restores
`sys.stdout`. Output from C code that writes to file descriptor 1 directly is
not redirected.
"""
from __future__ import annotations

import sys
import threading

_lock = threading.Lock()
_state = None  # the active _State, or None


class _State:
    def __init__(self, real, replaced):
        self.real = real
        self.replaced = replaced  # what sys.stdout was set to


class Guard:
    """Handle to the kept stream. A context manager."""

    def __init__(self, state: _State, owner: bool):
        self._state = state
        self._owner = owner

    @property
    def stdout(self):
        """The real stdout, kept for the protocol or the result."""
        return self._state.real

    @property
    def binary(self):
        """The real stdout's binary buffer, or None when it has none."""
        return getattr(self._state.real, "buffer", None)

    def restore(self) -> None:
        """Put `sys.stdout` back. A no-op for a handle that didn't install the
        guard, and for one already restored."""
        global _state
        if not self._owner:
            return
        with _lock:
            if _state is not self._state:
                return
            try:
                self._state.replaced.flush()
            except (OSError, ValueError):
                pass
            if sys.stdout is self._state.replaced:
                sys.stdout = self._state.real
            _state = None
        self._owner = False

    def __enter__(self) -> "Guard":
        return self

    def __exit__(self, *exc) -> None:
        self.restore()


def guard() -> Guard:
    """Point `sys.stdout` at stderr and keep the real stream on the handle."""
    global _state
    with _lock:
        if _state is not None:
            return Guard(_state, owner=False)
        real = sys.stdout
        try:
            real.flush()
        except (OSError, ValueError):
            pass
        sys.stdout = sys.stderr
        _state = _State(real, sys.stderr)
        return Guard(_state, owner=True)


def active() -> Guard | None:
    """A non-owning handle to the active guard, if any."""
    with _lock:
        return Guard(_state, owner=False) if _state is not None else None
