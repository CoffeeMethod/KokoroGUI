"""The app's log file, the stdout/stderr tee and the exception hooks.

`main.py` is the only caller. `setup_logging` points the root logger at
`<log_dir>/kokorogui.log` (a rotating file) and copies everything the app
prints into the same file, so the many `print(...)` calls need no edit.
`install_excepthooks` logs an uncaught exception from any thread and hands
it to a callback, which the GUI uses to open the crash dialog.

No Qt imports here. Tests call these functions with a `tmp_path` and restore
`sys.stdout`, `sys.stderr` and both excepthooks afterwards, so they never
leak into the next test.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
import traceback
from typing import Callable

LOG_SUBDIR = "logs"
LOG_FILENAME = "kokorogui.log"
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"

# The tee's own loggers write to the file handler only. The console already
# gets the text through the tee's pass-through, so a second copy from the
# root's stream handler would print every line twice.
STDOUT_LOGGER = "stdout"
STDERR_LOGGER = "stderr"

_state: dict = {"path": None, "file_handler": None, "stream_handler": None, "tees": {}, "root_level": None}


def resolve_log_path(cache_dir: str) -> str:
    """The log file `setup_logging` is writing, or where it would go under
    `cache_dir` when logging isn't set up (tests, the screenshot script)."""
    return _state["path"] or os.path.abspath(os.path.join(cache_dir, LOG_SUBDIR, LOG_FILENAME))


class _Tee:
    """A text stream that passes every write to the real stream and logs
    each completed line. `real` can be None (no console under pythonw)."""

    def __init__(self, real, logger: logging.Logger, level: int):
        self._real = real
        self._logger = logger
        self._level = level
        self._pending = ""
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def real(self):
        return self._real

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        written = len(text)
        if self._real is not None:
            try:
                written = self._real.write(text)
            except Exception:
                pass
        # A logging error prints to stderr, which is this tee: don't recurse.
        if not getattr(self._local, "busy", False):
            self._local.busy = True
            try:
                self._log_lines(text)
            finally:
                self._local.busy = False
        return written

    def _log_lines(self, text: str) -> None:
        with self._lock:
            self._pending += text
            *lines, self._pending = self._pending.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if line.strip():
                self._logger.log(self._level, "%s", line)

    def flush(self):
        if self._real is not None:
            try:
                self._real.flush()
            except Exception:
                pass

    def close_pending(self) -> None:
        """Logs a last line that never got its newline."""
        with self._lock:
            rest, self._pending = self._pending, ""
        if rest.strip():
            self._logger.log(self._level, "%s", rest)

    def __getattr__(self, name):
        # isatty, fileno, encoding and the rest come from the real stream.
        if self._real is None:
            raise AttributeError(name)
        return getattr(self._real, name)


def _unwrap(stream):
    return stream.real if isinstance(stream, _Tee) else stream


def teardown_logging() -> None:
    """Undoes `setup_logging`: removes and closes its handlers, unwraps the
    tee'd streams and restores the root level. Safe when nothing is set up."""
    root = logging.getLogger()
    for key in ("file_handler", "stream_handler"):
        handler = _state[key]
        if handler is not None:
            root.removeHandler(handler)
            logging.getLogger(STDOUT_LOGGER).removeHandler(handler)
            logging.getLogger(STDERR_LOGGER).removeHandler(handler)
            handler.close()
            _state[key] = None
    for name in ("stdout", "stderr"):
        tee = _state["tees"].pop(name, None)
        if tee is not None:
            tee.close_pending()
        stream = getattr(sys, name)
        if isinstance(stream, _Tee):
            setattr(sys, name, stream.real)
    if _state["root_level"] is not None:
        root.setLevel(_state["root_level"])
        _state["root_level"] = None
    _state["path"] = None


def setup_logging(log_dir: str) -> str:
    """Writes the root logger (INFO) to `<log_dir>/kokorogui.log`, rotating at
    2 MB with 5 backups, plus a stderr handler for developers, and tees
    `sys.stdout` and `sys.stderr` into the file. Returns the log path.
    Calling it again replaces the earlier setup instead of stacking handlers."""
    from kokoro_gui.engine.paths import ensure_private_dir

    teardown_logging()
    log_dir = ensure_private_dir(os.path.abspath(log_dir), fallback=False)
    path = os.path.join(log_dir, LOG_FILENAME)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    _state["root_level"] = root.level
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    _state["file_handler"] = file_handler

    # The stream handler holds the real stderr, never the tee, or its own
    # output would loop back into the log.
    real_stderr = _unwrap(sys.stderr)
    if real_stderr is not None:
        stream_handler = logging.StreamHandler(real_stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        _state["stream_handler"] = stream_handler

    for name, logger_name, level in (
        ("stdout", STDOUT_LOGGER, logging.INFO),
        ("stderr", STDERR_LOGGER, logging.WARNING),
    ):
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(level)
        logger.addHandler(file_handler)
        tee = _Tee(_unwrap(getattr(sys, name)), logger, level)
        _state["tees"][name] = tee
        setattr(sys, name, tee)

    _state["path"] = path
    return path


def format_exception_text(exc_type, exc, tb) -> str:
    return "".join(traceback.format_exception(exc_type, exc, tb))


def install_excepthooks(on_exception: Callable) -> Callable[[], None]:
    """Hooks `sys.excepthook` and `threading.excepthook`. Each uncaught
    exception is logged at ERROR with its traceback, then passed to
    `on_exception(exc_type, exc, tb)`. `KeyboardInterrupt` (and `SystemExit`
    in a thread) go to the default hook and never reach `on_exception`.
    A failing callback is logged and swallowed. Returns a function that puts
    the previous hooks back."""
    log = logging.getLogger("kokorogui.crash")
    previous_sys = sys.excepthook
    previous_thread = threading.excepthook

    def report(exc_type, exc, tb, where: str) -> None:
        log.error("Uncaught exception in %s", where, exc_info=(exc_type, exc, tb))
        try:
            on_exception(exc_type, exc, tb)
        except Exception:
            log.exception("The crash handler failed")

    def sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_sys(exc_type, exc, tb)
            return
        report(exc_type, exc, tb, threading.current_thread().name)

    def thread_hook(args):
        if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            previous_thread(args)
            return
        name = args.thread.name if args.thread is not None else "unknown thread"
        report(args.exc_type, args.exc_value, args.exc_traceback, name)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook

    def uninstall() -> None:
        if sys.excepthook is sys_hook:
            sys.excepthook = previous_sys
        if threading.excepthook is thread_hook:
            threading.excepthook = previous_thread

    return uninstall
