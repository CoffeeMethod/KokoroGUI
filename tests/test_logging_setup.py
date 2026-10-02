"""kokoro_gui/logging_setup.py: the rotating log file, the stdout/stderr tee
and the exception hooks. Every test restores the streams and hooks it
touched, or later tests (and pytest's own exception handling) break."""
import logging
import sys
import threading

import pytest

from kokoro_gui import logging_setup


@pytest.fixture
def log_dir(tmp_path):
    saved = (sys.stdout, sys.stderr, sys.excepthook, threading.excepthook)
    yield tmp_path / "logs"
    logging_setup.teardown_logging()
    sys.stdout, sys.stderr, sys.excepthook, threading.excepthook = saved


def _read(path):
    for handler in logging.getLogger().handlers:
        handler.flush()
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_log_file_is_created_under_the_given_dir(log_dir):
    path = logging_setup.setup_logging(str(log_dir))
    assert path == str(log_dir / "kokorogui.log")
    assert (log_dir / "kokorogui.log").exists()


def test_a_logging_error_lands_in_the_file_with_the_thread_name(log_dir):
    path = logging_setup.setup_logging(str(log_dir))
    logging.getLogger("kokorogui.test").error("disk on fire")
    text = _read(path)
    assert "ERROR MainThread kokorogui.test: disk on fire" in text


def test_print_and_stderr_writes_land_in_the_file_and_still_reach_the_stream(log_dir, capsys):
    path = logging_setup.setup_logging(str(log_dir))
    print("hello from print")
    print("a warning", file=sys.stderr)
    text = _read(path)
    assert "INFO MainThread stdout: hello from print" in text
    assert "WARNING MainThread stderr: a warning" in text
    out, err = capsys.readouterr()
    assert "hello from print" in out
    assert "a warning" in err


def test_a_partial_line_is_logged_once_its_newline_arrives(log_dir):
    path = logging_setup.setup_logging(str(log_dir))
    sys.stdout.write("half")
    assert "half" not in _read(path)
    sys.stdout.write(" a line\n")
    assert "stdout: half a line" in _read(path)


def test_a_stream_handler_error_does_not_recurse_through_the_tee(log_dir):
    logging_setup.setup_logging(str(log_dir))
    handler = logging.getLogger().handlers[-1]
    original = handler.emit

    def broken(record):
        try:
            raise OSError("stream closed")
        except OSError:
            handler.handleError(record)

    handler.emit = broken
    try:
        logging.getLogger("x").error("boom")  # would loop forever if the tee re-entered itself
    finally:
        handler.emit = original


def test_setup_twice_does_not_stack_handlers_or_tees(log_dir):
    logging_setup.setup_logging(str(log_dir))
    handlers = len(logging.getLogger().handlers)
    path = logging_setup.setup_logging(str(log_dir))
    assert len(logging.getLogger().handlers) == handlers
    print("once")
    assert _read(path).count("stdout: once") == 1
    assert not isinstance(sys.stdout.real, logging_setup._Tee)


def test_teardown_restores_streams_and_root_level(log_dir):
    before = (sys.stdout, sys.stderr, logging.getLogger().level, list(logging.getLogger().handlers))
    logging_setup.setup_logging(str(log_dir))
    logging_setup.teardown_logging()
    after = (sys.stdout, sys.stderr, logging.getLogger().level, list(logging.getLogger().handlers))
    assert after == before


def test_the_file_rotates(log_dir, monkeypatch):
    monkeypatch.setattr(logging_setup, "MAX_BYTES", 400)
    path = logging_setup.setup_logging(str(log_dir))
    log = logging.getLogger("rotate")
    for i in range(40):
        log.error("line %d %s", i, "x" * 40)
    assert (log_dir / "kokorogui.log.1").exists()
    assert (log_dir / "kokorogui.log").stat().st_size <= 400 + 200
    assert path.endswith("kokorogui.log")


def test_resolve_log_path_falls_back_to_the_cache_dir_when_logging_is_off(tmp_path):
    logging_setup.teardown_logging()
    expected = tmp_path / "cache" / "logs" / "kokorogui.log"
    assert logging_setup.resolve_log_path(str(tmp_path / "cache")) == str(expected)


def test_a_thread_that_raises_calls_on_exception_once_and_is_logged(log_dir):
    path = logging_setup.setup_logging(str(log_dir))
    seen = []
    logging_setup.install_excepthooks(lambda t, e, tb: seen.append((t, e)))

    def work():
        raise ValueError("worker failed")

    thread = threading.Thread(target=work, name="worker-1")
    thread.start()
    thread.join()

    assert len(seen) == 1
    assert seen[0][0] is ValueError
    text = _read(path)
    assert "ERROR worker-1 kokorogui.crash: Uncaught exception in worker-1" in text
    assert "ValueError: worker failed" in text


def test_sys_excepthook_logs_and_calls_on_exception(log_dir):
    path = logging_setup.setup_logging(str(log_dir))
    seen = []
    logging_setup.install_excepthooks(lambda t, e, tb: seen.append(e))
    try:
        raise RuntimeError("main failed")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    assert [str(e) for e in seen] == ["main failed"]
    assert "RuntimeError: main failed" in _read(path)


def test_keyboard_interrupt_goes_to_the_old_hook_not_the_callback(log_dir):
    logging_setup.setup_logging(str(log_dir))
    old = []
    sys.excepthook = lambda *a: old.append(a[0])
    seen = []
    logging_setup.install_excepthooks(lambda t, e, tb: seen.append(e))
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert seen == []
    assert old == [KeyboardInterrupt]


def test_a_failing_callback_is_swallowed_and_logged(log_dir):
    path = logging_setup.setup_logging(str(log_dir))

    def bad(*_):
        raise OSError("dialog died")

    logging_setup.install_excepthooks(bad)
    try:
        raise RuntimeError("first")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())  # must not raise
    assert "The crash handler failed" in _read(path)


def test_uninstall_puts_the_previous_hooks_back(log_dir):
    before = (sys.excepthook, threading.excepthook)
    uninstall = logging_setup.install_excepthooks(lambda *a: None)
    assert (sys.excepthook, threading.excepthook) != before
    uninstall()
    assert (sys.excepthook, threading.excepthook) == before
