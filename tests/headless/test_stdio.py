import io
import sys

import pytest

from kokoro_gui.headless import stdio


@pytest.fixture
def streams(monkeypatch):
    """Call it inside the test: pytest's capture swaps sys.stdout again after
    fixture setup, so a patch made here would be lost."""
    def install():
        out = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="\n")
        err = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        return out, err

    yield install
    # a failed test must not leave the guard installed
    stdio._state = None


def test_print_goes_to_stderr_while_guarded(streams):
    out, err = streams()
    with stdio.guard():
        print("stray engine line")
        assert sys.stdout is err
    assert "stray engine line" in err.getvalue()
    assert out.buffer.getvalue() == b""


def test_kept_stream_is_the_real_stdout(streams):
    out, _ = streams()
    with stdio.guard() as g:
        assert g.stdout is out
        g.stdout.write('{"ok": true}\n')
        g.stdout.flush()
    assert out.buffer.getvalue() == b'{"ok": true}\n'


def test_binary_buffer_for_stdout_dash(streams):
    out, _ = streams()
    with stdio.guard() as g:
        assert g.binary is out.buffer
        g.binary.write(b"RIFF")
    assert out.buffer.getvalue() == b"RIFF"


def test_binary_is_none_without_a_buffer(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    try:
        with stdio.guard() as g:
            assert g.binary is None
    finally:
        stdio._state = None


def test_restore_puts_stdout_back(streams):
    out, _ = streams()
    with stdio.guard():
        pass
    assert sys.stdout is out
    assert stdio.active() is None


def test_restore_without_with_block(streams):
    out, _ = streams()
    g = stdio.guard()
    assert sys.stdout is not out
    g.restore()
    g.restore()
    assert sys.stdout is out


def test_second_call_is_idempotent(streams):
    out, err = streams()
    first = stdio.guard()
    second = stdio.guard()
    assert second.stdout is out
    assert sys.stdout is err
    second.restore()  # not the owner: nothing happens
    assert sys.stdout is err
    first.restore()
    assert sys.stdout is out


def test_nested_with_restores_only_at_the_outer_exit(streams):
    out, err = streams()
    with stdio.guard():
        with stdio.guard() as inner:
            assert inner.stdout is out
        assert sys.stdout is err
    assert sys.stdout is out


def test_guard_after_restore_works_again(streams):
    out, err = streams()
    stdio.guard().restore()
    with stdio.guard() as g:
        assert g.stdout is out
        assert sys.stdout is err
    assert sys.stdout is out


def test_restore_leaves_stdout_alone_if_someone_else_replaced_it(streams):
    streams()
    other = io.StringIO()
    g = stdio.guard()
    sys.stdout = other
    g.restore()
    assert sys.stdout is other
