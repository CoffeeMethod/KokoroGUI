"""`post.render_async` and `post.prewarm` (kokoro_gui/audio/post.py): renders
on a worker pool whose results reach the memo and the counters only through
`drain()`. No Qt, no audio device."""
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.audio import post

RATE = 8000
CONFIG = {"volume": 2.0, "apply_fx": False}


def _tone(path, seconds=0.25, amplitude=0.25):
    t = np.arange(int(RATE * seconds)) / RATE
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32), RATE)
    return str(path)


@pytest.fixture(autouse=True)
def _clean_pool():
    post.wait_idle()
    post.clear_render_cache()
    post.set_notifier(None)
    yield
    post.wait_idle()
    post.set_notifier(None)
    post.clear_render_cache()


def _wait_for(condition, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("timed out")


def test_render_async_fills_the_entry_render_would_have_made(tmp_path):
    path = _tone(tmp_path / "a.wav")
    got = []

    post.render_async(path, CONFIG, RATE, None, got.append)
    post.wait_idle()

    assert len(got) == 1 and got[0] is not None
    assert post.render(path, CONFIG, RATE) is got[0]
    raw, _rate = sf.read(path, dtype="float32")
    assert np.allclose(got[0], raw * 2.0, atol=1e-6)
    assert post.render_count(path) == 1


def test_render_async_answers_at_once_when_the_memo_has_it(tmp_path):
    path = _tone(tmp_path / "a.wav")
    held = post.render(path, CONFIG, RATE)
    got = []

    post.render_async(path, CONFIG, RATE, None, got.append)

    assert got == [held]


def test_concurrent_calls_for_one_key_render_once(tmp_path, monkeypatch):
    path = _tone(tmp_path / "a.wav")
    calls = []
    real = post._render_uncached

    def slow(*args):
        calls.append(threading.current_thread().name)
        time.sleep(0.05)
        return real(*args)

    monkeypatch.setattr(post, "_render_uncached", slow)
    got = []
    for _ in range(5):
        post.render_async(path, CONFIG, RATE, None, got.append)
    post.wait_idle()

    assert len(calls) == 1 and calls[0] != threading.current_thread().name
    assert len(got) == 5 and all(g is got[0] for g in got)
    assert post.render_count(path) == 1


def test_a_worker_moves_no_counter_until_drain(tmp_path, monkeypatch):
    path = _tone(tmp_path / "a.wav")
    release = threading.Event()
    real = post._render_uncached
    monkeypatch.setattr(post, "_render_uncached", lambda *a: release.wait(5) and real(*a))
    before = post.RENDERS
    got = []

    post.render_async(path, CONFIG, RATE, None, got.append)
    release.set()
    _wait_for(lambda: post._FINISHED)

    assert post.RENDERS == before and post.render_count(path) == 0
    assert post.cached_render(path, CONFIG, RATE) is None and got == []
    assert post.drain() == 1
    assert post.RENDERS == before + 1 and post.render_count(path) == 1
    assert len(got) == 1 and post.cached_render(path, CONFIG, RATE) is got[0]


def test_an_unreadable_file_answers_none_and_is_not_memoized(tmp_path):
    missing = str(tmp_path / "nope.wav")
    got = []

    post.render_async(missing, CONFIG, RATE, None, got.append)
    post.wait_idle()

    assert got == [None]
    assert post.render_count(missing) == 0


def test_the_notifier_fires_once_per_burst(tmp_path):
    paths = [_tone(tmp_path / f"{i}.wav") for i in range(4)]
    pings = []
    post.set_notifier(lambda: pings.append(1))

    for path in paths:
        post.render_async(path, CONFIG, RATE, None, lambda _a: None)
    _wait_for(lambda: len(post._FINISHED) == 4)

    assert pings == [1]
    post.drain()
    post.render_async(paths[0], {"volume": 3.0, "apply_fx": False}, RATE, None, lambda _a: None)
    _wait_for(lambda: post._FINISHED)
    assert pings == [1, 1]


def test_prewarm_fills_the_memo_and_skips_what_is_there(tmp_path):
    paths = [_tone(tmp_path / f"{i}.wav") for i in range(6)]
    held = post.render(paths[0], CONFIG, RATE)

    post.prewarm([(p, CONFIG, RATE, None) for p in paths])
    post.wait_idle()

    assert all(post.cached_render(p, CONFIG, RATE) is not None for p in paths)
    assert post.cached_render(paths[0], CONFIG, RATE) is held
    assert [post.render_count(p) for p in paths] == [1] * 6


def test_a_render_async_does_not_wait_behind_a_prewarm_backlog(tmp_path, monkeypatch):
    backlog = [_tone(tmp_path / f"b{i}.wav") for i in range(12)]
    wanted = _tone(tmp_path / "wanted.wav")
    started = []
    real = post._render_uncached

    def slow(path, *args):
        started.append(path)
        time.sleep(0.03)
        return real(path, *args)

    monkeypatch.setattr(post, "_render_uncached", slow)
    post.prewarm([(p, CONFIG, RATE, None) for p in backlog])
    post.render_async(wanted, CONFIG, RATE, None, lambda _a: None)
    post.wait_idle()

    assert wanted in started
    assert started.index(wanted) <= post.RENDER_WORKERS + 1


def test_a_second_prewarm_replaces_the_backlog(tmp_path, monkeypatch):
    first = [_tone(tmp_path / f"f{i}.wav") for i in range(10)]
    second = [_tone(tmp_path / f"s{i}.wav") for i in range(2)]
    real = post._render_uncached

    def slow(*args):
        time.sleep(0.03)
        return real(*args)

    monkeypatch.setattr(post, "_render_uncached", slow)
    post.prewarm([(p, CONFIG, RATE, None) for p in first])
    post.prewarm([(p, CONFIG, RATE, None) for p in second])
    post.wait_idle()

    assert all(post.cached_render(p, CONFIG, RATE) is not None for p in second)
    assert sum(post.cached_render(p, CONFIG, RATE) is not None for p in first) < len(first)


def test_clear_render_cache_drops_the_backlog(tmp_path, monkeypatch):
    paths = [_tone(tmp_path / f"{i}.wav") for i in range(8)]
    real = post._render_uncached

    def slow(*args):
        time.sleep(0.03)
        return real(*args)

    monkeypatch.setattr(post, "_render_uncached", slow)
    post.prewarm([(p, CONFIG, RATE, None) for p in paths])
    post.clear_render_cache()
    post.wait_idle()

    assert not post._BACKLOG
