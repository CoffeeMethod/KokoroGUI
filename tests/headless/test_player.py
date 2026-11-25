"""headless/player.py: order, gaps, ranges, stop and interrupt, why nothing
plays, and the lock and stop file of a detached player. No audio device."""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.headless import paths
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.player import (
    PLAYBACK_ENV, Playback, Player, PlayerLock, check_mode, playback_reason, player_running, stop_detached,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Device:
    """Stands in for `playback`: a blocking play ends when the test says, or
    when it is stopped."""

    def __init__(self, mock):
        self.mock = mock
        self.played = []
        self.started = threading.Event()
        self.ended = threading.Event()
        self.hold = False
        mock.play.side_effect = self._play
        mock.stop.side_effect = self.ended.set

    def _play(self, path, blocking=False):
        assert blocking
        self.ended.clear()
        self.played.append(path)
        self.started.set()
        if self.hold:
            assert self.ended.wait(10)


@pytest.fixture
def device(headless_env, monkeypatch):
    monkeypatch.delenv(PLAYBACK_ENV, raising=False)
    return Device(headless_env)


@pytest.fixture
def player():
    made = Player()
    yield made
    made.close()
    assert made._thread is None or not made._thread.is_alive()


def test_files_play_in_order(device, player):
    items = [player.enqueue(name) for name in "abc"]
    assert player.wait()
    assert device.played == ["a", "b", "c"] and all(i.played for i in items)
    assert player.status() == {"playing": None, "elapsed_s": 0.0, "queued": 0}


def test_status_while_playing(device, player):
    device.hold = True
    player.enqueue("a")
    player.enqueue("b")
    assert device.started.wait(5)
    status = player.status()
    assert (status["playing"], status["queued"]) == ("a", 1) and status["elapsed_s"] >= 0
    player.stop()


def test_stop_ends_the_file_and_drops_the_rest(device, player):
    device.hold = True
    items = [player.enqueue(name) for name in "abc"]
    assert device.started.wait(5)
    player.stop()
    assert device.played == ["a"] and all(i.done.is_set() and i.dropped and not i.played for i in items)
    device.hold = False
    assert player.enqueue("d").done.wait(5) and device.played == ["a", "d"]


def test_a_stop_before_the_stream_opens_still_stops_it(device, player):
    """`sd.stop` does nothing until `sd.play` has run, so one call isn't enough."""
    opening, opened = threading.Event(), threading.Event()

    def play(path, blocking):
        opening.set()
        opened.wait(5)
        device.ended.clear()
        assert device.ended.wait(5)

    device.mock.play.side_effect = play
    item = player.enqueue("a")
    assert opening.wait(5)
    threading.Timer(0.1, opened.set).start()
    player.stop()
    assert item.done.is_set() and device.mock.stop.call_count > 1


def test_interrupt_plays_the_new_file_next(device, player):
    device.hold = True
    player.enqueue("a")
    player.enqueue("b")
    assert device.started.wait(5)
    device.hold = False
    assert player.enqueue("c", interrupt=True).done.wait(5)
    assert device.played == ["a", "c"]


def test_stopping_one_group_leaves_the_others(device, player):
    device.hold = True
    mine, theirs = object(), object()
    player.enqueue("a", group=theirs)
    dropped = player.enqueue("b", group=mine)
    kept = player.enqueue("c", group=theirs)
    assert device.started.wait(5)
    player.stop(mine)
    assert dropped.dropped and not kept.dropped and device.mock.stop.call_count == 0
    device.hold = False
    device.ended.set()
    assert player.wait(theirs) and device.played == ["a", "c"]


def test_a_gap_is_waited_out_before_the_file(device, player):
    start = time.monotonic()
    item = player.enqueue("a", gap_s=0.2)
    assert item.done.wait(5) and time.monotonic() - start >= 0.19 and item.played


def test_stop_ends_a_gap(device, player):
    item = player.enqueue("a", gap_s=60)
    time.sleep(0.05)
    start = time.monotonic()
    player.stop()
    assert item.done.is_set() and time.monotonic() - start < 5 and device.played == []


def test_a_range_goes_through_play_range_and_takes_its_time(device, player, tmp_path):
    path = str(tmp_path / "a.wav")
    sf.write(path, np.zeros(24000), 24000)
    start = time.monotonic()
    item = player.enqueue(path, 0.1, 0.3)
    assert item.done.wait(5) and item.played and time.monotonic() - start >= 0.19
    device.mock.play_range.assert_called_once_with(path, 0.1, 0.3)
    player.enqueue(path, 0.95).done.wait(5)
    assert device.mock.play_range.call_args.args == (path, 0.95, 1.0)


def test_a_file_that_fails_does_not_stop_the_next(device, player):
    device.mock.play.side_effect = [RuntimeError("no device"), None]
    first, second = player.enqueue("a"), player.enqueue("b")
    assert player.wait()
    assert isinstance(first.error, RuntimeError) and not first.played and second.played


def test_a_scratch_file_is_deleted_once_played_or_dropped(device, player, tmp_path):
    paths_ = [tmp_path / f"{n}.wav" for n in range(3)]
    for path in paths_:
        path.write_bytes(b"x")
    device.hold = True
    player.enqueue(str(paths_[0]), delete=True)
    player.enqueue(str(paths_[1]), delete=True)
    player.enqueue(str(paths_[2]))
    assert device.started.wait(5)
    player.stop()
    assert [p.exists() for p in paths_] == [False, False, True]


def test_wait_stops_when_cancelled(device, player):
    device.hold = True
    item = player.enqueue("a")
    assert device.started.wait(5)
    assert player.wait(cancel=lambda: True) is False
    assert item.done.is_set() and not item.played


def test_a_closed_player_takes_nothing(device):
    player = Player()
    player.enqueue("a")
    player.close()
    player.close()
    with pytest.raises(HeadlessError) as caught:
        player.enqueue("b")
    assert caught.value.code == "interrupted"


def test_why_nothing_plays(device, monkeypatch):
    assert playback_reason() is None
    device.mock.AVAILABLE = False
    assert playback_reason() == "no_audio_device"
    monkeypatch.setenv(PLAYBACK_ENV, " OFF ")
    assert playback_reason() == "disabled"
    monkeypatch.setenv(PLAYBACK_ENV, "on")
    device.mock.AVAILABLE = True
    assert playback_reason() is None


def test_modes():
    for mode in ("off", "after", "stream"):
        assert check_mode(mode) == mode
    for mode in ("", "now", None, 1):
        with pytest.raises(HeadlessError) as caught:
            check_mode(mode)
        assert (caught.value.code, caught.value.valid) == ("bad_arguments", ["off", "after", "stream"])


# -- one call's playback -----------------------------------------------------------------


def test_off_makes_no_player(device):
    playback = Playback("off", None)
    playback.add("a")
    assert playback.finish() == (False, None) and device.played == []


@pytest.mark.parametrize("reason", ["disabled", "no_audio_device"])
def test_nothing_can_play(device, monkeypatch, tmp_path, reason):
    if reason == "disabled":
        monkeypatch.setenv(PLAYBACK_ENV, "off")
    device.mock.AVAILABLE = reason == "disabled"
    playback = Playback("stream", None)
    scratch = tmp_path / "s.wav"
    scratch.write_bytes(b"x")
    playback.add(str(scratch), delete=True)
    assert playback.finish() == (False, reason) and device.played == [] and not scratch.exists()
    assert not playback.streaming


def test_without_a_player_it_waits(device):
    playback = Playback("after", None)
    playback.add("a")
    assert playback.finish() == (True, None) and device.played == ["a"]
    assert not playback._player._thread.is_alive()


def test_a_given_player_is_not_waited_for_unless_asked(device, player):
    device.hold = True
    playback = Playback("after", player)
    playback.add("a")
    assert playback.finish() == (True, None)
    waiting = Playback("after", player, wait=True)
    waiting.add("b")
    assert device.started.wait(5)
    device.hold = False
    device.ended.set()
    assert waiting.finish() == (True, None) and device.played == ["a", "b"]


def test_a_failed_file_and_a_stopped_one_are_reported(device, player):
    device.mock.play.side_effect = RuntimeError("no device")
    playback = Playback("after", player, wait=True)
    playback.add("a")
    assert playback.finish() == (False, "playback failed: no device")
    device.mock.play.side_effect = None
    assert Playback("after", player, wait=True).finish() == (False, None)
    stopped = Playback("after", player, wait=True)
    stopped.add("a", gap_s=60)
    assert stopped.finish(cancel=lambda: True) == (False, "interrupted")


def test_abort_drops_only_its_own(device, player):
    theirs = player.enqueue("a", gap_s=0.1)
    playback = Playback("stream", player)
    playback.add("b")
    playback.abort()
    assert player.wait() and theirs.played and device.played == ["a"]


# -- detached players ---------------------------------------------------------------------


def test_the_lock_and_the_stop_file():
    root = paths.renders_root()
    assert not player_running() and stop_detached() is False and not os.path.exists(root)
    with PlayerLock() as lock:
        assert player_running() and not lock.stop_requested()
        assert stop_detached(timeout_s=0.1) is False  # it doesn't answer
        assert not lock.stop_requested()
    assert not player_running() and sorted(os.listdir(root)) == ["player.lock"]


def test_a_stop_meant_for_a_player_that_died_is_forgotten():
    PlayerLock().acquire().release()
    with open(os.path.join(paths.renders_root(), "player.stop"), "w"):
        pass
    with PlayerLock() as lock:
        assert not lock.stop_requested()


def test_a_second_player_waits_for_the_first():
    first = PlayerLock().acquire()
    got = threading.Event()
    second = PlayerLock()
    thread = threading.Thread(target=lambda: (second.acquire(), got.set()))
    thread.start()
    assert not got.wait(0.2) and not first.stop_requested()
    first.release()
    assert got.wait(5)
    thread.join()
    second.release()


def test_interrupt_asks_the_first_to_stop():
    first = PlayerLock().acquire()

    def hold():
        while not first.stop_requested():
            time.sleep(0.01)
        first.release()

    thread = threading.Thread(target=hold)
    thread.start()
    second = PlayerLock().acquire(interrupt=True)
    thread.join()
    assert not second.stop_requested()
    second.release()


def test_a_cancel_ends_the_wait():
    with PlayerLock():
        with pytest.raises(HeadlessError) as caught:
            PlayerLock().acquire(cancel=lambda: True)
    assert caught.value.code == "interrupted"


CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from kokoro_gui.headless.player import PlayerLock
with PlayerLock() as lock:
    print("start", time.time(), flush=True)
    deadline = time.time() + float(sys.argv[2])
    while time.time() < deadline and not lock.stop_requested():
        time.sleep(0.01)
    print("end", time.time(), flush=True)
"""


def spawn(tmp_path, seconds):
    return subprocess.Popen([sys.executable, "-c", CHILD, ROOT, str(seconds)], cwd=tmp_path,
                            stdout=subprocess.PIPE, text=True)


def test_two_processes_take_turns(tmp_path):
    children = [spawn(tmp_path, 0.4) for _ in range(2)]
    spans = []
    for child in children:
        out, _err = child.communicate(timeout=120)
        assert child.returncode == 0
        spans.append([float(line.split()[1]) for line in out.splitlines()])
    (a_start, a_end), (b_start, b_end) = sorted(spans)
    assert a_end - a_start >= 0.39 and b_start >= a_end - 0.01 and b_end > b_start


def test_stop_ends_a_player_in_another_process(tmp_path):
    child = spawn(tmp_path, 60)
    try:
        assert child.stdout.readline().startswith("start")
        assert player_running() and stop_detached(timeout_s=20) is True
        child.communicate(timeout=20)
        assert child.returncode == 0 and not player_running()
    finally:
        child.kill()


def test_a_player_that_dies_frees_the_lock(tmp_path):
    child = spawn(tmp_path, 60)
    assert child.stdout.readline().startswith("start")
    child.kill()
    child.wait(20)
    PlayerLock().acquire().release()
