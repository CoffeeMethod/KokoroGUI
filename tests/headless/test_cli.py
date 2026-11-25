"""headless/cli.py through `main(argv)`, and through a real process for stdout,
stdin and `--detach`. Also the script and settings files it reads."""
import io
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.engine import text_extraction
from kokoro_gui.headless import cli, stdio
from kokoro_gui.headless.errors import HeadlessError
from kokoro_gui.headless.inputs import default_engine, load_settings, read_script
from kokoro_gui.headless.player import PLAYBACK_ENV, PlayerLock, player_running
from kokoro_gui.headless.render import render_script
from kokoro_gui.qt import project as project_io

from .support import SCRIPT, files_under, working_dirs

pytestmark = pytest.mark.usefixtures("offline")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def playable(monkeypatch):
    monkeypatch.delenv(PLAYBACK_ENV, raising=False)
    yield
    assert stdio.active() is None


@pytest.fixture
def run(capsys):
    """`(exit code, stdout, stderr)` of one command."""
    def call(*argv):
        code = cli.main([str(a) for a in argv])
        out, err = capsys.readouterr()
        return code, out, err

    return call


@pytest.fixture
def stdin(monkeypatch):
    def give(data):
        data = data.encode("utf-8") if isinstance(data, str) else data
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"))

    return give


def events(err):
    return [json.loads(line) for line in err.splitlines()]


# -- listings -------------------------------------------------------------------------


def test_engines(fake, run):
    code, out, err = run("engines")
    assert code == 0 and "fake\tFake" in out.splitlines() and "kokoro\tKokoro (local)\tdefault" in out and err == ""
    code, out, _err = run("engines", "--json")
    rows = {e["id"]: e for e in json.loads(out)["engines"]}
    assert code == 0 and rows["fake"] == {"id": "fake", "name": "Fake", "installed": True, "reason": None, "default": False}


def test_an_engine_that_is_not_installed_is_listed_with_its_reason(run):
    from kokoro_gui.engines import registry

    registry.mark_unavailable("gone", "Gone", "no module named gone")
    try:
        _code, out, _err = run("engines")
        assert "gone\tGone\tnot installed (no module named gone)" in out
    finally:
        registry.unregister_engine("gone")


def test_voices(fake, run):
    assert run("voices", "--engine", "fake") == (0, "v1\ta\t\tdefault\nv2\ta\nw1\tb\n", "")
    _code, out, _err = run("--json", "voices", "--engine", "fake", "--lang", "b")
    assert json.loads(out) == {"voices": [{"id": "w1", "name": "w1", "lang_code": "b", "custom": False, "default": False}]}


def test_devices(run, headless_env):
    code, out, _err = run("devices", "--json")
    found = json.loads(out)
    assert code == 0 and found["devices"][:2] == ["auto", "cpu"] and found["playback"] is None
    headless_env.AVAILABLE = False
    assert "playback: no_audio_device" in run("devices")[1]


# -- say ----------------------------------------------------------------------------------


def test_say_prints_the_path_and_nothing_else(fake, run):
    code, out, err = run("say", "Hello there.", "--engine", "fake", "--voice", "v2", "--quiet")
    assert (code, err) == (0, "") and sf.info(out.strip()).frames == 4800
    assert fake.spoken == [("Hello there.", "v2", 1.0, "a")]


def test_say_as_json(fake, run):
    code, out, _err = run("say", "Hello.", "--engine", "fake", "--json", "--speed", "2", "--pitch", "-1")
    data = json.loads(out)
    assert code == 0 and out.count("\n") == 1 and data["engine"] == "fake" and len(data["words"]) == 1
    assert os.path.isfile(data["path"])


def test_say_from_stdin_with_no_file(fake, run, stdin, tmp_path):
    stdin("﻿From stdin.\n")
    assert run("say", "--engine", "fake", "--no-save", "--quiet")[:2] == (0, "0.200 s\n")
    assert fake.spoken[0][0] == "From stdin." and not (tmp_path / "audio_output").exists()


def test_stdin_that_is_not_text(fake, run, stdin):
    stdin(b"\xff\xfe\x00")
    assert run("say", "--engine", "fake")[0] == 2


def test_say_to_stdout_is_wav_bytes_and_nothing_else(fake, capsysbinary, tmp_path):
    assert cli.main(["say", "Hello there.", "--engine", "fake", "-o", "-"]) == 0
    out, err = capsysbinary.readouterr()
    samples, rate = sf.read(io.BytesIO(out))
    assert out[:4] == b"RIFF" and (len(samples), rate) == (4800, 24000)
    assert len(out) == 44 + 2 * 4800
    assert all(json.loads(line) for line in err.splitlines()) and not (tmp_path / "audio_output").exists()


def test_a_print_in_an_engine_does_not_reach_stdout(fake, capsysbinary):
    fake.speak = lambda text: print("noise")
    assert cli.main(["say", "Hello.", "--engine", "fake", "-o", "-", "--quiet"]) == 0
    out, err = capsysbinary.readouterr()
    assert len(out) == 44 + 2 * 2400 and b"noise" in err


@pytest.mark.parametrize("extra", [["--json"], ["--no-save"]])
def test_stdout_audio_with_something_else(fake, run, extra):
    assert run("say", "x", "--engine", "fake", "-o", "-", *extra)[0] == 2
    assert fake.spoken == []


def test_plain_text_keeps_its_stars(fake, run):
    run("say", "a *b*", "--engine", "fake", "--plain")
    run("say", "a *b*", "--engine", "fake")
    assert [s[0] for s in fake.spoken] == ["a *b*", "a b"]


def test_text_that_looks_like_an_option(fake, run):
    assert run("say", "--engine", "fake", "--", "--detach now")[0] == 0
    assert fake.spoken[0][0] == "--detach now"


# -- progress ---------------------------------------------------------------------------------


def test_progress_is_json_lines_on_stderr(fake, run, tmp_path):
    (tmp_path / "s.txt").write_text(SCRIPT)
    _code, out, err = run("render", "s.txt", "--engine", "fake")
    rows = events(err)
    assert {"event": "progress", "stage": "generate", "done": 3, "total": 3} in rows
    assert {r["event"] for r in rows} == {"progress", "status", "warning"}
    assert len(out.splitlines()) == 2
    assert run("render", "s.txt", "--engine", "fake", "--quiet")[2] == ""


def test_a_closed_stderr_does_not_stop_the_command(fake, run, monkeypatch):
    real = cli._write

    def write(stream, text):
        if stream is sys.stderr:
            raise BrokenPipeError
        real(stream, text)

    monkeypatch.setattr(cli, "_write", write)
    assert run("say", "Hello.", "--engine", "fake")[0] == 0


def test_a_console_that_cannot_show_a_character():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    cli._write(stream, "a中")
    assert stream.buffer.getvalue() == b"a\\u4e2d"


# -- render -------------------------------------------------------------------------------------


def test_render_a_file_with_a_cast(fake, run, tmp_path):
    (tmp_path / "book.txt").write_text(SCRIPT, encoding="utf-8")
    (tmp_path / "cast.json").write_text(json.dumps({"Sam": "v1", "Ann": {"voice": "v1", "speed": 2}}))
    code, out, _err = run("render", "book.txt", "--engine", "fake", "--cast-file", "cast.json", "--cast", "sam = v2",
                          "--title", "Book", "-o", "out", "--channels", "2", "--json")
    data = json.loads(out)
    assert code == 0 and data["channels"] == 2 and files_under(tmp_path / "out") == ["Book.tbaw", "Book.wav"]
    assert [s[1] for s in fake.spoken] == ["v1", "v2", "v1"] and fake.spoken[2][2] == 2


def test_render_markdown_keeps_its_tags(fake, run, tmp_path):
    (tmp_path / "book.md").write_text("# Title\n\n[Sam]: Some **bold** text.\n", encoding="utf-8")
    run("render", "book.md", "--engine", "fake", "--cast", "Sam=v2")
    assert [s[:2] for s in fake.spoken] == [("Title.", "v1"), ("Some bold text.", "v2")]


def test_render_stdin_then_update(fake, run, stdin):
    stdin(SCRIPT)
    _code, out, _err = run("render", "--engine", "fake")
    project, mixdown = out.splitlines()
    stdin(SCRIPT.replace("Who", "And who"))
    code, out, _err = run("render", "-", "--engine", "fake", "--update", project, "--json")
    data = json.loads(out)
    assert (code, data["regenerated"], data["project_path"], data["mixdown_path"]) == (0, 1, project, mixdown)


def test_discard_session(fake, run, stdin):
    result = render_script(SCRIPT, engine="fake")
    (folder,) = working_dirs()
    project_io.write_session(folder, {**project_io.read_session(folder), "dirty": True})
    stdin(SCRIPT)
    code, out, _err = run("render", "--engine", "fake", "--update", result.project_path, "--json")
    assert code == 2 and json.loads(out)["error"] == "unsaved_session"
    stdin(SCRIPT)
    assert run("render", "--engine", "fake", "--update", result.project_path, "--discard-session")[0] == 0


@pytest.mark.parametrize("cast", ["Sam", "=v1", "Sam=", " = "])
def test_a_cast_that_is_not_speaker_equals_voice(fake, run, stdin, cast):
    stdin(SCRIPT)
    assert run("render", "--engine", "fake", "--cast", cast)[0] == 2 and fake.spoken == []


@pytest.mark.parametrize("content", ["[1]", "{", '"x"', '{"Sam": 4}', "[" * 100000],
                         ids=["list", "broken", "text", "number", "deep"])
def test_a_bad_cast_file(fake, run, stdin, tmp_path, content):
    (tmp_path / "cast.json").write_text(content)
    stdin(SCRIPT)
    assert run("render", "--engine", "fake", "--cast-file", "cast.json")[0] == 2 and fake.spoken == []


def test_a_cast_file_that_is_too_big_or_not_there(fake, run, stdin, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MAX_CAST_FILE", 10)
    (tmp_path / "cast.json").write_text(json.dumps({"Sam": "v1"}))
    for name in ("cast.json", "missing.json"):
        stdin(SCRIPT)
        assert run("render", "--engine", "fake", "--cast-file", name)[0] == 2


# -- errors and exit codes ----------------------------------------------------------------------------


@pytest.mark.parametrize("argv, code, error", [
    ([], 2, "bad_arguments"),
    (["sing"], 2, "bad_arguments"),
    (["say", "x", "--bogus"], 2, "bad_arguments"),
    (["say", "x", "--speed", "fast"], 2, "bad_arguments"),
    (["say", "x", "--engine", "fake", "--speed", "99"], 2, "bad_arguments"),
    (["say", "x", "--engine", "fake", "--play", "later"], 2, "bad_arguments"),
    (["say", "x", "--engine", "fake", "-o", "../x.wav"], 2, "out_not_allowed"),
    (["say", "x", "--engine", "fake", "--device", "tpu"], 2, "bad_arguments"),
    (["say", "x", "--engine", "fake", "--settings", "none.json"], 2, "bad_arguments"),
    (["render", "none.txt", "--engine", "fake"], 2, "bad_arguments"),
    (["render", "x.pdf", "--engine", "fake"], 2, "bad_arguments"),
    (["render", "x.txt", "--channels", "3"], 2, "bad_arguments"),
    (["say", "x", "--engine", "nope"], 3, "engine_not_installed"),
    (["voices", "--engine", "nope"], 3, "engine_not_installed"),
    (["say", "x", "--engine", "fake", "--voice", "nope"], 4, "unknown_voice"),
])
def test_exit_codes(fake, run, argv, code, error):
    got, out, err = run(*argv, "--json") if argv else run("--json")
    assert got == code and json.loads(out)["error"] == error and set(json.loads(out)) == {"error", "message", "valid"}
    got, out, err = run(*argv)
    assert got == code and out == "" and err.splitlines()[-1 if "valid: " not in err else -2].startswith(f"{error}: ")


def test_a_model_that_is_not_downloaded_is_5(fake, run):
    def load():
        from huggingface_hub.errors import LocalEntryNotFoundError

        raise LocalEntryNotFoundError("weights")

    fake.load = load
    assert run("say", "x", "--engine", "fake")[0] == 5


def test_a_held_lock_is_6_and_a_failure_is_7(fake, run, stdin):
    result = render_script(SCRIPT, engine="fake")
    lock = project_io.ProjectLock(working_dirs()[0]).acquire()
    try:
        stdin(SCRIPT)
        assert run("render", "--engine", "fake", "--update", result.project_path)[0] == 6
    finally:
        lock.release()
    fake.speak = lambda text: 1 / 0
    assert run("say", "Something else.", "--engine", "fake")[0] == 7


def test_ctrl_c_is_130(fake, run, monkeypatch):
    def stop(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.listing, "list_engines", stop)
    assert run("engines")[0] == 130


def test_the_valid_names_are_listed(fake, run):
    assert "valid: v1, v2, w1" in run("say", "x", "--engine", "fake", "--voice", "nope")[2]


def test_help_goes_to_stdout_and_exits_0(run):
    code, out, err = run("--help")
    assert code == 0 and out.startswith("usage: kokorogui") and err == ""
    code, out, _err = run("say", "--help")
    assert code == 0 and "--voice" in out and "--detached" not in out


def test_an_abbreviated_option_is_refused(fake, run):
    assert run("say", "x", "--engine", "fake", "--deta")[0] == 2


# -- settings -------------------------------------------------------------------------------------------


def test_settings_give_the_lexicon_and_the_engine(fake, run, tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({
        "default_engine": "fake", "lexicon": {"door": "gate"}, "speed": 3,
        "engines": {"fake": {"voice": "w1", "lang_code": "b", "num_threads": "many", "junk": 1}}}))
    assert run("say", "The door.", "--settings", "c.json", "--quiet")[0] == 0
    assert fake.spoken == [("The gate.", "w1", 1.0, "b")]


def test_config_qt_json_is_not_read_by_itself(fake, run, tmp_path):
    (tmp_path / "config_qt.json").write_text(json.dumps({"lexicon": {"door": "gate"}, "default_engine": "fake"}))
    run("say", "The door.", "--engine", "fake")
    assert fake.spoken[0][0] == "The door."


@pytest.mark.parametrize("content", ["", "[]", "{", '{"engines": 4}', '{"engines": {"fake": "x"}, "lexicon": 4}',
                                     '{"engines": {"fake": {"voice": 4, "lang_code": []}}, "default_engine": ["fake"]}'])
def test_damaged_settings_fall_back(fake, tmp_path, content):
    path = tmp_path / "c.json"
    path.write_text(content)
    settings = load_settings(str(path))
    assert settings["engines"]["fake"] == {"num_threads": 1}  # the app's own default
    assert default_engine(settings) == "kokoro"


# -- script files -----------------------------------------------------------------------------------------


def script_error(path, **kwargs):
    with pytest.raises(HeadlessError) as caught:
        read_script(path, **kwargs)
    assert caught.value.code == "bad_arguments"
    return caught.value


def test_a_script_file(tmp_path):
    (tmp_path / "a.TXT").write_text("﻿Hi *you*.", encoding="utf-8")
    assert read_script("a.TXT").endswith("Hi *you*.")
    (tmp_path / "a.md").write_text("Hi *you*.")
    assert read_script(tmp_path / "a.md") == "Hi you."


@pytest.mark.parametrize("name", ["a.pdf", "a", "a.txt.exe", "none.txt", "dir.txt"])
def test_not_a_script_file(tmp_path, name):
    (tmp_path / "dir.txt").mkdir()
    for other in ("a.pdf", "a", "a.txt.exe"):
        (tmp_path / other).write_text("x")
    script_error(name)


def test_a_script_over_the_limits_or_not_utf8(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_bytes(b"\xff\xfe")
    assert "UTF-8" in script_error("a.txt").message
    (tmp_path / "a.txt").write_text("x" * 100)
    monkeypatch.setattr(text_extraction, "MAX_TEXT_CHARS", 50)
    assert "limit" in script_error("a.txt").message
    monkeypatch.setattr(text_extraction, "MAX_BOOK_FILE_BYTES", 10)
    assert "limit" in script_error("a.txt").message


@pytest.mark.parametrize("path", ["../s.txt", "//host/share/s.txt", "\\\\host\\share\\s.txt", "nul.txt", "a:b.txt", "", 4])
def test_a_guarded_script_path(tmp_path, monkeypatch, path):
    (tmp_path / "s.txt").write_text("secret")
    inner = tmp_path / "inner"
    inner.mkdir()
    monkeypatch.chdir(inner)
    script_error(path, guarded=True)
    (inner / "s.txt").write_text("mine")
    assert read_script("s.txt", guarded=True) == "mine"


# -- play and stop --------------------------------------------------------------------------------------------


@pytest.fixture
def tone(tmp_path):
    path = tmp_path / "t.wav"
    sf.write(path, np.zeros(2400), 24000)
    return str(path)


def test_play(tone, run, headless_env):
    assert run("play", tone) == (0, "played\n", "")
    headless_env.play.assert_called_once_with(tone, blocking=True)
    assert json.loads(run("play", tone, "--start", "0.02", "--json")[1]) == {"played": True, "reason": None}
    assert headless_env.play_range.call_args.args == (tone, 0.02, 0.1)


def test_play_when_nothing_can(tone, run, monkeypatch):
    monkeypatch.setenv(PLAYBACK_ENV, "off")
    assert run("play", tone)[:2] == (0, "not played: disabled\n")


@pytest.mark.parametrize("extra", [["--start", "-1"], ["--start", "5"], ["--end", "0"], ["--start", "nan"]])
def test_play_a_bad_range(tone, run, extra):
    assert run("play", tone, *extra)[0] == 2


def test_play_something_that_is_not_audio(tmp_path, run):
    (tmp_path / "x.wav").write_text("no")
    assert run("play", "x.wav")[0] == 2 and run("play", "none.wav")[0] == 2


def test_a_device_that_fails_is_an_error(tone, run, headless_env):
    headless_env.play.side_effect = RuntimeError("no device")
    code, _out, err = run("play", tone)
    assert code == 7 and "no device" in err


def test_say_plays(fake, run, headless_env):
    assert json.loads(run("say", "Hello.", "--engine", "fake", "--play", "after", "--json")[1])["played"] is True
    assert headless_env.play.call_count == 1


def test_stop(run):
    assert run("stop") == (0, "nothing is playing\n", "")
    lock = PlayerLock().acquire()

    def hold():
        while not lock.stop_requested():
            time.sleep(0.01)
        lock.release()

    thread = threading.Thread(target=hold)
    thread.start()
    assert json.loads(run("stop", "--json")[1]) == {"stopped": True}
    thread.join()


def test_interrupt_stops_a_detached_player_first(tone, run):
    lock = PlayerLock().acquire()
    asked = []

    def hold():
        while not lock.stop_requested():
            time.sleep(0.01)
        asked.append(True)
        lock.release()

    thread = threading.Thread(target=hold)
    thread.start()
    assert run("play", tone, "--interrupt")[0] == 0
    thread.join()
    assert asked


# -- detaching -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["say", "x", "--detach"], ["render", "s.txt", "--detach"],
                                  ["say", "x", "--detach", "--play", "after", "-o", "-"]])
def test_detach_needs_something_to_play(run, argv, monkeypatch):
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started"))
    assert run(*argv)[0] == 2


def test_the_childs_arguments():
    assert cli._child_argv(["--json", "say", "--detach", "--play", "after", "--quiet", "--", "--detach"]) == \
        ["--json", "say", "--play", "after", "--detached", "--quiet", "--", "--detach"]


def test_a_detached_command_holds_the_lock_and_stops_when_asked(fake, run, headless_env):
    ended = threading.Event()
    seen = []

    def play(path, blocking):
        seen.append(player_running())
        threading.Thread(target=run_stop).start()
        ended.wait(10)

    def run_stop():
        cli.stop_detached()

    headless_env.play.side_effect = play
    headless_env.stop.side_effect = ended.set
    code, out, _err = run("say", "Hello.", "--engine", "fake", "--play", "after", "--detached", "--json")
    data = json.loads(out)
    assert code == 0 and seen == [True] and (data["played"], data["reason"]) == (False, "interrupted")
    assert not player_running()


# -- a real process ----------------------------------------------------------------------------------------------


def environment(**extra):
    env = {**os.environ, "PYTHONPATH": ROOT, PLAYBACK_ENV: "off", **extra}
    env.pop("HF_HUB_OFFLINE", None)
    return env


def test_a_process_reads_stdin_and_writes_only_the_result(tmp_path):
    done = subprocess.run([sys.executable, "-m", "kokoro_gui.headless", "say", "--engine", "dummy", "-o", "-"],
                          input="Hello there.".encode(), capture_output=True, cwd=tmp_path, env=environment(), timeout=180)
    assert done.returncode == 0, done.stderr
    assert done.stdout[:4] == b"RIFF" and len(sf.read(io.BytesIO(done.stdout))[0]) > 2400
    assert all(json.loads(line) for line in done.stderr.decode().splitlines())


def test_a_detached_process_outlives_its_parent(tmp_path):
    text = "Hello there. " * 12000  # more than a pipe holds
    done = subprocess.run([sys.executable, "-m", "kokoro_gui.headless", "render", "--engine", "dummy", "--play", "after",
                           "--detach", "--json", "-o", "out", "--title", "Book", "--channels", "1"],
                          input=text[:400].encode(), capture_output=True, cwd=tmp_path, env=environment(), timeout=180)
    assert done.returncode == 0, done.stderr
    assert isinstance(json.loads(done.stdout)["pid"], int)
    deadline = time.monotonic() + 120
    while not (tmp_path / "out" / "Book.wav").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    log = tmp_path / "audio_output" / "headless" / "player.log"
    while player_running() and time.monotonic() < deadline:
        time.sleep(0.1)
    while not log.read_bytes().strip() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert json.loads(log.read_text())["reason"] == "disabled"
    assert files_under(tmp_path / "out") == ["Book.tbaw", "Book.wav"]


def test_a_waiting_child_has_already_read_its_text(tmp_path):
    """The parent must not hang on a full pipe while another player plays."""
    lock = PlayerLock(str(tmp_path / "audio_output" / "headless")).acquire()
    try:
        done = subprocess.run([sys.executable, "-m", "kokoro_gui.headless", "say", "--engine", "dummy", "--play", "after",
                               "--detach", "--no-save"], input=("Hi there. " * 50000).encode(), capture_output=True,
                              cwd=tmp_path, env=environment(), timeout=120)
        assert done.returncode == 0, done.stderr
    finally:
        # Ends the child before it speaks 100,000 words.
        pid = int(done.stdout) if done.stdout.strip().isdigit() else None
        if pid is not None:
            _kill(pid)
        lock.release()


def _kill(pid):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal

        os.kill(pid, signal.SIGKILL)
