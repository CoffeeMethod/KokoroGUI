"""headless/mcp_server.py through the SDK's own client: in this process, and over
stdio to a real one. Skipped where `requirements-mcp.txt` isn't installed."""
import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("mcp.server.mcpserver", exc_type=ImportError)

import anyio  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

from kokoro_gui.headless import stdio  # noqa: E402
from kokoro_gui.headless.errors import HeadlessError  # noqa: E402
from kokoro_gui.headless.mcp_server import Service, build_server  # noqa: E402
from kokoro_gui.headless.player import PLAYBACK_ENV  # noqa: E402
from kokoro_gui.qt import project as project_io  # noqa: E402

from .support import SCRIPT, files_under, working_dirs  # noqa: E402

pytestmark = pytest.mark.usefixtures("offline")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = ["list_engines", "list_voices", "speak", "render_script", "play", "stop_playback", "playback_status"]


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in ("KOKOROGUI_VOICE", "KOKOROGUI_DEVICE", "KOKOROGUI_SETTINGS", PLAYBACK_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KOKOROGUI_ENGINE", "fake")
    yield
    assert stdio.active() is None


def session(work, mode="auto"):
    """Runs `await work(client, service)` against a server of its own."""
    service = Service()

    async def main():
        async with Client(build_server(service), mode=mode) as client:
            return await work(client, service)

    try:
        return asyncio.run(asyncio.wait_for(main(), 60))
    finally:
        assert not service._generating.locked() and service._calls == set()
        assert service.host._closing and service.player._closed


def call(name, arguments=None, **kwargs):
    async def work(client, _service):
        return await client.call_tool(name, arguments or {}, **kwargs)

    return session(work)


def data(result):
    """A good result's object, the same as text and as structured content."""
    assert not result.is_error, result.content
    assert json.loads(result.content[0].text) == result.structured_content and len(result.content) == 1
    return result.structured_content


def error(result):
    assert result.is_error
    assert set(result.structured_content) == {"error", "message", "valid"}
    assert json.loads(result.content[0].text) == result.structured_content
    return result.structured_content


# -- what it offers ----------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_the_tools_and_their_schemas(fake, mode):
    async def work(client, _service):
        return (await client.list_tools()).tools

    tools = {tool.name: tool for tool in session(work, mode)}
    assert list(tools) == TOOLS
    assert all(tool.description and "ctx" not in tool.input_schema["properties"] for tool in tools.values())
    speak = tools["speak"].input_schema
    assert speak["required"] == ["text"]
    assert speak["properties"]["play"]["default"] == "stream"  # grill PG25
    assert speak["properties"]["wait_for_playback"]["default"] is False
    render = tools["render_script"].input_schema
    assert render.get("required", []) == [] and render["properties"]["play"]["default"] == "off"
    assert render["properties"]["channels"]["enum"] == [1, 2] and render["properties"]["channels"]["default"] == 1
    assert tools["playback_status"].annotations.read_only_hint is True
    assert tools["render_script"].annotations.destructive_hint is True


def test_engines_and_voices(fake):
    rows = {e["id"]: e for e in data(call("list_engines"))["engines"]}
    assert rows["fake"]["default"] is True and rows["kokoro"]["default"] is False
    assert [v["id"] for v in data(call("list_voices"))["voices"]] == ["v1", "v2", "w1"]
    assert [v["id"] for v in data(call("list_voices", {"lang_code": "b"}))["voices"]] == ["w1"]
    assert error(call("list_voices", {"engine": "nope"}))["error"] == "engine_not_installed"


# -- speak ----------------------------------------------------------------------------------


def test_speak_streams_by_default_and_returns_a_path(fake, headless_env):
    result = data(call("speak", {"text": "Hello **there**.", "voice": "v2", "wait_for_playback": True}))
    assert os.path.isfile(result["path"]) and (result["played"], result["reason"]) == (True, None)
    assert [w["word"] for w in result["words"]] == ["Hello", "there."]
    assert fake.spoken == [("Hello there.", "v2", 1.0, "a")] and headless_env.play.call_count == 1


def test_speak_does_not_wait_unless_asked(fake, headless_env):
    ended = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(20)
    headless_env.stop.side_effect = ended.set

    async def work(client, _service):
        result = data(await client.call_tool("speak", {"text": "Hello."}))
        status = data(await client.call_tool("playback_status", {}))
        stopped = data(await client.call_tool("stop_playback", {}))
        return result, status, stopped

    result, status, stopped = session(work)
    assert result["played"] is True and status["playing"] and status["reason"] is None
    assert stopped == {"playing": None, "elapsed_s": 0.0, "queued": 0, "reason": None}


def test_a_waiting_speak_lets_another_call_generate(fake, headless_env):
    ended = threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(20)
    headless_env.stop.side_effect = ended.set

    async def work(client, _service):
        results = []

        async def first():
            results.append(data(await client.call_tool("speak", {"text": "First.", "wait_for_playback": True})))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first)
            while headless_env.play.call_count == 0:
                await anyio.sleep(0.01)
            second = data(await client.call_tool("speak", {"text": "Second.", "play": "off"}))
            assert second["path"] and not results
            await client.call_tool("stop_playback", {})
        return results[0]

    assert (session(work)["played"]) is False


def test_speak_saves_nothing_when_told(fake, tmp_path):
    result = data(call("speak", {"text": "Hello.", "save": False, "play": "off"}))
    assert result["path"] is None and not (tmp_path / "audio_output").exists()


def test_the_default_voice_comes_from_the_environment(fake, monkeypatch):
    monkeypatch.setenv("KOKOROGUI_VOICE", "v2")
    call("speak", {"text": "One.", "play": "off"})
    call("speak", {"text": "Two.", "play": "off", "voice": "v1"})
    assert [s[1] for s in fake.spoken] == ["v2", "v1"]
    assert data(call("speak", {"text": "Three.", "play": "off", "engine": "dummy"}))["voice"] == "dummy"


@pytest.mark.parametrize("text", ["123", "true", "null", "[1, 2]", '{"a": 1}', '"quoted"'])
def test_text_that_looks_like_json_is_text(fake, text):
    assert not call("speak", {"text": text, "markdown": False, "play": "off"}).is_error
    assert not call("render_script", {"script": text}).is_error
    assert [s[0] for s in fake.spoken] in ([text, text], [text])


# -- render_script -----------------------------------------------------------------------------


def test_render_text_then_update(fake):
    first = data(call("render_script", {"script": SCRIPT, "cast": {"Sam": "v2", "Ann": {"voice": "v1", "speed": 2}}}))
    assert (first["clips"], first["channels"], first["played"]) == (3, 1, False)
    assert [s[1] for s in fake.spoken] == ["v1", "v2", "v1"]
    second = data(call("render_script", {"script": SCRIPT.replace("Who", "And who"), "update": first["project_path"]}))
    assert second["regenerated"] == 1 and second["mixdown_path"] == first["mixdown_path"]
    project_io.inspect_bundle(first["project_path"])


def test_render_a_file(fake, tmp_path):
    (tmp_path / "s.md").write_text("# Title\n\n[Sam]: Hi.")
    assert data(call("render_script", {"path": "s.md"}))["clips"] == 2


@pytest.mark.parametrize("arguments", [{}, {"script": "x", "path": "s.txt"}, {"path": "none.txt"}, {"path": "../s.txt"},
                                       {"path": "x.pdf"}, {"script": "x", "cast": {"Narrator": 4}}, {"script": "[A]:"}])
def test_a_script_that_is_not_one(fake, tmp_path, arguments):
    (tmp_path / "s.txt").write_text("text")
    assert error(call("render_script", arguments))["error"] == "bad_arguments"
    assert fake.spoken == []


@pytest.mark.parametrize("name, arguments", [
    ("speak", {"text": "x", "play": "later"}), ("speak", {}), ("speak", {"text": "x", "speed": "fast"}),
    ("render_script", {"script": "x", "channels": 3}), ("render_script", {"script": "x", "cast": ["Sam"]}),
])
def test_arguments_the_schema_refuses(fake, name, arguments):
    assert call(name, arguments).is_error and fake.spoken == []


# -- errors ----------------------------------------------------------------------------------------


def test_errors_carry_the_code_and_what_would_be_valid(fake, tmp_path):
    found = error(call("speak", {"text": "x", "voice": "nope"}))
    assert (found["error"], found["valid"]) == ("unknown_voice", ["v1", "v2", "w1"])
    assert error(call("speak", {"text": "x", "engine": "nope"}))["error"] == "engine_not_installed"
    assert error(call("speak", {"text": "x", "out": "../x.wav"}))["error"] == "out_not_allowed"
    assert error(call("render_script", {"script": "x", "update": "../x.tbaw"}))["error"] == "out_not_allowed"
    fake.speak = lambda text: 1 / 0
    assert error(call("speak", {"text": "something new"}))["error"] == "generation_failed"


def test_no_result_holds_audio(fake):
    text = call("render_script", {"script": SCRIPT}).content[0].text
    assert "RIFF" not in text and len(text) < 4000


# -- progress, concurrency, cancelling ---------------------------------------------------------------


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_progress_goes_up_to_100(fake, mode):
    seen = []

    async def work(client, _service):
        async def progress(value, total, message):
            seen.append((value, total, message))

        return await client.call_tool("render_script", {"script": SCRIPT}, progress_callback=progress)

    data(session(work, mode))
    values = [v for v, _t, _m in seen]
    assert values == sorted(set(values)) and values[-1] == 100.0 and all(t == 100.0 for _v, t, _m in seen)


def test_stop_playback_gets_through_while_a_script_renders(fake, headless_env):
    speaking, release, ended = threading.Event(), threading.Event(), threading.Event()
    headless_env.play.side_effect = lambda path, blocking: ended.wait(20)
    headless_env.stop.side_effect = ended.set

    def say(text):
        if len(fake.spoken) == 2:
            speaking.set()
            assert release.wait(20)

    fake.speak = say

    async def work(client, _service):
        results = []

        async def render():
            results.append(data(await client.call_tool("render_script", {"script": SCRIPT, "play": "stream"})))

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(render)
            while not speaking.is_set():
                await anyio.sleep(0.01)
            stopped = data(await client.call_tool("stop_playback", {}))
            assert not results
            release.set()
        return stopped, results[0]

    stopped, rendered = session(work)
    assert stopped["playing"] is None and ended.is_set() and os.path.isfile(rendered["mixdown_path"])


def test_calls_generate_one_at_a_time(fake):
    running, most = [], []

    def say(text):
        running.append(text)
        most.append(len(running))
        time.sleep(0.05)
        running.remove(text)

    fake.speak = say

    async def work(client, _service):
        async with anyio.create_task_group() as tasks:
            for n in range(4):
                tasks.start_soon(client.call_tool, "speak", {"text": f"Number {n}.", "play": "off"})

    session(work)
    assert len(fake.spoken) == 4 and max(most) == 1


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_a_cancelled_call_stops_generating_and_leaves_nothing(fake, tmp_path, mode):
    fake.speak = lambda text: time.sleep(0.05)
    script = "\n\n".join(f"Paragraph {n}." for n in range(200))

    async def work(client, service):
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(client.call_tool, "render_script", {"script": script})
                while not fake.spoken:
                    await anyio.sleep(0.01)
                scope.cancel()
        with anyio.fail_after(20):
            while service._generating.locked():
                await anyio.sleep(0.01)
        done = len(fake.spoken)
        after = data(await client.call_tool("speak", {"text": "Still here.", "play": "off", "save": False}))
        return done, after

    done, after = session(work, mode)
    assert done < 200 and after["duration_s"] > 0
    assert files_under(tmp_path / "audio_output") == []
    assert not any(project_io.is_locked(d) for d in working_dirs())


def test_the_server_ending_stops_a_render(fake, tmp_path):
    fake.speak = lambda text: time.sleep(0.05)
    script = "\n\n".join(f"Paragraph {n}." for n in range(400))
    service = Service()

    async def main():
        async with Client(build_server(service)) as client:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(client.call_tool, "render_script", {"script": script})
                while not fake.spoken:
                    await anyio.sleep(0.01)
                tasks.cancel_scope.cancel()

    asyncio.run(asyncio.wait_for(main(), 60))
    assert len(fake.spoken) < 400 and not service._generating.locked() and fake.closed == 1
    assert files_under(tmp_path / "audio_output") == []


def test_one_load_for_the_whole_session(fake):
    loads = []
    fake.load = lambda: loads.append(1)

    async def work(client, _service):
        for text in ("One.", "Two."):
            data(await client.call_tool("speak", {"text": text, "play": "off"}))

    session(work)
    assert loads == [1]


# -- play -------------------------------------------------------------------------------------------------


@pytest.fixture
def tone(tmp_path):
    sf.write(tmp_path / "t.wav", np.zeros(2400), 24000)
    return "t.wav"


def test_play_and_a_range(tone, headless_env):
    assert data(call("play", {"path": tone, "wait_for_playback": True})) == \
        {"played": True, "reason": None, "duration_s": 0.1}
    assert data(call("play", {"path": tone, "start_s": 0.02, "wait_for_playback": True}))["played"]
    assert headless_env.play_range.call_args.args[1:] == (0.02, 0.1)


@pytest.mark.parametrize("arguments", [{"path": "none.wav"}, {"path": "../t.wav"}, {"path": "//host/t.wav"},
                                       {"path": "t.wav", "start_s": 5}, {"path": "t.wav", "end_s": 0},
                                       {"path": "t.txt"}])
def test_play_refuses(tone, tmp_path, headless_env, arguments):
    (tmp_path / "t.txt").write_text("no")
    assert error(call("play", arguments))["error"] == "bad_arguments"
    assert headless_env.play.call_count == 0


def test_files_outside_the_allowed_folders_are_refused(tone, fake, tmp_path, monkeypatch, headless_env):
    """They are there and they are readable."""
    (tmp_path / "s.txt").write_text("secret")
    inner = tmp_path / "inner"
    inner.mkdir()
    monkeypatch.chdir(inner)
    for outside in ("../s.txt", str(tmp_path / "s.txt")):
        assert "inside" in error(call("render_script", {"path": outside}))["message"]
    for outside in ("../t.wav", str(tmp_path / "t.wav")):
        assert "inside" in error(call("play", {"path": outside}))["message"]
    assert fake.spoken == [] and headless_env.play.call_count == 0


def test_prints_go_to_stderr_while_it_serves(fake):
    async def work(_client, _service):
        return sys.stdout is sys.stderr

    assert session(work) is True


def test_why_nothing_plays(tone, fake, monkeypatch):
    monkeypatch.setenv(PLAYBACK_ENV, "off")
    assert data(call("play", {"path": tone}))["reason"] == "disabled"
    assert data(call("speak", {"text": "x"}))["reason"] == "disabled"
    assert data(call("playback_status"))["reason"] == "disabled"


# -- the environment ------------------------------------------------------------------------------------------


def test_settings_and_device(fake, monkeypatch, tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"lexicon": {"door": "gate"}, "engines": {"fake": {"voice": "w1"}}}))
    monkeypatch.setenv("KOKOROGUI_SETTINGS", "c.json")
    call("speak", {"text": "The door.", "play": "off"})
    assert fake.spoken[0][:2] == ("The gate.", "v1")  # "w1" is another language's
    monkeypatch.setenv("KOKOROGUI_DEVICE", "cpu")
    assert Service().host.device == "cpu"
    monkeypatch.setenv("KOKOROGUI_DEVICE", "tpu")
    with pytest.raises(HeadlessError) as caught:
        Service()
    assert caught.value.code == "bad_arguments"


def test_a_default_engine_that_is_not_there_is_an_error_per_call(monkeypatch):
    monkeypatch.setenv("KOKOROGUI_ENGINE", "nope")
    assert error(call("speak", {"text": "x"}))["error"] == "engine_not_installed"
    assert "engines" in data(call("list_engines"))


# -- a real process -------------------------------------------------------------------------------------------


def process_env(**extra):
    env = {**os.environ, "PYTHONPATH": ROOT, PLAYBACK_ENV: "off", "KOKOROGUI_ENGINE": "dummy", **extra}
    env.pop("HF_HUB_OFFLINE", None)
    return env


NOISY = """
import os, sys
from kokoro_gui.engines.dummy import DummyModel
real = DummyModel.synthesize
def synthesize(self, *args):
    print("noise on stdout")
    os.write(1, b"noise on descriptor 1\\n")
    return real(self, *args)
DummyModel.synthesize = synthesize
from kokoro_gui.headless import mcp_server
sys.exit(mcp_server.main())
"""


@pytest.mark.parametrize("command", [["-m", "kokoro_gui.headless.mcp_server"], ["-c", NOISY]], ids=["plain", "noisy"])
def test_over_stdio(tmp_path, command):
    """An engine that prints, even straight to descriptor 1, doesn't break the
    protocol."""
    async def main():
        server = StdioServerParameters(command=sys.executable, args=command, env=process_env(), cwd=str(tmp_path))
        async with Client(server) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            spoken = data(await client.call_tool("speak", {"text": "Hello there."}))
            rendered = data(await client.call_tool("render_script", {"script": SCRIPT}))
            return names, spoken, rendered

    names, spoken, rendered = asyncio.run(asyncio.wait_for(main(), 120))
    assert names == TOOLS and spoken["reason"] == "disabled" and rendered["clips"] == 3
    assert os.path.isfile(spoken["path"]) and os.path.isfile(rendered["project_path"])


def test_a_bad_environment_ends_the_process_with_its_code(tmp_path):
    done = subprocess.run([sys.executable, "-m", "kokoro_gui.headless.mcp_server"], cwd=tmp_path, stdin=subprocess.DEVNULL,
                          env=process_env(KOKOROGUI_DEVICE="tpu"), capture_output=True, timeout=120)
    assert done.returncode == 2 and done.stdout == b"" and b"bad_arguments" in done.stderr


def test_the_process_ends_when_stdin_does(tmp_path):
    done = subprocess.run([sys.executable, "-m", "kokoro_gui.headless.mcp_server"], cwd=tmp_path, stdin=subprocess.DEVNULL,
                          env=process_env(), capture_output=True, timeout=120)
    assert done.returncode == 0 and done.stdout == b""


def test_without_mcp_it_says_what_to_install(tmp_path):
    code = ("import runpy, sys; sys.modules['mcp'] = None; "
            "runpy.run_module('kokoro_gui.headless.mcp_server', run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=process_env(), capture_output=True, timeout=120)
    assert done.returncode == 1 and done.stdout == b""
    assert b"requirements-mcp.txt" in done.stderr and b"Traceback" not in done.stderr


def test_the_module_imports_no_qt(tmp_path):
    from .test_imports import FORBIDDEN

    code = "import json, sys, kokoro_gui.headless.mcp_server; print(json.dumps(sorted(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=process_env(), capture_output=True, timeout=120)
    assert done.returncode == 0, done.stderr
    loaded = json.loads(done.stdout.splitlines()[-1])
    assert [n for n in loaded if any(n == f or n.startswith(f + ".") for f in FORBIDDEN)] == []
