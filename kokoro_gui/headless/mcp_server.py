"""The MCP server, over stdio: `python -m kokoro_gui.headless.mcp_server`.

Needs `pip install -r requirements-mcp.txt`. The process lives for the agent's
session, so a model loads once. One call generates at a time, since an engine
has one cancel event; playback goes on meanwhile and `stop_playback` always gets
through. Tools run in threads, report progress, and stop when the client cancels
the call. A failure is an `isError` result holding `{error, message, valid}`.
Results hold paths and numbers, never audio.

Every path a tool takes is guarded: inside `audio_output/headless/` or the
directory the server was started in, links followed. Defaults come from the
environment: `KOKOROGUI_ENGINE`, `KOKOROGUI_VOICE`, `KOKOROGUI_DEVICE`,
`KOKOROGUI_SETTINGS` (a `config_qt.json`), `KOKOROGUI_ALLOW_DOWNLOAD` and
`KOKOROGUI_PLAYBACK`.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import sys
import threading
import time
from typing import Any, Callable, Literal, Optional

import anyio
import anyio.to_thread

try:
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.types import CallToolResult, TextContent, ToolAnnotations
except ImportError as e:
    _MISSING = f"the MCP server needs mcp 2 or later: pip install -r requirements-mcp.txt ({e})"
    if __name__ == "__main__":
        sys.exit(_MISSING)
    raise ImportError(_MISSING) from e

from kokoro_gui.engine import runtime

from . import listing, paths, stdio
from .errors import HeadlessError
from .host import POLL_S, Host
from .inputs import default_engine, load_settings, read_script
from .player import Player, playback_reason
from .render import render_script as _render_script
from .speak import speak as _speak

SERVER_NAME = "kokorogui"
ENGINE_ENV, VOICE_ENV, DEVICE_ENV, SETTINGS_ENV = (
    "KOKOROGUI_ENGINE", "KOKOROGUI_VOICE", "KOKOROGUI_DEVICE", "KOKOROGUI_SETTINGS")
DEVICES = ("auto", "cpu", "cuda")
# Where each stage sits in one call's 0 to 100.
_STAGES = {"generate": (0.0, 70.0), "align": (70.0, 80.0), "mix": (80.0, 95.0), "save": (95.0, 100.0)}
INSTRUCTIONS = (
    "Text to speech on this machine. speak says a short text. render_script turns a script, optionally tagged "
    "'[Speaker]: text' with '[pause:1.5]' markers, into a KokoroGUI project and a WAV mixdown; pass update to change "
    "one it made. Call list_voices before choosing a voice. Results name files; no audio is returned.")


def _env(name: str) -> Optional[str]:
    return os.environ.get(name, "").strip() or None


class _Recorder:
    """A player that remembers what one call queued, so the call can wait for
    it after another call may generate."""

    def __init__(self, player: Player):
        self._player = player
        self.items: list = []

    def enqueue(self, *args, **kwargs):
        item = self._player.enqueue(*args, **kwargs)
        self.items.append(item)
        return item

    def stop(self, group=None) -> None:
        self._player.stop(group)

    def wait_for_all(self, cancel: Callable[[], bool]) -> tuple:
        """`(played, reason)`. A cancel drops what is left."""
        for item in self.items:
            while not item.done.wait(POLL_S):
                if cancel():
                    self._player.stop(item.group)
        failed = next((i.error for i in self.items if i.error is not None), None)
        if failed is not None:
            return False, f"playback failed: {failed}"
        played = all(item.played for item in self.items)
        return played, None if played else "interrupted"


class _Progress:
    """Progress from any thread to one call's client, never going back."""

    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()
        self._latest = (0.0, "")
        self._sent = 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        low, high = _STAGES.get(stage, (0.0, 100.0))
        value = low + (high - low) * min(1.0, done / max(1, total))
        with contextlib.suppress(RuntimeError):  # the loop is closed
            self._loop.call_soon_threadsafe(self._set, value, stage)

    def _set(self, value: float, stage: str) -> None:
        if value > self._latest[0]:
            self._latest = (value, stage)
            self._changed.set()

    async def flush(self) -> None:
        value, stage = self._latest
        if value > self._sent:
            self._sent = value
            await self._ctx.report_progress(value, 100.0, stage)

    async def pump(self) -> None:
        while True:
            await self._changed.wait()
            self._changed.clear()
            await self.flush()


def _result(data: dict) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))], structured_content=data)


def _error(error: HeadlessError) -> CallToolResult:
    data = error.to_json()
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))], structured_content=data,
                          is_error=True)


class Service:
    """What the tools share: the host, the player and the generating lock."""

    def __init__(self):
        settings_path = _env(SETTINGS_ENV)
        settings = load_settings(settings_path) if settings_path else None
        device = _env(DEVICE_ENV)
        if device is not None and device not in DEVICES:
            raise HeadlessError("bad_arguments", f"{DEVICE_ENV} must be one of {', '.join(DEVICES)}", valid=DEVICES)
        self.engine = _env(ENGINE_ENV) or default_engine(settings)
        self.voice = _env(VOICE_ENV)
        self.host = Host(self.engine, device=device, settings=settings)
        self.player = Player()
        self._generating = threading.Lock()
        self._calls: set = set()

    @contextlib.contextmanager
    def generating(self, cancel: Callable[[], bool]):
        while not self._generating.acquire(timeout=POLL_S):
            if cancel():
                raise HeadlessError("interrupted", "cancelled while another call was generating")
        try:
            yield
        finally:
            self._generating.release()

    async def call(self, ctx: Context, work: Callable[[Callable[[], bool], _Progress], dict]) -> CallToolResult:
        """`work(cancel, progress)` in a thread. When the client cancels the
        call, or the server ends, `cancel()` turns true and the thread ends by
        itself."""
        cancelled = threading.Event()
        self._calls.add(cancelled)
        progress = _Progress(ctx)
        result = None
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(progress.pump)
                try:
                    result = _result(await anyio.to_thread.run_sync(
                        functools.partial(work, cancelled.is_set, progress), abandon_on_cancel=True))
                except HeadlessError as e:
                    result = _error(e)
                await anyio.sleep(0)  # what the thread reported last is in the loop's queue
                await progress.flush()
                tasks.cancel_scope.cancel()
            return result
        finally:
            cancelled.set()
            self._calls.discard(cancelled)

    def close(self) -> None:
        for cancelled in list(self._calls):
            cancelled.set()
        self.player.close()
        self.host.close()
        # A thread nobody waits for any more still has to unlock its project.
        deadline = time.monotonic() + 10.0
        while self._generating.locked() and time.monotonic() < deadline:
            time.sleep(POLL_S)

    # -- tools, on a worker thread -----------------------------------------------

    def speak(self, cancel, progress, *, wait_for_playback: bool, interrupt: bool, engine, voice, **kwargs) -> dict:
        if interrupt:
            self.player.stop()
        player = _Recorder(self.player)
        if voice is None and (engine or self.engine) == self.engine:
            voice = self.voice
        with self.generating(cancel):
            result = _speak(host=self.host, engine=engine, voice=voice, player=player, interrupt=interrupt,
                            on_progress=progress, cancel=cancel, **kwargs)
        return self._played(result, player, wait_for_playback, cancel)

    def render(self, cancel, progress, *, script, path, wait_for_playback: bool, interrupt: bool, **kwargs) -> dict:
        if bool(script) == bool(path):
            raise HeadlessError("bad_arguments", "pass script or path, one of them")
        if path:
            script = read_script(path, guarded=True)
        if interrupt:
            self.player.stop()
        player = _Recorder(self.player)
        with self.generating(cancel):
            result = _render_script(script, host=self.host, player=player, interrupt=interrupt, on_progress=progress,
                                    cancel=cancel, **kwargs)
        return self._played(result, player, wait_for_playback, cancel)

    @staticmethod
    def _played(result, player: _Recorder, wait: bool, cancel) -> dict:
        if wait and player.items:
            result.played, result.reason = player.wait_for_all(cancel)
        return result.to_json()

    def play(self, cancel, _progress, *, path, start_s, end_s, interrupt: bool, wait_for_playback: bool) -> dict:
        import soundfile as sf

        try:
            path = paths.check_out_path(path)
        except HeadlessError as e:
            raise HeadlessError("bad_arguments", "path must be inside audio_output/headless or the current directory",
                                valid=e.valid) from e
        try:
            length = sf.info(path).duration
        except (OSError, RuntimeError) as e:  # LibsndfileError is a RuntimeError
            raise HeadlessError("bad_arguments", f"that file can't be played: {e}") from e
        first, last = 0.0 if start_s is None else start_s, length if end_s is None else end_s
        if not 0 <= first < last:  # a NaN fails this too
            raise HeadlessError("bad_arguments", "start_s must be at least 0 and before end_s")
        reason = playback_reason()
        if reason is not None:
            return {"played": False, "reason": reason, "duration_s": length}
        player = _Recorder(self.player)
        player.enqueue(path, start_s, end_s, interrupt=interrupt, group=player)
        played, reason = player.wait_for_all(cancel) if wait_for_playback else (True, None)
        return {"played": played, "reason": reason, "duration_s": length}

    def stop(self, _cancel, _progress) -> dict:
        self.player.stop()
        return self.status(None, None)

    def status(self, _cancel, _progress) -> dict:
        return {**self.player.status(), "reason": playback_reason()}

    def engines(self, _cancel, _progress) -> dict:
        return {"engines": listing.list_engines(self.engine)}

    def voices(self, _cancel, _progress, *, engine, lang_code) -> dict:
        return {"voices": listing.list_voices(self.host, engine, lang_code)}


def build_server(service: Service) -> MCPServer:
    """The server and its tools. `service` is closed when the server ends."""

    @contextlib.asynccontextmanager
    async def lifespan(_server):
        # The transport has its own copy of stdout by now; anything an engine
        # prints goes to stderr.
        with stdio.guard():
            try:
                yield service
            finally:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(service.close)

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS, lifespan=lifespan, log_level="WARNING")
    read_only = ToolAnnotations(read_only_hint=True, open_world_hint=False)
    writes = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
    rewrites = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

    def run(ctx: Context, method: str, **kwargs) -> Any:
        return service.call(ctx, functools.partial(getattr(service, method), **kwargs))

    @server.tool(annotations=read_only)
    async def list_engines(ctx: Context) -> CallToolResult:
        """The speech engines: id, name, whether each is installed, and the default."""
        return await run(ctx, "engines")

    @server.tool(annotations=read_only)
    async def list_voices(ctx: Context, engine: Optional[str] = None,
                          lang_code: Optional[str] = None) -> CallToolResult:
        """An engine's voices, of one language when lang_code is given."""
        return await run(ctx, "voices", engine=engine, lang_code=lang_code)

    @server.tool(annotations=writes)
    async def speak(ctx: Context, text: str, engine: Optional[str] = None, voice: Optional[str] = None,
                    speed: Optional[float] = None, pitch: Optional[float] = None, markdown: bool = True,
                    out: Optional[str] = None, save: bool = True, play: Literal["off", "after", "stream"] = "stream",
                    wait_for_playback: bool = False, interrupt: bool = False) -> CallToolResult:
        """Speaks a short text into a WAV, and by default plays it while it is made.

        markdown=True removes Markdown first. out is a new .wav; save=False writes none. speed is 0.1 to 4, pitch is
        -12 to 12 semitones. interrupt stops what is playing. Returns the path, the duration and word times."""
        return await run(ctx, "speak", text=text, engine=engine, voice=voice, speed=speed, pitch=pitch,
                         markdown=markdown, out=out, save=save, play=play, wait_for_playback=wait_for_playback,
                         interrupt=interrupt)

    @server.tool(annotations=rewrites)
    # Plain `str`: the SDK parses an `Optional[str]` that looks like JSON, and a
    # script may be "[1, 2]".
    async def render_script(ctx: Context, script: str = "", path: str = "",
                            engine: Optional[str] = None, cast: Optional[dict[str, Any]] = None,
                            out: Optional[str] = None, update: Optional[str] = None, discard_session: bool = False,
                            title: Optional[str] = None, channels: Literal[1, 2] = 1, align: bool = False,
                            play: Literal["off", "after", "stream"] = "off", wait_for_playback: bool = False,
                            interrupt: bool = False) -> CallToolResult:
        """Renders a script to a KokoroGUI project (.tbaw) and a WAV mixdown.

        Pass script (the text) or path (a .txt or .md file). Lines start with '[Speaker]:' to name a speaker, and
        '[pause:1.5]' adds seconds of silence. cast maps a speaker to a voice, or to {voice, speed, pitch, fx_preset}.
        out is the output folder. update is a project to render again in place: only changed paragraphs are generated,
        and edits made to its clips in the app are lost. discard_session also drops changes the app has not saved."""
        return await run(ctx, "render", script=script, path=path, engine=engine, cast=cast, out=out, update=update,
                         discard_session=discard_session, title=title, channels=channels, align=align, play=play,
                         wait_for_playback=wait_for_playback, interrupt=interrupt)

    @server.tool(annotations=writes)
    async def play(ctx: Context, path: str, start_s: Optional[float] = None, end_s: Optional[float] = None,
                   interrupt: bool = False, wait_for_playback: bool = False) -> CallToolResult:
        """Plays an audio file, or a range of it, after what is already queued."""
        return await run(ctx, "play", path=path, start_s=start_s, end_s=end_s, interrupt=interrupt,
                         wait_for_playback=wait_for_playback)

    @server.tool(annotations=writes)
    async def stop_playback(ctx: Context) -> CallToolResult:
        """Stops what is playing and drops what is queued. A render goes on."""
        return await run(ctx, "stop")

    @server.tool(annotations=read_only)
    async def playback_status(ctx: Context) -> CallToolResult:
        """What is playing, for how long, and how much is queued."""
        return await run(ctx, "status")

    return server


def main() -> int:
    runtime.prepare_storage()
    try:
        with stdio.guard():
            service = Service()
        # Not under `stdio.guard`: the transport takes the real stdout, and
        # moves file descriptor 1 to stderr, before the lifespan guards it.
        build_server(service).run("stdio")
    except HeadlessError as e:
        print(e, file=sys.stderr)
        return e.exit_code
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
