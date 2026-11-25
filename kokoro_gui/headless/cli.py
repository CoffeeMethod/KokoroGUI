"""The command line: `python -m kokoro_gui.headless <command>`.

stdout holds the result and nothing else: paths, a JSON object with `--json`, or
WAV bytes with `say -o -`. Progress and status are JSON lines on stderr, off
with `--quiet`. An error is a line on stderr, or with `--json` the object
`{error, message, valid}` on stdout, and its exit code (`errors.EXIT_CODES`).

`--detach` runs the same command in a child process that outlives this one and
prints its pid. The child holds the player's lock (grill PG29) and stops when
`stop` asks. What it would have printed goes to `audio_output/headless/player.log`.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
from typing import Optional

from kokoro_gui.engine import runtime

from . import listing, paths, stdio
from .errors import EXIT_INTERRUPTED, EXIT_OK, HeadlessError
from .host import Host
from .inputs import default_engine, load_settings, read_script
from .player import MODES, Player, PlayerLock, check_mode, playback_reason, stop_detached

LOG_NAME = "player.log"
MAX_CAST_FILE = 1 << 20
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Parser(argparse.ArgumentParser):
    """Errors are `HeadlessError`s, and help goes to the real stdout. No
    abbreviations: `_child_argv` has to know `--detach` when it sees it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, allow_abbrev=False, **kwargs)

    def error(self, message):
        raise HeadlessError("bad_arguments", message)

    def _print_message(self, message, file=None):
        guard = stdio.active()
        if message:
            _write(guard.stdout if guard is not None else sys.stdout, message)


def _write(stream, text: str) -> None:
    try:
        stream.write(text)
    except UnicodeEncodeError:  # a console code page
        stream.write(text.encode("ascii", "backslashreplace").decode("ascii"))
    stream.flush()


def build_parser() -> argparse.ArgumentParser:
    # Suppressed defaults, so a global works before and after the command.
    common = _Parser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("--json", action="store_true", help="print the result as JSON")
    common.add_argument("--quiet", action="store_true", help="no progress on stderr")
    common.add_argument("--device", help="auto, cpu or cuda")
    common.add_argument("--settings", metavar="PATH", help="a config_qt.json to take the lexicon and engine values from")
    common.add_argument("--allow-download", action="store_true", help="allow model downloads")

    playing = _Parser(add_help=False)
    playing.add_argument("--interrupt", action="store_true", help="stop what a detached player plays first")
    playing.add_argument("--detach", action="store_true", help="play in a process of its own and return at once")
    playing.add_argument("--detached", action="store_true", help=argparse.SUPPRESS)

    parser = _Parser(prog="kokorogui", parents=[common], description="Text and scripts to audio, without a window.")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    commands.add_parser("engines", parents=[common], help="list the engines")
    voices = commands.add_parser("voices", parents=[common], help="list an engine's voices")
    voices.add_argument("--engine")
    voices.add_argument("--lang", help="one language's voices")
    commands.add_parser("devices", parents=[common], help="list the devices")

    say = commands.add_parser("say", parents=[common, playing], help="speak text into a WAV")
    say.add_argument("text", nargs="?", default="-", help="the text, or - for stdin (the default)")
    say.add_argument("--engine")
    say.add_argument("--voice")
    say.add_argument("--speed", type=float)
    say.add_argument("--pitch", type=float)
    say.add_argument("--plain", action="store_true", help="the text isn't Markdown")
    say.add_argument("-o", "--out", help="a new .wav file, or - for stdout")
    say.add_argument("--no-save", action="store_true", help="write no file")
    say.add_argument("--play", choices=MODES, default="off")

    render = commands.add_parser("render", parents=[common, playing], help="render a script to a project and a mixdown")
    render.add_argument("script", nargs="?", default="-", help="a .txt or .md file, or - for stdin (the default)")
    render.add_argument("--engine")
    render.add_argument("--cast", action="append", default=[], metavar="SPEAKER=VOICE")
    render.add_argument("--cast-file", metavar="PATH", help="a JSON object of speaker to voice or settings")
    render.add_argument("--title")
    render.add_argument("-o", "--out", help="the output folder")
    render.add_argument("--update", metavar="PROJECT", help="a .tbaw to render again in place")
    render.add_argument("--discard-session", action="store_true", help="drop the project's unsaved changes")
    render.add_argument("--channels", type=int, choices=(1, 2), default=1)
    render.add_argument("--align", action="store_true", help="word times from Whisper where the engine gives none")
    render.add_argument("--play", choices=MODES, default="off")

    play = commands.add_parser("play", parents=[common, playing], help="play an audio file")
    play.add_argument("file")
    play.add_argument("--start", type=float)
    play.add_argument("--end", type=float)

    commands.add_parser("stop", parents=[common], help="stop the detached player")
    return parser


class _Session:
    """One command: its arguments, the real stdout, and progress."""

    def __init__(self, args, guard, argv):
        self.args = args
        self.guard = guard
        self.argv = argv
        self.json = getattr(args, "json", False)
        self.quiet = getattr(args, "quiet", False)
        self.cancel = None
        self._lock = threading.Lock()
        self._stdin = None

    # -- output ----------------------------------------------------------------

    def event(self, **fields) -> None:
        if self.quiet:
            return
        with self._lock:
            try:
                _write(sys.stderr, json.dumps(fields) + "\n")
            except OSError:
                self.quiet = True  # nobody reads it

    def progress(self, stage: str, done: int, total: int) -> None:
        self.event(event="progress", stage=stage, done=done, total=total)

    def status(self, engine_id: str, message: str, is_error: bool) -> None:
        self.event(event="status", engine=engine_id, message=message, error=is_error)

    def result(self, data, lines) -> None:
        _write(self.guard.stdout, (json.dumps(data) if self.json else "\n".join(lines)) + "\n")

    # -- inputs ----------------------------------------------------------------

    def settings(self) -> Optional[dict]:
        path = getattr(self.args, "settings", None)
        return load_settings(path) if path else None

    def allow_download(self) -> Optional[bool]:
        return True if getattr(self.args, "allow_download", False) else None

    def host(self) -> Host:
        device = getattr(self.args, "device", None)
        if device is not None and device not in ("auto", "cpu", "cuda"):
            raise HeadlessError("bad_arguments", f"there is no device {device!r}", valid=["auto", "cpu", "cuda"])
        settings = self.settings()
        return Host(default_engine(settings), allow_download=self.allow_download(), device=device,
                    settings=settings, on_status=self.status)

    def stdin(self) -> str:
        if self._stdin is None:
            self._stdin = self._read_stdin()
        return self._stdin

    @staticmethod
    def _read_stdin() -> str:
        try:
            data = sys.stdin.buffer.read() if hasattr(sys.stdin, "buffer") else sys.stdin.read().encode("utf-8")
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise HeadlessError("bad_arguments", "stdin isn't UTF-8 text") from e

    def cast(self) -> Optional[dict]:
        cast: dict = {}
        path = self.args.cast_file
        if path:
            try:
                if os.path.getsize(path) > MAX_CAST_FILE:
                    raise HeadlessError("bad_arguments", "the cast file is too big")
                with open(path, "r", encoding="utf-8") as f:
                    cast = json.load(f)
            except (OSError, ValueError, RecursionError) as e:
                raise HeadlessError("bad_arguments", f"the cast file can't be read: {e}") from e
            if not isinstance(cast, dict):
                raise HeadlessError("bad_arguments", "the cast file must hold an object")
        for entry in self.args.cast:
            name, sep, voice = entry.partition("=")
            if not sep or not name.strip() or not voice.strip():
                raise HeadlessError("bad_arguments", f"--cast takes SPEAKER=VOICE, not {entry!r}")
            for key in [k for k in cast if k.strip().lower() == name.strip().lower()]:
                del cast[key]
            cast[name.strip()] = voice.strip()
        return cast or None


# -- commands -------------------------------------------------------------------


def _engines(session: _Session) -> None:
    engines = listing.list_engines(default_engine(session.settings()))
    session.result({"engines": engines}, [
        f"{e['id']}\t{e['name']}" + ("" if e["installed"] else f"\tnot installed ({e['reason']})")
        + ("\tdefault" if e["default"] else "") for e in engines])


def _voices(session: _Session) -> None:
    with session.host() as host:
        voices = listing.list_voices(host, session.args.engine, session.args.lang)
    session.result({"voices": voices}, [
        "\t".join([v["id"], v["lang_code"] or "", "custom" if v["custom"] else "", "default" if v["default"] else ""])
        .rstrip() for v in voices])


def _devices(session: _Session) -> None:
    found = listing.list_devices()
    session.result(found, [*found["devices"], f"playback: {found['playback'] or 'available'}"])


def _say(session: _Session) -> None:
    from .speak import speak

    args = session.args
    to_stdout = args.out == "-"
    if to_stdout and (session.json or args.no_save):
        raise HeadlessError("bad_arguments", "-o - can't be used with --json or --no-save")
    if to_stdout and session.guard.binary is None:
        raise HeadlessError("bad_arguments", "stdout takes no bytes here")
    text = session.stdin() if args.text == "-" else args.text
    buffer = io.BytesIO() if to_stdout else None
    runtime.prepare_storage()
    with session.host() as host:
        result = speak(text, host=host, engine=args.engine, voice=args.voice, speed=args.speed, pitch=args.pitch,
                       markdown=not args.plain, out=None if to_stdout else args.out,
                       save=not (args.no_save or to_stdout), wav_file=buffer, play=args.play,
                       on_progress=session.progress, cancel=session.cancel)
    if to_stdout:
        session.guard.binary.write(buffer.getvalue())
        session.guard.binary.flush()
        return
    session.result(result.to_json(), [result.path] if result.path else [f"{result.duration_s:.3f} s"])


def _render(session: _Session) -> None:
    from .render import render_script

    args = session.args
    script = session.stdin() if args.script == "-" else read_script(args.script)
    cast = session.cast()
    runtime.prepare_storage()
    with session.host() as host:
        result = render_script(script, host=host, engine=args.engine, cast=cast, out=args.out, update=args.update,
                               discard_session=args.discard_session, title=args.title, channels=args.channels,
                               align=args.align, play=args.play, on_progress=session.progress, cancel=session.cancel)
    for warning in result.warnings:
        session.event(event="warning", message=warning)
    session.result(result.to_json(), [result.project_path, result.mixdown_path])


def _play(session: _Session) -> None:
    import soundfile as sf

    args = session.args
    try:
        length = sf.info(args.file).duration
    except (OSError, RuntimeError) as e:  # LibsndfileError is a RuntimeError
        raise HeadlessError("bad_arguments", f"{args.file} can't be played: {e}") from e
    start, end = args.start, args.end
    first, last = 0.0 if start is None else start, length if end is None else end
    if not 0 <= first < last:  # a NaN fails this too
        raise HeadlessError("bad_arguments", "--start must be at least 0 and before --end")
    played, reason = False, playback_reason()
    if reason is None:
        player = Player()
        try:
            item = player.enqueue(args.file, start, end)
            finished = player.wait(cancel=session.cancel)
        finally:
            player.close()
        if item.error is not None:
            raise HeadlessError("generation_failed", f"playback failed: {item.error}")
        played, reason = item.played, None if finished else "interrupted"
    session.result({"played": played, "reason": reason}, ["played" if played else f"not played: {reason}"])


def _stop(session: _Session) -> None:
    stopped = stop_detached()
    session.result({"stopped": stopped}, ["stopped" if stopped else "nothing is playing"])


_COMMANDS = {"engines": _engines, "voices": _voices, "devices": _devices, "say": _say, "render": _render,
             "play": _play, "stop": _stop}


# -- detaching -------------------------------------------------------------------


def _takes_stdin(args) -> bool:
    name = {"say": "text", "render": "script"}.get(args.command)
    return name is not None and getattr(args, name) == "-"


def _detach(session: _Session) -> None:
    """Starts this command again as a child that outlives this process."""
    args = session.args
    if args.command != "play" and check_mode(args.play) == "off":
        raise HeadlessError("bad_arguments", "--detach needs --play after or --play stream")
    if getattr(args, "out", None) == "-":
        raise HeadlessError("bad_arguments", "--detach can't be used with -o -")
    text = session.stdin() if _takes_stdin(args) else None
    root = paths.renders_root()
    os.makedirs(root, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (_ROOT, env.get("PYTHONPATH")) if p)
    flags = {"start_new_session": True}
    if sys.platform == "win32":
        flags = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with open(os.path.join(root, LOG_NAME), "wb") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "kokoro_gui.headless", *_child_argv(session.argv)], env=env, close_fds=True,
            stdin=subprocess.PIPE if text is not None else subprocess.DEVNULL, stdout=log, stderr=log, **flags)
    if text is not None:
        try:
            child.stdin.write(text.encode("utf-8"))
            child.stdin.close()
        except OSError as e:
            raise HeadlessError("generation_failed", f"the detached process ended: {e}") from e
    child.returncode = 0  # it outlives this process; nothing waits for it
    session.result({"pid": child.pid}, [str(child.pid)])


def _child_argv(argv: list) -> list:
    """The same command, marked as the child. An argument after `--` is text."""
    cut = argv.index("--") if "--" in argv else len(argv)
    options = [a for a in argv[:cut] if a not in ("--detach", "--quiet")]
    return [*options, "--detached", "--quiet", *argv[cut:]]


def _run(session: _Session) -> None:
    args = session.args
    command = _COMMANDS[args.command]
    if getattr(args, "detach", False) and not args.detached:
        _detach(session)
        return
    if not getattr(args, "detached", False):
        if getattr(args, "interrupt", False):
            stop_detached()
        command(session)
        return
    if _takes_stdin(args):
        session.stdin()  # the parent waits until it is read, and the lock can take a while
    with PlayerLock().acquire(interrupt=args.interrupt) as lock:
        session.cancel = lock.stop_requested
        command(session)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    with stdio.guard() as guard:
        session = None
        try:
            args = build_parser().parse_args(argv)
            session = _Session(args, guard, argv)
            _run(session)
            return EXIT_OK
        except HeadlessError as e:
            return _report(e, guard, session.json if session is not None else "--json" in argv)
        except KeyboardInterrupt:
            return _report(HeadlessError("interrupted", "cancelled"), guard,
                           session is not None and session.json)
        except SystemExit as e:  # --help
            return int(e.code or 0)
        except BrokenPipeError:
            return EXIT_INTERRUPTED


def _report(error: HeadlessError, guard, as_json: bool) -> int:
    try:
        if as_json:
            _write(guard.stdout, json.dumps(error.to_json()) + "\n")
        else:
            lines = f"{error}\n"
            if error.valid:
                lines += f"valid: {', '.join(map(str, error.valid))}\n"
            _write(sys.stderr, lines)
    except OSError:
        pass
    return error.exit_code
