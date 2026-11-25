"""Short text to audio, with no project.

Segments are generated under their key into `<CACHE_DIR>/headless_speak/`, so a
repeated phrase is a file read. The file a caller gets is those segments after
the character's post-processing (a pitch), joined. A segment's word times sit
beside it in `<segment>.words.json`, since a cache hit doesn't run the model.

`play="stream"` generates the text in pieces of about `STREAM_WORDS` words and
queues each as it lands, on every engine (grill PG25).
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import BinaryIO, Callable, Optional

import numpy as np
import soundfile as sf

from kokoro_gui.audio import post
from kokoro_gui.core import config as core_config
from kokoro_gui.core import engines as core_engines
from kokoro_gui.core import fx as core_fx
from kokoro_gui.daw.models import Document
from kokoro_gui.engine.caching import split_segments
from kokoro_gui.engine.lexicon import normalize_rules, spoken
from kokoro_gui.engine.text_extraction import strip_markup
from kokoro_gui.qt import project as project_io

from . import paths
from .clean import clean_markdown, normalise_text
from .errors import HeadlessError
from .host import Host
from .output import Output
from .player import Playback, Player
from .render import _bad, _poll, _quietly, _stop_if_cancelled
from .results import SpeakResult
from .scratch import scratch_path, speak_dir
from .script import _check_number

STREAM_WORDS = 15
MAX_WORDS_FILE = 1 << 20


def _words_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".words.json"


def _store_words(path: str, words: list) -> None:
    target = _words_path(path)
    tmp = f"{target}.{os.urandom(4).hex()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(words, f)
        os.replace(tmp, target)
    except OSError:  # the audio is there; the words are a courtesy
        try:
            os.remove(tmp)
        except OSError:
            pass


def _stored_words(path: str) -> list:
    """The words beside a segment, or [] for anything that isn't
    `[[text, start, end], ...]`."""
    target = _words_path(path)
    try:
        if os.path.getsize(target) > MAX_WORDS_FILE:
            return []
        with open(target, "r", encoding="utf-8") as f:
            return [[str(w[0]), float(w[1]), float(w[2])] for w in json.load(f)]
    except (OSError, ValueError, TypeError, IndexError, KeyError, RecursionError):
        return []


def _check_out(out) -> Optional[str]:
    if out is None:
        return None
    real = paths.check_out_path(out)
    if os.path.splitext(real)[1].lower() != ".wav":
        raise _bad("out must be a .wav file")
    return real


class _Voice:
    """One call's character: its config, and its post-processing."""

    def __init__(self, host: Host, backend, voice: Optional[str], speed, pitch):
        self.backend = backend
        settings = core_engines.engine_settings(host.settings, backend.id)
        known = core_engines.voice_names(backend, None)
        self.name = voice if voice is not None else core_engines.default_voice(backend, settings)
        if self.name not in known:
            what = f"no voice {str(self.name)[:200]!r}" if self.name else "no default voice"
            raise HeadlessError("unknown_voice", f"{backend.display_name} has {what}", valid=known)
        preset = {"voice": self.name}
        for key, value in (("speed", speed), ("pitch", pitch)):
            if value is not None:
                try:
                    preset[key] = _check_number("voice", key, value)
                except ValueError as e:
                    raise _bad(str(e).split(": ", 1)[-1]) from e
        character = core_engines.new_character("Speech", "#000000", backend, settings)
        character.preset_data = preset
        document = Document.from_plain_text("x", characters=[character])
        clip = document.assign_character_to_range(0, 1, character.id)
        inputs = core_config.GenerationInputs.defaults()
        if host.settings.get("lexicon"):
            inputs = replace(inputs, lexicon=host.settings["lexicon"])
        self.config = core_config.clip_config(
            inputs, core_config.OutputInputs.defaults(), settings, document, clip, backend, None, {},
            core_fx.FxResolution(apply_fx=False))
        self.config.update(out_dir=speak_dir(), segment_naming="cache_key", format="wav")
        self.post = core_config.post_config(self.config)
        self.lexicon = normalize_rules(inputs.lexicon)

    @property
    def plain(self) -> bool:
        return post.is_identity(self.post)


class _Speech:
    """The audio and words of the pieces so far."""

    def __init__(self, voice: _Voice, rate: int):
        self.voice = voice
        self.rate = rate
        self.parts: list = []
        self.words: list = []
        self.cached = True

    def add(self, results: list, before: set) -> list:
        """Lands one generate call; returns `[(segment path, samples)]`."""
        added = []
        for result in results:
            hit = os.path.basename(result["path"]) in before
            self.cached = self.cached and hit
            words = result.get("words") or []
            if words and not hit:
                _store_words(result["path"], words)
            elif not words:
                words = _stored_words(result["path"])
            samples = post.render(result["path"], self.voice.post, self.rate)
            start = sum(len(p) for p in self.parts) / self.rate
            raw = float(result["duration"]) or 1.0
            scale = len(samples) / self.rate / raw
            self.words += [{"word": text, "start": round(start + a * scale, 4), "end": round(start + b * scale, 4)}
                           for text, a, b in words]
            self.parts.append(samples)
            added.append((result["path"], samples))
        return added

    def samples(self) -> np.ndarray:
        return np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.float32)


def _present() -> set:
    try:
        return set(os.listdir(speak_dir()))
    except OSError:
        return set()


def speak(text: str, *, host: Optional[Host] = None, engine: Optional[str] = None, voice: Optional[str] = None,
          speed: Optional[float] = None, pitch: Optional[float] = None, markdown: bool = True, out=None,
          save: bool = True, wav_file: Optional[BinaryIO] = None, play: str = "off",
          player: Optional[Player] = None, wait_for_playback: bool = False, interrupt: bool = False,
          settings: Optional[dict] = None, allow_download: Optional[bool] = None,
          on_progress: Optional[Callable[[str, int, int], None]] = None,
          cancel: Optional[Callable[[], bool]] = None) -> SpeakResult:
    """Speaks `text` into one mono WAV.

    - `markdown` removes markup first (`clean_markdown`). Speaker tags and
      `[pause:x]` markers are never read out; this is one voice with no gaps.
    - `out` is a new `.wav` (`paths.check_out_path`). None makes
      `audio_output/headless/<stamp>_<slug>/<slug>.wav`. `save=False` writes
      neither. `wav_file` also gets the WAV's bytes.
    - `play` is "off", "after" or "stream". A given `player` stays open and only
      `wait_for_playback` waits for it; without one this waits. `interrupt` stops
      what the player has.
    - `host`, `engine`, `settings`, `allow_download`, `on_progress` (stage
      "generate") and `cancel` are `render_script`'s.

    Raises `HeadlessError` with `render_script`'s codes."""
    if not isinstance(text, str):
        raise _bad("the text must be text")
    if settings is not None and not isinstance(settings, dict):
        raise _bad("settings must be a mapping")
    if host is not None and settings is not None:
        raise _bad("pass settings to the host, not to a call on one")
    if out is not None and not save:
        raise _bad("pass out or save=False, not both")
    text = clean_markdown(text) if markdown else normalise_text(text)[0]
    text = strip_markup(text).strip()
    if not text:
        raise _bad("there is nothing to speak")
    out = _check_out(out)
    poll = _poll(cancel)
    playback = Playback(play, player, wait=wait_for_playback, interrupt=interrupt)

    own_host = host is None
    if own_host:
        host = Host(settings=settings, allow_download=allow_download)
    output = None
    done = False
    try:
        try:
            _stop_if_cancelled(poll)
            backend = host.backend(engine, allow_download=allow_download)
            chosen = _Voice(host, backend, voice, speed, pitch)
            if not spoken(text, chosen.lexicon).strip():
                raise _bad("there is nothing to speak")
            if save:
                output = _output_for(out, text)
            host.load(backend.id, allow_download=allow_download, cancel=poll)
            speech = _Speech(chosen, backend.sample_rate)
            config = chosen.config
            units = [text]
            if playback.streaming:
                config = {**config, "segment_target_words": STREAM_WORDS}
                units = split_segments(text, config)
            for index, unit in enumerate(units):
                _stop_if_cancelled(poll)
                before = _present()
                results = host.generate_clip(backend.id, (index, unit, config), allow_download=allow_download,
                                             cancel=poll)
                if not results and spoken(unit, chosen.lexicon).strip():
                    raise HeadlessError("generation_failed", f"{backend.display_name} made no audio for {unit[:60]!r}")
                for path, samples in speech.add(results, before):
                    if playback.streaming:
                        _queue(playback, chosen, path, samples, backend.sample_rate)
                _quietly(on_progress, "generate", index + 1, len(units))
            _stop_if_cancelled(poll)
            samples = speech.samples()
            path = _write(output, samples, backend.sample_rate) if output is not None else None
            if wav_file is not None:
                sf.write(wav_file, samples, backend.sample_rate, format="WAV", subtype="PCM_16")
            done = True
            if playback.active and not playback.streaming:
                scratch = None if path else scratch_path()
                if scratch:
                    sf.write(scratch, samples, backend.sample_rate)
                playback.add(path or scratch, delete=scratch is not None)
            played, reason = playback.finish(poll)
            return SpeakResult(path=path, duration_s=len(samples) / backend.sample_rate,
                               sample_rate=backend.sample_rate, engine=backend.id, voice=chosen.name,
                               words=speech.words, cached=speech.cached, played=played, reason=reason)
        except BaseException as e:
            playback.abort()
            if output is not None and not done:
                output.discard()
            if isinstance(e, HeadlessError) or not isinstance(e, (Exception, KeyboardInterrupt)):
                raise
            if isinstance(e, KeyboardInterrupt):
                raise HeadlessError("interrupted", "cancelled") from e
            raise HeadlessError("generation_failed", f"speaking failed: {str(e).strip() or type(e).__name__}") from e
    finally:
        post.clear_render_cache()
        if own_host:
            host.close()


def _output_for(out: Optional[str], text: str) -> Output:
    stem = paths.slugify(text[:paths.MAX_SLUG * 2], fallback="speech")
    try:
        output = Output(None, stem) if out is None else Output(os.path.dirname(out), "speech")
    except OSError as e:
        raise _bad(f"the output folder can't be made: {e}") from e
    if out is not None:
        output.mixdown_path = out
    if os.path.lexists(output.mixdown_path):
        output.discard()
        raise _bad(f"{output.mixdown_path} already exists")
    return output


def _write(output: Output, samples, rate: int) -> str:
    token = os.urandom(8).hex()
    part = os.path.join(output.dir, f".{token}.part")
    output.claim(token, part)
    sf.write(part, samples, rate, format="WAV", subtype="PCM_16")
    output.claim(token, output.mixdown_path)
    project_io._replace_with_retries(part, output.mixdown_path)
    return output.mixdown_path


def _queue(playback: Playback, voice: _Voice, path: str, samples, rate: int) -> None:
    """One segment, as it will sound: its own file when nothing changes it."""
    if voice.plain:
        playback.add(path)
        return
    scratch = scratch_path()
    sf.write(scratch, samples, rate)
    playback.add(scratch, delete=True)
