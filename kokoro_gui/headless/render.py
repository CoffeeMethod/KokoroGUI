"""A script becomes a `.tbaw` project and a WAV mixdown (plan section 6, H4).

    with Host(allow_download=False) as host:
        result = render_script("[Sam]: Hello.", host=host, cast={"Sam": "af_bella"})

`render_script` runs the steps the GUI runs, through the same functions:
`build_document`, a locked working folder (`begin_project_dir`), `plan_batch`,
`Host.generate_clips`, `apply_outcomes`, `compute_arrangement`, `export_mix`
and `save_project`. Every value comes from `SETTINGS_DEFAULTS` and the engine's
schema defaults. `config_qt.json`, the character library and the Lexicon are
not read; a host made with `settings` adds its engine buckets and its lexicon.
An FX preset is read only when the script or the cast names it.

Every failure is a `HeadlessError`. The mixdown is written under a private
name and gets its own once the bundle is saved, and a failure or a cancel takes
back every file and folder this call put at the output path. The working folder
stays (grill PG24), after a failure too, and its lock is always released.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Collection, Mapping
from dataclasses import replace
from typing import Callable, Optional

from kokoro_gui.audio import post
from kokoro_gui.core import config as core_config
from kokoro_gui.core import engines as core_engines
from kokoro_gui.core import fx as core_fx
from kokoro_gui.core import generation as core_generation
from kokoro_gui.core import keys as core_keys
from kokoro_gui.core import render as core_render
from kokoro_gui.daw import revision, wordalign
from kokoro_gui.daw.arrangement import compute_arrangement
from kokoro_gui.daw.dirty import apply_results, carry_segment_timing
from kokoro_gui.daw.mixdown import word_rows
from kokoro_gui.daw.share import clean_stem
from kokoro_gui.engines import registry
from kokoro_gui.qt import project as project_io
from kokoro_gui.qt import spec

from . import paths
from .clean import normalise_text
from .errors import HeadlessError
from .host import Host, download_gate, downloads_allowed
from .results import RenderResult
from .script import build_document

log = logging.getLogger(__name__)

FX_PRESETS_DIR = os.path.join("presets", "fx")
MAX_TITLE_CHARS = 200
# An `out` with one of these is a file name, and `out` is a folder.
_FILE_SUFFIXES = (".tbaw", ".json", ".wav", ".flac", ".mp3", ".ogg", ".m4b")


class _FxPresets(Collection):
    """The names of the FX presets in `directory`. Nothing is read until
    something is looked up. A name counts when it is a plain file stem, its file
    stays inside the folder and it holds preset values."""

    def __init__(self, directory: str):
        self.directory = directory

    def __contains__(self, name) -> bool:
        if clean_stem(name) is None or name not in self._stems():
            return False
        root = os.path.realpath(self.directory)
        path = os.path.realpath(os.path.join(root, f"{name}.json"))
        return os.path.dirname(path) == root and os.path.isfile(path) \
            and bool(core_fx.load_fx_preset_values(name, None, self.directory))

    def __iter__(self):
        return iter(sorted(n for n in self._stems() if n in self))

    def __len__(self) -> int:
        return sum(1 for _name in self)

    def _stems(self) -> list:
        try:
            return [f[:-5] for f in os.listdir(self.directory) if f.endswith(".json")]
        except OSError:
            return []


class _Output:
    """The files this call writes and the folders it makes. `discard` takes
    back exactly those; a file that was there first is never replaced."""

    def __init__(self, directory: Optional[str], stem: str):
        target = directory or paths.renders_root()
        self._made = [d for d in _with_parents(target) if not os.path.isdir(d)]
        self._files: list = []
        if directory is None:
            self.dir = paths.new_render_dir(stem)
            self._made.append(self.dir)
        else:
            os.makedirs(directory, exist_ok=True)
            self.dir = directory
        self.project_path = os.path.join(self.dir, stem + project_io.DEFAULT_EXTENSION)
        self.mixdown_path = os.path.join(self.dir, stem + ".wav")
        try:
            for path in (self.project_path, self.project_path + ".tmp", self.mixdown_path):
                self._refuse_existing(path)
        except HeadlessError:
            self.discard()
            raise

    @staticmethod
    def _refuse_existing(path: str) -> None:
        if os.path.lexists(path):
            raise HeadlessError("bad_arguments", f"{path} already exists")

    def claim(self, *names: str) -> None:
        """Called right before writing: another render may have taken the
        name since the first check."""
        for path in names:
            self._refuse_existing(path)
        self._files.extend(names)

    def discard(self) -> None:
        for path in self._files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                log.warning("couldn't remove %s", path)
        for directory in reversed(self._made):
            try:
                os.rmdir(directory)
            except OSError:  # gone, or someone else uses it
                pass


def _with_parents(path: str) -> list:
    """`path` and every folder above it, outermost first."""
    chain = [os.path.abspath(path)]
    while os.path.dirname(chain[-1]) != chain[-1]:
        chain.append(os.path.dirname(chain[-1]))
    return chain[::-1]


def _bad(message: str, valid=None) -> HeadlessError:
    return HeadlessError("bad_arguments", message, valid=valid)


def _quietly(hook, *args) -> None:
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001 - a caller's hook must not fail the render
        log.exception("a headless hook failed")


def _poll(cancel: Optional[Callable[[], bool]]) -> Callable[[], bool]:
    """`cancel` as a function that never raises. A hook that does counts as a
    cancel: the host has to see its job end before the working folder is
    unlocked, and an exception out of the hook would end the wait early."""
    def poll() -> bool:
        if cancel is None:
            return False
        try:
            return bool(cancel())
        except Exception:  # noqa: BLE001
            log.exception("the cancel hook failed")
            return True

    return poll


def _mapped(error: BaseException, stage: str) -> HeadlessError:
    if isinstance(error, HeadlessError):
        return error
    if isinstance(error, KeyboardInterrupt):
        return HeadlessError("interrupted", "cancelled")
    if isinstance(error, project_io.ProjectLockedError):
        return HeadlessError("project_locked", str(error))
    return HeadlessError("generation_failed", f"{stage} failed: {str(error).strip() or type(error).__name__}")


def _clean_title(title) -> str:
    if title is None:
        return ""
    if not isinstance(title, str):
        raise _bad("title must be text")
    return " ".join(normalise_text(title)[0].split())[:MAX_TITLE_CHARS]


def _check_out(out) -> Optional[str]:
    if out is None:
        return None
    real = paths.check_out_path(out)
    if real.lower().endswith(_FILE_SUFFIXES):
        raise _bad("out is a folder; the project and the mixdown are named in it")
    if os.path.lexists(real) and not os.path.isdir(real):
        raise _bad(f"out is not a folder: {real}")
    return real


def _cast_values(cast, key: str):
    """`(speaker, value)` for each cast entry that sets `key`. The cast has
    been through `build_document`, so each entry is a voice name or a mapping."""
    for name, entry in (cast or {}).items():
        value = entry.get(key) if isinstance(entry, Mapping) else (entry if key == "voice" else None)
        if value is not None:
            yield name, value


def _check_voices(cast, document, backend, lang_code) -> list:
    """Raises `unknown_voice` for a voice the engine doesn't list in any
    language, also for a cast entry no line uses. Returns a warning per speaker
    whose voice belongs to another language. A voice is a name from the list
    and never a path, so only the backend turns one into a file."""
    known = core_engines.voice_names(backend, None)
    used = [(c.name, (c.preset_data or {}).get("voice")) for c in document.characters]
    for name, voice in [*_cast_values(cast, "voice"), *used]:
        if voice not in known:
            what = f"no voice {voice!r}" if voice else "no default voice"
            raise HeadlessError("unknown_voice", f"{name!r}: {backend.display_name} has {what}", valid=known)
    in_language = core_engines.voice_names(backend, lang_code)
    return [f"Speaker {name!r}: {voice!r} is not a {lang_code!r} voice; it is read with that language's rules."
            for name, voice in used if voice not in in_language]


def _check_cast_fx(cast, known_fx: _FxPresets) -> None:
    for name, preset in _cast_values(cast, "fx_preset"):
        if preset not in known_fx:
            raise _bad(f"cast[{name!r}]: there is no FX preset {preset!r}", valid=list(known_fx))


def _share_timing(document) -> None:
    """Two paragraphs with the same text and settings share a file, and the
    second one is a cache hit with no words of its own."""
    segments = [s for clip in document.clips for s in clip.segments]
    carry_segment_timing(segments, [[s for s in segments if s.words], segments])


def _align_words(document, allow: bool, poll, on_progress, warnings: list) -> None:
    """Whisper word times for the segments whose engine gave none. Runs when
    the model is in the cache, or downloads are allowed."""
    todo = [s for clip in document.clips for s in clip.segments if s.audio_path and not s.words]
    if not todo:
        return
    from kokoro_gui.engine import asr

    errors = []
    with download_gate(allow):
        if not allow and not asr.whisper_model_cached():
            warnings.append("The words were not aligned: the Whisper model isn't downloaded and downloads are off.")
            return
        for done, segment in enumerate(todo, start=1):
            _stop_if_cancelled(poll)
            try:
                segment.words = wordalign.align(segment.text, asr.transcribe_wav_words(segment.audio_path, "whisper"))
            except Exception as e:  # noqa: BLE001 - one bad file leaves one segment without words
                errors.append(e)
            _quietly(on_progress, "align", done, len(todo))
    if errors:
        warnings.append(f"The words of {len(errors)} of {len(todo)} segment(s) were not aligned: {errors[0]}")


def _stop_if_cancelled(poll) -> None:
    if poll():
        raise HeadlessError("interrupted", "cancelled")


def _failed_clip(document, outcomes) -> Optional[HeadlessError]:
    """The failure to report for a batch, with the paragraph it belongs to: a
    missing download before anything else."""
    failed = [o for o in outcomes if not o["success"]]
    if not failed:
        return None
    worst = next((o for o in failed if o["failure"].code == "model_not_downloaded"), failed[0])
    failure = worst["failure"]
    if failure.code != "generation_failed":
        return failure
    number = next((i for i, c in enumerate(document.clips, start=1) if c.id == worst["clip_id"]), 0)
    error = HeadlessError(failure.code, f"paragraph {number} of {len(document.clips)}: {failure.message}")
    error.__cause__ = failure
    return error


class _Render:
    """One `render_script` call: what the stages share."""

    def __init__(self, host: Host, backend, document, warnings: list, *, title: str, poll, on_progress):
        self.host = host
        self.backend = backend
        self.document = document
        self.warnings = warnings
        self.poll = poll
        self.on_progress = on_progress
        self.project_settings = {"title": title} if title else {}
        self.project_dir = None
        self.project_id = None
        self.lock = None
        self.stage = "preparing"
        self._memo: dict = {}
        self._inputs = core_config.GenerationInputs.defaults()
        if host.settings.get("lexicon"):
            self._inputs = replace(self._inputs, lexicon=host.settings["lexicon"])
        self._output = core_config.OutputInputs.defaults()
        self._project_fx = {key: spec.SETTINGS_DEFAULTS[key] for key in spec.FX_PRESET_KEYS
                            if key in spec.SETTINGS_DEFAULTS}

    # -- configs ---------------------------------------------------------------

    def _backend_for(self, clip):
        return self.host.engines.for_clip(self.document, clip)

    def _engine_settings(self, backend) -> dict:
        return core_engines.engine_settings(self.host.settings, backend.id, self._memo)

    def _generation_config(self, clip) -> dict:
        backend = self._backend_for(clip)
        return core_config.generation_config(self._inputs, self._engine_settings(backend), self.document, clip,
                                             backend, self.project_dir, self._memo)

    def _clip_config(self, clip) -> dict:
        backend = self._backend_for(clip)
        fx = core_fx.resolve_fx(self.document, clip, project_fx=self._project_fx,
                                master_apply=spec.SETTINGS_DEFAULTS["apply_fx"], project_preset=None,
                                project_dir=self.project_dir, presets_dir=FX_PRESETS_DIR)
        return core_config.clip_config(self._inputs, self._output, self._engine_settings(backend), self.document,
                                       clip, backend, self.project_dir, self.project_settings, fx, self._memo)

    # -- stages ----------------------------------------------------------------

    def begin(self) -> None:
        self.stage = "making the working folder"

        def locked(project_dir, project_id, lock):
            # Before the first write, so a failed one still gets unlocked here.
            self.project_dir, self.project_id, self.lock = project_dir, project_id, lock

        project_io.begin_project_dir(self.document, self.project_settings, on_locked=locked)
        core_keys.install_key_fn(self.document, self._backend_for, self._generation_config,
                                 self._generation_config)

    def generate(self, allow_download) -> int:
        """Generates the stale clips and returns how many there were."""
        self.stage = "generating"
        _stop_if_cancelled(self.poll)
        batch = core_generation.plan_batch(self.document, self.document.dirty_clips(), self._backend_for,
                                           self._clip_config)
        total = len(batch.clips)
        done = []

        def progress(clip_id, _success):
            done.append(clip_id)
            _quietly(self.on_progress, "generate", len(done), total)

        for engine_id, (_backend, items) in batch.groups.items():
            self.host.load(engine_id, allow_download=allow_download, cancel=self.poll)
            outcomes = self.host.generate_clips(engine_id, items, progress=progress, partial=True,
                                                allow_download=allow_download, cancel=self.poll)
            core_generation.apply_outcomes(self.document, outcomes, self._apply)
            self.autosave()  # what did land, also when the rest failed
            failure = _failed_clip(self.document, outcomes)
            if failure is not None:
                raise failure
        _stop_if_cancelled(self.poll)
        _share_timing(self.document)
        return total

    @staticmethod
    def _apply(clip, results) -> None:
        revision.bump_files()  # generation wrote new segment files
        apply_results(clip, results)

    def arrange(self):
        self.stage = "arranging"
        rate = self.backend.sample_rate
        # Every clip has audio, so the guess for one without (which reads the
        # generation stats file) is never used.
        return compute_arrangement(
            self.document, engine_id=self.backend.id, chars_per_second=1.0,
            clip_duration=lambda clip: core_render.clip_duration_s(
                clip, core_config.post_config(self._clip_config(clip)), rate),
            clip_post_config=lambda clip: core_config.post_config(self._clip_config(clip)))

    def mix(self, arrangement, path: str, channels: int):
        self.stage = "mixing"

        def progress(fraction, _message):
            _stop_if_cancelled(self.poll)
            _quietly(self.on_progress, "mix", int(fraction * 100), 100)

        inputs = core_render.mix_inputs(arrangement, lambda clip: core_config.post_config(self._clip_config(clip)),
                                        None, self.backend.sample_rate, self.backend.id)
        options = core_render.export_options(
            {"format": "wav", "srt": False, "keep_clip_files": False, "channels": channels}, inputs,
            progress=progress)
        result = core_render.export_mix(self.document, path, options)
        if result.skipped_clip_ids:
            raise RuntimeError(f"{len(result.skipped_clip_ids)} clip(s) couldn't be read")
        self.warnings.extend(result.warnings)
        return result

    def autosave(self) -> None:
        """The working folder's `document.json` with the generated segments.
        Save records the folder as a clean extraction of the bundle, so Open
        may read this file instead of the bundle's."""
        project_io.autosave_to_dir(self.document, self.project_settings, self.project_dir)

    def save(self, path: str) -> None:
        self.stage = "saving"
        _stop_if_cancelled(self.poll)
        self.autosave()
        try:
            result = project_io.save_project(
                self.document, path, self.project_settings, self.project_dir, self.project_id,
                backend_for=self.host.engines.get, fx_presets_dir=FX_PRESETS_DIR,
                known_engine_ids=registry.list_engines(), project_fx=self._project_fx)
        except project_io.ProjectError as e:
            if isinstance(e.__context__, PermissionError):
                # Its message points at the `.tmp`, which `discard` removes.
                raise RuntimeError(f"couldn't write {path}: {e.__context__}") from e
            raise
        self.warnings.extend(result.warnings)
        _quietly(self.on_progress, "save", 1, 1)
        _stop_if_cancelled(self.poll)

    def failure(self, error: BaseException) -> HeadlessError:
        mapped = _mapped(error, self.stage)
        if self.project_dir is None:
            return mapped
        return HeadlessError(mapped.code, f"{mapped.message} (the working folder stays: {self.project_dir})",
                             valid=mapped.valid, exit_code=mapped.exit_code)


def render_script(script: str, *, host: Optional[Host] = None, engine: Optional[str] = None,
                  cast: Optional[Mapping] = None, out=None, title: Optional[str] = None, channels: int = 1,
                  align: bool = False, settings: Optional[dict] = None, allow_download: Optional[bool] = None,
                  on_progress: Optional[Callable[[str, int, int], None]] = None,
                  cancel: Optional[Callable[[], bool]] = None) -> RenderResult:
    """Renders `script` to `<slug>.tbaw` and `<slug>.wav` and returns where
    they are.

    - `host` runs the engines and stays open. Without one, a host is made from
      `settings` (a `config_qt.json` dict, or None for the defaults) and closed
      at the end. `engine` None is the host's default.
    - `cast` is `build_document`'s. Its voices and FX presets must exist. A tag's
      FX preset that doesn't is a warning.
    - `out` is a folder (`paths.check_out_path`), made when it's missing. None
      makes `audio_output/headless/<stamp>_<slug>/` (grill PG22). Existing files
      are not replaced. The slug is the title's, else the script's first words.
    - `channels` is 1 or 2 (grill PG23).
    - `align` adds Whisper word times where the engine gives none, when the
      model is in the cache or downloads are allowed.
    - `on_progress(stage, done, total)`: stages "generate", "align", "mix" and
      "save". `cancel()` is polled while the engine runs, per segment aligned, per
      clip mixed, and between the stages.

    Raises `HeadlessError`: `bad_arguments`, `out_not_allowed`,
    `engine_not_installed`, `unknown_voice`, `model_not_downloaded`,
    `project_locked`, `generation_failed` (mixing and saving too) or
    `interrupted`."""
    if not isinstance(script, str) or not script.strip():
        raise _bad("the script is empty" if isinstance(script, str) else "the script must be text")
    if isinstance(channels, bool) or channels not in (1, 2):
        raise _bad("channels must be 1 or 2", valid=["1", "2"])
    if settings is not None and not isinstance(settings, Mapping):
        raise _bad("settings must be a mapping")
    if host is not None and settings is not None:
        raise _bad("pass settings to the host, not to a render on one")
    title = _clean_title(title)
    out_dir = _check_out(out)
    poll = _poll(cancel)

    own_host = host is None
    if own_host:
        host = Host(settings=settings, allow_download=allow_download)
    render = output = None
    try:
        try:
            _stop_if_cancelled(poll)
            backend = host.backend(engine, allow_download=allow_download)
            engine_settings = core_engines.engine_settings(host.settings, backend.id)
            known_fx = _FxPresets(FX_PRESETS_DIR)
            try:
                document, warnings = build_document(script, backend, cast, engine_settings=engine_settings,
                                                    known_fx=known_fx)
            except ValueError as e:
                raise _bad(str(e)) from e
            if not document.clips:
                raise _bad("the script has nothing to speak")
            _stop_if_cancelled(poll)
            warnings += _check_voices(cast, document, backend, engine_settings.get("lang_code"))
            _check_cast_fx(cast, known_fx)

            render = _Render(host, backend, document, warnings, title=title, poll=poll, on_progress=on_progress)
            try:
                output = _Output(out_dir, paths.slugify(title or document.text[:paths.MAX_SLUG * 2]))
            except OSError as e:
                raise _bad(f"the output folder can't be made: {e}") from e
            # Before the working folder, so a model that isn't there leaves none.
            host.load(backend.id, allow_download=allow_download, cancel=poll)
            _stop_if_cancelled(poll)
            render.begin()
            regenerated = render.generate(allow_download)
            if align:
                render.stage = "aligning"
                allow = downloads_allowed(host.allow_download if allow_download is None else allow_download)
                _align_words(document, allow, poll, on_progress, warnings)
            arrangement = render.arrange()
            part = os.path.join(output.dir, f".{render.project_id}.part")
            output.claim(part)
            mixed = render.mix(arrangement, part, channels)
            output.claim(output.project_path, output.project_path + ".tmp")
            render.save(output.project_path)
            output.claim(output.mixdown_path)
            project_io._replace_with_retries(part, output.mixdown_path)
            return RenderResult(
                project_path=output.project_path, mixdown_path=output.mixdown_path, out_dir=output.dir,
                duration_s=mixed.duration_s, sample_rate=backend.sample_rate, channels=channels, engine=backend.id,
                clips=len(document.clips), regenerated=regenerated,
                speakers=[character.name for character in document.characters],
                words=[{"word": word, "start": round(start, 4), "end": round(end, 4)}
                       for start, end, word in word_rows(arrangement)],
                warnings=warnings)
        except BaseException as e:
            if output is not None:
                output.discard()
            if not isinstance(e, (Exception, KeyboardInterrupt)):
                raise
            failure = render.failure(e) if render is not None else _mapped(e, "preparing")
            if failure is e:
                raise
            raise failure from e
    finally:
        try:
            post.clear_render_cache()  # every clip's samples, which a long-lived process would keep
            if own_host:
                host.close()  # its jobs write into the working folder
        finally:
            if render is not None and render.lock is not None:
                render.lock.release()
