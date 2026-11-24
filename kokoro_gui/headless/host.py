"""The engines a headless process runs, and the gate that keeps them from
downloading weights.

`Host` owns an `EngineSet`. Its `on_new_backend` points an adapter's callbacks
at plain functions (no Qt signals) and wraps the adapter's `run`, the one way a
job reaches an engine's worker. Every job therefore holds the download gate for
as long as it runs, also one started with `backend.generate_clip(...)` directly
and one whose caller stopped waiting.

    with Host(allow_download=False, on_status=print) as host:
        host.load("kokoro")
        results = host.generate_clip("kokoro", (0, text, config))

Failures are `HeadlessError`s with the original exception as `__cause__`:
`engine_not_installed`, `model_not_downloaded`, `interrupted` (the cancel hook,
`cancel()`, `close()`), and `generation_failed` for everything else.

The gate. Downloads are off unless `allow_download=True` or
`KOKOROGUI_ALLOW_DOWNLOAD=1`. `HF_HUB_OFFLINE=1` alone does nothing once
huggingface_hub is imported, and the Kokoro adapter imports it before any load:
huggingface_hub copies the variable into `constants.HF_HUB_OFFLINE` at import
and keeps one HTTP session per thread, and transformers copies that constant
into `utils.hub._is_offline_mode` at its own import. So while the gate is shut:

- huggingface_hub's HTTP backend is one whose sessions refuse every request.
  This is what stops Kokoro, LuxTTS and any other `hf_hub_download` or
  `snapshot_download`; a file in the cache is still found, since the hub falls
  back to the cache when a request raises `OfflineModeIsEnabled`.
- `constants.HF_HUB_OFFLINE` and transformers' `_is_offline_mode` are set.
  Without transformers' flag a cached tokenizer with over 100,000 tokens
  (Audio8's) fails: it asks the hub for the model's tags and doesn't catch the
  refusal.
- `spacy.cli.download`, which Kokoro's English G2P calls when
  `en_core_web_sm` isn't installed, refuses. It would run pip.
- `HF_HUB_OFFLINE` is in the environment, for child processes.

All of it is process-wide, so jobs that allow downloads and jobs that don't
never overlap: one kind waits for the other to end. Opening the gate puts every
value back. An allowed job leaves a `HF_HUB_OFFLINE` the user set alone.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import os
import re
import sys
import threading
from typing import Callable, Optional
from urllib.parse import unquote, urlsplit

from kokoro_gui.core import engines as core_engines
from kokoro_gui.engines import registry

from .errors import HeadlessError

log = logging.getLogger(__name__)

ALLOW_DOWNLOAD_ENV = "KOKOROGUI_ALLOW_DOWNLOAD"
OFFLINE_ENV = "HF_HUB_OFFLINE"
POLL_S = 0.05

_WEIGHT_FILES = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt", ".index.json")
_RESOLVE_URL = re.compile(r"([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")


def downloads_allowed(allow: Optional[bool] = None) -> bool:
    """`allow` when the caller said, else whether `KOKOROGUI_ALLOW_DOWNLOAD`
    is `1`. Any other value of the variable means no."""
    if allow is not None:
        return bool(allow)
    return os.environ.get(ALLOW_DOWNLOAD_ENV, "").strip() == "1"


class DownloadRefused(RuntimeError):
    """A download that doesn't go through huggingface_hub, refused by the gate."""


class _Misses:
    """What one piece of work looked for and didn't find. `files` can't be
    anything but required. `maybe` is what transformers looked for, some of
    which a repo is free not to have."""

    def __init__(self):
        self.files: list = []
        self.maybe: list = []

    def weights(self) -> list:
        return [name for name in self.maybe if name.endswith(_WEIGHT_FILES)]


# Both travel with a job: the worker's task and the threads it starts run in a
# copy of the context the job was submitted from.
_JOB: contextvars.ContextVar = contextvars.ContextVar("kokorogui_headless_job", default=None)
_MISSES: contextvars.ContextVar = contextvars.ContextVar("kokorogui_headless_misses", default=None)
# The job a Host method is about to submit, read by the wrapped `run`.
_NEXT_JOB: contextvars.ContextVar = contextvars.ContextVar("kokorogui_headless_next_job", default=None)
# The gate mode this context holds (True = downloads allowed).
_HELD: contextvars.ContextVar = contextvars.ContextVar("kokorogui_headless_gate", default=None)


def _refused(url: str) -> None:
    """Notes the file behind a refused hub URL when the cache lacks it."""
    misses = _MISSES.get()
    match = _RESOLVE_URL.search(urlsplit(url).path)
    if misses is None or match is None:
        return
    from huggingface_hub import try_to_load_from_cache

    repo, revision, filename = (unquote(part) for part in match.groups())
    if not isinstance(try_to_load_from_cache(repo, filename, revision=revision), str):
        misses.files.append(filename)


def _refusing_session():
    """A `requests` session for huggingface_hub that sends nothing. It raises
    what the hub's own offline adapter raises, which the hub takes as "look in
    the cache"."""
    import requests
    from huggingface_hub.errors import OfflineModeIsEnabled

    class RefusingAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, *args, **kwargs):
            _refused(request.url)
            raise OfflineModeIsEnabled(f"Cannot reach {request.url}: downloads are off.")

    session = requests.Session()
    session.mount("http://", RefusingAdapter())
    session.mount("https://", RefusingAdapter())
    return session


def _refuse_spacy(model, *args, **kwargs):
    misses = _MISSES.get()
    if misses is not None:
        misses.files.append(str(model))
    raise DownloadRefused(f"spaCy's {model} isn't installed and downloads are off")


def _noting_misses(download):
    """transformers' `hf_hub_download`, noting a file the cache lacks: it
    swallows the miss and can fail later with an unrelated `AttributeError`."""
    @functools.wraps(download)
    def wrapped(repo_id, filename, *args, **kwargs):
        try:
            return download(repo_id, filename, *args, **kwargs)
        except FileNotFoundError:  # LocalEntryNotFoundError is one
            misses = _MISSES.get()
            if misses is not None:
                misses.maybe.append(filename)
            raise

    return wrapped


class _Gate:
    """Process-wide. Any number of jobs of one kind hold it at once, and the
    other kind waits. Released by whichever thread a job ends on."""

    def __init__(self):
        self._cond = threading.Condition()
        self._allow = None
        self._holders = 0
        self._undo: list = []
        self._patched: set = set()
        self._baseline = False

    def acquire(self, allow: bool, stopped: Optional[Callable[[], bool]] = None) -> None:
        held = _HELD.get()
        if held is not None and held != allow:
            raise RuntimeError("a download gate can't be nested inside one of the other kind")
        with self._cond:
            while self._holders and self._allow != allow:
                if stopped is not None and stopped():
                    raise HeadlessError("interrupted", "cancelled while waiting for another job")
                self._cond.wait(POLL_S)
            if not allow:
                try:
                    self._shut()
                except BaseException:
                    if not self._holders:
                        self._open()
                    raise
            self._allow = allow
            self._holders += 1

    def release(self) -> None:
        with self._cond:
            self._holders -= 1
            if self._holders == 0:
                if not self._allow:
                    self._open()
                self._allow = None
            self._cond.notify_all()

    def _set(self, obj, name: str, value) -> None:
        if (id(obj), name) in self._patched:
            return
        old = getattr(obj, name)
        setattr(obj, name, value)
        self._patched.add((id(obj), name))
        self._undo.append(lambda: setattr(obj, name, old))

    def _shut(self) -> None:
        """Runs for every holder: a module imported since the last one needs
        its own patch."""
        if not self._undo:
            self._baseline = False
            # Before the variable is set, so the constant's value is the user's.
            try:
                from huggingface_hub import constants
                from huggingface_hub.utils import _http
            except ImportError:
                constants = _http = None
            # Undone in reverse, so the backend goes back (which also drops the
            # refusing sessions) after the constant.
            if _http is not None and hasattr(_http, "configure_http_backend"):
                old_factory = _http._GLOBAL_BACKEND_FACTORY
                _http.configure_http_backend(_refusing_session)
                self._undo.append(lambda: _http.configure_http_backend(old_factory))
            old_env = os.environ.get(OFFLINE_ENV)
            os.environ[OFFLINE_ENV] = "1"
            self._undo.append(lambda: self._restore_env(old_env))
            if constants is not None:
                self._baseline = constants.HF_HUB_OFFLINE
                self._set(constants, "HF_HUB_OFFLINE", True)
        hub = sys.modules.get("transformers.utils.hub")
        if hub is not None and hasattr(hub, "_is_offline_mode"):
            self._set(hub, "_is_offline_mode", True)
            if hasattr(hub, "hf_hub_download"):
                self._set(hub, "hf_hub_download", _noting_misses(hub.hf_hub_download))
        spacy_cli = sys.modules.get("spacy.cli")
        if spacy_cli is not None and callable(getattr(spacy_cli, "download", None)):
            self._set(spacy_cli, "download", _refuse_spacy)

    @staticmethod
    def _restore_env(old: Optional[str]) -> None:
        if old is None:
            os.environ.pop(OFFLINE_ENV, None)
        else:
            os.environ[OFFLINE_ENV] = old

    def _open(self) -> None:
        hub = sys.modules.get("transformers.utils.hub")
        if hub is not None and (id(hub), "_is_offline_mode") not in self._patched and hasattr(hub, "_is_offline_mode"):
            # Imported while the gate was shut, so it copied our value.
            hub._is_offline_mode = self._baseline
        while self._undo:
            self._undo.pop()()
        self._patched.clear()


_GATE = _Gate()


@contextlib.contextmanager
def download_gate(allow: Optional[bool] = None):
    """Holds the gate for work that isn't an engine's job (a Whisper
    transcription). A no-op besides the waiting when downloads are allowed."""
    allow = downloads_allowed(allow)
    _GATE.acquire(allow)
    token = _HELD.set(allow)
    try:
        yield
    finally:
        _HELD.reset(token)
        _GATE.release()


class _Job:
    """One coroutine on an engine's worker. `done` is set after the gate is
    released, so a caller that waited for it sees the process as it was."""

    def __init__(self, allow: bool, *, load: bool = False, on_status=None):
        self.allow = allow
        self.load = load
        self.on_status = on_status
        self.poll = None             # the caller's cancel hook
        self.backend = None
        self.future = None
        self.coros = ()
        self.holds_gate = False
        self.cancelled = threading.Event()
        self.stopping = False
        self.done = threading.Event()
        self.finishing = False
        self.at_end: list = []      # func(job), before `done`
        self.misses = _Misses()
        self.error = None           # what the engine was handling when it reported an error
        self.error_text = ""
        self.clip_errors: dict = {}  # a clip's index -> (exception, its _Misses)


async def _contain(coro):
    """A `SystemExit` (spaCy's downloader calls `sys.exit`) would end the
    worker's loop and leave every Future on it pending."""
    try:
        return await coro
    except (SystemExit, KeyboardInterrupt) as e:
        raise RuntimeError(f"the engine tried to exit ({type(e).__name__}: {e})") from e


def _chain(error) -> list:
    """`error` and everything it was raised from or while handling."""
    seen = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _in_hub_cache(path) -> bool:
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is None or not isinstance(path, (str, os.PathLike)):
        return False
    path, cache = (os.path.normcase(os.path.abspath(p)) for p in (path, constants.HF_HUB_CACHE))
    try:
        return os.path.commonpath([path, cache]) == cache
    except ValueError:  # another drive
        return False


def _not_downloaded(error, misses: _Misses, allow: bool) -> bool:
    """Whether `error` means a file that has to be downloaded isn't there."""
    chain = _chain(error)
    hub_errors = sys.modules.get("huggingface_hub.errors")
    if hub_errors is not None:
        types = (hub_errors.LocalEntryNotFoundError, hub_errors.OfflineModeIsEnabled)
        if any(isinstance(e, types) for e in chain):
            return True
    if any(isinstance(e, DownloadRefused) for e in chain):
        return True
    for e in chain:
        # `snapshot_download` returns a half-downloaded snapshot (LuxTTS), and
        # opening what it lacks fails with the cache path. onnxruntime has its
        # own error for it.
        if isinstance(e, FileNotFoundError) and _in_hub_cache(e.filename):
            return True
        if type(e).__name__ == "NoSuchFile" and "models--" in str(e):
            return True
    if allow:
        return False
    if misses.files:
        # Kokoro's per-thread pipeline prints its error and returns None.
        return True
    if any(isinstance(e, OSError) and "does not appear to have" in str(e) for e in chain):
        return True
    # transformers with a cached config and no weights raises AttributeError
    # (it indexes a None path), and its `pipeline()` a ValueError with the
    # causes as text.
    return bool(misses.weights()) and isinstance(
        chain[-1], (AttributeError, OSError, TypeError, ValueError))


def _text(error) -> str:
    return str(error).strip() or type(error).__name__


def _failure(error, misses: _Misses, name: str, allow: bool, text: str = "") -> HeadlessError:
    """The `HeadlessError` for `error` (None when the engine only said it
    failed), with `error` as its cause."""
    if isinstance(error, HeadlessError):
        return error
    if error is not None and _not_downloaded(error, misses, allow):
        if allow:
            message = f"{name} needs files that couldn't be downloaded: {_text(error)}"
        else:
            files = list(dict.fromkeys(misses.files or misses.weights()))
            listed = f" ({', '.join(files[:3])}{', ...' if len(files) > 3 else ''})" if files else ""
            message = (f"{name} needs files that aren't downloaded{listed}, and downloads are off. "
                       f"Allow them with --allow-download or {ALLOW_DOWNLOAD_ENV}=1.")
        failure = HeadlessError("model_not_downloaded", message)
    else:
        failure = HeadlessError("generation_failed", text or (_text(error) if error else f"{name} failed"))
    failure.__cause__ = error
    return failure


def _interrupted(message: str = "cancelled") -> HeadlessError:
    return HeadlessError("interrupted", message)


def _quietly(hook, *args) -> None:
    """Calls a caller's hook. One that raises must not fail the job."""
    if hook is None:
        return
    try:
        hook(*args)
    except Exception:  # noqa: BLE001
        log.exception("a headless hook failed")


class Host:
    """The engines of one headless process.

    `allow_download` None follows `KOKOROGUI_ALLOW_DOWNLOAD`, and each call can
    override it. `settings` is a `config_qt.json` dict (`--settings`) or None
    for the schema defaults. `on_status(engine_id, message, is_error)` and
    `on_progress(engine_id, percent, elapsed, eta, detail)` are called on an
    engine's threads. A `cancel` argument is a function polled while the call
    waits (`threading.Event().is_set` works).

    Thread-safe. `engines` is the `EngineSet`, for `core`'s `backend_for_clip`
    callbacks; jobs started on its adapters go through the gate with the
    host's own setting.
    """

    def __init__(self, default_engine_id: str = registry.DEFAULT_ENGINE_ID, *,
                 allow_download: Optional[bool] = None, device: Optional[str] = None,
                 settings: Optional[dict] = None, on_status=None, on_progress=None):
        self.allow_download = allow_download
        self.settings = dict(settings or {})
        self.device = device or self.settings.get("device", "auto")
        self.on_status = on_status
        self.on_progress = on_progress
        self.engines = core_engines.EngineSet(default_engine_id, registry.DEFAULT_ENGINE_ID, self._on_new_backend)
        self._lock = threading.RLock()
        self._build_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._jobs: set = set()
        self._loaded: dict = {}      # engine id -> (lang_code, device)
        self._load_locks: dict = {}
        self._closing = False

    def __enter__(self) -> "Host":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- engines ---------------------------------------------------------------

    def backend(self, engine_id: Optional[str] = None, *, allow_download: Optional[bool] = None):
        """The resident adapter for `engine_id` (None is the default engine),
        built without loading a model. Raises `engine_not_installed`, with the
        installed ids in `valid`."""
        return self._backend(engine_id, self._allow(allow_download), None)

    def _backend(self, engine_id: Optional[str], allow: bool, cancel):
        engine_id = engine_id or self.engines.default_engine_id
        if self._closing:
            raise _interrupted("the host is closed")
        backend = self.engines.resident.get(engine_id)
        if backend is not None:
            return backend
        with self._build_lock:
            if self._closing:
                raise _interrupted("the host is closed")
            _GATE.acquire(allow, lambda: self._closing or bool(cancel and cancel()))
            try:
                return self._build(engine_id, allow)
            finally:
                _GATE.release()

    def _build(self, engine_id: str, allow: bool):
        backend = self.engines.get(engine_id)
        if backend is not None:
            return backend
        if not registry.is_registered(engine_id):
            reason = registry.unavailable_reason(engine_id)
            known = engine_id in registry.list_all_engines()
            message = f"{registry.get_display_name(engine_id)} isn't installed" if known else \
                f"there is no engine called {engine_id!r}"
            raise HeadlessError("engine_not_installed", f"{message} ({reason})" if reason else message,
                                valid=registry.list_engines())
        # Registered, but its constructor failed, and `EngineSet` drops the
        # error. Ask once more for it.
        try:
            backend = registry.get_engine(engine_id)
        except Exception as e:  # noqa: BLE001 - mapped
            raise _failure(e, _Misses(), registry.get_display_name(engine_id), allow)
        self.engines.resident[backend.id] = backend
        self._on_new_backend(backend)
        return backend

    def _on_new_backend(self, backend) -> None:
        """`EngineSet`'s callback. Callbacks without Qt signals, every job
        through the gate, and a clip's own exception kept: a batch's outcome
        has only its text."""
        backend.set_callbacks(on_status=functools.partial(self._status, backend.id),
                              on_progress=functools.partial(self._progress, backend.id))
        backend.run = functools.partial(self._run, backend, backend.run)
        generate = getattr(backend.engine, "generate_clip_audio", None)
        if asyncio.iscoroutinefunction(generate):
            backend.engine.generate_clip_audio = functools.partial(_noting_clip_error, generate)

    def _status(self, engine_id: str, message: str, is_error: bool = False) -> None:
        job = _JOB.get()
        if is_error and job is not None:
            # `EngineRunner.init_pipeline_async` reports from inside its
            # `except` and then returns False, so this is the only place the
            # exception itself can be had.
            job.error = sys.exc_info()[1]
            job.error_text = str(message)
        hook = (job.on_status if job is not None else None) or self.on_status
        _quietly(hook, engine_id, message, bool(is_error))

    def _progress(self, engine_id: str, *values) -> None:
        _quietly(self.on_progress, engine_id, *values)

    # -- jobs ------------------------------------------------------------------

    def load(self, engine_id: Optional[str] = None, *, lang_code: Optional[str] = None,
             device: Optional[str] = None, allow_download: Optional[bool] = None,
             cancel: Optional[Callable[[], bool]] = None, on_status=None):
        """Loads the engine's model and returns its adapter. A second call with
        the same language and device returns at once.

        A model that is loading can't be stopped. `cancel` ends the wait with
        `interrupted`, and the load finishes behind it, gate and all; the next
        `load` and `close()` wait for that."""
        allow = self._allow(allow_download)
        backend = self._backend(engine_id, allow, cancel)
        if lang_code is None:
            lang_code = core_engines.engine_settings(self.settings, backend.id).get("lang_code")
        device = device or self.device
        with self._lock:
            lock = self._load_locks.setdefault(backend.id, threading.Lock())
        while not lock.acquire(timeout=POLL_S):
            if self._closing or (cancel is not None and cancel()):
                raise _interrupted("cancelled while another load was running")
        if self._loaded.get(backend.id) == (lang_code, device) and backend.is_ready():
            lock.release()
            return backend
        job = _Job(allow, load=True, on_status=on_status)
        job.at_end.append(lambda j: self._note_load(backend, j, (lang_code, device)))
        job.at_end.append(lambda j: lock.release())
        self._submit(job, lambda: backend.ensure_ready(lang_code, device=device), cancel)
        result = self._wait(job, cancel)
        if result is False:
            raise _failure(job.error, job.misses, backend.display_name, job.allow, job.error_text)
        return backend

    def generate_clip(self, engine_id: Optional[str], chunk, *, allow_download: Optional[bool] = None,
                      cancel: Optional[Callable[[], bool]] = None) -> list:
        """One clip's `(index, text, config)` through `backend.generate_clip`."""
        allow = self._allow(allow_download)
        backend = self._backend(engine_id, allow, cancel)
        job = _Job(allow)
        self._submit(job, lambda: backend.generate_clip(chunk), cancel)
        results = self._wait(job, cancel)
        if not results and backend.was_cancelled():
            raise _interrupted()
        return results

    def generate_clips(self, engine_id: Optional[str], items, *, progress=None, partial: bool = False,
                       allow_download: Optional[bool] = None,
                       cancel: Optional[Callable[[], bool]] = None) -> list:
        """`[(clip_id, text, config), ...]` through `backend.generate_clips`;
        one outcome dict per clip. A failed one also has `failure`, its
        `HeadlessError`. Raises the first failure, a missing download before
        any other, unless `partial`. `progress(clip_id, success)` is called
        on the engine's thread."""
        allow = self._allow(allow_download)
        backend = self._backend(engine_id, allow, cancel)
        items = list(items)
        job = _Job(allow)
        with self._lock:
            # `generate_dirty_clips` checks the event a cancelled job left set
            # before anything clears it.
            event = getattr(backend.engine, "cancel_event", None)
            if event is not None and not any(j.backend is backend and not j.load for j in self._jobs):
                event.clear()
        self._submit(job, lambda: backend.generate_clips(items, progress=functools.partial(_quietly, progress)), cancel)
        outcomes = self._wait(job, cancel)
        failures = []
        for index, outcome in enumerate(outcomes):
            if outcome["success"]:
                continue
            error, misses = job.clip_errors.get(index, (None, job.misses))
            if outcome.get("cancelled") or (error is None and backend.was_cancelled()):
                outcome["failure"] = _interrupted()
            else:
                outcome["failure"] = _failure(error, misses, backend.display_name, job.allow, outcome.get("error") or "")
            failures.append(outcome["failure"])
        if failures and not partial:
            raise next((f for f in failures if f.code == "model_not_downloaded"), failures[0])
        return outcomes

    def cancel(self, engine_id: Optional[str] = None) -> None:
        """Cancels what is running (on one engine, or on all). The calls
        waiting for it raise `interrupted`. An engine has one cancel event, so
        its other jobs stop too."""
        with self._lock:
            jobs = [j for j in self._jobs if engine_id is None or j.backend.id == engine_id]
        for job in jobs:
            job.cancelled.set()
            self._request_stop(job)

    def close(self) -> None:
        """Cancels and waits for every job, then stops and joins each engine's
        worker thread and its pool. Safe to call again, from any thread. A load
        can't be cancelled, so this waits for one that is running."""
        self._closing = True
        with self._close_lock:
            while True:
                with self._lock:
                    jobs = list(self._jobs)
                if not jobs:
                    break
                for job in jobs:
                    self._request_stop(job)
                    job.done.wait(POLL_S)
                    self._check_worker(job)
            with self._build_lock:
                for engine_id, backend in list(self.engines.resident.items()):
                    _stop_backend(backend)
                    self.engines.resident.pop(engine_id, None)
                self._loaded.clear()

    # -- internals -------------------------------------------------------------

    def _allow(self, allow: Optional[bool]) -> bool:
        return downloads_allowed(self.allow_download if allow is None else allow)

    def _submit(self, job: _Job, start, cancel) -> None:
        """Runs `start` (a `backend.*` call that ends in `backend.run`) as `job`."""
        job.poll = cancel
        token = _NEXT_JOB.set(job)
        try:
            start()
            if job.future is None:
                raise HeadlessError("generation_failed", "the engine doesn't start its jobs through run()")
        except BaseException:
            self._finish(job)  # when it failed before `_run` could
            raise
        finally:
            _NEXT_JOB.reset(token)

    def _run(self, backend, run, coro):
        """`backend.run`, holding the gate until the coroutine has ended. The
        gate belongs to the job, not to whoever waits for it: a load nobody
        waits for any more must not start downloading."""
        job = _NEXT_JOB.get()
        if job is None or job.backend is not None:
            job = _Job(self._allow(None))
        job.backend = backend
        try:
            with self._lock:
                if self._closing:
                    raise _interrupted("the host is closed")
                self._jobs.add(job)
            _GATE.acquire(job.allow, lambda: self._closing or job.cancelled.is_set() or bool(job.poll and job.poll()))
            job.holds_gate = True
            context = contextvars.copy_context()
            for var, value in ((_JOB, job), (_MISSES, job.misses), (_HELD, job.allow), (_NEXT_JOB, None)):
                context.run(var.set, value)
            contained = _contain(coro)
            job.coros = (contained, coro)
            try:
                job.future = context.run(run, contained)
            except BaseException:
                contained.close()
                raise
        except BaseException:
            coro.close()
            self._finish(job)
            raise
        job.future.add_done_callback(lambda _future: self._finish(job))
        return job.future

    def _finish(self, job: _Job) -> None:
        with self._lock:
            if job.finishing:
                return
            job.finishing = True
        try:
            for func in job.at_end:
                func(job)
        finally:
            if job.holds_gate:
                _GATE.release()
            with self._lock:
                self._jobs.discard(job)
            job.done.set()

    def _note_load(self, backend, job: _Job, key) -> None:
        future = job.future
        ok = future is not None and future.done() and not future.cancelled() \
            and future.exception() is None and future.result() is not False
        if ok:
            self._loaded[backend.id] = key
        else:
            self._loaded.pop(backend.id, None)

    def _request_stop(self, job: _Job) -> None:
        """Again on every poll: `generate_clip_audio` clears the cancel event
        when it starts. `backend.cancel()` only once, since Kokoro's also stops
        playback."""
        if job.load or job.backend is None:
            return
        if not job.stopping:
            job.stopping = True
            job.backend.cancel()
            return
        event = getattr(job.backend.engine, "cancel_event", None)
        if event is not None:
            event.set()

    def _check_worker(self, job: _Job) -> bool:
        """Ends a job whose worker thread has died, since its Future never
        will. True when it did."""
        worker = getattr(job.backend.engine, "worker", None) if job.backend is not None else None
        is_alive = getattr(worker, "is_alive", None)
        if job.done.is_set() or is_alive is None or is_alive() or job.future is None or job.future.done():
            return False
        for coro in job.coros:
            if inspect.getcoroutinestate(coro) == inspect.CORO_CREATED:
                coro.close()
        self._finish(job)
        return True

    def _wait(self, job: _Job, cancel):
        """`job`'s result, or the `HeadlessError` for its failure."""
        name = job.backend.display_name
        try:
            while not job.done.wait(POLL_S):
                if self._closing or (cancel is not None and cancel()):
                    job.cancelled.set()
                if job.cancelled.is_set():
                    if job.load:
                        raise _interrupted("cancelled while a model was loading")
                    self._request_stop(job)
                if self._check_worker(job):
                    raise HeadlessError("generation_failed", f"{name}'s worker thread ended")
        except HeadlessError:
            raise
        except BaseException:  # Ctrl+C, or a cancel hook that raised
            job.cancelled.set()
            self._request_stop(job)
            raise
        if job.cancelled.is_set() or job.future.cancelled():
            raise _interrupted()
        error = job.future.exception()
        if error is None:
            return job.future.result()
        misses = next((m for e, m in job.clip_errors.values() if e is error), job.misses)
        raise _failure(error, misses, name, job.allow)


async def _noting_clip_error(generate, chunk_data, *args, **kwargs):
    """`generate_clip_audio` with its own miss list, and its exception kept on
    the job by the clip's index."""
    misses = _Misses()
    token = _MISSES.set(misses)
    try:
        return await _contain(generate(chunk_data, *args, **kwargs))
    except Exception as e:
        job = _JOB.get()
        if job is not None:
            job.clip_errors[chunk_data[0]] = (e, misses)
        raise
    finally:
        _MISSES.reset(token)


def _stop_backend(backend) -> None:
    """Joins the threads `asyncio.to_thread` started on the worker's loop,
    then the worker. `AsyncLoopThread.stop` joins only itself."""
    worker = getattr(backend.engine, "worker", None)
    loop = getattr(worker, "loop", None)
    try:
        backend.set_callbacks()
        if isinstance(worker, threading.Thread) and loop is not None:
            if worker.is_alive():
                asyncio.run_coroutine_threadsafe(loop.shutdown_default_executor(), loop).result()
            else:
                executor = getattr(loop, "_default_executor", None)
                if executor is not None:
                    executor.shutdown(wait=True)
        backend.stop()
        if isinstance(worker, threading.Thread) and not worker.is_alive() and loop is not None:
            loop.close()
    except Exception:  # noqa: BLE001 - the other engines still have to stop
        log.exception("stopping %s failed", backend.id)
