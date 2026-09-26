"""The stand-in for an engine a project uses but this install doesn't have
(grill EN6): an engine id nothing is registered under, or one whose module
failed to import (`registry.unavailable_reason`).

`MissingBackend` answers everything the GUI asks of a backend without a
model: every capability off, no voices, no schema, an `engine_version()` of
None. Its clips play from the files they have and aren't stale while those
files exist (the dirty check compares the stored key, see
`kokoro_gui/daw/dirty.py`), and anything that would generate refuses with
`message`. The project keeps the characters' `backend_id`, so opening it
where the engine is installed brings everything back.
"""
from __future__ import annotations

import concurrent.futures
import threading
from typing import Optional

from kokoro_gui.engines.base import BackendHooksMixin, EngineCapabilities
from kokoro_gui.engines.registry import get_display_name


class MissingEngineError(RuntimeError):
    """Raised by a generate or preview on a missing engine."""


class _MissingEngine:
    """The `engine` side the app wires callbacks into and cancels: no
    worker thread, no pipeline. `worker.run_coro` closes the coroutine and
    returns a failed Future, so a caller that gets past the app's own checks
    sees the refusal instead of a hang."""

    def __init__(self, backend: "MissingBackend"):
        self._backend = backend
        self.pipeline = None
        self.cancel_event = threading.Event()
        self.on_progress = None
        self.on_status = None
        self.on_finish = None
        self.worker = self
        self.SAMPLE_RATE = 24000

    def run_coro(self, coro):
        if hasattr(coro, "close"):
            coro.close()
        future = concurrent.futures.Future()
        future.set_exception(MissingEngineError(self._backend.message))
        return future

    async def init_pipeline_async(self, lang_code=None, device=None):
        if self.on_status:
            self.on_status(self._backend.message, True)
        return False

    def _refuse(self, *_args, **_kwargs):
        raise MissingEngineError(self._backend.message)

    generate_clip_audio = generate_dirty_clips = generate_preview = _refuse
    start_conversion = start_jit_conversion = _refuse

    def engine_version(self):
        return None

    def cache_key_extra(self, config):
        return {}

    def resolve_voice_path(self, name, project_dir=None):
        return name

    def resolve_voice_file(self, name, project_dir=None):
        return None

    def cancel(self):
        self.cancel_event.set()

    def stop(self):
        pass


class MissingBackend(BackendHooksMixin):
    capabilities = EngineCapabilities(
        supports_voice_mixing=False, supports_voice_cloning=False, supports_multi_speaker_script=False,
        is_local_model=False, supports_jit_streaming=False, supports_word_timing=False,
        supports_speed=False,
    )

    def __init__(self, engine_id: str, reason: Optional[str] = None):
        self.id = engine_id
        self.name = get_display_name(engine_id)
        self.display_name = f"{self.name} (not installed)"
        self.reason = reason
        self.message = f"{self.name} isn't installed"
        self._engine = _MissingEngine(self)

    @property
    def engine(self):
        return self._engine

    def get_config_schema(self) -> list:
        return []

    def get_voices(self, lang_code=None) -> list:
        return []

    def engine_version(self):
        return None

    def cancel(self) -> None:
        self._engine.cancel()
