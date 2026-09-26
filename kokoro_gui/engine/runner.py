"""`EngineRunner`: everything an engine does that isn't the model.

A backend used to be a full engine class (`KokoroEngine`, `DummyEngine`,
`Audio8Engine`), each repeating the worker thread, the cancel event, the
callbacks and the mixins around its own KPipeline-shaped callable. Now the
model is a `SynthesisModel` (kokoro_gui/engines/base.py): load, synthesize
one piece, say its version and cache inputs. The runner wraps one with the
model-agnostic mixins (FX, caching, conversion, JIT, lexicon, presets, SRT,
text extraction), and `process_chunk_task` and `generate_preview` call
`_synthesize`, which honours the model's `concurrency`: "per_thread" calls
straight through (the model keeps its per-thread state, as Kokoro's
thread-local `KPipeline` does), "shared" serializes calls through a lock.

`ModelBase` holds the defaults a model inherits: no extra cache inputs, the
package version from the registry, a voice name that resolves to its own
basename, nothing to load.
"""
from __future__ import annotations

import asyncio
import os
import threading
from typing import Optional

from kokoro_gui.engine import caching, runtime
from kokoro_gui.engine.audio_fx import AudioFXMixin
from kokoro_gui.engine.caching import CachingMixin
from kokoro_gui.engine.conversion import ConversionMixin
from kokoro_gui.engine.jit import JITMixin
from kokoro_gui.engine.lexicon import LexiconMixin
from kokoro_gui.engine.presets import PresetsMixin
from kokoro_gui.engine.srt import SrtMixin
from kokoro_gui.engine.text_extraction import TextExtractionMixin


class ModelBase:
    """Defaults for a `SynthesisModel`. A model sets `engine_id`,
    `sample_rate` and `concurrency` and implements `synthesize`."""

    engine_id = "unknown"
    sample_rate = 24000
    concurrency = "per_thread"
    display_name = "Engine"

    def load(self, lang_code: Optional[str], device: Optional[str]):
        return True

    def loading_message(self) -> Optional[str]:
        """Status shown before `load` runs (a download can take minutes)."""
        return None

    def ready_message(self, lang_code: Optional[str]) -> str:
        return f"{self.display_name} ready ({lang_code})."

    def load_error(self, error: Exception, lang_code: Optional[str]) -> str:
        return f"{self.display_name} failed to load: {error}"

    def synthesize(self, text, voice, speed, lang_code, params):
        raise NotImplementedError

    def engine_version(self) -> str:
        # Looked up on the module at call time so a test can patch it.
        return caching.get_engine_version(self.engine_id)

    def cache_key_extra(self, config: dict) -> dict:
        return {}

    def resolve_voice_path(self, name, project_dir=None):
        return os.path.basename(name) if name else name


class EngineRunner(
    AudioFXMixin, CachingMixin, ConversionMixin, JITMixin, LexiconMixin, PresetsMixin,
    SrtMixin, TextExtractionMixin,
):
    """One engine: a `SynthesisModel` and its own asyncio worker thread. The
    adapter in `kokoro_gui/engines/` wraps one of these for the GUI."""

    def __init__(self, model):
        self.model = model
        self.id = model.engine_id
        self.SAMPLE_RATE = model.sample_rate
        self.worker = runtime.AsyncLoopThread()
        self.worker.start()
        self.cancel_event = threading.Event()
        # The readiness token `load` returned (Kokoro's main KPipeline, which
        # voice mixing uses); falsy until the model has loaded.
        self.pipeline = None
        self._shared_lock = threading.Lock()

        # Callbacks
        self.on_progress = None  # func(percentage, time_elapsed, eta, detail_text)
        self.on_status = None    # func(msg, is_error)
        self.on_finish = None    # func()

        self._lexicon_cache = {}  # compiled lexicon regexes

    # -- the model's answers, for the segment key -------------------------------

    def engine_version(self):
        return self.model.engine_version()

    def cache_key_extra(self, config):
        return self.model.cache_key_extra(config)

    def resolve_voice_path(self, voice_name, project_dir=None):
        return self.model.resolve_voice_path(voice_name, project_dir)

    # -- loading and synthesis ---------------------------------------------------

    async def init_pipeline_async(self, lang_code=None, device=None):
        message = self.model.loading_message()
        if message and self.on_status:
            self.on_status(message, False)
        try:
            self.pipeline = await asyncio.to_thread(self.model.load, lang_code, device)
        except Exception as e:  # noqa: BLE001 - reported, never raised into the worker
            self.pipeline = None
            if self.on_status:
                self.on_status(self.model.load_error(e, lang_code), True)
            return False
        if self.on_status:
            self.on_status(self.model.ready_message(lang_code), False)
        return True

    def _synthesize(self, text, config, cancellable=True):
        """`model.synthesize` over one piece with `config`'s voice, speed and
        language (the caller has already applied pitch compensation to
        `speed`), serialized when the model is "shared". A cancellable call
        gets the cancel event in its params."""
        params = dict(config)
        if cancellable:
            params["cancel_event"] = self.cancel_event
        args = (text, config.get("voice"), config.get("speed", 1.0), config.get("lang_code"), params)
        if getattr(self.model, "concurrency", "per_thread") == "shared":
            with self._shared_lock:
                return self.model.synthesize(*args)
        return self.model.synthesize(*args)

    def cancel(self):
        self.cancel_event.set()
