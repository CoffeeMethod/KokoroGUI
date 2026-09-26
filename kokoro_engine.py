"""Kokoro's engine: `KokoroModel` (a thread-local `kokoro.KPipeline` as a
`SynthesisModel`), `KokoroEngine` (the `EngineRunner` around it, plus voice
mixing) and the pipeline getter tests patch.

The shared storage dirs, `playback` and `AsyncLoopThread` live in
`kokoro_gui/engine/runtime.py`, so the other backends import without the
`kokoro` package. The old names still resolve here (module `__getattr__`),
read from `runtime` at call time; patch them there.
"""
import importlib.metadata
import threading
from kokoro import KPipeline

import numpy as np

from kokoro_gui.engine import runtime
from kokoro_gui.engine.caching import to_numpy
from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engine.runtime import AsyncLoopThread  # noqa: F401 - old import path
from kokoro_gui.engine.voices import VoiceMixingMixin
from kokoro_gui.engine.wordtiming import words_from_tokens
from kokoro_gui.engines.base import Synthesis

_RUNTIME_NAMES = ("CUSTOM_VOICES_DIR", "CACHE_DIR", "STATS_FILE", "playback")


def __getattr__(name):
    """The names that moved to `runtime`, read from there at call time."""
    if name in _RUNTIME_NAMES:
        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def kokoro_package_version():
    """The installed `kokoro` package's version, "unknown" without one."""
    try:
        return importlib.metadata.version("kokoro")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# --- Thread Local Storage ---
thread_local = threading.local()

# Options > Device in the Qt shell writes settings["device"] ("auto" | "cpu" |
# "cuda"); init_pipeline_async() copies it here so every worker thread's
# KPipeline lands on the same device. None means "let kokoro pick".
PIPELINE_DEVICE = None


def _pipeline_kwargs():
    return {"device": PIPELINE_DEVICE} if PIPELINE_DEVICE else {}


def get_thread_pipeline(lang_code="a"):
    """Get or create a KPipeline instance for the current thread."""
    current = getattr(thread_local, "pipeline", None)
    if current is None or getattr(current, "lang_code", None) != lang_code:
        try:
            thread_local.pipeline = KPipeline(lang_code=lang_code, **_pipeline_kwargs())
        except Exception as e:
            print(f"Error init pipeline in thread {threading.get_ident()}: {e}")
            return None
    return thread_local.pipeline

class KokoroModel(ModelBase):
    """Kokoro as a `SynthesisModel`: one `KPipeline` per worker thread and
    language (`get_thread_pipeline`, called by name so tests can patch it),
    24000Hz, word timings from the pipeline's tokens."""

    engine_id = "kokoro"
    sample_rate = 24000
    concurrency = "per_thread"
    display_name = "Kokoro"

    def __init__(self):
        self.loaded_lang = None

    def load(self, lang_code, device):
        """The main pipeline (voice mixing loads tensors through it). A
        `lang_code` left over from another engine ("English") that KPipeline
        rejects gets one retry with Kokoro's "a" rather than surfacing
        KPipeline's raw AssertionError; "a" failing is a real problem and
        raises."""
        global PIPELINE_DEVICE
        if device is not None:
            PIPELINE_DEVICE = None if device == "auto" else device
        lang_code = lang_code or "a"
        try:
            pipeline = KPipeline(lang_code=lang_code, **_pipeline_kwargs())
        except Exception:
            if lang_code == "a":
                raise
            pipeline = KPipeline(lang_code="a", **_pipeline_kwargs())
            lang_code = "a"
        self.loaded_lang = lang_code
        return pipeline

    def ready_message(self, lang_code):
        return f"Pipeline Initialized ({self.loaded_lang or lang_code})."

    def load_error(self, error, lang_code):
        msg = f"Pipeline Init Failed: {error}"
        err_str = str(error).lower()
        if lang_code == "j" and ("fugashi" in err_str or "unidic" in err_str):
            msg += "\n(Try: pip install fugashi unidic-lite)"
        elif lang_code == "z" and "pypinyin" in err_str:
            msg += "\n(Try: pip install pypinyin)"
        return msg

    def synthesize(self, text, voice, speed, lang_code, params):
        """Every result KPipeline yields for `text`, concatenated; a result
        after the first starts where the audio so far ends, and its tokens'
        times are offset to match. A `voice_tensor` in `params` (a mix being
        previewed) is registered under `voice` first."""
        pipeline = get_thread_pipeline(lang_code) if lang_code else get_thread_pipeline()
        if not pipeline:
            raise RuntimeError(f"Failed to initialize pipeline ({lang_code}) in thread.")
        if params.get("voice_tensor") is not None:
            pipeline.voices[voice] = params["voice_tensor"]
        cancel_event = params.get("cancel_event")
        arrays, words, frames = [], [], 0
        for item in pipeline(text, voice=voice, speed=speed, split_pattern=None):
            if cancel_event is not None and cancel_event.is_set():
                break
            _graphemes, _phonemes, audio = item
            if audio is None:
                continue
            audio = np.asarray(to_numpy(audio), dtype=np.float32).reshape(-1)
            words.extend(words_from_tokens(getattr(item, "tokens", None), frames / float(self.sample_rate)))
            arrays.append(audio)
            frames += len(audio)
        if not arrays:
            return Synthesis(np.zeros(0, dtype=np.float32), [])
        return Synthesis(arrays[0] if len(arrays) == 1 else np.concatenate(arrays), words)


class KokoroEngine(VoiceMixingMixin, EngineRunner):
    """The runner around `KokoroModel`, plus `.pt` voice mixing and custom
    voice resolution (`VoiceMixingMixin`)."""

    def __init__(self):
        # Private (0o700) on POSIX; see runtime.prepare_storage.
        runtime.prepare_storage()
        super().__init__(KokoroModel())

    def get_thread_pipeline(self, lang_code="a"):
        """This thread's KPipeline for `lang_code` (the module function, by
        name, so a test's patch takes effect)."""
        return get_thread_pipeline(lang_code)

    def cancel(self):
        self.cancel_event.set()
        try:
            # Stop any current playback immediately
            runtime.playback.stop()
        except Exception:
            pass
