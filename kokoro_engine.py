import os
import threading
import asyncio
import time
import pypdf
import ebooklib
from ebooklib import epub
import warnings
import playback
from kokoro import KPipeline

from kokoro_gui.engine import (
    AudioFXMixin, CachingMixin, ConversionMixin, JITMixin, LexiconMixin,
    PresetsMixin, SrtMixin, TextExtractionMixin, VoiceMixingMixin,
)

# Suppress ebooklib warnings
warnings.filterwarnings("ignore", category=UserWarning, module='ebooklib')
warnings.filterwarnings("ignore", category=FutureWarning, module='ebooklib')

CUSTOM_VOICES_DIR = "custom_voices"
CACHE_DIR = "cache"
STATS_FILE = "generation_stats.json"  # per-engine generation-history, see kokoro_gui/engine/stats.py

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

class AsyncLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        self.running = True

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()

    def run_coro(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class KokoroEngine(
    AudioFXMixin, CachingMixin, ConversionMixin, JITMixin, LexiconMixin,
    PresetsMixin, SrtMixin, TextExtractionMixin, VoiceMixingMixin,
):
    def __init__(self):
        self.worker = AsyncLoopThread()
        self.worker.start()
        self.cancel_event = threading.Event()
        self.pipeline = None # Main pipeline for single thread check or init

        if not os.path.exists(CUSTOM_VOICES_DIR):
            os.makedirs(CUSTOM_VOICES_DIR)

        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        # Callbacks
        self.on_progress = None # func(percentage, time_elapsed, eta, detail_text)
        self.on_status = None   # func(msg, is_error)
        self.on_finish = None   # func()

        self._lexicon_cache = {} # Cache for compiled regexes

    def get_thread_pipeline(self, lang_code="a"):
        """Instance-method indirection to the module-level thread-local
        KPipeline getter, so the generic mixins (caching.py, conversion.py)
        can call `self.get_thread_pipeline(...)` polymorphically instead of
        hard-coding `kokoro_engine.get_thread_pipeline` - the one piece of
        that shared pipeline that's genuinely Kokoro-specific (see
        kokoro_gui/engines/dummy.py for a from-scratch, non-Kokoro backend
        built on the same generic mixins). Calls the free function by name
        (not a direct reference) so `monkeypatch.setattr(kokoro_engine,
        "get_thread_pipeline", ...)` in tests still takes effect."""
        return get_thread_pipeline(lang_code)

    async def init_pipeline_async(self, lang_code="a", device=None):
        global PIPELINE_DEVICE
        if device is not None:
            PIPELINE_DEVICE = None if device == "auto" else device
        try:
            try:
                self.pipeline = await asyncio.to_thread(KPipeline, lang_code=lang_code, **_pipeline_kwargs())
            except Exception:
                if lang_code == "a":
                    raise
                # A lang_code value left over from a different engine
                # backend (e.g. Audio8 stores full language names like
                # "English", not Kokoro's single-letter codes) can reach
                # here despite the Settings dock's own combo-fallback
                # reconciliation (kokoro_gui/qt/docks/settings_dock.py's
                # SchemaFormWidget._set_combo) - KPipeline itself rejects it
                # with a raw AssertionError against its own internal
                # LANG_CODES table. Retry once with Kokoro's own safe
                # default rather than surface that to the user; if "a"
                # itself fails (a real problem - missing model, no network,
                # etc.), let that failure propagate normally below.
                self.pipeline = await asyncio.to_thread(KPipeline, lang_code="a", **_pipeline_kwargs())
                lang_code = "a"
            if self.on_status: self.on_status(f"Pipeline Initialized ({lang_code}).", False)
            return True
        except Exception as e:
            msg = f"Pipeline Init Failed: {e}"
            err_str = str(e).lower()
            if lang_code == 'j' and ("fugashi" in err_str or "unidic" in err_str):
                 msg += "\n(Try: pip install fugashi unidic-lite)"
            elif lang_code == 'z' and "pypinyin" in err_str:
                 msg += "\n(Try: pip install pypinyin)"

            if self.on_status: self.on_status(msg, True)
            return False

    def cancel(self):
        self.cancel_event.set()
        try:
            # Stop any current playback immediately
            playback.stop()
        except Exception:
            pass
