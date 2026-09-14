"""A from-scratch, non-Kokoro backend for exercising the engine-switching UI
without a real model - PLAN_qt_and_engine_abstraction.md workstream 1, step 5:
"consider a second backend (even a stub/fake one) to prove the abstraction
isn't over-fit to Kokoro".

`DummyEngine` reuses every mixin in `kokoro_gui/engine/` that turned out to be
genuinely model-agnostic (FX, caching, conversion orchestration, JIT
streaming, lexicon, presets, SRT export, text extraction) unmodified, and
only supplies its own `get_thread_pipeline` (a fake generator that yields
short sine-wave tones instead of real speech). `CachingMixin` keys every
entry on `engine_id`, so a dummy tone can't collide with a Kokoro segment
for the same text; its schema still defaults `caching` off, since there's
nothing worth caching.

No `VoiceMixingMixin` - `capabilities.supports_voice_mixing=False`, so the
Mixing dock is not shown while this backend is active (see the Qt frontend's
`kokoro_gui/qt/app.py`'s `_sync_mixing_dock`), demonstrating that gate
actually works.
"""
from __future__ import annotations

import re
import threading

import numpy as np

from kokoro_engine import AsyncLoopThread
from kokoro_gui.engine import (
    AudioFXMixin, CachingMixin, ConversionMixin, JITMixin, LexiconMixin, PresetsMixin,
    SrtMixin, TextExtractionMixin,
)
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, VoiceInfo,
    COMMON_SPLIT_PATTERN_CHOICES, COMMON_OUTPUT_FORMAT_CHOICES,
)
from kokoro_gui.engines.registry import register_engine

SAMPLE_RATE = 24000


class DummyPipeline:
    """Fakes `kokoro.KPipeline`'s callable-generator surface closely enough
    for the generic mixins to drive it: `pipeline(text, voice=, speed=,
    split_pattern=)` yields `(graphemes, phonemes, audio)` triples, `audio`
    a mono float32 ndarray at `SAMPLE_RATE`. No model, no weights, no
    eSpeak - a short sine tone stands in for speech, its pitch derived from
    the voice name so different "voices" are at least audibly different."""

    def __init__(self, lang_code="a"):
        self.lang_code = lang_code

    def __call__(self, text, voice="dummy", speed=1.0, split_pattern=r"\n+"):
        segments = [s.strip() for s in re.split(split_pattern, text) if s.strip()]
        if not segments and text.strip():
            segments = [text.strip()]
        for seg in segments:
            yield seg, "", _tone_for(seg, speed, voice)


def _tone_for(text, speed, voice):
    duration = max(0.3, min(4.0, len(text) * 0.05 / max(speed, 0.1)))
    n = max(1, int(duration * SAMPLE_RATE))
    t = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float32)
    freq = 220.0 + (abs(hash(voice)) % 400)  # different "voices" -> different pitch
    tone = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = min(200, n // 4)
    if fade > 0:
        env = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        tone = tone * env
    return tone


class DummyEngine(
    AudioFXMixin, CachingMixin, ConversionMixin, JITMixin, LexiconMixin, PresetsMixin,
    SrtMixin, TextExtractionMixin,
):
    """KokoroEngine-shaped enough for the GUI to drive directly (same
    `worker`/`cancel_event`/`pipeline`/`on_progress`/`on_status`/`on_finish`/
    `start_conversion`/`start_jit_conversion`/`generate_preview`/`cancel`
    surface), but with no real synthesis underneath."""

    id = "dummy"
    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self):
        self.worker = AsyncLoopThread()
        self.worker.start()
        self.cancel_event = threading.Event()
        self.pipeline = True  # no model to load - "ready" immediately

        self.on_progress = None
        self.on_status = None
        self.on_finish = None

        self._lexicon_cache = {}

    async def init_pipeline_async(self, lang_code="a", device=None):
        self.pipeline = True
        if self.on_status:
            self.on_status(f"Dummy pipeline ready ({lang_code}).", False)
        return True

    def get_thread_pipeline(self, lang_code="a"):
        return DummyPipeline(lang_code)

    def resolve_voice_path(self, voice_name, project_dir=None):
        # No custom-voice directory concept for the dummy backend - voice
        # names are just labels that pick a tone pitch (see _tone_for).
        return voice_name

    def cancel(self):
        self.cancel_event.set()


class DummyBackendAdapter(BackendHooksMixin):
    id = "dummy"
    display_name = "Dummy (offline test tone)"
    capabilities = EngineCapabilities(
        supports_voice_mixing=False,
        supports_voice_cloning=False,
        supports_multi_speaker_script=True,
        is_local_model=True,
        supports_jit_streaming=True,
    )

    def __init__(self, engine=None):
        """Same convention as `KokoroBackendAdapter`: wraps an existing
        `DummyEngine` when given (tests), otherwise builds its own - used
        when the GUI switches its active backend at runtime."""
        self._engine = engine if engine is not None else DummyEngine()

    @property
    def engine(self):
        return self._engine

    def get_config_schema(self) -> list:
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="a", group="Generation"),
            ConfigField("voice", "Voice", ConfigFieldType.CHOICE,
                        default="dummy", group="Generation"),
            ConfigField("speed", "Speed", ConfigFieldType.SLIDER,
                        default=1.0, min=0.5, max=2.0, step=0.1, group="Generation"),
            ConfigField("pitch", "Pitch", ConfigFieldType.SLIDER,
                        default=0.0, min=-12, max=12, step=1, group="Audio"),
            ConfigField("split_pattern", "Split By", ConfigFieldType.CHOICE,
                        default=r"\n+", choices=list(COMMON_SPLIT_PATTERN_CHOICES), group="Generation"),
            ConfigField("format", "Output Format", ConfigFieldType.CHOICE,
                        default="wav", choices=list(COMMON_OUTPUT_FORMAT_CHOICES), group="Generation"),
            ConfigField("num_threads", "Parallel Threads", ConfigFieldType.INT,
                        default=1, min=1, max=32, step=1, group="Advanced"),
            ConfigField("caching", "Enable Segment Cache", ConfigFieldType.BOOL,
                        default=False, group="Advanced"),
        ]

    def get_voices(self, lang_code=None) -> list:
        return [VoiceInfo(id="dummy", display_name="Dummy Tone", lang_code=None, is_custom=False)]

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("dummy", DummyBackendAdapter, display_name=DummyBackendAdapter.display_name)
