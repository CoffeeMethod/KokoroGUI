"""A from-scratch, non-Kokoro backend for exercising the engine-switching UI
without a real model - PLAN_qt_and_engine_abstraction.md workstream 1, step 5:
"consider a second backend (even a stub/fake one) to prove the abstraction
isn't over-fit to Kokoro".

`DummyModel` is a `SynthesisModel` (kokoro_gui/engines/base.py) that speaks a
short sine tone per piece, its pitch taken from the voice name so different
"voices" are at least audibly different, with evenly spaced word timings.
`DummyEngine` is the `EngineRunner` around it: every model-agnostic mixin
(FX, caching, conversion orchestration, JIT streaming, lexicon, presets,
SRT export, text extraction) unmodified. `CachingMixin` keys every entry on
`engine_id`, so a dummy tone can't collide with a Kokoro segment for the
same text; its schema still defaults `caching` off, since there's nothing
worth caching.

No voice mixing and no cloning, so the Voices tab has no editor for it (see
the Qt frontend's `_follow_active_for_voices`).
"""
from __future__ import annotations

import numpy as np

from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engine.wordtiming import even_tokens, words_from_tokens
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, Synthesis, VoiceInfo,
    common_fields,
)
from kokoro_gui.engines.registry import register_engine

SAMPLE_RATE = 24000

# Kokoro's language labels and codes, so a test can swap engines without its
# lang_code going out of range. Its own copy: Dummy must import without the
# `kokoro` package.
LANGUAGES = [
    ("American English", "a"), ("British English", "b"), ("Spanish", "e"), ("French", "f"),
    ("Italian", "i"), ("Portuguese", "p"), ("Japanese", "j"), ("Chinese", "z"),
]


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


class DummyModel(ModelBase):
    """No model, no weights, no eSpeak: a tone per piece, ready at once."""

    engine_id = "dummy"
    sample_rate = SAMPLE_RATE
    concurrency = "per_thread"
    display_name = "Dummy pipeline"

    def synthesize(self, text, voice, speed, lang_code, params):
        text = (text or "").strip()
        if not text:
            return Synthesis(np.zeros(0, dtype=np.float32), [])
        tone = _tone_for(text, speed, voice)
        # Evenly spaced words over the tone, so word timing has data to
        # show without a model.
        return Synthesis(tone, words_from_tokens(even_tokens(text, len(tone) / SAMPLE_RATE), 0.0))

    def resolve_voice_path(self, name, project_dir=None):
        # No custom-voice directory: a voice name only picks the tone's pitch.
        return name


class DummyEngine(EngineRunner):
    """The runner around `DummyModel`, ready without a load."""

    def __init__(self):
        super().__init__(DummyModel())
        self.pipeline = True  # no model to load - "ready" immediately


class DummyBackendAdapter(BackendHooksMixin):
    id = "dummy"
    display_name = "Dummy (offline test tone)"
    capabilities = EngineCapabilities(
        supports_voice_mixing=False,
        supports_voice_cloning=False,
        supports_multi_speaker_script=True,
        is_local_model=True,
        supports_jit_streaming=True,
        supports_word_timing=True,
    )

    def __init__(self, engine=None):
        """Same convention as `KokoroBackendAdapter`: wraps an existing
        `DummyEngine` when given (tests), otherwise builds its own - used
        when the GUI makes the backend resident for a character."""
        self._engine = engine if engine is not None else DummyEngine()

    @property
    def engine(self):
        return self._engine

    @classmethod
    def get_config_schema(cls) -> list:
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="a", choices=list(LANGUAGES), group="Generation"),
            ConfigField("voice", "Voice", ConfigFieldType.CHOICE,
                        default="dummy", group="Generation"),
            *common_fields(caching_default=False),
        ]

    def get_voices(self, lang_code=None) -> list:
        return [VoiceInfo(id="dummy", display_name="Dummy Tone", lang_code=None, is_custom=False)]

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("dummy", DummyBackendAdapter, display_name=DummyBackendAdapter.display_name)
