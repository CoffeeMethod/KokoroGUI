"""Adapts the existing `KokoroEngine` to the `TTSEngineBackend` surface
(kokoro_gui/engines/base.py).

Composition, not rewrite: `KokoroBackendAdapter` wraps a `KokoroEngine`
instance built and driven exactly as before - `kokoro_engine.py`'s
`AsyncLoopThread`/thread-pool internals and the GUI's callback wiring
(`on_progress`/`on_status`/`on_finish`) are untouched. This module changes no
behavior; it only describes that existing surface through the schema/
capabilities contract so a schema-driven GUI panel and, eventually, a second
backend have something concrete to target (PLAN_qt_and_engine_abstraction.md
workstream 1).

An "embedding" engine: its custom mixes are `KOKORO_VOICES`
(kokoro_gui/engine/voices.py), which gives `get_voices` and the bundle its
files. Imports `kokoro_engine` (and so `kokoro`) at the top:
`kokoro_gui/engines/__init__.py` catches the ImportError and marks the
engine unavailable.
"""
from __future__ import annotations

from typing import Optional

import kokoro_engine
from kokoro_gui.engine.voices import KOKORO_VOICES
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, VoiceInfo,
    COMMON_OUTPUT_FORMAT_CHOICES as OUTPUT_FORMAT_CHOICES,  # noqa: F401 - tests import it from here
    common_fields,
)
from kokoro_gui.engines.registry import register_engine

# Kokoro's languages as `(label, lang_code)`, in the order the GUI lists them.
LANGUAGES = [
    ("American English", "a"),
    ("British English", "b"),
    ("Spanish", "e"),
    ("French", "f"),
    ("Italian", "i"),
    ("Portuguese", "p"),
    ("Japanese", "j"),
    ("Chinese", "z"),
]

# The voices built into the model, per lang_code.
VOICE_DB = {
    "a": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa"],
    "b": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex", "em_santa"],
    "f": ["ff_siwis"],
    "i": ["if_sara", "im_nicola"],
    "p": ["pf_dora", "pm_alex"],
    "j": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"],
    "z": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunjian"],
}

# The Mixing dock's preview sentence per lang_code.
MIX_PREVIEW_TEXT = {
    "f": "Ceci est un aperçu de votre voix personnalisée.",
    "e": "Esta es una vista previa de su voz personalizada.",
    "i": "Questa è un'anteprima della tua voce personalizzata.",
    "p": "Esta é uma prévia da sua voz personalizada.",
    "j": "これはカスタム合成音声のプレビューです。",
    "z": "这是您的自定义混合语音预览。",
}
MIX_PREVIEW_TEXT_DEFAULT = "This is a preview of your custom mixed voice."


class KokoroBackendAdapter(BackendHooksMixin):
    id = "kokoro"
    display_name = "Kokoro (local)"
    capabilities = EngineCapabilities(
        supports_voice_mixing=True,
        supports_voice_cloning=False,
        supports_multi_speaker_script=True,
        is_local_model=True,
        supports_jit_streaming=True,
        supports_word_timing=True,
    )

    voice_kind = "embedding"
    voice_store = KOKORO_VOICES

    def __init__(self, engine=None):
        """`engine`, when given, is an existing `KokoroEngine` instance the
        adapter wraps rather than constructing its own (used by the GUI at
        startup and by tests). When omitted, the adapter builds a fresh
        `KokoroEngine()` itself - used when the GUI makes a backend
        resident on first use (the Qt frontend's `_backend_for`), where
        nothing already owns an engine instance to hand in."""
        self._engine = engine if engine is not None else kokoro_engine.KokoroEngine()

    @property
    def engine(self):
        """The wrapped `KokoroEngine` instance - the GUI re-points
        `self.engine` at this on every backend switch so the many existing
        `self.engine.*` call sites keep working unchanged."""
        return self._engine

    @classmethod
    def get_config_schema(cls) -> list:
        """Today's `KokoroEngine` config-dict fields (per CLAUDE.md: "Config
        dicts, not typed objects" - this schema describes that dict, it
        doesn't replace it). "voice" leaves `choices=None`: the GUI asks
        `get_voices(lang_code)`, which lists the built-ins for the language
        and the custom mixes on disk."""
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="a", choices=list(LANGUAGES), group="Generation"),
            ConfigField("voice", "Voice", ConfigFieldType.CHOICE,
                        default="af_heart", group="Generation"),
            *common_fields(),
            ConfigField("lexicon", "Lexicon Substitutions", ConfigFieldType.TEXT,
                        default={}, group="Advanced"),
        ]

    def builtin_voices(self, lang_code: Optional[str] = None) -> list:
        """The voices built into the model for `lang_code` (every
        language's when None). The custom `.pt` mixes follow them in
        `get_voices` (`KOKORO_VOICES`, the project's first, grill TB3)."""
        if lang_code is None:
            pairs = [(code, name) for code, names in VOICE_DB.items() for name in names]
        else:
            pairs = [(lang_code, name) for name in VOICE_DB.get(lang_code, [])]
        return [VoiceInfo(id=name, display_name=name, lang_code=code, is_custom=False) for code, name in pairs]

    def preview_text(self, lang_code: Optional[str] = None) -> str:
        return MIX_PREVIEW_TEXT.get(lang_code, MIX_PREVIEW_TEXT_DEFAULT)

    def word_timing_for(self, lang_code: Optional[str]) -> bool:
        """KPipeline yields token timings for English only."""
        return super().word_timing_for(lang_code) and lang_code in ("a", "b")

    @classmethod
    def package_version(cls) -> str:
        """The installed `kokoro` package version: what a Kokoro segment
        key carries (`caching.get_engine_version`)."""
        return kokoro_engine.kokoro_package_version()

    async def mix_voices(self, v1_name: str, v2_name: str, ratio: float,
                          new_name: str, op: str = "mix"):
        """`SupportsVoiceMixing` extension - delegates straight to the
        wrapped engine's tensor math (kokoro_gui/engine/voices.py), which
        stays exactly where it is per the plan."""
        return await self._engine.mix_voices(v1_name, v2_name, ratio, new_name, op)

    async def preview_mix(self, tensor, voice_name: str, text: str, output_path: str, lang_code: str):
        """Speaks `text` with an unsaved mix `tensor`, registered as
        `voice_name`, into `output_path` (the Mixing dock's Preview)."""
        return await self._engine.generate_preview(text, voice_name, 1.0, output_path, voice_tensor=tensor,
                                                   lang_code=lang_code)

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("kokoro", KokoroBackendAdapter, display_name=KokoroBackendAdapter.display_name)
