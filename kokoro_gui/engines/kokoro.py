"""Adapts the existing `KokoroEngine` to the `TTSEngineBackend` surface
(kokoro_gui/engines/base.py).

Composition, not rewrite: `KokoroBackendAdapter` wraps a `KokoroEngine`
instance built and driven exactly as before - `kokoro_engine.py`'s
`AsyncLoopThread`/thread-pool internals, and `gui.py`'s callback wiring
(`on_progress`/`on_status`/`on_finish`) are untouched. This module changes no
behavior; it only describes that existing surface through the schema/
capabilities contract so a schema-driven GUI panel and, eventually, a second
backend have something concrete to target (PLAN_qt_and_engine_abstraction.md
workstream 1).

Reads `kokoro_engine.CUSTOM_VOICES_DIR` qualified, at call time (not via
`from kokoro_engine import CUSTOM_VOICES_DIR`), so tests can keep
monkeypatching that name on the `kokoro_engine` module - same convention
`kokoro_gui/engine/voices.py` already uses.
"""
from __future__ import annotations

import os
from typing import Optional

import kokoro_engine
from kokoro_gui.engines.base import ConfigField, ConfigFieldType, EngineCapabilities, VoiceInfo
from kokoro_gui.engines.registry import register_engine

# Single source of truth for the split-pattern presets the Generation tab
# offers - moved here (out of kokoro_gui/ui/generation_tab.py's widget-
# building code) so the GUI consumes it from the schema instead of hard-
# coding it a second time. Keys are what the GUI shows; values are what
# KokoroEngine actually splits on.
SPLIT_PATTERN_CHOICES = [
    ("Natural (Newlines)", r"\n+"),
    ("Paragraphs (Double Newline)", r"\n\n+"),
    ("Sentences (.!?)", r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"),
]

OUTPUT_FORMAT_CHOICES = [("wav", "wav"), ("flac", "flac"), ("mp3", "mp3"), ("ogg", "ogg")]


class KokoroBackendAdapter:
    id = "kokoro"
    display_name = "Kokoro (local)"
    capabilities = EngineCapabilities(
        supports_voice_mixing=True,
        supports_voice_cloning=False,
        supports_multi_speaker_script=True,
        is_local_model=True,
        supports_jit_streaming=True,
    )

    def __init__(self, engine):
        """`engine` is an existing `KokoroEngine` instance - the adapter
        never constructs its own; it wraps whichever one the caller (GUI or
        test) already owns and drives."""
        self._engine = engine

    def get_config_schema(self) -> list:
        """Reflects today's actual KokoroEngine config-dict fields (per
        CLAUDE.md: "Config dicts, not typed objects" - this schema describes
        that dict, it doesn't replace it).

        "voice" and "lang_code" deliberately leave `choices=None`: the voice
        catalog (`TTSApp.VOICE_DB`/`LANGUAGES`) is still GUI-owned display
        data as of this workstream, not engine data - `get_voices()` below
        only covers the part of the catalog that *is* genuinely engine/
        filesystem state (custom voice files). Migrating the built-in voice
        table itself behind the adapter is follow-on work, not required to
        make the Generation tab's other fields (split pattern, format,
        speed) schema-driven.
        """
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="a", group="Generation"),
            ConfigField("voice", "Voice", ConfigFieldType.CHOICE,
                        default="af_heart", group="Generation"),
            ConfigField("speed", "Speed", ConfigFieldType.SLIDER,
                        default=1.0, min=0.5, max=2.0, step=0.1, group="Generation"),
            ConfigField("pitch", "Pitch", ConfigFieldType.SLIDER,
                        default=0.0, min=-12, max=12, step=1, group="Audio"),
            ConfigField("split_pattern", "Split By", ConfigFieldType.CHOICE,
                        default=r"\n+", choices=list(SPLIT_PATTERN_CHOICES), group="Generation"),
            ConfigField("format", "Output Format", ConfigFieldType.CHOICE,
                        default="wav", choices=list(OUTPUT_FORMAT_CHOICES), group="Generation"),
            ConfigField("num_threads", "Parallel Threads", ConfigFieldType.INT,
                        default=1, min=1, max=32, step=1, group="Advanced"),
            ConfigField("caching", "Enable Segment Cache", ConfigFieldType.BOOL,
                        default=True, group="Advanced"),
            ConfigField("lexicon", "Lexicon Substitutions", ConfigFieldType.TEXT,
                        default={}, group="Advanced"),
        ]

    def get_voices(self, lang_code: Optional[str] = None) -> list:
        """Custom voices discovered under `CUSTOM_VOICES_DIR` - the built-in
        named voices (af_heart, bm_daniel, ...) aren't listed here; see the
        `get_config_schema` docstring for why."""
        custom_dir = kokoro_engine.CUSTOM_VOICES_DIR
        if not os.path.isdir(custom_dir):
            return []
        return [
            VoiceInfo(id=f[:-3], display_name=f[:-3], lang_code=None, is_custom=True)
            for f in sorted(os.listdir(custom_dir))
            if f.endswith(".pt")
        ]

    async def mix_voices(self, v1_name: str, v2_name: str, ratio: float,
                          new_name: str, op: str = "mix"):
        """`SupportsVoiceMixing` extension - delegates straight to the
        wrapped engine's tensor math (kokoro_gui/engine/voices.py), which
        stays exactly where it is per the plan."""
        return await self._engine.mix_voices(v1_name, v2_name, ratio, new_name, op)

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("kokoro", KokoroBackendAdapter, display_name=KokoroBackendAdapter.display_name)
