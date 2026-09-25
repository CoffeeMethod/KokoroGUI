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

Reads `kokoro_engine.CUSTOM_VOICES_DIR` qualified, at call time (not via
`from kokoro_engine import CUSTOM_VOICES_DIR`), so tests can keep
monkeypatching that name on the `kokoro_engine` module - same convention
`kokoro_gui/engine/voices.py` already uses.
"""
from __future__ import annotations

import os
from typing import Optional

import kokoro_engine
from kokoro_gui.engine.voices import project_voice_dir
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, VoiceInfo,
    COMMON_OUTPUT_FORMAT_CHOICES as OUTPUT_FORMAT_CHOICES, bundle_asset_for, segmentation_fields,
)
from kokoro_gui.engines.registry import register_engine


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
            *segmentation_fields(),
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
        """Custom voices: the open project's `engines/kokoro/voices/` first,
        then `CUSTOM_VOICES_DIR`; a name in both shows once and resolves to
        the project copy (grill TB3). The built-in named voices (af_heart,
        bm_daniel, ...) aren't listed here; see the `get_config_schema`
        docstring for why."""
        dirs = []
        if self.project_dir:
            dirs.append(project_voice_dir(self.project_dir))
        dirs.append(kokoro_engine.CUSTOM_VOICES_DIR)
        seen = []
        for directory in dirs:
            if not os.path.isdir(directory):
                continue
            for f in sorted(os.listdir(directory)):
                if f.endswith(".pt") and f[:-3] not in seen:
                    seen.append(f[:-3])
        return [VoiceInfo(id=name, display_name=name, lang_code=None, is_custom=True) for name in seen]

    def collect_project_assets(self, voice_names, project_dir=None) -> tuple:
        """The custom `.pt` mixes among `voice_names`, as
        `engines/kokoro/voices/<name>.pt`. Built-in voices resolve to no
        file and are skipped; so is a mix the user has deleted."""
        assets = []
        for name in sorted(voice_names):
            asset = bundle_asset_for(name, kokoro_engine.CUSTOM_VOICES_DIR, ".pt",
                                     "engines/kokoro/voices", project_dir)
            if asset is not None:
                assets.append(asset)
        return assets, {}

    async def mix_voices(self, v1_name: str, v2_name: str, ratio: float,
                          new_name: str, op: str = "mix"):
        """`SupportsVoiceMixing` extension - delegates straight to the
        wrapped engine's tensor math (kokoro_gui/engine/voices.py), which
        stays exactly where it is per the plan."""
        return await self._engine.mix_voices(v1_name, v2_name, ratio, new_name, op)

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("kokoro", KokoroBackendAdapter, display_name=KokoroBackendAdapter.display_name)
