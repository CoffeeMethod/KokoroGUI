"""A test-only engine, loaded the way a third-party one is: through the
`kokorogui.engines` entry point (tests patch `importlib.metadata.entry_points`
to hand it out). One module, a model and an adapter, nothing under `qt/`,
`daw/` or `audio/`: the ENGINE_AGNOSTIC plan's "done when" for C6.

It clones (a "reference" engine, so it gets the Voice Reference editor and
bundles its refs) and speaks a flat 16kHz tone, 20 ms per word, so a test
can check what it generated without a model.
"""
from __future__ import annotations

import os

import numpy as np

from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, Synthesis, common_fields,
)
from kokoro_gui.engines.voice_store import ReferenceStore

ENGINE_ID = "toneclone"
SAMPLE_RATE = 16000
STORE = ReferenceStore(ENGINE_ID)  # custom_voices/toneclone_refs/


class ToneCloneModel(ModelBase):
    engine_id = ENGINE_ID
    sample_rate = SAMPLE_RATE
    concurrency = "shared"
    display_name = "ToneClone"

    def synthesize(self, text, voice, speed, lang_code, params):
        words = max(1, len((text or "").split()))
        return Synthesis(np.full(int(SAMPLE_RATE * 0.02 * words), 0.1, dtype=np.float32), [])

    def resolve_voice_path(self, name, project_dir=None):
        return STORE.find_wav(name, project_dir) or os.path.basename(name or "")

    def cache_key_extra(self, config):
        voice = config.get("voice") or ""
        wav = voice if os.path.isabs(voice) else STORE.find_wav(voice, config.get("project_dir"))
        return {"ref_transcript": STORE.read_transcript_file(os.path.splitext(wav)[0] + ".txt") if wav else ""}


class ToneCloneEngine(EngineRunner):
    def __init__(self):
        super().__init__(ToneCloneModel())


class ToneCloneAdapter(BackendHooksMixin):
    id = ENGINE_ID
    display_name = "ToneClone (test)"
    capabilities = EngineCapabilities(supports_voice_cloning=True, supports_jit_streaming=False)
    voice_kind = "reference"
    voice_store = STORE

    def __init__(self, engine=None):
        self._engine = engine if engine is not None else ToneCloneEngine()

    @property
    def engine(self):
        return self._engine

    @classmethod
    def get_config_schema(cls) -> list:
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="en", choices=[("English", "en")], group="Generation"),
            ConfigField("voice", "Voice Reference", ConfigFieldType.CHOICE, default=None, group="Generation"),
            *common_fields(pitch=False),
        ]

    @classmethod
    def make_contract_engine(cls):
        """The engine the contract tests run (no weights to fake here)."""
        return ToneCloneEngine()

    def cancel(self) -> None:
        self._engine.cancel()
