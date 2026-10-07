"""Optional LuxTTS backend. Importing this module downloads and loads nothing.

The runner serializes synthesis; the model lock also protects device reloads
and the bounded, in-memory reference cache. Derived prompt tensors are never
written into a project or loaded from untrusted files.
"""
from __future__ import annotations

import importlib.metadata
import math
import os
import threading
from collections import OrderedDict

import numpy as np

from kokoro_gui.engine.caching import voice_fingerprint
from kokoro_gui.engine.runner import EngineRunner, ModelBase
from kokoro_gui.engines.base import (
    BackendHooksMixin, ConfigField, ConfigFieldType, EngineCapabilities, Synthesis, common_fields,
)
from kokoro_gui.engines.registry import register_engine
from kokoro_gui.engines.voice_store import ReferenceStore

MODEL_ID = "YatharthS/LuxTTS"
UPSTREAM_REVISION = "28ae6a61151684fffc9d1a7aa15eafa02286fe0b"
LuxTTSReferenceStore = ReferenceStore("luxtts", requires_transcript=False)
DEFAULTS = {
    "num_steps": 4, "guidance_scale": 3.0, "t_shift": 0.9,
    "ref_duration": 5.0, "ref_rms": 0.01, "return_smooth": False,
}


def _load_lux(device):
    try:
        from zipvoice.luxvoice import LuxTTS
    except ImportError as error:
        raise RuntimeError("LuxTTS dependencies are missing. In the app's Python environment, "
                           "run: python -m pip install -r requirements-luxtts.txt. "
                           f"Details: {error}") from error
    return LuxTTS(MODEL_ID, device=device, threads=2)


def _device(device):
    # Match the app's Auto policy; explicit MPS can also be used by API callers.
    if device in (None, "auto"):
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported LuxTTS device: {device}")
    return device


def _encode_prompt(lux, path, settings, transcript):
    """Keep LuxTTS's audio preprocessing, supplying reviewed text when saved.

    Upstream encode_prompt always runs ASR. Its process_audio helper accepts
    a transcriber callback, so supplying text needs no mutable model patch.
    """
    if not transcript:
        return lux.encode_prompt(path, duration=settings["ref_duration"], rms=settings["ref_rms"])
    from zipvoice.modeling_utils import process_audio

    values = process_audio(
        path, lambda audio: {"text": transcript}, lux.tokenizer,
        lux.feature_extractor, lux.device, target_rms=settings["ref_rms"],
        duration=settings["ref_duration"],
    )
    return dict(zip(("prompt_tokens", "prompt_features_lens", "prompt_features", "prompt_rms"), values))


class LuxTTSModel(ModelBase):
    engine_id = "luxtts"
    display_name = "LuxTTS"
    sample_rate = 48000
    concurrency = "shared"

    def __init__(self, loader=None):
        self._loader = loader or _load_lux
        self._lux = None
        self._device = None
        self._lock = threading.RLock()
        self._prompts = OrderedDict()

    def load(self, lang_code=None, device=None):
        chosen = _device(device)
        with self._lock:
            if self._lux is None or self._device != chosen:
                self._lux = None
                self._prompts.clear()
                self._device = None
                self._lux = self._loader(chosen)
                self._device = chosen
            return self._lux

    def loading_message(self):
        return "Loading LuxTTS (first use downloads speech and reference transcription models)..."

    def ready_message(self, lang_code):
        device = getattr(self._lux, "device", self._device)
        return f"LuxTTS ready ({device}, 48 kHz)."

    def engine_version(self):
        return f"{MODEL_ID}:{UPSTREAM_REVISION}:adapter-2:{LuxTTSBackendAdapter.package_version()}"

    def cache_key_extra(self, config):
        settings = {key: config.get(key, default) for key, default in DEFAULTS.items()}
        path = self.resolve_voice_path(config.get("voice"), config.get("project_dir"))
        settings["ref_transcript"] = self._transcript(path)
        return settings

    @staticmethod
    def _transcript(path):
        if path and os.path.isfile(path):
            return LuxTTSReferenceStore.read_transcript_file(os.path.splitext(path)[0] + ".txt")
        return ""

    def resolve_voice_path(self, name, project_dir=None):
        if name and os.path.isabs(name) and os.path.isfile(name):
            return name
        return LuxTTSReferenceStore.find_wav(name, project_dir) or (os.path.basename(name) if name else None)

    @staticmethod
    def _validate_reference(path):
        import soundfile as sf

        if not path or not os.path.isfile(path):
            raise ValueError("Select and save a LuxTTS voice reference before generating.")
        if os.path.getsize(path) > 200 * 1024 * 1024:
            raise ValueError("LuxTTS reference audio must be no larger than 200 MB.")
        info = sf.info(path)
        if info.duration < 3:
            raise ValueError("LuxTTS reference audio must contain at least 3 seconds of speech.")

    def synthesize(self, text, voice, speed, lang_code, params):
        if not text or not text.strip():
            return Synthesis(np.zeros(0, dtype=np.float32))
        cancel = params.get("cancel_event")
        if cancel is not None and cancel.is_set():
            return Synthesis(np.zeros(0, dtype=np.float32))
        settings = self.cache_key_extra(params)
        bounds = {"num_steps": (1, 32), "guidance_scale": (0, 10), "t_shift": (0.1, 1),
                  "ref_duration": (3, 1000), "ref_rms": (0.001, 0.1)}
        for key, (low, high) in bounds.items():
            value = float(settings[key])
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"LuxTTS {key} must be between {low} and {high}.")
        if float(settings["num_steps"]) != int(settings["num_steps"]):
            raise ValueError("LuxTTS num_steps must be an integer.")
        if not math.isfinite(float(speed)) or not 0.1 <= float(speed) <= 4:
            raise ValueError("LuxTTS speed must be between 0.1 and 4.")
        path = self.resolve_voice_path(voice, params.get("project_dir"))
        self._validate_reference(path)
        with self._lock:
            if self._lux is None:
                self.load(lang_code)
            transcript = self._transcript(path)
            key = (voice_fingerprint(os.path.abspath(path)), transcript,
                   settings["ref_duration"], settings["ref_rms"])
            if key not in self._prompts:
                prompt = _encode_prompt(self._lux, path, settings, transcript)
                self._prompts[key] = prompt
                if len(self._prompts) > 8:
                    self._prompts.popitem(last=False)
            self._prompts.move_to_end(key)
            if cancel is not None and cancel.is_set():
                return Synthesis(np.zeros(0, dtype=np.float32))
            wav = self._lux.generate_speech(
                text, self._prompts[key], speed=speed, num_steps=int(settings["num_steps"]),
                guidance_scale=settings["guidance_scale"], t_shift=settings["t_shift"],
                return_smooth=settings["return_smooth"],
            )
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu().numpy()
            # Both vocoder paths return 48 kHz (the smooth path resamples its
            # 24 kHz head internally). No sample-rate guessing or time stretch.
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            if not np.isfinite(audio).all() or audio.size == 0:
                raise ValueError("LuxTTS returned empty or invalid audio.")
            return Synthesis(audio)


class LuxTTSBackendAdapter(BackendHooksMixin):
    id = "luxtts"
    display_name = "LuxTTS (voice cloning)"
    voice_kind = "reference"
    voice_store = LuxTTSReferenceStore
    capabilities = EngineCapabilities(
        supports_voice_cloning=True, supports_multi_speaker_script=True,
        supports_jit_streaming=False, supports_speed=True,
    )

    def __init__(self, engine=None):
        self._engine = engine if engine is not None else EngineRunner(LuxTTSModel())

    @property
    def engine(self):
        return self._engine

    @classmethod
    def package_version(cls):
        try:
            return importlib.metadata.version("zipvoice")
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    @classmethod
    def get_config_schema(cls):
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="en", choices=[("English", "en")], group="Generation"),
            ConfigField("voice", "Voice Reference", ConfigFieldType.CHOICE, group="Generation"),
            *common_fields(max_threads=1),
            ConfigField("num_steps", "Sampling Steps", ConfigFieldType.INT,
                        default=4, min=1, max=32, step=1, group="Model"),
            ConfigField("guidance_scale", "Guidance Scale", ConfigFieldType.FLOAT,
                        default=3.0, min=0, max=10, step=0.1, group="Model"),
            ConfigField("t_shift", "Time Shift", ConfigFieldType.FLOAT,
                        default=0.9, min=0.1, max=1, step=0.05, group="Model"),
            ConfigField("ref_duration", "Reference Duration (seconds)", ConfigFieldType.FLOAT,
                        default=5.0, min=3, max=1000, step=1, group="Model"),
            ConfigField("ref_rms", "Reference RMS", ConfigFieldType.FLOAT,
                        default=0.01, min=0.001, max=0.1, step=0.001, group="Model"),
            ConfigField("return_smooth", "Smooth Audio", ConfigFieldType.BOOL,
                        default=False, group="Model"),
        ]

    def cancel(self):
        self._engine.cancel()

    def reference_transcription_duration(self, config):
        return config.get("ref_duration", DEFAULTS["ref_duration"])


register_engine("luxtts", LuxTTSBackendAdapter, display_name=LuxTTSBackendAdapter.display_name)
