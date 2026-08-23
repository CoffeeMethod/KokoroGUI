"""A real (non-Kokoro) second backend:
https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b - a 0.6B DualAR
zero-shot voice-cloning model. Unlike Kokoro's named/mixed `.pt` voice
embeddings, Audio8 clones a voice from a *reference WAV + a transcript of
what's said in that WAV* (`capabilities.supports_voice_cloning=True` - see
kokoro_gui/qt/docks/voice_clone_dock.py, the "Voice Reference" tab shown
only for a backend with this capability).

Follows the `dummy.py`/`kokoro.py` contract (TTSEngineBackend, base.py) and
reuses the same model-agnostic mixins `DummyEngine` does
(AudioFXMixin/ConversionMixin/JITMixin/LexiconMixin/PresetsMixin/SrtMixin/
TextExtractionMixin from kokoro_gui/engine/__init__.py) - but two things are
genuinely different from both existing backends, both explained where they
happen below:

1. Output is 44.1kHz, not Kokoro/Dummy's 24000Hz. `ConversionMixin` reads an
   instance `self.SAMPLE_RATE` (defaulting to 24000 via `getattr` when a
   backend doesn't set one) instead of a hardcoded literal, specifically so
   this engine can override it - see kokoro_gui/engine/conversion.py.
2. `get_thread_pipeline` does *not* hand out one model per worker thread the
   way Kokoro's `KPipeline` does - see `_Audio8Pipeline` and `_get_model`
   below for why and how the model is shared instead.

`transformers` itself is only imported inside `_get_model()`, not at this
module's top level - though it's already an indirect hard dependency of
this app regardless (the `kokoro` package imports it internally), so that
isn't actually deferring much on its own. What genuinely stays deferred
until `init_pipeline_async`/first generation: `AutoModel.from_pretrained(...)`
actually running - the network fetch (first run) and the model weights
landing in memory - so a user who never switches to this engine never pays
that cost merely by the `kokoro_gui.engines` package registering it at
startup. Loads with `trust_remote_code=True` (the model ships custom
modeling code in its HF repo) - see this module's sibling
kokoro_gui/engine/asr.py for the same note about what that means.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import threading
from typing import Optional

import numpy as np
import soundfile as sf
from pedalboard.io import AudioFile

import kokoro_engine
from kokoro_engine import AsyncLoopThread
from kokoro_gui.engine import (
    AudioFXMixin, ConversionMixin, JITMixin, LexiconMixin, PresetsMixin,
    SrtMixin, TextExtractionMixin,
)
from kokoro_gui.engine.caching import compute_cache_key, voice_fingerprint
from kokoro_gui.engines.base import (
    ConfigField, ConfigFieldType, EngineCapabilities, VoiceInfo,
    COMMON_SPLIT_PATTERN_CHOICES, COMMON_OUTPUT_FORMAT_CHOICES,
)
from kokoro_gui.engines.registry import register_engine

TTS_MODEL_ID = "Audio8/Audio8-TTS-Preview-0.6b"
SAMPLE_RATE = 44100

# Saved wav+transcript voice references live as sidecar file pairs here:
# <AUDIO8_REFS_DIR>/<name>.wav and <AUDIO8_REFS_DIR>/<name>.txt. Mirrors
# kokoro_engine.CUSTOM_VOICES_DIR's flat, name-keyed convention, just with
# two files per entry instead of one .pt. Read qualified (module-global, not
# rebound to a local default) so tests can monkeypatch
# `kokoro_gui.engines.audio8_tts.AUDIO8_REFS_DIR` the same way
# `isolated_dirs` monkeypatches `kokoro_engine.CUSTOM_VOICES_DIR`.
AUDIO8_REFS_DIR = os.path.join("custom_voices", "audio8_refs")


def _ref_codes_cache_dir() -> str:
    """Persisted cache of *encoded* reference audio ("reference codes" - see
    `Audio8Engine._reference_codes_path`/module docstring below): one `.npy`
    per distinct reference wav's content, named by its `voice_fingerprint`
    (sha256-based, mtime-cached) rather than by reference name, so re-saving
    a reference under a new name (or two references sharing identical
    audio) reuses the same cache entry, and re-saving one *name* with
    different audio correctly misses. Gated by the "cache_reference_codes"
    config field (Audio8BackendAdapter.get_config_schema) - default on. Like
    `CACHE_DIR`, this grows unbounded; no eviction policy yet (ROADMAP).

    Nested under the *current* `AUDIO8_REFS_DIR`, resolved fresh on every
    call (not baked in as a module-level constant at import time) so that
    tests monkeypatching `AUDIO8_REFS_DIR` (see `isolated_audio8_refs` in
    tests/test_engines_audio8.py) redirect this cache too, the same way
    they already redirect reference wav/transcript storage - otherwise this
    would keep writing into the real `custom_voices/audio8_refs/` regardless
    of that patch.
    """
    return os.path.join(AUDIO8_REFS_DIR, ".ref_codes_cache")

# The model's supported languages (per its model card) - passed through as
# plain strings to whatever `language=` argument the processor expects.
# There's no published short-code table for this model the way Kokoro has
# single-letter lang codes, so the value *is* the display label; worth
# double-checking against the processor's actual accepted values on first
# real run.
AUDIO8_LANGUAGE_CHOICES = [
    (name, name) for name in (
        "English", "Chinese", "Cantonese", "French", "German", "Italian",
        "Japanese", "Korean", "Dutch", "Polish", "Spanish",
    )
]


class Audio8ReferenceStore:
    """CRUD over the saved wav+transcript voice-reference pairs under
    `AUDIO8_REFS_DIR`. Plain functions, not a mixin - unlike custom-voice
    resolution (which needs a live pipeline to load a `.pt` tensor through),
    saving/listing/deleting these sidecar files needs no model, so there's
    no reason to route it through `Audio8Engine`."""

    @staticmethod
    def _safe_name(name: str) -> str:
        # Same path-traversal guard as VoiceMixingMixin.resolve_voice_path/
        # mix_voices (kokoro_gui/engine/voices.py).
        return os.path.basename(name)

    @staticmethod
    def save_reference(name: str, wav_path: str, transcript: str) -> str:
        """Copies `wav_path` and writes `transcript` under a sanitized
        `name`, creating `AUDIO8_REFS_DIR` if needed. Returns the saved wav's
        absolute path."""
        safe_name = Audio8ReferenceStore._safe_name(name)
        if not safe_name:
            raise ValueError("Reference name must not be empty.")
        os.makedirs(AUDIO8_REFS_DIR, exist_ok=True)
        out_wav = os.path.join(AUDIO8_REFS_DIR, f"{safe_name}.wav")
        out_txt = os.path.join(AUDIO8_REFS_DIR, f"{safe_name}.txt")
        shutil.copyfile(wav_path, out_wav)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(transcript.strip())
        return os.path.abspath(out_wav)

    @staticmethod
    def list_references() -> list:
        """Returns sorted `[name, ...]` for every wav+txt sidecar pair found
        (a lone `.wav` or `.txt` without its partner is skipped - an
        incomplete/interrupted save, not a usable reference)."""
        if not os.path.isdir(AUDIO8_REFS_DIR):
            return []
        names = []
        for f in os.listdir(AUDIO8_REFS_DIR):
            if not f.endswith(".wav"):
                continue
            name = f[:-4]
            if os.path.isfile(os.path.join(AUDIO8_REFS_DIR, f"{name}.txt")):
                names.append(name)
        return sorted(names)

    @staticmethod
    def get_transcript(name: str) -> str:
        safe_name = Audio8ReferenceStore._safe_name(name)
        txt_path = os.path.join(AUDIO8_REFS_DIR, f"{safe_name}.txt")
        if not os.path.isfile(txt_path):
            return ""
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def delete_reference(name: str) -> None:
        safe_name = Audio8ReferenceStore._safe_name(name)
        for ext in (".wav", ".txt"):
            path = os.path.join(AUDIO8_REFS_DIR, f"{safe_name}{ext}")
            if os.path.exists(path):
                os.remove(path)


# --- Shared model singleton -------------------------------------------------
#
# A 0.6B-parameter model loaded once per worker thread (Kokoro's KPipeline
# convention) would multiply GPU/RAM use by `num_threads` for zero benefit -
# unlike KPipeline, nothing about loading this model is thread-specific.
# Instead it's loaded once, process-wide, and every thread's generation call
# is serialized through `_model_lock` - safe (no concurrent `.generate()`
# calls into one model instance) at the cost of chunk generation itself not
# actually parallelizing across `num_threads` (I/O and pre/post-processing
# still overlap). See `Audio8BackendAdapter.get_config_schema`'s lower
# `num_threads` max, which reflects that.
_model_lock = threading.Lock()
_model = None
_processor = None


def _get_model():
    global _model, _processor
    with _model_lock:
        if _model is not None:
            return _model, _processor
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "The Audio8 engine needs the 'transformers' package "
                "(pip install -r requirements.txt)."
            ) from e
        try:
            processor = AutoProcessor.from_pretrained(TTS_MODEL_ID, trust_remote_code=True)
            model = AutoModel.from_pretrained(TTS_MODEL_ID, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {TTS_MODEL_ID}: {e}") from e
        _model, _processor = model, processor
        return _model, _processor


class _Audio8Pipeline:
    """Presents the shared singleton model as a `kokoro.KPipeline`-shaped
    callable-generator (`pipeline(text, voice=, speed=, split_pattern=)` ->
    `(graphemes, phonemes, audio)` triples, `audio` mono float32 at
    `SAMPLE_RATE`), the same convention `DummyPipeline` mimics
    (kokoro_gui/engines/dummy.py), so this engine's own `process_chunk_task`
    and the generic `ConversionMixin.generate_preview`/`smart_combine` can
    all drive it uniformly.

    `voice` here is always an already-*resolved* reference wav path (by the
    time any of the generic mixins call a pipeline, `config['voice']` has
    already been run through `Audio8Engine.resolve_voice_path` - see
    `ConversionMixin.start_conversion`/`generate_preview`) - the matching
    transcript sidecar is looked up from that path here, once per call,
    rather than needing a separate "voice name" threaded through everywhere.
    """

    def __init__(self, engine: "Audio8Engine", lang_code: str = "English"):
        self._engine = engine
        self.lang_code = lang_code

    def __call__(self, text, voice=None, speed=1.0, split_pattern=r"\n+"):
        try:
            segments = [s.strip() for s in re.split(split_pattern, text) if s.strip()]
        except re.error:
            segments = []
        if not segments and text.strip():
            segments = [text.strip()]

        ref_transcript = self._engine.resolve_voice_transcript(voice)
        for seg in segments:
            audio = self._engine.generate_segment(seg, voice, ref_transcript, speed, self.lang_code)
            yield seg, "", audio


class Audio8Engine(
    AudioFXMixin, ConversionMixin, JITMixin, LexiconMixin, PresetsMixin,
    SrtMixin, TextExtractionMixin,
):
    """KokoroEngine-shaped enough for the GUI to drive directly - same
    required surface as `DummyEngine` (worker/cancel_event/pipeline/
    on_progress/on_status/on_finish/init_pipeline_async/get_thread_pipeline/
    resolve_voice_path/cancel), plus `SAMPLE_RATE=44100` and its own
    `process_chunk_task` (see module docstring for both)."""

    SAMPLE_RATE = SAMPLE_RATE

    def __init__(self):
        self.worker = AsyncLoopThread()
        self.worker.start()
        self.cancel_event = threading.Event()
        self.pipeline = False  # not ready until init_pipeline_async loads the model

        self.on_progress = None
        self.on_status = None
        self.on_finish = None

        # Whether `generate_segment` should reuse a persisted, content-keyed
        # encoding of the reference wav instead of re-running the model's
        # audio encoder on every segment (see `_reference_codes_path`).
        # `process_chunk_task` overwrites this from `config['cache_reference_codes']`
        # each run - the `True` here only matters for callers that skip
        # `process_chunk_task` (e.g. calling `generate_segment` directly).
        self.cache_reference_codes = True

        # `ArkttsModel.generate`/`generate_audio` sampling knobs, exposed as
        # config fields (Audio8BackendAdapter.get_config_schema, "Generation"
        # group) rather than hardcoded - `process_chunk_task` overwrites
        # these from `config` each run, same pattern as `cache_reference_codes`
        # above. Defaults match this engine's original hardcoded values,
        # except `max_new_tokens` (was 4096, clamped internally to whatever
        # room is left under the model's `max_seq_len=2048` anyway - 1024
        # is a more honest default that still leaves prompt room).
        self.max_new_tokens = 1024
        self.temperature = 0.8
        self.top_p = 0.95
        self.top_k = 50

        self._lexicon_cache = {}

        os.makedirs(AUDIO8_REFS_DIR, exist_ok=True)

    async def init_pipeline_async(self, lang_code="a"):
        if self.on_status:
            self.on_status("Loading Audio8 TTS model (first use downloads it)...", False)
        try:
            await asyncio.to_thread(_get_model)
        except Exception as e:
            self.pipeline = False
            if self.on_status:
                self.on_status(f"Audio8 model load failed: {e}", True)
            return False
        self.pipeline = True
        if self.on_status:
            self.on_status("Audio8 TTS ready.", False)
        return True

    def get_thread_pipeline(self, lang_code="English"):
        return _Audio8Pipeline(self, lang_code)

    def resolve_voice_path(self, voice_name: str) -> str:
        """Resolves a saved reference name to its absolute wav path
        (sanitized-basename convention, matching
        `VoiceMixingMixin.resolve_voice_path`). Falls back to treating
        `voice_name` as a literal existing file path (a wav dropped straight
        into the Voice Reference dock and generated with before ever being
        saved under a name), and finally to returning it unchanged (will
        fail clearly at generation time rather than silently). Tolerates a
        falsy `voice_name` (e.g. the Voice dropdown is empty because no
        reference has been saved yet) by returning it as-is rather than
        raising here - the resulting generation failure is reported through
        the normal per-chunk error path (`_process_text_async`'s
        `asyncio.gather(..., return_exceptions=True)`) instead of crashing
        synchronously on the Qt main thread inside `start_conversion`."""
        if not voice_name:
            return voice_name
        safe_name = os.path.basename(voice_name)
        saved_path = os.path.join(AUDIO8_REFS_DIR, f"{safe_name}.wav")
        if os.path.exists(saved_path):
            return os.path.abspath(saved_path)
        if os.path.isabs(voice_name) and os.path.isfile(voice_name):
            return voice_name
        # Neither a saved reference nor an existing absolute file: return the
        # sanitized basename, not the raw string, so a preset-supplied
        # relative/UNC path can't be used as a literal path downstream (same
        # traversal fix as VoiceMixingMixin.resolve_voice_path - see
        # Claude/SECURITY_AUDIT.md). This still "fails clearly at generation
        # time" per the docstring above, just without ever touching the
        # unsanitized string first.
        return safe_name

    def resolve_voice_transcript(self, resolved_voice_path: str) -> str:
        """Given an already-*resolved* reference wav path (see
        `resolve_voice_path`), returns the transcript from its sidecar
        `.txt` file (same base name, `.wav` -> `.txt`), or `""` if none
        exists (including when `resolved_voice_path` itself is falsy)."""
        if not resolved_voice_path:
            return ""
        txt_path = os.path.splitext(resolved_voice_path)[0] + ".txt"
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def _reference_codes_path(self, ref_wav_path: str, ref_transcript: str) -> Optional[str]:
        """Returns the path to a persisted `.npy` of `ref_wav_path`'s
        *encoded* reference ("reference codes" - `ArkttsModel.encode_audio`'s
        output), computing and caching it on first use under
        `_ref_codes_cache_dir()`. Returns `None` when caching isn't
        applicable (no on-disk wav to fingerprint, no transcript to run the
        one-off encode with) or if the encode itself fails - the caller
        falls back to passing raw `reference_audio` on every call in that
        case, exactly like before this cache existed.

        This is the one genuine "reference audio -> tensor" step: the model
        encodes `reference_audio_values` into `reference_codes` via its own
        audio codec (`ArkttsModel.encode_audio`, a real forward pass through
        `ArkttsCodec` - not free) inside `_prepare_prompt` on *every*
        `generate`/`generate_audio` call that's given raw audio. Passing
        `reference_codes=` instead (which `ArkttsProcessor.__call__` accepts
        as a path it `np.load`s itself, per `processing_arktts.py`) skips
        that re-encode entirely - the same reference wav produces identical
        codes every time, so encoding it once and reusing the codes across
        every segment/chunk that shares a voice reference is a correctness-
        preserving cache, not an approximation.
        """
        if not ref_wav_path:
            return None
        fp = voice_fingerprint(ref_wav_path)
        if fp == ref_wav_path:
            return None  # not an existing absolute file - can't fingerprint/cache it
        cache_dir = _ref_codes_cache_dir()
        cache_path = os.path.join(cache_dir, f"{fp}.npy")
        if os.path.isfile(cache_path):
            return cache_path
        if not ref_transcript:
            return None  # a reference-conditioned encode requires reference_text too

        try:
            model, processor = _get_model()
            with _model_lock:
                probe = processor(
                    text="x", reference_audio=ref_wav_path, reference_text=ref_transcript,
                    return_tensors="pt",
                )
                codes, code_lengths = model.encode_audio(
                    probe["reference_audio_values"], probe["reference_audio_lengths"],
                )
                trimmed = codes[0, :, : int(code_lengths[0])].detach().cpu().numpy().astype(np.int64)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, trimmed)
        except Exception as e:
            print(f"Audio8 reference-codes cache write error: {e}")
            return None
        return cache_path

    def generate_segment(self, text: str, ref_wav_path: str, ref_transcript: str,
                          speed: float, lang_code: str) -> np.ndarray:
        """Runs one segment through the shared model, serialized via
        `_model_lock` (see module docstring). Returns mono float32 audio at
        `SAMPLE_RATE`.

        Checked against the installed model's actual `processing_arktts.py`/
        `modeling_arktts.py` (the model card guess this originally shipped
        with was wrong on every point below):

        - The processor's real kwargs are `reference_audio`/`reference_text`,
          not `ref_audio`/`ref_text`.
        - Neither `ArkttsProcessor.__call__` nor `ArkttsModel.generate` take
          a `language` or `speed` argument at all - both raise `TypeError`
          on any kwarg they don't recognize, which is what surfaced as
          "Unexpected processor arguments: [...]". `speed`/`lang_code` stay
          in this method's signature only so it keeps matching
          `_Audio8Pipeline`/`process_chunk_task`'s generic
          `(text, voice, speed, lang_code)` shape shared with Kokoro/Dummy -
          the model always synthesizes at its own pace and infers language
          from the text itself, so both are accepted here and silently
          unused rather than forwarded.
        - `processor.decode(...)` is just `tokenizer.decode` (text token
          decoding) - it was never how to get audio out. The real path is
          `model.generate(**inputs)` -> codes -> `model.decode_audio(codes)`,
          or the combined `model.generate_audio(**inputs, ...)` used below,
          which returns `(waveforms, lengths, codes)` directly.

        When `self.cache_reference_codes` is on (see `process_chunk_task`),
        looks up/populates a persisted reference-codes cache first (see
        `_reference_codes_path`) and passes `reference_codes=` instead of
        `reference_audio=`/`reference_text=` on a hit - same output, skips
        re-encoding the reference wav through the model's audio codec.
        """
        model, processor = _get_model()
        cached_codes_path = (
            self._reference_codes_path(ref_wav_path, ref_transcript)
            if self.cache_reference_codes else None
        )
        with _model_lock:
            if cached_codes_path:
                # `reference_text` isn't only an input to the audio encode -
                # `ArkttsProcessor._prompt_segments` bakes it into the *text*
                # prompt tokens whenever `has_reference` is True (set by
                # either `reference_audio` or `reference_codes`), so it's
                # still required here even though the audio side is cached.
                inputs = processor(
                    text=text, reference_codes=cached_codes_path,
                    reference_text=ref_transcript or None, return_tensors="pt",
                )
            else:
                inputs = processor(
                    text=text,
                    reference_audio=ref_wav_path or None,
                    reference_text=ref_transcript or None,
                    return_tensors="pt",
                )
            waveforms, lengths, _codes = model.generate_audio(
                **inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                top_p=self.top_p, top_k=self.top_k,
            )
            audio = waveforms[0, : lengths[0]].detach().cpu().numpy()

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio

    def process_chunk_task(self, chunk_data, progress_callback):
        """Same shape/return contract, and the same "predict the segment
        split, check every expected file exists" cache-validity check, as
        `CachingMixin.process_chunk_task` (kokoro_gui/engine/caching.py) -
        hand-rolled rather than inherited because that mixin hardcodes
        24000Hz in several places and this engine outputs 44100Hz. Calls
        `compute_cache_key` directly as a library function instead, passing
        `extra={"ref_transcript": ...}` so a reference's transcript is part
        of the cache key too - changing just the transcript for the same wav
        (a real "the auto-transcript was wrong, I fixed it" workflow)
        correctly invalidates old cache entries.
        """
        index, text, config = chunk_data
        if self.cancel_event.is_set():
            return []

        lang_code = config.get('lang_code', 'English')
        eff_speed = config['speed']
        ref_wav = config['voice']  # already resolved by ConversionMixin.start_conversion
        ref_transcript = self.resolve_voice_transcript(ref_wav)
        split_pattern = config.get('split_pattern', r"\n+")
        # See `_reference_codes_path`/module docstring - independent of the
        # per-segment WAV cache below (`use_cache`/`caching`).
        self.cache_reference_codes = config.get('cache_reference_codes', True)
        # `ArkttsModel.generate` sampling knobs - see `__init__`'s docstring
        # on these same attributes. Read into `extra` below too: they change
        # what gets generated, so a stale segment cached under old values
        # must miss rather than silently keep serving old audio.
        self.max_new_tokens = config.get('max_new_tokens', 1024)
        self.temperature = config.get('temperature', 0.8)
        self.top_p = config.get('top_p', 0.95)
        self.top_k = config.get('top_k', 50)

        use_cache = config.get('caching', False)
        cache_hash = None
        cached_segments = []  # [(graphemes, audio), ...]

        if use_cache:
            cache_hash = compute_cache_key(
                text, ref_wav, eff_speed, lang_code, engine_id="audio8",
                extra={
                    "ref_transcript": ref_transcript,
                    "max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
                    "top_p": self.top_p, "top_k": self.top_k,
                },
            )
            try:
                predicted_texts = [t.strip() for t in re.split(split_pattern, text) if t.strip()]
            except re.error:
                predicted_texts = []
            if not predicted_texts and text.strip():
                predicted_texts = [text.strip()]

            if predicted_texts:
                loaded = []
                all_exist = True
                for i, seg_text in enumerate(predicted_texts):
                    f_path = os.path.join(kokoro_engine.CACHE_DIR, f"{cache_hash}_{i}.wav")
                    if not os.path.exists(f_path):
                        all_exist = False
                        break
                    try:
                        audio_data, _ = sf.read(f_path)
                    except Exception as e:
                        print(f"Audio8 cache read error: {e}")
                        all_exist = False
                        break
                    loaded.append((seg_text, audio_data))
                if all_exist:
                    cached_segments = loaded

        chunk_files = []
        base_name = f"{config.get('filename', 'output')}_{config.get('time_id', '0')}_part{index}"

        def write_output(graphemes, audio, sub_idx):
            processed = self.process_audio(audio, self.SAMPLE_RATE, config)
            fmt = config.get('format', 'wav').lower()
            if fmt not in ('wav', 'flac', 'mp3', 'ogg'):
                fmt = 'wav'
            file_name = f"{base_name}_{sub_idx}.{fmt}"
            path = os.path.join(config['out_dir'], file_name)
            try:
                with AudioFile(path, 'w', samplerate=self.SAMPLE_RATE, num_channels=1) as f:
                    f.write(processed)
            except Exception as e:
                print(f"Audio8 write failed: {e}. Fallback to soundfile.")
                sf.write(path, processed, self.SAMPLE_RATE)
            return {
                "path": path, "text": graphemes,
                "duration": len(processed) / self.SAMPLE_RATE, "seg_idx": index,
            }

        if cached_segments:
            for sub_idx, (graphemes, audio) in enumerate(cached_segments):
                if self.cancel_event.is_set():
                    break
                if progress_callback:
                    progress_callback(len(graphemes), graphemes)
                chunk_files.append(write_output(graphemes, audio, sub_idx))
        else:
            pipeline = self.get_thread_pipeline(lang_code)
            generator = pipeline(text, voice=ref_wav, speed=eff_speed, split_pattern=split_pattern)
            for sub_idx, (graphemes, _phonemes, audio) in enumerate(generator):
                if self.cancel_event.is_set():
                    break
                if progress_callback:
                    progress_callback(len(graphemes), graphemes)

                if use_cache and cache_hash:
                    try:
                        sf.write(
                            os.path.join(kokoro_engine.CACHE_DIR, f"{cache_hash}_{sub_idx}.wav"),
                            audio, self.SAMPLE_RATE,
                        )
                    except Exception as e:
                        print(f"Audio8 cache write error: {e}")

                chunk_files.append(write_output(graphemes, audio, sub_idx))

        return chunk_files

    def cancel(self) -> None:
        self.cancel_event.set()


class Audio8BackendAdapter:
    id = "audio8"
    display_name = "Audio8 TTS (voice cloning)"
    capabilities = EngineCapabilities(
        supports_voice_mixing=False,
        supports_voice_cloning=True,
        supports_multi_speaker_script=True,
        is_local_model=True,
        supports_jit_streaming=False,
    )

    def __init__(self, engine: Optional[Audio8Engine] = None):
        self._engine = engine if engine is not None else Audio8Engine()

    @property
    def engine(self):
        return self._engine

    def get_config_schema(self) -> list:
        return [
            ConfigField("lang_code", "Language", ConfigFieldType.CHOICE,
                        default="English", choices=list(AUDIO8_LANGUAGE_CHOICES), group="Generation"),
            ConfigField("voice", "Voice Reference", ConfigFieldType.CHOICE,
                        default=None, group="Generation"),
            ConfigField("speed", "Speed", ConfigFieldType.SLIDER,
                        default=1.0, min=0.5, max=2.0, step=0.1, group="Generation"),
            ConfigField("split_pattern", "Split By", ConfigFieldType.CHOICE,
                        default=r"\n+", choices=list(COMMON_SPLIT_PATTERN_CHOICES), group="Generation"),
            ConfigField("format", "Output Format", ConfigFieldType.CHOICE,
                        default="wav", choices=list(COMMON_OUTPUT_FORMAT_CHOICES), group="Generation"),
            ConfigField("num_threads", "Parallel Threads", ConfigFieldType.INT,
                        default=1, min=1, max=4, step=1, group="Advanced"),
            ConfigField("caching", "Enable Segment Cache", ConfigFieldType.BOOL,
                        default=True, group="Advanced"),
            ConfigField("cache_reference_codes", "Cache Reference Encoding", ConfigFieldType.BOOL,
                        default=True, group="Advanced"),
            # `ArkttsModel.generate`/`generate_audio` sampling knobs (see
            # `Audio8Engine.__init__`/`process_chunk_task`/`generate_segment`)
            # - model-specific, unlike everything above, so broken out into
            # their own group rather than folded into "Generation"/"Advanced".
            # `max_new_tokens` above `max_seq_len - <prompt length>` (2048
            # total, per the model's config) is clamped internally by
            # `ArkttsModel.generate` - the 2048 ceiling here just matches
            # that reality instead of offering a value that's silently capped.
            ConfigField("max_new_tokens", "Max New Tokens", ConfigFieldType.INT,
                        default=1024, min=64, max=2048, step=64, group="Model"),
            ConfigField("temperature", "Temperature", ConfigFieldType.SLIDER,
                        default=0.8, min=0.1, max=2.0, step=0.05, group="Model"),
            ConfigField("top_p", "Top P", ConfigFieldType.SLIDER,
                        default=0.95, min=0.0, max=1.0, step=0.01, group="Model"),
            ConfigField("top_k", "Top K", ConfigFieldType.INT,
                        default=50, min=0, max=200, step=1, group="Model"),
        ]

    def get_voices(self, lang_code: Optional[str] = None) -> list:
        """Saved wav+transcript references - see `Audio8ReferenceStore`.
        Unlike Kokoro, there are no built-in named voices at all; every
        selectable "voice" here is a user-saved reference."""
        return [
            VoiceInfo(id=name, display_name=name, lang_code=None, is_custom=True)
            for name in Audio8ReferenceStore.list_references()
        ]

    def cancel(self) -> None:
        self._engine.cancel()


register_engine("audio8", Audio8BackendAdapter, display_name=Audio8BackendAdapter.display_name)
