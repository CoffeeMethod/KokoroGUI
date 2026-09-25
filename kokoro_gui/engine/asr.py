"""Auto-transcription helpers for reference audio, used by the Voice
Reference dock (kokoro_gui/qt/docks/voice_clone_dock.py) to pre-fill an
editable transcript for a WAV a user is about to use as an Audio8-TTS voice
reference.

Three engines are registered in `ASR_ENGINES`, selectable from the dock's
engine picker:

- `"whisper"` (the default) - faster-whisper (CTranslate2) running
  `large-v3-turbo`, int8 on CPU, float16 on CUDA. MIT. The weights (about
  1.6 GB) download from Hugging Face on first use; `whisper_model_cached()`
  tells a caller whether that's about to happen so it can ask first. The
  model is `WHISPER_MODEL` in `.env` (default `large-v3-turbo`), so a weak
  machine can pick a smaller one. The only engine with reliable word
  timestamps.
- `"audio8"` - https://huggingface.co/Audio8/Audio8-ASR-0.1B. Higher quality,
  needs `transformers`/`torch`, downloads multi-GB weights from Hugging Face
  on first use, and is CC-BY-NC-4.0 (non-commercial).
- `"vosk"` - https://alphacephei.com/vosk. Fully offline and Apache-2.0
  (commercial-friendly), but needs a model directory downloaded and unzipped
  by hand from https://alphacephei.com/vosk/models (there's no pip-installable
  weights the way `transformers.from_pretrained` fetches Audio8's) and is
  generally lower quality, especially on non-English audio. The model folder
  lives in `VOSK_MODEL_PATH`, in a `.env` file at the project root (see
  `.env.example`) rather than in `config_qt.json` - unlike the engine choice
  itself, it's treated as deployment config, not a per-session GUI
  preference. The Voice Reference dock can still edit it though:
  `set_vosk_model_path` writes the change straight into `.env` (creating the
  file if needed) via `dotenv.set_key`, rather than the dock hand-rolling its
  own settings round-trip for one value. Whatever WAV format the reference
  audio is in, it's converted to the 16-bit mono PCM Vosk requires before
  recognition runs (see `_ensure_pcm16_mono` below) - the caller never has to
  pre-convert it.

Audio8's zero-shot voice cloning needs a transcript of the reference audio,
not just the audio itself - getting that by hand is tedious, so either engine
here is a starting point the user reviews/corrects, not a ground-truth
oracle.

`transcribe_wav` returns text. `transcribe_wav_words` returns
`[(word, start_s, end_s), ...]` and is the entry point for anything that
needs timing (Whisper's are tight, Vosk's loose, Audio8 has none and
raises `NotImplementedError`).

Every engine is lazy-imported (only inside its `_get_*_model()` helper,
not at this module's top level) so importing this module - which happens
whenever the Voice Reference dock is built - never triggers a model
load/download by itself; only calling `transcribe_wav` does. Note
`transformers` is already an indirect hard dependency of this app (the
`kokoro` package imports it internally), so deferring *its* import isn't
about avoiding the import itself - what's actually deferred is
`AutoModel.from_pretrained(...)`/`AutoProcessor.from_pretrained(...)`, i.e.
the network fetch and weights landing in memory. Audio8 loads with
`trust_remote_code=True`, which executes Python code shipped in the model's
HF repo the first time it's loaded - inherent to how this model is
distributed, not something this module can avoid while still using it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import wave
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from dotenv import find_dotenv, load_dotenv, set_key

AUDIO8_MODEL_ID = "Audio8/Audio8-ASR-0.1B"
# Kept as an alias - some external callers/notes may still reference the old
# name from before this module supported more than one engine.
ASR_MODEL_ID = AUDIO8_MODEL_ID

VOSK_MODEL_PATH_ENV_KEY = "VOSK_MODEL_PATH"

WHISPER_MODEL_ENV_KEY = "WHISPER_MODEL"
DEFAULT_WHISPER_MODEL = "large-v3-turbo"
# Download size per faster-whisper model name, for the "downloads about X
# the first time" prompt. A name missing here shows no size.
WHISPER_MODEL_SIZES = {
    "tiny": "75 MB",
    "tiny.en": "75 MB",
    "base": "145 MB",
    "base.en": "145 MB",
    "small": "484 MB",
    "small.en": "484 MB",
    "distil-small.en": "336 MB",
    "medium": "1.5 GB",
    "medium.en": "1.5 GB",
    "distil-medium.en": "789 MB",
    "large-v1": "3.1 GB",
    "large-v2": "3.1 GB",
    "large-v3": "3.1 GB",
    "large": "3.1 GB",
    "distil-large-v2": "1.5 GB",
    "distil-large-v3": "1.5 GB",
    "large-v3-turbo": "1.6 GB",
    "turbo": "1.6 GB",
}


def _dotenv_path() -> str:
    """Resolves the `.env` file both `_load_dotenv`/`set_vosk_model_path`
    act on: whichever one `find_dotenv(usecwd=True)` locates walking up from
    the current working directory, or - since that returns "" when none
    exists yet - a new `.env` at the cwd (the project root, by this
    codebase's convention), so saving from the GUI works on a first run with
    no `.env` file at all."""
    return find_dotenv(usecwd=True) or os.path.join(os.getcwd(), ".env")


def _load_dotenv(override: bool = False) -> None:
    """Populates `os.environ` from a `.env` file at (or above) the current
    working directory, if one exists - a no-op otherwise. Safe to call at
    import time: it's a local file read, not a network call, and with
    `override=False` (the default) never overwrites a variable the
    environment already set - a real env var still wins over a stale `.env`
    entry."""
    load_dotenv(_dotenv_path(), override=override)


_load_dotenv()


def get_vosk_model_path() -> str:
    """Reads the Vosk model folder from `VOSK_MODEL_PATH` - not a GUI
    setting saved in `config_qt.json`, see this module's docstring -
    defaulting to "" (unconfigured) if unset."""
    return os.environ.get(VOSK_MODEL_PATH_ENV_KEY, "").strip()


def set_vosk_model_path(path: str) -> None:
    """Writes `VOSK_MODEL_PATH` into `.env` (`dotenv.set_key` creates the
    file if it doesn't exist yet and rewrites just that one line, leaving
    any other keys already in it alone) and updates `os.environ` so the
    running process picks up the change immediately - no reload needed.
    Wired to the Voice Reference dock's "Save" button, the write-side
    counterpart to `reload_vosk_model_path`'s read-side (file -> process)."""
    path = path.strip()
    set_key(_dotenv_path(), VOSK_MODEL_PATH_ENV_KEY, path)
    os.environ[VOSK_MODEL_PATH_ENV_KEY] = path


def reload_vosk_model_path() -> str:
    """Re-reads `.env` into `os.environ`, picking up an edit made to it while
    the app is already running (plain `load_dotenv()` leaves already-set
    variables alone, so this passes `override=True`), and returns the
    resulting path. Wired to the Voice Reference dock's "Reload" button, the
    read-side counterpart to `set_vosk_model_path`."""
    _load_dotenv(override=True)
    return get_vosk_model_path()


@dataclass(frozen=True)
class AsrEngineInfo:
    id: str
    display_name: str
    description: str


ASR_ENGINES = [
    AsrEngineInfo(
        "whisper",
        "Whisper large-v3-turbo (local, word timestamps)",
        "faster-whisper, runs locally (int8 on CPU, float16 on CUDA). Downloads the "
        f"model from Hugging Face on first use, about {WHISPER_MODEL_SIZES[DEFAULT_WHISPER_MODEL]}; "
        f"set {WHISPER_MODEL_ENV_KEY} in .env for a smaller one. MIT.",
    ),
    AsrEngineInfo(
        "audio8",
        "Audio8-ASR-0.1B (online, higher quality)",
        f"Downloads {AUDIO8_MODEL_ID} from Hugging Face on first use. "
        "CC-BY-NC-4.0 - non-commercial use only.",
    ),
    AsrEngineInfo(
        "vosk",
        "Vosk (offline)",
        "Fully offline once a model is downloaded from "
        f"https://alphacephei.com/vosk/models and its folder is set as "
        f"{VOSK_MODEL_PATH_ENV_KEY} in a .env file. Apache-2.0.",
    ),
]
DEFAULT_ASR_ENGINE = ASR_ENGINES[0].id


def get_asr_engine(engine_id: str) -> AsrEngineInfo:
    for engine in ASR_ENGINES:
        if engine.id == engine_id:
            return engine
    raise ValueError(f"Unknown ASR engine '{engine_id}'")


# --- Audio8-ASR-0.1B ---------------------------------------------------

_model_lock = threading.Lock()
_model = None
_processor = None


def _get_model():
    """Lazily loads and caches the Audio8 ASR model/processor as a
    process-wide singleton (guarded by a lock so two near-simultaneous
    "Auto-Transcribe" clicks - or a batch run alongside one - don't each
    start their own multi-GB download/load)."""
    global _model, _processor

    with _model_lock:
        if _model is not None:
            return _model, _processor

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "Auto-transcription needs the 'transformers' package "
                "(pip install -r requirements.txt)."
            ) from e

        try:
            processor = AutoProcessor.from_pretrained(AUDIO8_MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(AUDIO8_MODEL_ID, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {AUDIO8_MODEL_ID}: {e}") from e

        _model, _processor = model, processor
        return _model, _processor


def _transcribe_wav_audio8(wav_path: str, max_new_tokens: int = 128) -> str:
    """Transcribes `wav_path` (16kHz mono expected; the model resamples if
    needed per its model card) to plain text via Audio8-ASR-0.1B's
    chat-template audio interface. Blocking/CPU-or-GPU-bound - callers from
    the GUI should run this off the main thread (see
    `VoiceCloneDock._on_transcribe_clicked`, which schedules it via
    `asyncio.to_thread` on the active engine's worker)."""
    model, processor = _get_model()

    conversation = [
        {
            "role": "user",
            "content": [{"type": "audio", "path": wav_path}],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Only decode the newly-generated tokens, not the echoed prompt.
        prompt_len = inputs["input_ids"].shape[-1]
        text = processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e


# --- Whisper (faster-whisper) --------------------------------------------

_whisper_lock = threading.Lock()
_whisper_models: dict[tuple[str, str], object] = {}  # (model name, device) -> loaded WhisperModel
# Set once a CUDA run fails (e.g. `cublas64_12.dll` missing: CTranslate2 sees
# the GPU through the driver but the CUDA 12 runtime libraries aren't
# installed). Every later load goes to the CPU for the rest of the process.
_whisper_cuda_failed = False


def get_whisper_model_name() -> str:
    """`WHISPER_MODEL` from the environment / `.env`, default
    `large-v3-turbo`."""
    return os.environ.get(WHISPER_MODEL_ENV_KEY, "").strip() or DEFAULT_WHISPER_MODEL


def whisper_model_size(name: str | None = None) -> str | None:
    """The download size string for `name` (default: the configured
    model), or None when `WHISPER_MODEL_SIZES` doesn't know it."""
    return WHISPER_MODEL_SIZES.get(name or get_whisper_model_name())


def whisper_model_cached(name: str | None = None) -> bool:
    """True when the model's weights are already on disk, so loading it
    won't download. A `local_files_only` probe of the Hugging Face cache;
    a path to a local model folder counts as cached when it exists. False
    when faster-whisper isn't installed (loading would fail anyway, and the
    caller's prompt is harmless)."""
    name = name or get_whisper_model_name()
    if os.path.isdir(name):
        return True
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        return False
    try:
        download_model(name, local_files_only=True)
        return True
    except Exception:
        return False


def _whisper_device_and_compute_type() -> tuple[str, str]:
    if _whisper_cuda_failed:
        return "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _get_whisper_model(name: str | None = None):
    """Lazily loads and caches a faster-whisper model by name, behind a
    lock (like `_get_model`), so two callers don't both start the download.
    The first call for a model not yet in the Hugging Face cache downloads
    it; callers that can be first ask the user before that
    (`whisper_model_cached`)."""
    name = name or get_whisper_model_name()
    with _whisper_lock:
        device, compute_type = _whisper_device_and_compute_type()
        model = _whisper_models.get((name, device))
        if model is not None:
            return model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "Whisper transcription needs the 'faster-whisper' package (pip install -r requirements.txt)."
            ) from e
        try:
            model = WhisperModel(name, device=device, compute_type=compute_type)
        except Exception as e:
            if device != "cuda":
                raise RuntimeError(f"Failed to load Whisper model '{name}': {e}") from e
            _mark_whisper_cuda_failed(e)
            try:
                model = WhisperModel(name, device="cpu", compute_type="int8")
            except Exception as e2:
                raise RuntimeError(f"Failed to load Whisper model '{name}': {e2}") from e2
            device = "cpu"
        model.kokoro_device = device  # which device this instance runs on
        _whisper_models[(name, device)] = model
        return model


def _mark_whisper_cuda_failed(error) -> None:
    global _whisper_cuda_failed
    if not _whisper_cuda_failed:
        print(f"Whisper on CUDA failed ({error}); using the CPU from now on.")
    _whisper_cuda_failed = True


def _transcribe_wav_whisper(wav_path: str) -> tuple[str, list[tuple[str, float, float]]]:
    """`(text, [(word, start_s, end_s), ...])` for `wav_path`, with word
    timestamps and the VAD filter on. faster-whisper decodes and resamples
    the file itself. CTranslate2 loads the CUDA libraries lazily, so a
    missing one only shows up here: a run that fails on CUDA is retried
    once on the CPU, which every later load then uses."""
    model = _get_whisper_model()
    try:
        return _run_whisper(model, wav_path)
    except Exception as e:
        if getattr(model, "kokoro_device", "cpu") != "cuda":
            raise RuntimeError(f"Transcription failed: {e}") from e
        _mark_whisper_cuda_failed(e)
    try:
        return _run_whisper(_get_whisper_model(), wav_path)
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e


def _run_whisper(model, wav_path: str) -> tuple[str, list[tuple[str, float, float]]]:
    segments, _info = model.transcribe(wav_path, word_timestamps=True, vad_filter=True)
    texts = []
    words = []
    for segment in segments:  # a generator: decoding happens here
        texts.append(segment.text.strip())
        for word in segment.words or ():
            token = word.word.strip()
            if token:
                words.append((token, float(word.start), float(word.end)))
    return " ".join(t for t in texts if t).strip(), words


# --- Vosk ---------------------------------------------------------------

_vosk_lock = threading.Lock()
_vosk_models: dict[str, object] = {}  # model dir path -> loaded vosk.Model


def _get_vosk_model(model_path: str):
    """Lazily loads and caches a Vosk model directory as a singleton keyed by
    path, so switching between two downloaded models (or repeated
    transcriptions against the same one) doesn't reload from disk each time.
    Unlike Audio8, there's no single well-known model id - the caller must
    point at a model folder they've downloaded and unzipped themselves."""
    with _vosk_lock:
        model = _vosk_models.get(model_path)
        if model is not None:
            return model

        try:
            import vosk
        except ImportError as e:
            raise RuntimeError(
                "Vosk transcription needs the 'vosk' package (pip install -r requirements.txt)."
            ) from e

        if not model_path:
            raise RuntimeError(
                "Vosk transcription needs a model folder. Download one from "
                "https://alphacephei.com/vosk/models, unzip it, and set "
                f"{VOSK_MODEL_PATH_ENV_KEY}=<path> in a .env file at the project "
                "root (see .env.example)."
            )
        # No existence pre-check: `vosk.Model` fails for a missing folder
        # the same way it fails for a folder without model files, and one
        # message covers both.
        try:
            vosk.SetLogLevel(-1)  # silence Kaldi's default stderr logging
            model = vosk.Model(model_path)
        except Exception as e:
            raise RuntimeError(
                f"Couldn't load a Vosk model from '{model_path}' ({e}). The folder must exist "
                "and hold an unzipped model from https://alphacephei.com/vosk/models."
            ) from e

        _vosk_models[model_path] = model
        return model


def _ensure_pcm16_mono(wav_path: str) -> tuple[str, str | None]:
    """Returns `(path, temp_path)`: a path to a 16-bit mono PCM WAV holding
    `wav_path`'s audio - the exact format `vosk.KaldiRecognizer` requires -
    converting into a temp file first if the original isn't already in that
    format (stereo, float samples, 8/24/32-bit PCM, etc.). `temp_path` is
    that temp file, for the caller to delete when done, and `None` in the
    common case (already-correct input), where `path` is `wav_path` itself
    unchanged. The caller only ever deletes `temp_path`, so the original
    can't be removed by mistake.

    The initial probe goes through the stdlib `wave` module rather than
    `soundfile`, since a plain `wave.open` + `getnchannels`/`getsampwidth`
    check is cheap and covers the WAV files this format check actually needs
    to reject. `wave` can't even open every valid WAV (e.g. 32-bit float
    PCM raises `wave.Error: unknown format: 3`), so that failure also routes
    into the conversion path below rather than propagating."""
    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() == 1 and wf.getsampwidth() == 2:
                return wav_path, None
    except wave.Error:
        pass  # not something `wave` can parse at all - fall through and convert

    # `soundfile` reads virtually any WAV encoding and, given dtype="int16",
    # handles the bit-depth conversion itself (float/8/24/32-bit -> int16).
    # `always_2d` keeps mono files at shape (n, 1) so the downmix branch
    # below is unconditional regardless of the source channel count.
    data, samplerate = sf.read(wav_path, dtype="int16", always_2d=True)
    if data.shape[1] > 1:
        # Average in a wider dtype first so summing several 16-bit channels
        # can't wrap around int16 before the divide brings it back in range.
        mono = data.astype(np.int32).mean(axis=1).astype(np.int16)
    else:
        mono = data[:, 0]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="kokoro_vosk_")
    os.close(fd)
    try:
        with wave.open(tmp_path, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(samplerate)
            out.writeframes(mono.tobytes())
    except Exception:
        os.remove(tmp_path)
        raise
    return tmp_path, tmp_path


def _transcribe_wav_vosk(wav_path: str, model_path: str) -> str:
    """Transcribes `wav_path` via Vosk's offline recognizer, first converting
    it to the 16-bit mono PCM WAV Vosk requires (see `_ensure_pcm16_mono`) -
    unlike Audio8-ASR, Vosk itself doesn't resample or downmix its input."""
    return _run_vosk(wav_path, model_path, with_words=False)[0]


def _run_vosk(wav_path: str, model_path: str, with_words: bool) -> tuple[str, list[tuple[str, float, float]]]:
    """`(text, words)` from Vosk. `words` is filled only `with_words`, from
    the recognizer's per-word `result` entries (loose timings)."""
    model = _get_vosk_model(model_path)
    import vosk

    words: list[tuple[str, float, float]] = []

    def _take(raw):
        result = json.loads(raw)
        if with_words:
            for item in result.get("result", ()):
                words.append((str(item.get("word", "")), float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        return result.get("text", "")

    converted_path, temp_path = _ensure_pcm16_mono(wav_path)
    try:
        with wave.open(converted_path, "rb") as wf:
            recognizer = vosk.KaldiRecognizer(model, wf.getframerate())
            recognizer.SetWords(with_words)

            pieces = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    pieces.append(_take(recognizer.Result()))
            pieces.append(_take(recognizer.FinalResult()))
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return " ".join(p for p in pieces if p).strip(), words


# --- dispatch -------------------------------------------------------------

def transcribe_wav(
    wav_path: str,
    engine: str = DEFAULT_ASR_ENGINE,
    max_new_tokens: int = 128,
    model_path: str | None = None,
) -> str:
    """Transcribes `wav_path` via the named engine ("whisper", "audio8" or
    "vosk").
    Blocking/CPU-or-GPU-bound - callers from the GUI should run this off the
    main thread (see `VoiceCloneDock._on_transcribe_clicked`, which schedules
    it via `asyncio.to_thread` on the active engine's worker).

    `model_path` only matters for `engine="vosk"`; leaving it `None` (the
    GUI's normal path) falls back to `get_vosk_model_path()` (the
    `VOSK_MODEL_PATH` env var) rather than requiring every caller to read
    that themselves. Passing a path explicitly - the standalone CLI does -
    overrides the environment for that one call."""
    if engine == "whisper":
        return _transcribe_wav_whisper(wav_path)[0]
    elif engine == "audio8":
        return _transcribe_wav_audio8(wav_path, max_new_tokens=max_new_tokens)
    elif engine == "vosk":
        return _transcribe_wav_vosk(wav_path, model_path if model_path is not None else get_vosk_model_path())
    else:
        raise ValueError(f"Unknown ASR engine '{engine}'")


def transcribe_wav_words(
    wav_path: str,
    engine: str = DEFAULT_ASR_ENGINE,
    model_path: str | None = None,
) -> list[tuple[str, float, float]]:
    """`[(word, start_s, end_s), ...]` for `wav_path`, in order. The one
    entry point for anything that needs word timing; nothing calls the
    engine functions directly. Whisper's timings are tight; Vosk's are
    loose; Audio8 has none and raises `NotImplementedError`. Blocking, like
    `transcribe_wav`."""
    if engine == "whisper":
        return _transcribe_wav_whisper(wav_path)[1]
    elif engine == "vosk":
        return _run_vosk(wav_path, model_path if model_path is not None else get_vosk_model_path(),
                         with_words=True)[1]
    elif engine == "audio8":
        raise NotImplementedError("Audio8-ASR has no word timestamps")
    else:
        raise ValueError(f"Unknown ASR engine '{engine}'")


def _main(argv) -> int:
    words = "--words" in argv
    args = [a for a in argv if a != "--words"]
    if not args:
        print(
            "Usage: python -m kokoro_gui.engine.asr <path/to/reference.wav> "
            "[engine=whisper|audio8|vosk] [vosk_model_path] [--words]\n"
            f"(vosk_model_path defaults to the {VOSK_MODEL_PATH_ENV_KEY} env var / .env entry if omitted;\n"
            " --words prints one timed word per line: start end word)"
        )
        return 1
    wav_path = args[0]
    engine = args[1] if len(args) > 1 else DEFAULT_ASR_ENGINE
    model_path = args[2] if len(args) > 2 else None
    if words:
        for word, start, end in transcribe_wav_words(wav_path, engine=engine, model_path=model_path):
            print(f"{start:8.2f} {end:8.2f}  {word}")
    else:
        print(transcribe_wav(wav_path, engine=engine, model_path=model_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
