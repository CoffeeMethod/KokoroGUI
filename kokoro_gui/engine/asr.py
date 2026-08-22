"""Auto-transcription helper for reference audio, built on
https://huggingface.co/Audio8/Audio8-ASR-0.1B - used by the Voice Reference
dock (kokoro_gui/qt/docks/voice_clone_dock.py) to pre-fill an editable
transcript for a WAV a user is about to use as an Audio8-TTS voice reference.

Audio8-TTS-Preview-0.6b's zero-shot voice cloning needs a transcript of the
reference audio, not just the audio itself - getting that by hand is tedious,
so this is a starting point the user reviews/corrects, not a ground-truth
oracle. `Audio8/Audio8-ASR-0.1B` is licensed CC-BY-NC-4.0 (non-commercial);
fine for this project's own use, but worth knowing if you fork this for
something commercial.

`transformers` is only imported inside `_get_model()`, not at this module's
top level - though note `transformers` itself is already an indirect hard
dependency of this app (the `kokoro` package imports it internally), so this
isn't about avoiding the `transformers` import itself. What *is* still
lazy, and is the actual cost worth deferring: `AutoModel.from_pretrained(...)`
/`AutoProcessor.from_pretrained(...)` - the network fetch (first run) and
the model weights actually landing in memory - only happens on first call to
`transcribe_wav`/`_get_model`, not merely by importing this module (e.g.
whenever the Voice Reference dock is built). Loads with
`trust_remote_code=True`, which executes Python code shipped in the model's
HF repo the first time it's loaded - inherent to how this model is
distributed, not something this module can avoid while still using it.
"""
from __future__ import annotations

import sys
import threading

ASR_MODEL_ID = "Audio8/Audio8-ASR-0.1B"

_model_lock = threading.Lock()
_model = None
_processor = None


def _get_model():
    """Lazily loads and caches the ASR model/processor as a process-wide
    singleton (guarded by a lock so two near-simultaneous "Auto-Transcribe"
    clicks - or a batch run alongside one - don't each start their own
    multi-GB download/load)."""
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
            processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(ASR_MODEL_ID, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load {ASR_MODEL_ID}: {e}") from e

        _model, _processor = model, processor
        return _model, _processor


def transcribe_wav(wav_path: str, max_new_tokens: int = 128) -> str:
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m kokoro_gui.engine.asr <path/to/reference.wav>")
        raise SystemExit(1)
    print(transcribe_wav(sys.argv[1]))
