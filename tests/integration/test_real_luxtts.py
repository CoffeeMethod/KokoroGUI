"""Opt-in real CPU smoke test; set LUXTTS_REFERENCE_WAV to clear speech >=3s.

Run: python -m pytest -m integration tests/integration/test_real_luxtts.py -s
Install requirements-luxtts.txt first. Weights download on first use.
"""
import os

import numpy as np
import pytest
import soundfile as sf

from kokoro_gui.engines.luxtts import LuxTTSModel

pytestmark = pytest.mark.integration


def test_real_luxtts_cpu(timestamped_output_dir):
    reference = os.environ.get("LUXTTS_REFERENCE_WAV")
    if not reference:
        pytest.skip("Set LUXTTS_REFERENCE_WAV to a >=3-second reference speech WAV.")
    model = LuxTTSModel()
    assert model.load("en", "cpu")
    text = "This is a Lux T T S integration test. The quick brown fox jumps over the lazy dog."
    for smooth in (False, True):
        result = model.synthesize(text, reference, 1.0, "en", {"return_smooth": smooth})
        assert result.audio.dtype == np.float32 and result.audio.ndim == 1
        assert result.audio.size > 48000 and np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) > 0.001
        path = timestamped_output_dir / f"luxtts_{'smooth' if smooth else 'standard'}.wav"
        sf.write(str(path), result.audio, model.sample_rate)
        assert sf.info(str(path)).samplerate == 48000
        print(f"LuxTTS sample: {path}")
