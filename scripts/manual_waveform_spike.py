"""Manual QA only - NOT part of the app, NOT imported by anything in
kokoro_gui/, safe to delete once this spike (Workstream 3 of
Claude/PLAN_daw_ui_ux_redesign.md) is visually signed off.

Launches the standalone WaveformPanel spike widget (kokoro_gui/qt/waveform_view.py)
so its rendering and playhead motion can be eyeballed - the one thing
automated tests can't judge ("does this look right?").

Usage:
    python scripts/manual_waveform_spike.py [path/to/audio.wav]

With no argument, generates a short synthetic tone on the fly so this runs
with zero fixtures.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "xcb")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kokoro_gui.qt.waveform_view import WaveformPanel  # noqa: E402


def _make_synthetic_tone_wav() -> str:
    sample_rate = 24000
    duration_seconds = 3.0
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # A slowly-sweeping tone plus a touch of amplitude variation, so the
    # waveform has visible structure to eyeball rather than a flat sine.
    freq = 220 + 220 * (t / duration_seconds)
    data = (0.6 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))).astype(np.float32)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, data, sample_rate)
    return path


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else _make_synthetic_tone_wav()

    app = QApplication(sys.argv)
    panel = WaveformPanel()
    panel.setWindowTitle("Waveform Spike - manual QA")
    panel.resize(800, 200)
    panel.load_audio(audio_path)
    panel.show()
    app.exec()


if __name__ == "__main__":
    main()
