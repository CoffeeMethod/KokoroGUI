"""Cross-platform audio playback for preview buttons and JIT streaming.

Wraps `sounddevice` (PortAudio) instead of the Windows-only `winsound`
module, so preview/JIT playback works on Windows, macOS, and Linux.

On Linux, `sounddevice` needs the system PortAudio shared library
(`libportaudio2` / `portaudio19-dev`) installed. If it's missing, importing
`sounddevice` raises OSError - we catch that and degrade to a no-op instead
of crashing import of kokoro_engine/gui on machines without it.
"""
import soundfile as sf

try:
    import sounddevice as sd
    AVAILABLE = True
except OSError:
    sd = None
    AVAILABLE = False


def play(path: str, blocking: bool = False) -> None:
    """Play an audio file.

    blocking=True waits for playback to finish (used to pace the JIT
    playback loop, matching the old `winsound.PlaySound(..., SND_FILENAME)`
    behavior). blocking=False fires and forgets (used by preview buttons,
    matching the old `SND_ASYNC` behavior).
    """
    if not AVAILABLE:
        return
    data, samplerate = sf.read(path, dtype='float32')
    sd.play(data, samplerate)
    if blocking:
        sd.wait()


def stop() -> None:
    """Stop any currently playing audio immediately (old SND_PURGE)."""
    if not AVAILABLE:
        return
    sd.stop()
