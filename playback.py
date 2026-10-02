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


def play_range(path: str, start_s: float, end_s: float) -> None:
    """Play `[start_s, end_s]` of an audio file without waiting (the Import
    Recording review dialog's per-line play buttons)."""
    if not AVAILABLE:
        return
    with sf.SoundFile(path) as f:
        rate = f.samplerate
        start = max(0, min(int(round(start_s * rate)), f.frames))
        f.seek(start)
        data = f.read(max(0, int(round(end_s * rate)) - start), dtype='float32')
    sd.play(data, rate)


def stop() -> None:
    """Stop any currently playing audio immediately (old SND_PURGE)."""
    if not AVAILABLE:
        return
    sd.stop()
