"""Cross-platform audio playback for preview buttons and JIT streaming.

Wraps `sounddevice` (PortAudio) instead of the Windows-only `winsound`
module, so preview/JIT playback works on Windows, macOS, and Linux.

On Linux, `sounddevice` needs the system PortAudio shared library
(`libportaudio2` / `portaudio19-dev`) installed. If it's missing, importing
`sounddevice` raises OSError - we catch that and degrade to a no-op instead
of crashing import of kokoro_engine/gui on machines without it.

`stream_blocksize(rate)` is the `blocksize` every output stream asks for,
here and in the arrangement transport (`kokoro_gui/audio/transport.py`).
On macOS it's a fixed block: CoreAudio rate-converts a stream opened below
the device's rate (24000 Hz against 44100 or 48000), and left to PortAudio
the callback then gets about 14 frames, some 1700 calls a second, and
playback clicks (GitHub PR 56). Windows and Linux keep 0, PortAudio's
choice. `KOKOROGUI_BLOCKSIZE` overrides it on any platform.
"""
import logging
import os
import sys

import soundfile as sf

try:
    import sounddevice as sd
    AVAILABLE = True
except OSError:
    sd = None
    AVAILABLE = False

logger = logging.getLogger(__name__)

BLOCKSIZE_ENV = "KOKOROGUI_BLOCKSIZE"
# The fixed block covers at least this much audio: 1024 frames at 24000 Hz,
# the size the PR author measured and a macOS 15 user confirmed.
FIXED_BLOCK_S = 0.040
MAX_BLOCKSIZE = 1 << 16
_warned_env_values: set = set()


def choose_blocksize(platform: str, stream_rate: int, env=None) -> int:
    """Frames per callback for an output stream at `stream_rate`; 0 lets
    PortAudio choose. `env[KOKOROGUI_BLOCKSIZE]`, a whole number from 0 to
    65536, wins on any platform; anything else there is logged once and
    ignored. Otherwise macOS (`"darwin"`) gets the smallest power of two
    that holds `FIXED_BLOCK_S` at the stream rate, and every other platform
    gets 0."""
    raw = (env or {}).get(BLOCKSIZE_ENV)
    if raw is not None and str(raw).strip() != "":
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = -1
        if 0 <= value <= MAX_BLOCKSIZE:
            return value
        if raw not in _warned_env_values:
            _warned_env_values.add(raw)
            logger.warning("Ignoring %s=%r: expected a whole number from 0 to %d",
                           BLOCKSIZE_ENV, raw, MAX_BLOCKSIZE)
    if not str(platform).startswith("darwin"):
        return 0
    wanted = max(1, int(stream_rate) if stream_rate else 24000) * FIXED_BLOCK_S
    block = 256
    while block < wanted and block < MAX_BLOCKSIZE:
        block *= 2
    return block


def stream_blocksize(stream_rate: int) -> int:
    """`choose_blocksize` for this machine and environment."""
    return choose_blocksize(sys.platform, stream_rate, os.environ)


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
    sd.play(data, samplerate, blocksize=stream_blocksize(samplerate))
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
    sd.play(data, rate, blocksize=stream_blocksize(rate))


def stop() -> None:
    """Stop any currently playing audio immediately (old SND_PURGE)."""
    if not AVAILABLE:
        return
    sd.stop()
