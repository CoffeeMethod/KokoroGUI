"""Microphone input for a guide take (plan 31). No Qt.

`Recorder` wraps one `sounddevice.InputStream`. `start()` opens it, the
PortAudio callback copies each block into a list under a lock and keeps the
block's peak for `level()`, and `stop()` closes the stream and returns the
blocks joined into one mono float32 array. The callback allocates nothing
beyond the copy of the block: the peak comes from `max()` and `min()`.

The stream asks for `playback.stream_blocksize(sample_rate)`, the same
block-size rule as every output stream (a fixed block on macOS, PortAudio's
choice elsewhere, `KOKOROGUI_BLOCKSIZE` overrides).

`stream_factory(sample_rate, channels, device, callback)` is injectable, as
in `Transport`: tests pass a fake that calls the callback with known
blocks, so no test opens a real device. Without it the recorder looks up
`default_stream_factory` when `start()` runs, which reads `playback.sd`
then. `available()` is False when `sounddevice` or PortAudio is missing,
and `start()` raises `RecorderError` in that case.

`count_in(sample_rate)` is the three-beep count-in as an array: 1 kHz
beeps of 100 ms, one second apart, `COUNT_IN_S` long. The take starts
recording at the first beep, so the beeps are in the file and the take's
segment skips them with `Segment.range`.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

import playback

from kokoro_gui.i18n import tr

DEFAULT_SAMPLE_RATE = 48000
BEEPS = 3
BEEP_HZ = 1000.0
BEEP_S = 0.1
BEAT_S = 1.0
# The count-in's length: the clip's first word comes on the beat after the
# last beep.
COUNT_IN_S = BEEPS * BEAT_S
BEEP_GAIN = 0.5
# Fade at each end of a beep, so it doesn't click.
BEEP_RAMP_S = 0.005


class RecorderError(Exception):
    """No input device, or the stream wouldn't open."""


def available() -> bool:
    """True when `sounddevice` imported and found PortAudio."""
    return bool(playback.AVAILABLE) and playback.sd is not None


def input_devices() -> list:
    """`[(index, name, default_samplerate)]` for every device with an input
    channel; empty without PortAudio or when the query fails."""
    if not available():
        return []
    try:
        devices = playback.sd.query_devices()
    except Exception:
        return []
    found = []
    for index, info in enumerate(devices):
        try:
            if int(info.get("max_input_channels", 0)) > 0:
                found.append((index, str(info.get("name", f"Device {index}")),
                              int(info.get("default_samplerate") or DEFAULT_SAMPLE_RATE)))
        except (TypeError, ValueError, AttributeError):
            continue
    return found


def count_in(sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """`COUNT_IN_S` of mono float32: a `BEEP_HZ` beep of `BEEP_S` at the
    start of each beat, silence between."""
    rate = int(sample_rate)
    out = np.zeros(int(round(COUNT_IN_S * rate)), dtype=np.float32)
    length = int(round(BEEP_S * rate))
    t = np.arange(length, dtype=np.float32) / rate
    beep = (BEEP_GAIN * np.sin(2.0 * np.pi * BEEP_HZ * t)).astype(np.float32)
    ramp = max(1, int(round(BEEP_RAMP_S * rate)))
    envelope = np.ones(length, dtype=np.float32)
    envelope[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
    envelope[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    beep *= envelope
    for beat in range(BEEPS):
        start = int(round(beat * BEAT_S * rate))
        out[start:start + length] = beep[:len(out) - start]
    return out


def default_stream_factory(sample_rate: int, channels: int, device, callback: Callable):
    if not available():
        raise RecorderError(tr("No audio input: sounddevice or PortAudio is missing."))
    return playback.sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", device=device,
                                   blocksize=playback.stream_blocksize(sample_rate), callback=callback)


class Recorder:
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1, device=None,
                 stream_factory: Optional[Callable] = None):
        self.sample_rate = int(sample_rate)
        self.channels = max(1, int(channels))
        self.device = device
        self._stream_factory = stream_factory
        self._lock = threading.Lock()
        self._blocks: list = []
        self._frames = 0
        self._peak = 0.0
        self._stream = None
        # Input overflows the callback saw (an int, no logging there).
        self.overflows = 0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        """Opens and starts the input stream; clears anything recorded
        before. Raises `RecorderError` when the stream won't open."""
        if self._stream is not None:
            return
        with self._lock:
            self._blocks = []
            self._frames = 0
            self._peak = 0.0
            self.overflows = 0
        factory = self._stream_factory or default_stream_factory
        try:
            stream = factory(self.sample_rate, self.channels, self.device, self._callback)
            stream.start()
        except RecorderError:
            raise
        except Exception as e:
            raise RecorderError(tr("Couldn't open the input: {error}", error=e)) from e
        self._stream = stream

    def _callback(self, indata, frames, _time_info, status) -> None:
        if status:
            self.overflows += 1
        block = indata.copy()
        peak = max(float(block.max()), -float(block.min())) if len(block) else 0.0
        with self._lock:
            self._blocks.append(block)
            self._frames += len(block)
            self._peak = peak

    def stop(self) -> np.ndarray:
        """Stops and closes the stream and returns what it recorded, mono
        float32 (the first channel). Empty when nothing came in."""
        stream, self._stream = self._stream, None
        if stream is not None:
            for name in ("stop", "close"):
                try:
                    getattr(stream, name)()
                except Exception:
                    pass
        with self._lock:
            blocks, self._blocks = self._blocks, []
            self._peak = 0.0
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        joined = np.concatenate(blocks, axis=0)
        if joined.ndim > 1:
            joined = joined[:, 0]
        return np.ascontiguousarray(joined, dtype=np.float32)

    def level(self) -> float:
        """Linear peak of the latest block, 0.0 when stopped."""
        with self._lock:
            return self._peak

    def seconds(self) -> float:
        """How long the recording runs so far."""
        with self._lock:
            return self._frames / float(self.sample_rate)
