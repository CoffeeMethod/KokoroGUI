"""Position-tracking arrangement player (UI4 of
Claude/PLAN_ui_shell_redesign.md, section 5).

`playback.py` stays the fire-and-forget player for previews and JIT. This
`Transport` plays a whole arrangement: `load(schedule)` takes the
`ScheduledClip`s `kokoro_gui.daw.arrangement.compute_arrangement` produced
(estimated clips are skipped - they're silence), opens one
`sounddevice.OutputStream` at the project sample rate and fills each block
in the PortAudio callback by summing every clip overlapping it
(`kokoro_gui.audio.mixer.mix_block`), in stereo: each `ScheduledClip`
carries its track's gain, pan and automation and its own fades.

`loop_range` (frames, runtime only) wraps playback inside a region once the
playhead crosses its end; without one, `loop` wraps the whole arrangement.

Position is the callback's frame counter, sample accurate, published to
the GUI thread by a 30Hz `QTimer` as `positionChanged(float)`. `play()`,
`pause()`, `stop()`, `seek()`, `toggle()` are GUI-thread API.

`stream_factory` is injectable: tests pass a fake whose `pull(frames)`
drives the callback synchronously, so the mixing math and state machine
are testable without a device (same rule as `conftest.py`'s `playback`
mock). Without PortAudio (`playback.AVAILABLE` False) the transport still
loads and changes state, it just never advances.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

import playback
from kokoro_gui.audio import mixer

POSITION_TIMER_MS = 33
DEFAULT_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class ScheduledClip:
    clip_id: str
    start_s: float
    path: Optional[str]
    gain: float = 1.0
    # Read-time post-processing (kokoro_gui/audio/post.py) applied to
    # `path` on load; None plays the file as is.
    post_config: Optional[dict] = None
    pan: float = 0.0
    fade_in_s: float = 0.0
    fade_out_s: float = 0.0
    # The track's `[seconds, gain]` breakpoints, absolute on the timeline.
    automation: tuple = ()
    # `(start_s, end_s)` into `path`: only that range plays (a
    # `Segment.range`, a source track sliced per clip). None plays the file.
    slice: Optional[tuple] = None


def loaded_clip_for(item, samples: np.ndarray, sample_rate: int) -> "mixer.LoadedClip":
    """A `ScheduledClip`'s mixer entry at `sample_rate`: frames for the
    start and the fades, left/right gains from the pan, the automation as
    frame arrays. Shared with the exporter."""
    gain_l, gain_r = mixer.pan_gains(item.pan)
    return mixer.LoadedClip(
        clip_id=item.clip_id,
        start_frame=int(round(item.start_s * sample_rate)),
        samples=samples,
        gain=item.gain,
        gain_l=gain_l,
        gain_r=gain_r,
        fade_in_frames=int(round(max(0.0, item.fade_in_s) * sample_rate)),
        fade_out_frames=int(round(max(0.0, item.fade_out_s) * sample_rate)),
        automation=mixer.automation_arrays(item.automation, sample_rate),
    )


class _NullStream:
    """Stands in for `sounddevice.OutputStream` when PortAudio is missing."""

    def __init__(self, *_args, **_kwargs):
        self.active = False

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.active = False


def default_stream_factory(sample_rate: int, callback: Callable):
    if not playback.AVAILABLE or playback.sd is None:
        return _NullStream()
    return playback.sd.OutputStream(samplerate=sample_rate, channels=mixer.CHANNELS, dtype="float32",
                                    callback=callback)


class Transport(QObject):
    positionChanged = Signal(float)
    stateChanged = Signal(str)  # "playing" | "paused" | "stopped"
    finished = Signal()
    loaded = Signal()

    def __init__(self, parent=None, stream_factory: Optional[Callable] = None):
        super().__init__(parent)
        self._stream_factory = stream_factory or default_stream_factory
        self._lock = threading.Lock()
        self._clips: list = []
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._frame = 0
        self._total_frames = 0
        self._ended = False
        self._stream = None
        self._state = "stopped"
        self.loop = False
        # `(start_frame, end_frame)` or None; see the module docstring.
        self.loop_range: Optional[tuple] = None
        self._timer = QTimer(self)
        self._timer.setInterval(POSITION_TIMER_MS)
        self._timer.timeout.connect(self._on_tick)

    # -- introspection -----------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_playing(self) -> bool:
        return self._state == "playing"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def position(self) -> float:
        with self._lock:
            frame = self._frame
        return frame / float(self._sample_rate)

    def duration(self) -> float:
        return self._total_frames / float(self._sample_rate)

    def loaded_clips(self) -> list:
        with self._lock:
            return list(self._clips)

    # -- loading -----------------------------------------------------------------

    def load(self, schedule: list, sample_rate: Optional[int] = None,
             total_duration_s: Optional[float] = None) -> None:
        """Replace the arrangement. Keeps the current position and playing
        state so a freshly generated clip becomes audible mid-playback (this
        is also what `reload()` is for)."""
        if sample_rate:
            new_rate = int(sample_rate)
        else:
            new_rate = self._sample_rate
        clips = []
        for item in schedule:
            if not item.path:
                continue
            try:
                samples = mixer.load_clip_samples(item.path, new_rate, item.post_config, item.slice)
            except Exception:
                continue
            clips.append(loaded_clip_for(item, samples, new_rate))
        total = mixer.total_frames(clips)
        if total_duration_s is not None:
            total = max(total, int(round(total_duration_s * new_rate)))

        rate_changed = new_rate != self._sample_rate
        with self._lock:
            if rate_changed:
                self._frame = int(round(self._frame * new_rate / float(self._sample_rate)))
                if self.loop_range is not None:
                    ratio = new_rate / float(self._sample_rate)
                    self.loop_range = tuple(int(round(f * ratio)) for f in self.loop_range)
            self._sample_rate = new_rate
            self._clips = clips
            self._total_frames = total
            self._frame = min(self._frame, total)
            self._ended = False
        if rate_changed and self._stream is not None:
            was_playing = self.is_playing
            self._close_stream()
            if was_playing:
                self._open_stream()
        self.loaded.emit()
        self.positionChanged.emit(self.position())

    def reload(self, schedule: list, sample_rate: Optional[int] = None,
               total_duration_s: Optional[float] = None) -> None:
        self.load(schedule, sample_rate=sample_rate, total_duration_s=total_duration_s)

    # -- control -------------------------------------------------------------------

    def play(self) -> None:
        if self._total_frames == 0:
            return
        with self._lock:
            if self._frame >= self._total_frames:
                self._frame = 0
            self._ended = False
        self._open_stream()
        self._set_state("playing")
        self._timer.start()
        self.positionChanged.emit(self.position())

    def pause(self) -> None:
        if self._state != "playing":
            return
        self._close_stream()
        self._timer.stop()
        self._set_state("paused")
        self.positionChanged.emit(self.position())

    def stop(self) -> None:
        self._close_stream()
        self._timer.stop()
        with self._lock:
            self._frame = 0
            self._ended = False
        self._set_state("stopped")
        self.positionChanged.emit(0.0)

    def toggle(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def set_loop_range_s(self, start_s: Optional[float], end_s: Optional[float] = None) -> None:
        """Loop between two times in seconds; `None` clears the region."""
        if start_s is None or end_s is None:
            self.loop_range = None
            return
        lo, hi = sorted((max(0.0, float(start_s)), max(0.0, float(end_s))))
        start, end = int(round(lo * self._sample_rate)), int(round(hi * self._sample_rate))
        self.loop_range = (start, end) if end > start else None

    def seek(self, seconds: float) -> None:
        frame = int(round(max(0.0, seconds) * self._sample_rate))
        with self._lock:
            self._frame = min(frame, self._total_frames)
            self._ended = False
        self.positionChanged.emit(self.position())

    # -- internals -------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self.stateChanged.emit(state)

    def _open_stream(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._stream_factory(self._sample_rate, self._callback)
        try:
            self._stream.start()
        except Exception:
            self._stream = None

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        for method in ("stop", "close"):
            try:
                getattr(stream, method)()
            except Exception:
                pass

    def _callback(self, outdata, frames, _time_info=None, _status=None) -> None:
        """PortAudio callback thread. Never touches Qt."""
        with self._lock:
            frame = self._frame
            clips = self._clips
            total = self._total_frames
            loop = self.loop
            loop_range = self.loop_range
        new_frame = frame + frames
        if loop_range is not None and frame < loop_range[1] <= new_frame:
            # The block crosses the region's end: play up to it, then carry
            # on from the region's start.
            start, end = loop_range
            head = end - frame
            block = np.zeros((frames, mixer.CHANNELS), dtype=np.float32)
            block[:head] = mixer.mix_block(clips, frame, head)
            if frames > head:
                block[head:] = mixer.mix_block(clips, start, frames - head)
            new_frame = start + (frames - head)
        else:
            block = mixer.mix_block(clips, frame, frames)
        self._write_block(outdata, block)
        ended = False
        # Short of a loop region's end, playback runs on through silence.
        looping_region = loop_range is not None and new_frame < loop_range[1]
        if new_frame >= total and not looping_region:
            if loop and total > 0:
                new_frame = new_frame % total
            else:
                new_frame = total
                ended = True
        with self._lock:
            self._frame = new_frame
            if ended:
                self._ended = True

    @staticmethod
    def _write_block(outdata, block: np.ndarray) -> None:
        """A stereo block into whatever the device opened: both columns of a
        stereo buffer, the average on a mono one, silence on any extra."""
        if outdata.ndim == 2 and outdata.shape[1] >= 2:
            outdata[:, :2] = block
            if outdata.shape[1] > 2:
                outdata[:, 2:] = 0.0
        elif outdata.ndim == 2:
            outdata[:, 0] = block.mean(axis=1)
        else:
            outdata[:] = block.mean(axis=1)

    def _on_tick(self) -> None:
        with self._lock:
            ended = self._ended
        self.positionChanged.emit(self.position())
        if ended:
            self._close_stream()
            self._timer.stop()
            with self._lock:
                self._ended = False
            self._set_state("stopped")
            self.finished.emit()

    def process_pending(self) -> None:
        """Test hook: what the timer tick does, callable synchronously."""
        self._on_tick()


def render_block_for_test(transport: Transport, frames: int) -> np.ndarray:
    """Drives one callback synchronously and returns the mixed `(frames, 2)`
    block."""
    out = np.zeros((frames, mixer.CHANNELS), dtype=np.float32)
    transport._callback(out, frames)
    return out
