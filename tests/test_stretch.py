"""Tests for kokoro_gui/audio/stretch.py: the streaming WSOLA time stretch
the transport plays through at a rate other than 1.0 (plan 27). Offline,
numpy only, no Qt and no device."""
import numpy as np
import pytest

from kokoro_gui.audio import stretch
from kokoro_gui.audio.stretch import Stretcher, stretch_array

SR = 24000


def _tone(seconds=2.0, freq=440.0, sr=SR, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _peak_hz(samples, sr=SR):
    seg = samples[sr // 10:-sr // 10]
    spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    return float(np.argmax(spectrum)) * sr / len(seg)


@pytest.mark.parametrize("rate", [0.5, 0.75, 1.25, 1.5, 2.0])
def test_a_tone_keeps_its_pitch_at_any_rate(rate):
    out = stretch_array(_tone(), SR, rate)

    assert abs(_peak_hz(out) - 440.0) < 3.0


@pytest.mark.parametrize("rate", [0.5, 0.75, 1.25, 1.5, 2.0])
def test_the_output_is_the_input_length_over_the_rate(rate):
    out = stretch_array(_tone(), SR, rate)

    assert len(out) == int(round(SR * 2.0 / rate))


@pytest.mark.parametrize("rate", [0.5, 1.5, 2.0])
def test_a_tone_keeps_its_level(rate):
    out = stretch_array(_tone(amp=0.5), SR, rate)
    steady = out[SR // 10:-SR // 10]

    assert np.sqrt(np.mean(steady ** 2)) == pytest.approx(0.5 / np.sqrt(2.0), rel=0.02)
    assert np.abs(out).max() <= 0.5 + 1e-3


def test_rate_one_gives_the_input_back():
    noise = (np.random.default_rng(7).standard_normal(SR) * 0.1).astype(np.float32)

    out = stretch_array(noise, SR, 1.0)

    assert np.allclose(out, noise[:len(out)], atol=1e-5)


def test_silence_stays_silent_and_finite():
    out = stretch_array(np.zeros(SR, dtype=np.float32), SR, 1.7)

    assert np.all(np.isfinite(out)) and np.all(out == 0.0)


def test_channels_keep_their_own_content():
    left = _tone(1.0, 440.0)
    data = np.stack([left, np.zeros_like(left)], axis=1)

    out = stretch_array(data, SR, 1.5)

    assert out.shape[1] == 2
    assert np.abs(out[:, 1]).max() == 0.0
    assert abs(_peak_hz(out[:, 0]) - 440.0) < 3.0


def test_the_block_size_does_not_change_the_samples():
    noise = (np.random.default_rng(3).standard_normal((SR, 2)) * 0.1).astype(np.float32)

    def run(sizes):
        position = [0]

        def pull(count):
            chunk = noise[position[0]:position[0] + count]
            position[0] += count
            return np.concatenate((chunk, np.zeros((count - len(chunk), 2), dtype=np.float32)))

        stretcher = Stretcher(SR)
        stretcher.reset(1.5)
        return np.concatenate([stretcher.process(n, pull) for n in sizes])

    whole = run([8000])
    pieces = run([1024] * 7 + [832])

    assert np.array_equal(whole, pieces)
    assert np.array_equal(run([37] * 216 + [8]), whole)


def test_source_frames_counts_what_the_output_stands_for():
    stretcher = Stretcher(SR)
    stretcher.reset(2.0)
    stretcher.process(1000, lambda n: np.zeros((n, 2), dtype=np.float32))
    stretcher.process(500, lambda n: np.zeros((n, 2), dtype=np.float32))

    assert stretcher.source_frames == 3000


def test_reset_starts_a_new_stream_and_clamps_the_rate():
    stretcher = Stretcher(SR)
    stretcher.reset(1.5)
    stretcher.process(1000, lambda n: np.ones((n, 2), dtype=np.float32))
    stretcher.reset(9.0)

    assert stretcher.rate == stretch.RATE_MAX
    assert stretcher.played == 0
    stretcher.reset(0.01)
    assert stretcher.rate == stretch.RATE_MIN


def test_a_1024_frame_block_at_2x_costs_a_small_part_of_its_duration():
    """The transport calls this from the audio callback. 1024 frames at 24 kHz
    last 42.7 ms; the mean call has to stay far under that on any machine
    (it is about 0.15 ms where this was written)."""
    import time

    data = (np.random.default_rng(1).standard_normal((SR * 30, 2)) * 0.1).astype(np.float32)
    position = [0]

    def pull(count):
        chunk = data[position[0]:position[0] + count]
        position[0] += count
        return chunk

    stretcher = Stretcher(SR)
    stretcher.reset(2.0)
    started = time.perf_counter()
    for _ in range(300):
        stretcher.process(1024, pull)
    mean_ms = (time.perf_counter() - started) / 300 * 1000.0

    assert mean_ms < 1024 / SR * 1000.0 / 2
